//! 意图-嵌入核心引擎
//!
//! 这是整个系统的核心：意图和视频嵌入的双向循环

use std::sync::Arc;
use ndarray::Array1;
use tokio::sync::mpsc;
use tracing::{info, debug};

use crate::buffer::frame::Frame;
use crate::buffer::embedding::EmbeddingCache;
use crate::core::intent_loop::{
    IntentState, IntentType, IntentDrivenStore, EmbeddingAnalysis,
    ExtractionStrategy, ActivityLevel,
};

// ============================================================
// 核心引擎事件
// ============================================================

/// 引擎事件
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// 意图更新
    IntentUpdated {
        primary: IntentType,
        confidence: f32,
        iteration: u32,
    },
    /// 策略切换
    StrategyChanged {
        old: ExtractionStrategy,
        new: ExtractionStrategy,
    },
    /// 关键帧选中
    KeyframeSelected {
        timestamp: f64,
        reason: String,
    },
    /// 分析完成
    AnalysisComplete {
        activity: ActivityLevel,
        has_change: bool,
    },
}

// ============================================================
// 帧选择器 (根据策略选择关键帧)
// ============================================================

/// 帧选择器
pub struct FrameSelector {
    /// 当前策略
    strategy: ExtractionStrategy,
    /// 候选帧
    candidates: Vec<(Frame, f64, f32)>, // (frame, timestamp, score)
    /// 最大候选数
    max_candidates: usize,
}

impl FrameSelector {
    pub fn new(strategy: ExtractionStrategy) -> Self {
        Self {
            strategy,
            candidates: Vec::new(),
            max_candidates: 50,
        }
    }

    /// 更新策略
    pub fn set_strategy(&mut self, strategy: ExtractionStrategy) {
        if self.strategy != strategy {
            debug!("策略切换: {:?} -> {:?}", self.strategy, strategy);
            self.strategy = strategy;
            // 策略变化时可能需要重新评估候选帧
        }
    }

    /// 评估并添加候选帧
    pub fn evaluate(&mut self, frame: Frame, change_score: f32) -> bool {
        let timestamp = frame.timestamp;
        let should_select = self.should_select(timestamp, change_score);

        if should_select {
            self.candidates.push((frame, timestamp, change_score));

            // 维护数量限制
            if self.candidates.len() > self.max_candidates {
                self.prune();
            }
        }

        should_select
    }

    /// 根据策略判断是否选择
    fn should_select(&self, timestamp: f64, change_score: f32) -> bool {
        let min_interval = self.strategy.min_interval();

        // 检查与上一个候选帧的时间间隔
        if let Some((_, last_time, _)) = self.candidates.last() {
            if timestamp - last_time < min_interval {
                return false;
            }
        }

        match self.strategy {
            ExtractionStrategy::UniformWithChange => {
                // 变化分数超过阈值，或时间间隔足够长
                change_score > 0.05 || self.candidates.is_empty()
            }
            ExtractionStrategy::DenseWithPeaks => {
                // 更低的阈值，更密集
                change_score > 0.03 || self.candidates.is_empty()
            }
            ExtractionStrategy::StartEndTransition => {
                // 只关注开始、结束、转折
                change_score > 0.1 || self.candidates.len() < 2
            }
            ExtractionStrategy::Representative => {
                // 每隔一定时间选一帧
                true
            }
            ExtractionStrategy::StableFrames => {
                // 选择稳定帧 (低变化)
                change_score < 0.03
            }
            ExtractionStrategy::HighFrequency => {
                // 几乎所有帧
                true
            }
        }
    }

    /// 获取选中的关键帧
    pub fn get_keyframes(&self, count: usize) -> Vec<&Frame> {
        match self.strategy {
            ExtractionStrategy::UniformWithChange | ExtractionStrategy::DenseWithPeaks => {
                // 按变化分数排序，取最高的
                let mut sorted: Vec<_> = self.candidates.iter().collect();
                sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                sorted.into_iter().take(count).map(|(f, _, _)| f).collect()
            }
            ExtractionStrategy::StartEndTransition => {
                // 开始 + 最大变化 + 结束
                let mut result = Vec::new();
                if let Some((f, _, _)) = self.candidates.first() {
                    result.push(f);
                }
                if let Some((f, t, _)) = self.candidates.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) {
                    // 检查时间戳是否与已添加的不同
                    let already_added = result.iter().any(|r| (r.timestamp - t).abs() < 0.1);
                    if !already_added {
                        result.push(f);
                    }
                }
                if let Some((f, _, _)) = self.candidates.last() {
                    if result.len() < count {
                        result.push(f);
                    }
                }
                result
            }
            ExtractionStrategy::StableFrames => {
                // 按变化分数升序，取最稳定的
                let mut sorted: Vec<_> = self.candidates.iter().collect();
                sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                sorted.into_iter().take(count).map(|(f, _, _)| f).collect()
            }
            _ => {
                // 均匀采样
                let step = (self.candidates.len() / count).max(1);
                self.candidates.iter()
                    .step_by(step)
                    .take(count)
                    .map(|(f, _, _)| f)
                    .collect()
            }
        }
    }

    fn prune(&mut self) {
        // 根据策略保留重要的
        let keep = self.max_candidates / 2;

        match self.strategy {
            ExtractionStrategy::HighFrequency => {
                // 保留最近的
                self.candidates.drain(0..self.candidates.len() - keep);
            }
            _ => {
                // 保留变化分数高的 + 首尾
                let first = self.candidates.first().cloned();
                let last = self.candidates.last().cloned();

                self.candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                self.candidates.truncate(keep - 2);

                if let Some(f) = first {
                    if !self.candidates.iter().any(|(_, t, _)| (*t - f.1).abs() < 0.1) {
                        self.candidates.insert(0, f);
                    }
                }
                if let Some(l) = last {
                    if !self.candidates.iter().any(|(_, t, _)| (*t - l.1).abs() < 0.1) {
                        self.candidates.push(l);
                    }
                }

                self.candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
        }
    }

    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }
}

// ============================================================
// 核心引擎
// ============================================================

/// 意图-嵌入核心引擎
///
/// 实现意图和视频嵌入的双向循环：
/// 1. 用户查询 → 意图分析 → 提取策略
/// 2. 视频流 → 根据策略选择关键帧 → 存储 embedding
/// 3. embedding 分析 → 增强意图理解
/// 4. 意图更新 → 调整提取策略 → 回到 2
pub struct IntentEmbeddingEngine {
    /// 意图状态
    intent: IntentState,
    /// embedding 存储
    store: IntentDrivenStore,
    /// 帧选择器
    selector: FrameSelector,
    /// 事件发送器
    event_tx: Option<mpsc::UnboundedSender<EngineEvent>>,
    /// 上一次分析时间
    last_analysis_time: f64,
    /// 分析间隔
    analysis_interval: f64,
}

impl IntentEmbeddingEngine {
    /// 从用户查询创建引擎
    pub fn new(initial_query: &str) -> Self {
        let intent = IntentState::new(initial_query);
        let strategy = intent.current_strategy();

        info!(
            "引擎初始化: 意图={:?}, 策略={:?}",
            intent.primary, strategy
        );

        Self {
            intent,
            store: IntentDrivenStore::new(200),
            selector: FrameSelector::new(strategy),
            event_tx: None,
            last_analysis_time: 0.0,
            analysis_interval: 2.0, // 每2秒分析一次
        }
    }

    /// 设置事件通道
    pub fn with_event_channel(mut self, tx: mpsc::UnboundedSender<EngineEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// 处理新帧和 embedding
    ///
    /// 这是核心循环的一部分
    pub fn process(&mut self, frame: Frame, embedding: Array1<f32>, change_score: f32) {
        let timestamp = frame.timestamp;

        // 1. 根据当前策略评估帧
        let selected = self.selector.evaluate(frame, change_score);

        // 2. 存储 embedding
        self.store.store(timestamp, embedding, &self.intent, change_score);

        // 3. 如果选中了关键帧，发送事件
        if selected {
            self.emit(EngineEvent::KeyframeSelected {
                timestamp,
                reason: format!("change_score={:.3}", change_score),
            });
        }

        // 4. 定期分析 embedding 以增强意图
        if timestamp - self.last_analysis_time >= self.analysis_interval {
            self.analyze_and_update();
            self.last_analysis_time = timestamp;
        }
    }

    /// 用户反馈更新
    ///
    /// 用户的新查询可能调整意图
    pub fn update_from_user(&mut self, query: &str) {
        let old_strategy = self.intent.current_strategy();

        self.intent.update_from_user_feedback(query);

        let new_strategy = self.intent.current_strategy();

        // 更新选择器策略
        self.selector.set_strategy(new_strategy);

        // 发送事件
        self.emit(EngineEvent::IntentUpdated {
            primary: self.intent.primary,
            confidence: self.intent.primary_confidence,
            iteration: self.intent.iteration,
        });

        if old_strategy != new_strategy {
            self.emit(EngineEvent::StrategyChanged {
                old: old_strategy,
                new: new_strategy,
            });
        }
    }

    /// 分析 embedding 并更新意图
    fn analyze_and_update(&mut self) {
        let analysis = self.store.analyze();

        let old_strategy = self.intent.current_strategy();

        // 用分析结果更新意图
        self.intent.update_from_embedding_analysis(&analysis);

        let new_strategy = self.intent.current_strategy();

        // 更新选择器策略
        self.selector.set_strategy(new_strategy);

        // 发送事件
        self.emit(EngineEvent::AnalysisComplete {
            activity: analysis.activity_level,
            has_change: analysis.has_significant_change,
        });

        if old_strategy != new_strategy {
            self.emit(EngineEvent::StrategyChanged {
                old: old_strategy,
                new: new_strategy,
            });

            info!(
                "策略自动调整: {:?} -> {:?} (基于 embedding 分析)",
                old_strategy, new_strategy
            );
        }
    }

    /// 获取当前意图相关的 embedding
    pub fn get_relevant_embeddings(&self) -> Vec<(f64, &Array1<f32>)> {
        self.store.get_for_intent(&self.intent)
    }

    /// 获取选中的关键帧
    pub fn get_keyframes(&self, count: usize) -> Vec<&Frame> {
        self.selector.get_keyframes(count)
    }

    /// 获取当前意图状态
    pub fn intent(&self) -> &IntentState {
        &self.intent
    }

    /// 获取当前策略
    pub fn current_strategy(&self) -> ExtractionStrategy {
        self.intent.current_strategy()
    }

    /// 获取统计信息
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            intent: self.intent.primary,
            confidence: self.intent.primary_confidence,
            iteration: self.intent.iteration,
            strategy: self.intent.current_strategy(),
            embedding_count: self.store.len(),
            keyframe_count: self.selector.len(),
            semantic_cues: self.intent.semantic_cues.clone(),
        }
    }

    /// 重置
    pub fn reset(&mut self, new_query: &str) {
        self.intent = IntentState::new(new_query);
        self.store.clear();
        self.selector.clear();
        self.selector.set_strategy(self.intent.current_strategy());
        self.last_analysis_time = 0.0;
    }

    fn emit(&self, event: EngineEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }
}

/// 引擎统计
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub intent: IntentType,
    pub confidence: f32,
    pub iteration: u32,
    pub strategy: ExtractionStrategy,
    pub embedding_count: usize,
    pub keyframe_count: usize,
    pub semantic_cues: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_basic() {
        let mut engine = IntentEmbeddingEngine::new("描述这个视频");

        assert_eq!(engine.intent().primary, IntentType::Describe);
        assert_eq!(engine.current_strategy(), ExtractionStrategy::UniformWithChange);

        // 模拟处理
        for i in 0..10 {
            let frame = Frame::from_vec(vec![0u8; 100], i as f64 * 0.5, 10, 10);
            let emb = Array1::from_vec(vec![0.1f32; 1024]);
            let change = if i == 5 { 0.2 } else { 0.02 };
            engine.process(frame, emb, change);
        }

        assert!(engine.stats().embedding_count > 0);
    }

    #[test]
    fn test_intent_update() {
        let mut engine = IntentEmbeddingEngine::new("描述场景");

        // 用户更新查询
        engine.update_from_user("什么时候出现了人?");

        // 意图应该变为 Locate
        assert_eq!(engine.intent().primary, IntentType::Locate);
        assert_eq!(engine.current_strategy(), ExtractionStrategy::DenseWithPeaks);
    }
}

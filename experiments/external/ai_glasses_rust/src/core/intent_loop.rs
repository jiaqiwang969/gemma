//! 意图-嵌入核心循环
//!
//! 核心机制：
//! 1. 用户意图 → 决定 V-JEPA2 提取策略
//! 2. V-JEPA2 embedding → 增强意图分析
//! 3. 意图更新 → 调整提取策略
//! 4. 循环迭代直到理解充分

use std::collections::HashMap;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::buffer::embedding::EmbeddingCache;

// ============================================================
// 意图状态
// ============================================================

/// 用户意图类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentType {
    /// 描述场景
    Describe,
    /// 定位内容
    Locate,
    /// 对比变化
    Compare,
    /// 总结概括
    Summarize,
    /// 关注音频
    AudioFocus,
    /// 跟踪物体
    Track,
    /// 通用
    General,
}

impl IntentType {
    /// 从查询推断意图
    pub fn from_query(query: &str) -> (Self, f32) {
        let q = query.to_lowercase();

        let patterns: &[(Self, &[&str])] = &[
            (Self::Describe, &["描述", "说明", "看到", "是什么", "describe", "what", "show", "see"]),
            (Self::Locate, &["什么时候", "哪里", "出现", "找", "when", "where", "find", "locate"]),
            (Self::Compare, &["变化", "区别", "对比", "不同", "compare", "change", "differ"]),
            (Self::Summarize, &["总结", "概括", "summary", "brief", "主要"]),
            (Self::AudioFocus, &["说了", "声音", "音频", "听", "audio", "sound", "say", "speak"]),
            (Self::Track, &["跟踪", "追踪", "一直", "track", "follow"]),
        ];

        let mut best = (Self::General, 0usize);
        for (intent, keywords) in patterns {
            let count = keywords.iter().filter(|kw| q.contains(*kw)).count();
            if count > best.1 {
                best = (*intent, count);
            }
        }

        let confidence = (best.1 as f32 / 2.0).min(1.0).max(0.3);
        (best.0, confidence)
    }

    /// 获取对应的提取策略
    pub fn extraction_strategy(&self) -> ExtractionStrategy {
        match self {
            Self::Describe => ExtractionStrategy::UniformWithChange,
            Self::Locate => ExtractionStrategy::DenseWithPeaks,
            Self::Compare => ExtractionStrategy::StartEndTransition,
            Self::Summarize => ExtractionStrategy::Representative,
            Self::AudioFocus => ExtractionStrategy::StableFrames,
            Self::Track => ExtractionStrategy::HighFrequency,
            Self::General => ExtractionStrategy::UniformWithChange,
        }
    }
}

/// 意图状态 (动态演化)
#[derive(Debug, Clone)]
pub struct IntentState {
    /// 主要意图
    pub primary: IntentType,
    /// 主要意图置信度
    pub primary_confidence: f32,

    /// 次要意图 (可能同时存在多个)
    pub secondary: Vec<(IntentType, f32)>,

    /// 意图历史 (用于趋势分析)
    pub history: Vec<IntentType>,

    /// 累积的语义线索 (从 embedding 分析中获得)
    pub semantic_cues: Vec<String>,

    /// 迭代次数
    pub iteration: u32,
}

impl IntentState {
    pub fn new(query: &str) -> Self {
        let (primary, confidence) = IntentType::from_query(query);
        Self {
            primary,
            primary_confidence: confidence,
            secondary: Vec::new(),
            history: vec![primary],
            semantic_cues: Vec::new(),
            iteration: 0,
        }
    }

    /// 从 embedding 分析更新意图
    pub fn update_from_embedding_analysis(&mut self, analysis: &EmbeddingAnalysis) {
        self.iteration += 1;

        // 根据 embedding 分析调整意图
        if analysis.has_significant_change && self.primary != IntentType::Compare {
            self.add_secondary(IntentType::Compare, 0.5);
        }

        if analysis.has_stable_regions && self.primary != IntentType::AudioFocus {
            self.add_secondary(IntentType::AudioFocus, 0.3);
        }

        if analysis.has_unique_frames && self.primary != IntentType::Locate {
            self.add_secondary(IntentType::Locate, 0.4);
        }

        // 添加语义线索
        for cue in &analysis.detected_cues {
            if !self.semantic_cues.contains(cue) {
                self.semantic_cues.push(cue.clone());
            }
        }

        // 如果次要意图置信度超过主要意图，考虑切换
        self.maybe_switch_primary();
    }

    /// 用户反馈更新意图
    pub fn update_from_user_feedback(&mut self, new_query: &str) {
        let (new_intent, confidence) = IntentType::from_query(new_query);
        self.history.push(new_intent);

        // 如果新意图与主要意图不同，可能需要切换
        if new_intent != self.primary && confidence > self.primary_confidence {
            self.primary = new_intent;
            self.primary_confidence = confidence;
        } else {
            self.add_secondary(new_intent, confidence);
        }

        self.iteration += 1;
    }

    fn add_secondary(&mut self, intent: IntentType, confidence: f32) {
        if intent == self.primary {
            return;
        }

        // 查找是否已存在
        if let Some(pos) = self.secondary.iter().position(|(i, _)| *i == intent) {
            // 增加置信度
            self.secondary[pos].1 = (self.secondary[pos].1 + confidence).min(1.0);
        } else {
            self.secondary.push((intent, confidence));
        }

        // 按置信度排序
        self.secondary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 限制数量
        if self.secondary.len() > 3 {
            self.secondary.truncate(3);
        }
    }

    fn maybe_switch_primary(&mut self) {
        if let Some((intent, conf)) = self.secondary.first() {
            if *conf > self.primary_confidence + 0.2 {
                let old_primary = self.primary;
                self.primary = *intent;
                self.primary_confidence = *conf;
                self.secondary.retain(|(i, _)| *i != self.primary);
                self.secondary.push((old_primary, self.primary_confidence - 0.2));
            }
        }
    }

    /// 获取当前最优提取策略
    pub fn current_strategy(&self) -> ExtractionStrategy {
        self.primary.extraction_strategy()
    }

    /// 获取策略权重 (主+次要)
    pub fn strategy_weights(&self) -> Vec<(ExtractionStrategy, f32)> {
        let mut weights = vec![(self.primary.extraction_strategy(), self.primary_confidence)];
        for (intent, conf) in &self.secondary {
            weights.push((intent.extraction_strategy(), *conf * 0.5));
        }
        weights
    }
}

// ============================================================
// 提取策略
// ============================================================

/// 视频提取策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionStrategy {
    /// 均匀采样 + 变化检测
    UniformWithChange,
    /// 密集采样 + 峰值检测
    DenseWithPeaks,
    /// 首尾 + 转折点
    StartEndTransition,
    /// 代表性帧
    Representative,
    /// 稳定帧 (适合音频分析)
    StableFrames,
    /// 高频采样 (用于跟踪)
    HighFrequency,
}

impl ExtractionStrategy {
    /// 获取建议的关键帧数量
    pub fn suggested_keyframe_count(&self, video_duration: f64) -> usize {
        match self {
            Self::UniformWithChange => ((video_duration / 2.0) as usize).clamp(3, 8),
            Self::DenseWithPeaks => ((video_duration * 2.0) as usize).clamp(5, 15),
            Self::StartEndTransition => 4,
            Self::Representative => 3,
            Self::StableFrames => 2,
            Self::HighFrequency => ((video_duration * 5.0) as usize).clamp(10, 30),
        }
    }

    /// 获取最小采样间隔 (秒)
    pub fn min_interval(&self) -> f64 {
        match self {
            Self::UniformWithChange => 0.5,
            Self::DenseWithPeaks => 0.2,
            Self::StartEndTransition => 1.0,
            Self::Representative => 2.0,
            Self::StableFrames => 1.0,
            Self::HighFrequency => 0.1,
        }
    }
}

// ============================================================
// Embedding 分析结果
// ============================================================

/// Embedding 分析结果 (用于反馈意图)
#[derive(Debug, Clone, Default)]
pub struct EmbeddingAnalysis {
    /// 是否有显著变化
    pub has_significant_change: bool,
    /// 最大变化分数
    pub max_change_score: f32,
    /// 是否有稳定区域
    pub has_stable_regions: bool,
    /// 稳定区域时间范围
    pub stable_regions: Vec<(f64, f64)>,
    /// 是否有独特帧
    pub has_unique_frames: bool,
    /// 独特帧时间点
    pub unique_frame_times: Vec<f64>,
    /// 检测到的语义线索
    pub detected_cues: Vec<String>,
    /// 活动级别
    pub activity_level: ActivityLevel,
}

/// 活动级别
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ActivityLevel {
    Static,
    Low,
    #[default]
    Medium,
    High,
}

// ============================================================
// 意图驱动的 Embedding 存储
// ============================================================

/// 意图驱动的 Embedding 存储
///
/// 根据当前意图决定如何存储和索引 embedding
pub struct IntentDrivenStore {
    /// 按时间存储的 embedding
    temporal_embeddings: Vec<(f64, Array1<f32>)>,
    /// 按意图类型索引的 embedding
    intent_index: HashMap<IntentType, Vec<usize>>,
    /// 变化点 embedding
    change_point_embeddings: Vec<(f64, Array1<f32>, f32)>, // (time, emb, score)
    /// 代表性 embedding (全局语义)
    representative_embedding: Option<Array1<f32>>,
    /// 最大存储数量
    max_size: usize,
}

impl IntentDrivenStore {
    pub fn new(max_size: usize) -> Self {
        Self {
            temporal_embeddings: Vec::with_capacity(max_size),
            intent_index: HashMap::new(),
            change_point_embeddings: Vec::new(),
            representative_embedding: None,
            max_size,
        }
    }

    /// 根据意图存储 embedding
    pub fn store(&mut self, timestamp: f64, embedding: Array1<f32>, intent: &IntentState, change_score: f32) {
        let idx = self.temporal_embeddings.len();

        // 存储到时间序列
        self.temporal_embeddings.push((timestamp, embedding.clone()));

        // 根据意图类型索引
        self.intent_index
            .entry(intent.primary)
            .or_insert_with(Vec::new)
            .push(idx);

        // 如果是变化点，额外存储
        if change_score > 0.1 {
            self.change_point_embeddings.push((timestamp, embedding.clone(), change_score));
        }

        // 更新代表性 embedding
        self.update_representative(&embedding);

        // 维护大小限制
        self.prune_if_needed(intent);
    }

    /// 根据当前意图获取相关 embedding
    pub fn get_for_intent(&self, intent: &IntentState) -> Vec<(f64, &Array1<f32>)> {
        let strategy = intent.current_strategy();

        match strategy {
            ExtractionStrategy::UniformWithChange | ExtractionStrategy::DenseWithPeaks => {
                // 返回变化点 + 均匀采样
                let mut result: Vec<_> = self.change_point_embeddings
                    .iter()
                    .map(|(t, e, _)| (*t, e))
                    .collect();

                // 补充均匀采样
                let step = (self.temporal_embeddings.len() / 5).max(1);
                for (i, (t, e)) in self.temporal_embeddings.iter().enumerate() {
                    if i % step == 0 && !result.iter().any(|(rt, _)| (*rt - *t).abs() < 0.5) {
                        result.push((*t, e));
                    }
                }

                result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                result
            }

            ExtractionStrategy::StartEndTransition => {
                let mut result = Vec::new();
                // 开始
                if let Some((t, e)) = self.temporal_embeddings.first() {
                    result.push((*t, e));
                }
                // 最大变化点
                if let Some((t, e, _)) = self.change_point_embeddings.iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) {
                    result.push((*t, e));
                }
                // 结束
                if let Some((t, e)) = self.temporal_embeddings.last() {
                    result.push((*t, e));
                }
                result
            }

            ExtractionStrategy::Representative => {
                if let Some(ref rep) = self.representative_embedding {
                    // 找到最接近代表性 embedding 的
                    let mut scored: Vec<_> = self.temporal_embeddings.iter()
                        .map(|(t, e)| (*t, e, EmbeddingCache::cosine_similarity(e, rep)))
                        .collect();
                    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                    scored.into_iter().take(3).map(|(t, e, _)| (t, e)).collect()
                } else {
                    self.temporal_embeddings.iter().take(3).map(|(t, e)| (*t, e)).collect()
                }
            }

            ExtractionStrategy::StableFrames => {
                // 选择变化分数最低的
                let mut scored: Vec<_> = self.temporal_embeddings.iter().enumerate()
                    .map(|(i, (t, e))| {
                        let change = self.change_point_embeddings.iter()
                            .find(|(ct, _, _)| (*ct - *t).abs() < 0.3)
                            .map(|(_, _, s)| *s)
                            .unwrap_or(0.0);
                        (*t, e, change)
                    })
                    .collect();
                scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                scored.into_iter().take(3).map(|(t, e, _)| (t, e)).collect()
            }

            ExtractionStrategy::HighFrequency => {
                // 返回所有
                self.temporal_embeddings.iter().map(|(t, e)| (*t, e)).collect()
            }
        }
    }

    /// 分析 embedding 以增强意图理解
    pub fn analyze(&self) -> EmbeddingAnalysis {
        let mut analysis = EmbeddingAnalysis::default();

        if self.temporal_embeddings.is_empty() {
            return analysis;
        }

        // 分析变化
        if !self.change_point_embeddings.is_empty() {
            analysis.max_change_score = self.change_point_embeddings.iter()
                .map(|(_, _, s)| *s)
                .fold(0.0f32, f32::max);
            analysis.has_significant_change = analysis.max_change_score > 0.15;
        }

        // 检测稳定区域
        let mut stable_start: Option<f64> = None;
        for i in 1..self.temporal_embeddings.len() {
            let (t1, e1) = &self.temporal_embeddings[i - 1];
            let (t2, e2) = &self.temporal_embeddings[i];
            let sim = EmbeddingCache::cosine_similarity(e1, e2);

            if sim > 0.95 {
                if stable_start.is_none() {
                    stable_start = Some(*t1);
                }
            } else if let Some(start) = stable_start {
                if *t1 - start > 1.0 {
                    analysis.stable_regions.push((start, *t1));
                    analysis.has_stable_regions = true;
                }
                stable_start = None;
            }
        }

        // 检测独特帧
        if let Some(ref rep) = self.representative_embedding {
            for (t, e) in &self.temporal_embeddings {
                let sim = EmbeddingCache::cosine_similarity(e, rep);
                if sim < 0.8 {
                    analysis.unique_frame_times.push(*t);
                    analysis.has_unique_frames = true;
                }
            }
        }

        // 活动级别
        analysis.activity_level = if analysis.max_change_score < 0.05 {
            ActivityLevel::Static
        } else if analysis.max_change_score < 0.1 {
            ActivityLevel::Low
        } else if analysis.max_change_score < 0.2 {
            ActivityLevel::Medium
        } else {
            ActivityLevel::High
        };

        analysis
    }

    fn update_representative(&mut self, new_emb: &Array1<f32>) {
        match &mut self.representative_embedding {
            Some(rep) => {
                // 指数移动平均
                let alpha = 0.1;
                for (r, n) in rep.iter_mut().zip(new_emb.iter()) {
                    *r = *r * (1.0 - alpha) + *n * alpha;
                }
            }
            None => {
                self.representative_embedding = Some(new_emb.clone());
            }
        }
    }

    fn prune_if_needed(&mut self, intent: &IntentState) {
        if self.temporal_embeddings.len() <= self.max_size {
            return;
        }

        // 根据意图策略决定保留哪些
        let strategy = intent.current_strategy();
        let keep_count = self.max_size * 3 / 4;

        match strategy {
            ExtractionStrategy::DenseWithPeaks | ExtractionStrategy::HighFrequency => {
                // 保留最近的
                let remove_count = self.temporal_embeddings.len() - keep_count;
                self.temporal_embeddings.drain(0..remove_count);
            }
            _ => {
                // 保留变化点和均匀分布的
                let mut keep_indices: Vec<usize> = Vec::new();

                // 保留变化点
                for (ct, _, _) in &self.change_point_embeddings {
                    if let Some(idx) = self.temporal_embeddings.iter()
                        .position(|(t, _)| (*t - *ct).abs() < 0.1) {
                        keep_indices.push(idx);
                    }
                }

                // 均匀补充
                let step = self.temporal_embeddings.len() / (keep_count - keep_indices.len()).max(1);
                for i in (0..self.temporal_embeddings.len()).step_by(step) {
                    if !keep_indices.contains(&i) {
                        keep_indices.push(i);
                    }
                }

                keep_indices.sort();
                keep_indices.dedup();
                keep_indices.truncate(keep_count);

                let new_embs: Vec<_> = keep_indices.iter()
                    .filter_map(|&i| self.temporal_embeddings.get(i).cloned())
                    .collect();
                self.temporal_embeddings = new_embs;
            }
        }

        // 重建索引
        self.intent_index.clear();
    }

    pub fn len(&self) -> usize {
        self.temporal_embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.temporal_embeddings.is_empty()
    }

    pub fn clear(&mut self) {
        self.temporal_embeddings.clear();
        self.intent_index.clear();
        self.change_point_embeddings.clear();
        self.representative_embedding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_from_query() {
        let (intent, _) = IntentType::from_query("描述一下这个视频");
        assert_eq!(intent, IntentType::Describe);

        let (intent, _) = IntentType::from_query("什么时候出现了人");
        assert_eq!(intent, IntentType::Locate);

        let (intent, _) = IntentType::from_query("视频前后有什么变化");
        assert_eq!(intent, IntentType::Compare);
    }

    #[test]
    fn test_intent_state_update() {
        let mut state = IntentState::new("描述场景");
        assert_eq!(state.primary, IntentType::Describe);

        // 模拟 embedding 分析结果
        let analysis = EmbeddingAnalysis {
            has_significant_change: true,
            max_change_score: 0.3,
            ..Default::default()
        };

        state.update_from_embedding_analysis(&analysis);
        assert!(state.secondary.iter().any(|(i, _)| *i == IntentType::Compare));
    }
}

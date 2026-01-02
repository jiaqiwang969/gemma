//! 流处理器
//!
//! 实时视频/音频流处理的核心组件

use std::sync::Arc;
use ndarray::Array1;
use tokio::sync::mpsc;
use tracing::{info, debug};

use crate::buffer::{FrameBuffer, AudioBuffer, EmbeddingCache};
use crate::buffer::frame::Frame;
use crate::ai::client::AiClient;

/// 变化检测器
pub struct ChangeDetector {
    /// 基础阈值
    base_threshold: f32,
    /// 变化历史
    change_history: Vec<f32>,
    /// 最大历史长度
    max_history: usize,
    /// 上一个 embedding
    last_embedding: Option<Array1<f32>>,
}

impl ChangeDetector {
    pub fn new(base_threshold: f32) -> Self {
        Self {
            base_threshold,
            change_history: Vec::with_capacity(100),
            max_history: 100,
            last_embedding: None,
        }
    }

    /// 检测变化
    ///
    /// 返回 (变化分数, 是否显著变化)
    pub fn detect(&mut self, current: &Array1<f32>) -> (f32, bool) {
        let change_score = match &self.last_embedding {
            Some(last) => {
                let similarity = EmbeddingCache::cosine_similarity(last, current);
                (1.0 - similarity).max(0.0)
            }
            None => 0.0,
        };

        // 记录历史
        if self.change_history.len() >= self.max_history {
            self.change_history.remove(0);
        }
        self.change_history.push(change_score);

        // 自适应阈值
        let threshold = self.adaptive_threshold();
        let is_significant = change_score > threshold;

        // 更新 last embedding
        self.last_embedding = Some(current.clone());

        (change_score, is_significant)
    }

    fn adaptive_threshold(&self) -> f32 {
        if self.change_history.len() < 10 {
            return self.base_threshold;
        }

        let mean: f32 = self.change_history.iter().sum::<f32>() / self.change_history.len() as f32;
        let variance: f32 = self.change_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.change_history.len() as f32;
        let std = variance.sqrt();

        (mean + std).max(self.base_threshold)
    }

    /// 获取活动级别
    pub fn activity_level(&self) -> &'static str {
        if self.change_history.len() < 5 {
            return "unknown";
        }

        let max_change = self.change_history.iter().cloned().fold(0.0f32, f32::max);

        if max_change < 0.03 {
            "static"
        } else if max_change < 0.08 {
            "low"
        } else if max_change < 0.2 {
            "medium"
        } else {
            "high"
        }
    }

    pub fn reset(&mut self) {
        self.change_history.clear();
        self.last_embedding = None;
    }
}

/// 关键帧选择器
pub struct KeyframeSelector {
    /// 最大关键帧数
    max_keyframes: usize,
    /// 最小时间间隔
    min_interval: f64,
    /// 关键帧列表: (帧, 时间戳, 分数)
    keyframes: Vec<(Frame, f64, f32)>,
}

impl KeyframeSelector {
    pub fn new(max_keyframes: usize, min_interval: f64) -> Self {
        Self {
            max_keyframes,
            min_interval,
            keyframes: Vec::with_capacity(max_keyframes * 2),
        }
    }

    /// 添加关键帧候选
    pub fn add_candidate(&mut self, frame: Frame, change_score: f32, force: bool) -> bool {
        let timestamp = frame.timestamp;

        // 检查时间间隔
        if !force && !self.keyframes.is_empty() {
            let last_time = self.keyframes.last().unwrap().1;
            if timestamp - last_time < self.min_interval {
                return false;
            }
        }

        self.keyframes.push((frame, timestamp, change_score));

        // 修剪
        if self.keyframes.len() > self.max_keyframes * 2 {
            self.prune();
        }

        true
    }

    fn prune(&mut self) {
        if self.keyframes.len() <= self.max_keyframes {
            return;
        }

        // 保留首帧
        let first = self.keyframes.remove(0);

        // 按分数排序
        self.keyframes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // 保留最高分的
        self.keyframes.truncate(self.max_keyframes - 1);

        // 按时间排序
        self.keyframes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // 加回首帧
        self.keyframes.insert(0, first);
    }

    /// 获取关键帧
    pub fn get_keyframes(&self, count: Option<usize>) -> Vec<(Frame, f64)> {
        let count = count.unwrap_or(self.max_keyframes);
        let n = count.min(self.keyframes.len());

        if n == self.keyframes.len() {
            return self.keyframes.iter()
                .map(|(f, t, _)| (f.clone(), *t))
                .collect();
        }

        // 均匀采样
        let step = self.keyframes.len() as f64 / n as f64;
        (0..n)
            .map(|i| {
                let idx = (i as f64 * step) as usize;
                let (f, t, _) = &self.keyframes[idx];
                (f.clone(), *t)
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.keyframes.clear();
    }

    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

/// 流处理事件
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// 场景变化
    SceneChange { timestamp: f64, score: f32 },
    /// 新关键帧
    NewKeyframe { timestamp: f64 },
    /// Embedding 更新
    EmbeddingUpdate { timestamp: f64 },
}

/// 流处理器配置
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// 编码间隔 (秒)
    pub encoding_interval: f64,
    /// 目标关键帧数
    pub keyframe_count: usize,
    /// 变化检测阈值
    pub change_threshold: f32,
    /// 最小关键帧间隔
    pub min_keyframe_interval: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            encoding_interval: 0.5,
            keyframe_count: 5,
            change_threshold: 0.05,
            min_keyframe_interval: 0.5,
        }
    }
}

/// 流处理器状态
#[derive(Debug, Clone)]
pub struct StreamState {
    pub current_time: f64,
    pub frame_count: usize,
    pub audio_duration: f64,
    pub embedding_count: usize,
    pub keyframe_count: usize,
    pub activity_level: String,
}

/// 流处理器
pub struct StreamProcessor {
    /// 配置
    config: StreamConfig,
    /// 帧缓冲
    frame_buffer: FrameBuffer,
    /// 音频缓冲
    audio_buffer: AudioBuffer,
    /// Embedding 缓存
    embedding_cache: EmbeddingCache,
    /// 变化检测器
    change_detector: ChangeDetector,
    /// 关键帧选择器
    keyframe_selector: KeyframeSelector,
    /// AI 客户端
    ai_client: Option<Arc<AiClient>>,
    /// 上次编码时间
    last_encoding_time: f64,
    /// 当前时间
    current_time: f64,
    /// 事件发送器
    event_tx: Option<mpsc::UnboundedSender<StreamEvent>>,
}

impl StreamProcessor {
    /// 创建新的流处理器
    pub fn new(config: StreamConfig) -> Self {
        Self {
            frame_buffer: FrameBuffer::new(120),
            audio_buffer: AudioBuffer::new(16000, 30.0),
            embedding_cache: EmbeddingCache::new(60, 1024),
            change_detector: ChangeDetector::new(config.change_threshold),
            keyframe_selector: KeyframeSelector::new(config.keyframe_count, config.min_keyframe_interval),
            config,
            ai_client: None,
            last_encoding_time: 0.0,
            current_time: 0.0,
            event_tx: None,
        }
    }

    /// 设置 AI 客户端
    pub fn with_ai_client(mut self, client: Arc<AiClient>) -> Self {
        self.ai_client = Some(client);
        self
    }

    /// 设置事件通道
    pub fn with_event_channel(mut self, tx: mpsc::UnboundedSender<StreamEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// 处理视频帧
    pub async fn process_frame(&mut self, frame: Frame) -> anyhow::Result<()> {
        self.current_time = frame.timestamp;

        // 添加到缓冲区
        self.frame_buffer.push(frame.clone());

        // 检查是否需要编码
        if self.current_time - self.last_encoding_time >= self.config.encoding_interval {
            self.encode_and_detect().await?;
            self.last_encoding_time = self.current_time;
        }

        Ok(())
    }

    /// 处理音频块
    pub fn process_audio(&mut self, chunk: &[f32], timestamp: f64) {
        self.audio_buffer.push(chunk, timestamp);
    }

    /// 编码并检测变化
    async fn encode_and_detect(&mut self) -> anyhow::Result<()> {
        let ai_client = match &self.ai_client {
            Some(c) => c.clone(),
            None => return Ok(()), // 没有 AI 客户端，跳过
        };

        // 获取窗口
        let (frames, _) = self.frame_buffer.get_window(16);
        if frames.len() < 4 {
            return Ok(());
        }

        // V-JEPA2 编码
        let embeddings = ai_client.encode_frames(&frames).await?;

        // 使用最后一帧的 embedding
        if let Some(current) = embeddings.last() {
            // 添加到缓存
            self.embedding_cache.push(current.clone(), self.current_time);

            // 发送事件
            if let Some(tx) = &self.event_tx {
                let _ = tx.send(StreamEvent::EmbeddingUpdate {
                    timestamp: self.current_time,
                });
            }

            // 变化检测
            let (score, is_significant) = self.change_detector.detect(current);

            if is_significant {
                debug!("Scene change detected: score={:.3}", score);

                // 发送事件
                if let Some(tx) = &self.event_tx {
                    let _ = tx.send(StreamEvent::SceneChange {
                        timestamp: self.current_time,
                        score,
                    });
                }
            }

            // 关键帧选择
            if is_significant || self.keyframe_selector.is_empty() {
                if let Some(frame) = frames.last() {
                    let added = self.keyframe_selector.add_candidate(
                        frame.clone(),
                        score,
                        self.keyframe_selector.is_empty(),
                    );

                    if added {
                        if let Some(tx) = &self.event_tx {
                            let _ = tx.send(StreamEvent::NewKeyframe {
                                timestamp: self.current_time,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// 获取当前状态
    pub fn state(&self) -> StreamState {
        StreamState {
            current_time: self.current_time,
            frame_count: self.frame_buffer.len(),
            audio_duration: self.audio_buffer.duration(),
            embedding_count: self.embedding_cache.len(),
            keyframe_count: self.keyframe_selector.len(),
            activity_level: self.change_detector.activity_level().to_string(),
        }
    }

    /// 获取关键帧及其音频
    pub fn get_keyframes_with_audio(&self) -> Vec<KeyframeWithAudio> {
        let keyframes = self.keyframe_selector.get_keyframes(None);

        keyframes.into_iter().map(|(frame, timestamp)| {
            // 获取前后 1 秒的音频
            let audio = self.audio_buffer.get_segment(timestamp - 1.0, timestamp + 1.0);

            KeyframeWithAudio {
                frame,
                timestamp,
                audio,
            }
        }).collect()
    }

    /// 获取变化曲线
    pub fn change_profile(&self) -> Vec<f32> {
        self.change_detector.change_history.clone()
    }

    /// 获取全局 embedding
    pub fn global_embedding(&self) -> Option<Array1<f32>> {
        self.embedding_cache.get_global_embedding()
    }

    /// 重置
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.audio_buffer.clear();
        self.embedding_cache.clear();
        self.change_detector.reset();
        self.keyframe_selector.clear();
        self.last_encoding_time = 0.0;
        self.current_time = 0.0;
    }
}

/// 带音频的关键帧
#[derive(Debug, Clone)]
pub struct KeyframeWithAudio {
    pub frame: Frame,
    pub timestamp: f64,
    pub audio: Option<Vec<f32>>,
}

impl KeyframeWithAudio {
    pub fn has_audio(&self) -> bool {
        self.audio.as_ref().map(|a| !a.is_empty()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn test_change_detector() {
        let mut detector = ChangeDetector::new(0.05);

        // 相同的 embedding
        let emb1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let emb2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        let (_, is_sig) = detector.detect(&emb1);
        assert!(!is_sig); // 第一帧

        let (score, is_sig) = detector.detect(&emb2);
        assert!(score < 0.01); // 相同
        assert!(!is_sig);

        // 不同的 embedding
        let emb3 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let (score, _) = detector.detect(&emb3);
        assert!(score > 0.5); // 显著变化
    }

    #[test]
    fn test_keyframe_selector() {
        let mut selector = KeyframeSelector::new(3, 0.1);

        for i in 0..10 {
            let frame = Frame::from_vec(vec![0u8; 100], i as f64 * 0.2, 10, 10);
            selector.add_candidate(frame, i as f32 * 0.1, i == 0);
        }

        let keyframes = selector.get_keyframes(None);
        assert!(keyframes.len() <= 3);
    }
}

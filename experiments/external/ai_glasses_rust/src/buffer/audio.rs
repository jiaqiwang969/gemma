//! 音频缓冲区
//!
//! 实时音频流的环形缓冲区

use std::sync::Arc;
use parking_lot::RwLock;

/// 音频缓冲区
pub struct AudioBuffer {
    /// 音频数据 (f32, 单声道)
    buffer: Arc<RwLock<Vec<f32>>>,
    /// 采样率
    sample_rate: u32,
    /// 最大持续时间 (秒)
    max_duration: f64,
    /// 缓冲区开始时间
    start_time: Arc<RwLock<f64>>,
}

impl AudioBuffer {
    /// 创建新的音频缓冲区
    ///
    /// # Arguments
    /// * `sample_rate` - 采样率 (例如: 16000)
    /// * `max_duration` - 最大缓存时长 (秒)
    pub fn new(sample_rate: u32, max_duration: f64) -> Self {
        let max_samples = (sample_rate as f64 * max_duration) as usize;
        Self {
            buffer: Arc::new(RwLock::new(Vec::with_capacity(max_samples))),
            sample_rate,
            max_duration,
            start_time: Arc::new(RwLock::new(0.0)),
        }
    }

    /// 添加音频块
    pub fn push(&self, chunk: &[f32], timestamp: f64) {
        let mut buffer = self.buffer.write();
        let mut start_time = self.start_time.write();

        if buffer.is_empty() {
            *start_time = timestamp;
        }

        buffer.extend_from_slice(chunk);

        // 限制最大长度
        let max_samples = (self.sample_rate as f64 * self.max_duration) as usize;
        if buffer.len() > max_samples {
            let excess = buffer.len() - max_samples;
            buffer.drain(0..excess);
            *start_time = timestamp - self.max_duration;
        }
    }

    /// 获取时间段内的音频
    pub fn get_segment(&self, start: f64, end: f64) -> Option<Vec<f32>> {
        let buffer = self.buffer.read();
        let buf_start = *self.start_time.read();

        let start_idx = ((start - buf_start) * self.sample_rate as f64) as isize;
        let end_idx = ((end - buf_start) * self.sample_rate as f64) as isize;

        let start_idx = start_idx.max(0) as usize;
        let end_idx = (end_idx as usize).min(buffer.len());

        if start_idx >= end_idx {
            return None;
        }

        Some(buffer[start_idx..end_idx].to_vec())
    }

    /// 获取最近的音频
    pub fn get_recent(&self, duration: f64) -> Vec<f32> {
        let buffer = self.buffer.read();
        let samples = (duration * self.sample_rate as f64) as usize;
        let n = samples.min(buffer.len());
        buffer[buffer.len() - n..].to_vec()
    }

    /// 清空缓冲区
    pub fn clear(&self) {
        self.buffer.write().clear();
        *self.start_time.write() = 0.0;
    }

    /// 获取持续时间
    pub fn duration(&self) -> f64 {
        self.buffer.read().len() as f64 / self.sample_rate as f64
    }

    /// 获取样本数
    pub fn len(&self) -> usize {
        self.buffer.read().len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.buffer.read().is_empty()
    }

    /// 计算 RMS 能量
    pub fn rms_energy(&self, start: f64, end: f64) -> Option<f32> {
        let segment = self.get_segment(start, end)?;
        if segment.is_empty() {
            return None;
        }

        let sum_sq: f32 = segment.iter().map(|x| x * x).sum();
        Some((sum_sq / segment.len() as f32).sqrt())
    }

    /// 检测是否有语音活动 (简单能量阈值)
    pub fn has_speech(&self, start: f64, end: f64, threshold: f32) -> bool {
        self.rms_energy(start, end)
            .map(|energy| energy > threshold)
            .unwrap_or(false)
    }
}

impl Clone for AudioBuffer {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            sample_rate: self.sample_rate,
            max_duration: self.max_duration,
            start_time: Arc::clone(&self.start_time),
        }
    }
}

impl Default for AudioBuffer {
    fn default() -> Self {
        Self::new(16000, 30.0) // 16kHz, 30秒
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer() {
        let buffer = AudioBuffer::new(16000, 5.0);

        // 添加 1 秒的音频
        let chunk: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();
        buffer.push(&chunk, 0.0);

        assert_eq!(buffer.len(), 16000);
        assert!((buffer.duration() - 1.0).abs() < 0.01);

        // 获取片段
        let segment = buffer.get_segment(0.0, 0.5).unwrap();
        assert_eq!(segment.len(), 8000);
    }
}

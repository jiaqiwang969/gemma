//! 帧缓冲区
//!
//! 高性能的视频帧环形缓冲区

use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;

/// 视频帧
#[derive(Clone, Debug)]
pub struct Frame {
    /// 帧数据 (RGB, 零拷贝)
    pub data: Bytes,
    /// 时间戳 (秒)
    pub timestamp: f64,
    /// 宽度
    pub width: u32,
    /// 高度
    pub height: u32,
}

impl Frame {
    pub fn new(data: Bytes, timestamp: f64, width: u32, height: u32) -> Self {
        Self { data, timestamp, width, height }
    }

    /// 从 Vec<u8> 创建
    pub fn from_vec(data: Vec<u8>, timestamp: f64, width: u32, height: u32) -> Self {
        Self {
            data: Bytes::from(data),
            timestamp,
            width,
            height,
        }
    }
}

/// 帧缓冲区
///
/// 线程安全的环形缓冲区，用于存储最近的视频帧
pub struct FrameBuffer {
    frames: Arc<RwLock<VecDeque<Frame>>>,
    max_size: usize,
}

impl FrameBuffer {
    /// 创建新的帧缓冲区
    ///
    /// # Arguments
    /// * `max_size` - 最大帧数 (例如: 120 = 4秒 @ 30fps)
    pub fn new(max_size: usize) -> Self {
        Self {
            frames: Arc::new(RwLock::new(VecDeque::with_capacity(max_size))),
            max_size,
        }
    }

    /// 添加帧
    pub fn push(&self, frame: Frame) {
        let mut frames = self.frames.write();
        if frames.len() >= self.max_size {
            frames.pop_front();
        }
        frames.push_back(frame);
    }

    /// 获取最近的 N 帧
    pub fn get_recent(&self, count: usize) -> Vec<Frame> {
        let frames = self.frames.read();
        let n = count.min(frames.len());
        frames.iter().rev().take(n).cloned().collect::<Vec<_>>().into_iter().rev().collect()
    }

    /// 获取滑动窗口
    pub fn get_window(&self, window_size: usize) -> (Vec<Frame>, Vec<f64>) {
        let frames = self.get_recent(window_size);
        let timestamps: Vec<f64> = frames.iter().map(|f| f.timestamp).collect();
        (frames, timestamps)
    }

    /// 获取所有帧
    pub fn get_all(&self) -> Vec<Frame> {
        self.frames.read().iter().cloned().collect()
    }

    /// 清空缓冲区
    pub fn clear(&self) {
        self.frames.write().clear();
    }

    /// 获取帧数
    pub fn len(&self) -> usize {
        self.frames.read().len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.frames.read().is_empty()
    }

    /// 获取时间范围
    pub fn time_range(&self) -> Option<(f64, f64)> {
        let frames = self.frames.read();
        if frames.is_empty() {
            return None;
        }
        let first = frames.front()?.timestamp;
        let last = frames.back()?.timestamp;
        Some((first, last))
    }
}

impl Clone for FrameBuffer {
    fn clone(&self) -> Self {
        Self {
            frames: Arc::clone(&self.frames),
            max_size: self.max_size,
        }
    }
}

impl Default for FrameBuffer {
    fn default() -> Self {
        Self::new(120) // 4秒 @ 30fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_buffer() {
        let buffer = FrameBuffer::new(5);

        // 添加帧
        for i in 0..10 {
            let frame = Frame::from_vec(vec![0u8; 100], i as f64 * 0.1, 10, 10);
            buffer.push(frame);
        }

        // 检查大小 (应该是 5)
        assert_eq!(buffer.len(), 5);

        // 获取最近的帧
        let recent = buffer.get_recent(3);
        assert_eq!(recent.len(), 3);
        assert!((recent[0].timestamp - 0.7).abs() < 0.01);
    }
}

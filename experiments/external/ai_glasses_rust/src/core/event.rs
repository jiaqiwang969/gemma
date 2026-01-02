//! 事件检测器
//!
//! 检测各种事件：场景变化、用户交互、唤醒词等

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// 事件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// 唤醒词
    WakeWord,
    /// 用户语音
    UserSpeech,
    /// 场景变化
    SceneChange,
    /// 物体出现
    ObjectAppear,
    /// 用户手势
    Gesture,
    /// 超时
    Timeout,
    /// 用户开始说话
    SpeechStart,
    /// 用户停止说话
    SpeechEnd,
}

/// 事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// 事件类型
    pub event_type: EventType,
    /// 时间戳
    pub timestamp: f64,
    /// 置信度 (0-1)
    pub confidence: f32,
    /// 附加数据
    pub data: Option<String>,
}

impl Event {
    pub fn new(event_type: EventType, timestamp: f64) -> Self {
        Self {
            event_type,
            timestamp,
            confidence: 1.0,
            data: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_data(mut self, data: impl Into<String>) -> Self {
        self.data = Some(data.into());
        self
    }
}

/// 事件检测器配置
#[derive(Debug, Clone)]
pub struct EventDetectorConfig {
    /// 场景变化阈值
    pub scene_change_threshold: f32,
    /// 语音能量阈值
    pub speech_energy_threshold: f32,
    /// 语音超时 (秒)
    pub speech_timeout: f64,
    /// 唤醒词 (可选)
    pub wake_word: Option<String>,
}

impl Default for EventDetectorConfig {
    fn default() -> Self {
        Self {
            scene_change_threshold: 0.1,
            speech_energy_threshold: 0.02,
            speech_timeout: 2.0,
            wake_word: Some("Hey Glasses".to_string()),
        }
    }
}

/// 事件检测器
pub struct EventDetector {
    config: EventDetectorConfig,
    /// 事件发送器
    event_tx: Option<mpsc::UnboundedSender<Event>>,
    /// 是否正在说话
    is_speaking: bool,
    /// 最后语音时间
    last_speech_time: f64,
    /// 是否已唤醒
    is_awake: bool,
}

impl EventDetector {
    pub fn new(config: EventDetectorConfig) -> Self {
        Self {
            config,
            event_tx: None,
            is_speaking: false,
            last_speech_time: 0.0,
            is_awake: false,
        }
    }

    /// 设置事件通道
    pub fn with_event_channel(mut self, tx: mpsc::UnboundedSender<Event>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// 检测场景变化
    pub fn on_scene_change(&mut self, timestamp: f64, change_score: f32) {
        if change_score >= self.config.scene_change_threshold {
            self.emit(Event::new(EventType::SceneChange, timestamp)
                .with_confidence(change_score.min(1.0)));
        }
    }

    /// 检测语音活动
    pub fn on_audio_energy(&mut self, timestamp: f64, energy: f32) {
        let is_speech = energy > self.config.speech_energy_threshold;

        if is_speech && !self.is_speaking {
            // 开始说话
            self.is_speaking = true;
            self.last_speech_time = timestamp;
            self.emit(Event::new(EventType::SpeechStart, timestamp));
        } else if is_speech {
            // 持续说话
            self.last_speech_time = timestamp;
        } else if self.is_speaking {
            // 检查是否超时
            if timestamp - self.last_speech_time > self.config.speech_timeout {
                self.is_speaking = false;
                self.emit(Event::new(EventType::SpeechEnd, timestamp));
                self.emit(Event::new(EventType::UserSpeech, timestamp));
            }
        }
    }

    /// 检测唤醒词 (从转录文本)
    pub fn on_transcription(&mut self, timestamp: f64, text: &str) {
        if let Some(wake_word) = &self.config.wake_word {
            if text.to_lowercase().contains(&wake_word.to_lowercase()) {
                self.is_awake = true;
                self.emit(Event::new(EventType::WakeWord, timestamp)
                    .with_data(text));
            }
        }
    }

    /// 检测超时
    pub fn check_timeout(&mut self, current_time: f64, last_interaction: f64, timeout: f64) {
        if self.is_awake && current_time - last_interaction > timeout {
            self.is_awake = false;
            self.emit(Event::new(EventType::Timeout, current_time));
        }
    }

    /// 手动唤醒
    pub fn wake(&mut self) {
        self.is_awake = true;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        self.emit(Event::new(EventType::WakeWord, now));
    }

    /// 是否已唤醒
    pub fn is_awake(&self) -> bool {
        self.is_awake
    }

    /// 是否正在说话
    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }

    fn emit(&self, event: Event) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }
}

impl Default for EventDetector {
    fn default() -> Self {
        Self::new(EventDetectorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_detector() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut detector = EventDetector::new(EventDetectorConfig::default())
            .with_event_channel(tx);

        // 检测场景变化
        detector.on_scene_change(1.0, 0.15);

        // 应该收到事件
        let event = rx.try_recv().unwrap();
        assert_eq!(event.event_type, EventType::SceneChange);
    }

    #[test]
    fn test_speech_detection() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut detector = EventDetector::new(EventDetectorConfig {
            speech_energy_threshold: 0.01,
            speech_timeout: 0.5,
            ..Default::default()
        }).with_event_channel(tx);

        // 开始说话
        detector.on_audio_energy(0.0, 0.1);
        let event = rx.try_recv().unwrap();
        assert_eq!(event.event_type, EventType::SpeechStart);

        // 持续说话
        detector.on_audio_energy(0.2, 0.1);
        detector.on_audio_energy(0.4, 0.1);

        // 停止说话 + 超时
        detector.on_audio_energy(0.6, 0.001);
        detector.on_audio_energy(1.2, 0.001);

        // 应该收到 SpeechEnd 和 UserSpeech
        let event1 = rx.try_recv().unwrap();
        let event2 = rx.try_recv().unwrap();
        assert_eq!(event1.event_type, EventType::SpeechEnd);
        assert_eq!(event2.event_type, EventType::UserSpeech);
    }
}

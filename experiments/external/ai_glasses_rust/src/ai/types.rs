//! AI 相关类型定义

use serde::{Deserialize, Serialize};

/// 用户意图
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserIntent {
    /// 描述
    Describe,
    /// 总结
    Summarize,
    /// 定位
    Locate,
    /// 对比
    Compare,
    /// 计数
    Count,
    /// 解释
    Explain,
    /// 音频关注
    AudioFocus,
    /// 转录
    Transcribe,
    /// 通用
    General,
}

impl Default for UserIntent {
    fn default() -> Self {
        Self::General
    }
}

impl UserIntent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Describe => "describe",
            Self::Summarize => "summarize",
            Self::Locate => "locate",
            Self::Compare => "compare",
            Self::Count => "count",
            Self::Explain => "explain",
            Self::AudioFocus => "audio_focus",
            Self::Transcribe => "transcribe",
            Self::General => "general",
        }
    }

    /// 从查询推断意图 (简单关键词匹配)
    pub fn from_query(query: &str) -> (Self, f32) {
        let query_lower = query.to_lowercase();

        let patterns: Vec<(Self, Vec<&str>)> = vec![
            (Self::Describe, vec!["描述", "说明", "讲解", "看到什么", "发生什么", "describe", "what", "show"]),
            (Self::Summarize, vec!["总结", "概括", "摘要", "简述", "summarize", "summary", "brief"]),
            (Self::Locate, vec!["什么时候", "哪里", "找到", "定位", "出现", "when", "where", "find", "locate"]),
            (Self::Compare, vec!["对比", "比较", "区别", "变化", "前后", "compare", "difference", "change"]),
            (Self::Count, vec!["多少", "几个", "数量", "统计", "count", "how many", "number"]),
            (Self::Explain, vec!["为什么", "原因", "解释", "explain", "why", "reason"]),
            (Self::AudioFocus, vec!["说了什么", "音频", "声音", "对话", "audio", "sound", "speech", "say"]),
            (Self::Transcribe, vec!["转录", "字幕", "文字", "transcribe", "transcript", "subtitle"]),
        ];

        let mut best_intent = Self::General;
        let mut best_score = 0;

        for (intent, keywords) in patterns {
            let score = keywords.iter()
                .filter(|kw| query_lower.contains(*kw))
                .count();
            if score > best_score {
                best_score = score;
                best_intent = intent;
            }
        }

        let confidence = (best_score as f32 / 3.0).min(1.0);
        (best_intent, confidence.max(0.5))
    }
}

/// V-JEPA2 策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VJEPA2Strategy {
    /// 描述: 均匀覆盖 + 变化检测
    Describe,
    /// 总结: 全局语义 + 代表帧
    Summarize,
    /// 定位: 密集采样 + 峰值检测
    Locate,
    /// 对比: 首尾对比 + 转折点
    Compare,
    /// 音频聚焦: 稳定帧
    AudioFocus,
}

impl From<UserIntent> for VJEPA2Strategy {
    fn from(intent: UserIntent) -> Self {
        match intent {
            UserIntent::Describe | UserIntent::Explain | UserIntent::General => Self::Describe,
            UserIntent::Summarize => Self::Summarize,
            UserIntent::Locate | UserIntent::Count => Self::Locate,
            UserIntent::Compare => Self::Compare,
            UserIntent::AudioFocus | UserIntent::Transcribe => Self::AudioFocus,
        }
    }
}

/// 编码请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeRequest {
    /// 帧数据 (Base64 编码的图像列表)
    pub frames: Vec<String>,
    /// 时间戳
    pub timestamps: Vec<f64>,
}

/// 编码响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeResponse {
    /// Embeddings (每帧一个)
    pub embeddings: Vec<Vec<f32>>,
    /// 维度
    pub dim: usize,
}

/// 理解请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandRequest {
    /// 关键帧 (Base64)
    pub keyframes: Vec<String>,
    /// 关键帧时间戳
    pub keyframe_times: Vec<f64>,
    /// 音频 (Base64, 可选)
    pub audio: Option<String>,
    /// 音频采样率
    pub sample_rate: Option<u32>,
    /// 提示
    pub prompt: String,
    /// 对话历史
    pub messages: Option<Vec<serde_json::Value>>,
}

/// 理解响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandResponse {
    /// 响应文本
    pub response: String,
    /// 推断的意图
    pub intent: Option<String>,
    /// 处理时间 (毫秒)
    pub processing_time_ms: u64,
}

/// 转录请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeRequest {
    /// 音频 (Base64)
    pub audio: String,
    /// 采样率
    pub sample_rate: u32,
}

/// 转录响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    /// 转录文本
    pub text: String,
    /// 语言
    pub language: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_classification() {
        let (intent, _) = UserIntent::from_query("描述一下这个视频");
        assert_eq!(intent, UserIntent::Describe);

        let (intent, _) = UserIntent::from_query("什么时候出现了人");
        assert_eq!(intent, UserIntent::Locate);

        let (intent, _) = UserIntent::from_query("视频里说了什么");
        assert_eq!(intent, UserIntent::AudioFocus);
    }
}

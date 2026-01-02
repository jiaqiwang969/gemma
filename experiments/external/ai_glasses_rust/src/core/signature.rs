//! Thought Signature
//!
//! 视频思维签名 - V-JEPA2 和 Gemma 3n 交互的压缩记忆

use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// 关注类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusType {
    /// 全局扫描
    Global,
    /// 时间定位
    Temporal,
    /// 空间区域
    Spatial,
    /// 音频内容
    Audio,
    /// 细节分析
    Detail,
}

impl Default for FocusType {
    fn default() -> Self {
        Self::Global
    }
}

impl FocusType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Temporal => "temporal",
            Self::Spatial => "spatial",
            Self::Audio => "audio",
            Self::Detail => "detail",
        }
    }
}

/// 签名状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureState {
    /// 初始化
    Initial,
    /// 扫描中
    Scanning,
    /// 理解中
    Understanding,
    /// 聚焦中
    Focusing,
    /// 完成
    Complete,
}

impl Default for SignatureState {
    fn default() -> Self {
        Self::Initial
    }
}

/// Thought Signature 配置
#[derive(Debug, Clone)]
pub struct SignatureConfig {
    /// 最大关键帧时间点数
    pub max_keyframe_times: usize,
    /// 最大意图数
    pub max_intents: usize,
    /// 最大问题历史
    pub max_questions: usize,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            max_keyframe_times: 10,
            max_intents: 5,
            max_questions: 10,
        }
    }
}

/// 视频思维签名
///
/// 这是 V-JEPA2 和 Gemma 3n 交互的核心数据结构
/// 它存储了对视频的理解状态，并在交互过程中不断演化
#[derive(Debug, Clone)]
pub struct ThoughtSignature {
    /// 视频 ID
    pub video_id: String,
    /// 视频时长
    pub video_duration: f64,
    /// 创建时间
    pub created_at: f64,
    /// 最后更新时间
    pub updated_at: f64,

    // V-JEPA2 分析状态
    /// 全局 embedding
    pub global_embedding: Option<Array1<f32>>,
    /// 关键帧时间点
    pub keyframe_times: Vec<f64>,
    /// 变化曲线
    pub change_profile: Vec<f32>,
    /// 活动级别
    pub activity_level: String,

    // 当前关注状态
    /// 当前关注类型
    pub current_focus: FocusType,
    /// 关注时间区域 (start, end)
    pub focus_regions: Vec<(f64, f64)>,

    // Gemma 3n 理解状态
    /// 视觉理解
    pub visual_understanding: String,
    /// 音频理解
    pub audio_understanding: String,
    /// 时序理解
    pub temporal_understanding: String,
    /// 语义摘要
    pub semantic_summary: String,

    // 用户意图
    /// 推断的意图
    pub inferred_intents: Vec<String>,
    /// 用户问题历史
    pub user_questions: Vec<String>,

    // 迭代状态
    /// 迭代次数
    pub iteration: u32,
    /// 理解深度 (0-1)
    pub understanding_depth: f32,
    /// 当前状态
    pub state: SignatureState,

    /// 配置
    config: SignatureConfig,
}

impl ThoughtSignature {
    /// 创建新的 Thought Signature
    pub fn new(video_id: String, video_duration: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            video_id,
            video_duration,
            created_at: now,
            updated_at: now,
            global_embedding: None,
            keyframe_times: Vec::new(),
            change_profile: Vec::new(),
            activity_level: "unknown".to_string(),
            current_focus: FocusType::Global,
            focus_regions: Vec::new(),
            visual_understanding: String::new(),
            audio_understanding: String::new(),
            temporal_understanding: String::new(),
            semantic_summary: String::new(),
            inferred_intents: Vec::new(),
            user_questions: Vec::new(),
            iteration: 0,
            understanding_depth: 0.0,
            state: SignatureState::Initial,
            config: SignatureConfig::default(),
        }
    }

    /// 使用配置创建
    pub fn with_config(mut self, config: SignatureConfig) -> Self {
        self.config = config;
        self
    }

    /// 初始化 (来自 V-JEPA2 首次分析)
    pub fn initialize(
        &mut self,
        global_embedding: Array1<f32>,
        keyframe_times: Vec<f64>,
        change_profile: Vec<f32>,
        activity_level: &str,
    ) {
        self.global_embedding = Some(global_embedding);
        self.keyframe_times = keyframe_times;
        self.change_profile = change_profile;
        self.activity_level = activity_level.to_string();
        self.understanding_depth = 0.1; // 初始理解深度 10%
        self.state = SignatureState::Scanning;
        self.touch();
    }

    /// 更新视觉理解
    pub fn update_visual_understanding(&mut self, understanding: &str) {
        if self.visual_understanding.is_empty() {
            self.visual_understanding = understanding.to_string();
        } else {
            self.visual_understanding.push_str(&format!(
                "\n[Iter {}]: {}",
                self.iteration, understanding
            ));
        }
        self.increment_depth(0.15);
        self.touch();
    }

    /// 更新音频理解
    pub fn update_audio_understanding(&mut self, understanding: &str) {
        if self.audio_understanding.is_empty() {
            self.audio_understanding = understanding.to_string();
        } else {
            self.audio_understanding.push_str(&format!(
                "\n[Iter {}]: {}",
                self.iteration, understanding
            ));
        }
        self.increment_depth(0.1);
        self.touch();
    }

    /// 更新语义摘要
    pub fn update_semantic_summary(&mut self, summary: &str) {
        self.semantic_summary = summary.to_string();
        self.touch();
    }

    /// 更新时序理解
    pub fn update_temporal_understanding(&mut self, understanding: &str) {
        self.temporal_understanding = understanding.to_string();
        self.touch();
    }

    /// 添加推断的意图
    pub fn add_intent(&mut self, intent: &str) {
        if !self.inferred_intents.contains(&intent.to_string()) {
            self.inferred_intents.push(intent.to_string());
            if self.inferred_intents.len() > self.config.max_intents {
                self.inferred_intents.remove(0);
            }
        }
        self.touch();
    }

    /// 添加用户问题
    pub fn add_question(&mut self, question: &str) {
        self.user_questions.push(question.to_string());
        if self.user_questions.len() > self.config.max_questions {
            self.user_questions.remove(0);
        }
        self.touch();
    }

    /// 设置关注类型
    pub fn set_focus(&mut self, focus: FocusType) {
        self.current_focus = focus;
        self.state = SignatureState::Focusing;
        self.touch();
    }

    /// 设置关注区域
    pub fn set_focus_regions(&mut self, regions: Vec<(f64, f64)>) {
        self.focus_regions = regions;
        self.touch();
    }

    /// 增加迭代次数
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
        self.state = SignatureState::Understanding;
        self.touch();
    }

    /// 标记完成
    pub fn mark_complete(&mut self) {
        self.state = SignatureState::Complete;
        self.understanding_depth = 1.0;
        self.touch();
    }

    /// 生成上下文字符串 (供 LLM 使用)
    pub fn to_context_string(&self) -> String {
        let mut parts = Vec::new();

        parts.push(format!("## Video Context (Iteration {})", self.iteration));
        parts.push(format!("- Duration: {:.1}s", self.video_duration));
        parts.push(format!("- Activity: {}", self.activity_level));
        parts.push(format!("- Current Focus: {}", self.current_focus.as_str()));
        parts.push(format!("- Understanding Depth: {:.0}%", self.understanding_depth * 100.0));

        if !self.focus_regions.is_empty() {
            let regions: Vec<String> = self.focus_regions.iter()
                .map(|(s, e)| format!("{:.1}s-{:.1}s", s, e))
                .collect();
            parts.push(format!("- Focus Regions: {}", regions.join(", ")));
        }

        if !self.visual_understanding.is_empty() {
            parts.push(format!("\n### Visual Understanding\n{}", self.visual_understanding));
        }

        if !self.audio_understanding.is_empty() {
            parts.push(format!("\n### Audio Understanding\n{}", self.audio_understanding));
        }

        if !self.temporal_understanding.is_empty() {
            parts.push(format!("\n### Temporal Understanding\n{}", self.temporal_understanding));
        }

        if !self.semantic_summary.is_empty() {
            parts.push(format!("\n### Current Summary\n{}", self.semantic_summary));
        }

        if !self.inferred_intents.is_empty() {
            parts.push(format!("\n### Inferred User Intents\n- {}", self.inferred_intents.join("\n- ")));
        }

        parts.join("\n")
    }

    /// 序列化为 JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.to_serializable())
    }

    /// 转换为可序列化的结构
    fn to_serializable(&self) -> SignatureData {
        SignatureData {
            video_id: self.video_id.clone(),
            video_duration: self.video_duration,
            created_at: self.created_at,
            updated_at: self.updated_at,
            keyframe_times: self.keyframe_times.clone(),
            activity_level: self.activity_level.clone(),
            current_focus: self.current_focus,
            focus_regions: self.focus_regions.clone(),
            visual_understanding: self.visual_understanding.clone(),
            audio_understanding: self.audio_understanding.clone(),
            temporal_understanding: self.temporal_understanding.clone(),
            semantic_summary: self.semantic_summary.clone(),
            inferred_intents: self.inferred_intents.clone(),
            user_questions: self.user_questions.clone(),
            iteration: self.iteration,
            understanding_depth: self.understanding_depth,
            state: self.state,
        }
    }

    fn increment_depth(&mut self, amount: f32) {
        self.understanding_depth = (self.understanding_depth + amount).min(1.0);
    }

    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
    }
}

/// 可序列化的签名数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureData {
    pub video_id: String,
    pub video_duration: f64,
    pub created_at: f64,
    pub updated_at: f64,
    pub keyframe_times: Vec<f64>,
    pub activity_level: String,
    pub current_focus: FocusType,
    pub focus_regions: Vec<(f64, f64)>,
    pub visual_understanding: String,
    pub audio_understanding: String,
    pub temporal_understanding: String,
    pub semantic_summary: String,
    pub inferred_intents: Vec<String>,
    pub user_questions: Vec<String>,
    pub iteration: u32,
    pub understanding_depth: f32,
    pub state: SignatureState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_signature() {
        let mut sig = ThoughtSignature::new("test_video".to_string(), 10.0);

        // 初始化
        sig.initialize(
            Array1::from_vec(vec![1.0; 1024]),
            vec![0.0, 2.5, 5.0, 7.5, 10.0],
            vec![0.0, 0.05, 0.02, 0.1, 0.03],
            "medium",
        );

        assert_eq!(sig.activity_level, "medium");
        assert!(sig.understanding_depth > 0.0);

        // 更新理解
        sig.next_iteration();
        sig.update_visual_understanding("A person is walking in a park.");
        sig.add_intent("describe");

        assert!(sig.visual_understanding.contains("walking"));
        assert!(sig.inferred_intents.contains(&"describe".to_string()));

        // 生成上下文
        let context = sig.to_context_string();
        assert!(context.contains("Duration:"));
        assert!(context.contains("Visual Understanding"));
    }
}

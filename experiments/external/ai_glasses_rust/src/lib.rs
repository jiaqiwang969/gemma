//! AI 眼镜系统核心库
//!
//! 核心机制：意图-嵌入双向循环
//! - 用户意图 → 决定 V-JEPA2 提取策略
//! - V-JEPA2 embedding → 增强意图分析
//! - 意图更新 → 调整提取策略
//! - 循环迭代直到理解充分

pub mod buffer;
pub mod core;
pub mod ai;
pub mod utils;

// Re-exports - 核心引擎
pub use core::engine::{IntentEmbeddingEngine, EngineEvent, EngineStats, FrameSelector};
pub use core::intent_loop::{
    IntentState, IntentType, IntentDrivenStore, EmbeddingAnalysis,
    ExtractionStrategy, ActivityLevel,
};

// Re-exports - 流处理
pub use core::stream::StreamProcessor;
pub use core::signature::{ThoughtSignature, FocusType};
pub use core::event::{EventDetector, Event, EventType};
pub use core::dialogue::{DialogueManager, DialogueState, Message};

// Re-exports - 缓冲区
pub use buffer::{FrameBuffer, AudioBuffer, EmbeddingCache};
pub use buffer::frame::Frame;

// Re-exports - AI
pub use ai::client::AiClient;

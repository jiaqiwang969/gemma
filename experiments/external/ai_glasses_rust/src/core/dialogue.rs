//! 对话管理器
//!
//! 管理对话状态、多轮交互、上下文切换

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// 消息角色
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

/// 消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub timestamp: f64,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            timestamp: now(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            timestamp: now(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            timestamp: now(),
        }
    }
}

/// 对话状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialogueState {
    /// 空闲
    Idle,
    /// 等待用户输入
    WaitingForUser,
    /// 处理中
    Processing,
    /// 生成响应中
    Generating,
    /// 等待确认
    WaitingForConfirmation,
}

impl Default for DialogueState {
    fn default() -> Self {
        Self::Idle
    }
}

/// 对话管理器配置
#[derive(Debug, Clone)]
pub struct DialogueConfig {
    /// 最大历史消息数
    pub max_history: usize,
    /// 上下文窗口大小 (字符数)
    pub context_window: usize,
    /// 空闲超时 (秒)
    pub idle_timeout: f64,
}

impl Default for DialogueConfig {
    fn default() -> Self {
        Self {
            max_history: 20,
            context_window: 4000,
            idle_timeout: 300.0, // 5分钟
        }
    }
}

/// 对话管理器
pub struct DialogueManager {
    config: DialogueConfig,
    /// 对话历史
    history: VecDeque<Message>,
    /// 当前状态
    state: DialogueState,
    /// 最后交互时间
    last_interaction: f64,
    /// 会话 ID
    session_id: String,
    /// 系统提示
    system_prompt: Option<String>,
}

impl DialogueManager {
    pub fn new(config: DialogueConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            state: DialogueState::Idle,
            last_interaction: now(),
            session_id: uuid::Uuid::new_v4().to_string(),
            system_prompt: None,
        }
    }

    /// 设置系统提示
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// 添加用户消息
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        let msg = Message::user(content);
        self.add_message(msg);
        self.state = DialogueState::Processing;
    }

    /// 添加助手消息
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        let msg = Message::assistant(content);
        self.add_message(msg);
        self.state = DialogueState::WaitingForUser;
    }

    /// 添加消息
    fn add_message(&mut self, msg: Message) {
        self.history.push_back(msg);
        if self.history.len() > self.config.max_history {
            self.history.pop_front();
        }
        self.last_interaction = now();
    }

    /// 获取对话历史
    pub fn get_history(&self) -> Vec<&Message> {
        self.history.iter().collect()
    }

    /// 获取最近的 N 条消息
    pub fn get_recent(&self, n: usize) -> Vec<&Message> {
        self.history.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect()
    }

    /// 构建上下文字符串 (包含系统提示和历史)
    pub fn build_context(&self, video_context: Option<&str>) -> String {
        let mut parts = Vec::new();

        // 系统提示
        if let Some(prompt) = &self.system_prompt {
            parts.push(format!("System: {}", prompt));
        }

        // 视频上下文
        if let Some(ctx) = video_context {
            parts.push(format!("\n{}", ctx));
        }

        // 对话历史
        parts.push("\n## Conversation".to_string());
        for msg in self.history.iter() {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            parts.push(format!("{}: {}", role, msg.content));
        }

        let full = parts.join("\n");

        // 截断到上下文窗口
        if full.len() > self.config.context_window {
            full[full.len() - self.config.context_window..].to_string()
        } else {
            full
        }
    }

    /// 构建消息列表 (用于 API 调用)
    pub fn build_messages(&self) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // 系统提示
        if let Some(prompt) = &self.system_prompt {
            messages.push(serde_json::json!({
                "role": "system",
                "content": prompt
            }));
        }

        // 历史消息
        for msg in self.history.iter() {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            messages.push(serde_json::json!({
                "role": role,
                "content": msg.content
            }));
        }

        messages
    }

    /// 获取当前状态
    pub fn state(&self) -> DialogueState {
        self.state
    }

    /// 设置状态
    pub fn set_state(&mut self, state: DialogueState) {
        self.state = state;
    }

    /// 检查是否空闲超时
    pub fn is_idle_timeout(&self) -> bool {
        now() - self.last_interaction > self.config.idle_timeout
    }

    /// 获取最后交互时间
    pub fn last_interaction(&self) -> f64 {
        self.last_interaction
    }

    /// 获取会话 ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// 清空历史
    pub fn clear(&mut self) {
        self.history.clear();
        self.state = DialogueState::Idle;
    }

    /// 开始新会话
    pub fn new_session(&mut self) {
        self.clear();
        self.session_id = uuid::Uuid::new_v4().to_string();
        self.last_interaction = now();
    }
}

impl Default for DialogueManager {
    fn default() -> Self {
        Self::new(DialogueConfig::default())
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialogue_manager() {
        let mut manager = DialogueManager::new(DialogueConfig {
            max_history: 5,
            ..Default::default()
        }).with_system_prompt("You are a helpful AI assistant.");

        // 添加消息
        manager.add_user_message("Hello!");
        manager.add_assistant_message("Hi! How can I help you?");
        manager.add_user_message("What do you see?");

        // 检查历史
        assert_eq!(manager.get_history().len(), 3);
        assert_eq!(manager.state(), DialogueState::Processing);

        // 构建上下文
        let context = manager.build_context(None);
        assert!(context.contains("Hello!"));
        assert!(context.contains("You are a helpful AI assistant"));
    }

    #[test]
    fn test_history_limit() {
        let mut manager = DialogueManager::new(DialogueConfig {
            max_history: 3,
            ..Default::default()
        });

        for i in 0..5 {
            manager.add_user_message(format!("Message {}", i));
        }

        assert_eq!(manager.get_history().len(), 3);
        assert!(manager.get_history()[0].content.contains("Message 2"));
    }
}

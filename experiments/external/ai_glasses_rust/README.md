# AI 眼镜系统 (Rust 版)

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Rust Core                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    StreamProcessor                           │    │
│  │  - 实时视频/音频流处理                                        │    │
│  │  - 滑动窗口管理                                              │    │
│  │  - 关键帧选择                                                │    │
│  │  - 变化检测                                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   ThoughtSignature                           │    │
│  │  - 压缩记忆系统                                              │    │
│  │  - 理解状态管理                                              │    │
│  │  - 意图追踪                                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    EventDetector                             │    │
│  │  - 场景变化事件                                              │    │
│  │  - 用户交互事件                                              │    │
│  │  - 唤醒词检测                                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   DialogueManager                            │    │
│  │  - 对话状态机                                                │    │
│  │  - 多轮对话管理                                              │    │
│  │  - 上下文切换                                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ HTTP/gRPC
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Python AI Service                             │
│  ┌───────────────────┐  ┌───────────────────┐                       │
│  │     V-JEPA2       │  │    Gemma 3n       │                       │
│  │  视频语义编码      │  │  多模态理解       │                       │
│  └───────────────────┘  └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 模块结构

```
src/
├── lib.rs              # 库入口
├── main.rs             # 可执行入口
├── core/
│   ├── mod.rs
│   ├── stream.rs       # 流处理器
│   ├── signature.rs    # Thought Signature
│   ├── event.rs        # 事件检测
│   └── dialogue.rs     # 对话管理
├── buffer/
│   ├── mod.rs
│   ├── frame.rs        # 帧缓冲
│   ├── audio.rs        # 音频缓冲
│   └── embedding.rs    # Embedding 缓冲
├── ai/
│   ├── mod.rs
│   ├── client.rs       # AI 服务客户端
│   └── types.rs        # AI 相关类型
└── utils/
    ├── mod.rs
    └── time.rs         # 时间工具
```

## 设计原则

### 1. Rust Core + Python AI Service

- **Rust**: 实时流处理、状态管理、事件驱动
- **Python**: V-JEPA2 编码、Gemma 3n 推理

### 2. 异步优先

```rust
// 所有 IO 操作都是异步的
async fn process_frame(&mut self, frame: Frame) -> Result<()> {
    // 添加到缓冲
    self.buffer.push(frame).await;

    // 检查是否需要编码
    if self.should_encode() {
        let embedding = self.ai_client.encode(frames).await?;
        self.signature.update(embedding).await;
    }

    Ok(())
}
```

### 3. 无锁数据结构

```rust
// 使用 crossbeam 的无锁队列
use crossbeam::queue::ArrayQueue;

struct FrameBuffer {
    frames: ArrayQueue<Frame>,
}
```

### 4. 零拷贝

```rust
// 使用 bytes crate 实现零拷贝
use bytes::Bytes;

struct Frame {
    data: Bytes,  // 引用计数，零拷贝
    timestamp: f64,
}
```

## 快速开始

```bash
# 构建
cargo build --release

# 运行
cargo run --release

# 测试
cargo test
```

## AI 服务

Rust 核心通过 HTTP 与 Python AI 服务通信：

```bash
# 启动 Python AI 服务
cd ../
python -m ai_service.server --port 8080
```

API:
- `POST /encode` - V-JEPA2 视频编码
- `POST /understand` - Gemma 3n 多模态理解
- `POST /transcribe` - 音频转录

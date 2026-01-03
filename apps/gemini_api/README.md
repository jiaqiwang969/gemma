# Gemini API 兼容服务器

将本地 Gemma 3n 模型暴露为 Google Gemini API 兼容格式的服务器。

## 快速开始

```bash
# 启动服务器
python apps/gemini_api/server.py

# 服务器将在 http://localhost:5001 启动
```

## API 端点

### 生成内容
```
POST /v1beta/models/{model}:generateContent
```

### 流式生成内容
```
POST /v1beta/models/{model}:streamGenerateContent
```

支持的模型名称:
- `gemini-3-pro-preview`
- `gemini-3-flash-preview`
- `gemma-3n`

### 列出模型
```
GET /v1beta/models
```

### 健康检查
```
GET /health
```

## 使用示例

### 1. 基础文本调用

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "hi"}]}]
  }'
```

### 2. 系统指令

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "你好"}]}],
    "systemInstruction": {
      "parts": [{"text": "你是一个专业的中文助手，名叫小智。每次回答以「小智：」开头。"}]
    }
  }'
```

### 3. 思考等级配置

支持 `minimal`, `low`, `medium`, `high` 四个等级：

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "解释量子纠缠"}]}],
    "generationConfig": {
      "thinkingConfig": {"thinkingLevel": "high"}
    }
  }'
```

### 4. JSON Schema 结构化输出

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "列出三种编程语言"}]}],
    "generationConfig": {
      "responseMimeType": "application/json",
      "responseSchema": {
        "type": "object",
        "properties": {
          "languages": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "year": {"type": "integer"},
                "use_case": {"type": "string"}
              }
            }
          }
        }
      }
    }
  }'
```

### 5. 工具调用 (Function Call)

**Step 1: 发起工具调用**
```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "执行 ls 命令"}]}],
    "tools": [{
      "functionDeclarations": [{
        "name": "shell_command",
        "description": "Run shell command",
        "parameters": {
          "type": "object",
          "properties": {"command": {"type": "string"}},
          "required": ["command"]
        }
      }]
    }]
  }'
```

**Step 2: 回放 functionCall + functionResponse**
```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "执行 ls 命令"}]},
      {"role": "model", "parts": [{
        "functionCall": {"name": "shell_command", "args": {"command": "ls"}},
        "thoughtSignature": "<从Step1响应中获取>"
      }]},
      {"role": "user", "parts": [{
        "functionResponse": {
          "name": "shell_command",
          "response": {"output": "file1.txt\nfile2.txt", "success": true}
        }
      }]}
    ],
    "tools": [...]
  }'
```

### 6. 工具配置 (toolConfig)

支持 `AUTO`, `ANY`, `NONE` 三种模式：

```bash
# 强制使用工具
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "What is the weather?"}]}],
    "tools": [...],
    "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
  }'
```

### 7. 多轮对话

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "我叫小明"}]},
      {"role": "model", "parts": [
        {"text": "你好小明！"},
        {"thoughtSignature": "<从上一轮响应中获取>"}
      ]},
      {"role": "user", "parts": [{"text": "我叫什么名字？"}]}
    ]
  }'
```

### 8. 流式输出 (SSE)

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:streamGenerateContent?alt=sse" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Count from 1 to 10"}]}],
    "generationConfig": {"maxOutputTokens": 256}
  }'
```

### 9. 视觉多模态 (单图/多图)

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [
      {"text": "What is in this image?"},
      {"inlineData": {"mimeType": "image/png", "data": "<base64-encoded-image>"}}
    ]}],
    "generationConfig": {"maxOutputTokens": 256}
  }'
```

多图片对比：
```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [
      {"text": "Compare these two images"},
      {"inlineData": {"mimeType": "image/png", "data": "<image1-base64>"}},
      {"inlineData": {"mimeType": "image/png", "data": "<image2-base64>"}}
    ]}]
  }'
```

### 10. 音频多模态

支持的音频格式: `audio/wav`, `audio/mp3`, `audio/flac`, `audio/ogg`, `audio/aac`, `audio/webm`

```bash
curl -X POST "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [
      {"text": "Transcribe this audio"},
      {"inlineData": {"mimeType": "audio/flac", "data": "<base64-encoded-audio>"}}
    ]}],
    "generationConfig": {"maxOutputTokens": 1024}
  }'
```

## 支持的功能

| 功能 | 状态 | 说明 |
|------|------|------|
| generateContent | ✅ | 文本生成 |
| streamGenerateContent | ✅ | 流式输出 (SSE) |
| thinkingConfig | ✅ | 思考等级 (minimal/low/medium/high) |
| includeThoughts | ✅ | 控制思考过程是否显示 |
| systemInstruction | ✅ | 系统指令/人设 |
| responseMimeType + responseSchema | ✅ | JSON 结构化输出 |
| safetySettings | ✅ | 安全设置 (已解析，效果取决于模型) |
| Multi-turn conversation | ✅ | 多轮对话 |
| thoughtSignature | ✅ | 思维签名 (KV Cache 持久化) |
| functionDeclarations + functionCall | ✅ | 工具声明和调用 |
| functionResponse | ✅ | 工具响应回放 |
| toolConfig (AUTO/ANY/NONE) | ✅ | 工具调用模式配置 |
| parallel function calling | ✅ | 并行函数调用 |
| codeExecution | ⚠️ | 仅识别格式，不执行代码 |
| vision multimodal | ✅ | 视觉多模态 (单/多图片) |
| audio multimodal | ✅ | 音频多模态 (单音频) |

## 不支持的功能

| 功能 | 原因 |
|------|------|
| googleSearch | 需要外部 Google Search API |
| 图像生成 (gemini-3-pro-image-preview) | 本地模型不支持图像生成 |
| 多音频文件 | llama-mtmd-cli 限制，仅支持单个音频 |

## 响应格式

所有响应都遵循 Gemini API 格式:

```json
{
  "candidates": [{
    "content": {
      "role": "model",
      "parts": [
        {"text": "思考过程...", "thought": true},
        {"text": "回复内容", "thoughtSignature": "Base64签名..."}
      ]
    },
    "finishReason": "STOP",
    "index": 0,
    "safetyRatings": null
  }],
  "promptFeedback": {"safetyRatings": null},
  "usageMetadata": {
    "promptTokenCount": N,
    "candidatesTokenCount": N,
    "totalTokenCount": N,
    "thoughtsTokenCount": N,
    "promptTokensDetails": [
      {"modality": "TEXT", "tokenCount": N},
      {"modality": "AUDIO", "tokenCount": N}
    ]
  },
  "modelVersion": "gemma-3n-local",
  "responseId": "xxx"
}
```

## 测试工具

### 内置测试端点

```bash
# 基础测试
curl http://localhost:5001/test/basic

# 思考等级测试
curl http://localhost:5001/test/thinking

# 系统指令测试
curl http://localhost:5001/test/system

# JSON Schema 测试
curl http://localhost:5001/test/json-schema

# 工具调用测试
curl http://localhost:5001/test/function-call

# 多轮对话测试
curl http://localhost:5001/test/multi-turn

# toolConfig 模式测试
curl http://localhost:5001/test/tool-config-any
curl http://localhost:5001/test/tool-config-none

# 并行函数调用测试
curl http://localhost:5001/test/parallel-function-call

# 视觉多模态测试
curl http://localhost:5001/test/vision

# 音频状态检查
curl http://localhost:5001/test/audio-status
```

### HTML 对比测试

打开 `02-gemini-api-test.html` 可以进行真实 Gemini API 与本地服务的 JSON 字段逐项对比测试。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| LLAMA_MODEL | 自动检测 | GGUF 模型路径 |
| LLAMA_MODEL_AUDIO | 自动检测 | 音频推理专用模型 (原始模型，非微调) |
| LLAMA_SERVER_BIN | infra/llama.cpp/build/bin/llama-server | llama-server 二进制路径 |
| LLAMA_MTMD_BIN | infra/llama.cpp/build/bin/llama-mtmd-cli | llama-mtmd-cli 二进制路径 |
| LLAMA_MMPROJ_VISION | 自动检测 | 视觉投影器路径 |
| LLAMA_MMPROJ_AUDIO | 自动检测 | 音频投影器路径 |
| GEMINI_API_LLAMA_PORT | 8090 | llama-server 端口 |
| KV_CACHE_DIR | /tmp/gemma3n_thought_cache | KV Cache 存储目录 |
| KV_CACHE_ENABLED | true | 是否启用 KV Cache 持久化 |

## 与真实 Gemini API 的对比

本服务器的响应格式与真实 Google Gemini API 完全兼容，可以直接替换 `api.vectorengine.ai` 或 `generativelanguage.googleapis.com` 使用。

只需将 API 端点从:
```
https://api.vectorengine.ai/v1beta/models/gemini-3-pro-preview:generateContent
```

改为:
```
http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent
```

## 架构说明

```
┌─────────────────────────────────────────────────────────────┐
│                    Gemini API Server                        │
│                    (Flask, port 5001)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │ Text Only   │     │   Vision    │     │   Audio     │  │
│   │             │     │  (Images)   │     │             │  │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘  │
│          │                   │                   │         │
│          ▼                   ▼                   ▼         │
│   ┌─────────────────────────────┐     ┌─────────────────┐  │
│   │      llama-server           │     │ llama-mtmd-cli  │  │
│   │      (port 8090)            │     │   (CLI tool)    │  │
│   │  - 纯文本推理               │     │  - 音频推理     │  │
│   │  - 视觉多模态               │     │                 │  │
│   │  - 流式输出                 │     │                 │  │
│   └─────────────────────────────┘     └─────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 版本历史

- **v1.1.0** - 新增流式输出、多图片支持、完整功能对齐
- **v1.0.0** - 初始版本，基础文本和工具调用支持

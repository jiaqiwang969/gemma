<p align="center">
  <img src="https://img.shields.io/badge/🐉-灵空_AI-6366f1?style=for-the-badge" alt="LingKong AI">
</p>

<h1 align="center">灵空 AI</h1>

<p align="center">
  <strong>你的 AI. 你的数据. 你的掌控.</strong>
</p>

<p align="center">
  一个开源的本地多模态 AI 平台，让你在自己的设备上运行强大的 AI，无需将数据发送到任何云端。
</p>

<p align="center">
  <a href="https://github.com/jiaqiwang969/gemma/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://huggingface.co/jiaqiwang969/gemma3n-gguf"><img src="https://img.shields.io/badge/🤗-Models-yellow.svg" alt="HuggingFace"></a>
  <a href="https://lingkong.xyz"><img src="https://img.shields.io/badge/demo-live-brightgreen.svg" alt="Demo"></a>
</p>

---

## ⚡ 一键安装

```bash
curl -fsSL https://lingkong.xyz/install.sh | bash
```

安装完成后：

```bash
# 下载模型 (2.8GB)
lingkong-download

# 启动服务
lingkong-start

# 访问 http://localhost:5001
```

**无需 Python，无需复杂配置。三步启动你的私有 AI！**

---

## 🎯 为什么选择灵空 AI？

| 对比项 | 云端 AI | 灵空 AI |
|--------|---------|---------|
| 🔐 隐私 | 服务商能看到一切 | **100% 本地运行** |
| 💰 成本 | 按 Token 持续付费 | **一次性，永久免费** |
| ⚡ 速度 | 受网络延迟影响 | **94 tok/s 本地推理** |
| 📴 离线 | 必须联网 | **随处可用** |
| 🎛️ 控制 | 条款随时变更 | **你拥有完全掌控** |

---

## 📦 预编译模型

模型托管在 [HuggingFace](https://huggingface.co/jiaqiwang969/gemma3n-gguf)：

| 模型 | 大小 | 用途 |
|------|------|------|
| `gemma-3n-E2B-it-Q4_K_M.gguf` | 2.8GB | 主文本模型 (推荐) |
| `gemma-3n-vision-mmproj-f16.gguf` | 600MB | 视觉理解模块 |
| `gemma-3n-audio-mmproj-f16.gguf` | 1.4GB | 音频理解模块 |

```bash
# 下载指定模型
lingkong-download text    # 仅文本模型
lingkong-download vision  # 视觉模块
lingkong-download audio   # 音频模块
lingkong-download all     # 全部模型 (~5GB)
```

---

## 🌐 在线演示

| 页面 | 地址 | 说明 |
|------|------|------|
| 🏠 项目主页 | [lingkong.xyz](https://lingkong.xyz) | 功能介绍、快速开始 |
| 💬 聊天界面 | [/static/index.html](https://lingkong.xyz/static/index.html) | 多模态对话体验 |
| 📚 API 文档 | [/static/docs.html](https://lingkong.xyz/static/docs.html) | Gemini 兼容 API |
| 🛠️ Playground | [/static/playground.html](https://lingkong.xyz/static/playground.html) | 交互式 API 测试 |
| 📊 商业计划书 | [/static/pitch.html](https://lingkong.xyz/static/pitch.html) | 愿景与商业模式 |

---

## ✨ 核心特性

### 🔐 完全私密
- 数据永不离开你的设备
- 无需账号、无需登录
- 零知识架构设计

### 🎯 多模态能力
- **文本理解**: 对话、写作、编程
- **图像理解**: 描述、分析、OCR
- **音频理解**: 转录、翻译、总结

### ⚡ 高性能推理
- llama.cpp 引擎 (Metal/CUDA 加速)
- ~94 tokens/s (M4 Max)
- 支持 GGUF 量化模型

### 🔌 API 兼容
- 兼容 Google Gemini API
- 无缝替换现有应用
- 支持流式输出

---

## 🛠️ 使用方式

### Level 1: 一键部署 (小白用户)

```bash
# 安装
curl -fsSL https://lingkong.xyz/install.sh | bash

# 下载模型 + 启动
lingkong-download && lingkong-start
```

### Level 2: API 调用 (开发者)

```python
import requests

response = requests.post(
    "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent",
    json={
        "contents": [{"parts": [{"text": "你好，介绍一下你自己"}]}],
        "generationConfig": {"maxOutputTokens": 512}
    }
)

print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
```

### Level 3: 模型微调 (进阶用户)

```bash
# 克隆仓库
git clone https://github.com/jiaqiwang969/gemma.git
cd gemma

# 创建 Python 环境
conda create -n lingkong python=3.11
conda activate lingkong
pip install -e .

# 微调模型 (LoRA)
python scripts/finetune.py --data your_data.jsonl

# 合并权重并转换 GGUF
python scripts/merge_lora.py
python scripts/convert_to_gguf.py
```

---

## 📊 性能数据

在 Apple M4 Max (64GB) 上测试：

| 指标 | 数值 |
|------|------|
| 推理速度 (llama.cpp) | 94 tok/s |
| 推理速度 (PyTorch) | 16 tok/s |
| 首 Token 延迟 | ~200ms |
| 内存占用 (Q4_K_M) | ~4GB |
| 模型加载 | ~3.7s |

---

## 💻 硬件要求

| 配置 | 规格 | 适用场景 | 参考价格 |
|------|------|----------|----------|
| 入门级 | Mac Mini M2 8GB | 纯文本对话 | ~$600 |
| **推荐** | Mac Mini M4 24GB | 多模态推理 | ~$1,200 |
| 专业级 | Mac Studio / RTX 4090 | 微调训练 | ~$4,000+ |

---

## 📁 项目结构

```
gemma/
├── apps/                          # 应用层
│   ├── webui/                     # Web 聊天界面
│   │   ├── server.py              # Flask 服务器
│   │   └── static/                # 前端页面
│   └── gemini_api/                # Gemini 兼容 API
├── scripts/                       # 工具脚本
│   ├── quick-install.sh           # 一键安装脚本
│   └── install.sh                 # 完整安装脚本
├── artifacts/                     # 产物输出
│   ├── gguf/                      # GGUF 模型文件
│   └── lora/                      # LoRA 适配器
└── contexts/training/             # 微调脚本
```

---

## 🔗 链接

- 🌐 **官网**: [lingkong.xyz](https://lingkong.xyz)
- 📚 **API 文档**: [lingkong.xyz/static/docs.html](https://lingkong.xyz/static/docs.html)
- 🛠️ **Playground**: [lingkong.xyz/static/playground.html](https://lingkong.xyz/static/playground.html)
- 🤗 **模型**: [huggingface.co/jiaqiwang969/gemma3n-gguf](https://huggingface.co/jiaqiwang969/gemma3n-gguf)
- 📊 **商业计划**: [lingkong.xyz/static/pitch.html](https://lingkong.xyz/static/pitch.html)

---

## 📄 许可证

MIT License - 随意使用，保留版权声明即可。

---

<p align="center">
  <strong>🐉 灵空 AI</strong><br>
  你的 AI. 你的数据. 你的掌控.
</p>

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
  <a href="https://github.com/jiaqiwang969/gemma/actions"><img src="https://github.com/jiaqiwang969/gemma/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://github.com/jiaqiwang969/gemma/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="http://115.159.223.227"><img src="https://img.shields.io/badge/demo-live-brightgreen.svg" alt="Demo"></a>
</p>

<p align="center">
  <a href="#-快速开始">快速开始</a> •
  <a href="#-核心特性">核心特性</a> •
  <a href="#-在线演示">在线演示</a> •
  <a href="#-文档">文档</a> •
  <a href="#-商业计划书">商业计划书</a>
</p>

---

## 🎯 为什么选择灵空 AI？

| 特性 | 云端 AI | 灵空 AI |
|------|---------|---------|
| **隐私** | ❌ 服务商能看到一切 | ✅ 100% 本地，完全私密 |
| **成本** | ❌ 按 Token 持续付费 | ✅ 永久免费 (仅硬件成本) |
| **速度** | ❌ 受网络延迟影响 | ✅ 即时本地响应 |
| **离线** | ❌ 必须联网 | ✅ 随处可用 |
| **控制** | ❌ 条款随时变更 | ✅ 你拥有完全掌控 |

## 🚀 快速开始

### 方式一：从源码编译 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/jiaqiwang969/gemma.git
cd gemma

# 2. 安装 Python 依赖
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 启动 WebUI 服务器
python apps/webui/server.py
# 访问 http://localhost:5000
```

### 方式二：使用 LingKong CLI (Rust)

```bash
# 1. 编译 CLI 工具
cd ../LingKong-AI/src
cargo build --release

# 2. 安装到系统路径
cp target/release/lingkong ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"

# 3. 初始化环境 (下载 llama.cpp, 创建目录)
lingkong install

# 4. 下载模型
lingkong model pull gemma3n-2b-text

# 5. 启动服务
lingkong serve start
```

### 方式三：一键部署到服务器

```bash
# 在远程服务器上执行
ssh ubuntu@your-server 'bash -s' < apps/deploy_all.sh

# 上传代码
scp apps/webui/server_lite.py ubuntu@your-server:/opt/lingkong-webui/server.py
scp -r apps/webui/static/* ubuntu@your-server:/opt/lingkong-webui/static/

# 启动服务
ssh ubuntu@your-server 'sudo systemctl enable --now lingkong-webui'
```

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

## 🌐 在线演示

访问我们的在线演示站点：

| 页面 | 地址 | 说明 |
|------|------|------|
| 🏠 项目主页 | http://115.159.223.227 | 功能介绍、快速开始 |
| 💬 聊天界面 | http://115.159.223.227/static/index.html | 多模态对话体验 |
| 📚 API 文档 | http://115.159.223.227/static/docs.html | Gemini 兼容 API |
| 📊 商业计划书 | http://115.159.223.227/static/pitch.html | 愿景与商业模式 |

## 📁 项目结构

```
gemma/
├── apps/                          # 应用层
│   ├── webui/                     # Web 聊天界面
│   │   ├── server.py              # Flask 服务器 (完整版)
│   │   ├── server_lite.py         # Flask 服务器 (轻量部署版)
│   │   ├── static/                # 前端页面
│   │   │   ├── home.html          # 项目主页
│   │   │   ├── index.html         # 聊天界面
│   │   │   ├── docs.html          # API 文档
│   │   │   └── pitch.html         # 商业计划书
│   │   └── deploy/                # 部署脚本
│   ├── gemini_api/                # Gemini 兼容 API Gateway
│   └── examples/                  # 示例脚本
├── contexts/                      # 领域上下文
│   └── training/                  # 训练相关
│       └── scripts/               # 微调脚本
├── experiments/                   # 实验代码
│   ├── vision/                    # 图像理解实验
│   └── audio/                     # 音频理解实验
├── infra/                         # 基础设施
│   └── llama.cpp/                 # 推理引擎 (子模块)
├── artifacts/                     # 产物输出
│   ├── gguf/                      # GGUF 模型文件
│   ├── lora/                      # LoRA 适配器
│   └── merged_model/              # 合并后的模型
├── assets/                        # 资源文件
│   └── data/                      # 训练数据
├── docs/                          # 文档
│   └── 商业计划书/                # 商业计划书 PDF
└── scripts/                       # 工具脚本
```

## 🛠️ LingKong CLI 命令

LingKong CLI 是用 Rust 编写的命令行工具：

```bash
# 环境检查
lingkong doctor                    # 诊断系统环境

# 模型管理
lingkong model list                # 列出可用模型
lingkong model pull <name>         # 下载模型
lingkong model info <name>         # 查看模型信息
lingkong model remove <name>       # 删除模型
lingkong model link /path/to.gguf  # 链接本地模型

# 服务管理
lingkong serve start               # 启动服务
lingkong serve start --model <n>   # 指定模型启动
lingkong serve stop                # 停止服务
lingkong serve status              # 查看状态

# 配置管理
lingkong config show               # 显示配置
lingkong config edit               # 编辑配置
```

## 🎮 使用示例

### Python API 调用

```python
import requests

# 文本生成
response = requests.post(
    "http://localhost:5001/v1beta/models/gemma-3-pro-preview:generateContent",
    json={
        "contents": [{"parts": [{"text": "你好，介绍一下你自己"}]}]
    }
)
print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
```

### 多模态示例

```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import torch

model = Gemma3nForConditionalGeneration.from_pretrained(
    "google/gemma-3n-E2B-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained("google/gemma-3n-E2B-it")

# 图像理解
image = Image.open("your_image.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "描述这张图片"}
]}]

inputs = processor.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### LoRA 微调

```bash
# 准备训练数据 (assets/data/train.jsonl)
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# 运行微调
python contexts/training/scripts/finetune_lora.py

# 测试微调效果
python contexts/training/scripts/test_finetuned_model.py

# 合并 LoRA 权重
python contexts/training/scripts/merge_lora.py

# 转换为 GGUF
python infra/llama.cpp/convert_hf_to_gguf.py artifacts/merged_model \
    --outfile artifacts/gguf/model.gguf --outtype f16
```

## 💻 硬件要求

| 配置 | 规格 | 适用场景 | 参考价格 |
|------|------|----------|----------|
| 入门级 | Mac Mini M2 8GB | 纯文本对话 | ~$600 |
| **推荐** | Mac Mini M4 24GB | 多模态推理 | ~$1,200 |
| 专业级 | Mac Studio / RTX 4090 | 微调训练 | ~$4,000+ |

## 📊 性能指标

| 指标 | M4 Max 128GB |
|------|--------------|
| 模型加载 | ~3.7 秒 |
| 推理速度 (PyTorch) | ~16 tokens/s |
| 推理速度 (llama.cpp) | ~94 tokens/s |
| LoRA 微调 | ~37 秒/epoch |

## 🔧 开发指南

### 编译 LingKong CLI

```bash
cd ../LingKong-AI/src
cargo build --release
```

### 运行测试

```bash
# Python 测试
python -m pytest tests/

# Rust 测试
cd ../LingKong-AI/src
cargo test
```

### 代码风格

```bash
# Python
pip install black isort
black .
isort .

# Rust
cargo fmt
cargo clippy
```

## 📚 文档

- [API 文档](http://115.159.223.227/static/docs.html) - Gemini 兼容 API 使用指南
- [部署指南](apps/webui/deploy/README.md) - 服务器部署说明
- [架构设计](docs/architecture/README.md) - 系统架构文档
- [商业计划书](http://115.159.223.227/static/pitch.html) - 项目愿景与商业模式

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🔗 链接

- **在线演示**: http://115.159.223.227
- **GitHub**: https://github.com/jiaqiwang969/gemma
- **商业计划书**: http://115.159.223.227/static/pitch.html

---

<p align="center">
  <strong>灵空 AI</strong> - 你的 AI. 你的数据. 你的掌控.
</p>

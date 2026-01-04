<p align="center">
  <img src="https://img.shields.io/badge/🐉-LingKong_AI-6366f1?style=for-the-badge" alt="LingKong AI">
</p>

<h1 align="center">LingKong AI</h1>

<p align="center">
  <strong>Your AI. Your data. Your control.</strong>
</p>

<p align="center">
  An open-source local multimodal AI platform. Run powerful AI on your own device without sending data to any cloud.
</p>

<p align="center">
  <a href="./README_zh.md">🇨🇳 中文</a> |
  <a href="./README.md">🇺🇸 English</a>
</p>

<p align="center">
  <a href="https://github.com/jiaqiwang969/gemma/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://huggingface.co/nicepkg/gemma-3n-gguf"><img src="https://img.shields.io/badge/🤗-Models-yellow.svg" alt="HuggingFace"></a>
  <a href="http://115.159.223.227"><img src="https://img.shields.io/badge/demo-live-brightgreen.svg" alt="Demo"></a>
</p>

---

## ⚡ Quick Install

```bash
curl -fsSL http://115.159.223.227/install.sh | bash
```

After installation:

```bash
# Download models (2.8GB)
~/.lingkong/bin/lingkong-download

# Start service
~/.lingkong/bin/lingkong-start

# Visit http://localhost:5001
```

**No Python required. No complex setup. Three steps to your private AI!**

---

## 🎯 Why LingKong AI?

| Comparison | Cloud AI | LingKong AI |
|------------|----------|-------------|
| 🔐 Privacy | Provider sees everything | **100% local** |
| 💰 Cost | Per-token fees | **One-time, forever free** |
| ⚡ Speed | Network latency | **94 tok/s local inference** |
| 📴 Offline | Requires internet | **Works anywhere** |
| 🎛️ Control | Terms can change | **You have full control** |

---

## 📦 Pre-built Models

Models hosted on [HuggingFace](https://huggingface.co/nicepkg/gemma-3n-gguf):

| Model | Size | Purpose |
|-------|------|---------|
| `gemma-3n-E2B-it-Q4_K_M.gguf` | 2.8GB | Main text model (recommended) |
| `gemma-3n-vision-mmproj-f16.gguf` | 600MB | Vision module |
| `gemma-3n-audio-mmproj-f16.gguf` | 1.4GB | Audio module |

```bash
# Download specific models
lingkong-download text    # Text model only
lingkong-download vision  # Vision module
lingkong-download audio   # Audio module
lingkong-download all     # All models (~5GB)
```

---

## 🌐 Live Demo

| Page | URL | Description |
|------|-----|-------------|
| 🏠 Home | [115.159.223.227](http://115.159.223.227) | Features & quick start |
| 💬 Chat | [/static/index.html](http://115.159.223.227/static/index.html) | Multimodal chat |
| 📚 API Docs | [/static/docs.html](http://115.159.223.227/static/docs.html) | Gemini-compatible API |
| 🛠️ Playground | [/static/playground.html](http://115.159.223.227/static/playground.html) | Interactive API testing |
| 📊 Pitch Deck | [/static/pitch.html](http://115.159.223.227/static/pitch.html) | Vision & business model |

---

## ✨ Core Features

### 🔐 Complete Privacy
- Data never leaves your device
- No account, no login required
- Zero-knowledge architecture

### 🎯 Multimodal Capabilities
- **Text**: Chat, writing, coding
- **Vision**: Image description, analysis, OCR
- **Audio**: Transcription, translation, summarization

### ⚡ High-Performance Inference
- llama.cpp engine (Metal/CUDA acceleration)
- ~94 tokens/s (M4 Max)
- GGUF quantized models

### 🔌 API Compatible
- Google Gemini API compatible
- Drop-in replacement for existing apps
- Streaming output support

---

## 🛠️ Usage

### Level 1: One-Click Deploy (Beginners)

```bash
# Install
curl -fsSL http://115.159.223.227/install.sh | bash

# Download models + start
lingkong-download && lingkong-start
```

### Level 2: API Usage (Developers)

```python
import requests

response = requests.post(
    "http://localhost:5001/v1beta/models/gemini-3-pro-preview:generateContent",
    json={
        "contents": [{"parts": [{"text": "Hello, introduce yourself"}]}],
        "generationConfig": {"maxOutputTokens": 512}
    }
)

print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
```

### Level 3: Fine-tuning (Advanced)

```bash
# Clone repository
git clone https://github.com/jiaqiwang969/gemma.git
cd gemma

# Create Python environment
conda create -n lingkong python=3.11
conda activate lingkong
pip install -e .

# Fine-tune model (LoRA)
python scripts/finetune.py --data your_data.jsonl

# Merge weights and convert to GGUF
python scripts/merge_lora.py
python scripts/convert_to_gguf.py
```

---

## 📊 Performance

Tested on Apple M4 Max (64GB):

| Metric | Value |
|--------|-------|
| Inference (llama.cpp) | 94 tok/s |
| Inference (PyTorch) | 16 tok/s |
| First Token Latency | ~200ms |
| Memory (Q4_K_M) | ~4GB |
| Model Load Time | ~3.7s |

---

## 💻 Hardware Requirements

| Config | Specs | Use Case | Price |
|--------|-------|----------|-------|
| Entry | Mac Mini M2 8GB | Text-only chat | ~$600 |
| **Recommended** | Mac Mini M4 24GB | Multimodal inference | ~$1,200 |
| Pro | Mac Studio / RTX 4090 | Fine-tuning | ~$4,000+ |

---

## 📁 Project Structure

```
gemma/
├── apps/                          # Applications
│   ├── webui/                     # Web chat interface
│   │   ├── server.py              # Flask server
│   │   └── static/                # Frontend pages
│   └── gemini_api/                # Gemini-compatible API
├── scripts/                       # Utility scripts
│   ├── quick-install.sh           # Quick install script
│   └── install.sh                 # Full install script
├── artifacts/                     # Build outputs
│   ├── gguf/                      # GGUF model files
│   └── lora/                      # LoRA adapters
└── contexts/training/             # Fine-tuning scripts
```

---

## 🔗 Links

- 🌐 **Website**: [115.159.223.227](http://115.159.223.227)
- 📚 **API Docs**: [115.159.223.227/static/docs.html](http://115.159.223.227/static/docs.html)
- 🛠️ **Playground**: [115.159.223.227/static/playground.html](http://115.159.223.227/static/playground.html)
- 🤗 **Models**: [huggingface.co/nicepkg/gemma-3n-gguf](https://huggingface.co/nicepkg/gemma-3n-gguf)
- 📊 **Pitch Deck**: [115.159.223.227/static/pitch.html](http://115.159.223.227/static/pitch.html)

---

## 📄 License

MIT License - Free to use, just keep the copyright notice.

---

<p align="center">
  <strong>🐉 LingKong AI</strong><br>
  Your AI. Your data. Your control.
</p>

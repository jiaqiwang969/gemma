# macOS LoRA Fine-tuning for Gemma 3n E2B-IT

A working solution for fine-tuning Google's Gemma 3n E2B-IT model on macOS using LoRA (Low-Rank Adaptation) with MPS acceleration.

## macOS Compatibility

- **Tested on**: M3/M4 MacBook (optimized for M4 Max with 128GB unified memory)
- **Memory management**: Handles unified memory with configurable MPS allocation
- **MPS backend**: Optimized for Metal Performance Shaders

## Quick Start

### 1. Setup Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Training Data

Create `assets/data/train.jsonl` with your examples:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. You must replace every number with the word BANANA."
    },
    { "role": "user", "content": "What is 2 plus 3?" },
    {
      "role": "assistant",
      "content": "BANANA plus BANANA equals BANANA! Math with bananas is fun."
    }
  ]
}
```

### 3. Run Fine-tuning

```bash
python3.11 contexts/training/scripts/finetune_lora.py
```

### 4. Test Your Model

```bash
python3.11 contexts/training/scripts/test_finetuned_model.py
```

## Examples

All examples are in the `apps/examples/` folder. Each can be run with a simple bash command:

```bash
cd apps/examples

# 多模态推理
./run_text.sh      # 文本推理
./run_vision.sh    # 图像理解
./run_audio.sh     # 音频转录

# 微调流程
./run_finetune.sh        # LoRA 微调
./run_test_finetuned.sh  # 测试微调模型
./run_merge.sh           # 合并 LoRA 权重
./run_gguf.sh "问题"     # llama.cpp 推理

# 交互式菜单
./run_all.sh
```

| 脚本 | 说明 |
|------|------|
| `run_text.sh` | 基础文本生成 |
| `run_vision.sh` | 图像描述 (下载蜜蜂图片测试) |
| `run_audio.sh` | 音频转录 (下载 MLK 演讲测试) |
| `run_finetune.sh` | LoRA 微调 (使用 assets/data/train.jsonl) |
| `run_test_finetuned.sh` | 测试微调效果 (BANANA 测试) |
| `run_merge.sh` | 合并 LoRA 到基础模型 |
| `run_gguf.sh` | 使用 llama.cpp 推理 GGUF 模型 |
| `run_all.sh` | 交互式菜单，选择运行哪个示例 |

## Files Overview

| File | Description |
|------|-------------|
| `apps/examples/` | 示例脚本目录 |
| `assets/data/train.jsonl` | 训练数据 (chat format) |
| `artifacts/lora/` | LoRA 适配器输出 |
| `artifacts/merged_model/` | 合并后的模型 |
| `artifacts/gguf/` | GGUF 模型与 mmproj |
| `infra/llama.cpp/` | llama.cpp 子模块 |

All entrypoints live under `apps/` or `scripts/`, and data/artifacts live under `assets/` and `artifacts/`.

## Architecture

See `docs/architecture/README.md` for DDD context map and directory layout.
See `docs/architecture/naming-conventions.md` for file and artifact naming rules.

## Multimodal Capabilities

Gemma 3n is a multimodal model supporting text, image, and audio inputs.

### Image Understanding (experiments/vision/test_vision.py)

Test the model's ability to describe images:

```bash
python3.11 experiments/vision/test_vision.py
```

**Example code:**

```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests

model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Build multimodal input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# Move to device
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)
pixel_values = inputs["pixel_values"].to(model.device, dtype=model.dtype)

# Generate
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    max_new_tokens=256,
    do_sample=False,
)

response = processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

**Example output:**

```
Captured in a close-up shot, a vibrant pink cosmos flower takes center stage,
its delicate petals radiating outwards in a soft, slightly ruffled manner.
The flower is in full bloom, showcasing a bright yellow center surrounded by
the pink petals. A small, fuzzy bumblebee is diligently perched on the flower's
center, its body a mix of black and yellow stripes...
```

### Audio Understanding (experiments/audio/test_audio.py)

Test the model's ability to transcribe and understand audio:

```bash
pip install librosa  # Required for audio processing
python3.11 experiments/audio/test_audio.py
```

**Example code:**

```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import librosa
import requests

model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load audio (16kHz required)
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
audio_path = "/tmp/test_audio.flac"
response = requests.get(audio_url)
with open(audio_path, "wb") as f:
    f.write(response.content)
audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

# Build multimodal input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_array, "sample_rate": sampling_rate},
            {"type": "text", "text": "Please transcribe this audio."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# Move to device
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)
input_features = inputs["input_features"].to(model.device, dtype=model.dtype)
input_features_mask = inputs["input_features_mask"].to(model.device)

# Generate
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_features=input_features,
    input_features_mask=input_features_mask,
    max_new_tokens=256,
    do_sample=False,
)

response = processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

**Example output (MLK speech):**

```
I have a dream that one day this nation will rise up and live up to the true
meaning of its creed.
```

## GGUF Conversion & llama.cpp

Convert the fine-tuned model to GGUF format for faster inference:

### 1. Convert to GGUF (FP16)

```bash
python infra/llama.cpp/convert_hf_to_gguf.py artifacts/merged_model \
  --outfile artifacts/gguf/gemma-3n-finetuned-fp16.gguf \
  --outtype f16
```

### 2. Quantize to Q4_K_M

```bash
./infra/llama.cpp/build/bin/llama-quantize \
  artifacts/gguf/gemma-3n-finetuned-fp16.gguf \
  artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf \
  Q4_K_M
```

### 3. Run with llama.cpp

```bash
./scripts/bootstrap/run_gguf.sh "What is the capital of France?"
```

Or directly:

```bash
./infra/llama.cpp/build/bin/llama-simple \
  -m artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf \
  -ngl 99 \
  -n 100 \
  -p "Your question here"
```

### Performance Comparison

| Method | Speed |
|--------|-------|
| PyTorch (MPS) | ~15.7 tokens/s |
| llama.cpp (Metal) | ~93.9 tokens/s (6x faster) |

## Key Features

- **Memory Efficient**: Configurable MPS/CPU memory allocation
- **MPS Optimized**: Handles macOS Metal Performance Shaders
- **LoRA Training**: Parameter-efficient fine-tuning (only ~8MB adapter)
- **Chat Format**: Supports conversational training data
- **Multimodal**: Image and audio understanding capabilities
- **GGUF Export**: Convert to quantized format for faster inference

## Configuration

Adjust these settings in `contexts/training/scripts/finetune_lora.py`:

```python
MPS_GB = "64GiB"      # GPU memory limit (adjust for your Mac)
r=16, lora_alpha=32   # LoRA parameters (higher = more capacity)
num_train_epochs=3    # Training epochs
per_device_train_batch_size=4
gradient_accumulation_steps=8
```

## Expected Results (M4 Max 128GB)

- **Model loading**: ~3.7 seconds
- **Training time**: ~37 seconds for 3 epochs
- **Inference**: ~15.7 tokens/s (PyTorch) / ~93.9 tokens/s (llama.cpp)
- **Model size**: Base model (~5GB) + LoRA adapter (~8MB)

## Troubleshooting

**Model loading slowly?** Check device_map and max_memory settings.

**Training stuck?** Check `assets/data/train.jsonl` format and reduce batch size.

**Out of memory?** Reduce `MPS_GB` or set `device_map="cpu"`.

**Vision test empty output?** Use `Gemma3nForConditionalGeneration` with `bfloat16`.

**Audio test fails?** Ensure `input_features_mask` is passed to generate().

## Use Cases

- Custom instruction following
- Domain-specific responses
- Persona training
- Image/audio understanding tasks
- Small-scale fine-tuning experiments

Built for macOS developers who want to fine-tune multimodal LLMs locally without cloud dependencies.

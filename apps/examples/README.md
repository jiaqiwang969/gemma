# Gemma 3n 示例集

所有示例都可以通过简单的 bash 命令运行。

## 快速开始

```bash
cd apps/examples
./run_all.sh  # 交互式菜单
```

## 示例列表

### 多模态推理

| 脚本 | 说明 | 交互 |
|------|------|------|
| `./run_text.sh` | 文本推理 | 交互式问答 |
| `./run_vision.sh` | 图像理解 | 支持本地图片/URL，可打开图片预览 |
| `./run_audio.sh` | 音频转录 | 支持本地音频/URL |
| `./run_vision_multi.sh` | 多图批量处理 | 多张图片联合分析 |
| `./run_multimodal.sh` | 多模态联合分析 | 文字 + 图片 + 音频 |

#### llama.cpp 多模态（图 + 音）

- 推荐顺序：音频放在前、图片在后，提示中也按同样顺序放 `<__media__>`，可确保音频被正确转写。
- 若遇到 Metal 报错或音频被忽略，可先用 CPU 路径验证（`MTMD_BACKEND_DEVICE=CPU` + `--device none --n-gpu-layers 0`）。
- 若缺少 image mmproj，可先生成：
  `python infra/llama.cpp/convert_hf_to_gguf.py google/gemma-3n-E2B-it --mmproj-type vision --outfile artifacts/gguf/gemma-3n-image-mmproj-f16.gguf`
- 单模态（llama.cpp）快捷入口：
  - `./run_text_cpp.sh`
  - `./run_vision_cpp.sh`
  - `./run_audio_cpp.sh`
- 示例命令（双 mmproj，同轮转写 MLK 音频并描述猫/狗图）：
  ```bash
  cd /Users/jqwang/151-lego-gpt-wjq/01-gemma-3n/gemma-3n-finetuning
  MTMD_BACKEND_DEVICE=CPU ./infra/llama.cpp/build/bin/llama-mtmd-cli \
    --log-verbosity 2 \
    -m artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf \
    --mmproj "artifacts/gguf/gemma-3n-image-mmproj-f16.gguf,artifacts/gguf/gemma-3n-audio-mmproj-f16.gguf" \
    --audio assets/data/audio/mlk_speech.wav \
    --image assets/data/images/cat.jpg,assets/data/images/dog.jpg \
    -p "<__media__><__media__><__media__> 我将依次提供音频、图片1、图片2。请先转写音频，再描述两张图片并比较异同，最后说明三者是否有关联。" \
    -n 200 --temp 0 --no-warmup \
    --device none --n-gpu-layers 0
  ```

### 微调流程

| 脚本 | 说明 |
|------|------|
| `./run_finetune.sh` | LoRA 微调 (使用 assets/data/train.jsonl) |
| `./run_test_finetuned.sh` | 测试 BANANA 微调效果 |
| `./run_merge.sh` | 合并 LoRA 权重到基础模型 |
| `./run_gguf.sh` | llama.cpp 推理 (需要先转换 GGUF) |

## 使用示例

### 1. 文本推理

```bash
./run_text.sh
# 输入问题进行对话，输入 'quit' 退出
```

### 2. 图像理解

```bash
./run_vision.sh
# 直接回车使用默认蜜蜂图片
# 或输入本地路径: /path/to/image.jpg
# 或输入 URL: https://example.com/image.jpg
```

### 3. 音频转录

```bash
./run_audio.sh
# 直接回车使用默认 MLK 演讲
# 支持 wav, mp3, flac, ogg 等格式
```

### 4. 完整微调流程

```bash
# 步骤1: 训练
./run_finetune.sh

# 步骤2: 测试效果
./run_test_finetuned.sh

# 步骤3: 合并权重
./run_merge.sh

# 步骤4 (可选): 转换 GGUF 后运行
./run_gguf.sh "What is 2 plus 3?"
```

## 文件结构

```
apps/examples/
├── text_inference.py        # 文本推理
├── vision_inference.py      # 图像理解
├── audio_inference.py       # 音频转录
├── vision_multi_images.py   # 多图批量处理
├── multimodal_inference.py  # 多模态联合分析
├── text_cpp.sh              # llama.cpp 文本推理
├── vision_cpp.sh            # llama.cpp 图像理解
├── audio_cpp.sh             # llama.cpp 音频转录
├── finetune_lora.py         # LoRA 微调
├── test_finetuned_model.py  # 测试微调模型
├── merge_and_export.py      # 合并并导出
├── run_text.sh              # 运行脚本
├── run_vision.sh
├── run_audio.sh
├── run_vision_multi.sh
├── run_multimodal.sh
├── run_text_cpp.sh
├── run_vision_cpp.sh
├── run_audio_cpp.sh
├── run_finetune.sh
├── run_test_finetuned.sh
├── run_merge.sh
├── run_gguf.sh
├── run_all.sh             # 交互式菜单
└── README.md              # 本文件
```

## 依赖

确保已安装所有依赖:

```bash
pip install -r requirements.txt
pip install librosa  # 音频处理需要
```

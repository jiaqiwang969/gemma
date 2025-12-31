# Gemma 3n 示例集

所有示例都可以通过简单的 bash 命令运行。

## 快速开始

```bash
cd examples
./run_all.sh  # 交互式菜单
```

## 示例列表

### 多模态推理

| 脚本 | 说明 | 交互 |
|------|------|------|
| `./run_1_text.sh` | 文本推理 | 交互式问答 |
| `./run_2_vision.sh` | 图像理解 | 支持本地图片/URL，可打开图片预览 |
| `./run_3_audio.sh` | 音频转录 | 支持本地音频/URL |

### 微调流程

| 脚本 | 说明 |
|------|------|
| `./run_4_finetune.sh` | LoRA 微调 (使用 train.jsonl) |
| `./run_5_test_finetuned.sh` | 测试 BANANA 微调效果 |
| `./run_6_merge.sh` | 合并 LoRA 权重到基础模型 |
| `./run_7_gguf.sh` | llama.cpp 推理 (需要先转换 GGUF) |

## 使用示例

### 1. 文本推理

```bash
./run_1_text.sh
# 输入问题进行对话，输入 'quit' 退出
```

### 2. 图像理解

```bash
./run_2_vision.sh
# 直接回车使用默认蜜蜂图片
# 或输入本地路径: /path/to/image.jpg
# 或输入 URL: https://example.com/image.jpg
```

### 3. 音频转录

```bash
./run_3_audio.sh
# 直接回车使用默认 MLK 演讲
# 支持 wav, mp3, flac, ogg 等格式
```

### 4. 完整微调流程

```bash
# 步骤1: 训练
./run_4_finetune.sh

# 步骤2: 测试效果
./run_5_test_finetuned.sh

# 步骤3: 合并权重
./run_6_merge.sh

# 步骤4 (可选): 转换 GGUF 后运行
./run_7_gguf.sh "What is 2 plus 3?"
```

## 文件结构

```
examples/
├── 1_text_inference.py    # 文本推理
├── 2_vision.py            # 图像理解
├── 3_audio.py             # 音频转录
├── 4_finetune.py          # LoRA 微调
├── 5_test_finetuned.py    # 测试微调模型
├── 6_merge_and_export.py  # 合并并导出
├── run_1_text.sh          # 运行脚本
├── run_2_vision.sh
├── run_3_audio.sh
├── run_4_finetune.sh
├── run_5_test_finetuned.sh
├── run_6_merge.sh
├── run_7_gguf.sh
├── run_all.sh             # 交互式菜单
└── README.md              # 本文件
```

## 依赖

确保已安装所有依赖:

```bash
pip install -r requirements.txt
pip install librosa  # 音频处理需要
```

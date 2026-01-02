#!/bin/bash
# 示例4: LoRA 微调
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
source venv/bin/activate
python3.11 apps/examples/finetune_lora.py

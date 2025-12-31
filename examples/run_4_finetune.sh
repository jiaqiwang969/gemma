#!/bin/bash
# 示例4: LoRA 微调
cd "$(dirname "$0")/.."
source venv/bin/activate
python3.11 examples/4_finetune.py

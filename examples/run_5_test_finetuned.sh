#!/bin/bash
# 示例5: 测试微调后的模型
cd "$(dirname "$0")/.."
source venv/bin/activate
python3.11 examples/5_test_finetuned.py

#!/bin/bash
# 示例5: 测试微调后的模型
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
source venv/bin/activate
python3.11 apps/examples/test_finetuned_model.py

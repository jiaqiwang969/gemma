#!/bin/bash
# 示例6: 合并 LoRA 并导出
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
source venv/bin/activate
python3.11 apps/examples/merge_and_export.py

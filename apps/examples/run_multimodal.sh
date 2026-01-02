#!/bin/bash
# 示例2c: 多模态联合分析 (文字 + 图片 + 音频)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
source venv/bin/activate
python3.11 apps/examples/multimodal_inference.py

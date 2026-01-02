#!/bin/bash
# 示例3: 音频理解
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
source venv/bin/activate
python3.11 apps/examples/audio_inference.py

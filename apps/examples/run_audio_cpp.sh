#!/bin/bash
# 示例3: 音频理解 (llama.cpp)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
./apps/examples/audio_cpp.sh

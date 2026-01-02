#!/bin/bash
# 示例1: 文本推理 (llama.cpp)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
./apps/examples/text_cpp.sh

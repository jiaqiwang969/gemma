#!/bin/bash
# llama.cpp GGUF 模型运行脚本
# 用法: ./scripts/bootstrap/run_gguf.sh "你的问题"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"

MODEL="artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf"
PROMPT="${1:-What is the capital of France?}"
LLAMA_BIN_DIR="infra/llama.cpp/build/bin"
export DYLD_LIBRARY_PATH="$LLAMA_BIN_DIR:${DYLD_LIBRARY_PATH:-}"

echo "=========================================="
echo "问题: $PROMPT"
echo "=========================================="

./infra/llama.cpp/build/bin/llama-simple \
  -m "$MODEL" \
  -ngl 99 \
  -n 100 \
  -p "$PROMPT" \
  2>/dev/null | tail -n +2

echo ""
echo "=========================================="

#!/bin/bash
# llama.cpp GGUF 模型运行脚本
# 用法: ./run_gguf.sh "你的问题"

MODEL="outputs/gemma-3n-finetuned-Q4_K_M.gguf"
PROMPT="${1:-What is the capital of France?}"

echo "=========================================="
echo "问题: $PROMPT"
echo "=========================================="

./llama.cpp/build/bin/llama-simple \
  -m "$MODEL" \
  -ngl 99 \
  -n 100 \
  -p "$PROMPT" \
  2>/dev/null | tail -n +2

echo ""
echo "=========================================="

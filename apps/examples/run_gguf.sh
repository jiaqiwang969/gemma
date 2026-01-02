#!/bin/bash
# 示例7: 使用 llama.cpp 运行 GGUF 模型
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"

MODEL="artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf"
PROMPT="${1:-What is 2 plus 3?}"
LLAMA_BIN_DIR="infra/llama.cpp/build/bin"
export DYLD_LIBRARY_PATH="$LLAMA_BIN_DIR:${DYLD_LIBRARY_PATH:-}"

if [ ! -f "$MODEL" ]; then
    echo "错误: GGUF 模型不存在: $MODEL"
    echo "请先运行 run_merge.sh 然后手动转换为 GGUF"
    exit 1
fi

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

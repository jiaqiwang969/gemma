#!/bin/bash
# 示例1: 文本推理 - 交互式对话 (llama.cpp)
# 使用 llama-run 进行单次问答

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"

# 模型路径 (优先微调版，支持 LLAMA_MODEL 覆盖)
LLAMA_RUN="infra/llama.cpp/build/bin/llama-run"
LLAMA_BIN_DIR="$(dirname "$LLAMA_RUN")"
export DYLD_LIBRARY_PATH="$LLAMA_BIN_DIR:${DYLD_LIBRARY_PATH:-}"
MODEL="${LLAMA_MODEL:-}"
if [ -z "$MODEL" ]; then
    for candidate in \
        "artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf" \
        "artifacts/gguf/gemma-3n-E2B-it-Q4_K_M.gguf" \
        "artifacts/gguf/gemma-3n-E2B-it-fp16.gguf"; do
        if [ -f "$candidate" ]; then
            MODEL="$candidate"
            break
        fi
    done
fi

echo "============================================================"
echo "示例1: Gemma 3n 文本推理 (llama.cpp)"
echo "============================================================"

# 检查模型文件
echo ""
echo "[1] 检查模型..."
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "错误: 找不到 GGUF 模型"
    echo "请先生成模型或设置 LLAMA_MODEL，例如:"
    echo "  LLAMA_MODEL=artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf"
    exit 1
fi

if [ ! -f "$LLAMA_RUN" ]; then
    echo "错误: 找不到 llama-run"
    echo "请先编译 llama.cpp: cd infra/llama.cpp && cmake -B build && cmake --build build -j"
    exit 1
fi

echo "    使用模型: $MODEL"
echo "    准备就绪!"

echo ""
echo "============================================================"
echo "开始交互式问答 (输入 'quit' 退出)"
echo "============================================================"

while true; do
    echo ""
    read -p "请输入问题: " question
    question=$(echo "$question" | xargs)

    if [[ "$question" == "quit" || "$question" == "exit" || "$question" == "q" ]]; then
        echo "再见!"
        break
    fi

    if [ -z "$question" ]; then
        continue
    fi

    echo ""
    echo "问题: $question"
    echo "----------------------------------------"
    echo "生成中..."

    response=$($LLAMA_RUN \
        --ngl 999 \
        --temp 0.7 \
        "$MODEL" \
        "$question" \
        2>/tmp/llama-run.err)

    if [ -z "$response" ] && [ -s /tmp/llama-run.err ]; then
        echo ""
        echo "出现错误，请检查日志:"
        tail -n 5 /tmp/llama-run.err
    fi

    echo ""
    echo "回答: $response"
    echo "----------------------------------------"
done

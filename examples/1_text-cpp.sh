#!/bin/bash
# 示例1: 文本推理 - 交互式对话 (llama.cpp)
# 注意: Gemma 3n 使用 llama-cli 进行纯文本对话

cd "$(dirname "$0")/.."

# 模型路径
MODEL="outputs/gemma-3n-text.gguf"
LLAMA_CLI="llama.cpp/build/bin/llama-cli"

echo "============================================================"
echo "示例1: Gemma 3n 文本推理"
echo "============================================================"

# 检查模型文件
echo ""
echo "[1] 加载模型..."
if [ ! -f "$MODEL" ]; then
    echo "错误: 找不到模型文件 $MODEL"
    echo "请先运行 GGUF 转换脚本"
    exit 1
fi

if [ ! -f "$LLAMA_CLI" ]; then
    echo "错误: 找不到 llama-cli"
    echo "请先编译 llama.cpp: cd llama.cpp && cmake -B build && cmake --build build -j"
    exit 1
fi

echo "    模型加载完成!"

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

    response=$($LLAMA_CLI \
        -m "$MODEL" \
        -p "$question" \
        -n 200 \
        --temp 0 \
        -ngl 99 \
        --no-display-prompt \
        2>/dev/null)

    echo ""
    echo "回答: $response"
    echo "----------------------------------------"
done

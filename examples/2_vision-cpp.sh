#!/bin/bash
# 示例2: 图像理解 - 测试模型的视觉能力 (llama.cpp)
# 支持：本地图片路径 或 URL

cd "$(dirname "$0")/.."

# 模型路径
MODEL="outputs/gemma-3n-text.gguf"
MMPROJ="outputs/mmproj-gemma-3n-E2B-it-f16.gguf"
LLAMA_MTMD="llama.cpp/build/bin/llama-mtmd-cli"

# 默认测试图片
DEFAULT_URL="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
DEFAULT_IMAGE="data/images/bee.jpg"

echo "============================================================"
echo "示例2: Gemma 3n 图像理解"
echo "============================================================"

# 检查模型文件
echo ""
echo "[1] 加载模型..."
if [ ! -f "$MODEL" ]; then
    echo "错误: 找不到模型文件 $MODEL"
    echo "请先运行 GGUF 转换脚本"
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "错误: 找不到视觉编码器 $MMPROJ"
    echo "请先运行视觉编码器转换脚本"
    exit 1
fi

if [ ! -f "$LLAMA_MTMD" ]; then
    echo "错误: 找不到 llama-mtmd-cli"
    echo "请先编译 llama.cpp: cd llama.cpp && cmake -B build && cmake --build build -j"
    exit 1
fi

echo "    模型加载完成!"

# 辅助函数：显示图片信息
display_image_info() {
    local image_file="$1"
    echo ""
    echo "图片来源: $image_file"
    if command -v identify &> /dev/null; then
        local size=$(identify -format "%wx%h" "$image_file" 2>/dev/null)
        local mode=$(identify -format "%[colorspace]" "$image_file" 2>/dev/null)
        [ -n "$size" ] && echo "图片大小: $size"
        [ -n "$mode" ] && echo "图片模式: $mode"
    elif command -v file &> /dev/null; then
        file "$image_file" | sed 's/.*: //'
    fi
}

# 辅助函数：运行推理
ask_about_image() {
    local image_file="$1"
    local question="$2"

    $LLAMA_MTMD \
        -m "$MODEL" \
        -mm "$MMPROJ" \
        --image "$image_file" \
        -p "$question" \
        -ngl 99 \
        2>&1 | grep -v "^ggml\|^llama\|^print_info\|^load\|^common\|^clip\|^warmup\|^alloc\|^init\|^main\|^WARN\|^mtmd\|^Using"
}

echo ""
echo "============================================================"
echo "图像理解交互模式"
echo "============================================================"
echo ""
echo "输入图片路径或URL (直接回车使用默认蜜蜂图片)"
echo "输入 'quit' 退出"

current_image=""

while true; do
    echo ""

    if [ -z "$current_image" ]; then
        read -p "图片路径/URL: " path
        path=$(echo "$path" | xargs)  # trim whitespace

        if [[ "$path" == "quit" || "$path" == "exit" || "$path" == "q" ]]; then
            echo "再见!"
            break
        fi

        if [ -z "$path" ]; then
            # 使用默认图片
            if [ ! -f "$DEFAULT_IMAGE" ]; then
                echo "下载默认图片..."
                mkdir -p data/images
                curl -L -o "$DEFAULT_IMAGE" "$DEFAULT_URL"
            fi
            path="$DEFAULT_IMAGE"
            echo "使用默认图片: $path"
        fi

        # 检查是否为URL
        if [[ "$path" == http://* || "$path" == https://* ]]; then
            echo "下载图片中..."
            temp_image="/tmp/temp_image_file"
            curl -L -o "$temp_image" "$path" 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "加载图片失败"
                continue
            fi
            current_image="$temp_image"
        elif [ -f "$path" ]; then
            current_image="$path"
        else
            echo "加载图片失败: 文件不存在"
            continue
        fi

        display_image_info "$current_image"
    fi

    echo ""
    echo "可选操作:"
    echo "  1. 输入问题 - 询问关于图片的问题"
    echo "  2. 输入 'new' - 加载新图片"
    echo "  3. 输入 'quit' - 退出"

    read -p $'\n请输入问题或命令: ' action
    action=$(echo "$action" | xargs)

    if [[ "$action" == "quit" || "$action" == "exit" || "$action" == "q" ]]; then
        echo "再见!"
        break
    elif [ "$action" == "new" ]; then
        current_image=""
        continue
    elif [ -z "$action" ]; then
        action="Describe this image in detail."
        echo "使用默认问题: $action"
    fi

    echo ""
    echo "问题: $action"
    echo "----------------------------------------"
    echo "分析图片中..."

    response=$(ask_about_image "$current_image" "$action")

    echo ""
    echo "回答:"
    echo "$response"
    echo "----------------------------------------"
done

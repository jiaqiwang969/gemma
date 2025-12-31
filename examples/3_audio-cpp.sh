#!/bin/bash
# 示例3: 音频理解 - 测试模型的语音转录能力 (llama.cpp)
# 支持：本地音频文件 或 URL

cd "$(dirname "$0")/.."

# 模型路径
MODEL="outputs/gemma-3n-text.gguf"
MMPROJ="outputs/gemma-3n-audio-mmproj-f16.gguf"
LLAMA_MTMD="llama.cpp/build/bin/llama-mtmd-cli"

# 默认测试音频 (MLK "I Have a Dream" 片段)
DEFAULT_URL="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
DEFAULT_AUDIO="data/audio/mlk_speech.flac"

echo "============================================================"
echo "示例3: Gemma 3n 音频理解"
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
    echo "错误: 找不到音频编码器 $MMPROJ"
    echo "请先运行音频编码器转换脚本:"
    echo "  python llama.cpp/convert_hf_to_gguf.py google/gemma-3n-E2B-it \\"
    echo "    --mmproj-type audio \\"
    echo "    --outfile outputs/gemma-3n-audio-mmproj-f16.gguf"
    exit 1
fi

if [ ! -f "$LLAMA_MTMD" ]; then
    echo "错误: 找不到 llama-mtmd-cli"
    echo "请先编译 llama.cpp: cd llama.cpp && cmake -B build && cmake --build build -j"
    exit 1
fi

echo "    模型加载完成!"

# 辅助函数：显示音频信息
display_audio_info() {
    local audio_file="$1"
    echo ""
    echo "音频来源: $audio_file"
    if command -v ffprobe &> /dev/null; then
        local sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "$audio_file" 2>/dev/null)
        local duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$audio_file" 2>/dev/null)
        local samples=$(ffprobe -v error -select_streams a:0 -show_entries stream=duration_ts -of default=noprint_wrappers=1:nokey=1 "$audio_file" 2>/dev/null)
        [ -n "$sr" ] && echo "采样率: $sr Hz"
        [ -n "$duration" ] && printf "时长: %.2f 秒\n" "$duration"
        [ -n "$samples" ] && echo "样本数: $samples"
    fi
}

# 辅助函数：运行推理
ask_about_audio() {
    local audio_file="$1"
    local question="$2"

    $LLAMA_MTMD \
        -m "$MODEL" \
        -mm "$MMPROJ" \
        --audio "$audio_file" \
        -p "$question" \
        -ngl 99 \
        2>&1 | grep -v "^ggml\|^llama\|^print_info\|^load\|^common\|^clip\|^warmup\|^alloc\|^init\|^main\|^WARN\|^mtmd\|^gemma3na\|^encoding\|^decoding\|^audio\|^Using"
}

echo ""
echo "============================================================"
echo "音频理解交互模式"
echo "============================================================"
echo ""
echo "输入音频路径或URL (直接回车使用默认 MLK 演讲)"
echo "支持格式: wav, mp3, flac, ogg 等"
echo "输入 'quit' 退出"

current_audio=""

while true; do
    echo ""

    if [ -z "$current_audio" ]; then
        read -p "音频路径/URL: " path
        path=$(echo "$path" | xargs)  # trim whitespace

        if [[ "$path" == "quit" || "$path" == "exit" || "$path" == "q" ]]; then
            echo "再见!"
            break
        fi

        if [ -z "$path" ]; then
            # 使用默认音频
            if [ ! -f "$DEFAULT_AUDIO" ]; then
                echo "下载默认音频..."
                mkdir -p data/audio
                curl -L -o "$DEFAULT_AUDIO" "$DEFAULT_URL"
            fi
            path="$DEFAULT_AUDIO"
            echo "使用默认音频: MLK 'I Have a Dream' 演讲片段"
        fi

        # 检查是否为URL
        if [[ "$path" == http://* || "$path" == https://* ]]; then
            echo "加载音频中..."
            temp_audio="/tmp/temp_audio_file"
            curl -L -o "$temp_audio" "$path" 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "加载音频失败"
                continue
            fi
            current_audio="$temp_audio"
        elif [ -f "$path" ]; then
            current_audio="$path"
        else
            echo "加载音频失败: 文件不存在"
            continue
        fi

        display_audio_info "$current_audio"
    fi

    echo ""
    echo "可选操作:"
    echo "  1. 输入问题 - 询问关于音频的问题"
    echo "  2. 输入 'new' - 加载新音频"
    echo "  3. 输入 'quit' - 退出"
    echo ""
    echo "常用问题示例:"
    echo "  - Please transcribe this audio."
    echo "  - What language is spoken in this audio?"
    echo "  - Summarize the content of this audio."

    read -p $'\n请输入问题或命令: ' action
    action=$(echo "$action" | xargs)

    if [[ "$action" == "quit" || "$action" == "exit" || "$action" == "q" ]]; then
        echo "再见!"
        break
    elif [ "$action" == "new" ]; then
        current_audio=""
        continue
    elif [ -z "$action" ]; then
        action="Please transcribe this audio."
        echo "使用默认问题: $action"
    fi

    echo ""
    echo "问题: $action"
    echo "----------------------------------------"
    echo "分析音频中..."

    response=$(ask_about_audio "$current_audio" "$action")

    echo ""
    echo "回答:"
    echo "$response"
    echo "----------------------------------------"
done

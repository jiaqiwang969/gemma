#!/bin/bash
# 示例: 多模态推理 - 图片 + 音频 (llama.cpp)
# 使用 llama-mtmd-cli 进行多模态问答

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"

# 设置动态库路径
LLAMA_BIN_DIR="infra/llama.cpp/build/bin"
export DYLD_LIBRARY_PATH="$LLAMA_BIN_DIR:${DYLD_LIBRARY_PATH:-}"

# 模型路径
MODEL="${LLAMA_MODEL:-artifacts/gguf/gemma-3n-E2B-it-Q4_K_M.gguf}"
VISION_MMPROJ="artifacts/gguf/gemma-3n-vision-mmproj-f16.gguf"
AUDIO_MMPROJ="artifacts/gguf/gemma-3n-audio-mmproj-f16.gguf"
LLAMA_MTMD="infra/llama.cpp/build/bin/llama-mtmd-cli"

# 测试素材
TEST_IMAGE="assets/data/images/dog.jpg"
TEST_AUDIO="assets/data/audio/mlk_speech.wav"

echo "============================================================"
echo "Gemma 3n 多模态推理测试 (llama.cpp)"
echo "============================================================"

# 检查文件
echo ""
echo "[1] 检查必要文件..."
for file in "$MODEL" "$LLAMA_MTMD" "$VISION_MMPROJ" "$AUDIO_MMPROJ"; do
    if [ ! -f "$file" ]; then
        echo "错误: 文件不存在 - $file"
        exit 1
    fi
done
echo "    所有文件就绪!"

# 测试1: 纯图片
echo ""
echo "============================================================"
echo "[测试1] 图片理解"
echo "============================================================"
echo "图片: $TEST_IMAGE"
echo ""

$LLAMA_MTMD \
    -m "$MODEL" \
    --mmproj "$VISION_MMPROJ" \
    --image "$TEST_IMAGE" \
    -p "Describe this image in one sentence." \
    -n 64 \
    --temp 0 \
    --log-verbosity 0 \
    --no-warmup

# 测试2: 纯音频
echo ""
echo "============================================================"
echo "[测试2] 音频转录"
echo "============================================================"
echo "音频: $TEST_AUDIO"
echo ""

$LLAMA_MTMD \
    -m "$MODEL" \
    --mmproj "$AUDIO_MMPROJ" \
    --audio "$TEST_AUDIO" \
    -p "Transcribe this audio." \
    -n 128 \
    --temp 0 \
    --log-verbosity 0 \
    --no-warmup

# 测试3: 图片 + 音频
echo ""
echo "============================================================"
echo "[测试3] 双模态 (图片 + 音频)"
echo "============================================================"
echo "图片: $TEST_IMAGE"
echo "音频: $TEST_AUDIO"
echo ""

$LLAMA_MTMD \
    -m "$MODEL" \
    --mmproj "$VISION_MMPROJ,$AUDIO_MMPROJ" \
    --image "$TEST_IMAGE" \
    --audio "$TEST_AUDIO" \
    -p "Describe what you see and hear." \
    -n 128 \
    --temp 0 \
    --log-verbosity 0 \
    --no-warmup

echo ""
echo "============================================================"
echo "测试完成!"
echo "============================================================"

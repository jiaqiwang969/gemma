#!/bin/bash
# ============================================================
# V-JEPA2 环境安装脚本
# ============================================================
# 用途: 安装 V-JEPA2 依赖并下载模型
# 支持: macOS (MPS) / Linux (CUDA)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VJEPA2_DIR="$SCRIPT_DIR"
MODELS_DIR="$VJEPA2_DIR/models"

echo "============================================================"
echo "V-JEPA2 环境安装"
echo "============================================================"
echo "项目目录: $PROJECT_DIR"
echo "V-JEPA2目录: $VJEPA2_DIR"
echo "============================================================"

# 检测系统
if [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="macOS"
    DEVICE="mps"
else
    PLATFORM="Linux"
    DEVICE="cuda"
fi
echo "平台: $PLATFORM ($DEVICE)"

# 激活虚拟环境
cd "$PROJECT_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活虚拟环境: venv"
else
    echo "创建虚拟环境..."
    python3.11 -m venv venv
    source venv/bin/activate
fi

# 安装依赖
echo ""
echo "[1/4] 安装基础依赖..."
pip install -q --upgrade pip

# PyTorch (如果未安装)
python -c "import torch" 2>/dev/null || {
    echo "安装 PyTorch..."
    if [[ "$PLATFORM" == "macOS" ]]; then
        pip install torch torchvision torchaudio
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
}

# V-JEPA2 依赖
echo ""
echo "[2/4] 安装 V-JEPA2 依赖..."
pip install -q timm einops scipy

# macOS 需要特殊处理 decord
if [[ "$PLATFORM" == "macOS" ]]; then
    echo "macOS: 尝试安装 eva-decord (decord 替代方案)..."
    pip install -q eva-decord 2>/dev/null || {
        echo "警告: eva-decord 安装失败，将使用 opencv 作为后备方案"
        pip install -q opencv-python
    }
else
    pip install -q decord
fi

# HuggingFace transformers (用于加载模型)
pip install -q transformers accelerate

# 创建模型目录
echo ""
echo "[3/4] 准备模型目录..."
mkdir -p "$MODELS_DIR"

# 下载模型
echo ""
echo "[4/4] 下载 V-JEPA2 模型..."
echo "可选模型:"
echo "  1. ViT-L (300M 参数) - 推荐用于实时场景"
echo "  2. ViT-H (600M 参数) - 平衡选择"
echo "  3. ViT-g (1B 参数)   - 最高精度"
echo ""

# 默认下载 ViT-L (适合实时场景)
VITL_URL="https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"
VITL_PATH="$MODELS_DIR/vitl.pt"

if [ -f "$VITL_PATH" ]; then
    echo "ViT-L 模型已存在: $VITL_PATH"
else
    echo "下载 ViT-L 模型..."
    curl -L -o "$VITL_PATH" "$VITL_URL"
    echo "下载完成: $VITL_PATH"
fi

# 显示模型文件
echo ""
echo "============================================================"
echo "模型文件:"
ls -lh "$MODELS_DIR"
echo "============================================================"

# 验证安装
echo ""
echo "验证安装..."
python3.11 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

import timm
print(f'timm: {timm.__version__}')

import einops
print('einops: OK')

try:
    import decord
    print('decord: OK')
except:
    try:
        import cv2
        print('opencv: OK (decord 替代)')
    except:
        print('警告: 需要 decord 或 opencv')

print('')
print('V-JEPA2 环境安装完成!')
"

echo ""
echo "============================================================"
echo "安装完成!"
echo ""
echo "下一步:"
echo "  1. 运行测试: python3.11 vjepa2/test_vjepa2.py"
echo "  2. 如需更大模型，手动下载:"
echo "     - ViT-H: curl -o $MODELS_DIR/vith.pt https://dl.fbaipublicfiles.com/vjepa2/vith.pt"
echo "     - ViT-g: curl -o $MODELS_DIR/vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vitg.pt"
echo "============================================================"

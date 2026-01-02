#!/bin/bash
# ============================================================
# V-JEPA2 测试脚本
# ============================================================
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "============================================================"
echo "V-JEPA2 + Gemma 3n 视频理解测试"
echo "============================================================"

# 检查模型文件
if [ ! -f "vjepa2/models/vitl.pt" ]; then
    echo ""
    echo "⚠️  V-JEPA2 模型未下载"
    echo "   请先运行: ./vjepa2/setup.sh"
    echo ""
    echo "或手动下载:"
    echo "   curl -o vjepa2/models/vitl.pt https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"
    echo ""
fi

# 运行测试
python3.11 vjepa2/test_vjepa2.py "$@"

#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LingKong AI - 本地开发启动脚本
# ═══════════════════════════════════════════════════════════════════════════════
#
# 在本地 Mac 上启动 WebUI 和建立 SSH 隧道
#
# 使用方式:
#   ./apps/webui/deploy/start_local.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "LingKong AI - 本地开发环境"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "项目根目录: $ROOT_DIR"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建 Python 虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

# 安装依赖
echo "检查依赖..."
pip install -q flask flask-cors requests pillow soundfile numpy

# 选择启动模式
echo ""
echo "请选择启动模式:"
echo "  1) 仅本地开发 (localhost:5000)"
echo "  2) 本地 + SSH 隧道 (连接到 lingkong.xyz)"
echo "  3) 仅建立 SSH 隧道"
echo ""
read -p "选择 [1/2/3]: " mode

case $mode in
    1)
        echo ""
        echo "启动本地 WebUI..."
        echo "访问: http://localhost:5000"
        echo ""
        python3 apps/webui/server.py
        ;;
    2)
        echo ""
        echo "启动本地服务并建立 SSH 隧道..."
        echo ""
        echo "本地: http://localhost:5000"
        echo "公网: http://lingkong.xyz"
        echo ""

        # 在后台启动本地服务
        python3 apps/webui/server.py &
        LOCAL_PID=$!

        # 等待服务启动
        sleep 3

        # 建立 SSH 隧道
        echo "建立 SSH 隧道..."
        echo "隧道端口: 5001 (推理API) -> 远程"
        echo ""
        ssh -R 5001:127.0.0.1:5001 ubuntu@115.159.223.227

        # 清理
        kill $LOCAL_PID 2>/dev/null || true
        ;;
    3)
        echo ""
        echo "建立 SSH 隧道..."
        echo ""
        echo "确保本地服务已在运行:"
        echo "  - 推理服务: http://127.0.0.1:5001"
        echo "  - llama.cpp: http://127.0.0.1:8081"
        echo ""
        ssh -R 5001:127.0.0.1:5001 -R 8081:127.0.0.1:8081 ubuntu@115.159.223.227
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

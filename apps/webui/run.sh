#!/bin/bash
# 启动 Gemma 3n 多模态聊天 Web UI
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$ROOT_DIR/../.." && pwd -P)"
cd "$ROOT_DIR"
source venv/bin/activate

# 安装依赖（如果需要）
pip install flask flask-cors -q

echo "=========================================="
echo "Gemma 3n 多模态聊天 Web UI"
echo "=========================================="
echo ""
echo "启动服务器..."
echo "浏览器访问: http://localhost:5000"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "=========================================="

python3.11 apps/webui/server.py

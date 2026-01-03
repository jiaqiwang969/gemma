#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LingKong AI WebUI - 文件部署脚本
# ═══════════════════════════════════════════════════════════════════════════════
#
# 在本地运行，上传 WebUI 文件到远程服务器
#
# 使用方式:
#   ./apps/webui/deploy/deploy.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# 配置
REMOTE_HOST="ubuntu@115.159.223.227"
REMOTE_DIR="/opt/lingkong-webui"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "═══════════════════════════════════════════════════════════════════════"
echo "LingKong AI WebUI - 部署更新"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "本地目录: $LOCAL_DIR"
echo "远程地址: $REMOTE_HOST:$REMOTE_DIR"
echo ""

# 检查本地文件
if [ ! -f "$LOCAL_DIR/server.py" ]; then
    echo "错误: 找不到 server.py"
    echo "请确保在正确的目录运行此脚本"
    exit 1
fi

# 创建远程目录
echo "[1/4] 创建远程目录..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR/static"

# 上传核心文件
echo "[2/4] 上传核心文件..."
FILES=(
    "server.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  - $file"
        scp "$LOCAL_DIR/$file" "$REMOTE_HOST:$REMOTE_DIR/$file"
    fi
done

# 上传静态文件
echo "[3/4] 上传静态文件..."
if [ -d "$LOCAL_DIR/static" ]; then
    scp -r "$LOCAL_DIR/static/"* "$REMOTE_HOST:$REMOTE_DIR/static/"
    echo "  - static/* (所有静态文件)"
fi

# 重启服务
echo "[4/4] 重启服务..."
ssh $REMOTE_HOST "sudo systemctl restart lingkong-webui 2>/dev/null || echo '服务未运行，跳过重启'"

# 检查状态
echo ""
echo "检查服务状态..."
sleep 2
ssh $REMOTE_HOST "curl -s http://127.0.0.1:5000/api/status 2>/dev/null | head -20 || echo '服务正在启动...'"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "部署完成!"
echo ""
echo "访问地址:"
echo "  - http://lingkong.xyz (WebUI)"
echo "  - https://lingkong.xyz (HTTPS, 需先申请证书)"
echo ""
echo "查看日志:"
echo "  ssh $REMOTE_HOST 'sudo journalctl -u lingkong-webui -f'"
echo "═══════════════════════════════════════════════════════════════════════"

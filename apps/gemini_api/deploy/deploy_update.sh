#!/bin/bash
# 部署更新到远程服务器
# 用法: ./deploy_update.sh

set -e

REMOTE_HOST="ubuntu@115.159.223.227"
REMOTE_DIR="/opt/gemini-gateway"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================="
echo "灵空 AI Gateway 部署更新"
echo "=================================================="
echo "本地目录: $LOCAL_DIR"
echo "远程地址: $REMOTE_HOST:$REMOTE_DIR"
echo ""

# 需要上传的文件
FILES=(
    "gateway.py"
    "crypto_api.py"
    "static/index.html"
)

# 创建远程目录
echo "[1/4] 创建远程目录..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR/static"

# 上传文件
echo "[2/4] 上传文件..."
for file in "${FILES[@]}"; do
    echo "  - $file"
    scp "$LOCAL_DIR/$file" "$REMOTE_HOST:$REMOTE_DIR/$file"
done

# 上传加密模块 (如果存在编译好的 wheel)
WHEEL_FILE=$(ls -1 "$LOCAL_DIR/crypto/target/wheels/"*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL_FILE" ]; then
    echo "[2.5/4] 上传加密模块..."
    scp "$WHEEL_FILE" "$REMOTE_HOST:/tmp/"
    ssh $REMOTE_HOST "pip3 install /tmp/$(basename $WHEEL_FILE) --force-reinstall"
fi

# 重启服务
echo "[3/4] 重启 Gateway 服务..."
ssh $REMOTE_HOST "sudo systemctl restart gemini-gateway || echo 'Service not found, please start manually'"

# 检查状态
echo "[4/4] 检查服务状态..."
sleep 2
ssh $REMOTE_HOST "curl -s http://127.0.0.1:8080/health | head -5 || echo 'Health check failed'"

echo ""
echo "=================================================="
echo "部署完成!"
echo ""
echo "访问:"
echo "  - 网页: https://lingkong.xyz"
echo "  - API:  https://lingkong.xyz/v1beta/models/gemini-3-pro-preview:generateContent"
echo "=================================================="

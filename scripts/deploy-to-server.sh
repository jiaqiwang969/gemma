#!/bin/bash
# =============================================================================
# 灵空 AI - 服务器部署脚本
# =============================================================================
# 在本地运行此脚本，将更新推送到服务器
# 使用方法: ./deploy-to-server.sh
# =============================================================================

set -e

SERVER="ubuntu@115.159.223.227"
REMOTE_DIR="/var/www/html"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🐉 灵空 AI - 部署更新到服务器"
echo ""

# 测试连接
echo "▶ 测试服务器连接..."
if ! ssh -o ConnectTimeout=10 $SERVER "echo 'connected'" 2>/dev/null; then
    echo "❌ 无法连接服务器，请检查 SSH 配置"
    exit 1
fi

echo "✓ 连接成功"
echo ""

# 打包 WebUI
echo "▶ 打包 WebUI..."
cd "$PROJECT_DIR"
tar -czf /tmp/webui.tar.gz -C apps/webui .
echo "✓ WebUI 已打包 ($(du -h /tmp/webui.tar.gz | cut -f1))"

# 上传文件到临时目录
echo "▶ 上传安装脚本..."
scp scripts/quick-install.sh $SERVER:/tmp/install.sh
echo "✓ install.sh 已上传"

echo "▶ 上传 WebUI 包..."
scp /tmp/webui.tar.gz $SERVER:/tmp/webui.tar.gz
echo "✓ webui.tar.gz 已上传"

echo "▶ 上传首页..."
scp apps/webui/static/home.html $SERVER:/tmp/home.html
echo "✓ home.html 已上传"

echo "▶ 上传聊天界面..."
scp apps/webui/static/index.html $SERVER:/tmp/index.html
scp apps/webui/static/chat.html $SERVER:/tmp/chat.html
scp apps/webui/static/chat-lite.html $SERVER:/tmp/chat-lite.html 2>/dev/null || true
echo "✓ chat 界面已上传"

echo "▶ 上传 API 文档..."
scp apps/webui/static/docs.html $SERVER:/tmp/docs.html
echo "✓ docs.html 已上传"

echo "▶ 上传 Playground..."
scp apps/webui/static/playground.html $SERVER:/tmp/playground.html 2>/dev/null || true
echo "✓ playground.html 已上传"

echo "▶ 上传商业计划书..."
scp apps/webui/static/pitch.html $SERVER:/tmp/pitch.html 2>/dev/null || true
scp apps/webui/static/pitch.pdf $SERVER:/tmp/pitch.pdf 2>/dev/null || true
echo "✓ pitch 文件已上传"

echo "▶ 上传商业计划书图片..."
ssh $SERVER "mkdir -p /tmp/pitch"
scp apps/webui/static/pitch/*.jpg $SERVER:/tmp/pitch/ 2>/dev/null || true
echo "✓ pitch 图片已上传"

echo "▶ 上传灵空聊天界面..."
scp apps/webui/static/lingkong.html $SERVER:/tmp/lingkong.html 2>/dev/null || true
echo "✓ lingkong.html 已上传"

# 移动文件并设置权限
echo "▶ 部署文件..."
ssh $SERVER "sudo mkdir -p $REMOTE_DIR/static/pitch && \
    sudo mv /tmp/install.sh $REMOTE_DIR/install.sh && \
    sudo mv /tmp/webui.tar.gz $REMOTE_DIR/webui.tar.gz && \
    sudo mv /tmp/home.html $REMOTE_DIR/static/home.html && \
    sudo mv /tmp/index.html $REMOTE_DIR/static/index.html 2>/dev/null || true && \
    sudo mv /tmp/chat.html $REMOTE_DIR/static/chat.html 2>/dev/null || true && \
    sudo mv /tmp/lingkong.html $REMOTE_DIR/static/lingkong.html 2>/dev/null || true && \
    sudo mv /tmp/chat-lite.html $REMOTE_DIR/static/chat-lite.html 2>/dev/null || true && \
    sudo mv /tmp/docs.html $REMOTE_DIR/static/docs.html 2>/dev/null || true && \
    sudo mv /tmp/playground.html $REMOTE_DIR/static/playground.html 2>/dev/null || true && \
    sudo mv /tmp/pitch.html $REMOTE_DIR/static/pitch.html 2>/dev/null || true && \
    sudo mv /tmp/pitch.pdf $REMOTE_DIR/static/pitch.pdf 2>/dev/null || true && \
    sudo mv /tmp/pitch/*.jpg $REMOTE_DIR/static/pitch/ 2>/dev/null || true && \
    sudo rmdir /tmp/pitch 2>/dev/null || true && \
    sudo chmod 755 $REMOTE_DIR/install.sh && \
    sudo chmod 644 $REMOTE_DIR/webui.tar.gz && \
    sudo chmod 644 $REMOTE_DIR/static/*.html 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/*.pdf 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/pitch/*.jpg 2>/dev/null || true && \
    sudo chown -R ubuntu:ubuntu $REMOTE_DIR/static $REMOTE_DIR/install.sh $REMOTE_DIR/webui.tar.gz"
echo "✓ 文件已部署"

# 清理本地临时文件
rm -f /tmp/webui.tar.gz

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ 部署完成!"
echo ""
echo "  🌐 首页: http://115.159.223.227/static/home.html"
echo "  📦 安装脚本: http://115.159.223.227/install.sh"
echo "  📦 WebUI包: http://115.159.223.227/webui.tar.gz"
echo "  📚 文档: http://115.159.223.227/static/docs.html"
echo "═══════════════════════════════════════════════════════════"

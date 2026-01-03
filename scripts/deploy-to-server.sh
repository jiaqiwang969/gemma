#!/bin/bash
# =============================================================================
# 灵空 AI - 服务器部署脚本
# =============================================================================
# 在本地运行此脚本，将更新推送到服务器
# 使用方法: ./deploy-to-server.sh
# =============================================================================

SERVER="root@115.159.223.227"
REMOTE_DIR="/var/www/lingkong/public"

echo "🐉 灵空 AI - 部署更新到服务器"
echo ""

# 测试连接
echo "▶ 测试服务器连接..."
if ! ssh -o ConnectTimeout=10 $SERVER "echo 'connected'" 2>/dev/null; then
    echo "❌ 无法连接服务器，请检查 SSH 配置"
    echo "   你可能需要手动运行以下命令:"
    echo ""
    echo "   scp scripts/quick-install.sh $SERVER:$REMOTE_DIR/install.sh"
    echo "   scp apps/webui/static/home.html $SERVER:$REMOTE_DIR/static/"
    echo "   scp apps/webui/static/docs.html $SERVER:$REMOTE_DIR/static/"
    echo "   scp apps/webui/static/playground.html $SERVER:$REMOTE_DIR/static/"
    exit 1
fi

echo "✓ 连接成功"
echo ""

# 上传文件
echo "▶ 上传安装脚本..."
scp scripts/quick-install.sh $SERVER:$REMOTE_DIR/install.sh
echo "✓ install.sh 已上传"

echo "▶ 上传首页..."
scp apps/webui/static/home.html $SERVER:$REMOTE_DIR/static/
echo "✓ home.html 已上传"

echo "▶ 上传 API 文档..."
scp apps/webui/static/docs.html $SERVER:$REMOTE_DIR/static/
echo "✓ docs.html 已上传"

echo "▶ 上传 Playground..."
scp apps/webui/static/playground.html $SERVER:$REMOTE_DIR/static/
echo "✓ playground.html 已上传"

# 设置权限
echo "▶ 设置文件权限..."
ssh $SERVER "chmod +x $REMOTE_DIR/install.sh && chmod 644 $REMOTE_DIR/static/*.html"
echo "✓ 权限已设置"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ 部署完成!"
echo ""
echo "  🌐 首页: https://lingkong.xyz"
echo "  📦 安装脚本: https://lingkong.xyz/install.sh"
echo "  📚 文档: https://lingkong.xyz/static/docs.html"
echo "═══════════════════════════════════════════════════════════"

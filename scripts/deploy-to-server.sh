#!/bin/bash
# =============================================================================
# 灵空 AI - 服务器部署脚本
# =============================================================================
# 在本地运行此脚本，将更新推送到服务器
# 使用方法:
#   ./deploy-to-server.sh           # 部署代码和页面
#   ./deploy-to-server.sh models    # 部署代码 + 上传模型文件
# =============================================================================

set -e

SERVER="ubuntu@115.159.223.227"
REMOTE_DIR="/var/www/html"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$HOME/.lingkong/models"

DEPLOY_MODELS="${1:-}"

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

# 打包 Gemini API
echo "▶ 打包 Gemini API..."
cd "$PROJECT_DIR/apps/gemini_api"
tar -czf /tmp/gemini_api.tar.gz \
  --exclude='crypto/target' \
  --exclude='__pycache__' \
  --exclude='.server_keys.json' \
  server.py gateway.py crypto_api.py README.md THOUGHT_SIGNATURE.md static deploy
echo "✓ Gemini API 已打包 ($(du -h /tmp/gemini_api.tar.gz | cut -f1))"

# 上传文件到临时目录
echo "▶ 上传安装脚本..."
cd "$PROJECT_DIR"
scp scripts/quick-install.sh $SERVER:/tmp/install.sh
echo "✓ install.sh 已上传"

echo "▶ 上传 WebUI 包..."
scp /tmp/webui.tar.gz $SERVER:/tmp/webui.tar.gz
echo "✓ webui.tar.gz 已上传"

echo "▶ 上传 Gemini API 包..."
scp /tmp/gemini_api.tar.gz $SERVER:/tmp/gemini_api.tar.gz
echo "✓ gemini_api.tar.gz 已上传"

echo "▶ 上传首页..."
scp apps/webui/static/home.html $SERVER:/tmp/home.html
echo "✓ home.html 已上传"

echo "▶ 上传聊天界面..."
scp apps/webui/static/index.html $SERVER:/tmp/index.html
scp apps/webui/static/chat.html $SERVER:/tmp/chat.html
scp apps/webui/static/chat-lite.html $SERVER:/tmp/chat-lite.html 2>/dev/null || true
echo "✓ chat 界面已上传"

echo "▶ 上传下载页面..."
scp apps/webui/static/downloads.html $SERVER:/tmp/downloads.html
echo "✓ downloads.html 已上传"

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
ssh $SERVER "sudo mkdir -p $REMOTE_DIR/static/pitch $REMOTE_DIR/models && \
    sudo mv /tmp/install.sh $REMOTE_DIR/install.sh && \
    sudo mv /tmp/webui.tar.gz $REMOTE_DIR/webui.tar.gz && \
    sudo mv /tmp/gemini_api.tar.gz $REMOTE_DIR/gemini_api.tar.gz && \
    sudo mv /tmp/home.html $REMOTE_DIR/static/home.html && \
    sudo mv /tmp/index.html $REMOTE_DIR/static/index.html 2>/dev/null || true && \
    sudo mv /tmp/chat.html $REMOTE_DIR/static/chat.html 2>/dev/null || true && \
    sudo mv /tmp/downloads.html $REMOTE_DIR/static/downloads.html 2>/dev/null || true && \
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
    sudo chmod 644 $REMOTE_DIR/gemini_api.tar.gz && \
    sudo chmod 644 $REMOTE_DIR/static/*.html 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/*.pdf 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/pitch/*.jpg 2>/dev/null || true && \
    sudo chown -R ubuntu:ubuntu $REMOTE_DIR/static $REMOTE_DIR/install.sh $REMOTE_DIR/webui.tar.gz $REMOTE_DIR/gemini_api.tar.gz"
echo "✓ 文件已部署"

# 清理本地临时文件
rm -f /tmp/webui.tar.gz /tmp/gemini_api.tar.gz

# 生成校验和文件 (用于客户端检测更新)
echo "▶ 生成校验和文件..."
ssh $SERVER "cd $REMOTE_DIR && sha256sum install.sh webui.tar.gz gemini_api.tar.gz bin/llama-lingkong-macos-arm64.tar.gz 2>/dev/null | sudo tee checksums.sha256 > /dev/null && sudo chmod 644 checksums.sha256"
echo "✓ checksums.sha256 已生成"

# 上传模型文件 (可选)
if [[ "$DEPLOY_MODELS" == "models" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  上传模型文件到服务器 (4.5GB，需要较长时间)"
    echo "════════════════════════════════════════════════════════════"

    ssh $SERVER "sudo mkdir -p $REMOTE_DIR/models && sudo chown ubuntu:ubuntu $REMOTE_DIR/models"

    for model in gemma-3n-E2B-it-Q4_K_M.gguf gemma-3n-vision-mmproj-f16.gguf gemma-3n-audio-mmproj-f16.gguf; do
        if [[ -f "$MODELS_DIR/$model" ]]; then
            size=$(du -h "$MODELS_DIR/$model" | cut -f1)
            echo "▶ 上传 $model ($size)..."
            scp "$MODELS_DIR/$model" "$SERVER:/tmp/$model"
            ssh $SERVER "sudo mv /tmp/$model $REMOTE_DIR/models/$model && sudo chmod 644 $REMOTE_DIR/models/$model"
            echo "✓ $model 已上传"
        else
            echo "⚠ $MODELS_DIR/$model 不存在，跳过"
        fi
    done

    echo ""
    echo "✓ 模型文件上传完成"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ 部署完成!"
echo ""
echo "  🌐 首页: http://115.159.223.227/static/home.html"
echo "  📦 安装脚本: http://115.159.223.227/install.sh"
echo "  📦 WebUI包: http://115.159.223.227/webui.tar.gz"
echo "  📦 Gemini API包: http://115.159.223.227/gemini_api.tar.gz"
echo "  📥 下载中心: http://115.159.223.227/static/downloads.html"
echo "  📚 文档: http://115.159.223.227/static/docs.html"
if [[ "$DEPLOY_MODELS" == "models" ]]; then
echo "  🤖 模型文件: http://115.159.223.227/models/"
fi
echo "═══════════════════════════════════════════════════════════"

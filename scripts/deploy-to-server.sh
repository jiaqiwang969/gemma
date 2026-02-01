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

SERVER="${DEPLOY_SERVER:-ubuntu@115.159.223.227}"
REMOTE_DIR="${DEPLOY_REMOTE_DIR:-/var/www/html}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$HOME/.lingkong/models"

DEPLOY_MODELS="${1:-}"
DEPLOY_OPENCLAW="${DEPLOY_OPENCLAW:-1}"  # set to 0 to skip OpenClaw bundle upload
DEPLOY_WHISPER="${DEPLOY_WHISPER:-1}"   # set to 0 to skip whisper-cli bundle upload

echo "🐉 灵空 AI - 部署更新到服务器"
echo ""

# Ensure we can connect non-interactively (avoid host key prompts).
ensure_known_host() {
    local host="${SERVER#*@}"
    mkdir -p "$HOME/.ssh"
    if ! ssh-keygen -F "$host" >/dev/null 2>&1; then
        echo "▶ 添加 $host 到 known_hosts..."
        ssh-keyscan -H "$host" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
    fi
}

# Build OpenClaw bundle locally (macOS arm64) so the website installer can fetch it.
OPENCLAW_STABLE_TARBALL="/tmp/openclaw-macos-arm64.tar.gz"
WHISPER_STABLE_TARBALL="/tmp/whisper-cli-macos-arm64.tar.gz"
NODE_VERSION_DEFAULT="${NODE_VERSION_DEFAULT:-22.20.0}"
NODE_STABLE_TARBALL="/tmp/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz"

maybe_build_openclaw_bundle() {
    if [[ "$DEPLOY_OPENCLAW" != "1" ]]; then
        echo "▶ 跳过 OpenClaw bundle (DEPLOY_OPENCLAW=0)"
        return 0
    fi

    local platform
    platform="$(uname -s)-$(uname -m)"
    if [[ "$platform" != "Darwin-arm64" ]]; then
        echo "⚠ OpenClaw bundle 目前仅能在 macOS arm64 构建 (当前: $platform)，跳过"
        return 0
    fi

    echo "▶ 构建 OpenClaw bundle..."
    cd "$PROJECT_DIR"
    ./scripts/build-openclaw-bundle.sh
    local latest
    latest="$(ls -t artifacts/bundles/openclaw-macos-arm64-*.tar.gz 2>/dev/null | head -1)"
    if [[ -z "$latest" ]]; then
        echo "⚠ 未找到 OpenClaw bundle 产物 (artifacts/bundles/openclaw-macos-arm64-*.tar.gz)，跳过"
        return 0
    fi
    cp "$latest" "$OPENCLAW_STABLE_TARBALL"
    echo "✓ OpenClaw bundle 已构建: $OPENCLAW_STABLE_TARBALL ($(du -h "$OPENCLAW_STABLE_TARBALL" | cut -f1))"
}

maybe_build_whisper_bundle() {
    if [[ "$DEPLOY_WHISPER" != "1" ]]; then
        echo "▶ 跳过 whisper-cli bundle (DEPLOY_WHISPER=0)"
        return 0
    fi

    local platform
    platform="$(uname -s)-$(uname -m)"
    if [[ "$platform" != "Darwin-arm64" ]]; then
        echo "⚠ whisper-cli bundle 目前仅能在 macOS arm64 构建 (当前: $platform)，跳过"
        return 0
    fi

    echo "▶ 构建 whisper-cli bundle..."
    cd "$PROJECT_DIR"
    ./scripts/build-whisper-cli-bundle.sh
    local latest
    latest="$(ls -t artifacts/bundles/whisper-cli-macos-arm64-*.tar.gz 2>/dev/null | head -1)"
    if [[ -z "$latest" ]]; then
        echo "⚠ 未找到 whisper-cli bundle 产物 (artifacts/bundles/whisper-cli-macos-arm64-*.tar.gz)，跳过"
        return 0
    fi
    cp "$latest" "$WHISPER_STABLE_TARBALL"
    echo "✓ whisper-cli bundle 已构建: $WHISPER_STABLE_TARBALL ($(du -h "$WHISPER_STABLE_TARBALL" | cut -f1))"
}

maybe_download_node_runtime() {
    # Only needed when the installer runs on a machine without system node.
    if [[ "$DEPLOY_OPENCLAW" != "1" ]]; then
        return 0
    fi

    if [[ -f "$NODE_STABLE_TARBALL" ]]; then
        return 0
    fi

    echo "▶ 下载 Node.js runtime (macOS arm64)..."
    local url="https://nodejs.org/dist/v${NODE_VERSION_DEFAULT}/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz"
    curl -fL --progress-bar "$url" -o "$NODE_STABLE_TARBALL"
    echo "✓ Node runtime 已下载: $NODE_STABLE_TARBALL ($(du -h "$NODE_STABLE_TARBALL" | cut -f1))"
}

# 测试连接
echo "▶ 测试服务器连接..."
ensure_known_host
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

maybe_build_openclaw_bundle
maybe_build_whisper_bundle
maybe_download_node_runtime

if [[ "$DEPLOY_OPENCLAW" == "1" && -f "$OPENCLAW_STABLE_TARBALL" ]]; then
  echo "▶ 上传 OpenClaw bundle..."
  scp "$OPENCLAW_STABLE_TARBALL" $SERVER:/tmp/openclaw-macos-arm64.tar.gz
  echo "✓ openclaw-macos-arm64.tar.gz 已上传"
fi

if [[ "$DEPLOY_WHISPER" == "1" && -f "$WHISPER_STABLE_TARBALL" ]]; then
  echo "▶ 上传 whisper-cli bundle..."
  scp "$WHISPER_STABLE_TARBALL" $SERVER:/tmp/whisper-cli-macos-arm64.tar.gz
  echo "✓ whisper-cli-macos-arm64.tar.gz 已上传"
fi

if [[ "$DEPLOY_OPENCLAW" == "1" && -f "$NODE_STABLE_TARBALL" ]]; then
  echo "▶ 上传 Node runtime..."
  scp "$NODE_STABLE_TARBALL" $SERVER:/tmp/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz
  echo "✓ node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz 已上传"
fi

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

echo "▶ 上传 i18n 国际化文件..."
ssh $SERVER "mkdir -p /tmp/i18n /tmp/js /tmp/tinybox"
scp apps/webui/static/i18n/*.json $SERVER:/tmp/i18n/ 2>/dev/null || true
scp apps/webui/static/js/*.js $SERVER:/tmp/js/ 2>/dev/null || true
echo "✓ i18n 文件已上传"

echo "▶ 上传 TinyBox DIY 指南..."
scp -r apps/webui/static/tinybox/* $SERVER:/tmp/tinybox/ 2>/dev/null || true
echo "✓ tinybox 已上传"

echo "▶ 上传 Playground..."
ssh $SERVER "mkdir -p /tmp/playground"
scp -r apps/webui/static/playground/* $SERVER:/tmp/playground/ 2>/dev/null || true
echo "✓ playground 已上传"

echo "▶ 上传进化系统..."
ssh $SERVER "mkdir -p /tmp/evolution"
scp -r apps/webui/static/evolution/* $SERVER:/tmp/evolution/ 2>/dev/null || true
echo "✓ evolution 已上传"

echo "▶ 上传加密策略..."
ssh $SERVER "mkdir -p /tmp/encryption"
scp -r apps/webui/static/encryption/* $SERVER:/tmp/encryption/ 2>/dev/null || true
echo "✓ encryption 已上传"

echo "▶ 上传灵空聊天界面..."
scp apps/webui/static/lingkong.html $SERVER:/tmp/lingkong.html 2>/dev/null || true
echo "✓ lingkong.html 已上传"

# 移动文件并设置权限
echo "▶ 部署文件..."
ssh $SERVER "sudo mkdir -p $REMOTE_DIR/bin $REMOTE_DIR/static/pitch $REMOTE_DIR/static/i18n $REMOTE_DIR/static/js $REMOTE_DIR/static/tinybox $REMOTE_DIR/static/playground $REMOTE_DIR/static/evolution $REMOTE_DIR/static/encryption $REMOTE_DIR/models && \
    sudo mv /tmp/install.sh $REMOTE_DIR/install.sh && \
    sudo mv /tmp/webui.tar.gz $REMOTE_DIR/webui.tar.gz && \
    sudo mv /tmp/gemini_api.tar.gz $REMOTE_DIR/gemini_api.tar.gz && \
    (test -f /tmp/openclaw-macos-arm64.tar.gz && sudo mv /tmp/openclaw-macos-arm64.tar.gz $REMOTE_DIR/bin/openclaw-macos-arm64.tar.gz || true) && \
    (test -f /tmp/whisper-cli-macos-arm64.tar.gz && sudo mv /tmp/whisper-cli-macos-arm64.tar.gz $REMOTE_DIR/bin/whisper-cli-macos-arm64.tar.gz || true) && \
    (test -f /tmp/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz && sudo mv /tmp/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz $REMOTE_DIR/bin/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz || true) && \
    sudo mv /tmp/home.html $REMOTE_DIR/static/home.html && \
    sudo mv /tmp/index.html $REMOTE_DIR/static/index.html 2>/dev/null || true && \
    sudo mv /tmp/chat.html $REMOTE_DIR/static/chat.html 2>/dev/null || true && \
    sudo mv /tmp/downloads.html $REMOTE_DIR/static/downloads.html 2>/dev/null || true && \
    sudo mv /tmp/lingkong.html $REMOTE_DIR/static/lingkong.html 2>/dev/null || true && \
    sudo mv /tmp/chat-lite.html $REMOTE_DIR/static/chat-lite.html 2>/dev/null || true && \
    sudo mv /tmp/docs.html $REMOTE_DIR/static/docs.html 2>/dev/null || true && \
    sudo mv /tmp/pitch.html $REMOTE_DIR/static/pitch.html 2>/dev/null || true && \
    sudo mv /tmp/pitch.pdf $REMOTE_DIR/static/pitch.pdf 2>/dev/null || true && \
    sudo mv /tmp/pitch/*.jpg $REMOTE_DIR/static/pitch/ 2>/dev/null || true && \
    sudo cp -r /tmp/tinybox/* $REMOTE_DIR/static/tinybox/ 2>/dev/null || true && \
    sudo cp -r /tmp/playground/* $REMOTE_DIR/static/playground/ 2>/dev/null || true && \
    sudo cp -r /tmp/evolution/* $REMOTE_DIR/static/evolution/ 2>/dev/null || true && \
    sudo cp -r /tmp/encryption/* $REMOTE_DIR/static/encryption/ 2>/dev/null || true && \
    sudo mv /tmp/i18n/*.json $REMOTE_DIR/static/i18n/ 2>/dev/null || true && \
    sudo mv /tmp/js/*.js $REMOTE_DIR/static/js/ 2>/dev/null || true && \
    sudo rm -rf /tmp/pitch /tmp/i18n /tmp/js /tmp/tinybox /tmp/evolution /tmp/encryption 2>/dev/null || true && \
    sudo chmod 755 $REMOTE_DIR/install.sh && \
    sudo chmod 644 $REMOTE_DIR/webui.tar.gz && \
    sudo chmod 644 $REMOTE_DIR/gemini_api.tar.gz && \
    sudo chmod 644 $REMOTE_DIR/static/*.html 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/*.pdf 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/pitch/*.jpg 2>/dev/null || true && \
    sudo chmod -R 644 $REMOTE_DIR/static/tinybox/* 2>/dev/null || true && \
    sudo chmod 755 $REMOTE_DIR/static/tinybox $REMOTE_DIR/static/tinybox/assets $REMOTE_DIR/static/tinybox/images 2>/dev/null || true && \
    sudo chmod -R 644 $REMOTE_DIR/static/evolution/* 2>/dev/null || true && \
    sudo chmod 755 $REMOTE_DIR/static/evolution 2>/dev/null || true && \
    sudo chmod -R 644 $REMOTE_DIR/static/encryption/* 2>/dev/null || true && \
    sudo chmod 755 $REMOTE_DIR/static/encryption 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/i18n/*.json 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/static/js/*.js 2>/dev/null || true && \
    sudo chmod 644 $REMOTE_DIR/bin/*.tar.gz 2>/dev/null || true && \
    sudo chown -R ubuntu:ubuntu $REMOTE_DIR/static $REMOTE_DIR/bin $REMOTE_DIR/install.sh $REMOTE_DIR/webui.tar.gz $REMOTE_DIR/gemini_api.tar.gz"
echo "✓ 文件已部署"

# 同步静态文件到 WebUI 目录 (nginx 从这里读取)
echo "▶ 同步静态文件到 WebUI..."
WEBUI_STATIC="/opt/lingkong-webui/static"
ssh $SERVER "sudo mkdir -p $WEBUI_STATIC/i18n $WEBUI_STATIC/js $WEBUI_STATIC/tinybox/js $WEBUI_STATIC/tinybox/assets $WEBUI_STATIC/tinybox/images $WEBUI_STATIC/playground $WEBUI_STATIC/evolution $WEBUI_STATIC/encryption && \
    sudo cp $REMOTE_DIR/static/home.html $WEBUI_STATIC/ 2>/dev/null || true && \
    sudo cp $REMOTE_DIR/static/downloads.html $WEBUI_STATIC/ 2>/dev/null || true && \
    sudo cp $REMOTE_DIR/static/docs.html $WEBUI_STATIC/ 2>/dev/null || true && \
    sudo cp $REMOTE_DIR/static/pitch.html $WEBUI_STATIC/ 2>/dev/null || true && \
    sudo cp -r $REMOTE_DIR/static/tinybox/* $WEBUI_STATIC/tinybox/ 2>/dev/null || true && \
    sudo cp -r $REMOTE_DIR/static/playground/* $WEBUI_STATIC/playground/ 2>/dev/null || true && \
    sudo cp -r $REMOTE_DIR/static/evolution/* $WEBUI_STATIC/evolution/ 2>/dev/null || true && \
    sudo cp -r $REMOTE_DIR/static/encryption/* $WEBUI_STATIC/encryption/ 2>/dev/null || true && \
    sudo cp $REMOTE_DIR/static/i18n/*.json $WEBUI_STATIC/i18n/ 2>/dev/null || true && \
    sudo cp $REMOTE_DIR/static/js/*.js $WEBUI_STATIC/js/ 2>/dev/null || true && \
    sudo chmod 755 $WEBUI_STATIC $WEBUI_STATIC/i18n $WEBUI_STATIC/js $WEBUI_STATIC/tinybox $WEBUI_STATIC/tinybox/js $WEBUI_STATIC/tinybox/assets $WEBUI_STATIC/tinybox/images $WEBUI_STATIC/playground $WEBUI_STATIC/evolution $WEBUI_STATIC/encryption && \
    sudo chmod 644 $WEBUI_STATIC/*.html $WEBUI_STATIC/i18n/*.json $WEBUI_STATIC/js/*.js 2>/dev/null || true && \
    sudo chmod 644 $WEBUI_STATIC/tinybox/*.html $WEBUI_STATIC/tinybox/js/*.js $WEBUI_STATIC/tinybox/assets/* 2>/dev/null || true && \
    sudo chmod 644 $WEBUI_STATIC/playground/*.html 2>/dev/null || true && \
    sudo chmod 644 $WEBUI_STATIC/evolution/*.html $WEBUI_STATIC/evolution/*.png $WEBUI_STATIC/evolution/*.pdf 2>/dev/null || true && \
    sudo chmod 644 $WEBUI_STATIC/encryption/*.html $WEBUI_STATIC/encryption/*.png 2>/dev/null || true && \
    sudo chown -R ubuntu:ubuntu $WEBUI_STATIC"
echo "✓ WebUI 静态文件已同步"

# 清理本地临时文件
rm -f /tmp/webui.tar.gz /tmp/gemini_api.tar.gz

# 生成校验和文件 (用于客户端检测更新)
echo "▶ 生成校验和文件..."
ssh $SERVER "cd $REMOTE_DIR && sha256sum install.sh webui.tar.gz gemini_api.tar.gz bin/llama-lingkong-macos-arm64.tar.gz bin/openclaw-macos-arm64.tar.gz bin/whisper-cli-macos-arm64.tar.gz bin/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz 2>/dev/null | sudo tee checksums.sha256 > /dev/null && sudo chmod 644 checksums.sha256"
echo "✓ checksums.sha256 已生成"

# 上传模型文件 (可选)
if [[ "$DEPLOY_MODELS" == "models" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  上传模型文件到服务器 (4.5GB，需要较长时间)"
    echo "════════════════════════════════════════════════════════════"

    ssh $SERVER "sudo mkdir -p $REMOTE_DIR/models $REMOTE_DIR/models/whisper && sudo chown -R ubuntu:ubuntu $REMOTE_DIR/models"

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

    # Whisper STT model (optional, ~465MB)
    if [[ -f "$MODELS_DIR/whisper/ggml-small.bin" ]]; then
        size=$(du -h "$MODELS_DIR/whisper/ggml-small.bin" | cut -f1)
        echo "▶ 上传 whisper/ggml-small.bin ($size)..."
        scp "$MODELS_DIR/whisper/ggml-small.bin" "$SERVER:/tmp/ggml-small.bin"
        ssh $SERVER "sudo mv /tmp/ggml-small.bin $REMOTE_DIR/models/whisper/ggml-small.bin && sudo chmod 644 $REMOTE_DIR/models/whisper/ggml-small.bin"
        echo "✓ whisper/ggml-small.bin 已上传"
    else
        echo "⚠ $MODELS_DIR/whisper/ggml-small.bin 不存在，跳过"
    fi

    echo ""
    echo "✓ 模型文件上传完成"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ 部署完成!"
echo ""
echo "  🌐 首页: http://lingkong.xyz/static/home.html"
echo "  📦 安装脚本: http://lingkong.xyz/install.sh"
echo "  📦 WebUI包: http://lingkong.xyz/webui.tar.gz"
echo "  📦 Gemini API包: http://lingkong.xyz/gemini_api.tar.gz"
echo "  📥 下载中心: http://lingkong.xyz/static/downloads.html"
echo "  📚 文档: http://lingkong.xyz/static/docs.html"
if [[ "$DEPLOY_MODELS" == "models" ]]; then
echo "  🤖 模型文件: http://lingkong.xyz/models/"
fi
echo "═══════════════════════════════════════════════════════════"

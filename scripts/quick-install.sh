#!/bin/bash
# ============================================================================
# LingKong AI - Quick Installer (使用 llama.cpp)
# ============================================================================
# 你的 AI. 你的数据. 你的掌控.
#
# 使用方法:
#   curl -fsSL https://lingkong.xyz/install.sh | bash
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# 配置
INSTALL_DIR="${LINGKONG_HOME:-$HOME/.lingkong}"
MODELS_DIR="${INSTALL_DIR}/models"
BIN_DIR="${INSTALL_DIR}/bin"
HF_BASE_URL="https://huggingface.co/jiaqiwang969/gemma3n-gguf/resolve/main"

# 模型列表
declare -A MODELS
MODELS["text"]="gemma-3n-E2B-it-Q4_K_M.gguf|2.8GB|主文本模型 (推荐)"
MODELS["vision"]="gemma-3n-vision-mmproj-f16.gguf|600MB|视觉理解模块"
MODELS["audio"]="gemma-3n-audio-mmproj-f16.gguf|1.4GB|音频理解模块"

# ============================================================================
print_banner() {
    echo ""
    echo -e "${PURPLE}${BOLD}"
    echo "  ╔═══════════════════════════════════════════════════════════╗"
    echo "  ║                                                           ║"
    echo "  ║   🐉  灵空 AI  -  LingKong AI                             ║"
    echo "  ║                                                           ║"
    echo "  ║   你的 AI. 你的数据. 你的掌控.                            ║"
    echo "  ║                                                           ║"
    echo "  ╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step() { echo -e "\n${CYAN}${BOLD}▶ $1${NC}"; }

# 检测系统
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        darwin) OS="macos" ;;
        linux) OS="linux" ;;
        *) log_error "不支持的操作系统: $os"; exit 1 ;;
    esac

    case "$arch" in
        x86_64|amd64) ARCH="x86_64" ;;
        arm64|aarch64) ARCH="arm64" ;;
        *) log_error "不支持的架构: $arch"; exit 1 ;;
    esac

    PLATFORM="${OS}-${ARCH}"
    log_info "检测到系统: $PLATFORM"
}

# 检查依赖
check_dependencies() {
    log_step "检查系统依赖"

    # 检查下载工具
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -fSL --progress-bar"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget --show-progress -qO-"
    else
        log_error "需要 curl 或 wget"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 创建目录
create_directories() {
    log_step "创建安装目录"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$BIN_DIR"
    mkdir -p "$INSTALL_DIR/config"
    log_success "目录创建完成: $INSTALL_DIR"
}

# 安装 llama.cpp (使用 Homebrew 或预编译)
install_llama_cpp() {
    log_step "安装 llama.cpp 推理引擎"

    if command -v llama-server &> /dev/null; then
        log_info "llama.cpp 已安装"
        return 0
    fi

    if [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            log_info "使用 Homebrew 安装 llama.cpp..."
            brew install llama.cpp
            log_success "llama.cpp 安装完成"
            return 0
        fi
    fi

    # 下载预编译版本
    log_info "下载预编译的 llama.cpp..."
    local llama_url="https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-${PLATFORM}.zip"

    if [[ "$PLATFORM" == "macos-arm64" ]]; then
        llama_url="https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-macos-arm64.zip"
    elif [[ "$PLATFORM" == "macos-x86_64" ]]; then
        llama_url="https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-macos-x64.zip"
    elif [[ "$PLATFORM" == "linux-x86_64" ]]; then
        llama_url="https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-ubuntu-x64.zip"
    fi

    local tmp_dir=$(mktemp -d)
    $DOWNLOAD_CMD "$llama_url" -o "$tmp_dir/llama.zip" || {
        log_warning "无法下载预编译版本，请手动安装 llama.cpp"
        log_info "macOS: brew install llama.cpp"
        log_info "Linux: 从 https://github.com/ggerganov/llama.cpp 下载"
        return 1
    }

    unzip -q "$tmp_dir/llama.zip" -d "$tmp_dir"
    cp "$tmp_dir"/*/llama-server "$BIN_DIR/" 2>/dev/null || cp "$tmp_dir"/llama-server "$BIN_DIR/" 2>/dev/null
    chmod +x "$BIN_DIR/llama-server"
    rm -rf "$tmp_dir"

    log_success "llama.cpp 安装完成"
}

# 下载模型
download_model() {
    local model_key="$1"
    local model_info="${MODELS[$model_key]}"
    local model_file=$(echo "$model_info" | cut -d'|' -f1)
    local model_size=$(echo "$model_info" | cut -d'|' -f2)
    local model_desc=$(echo "$model_info" | cut -d'|' -f3)

    local model_path="${MODELS_DIR}/${model_file}"

    if [[ -f "$model_path" ]]; then
        log_info "模型已存在: $model_file"
        return 0
    fi

    log_step "下载模型: $model_desc ($model_size)"
    local model_url="${HF_BASE_URL}/${model_file}"

    log_info "下载地址: $model_url"
    log_warning "文件较大 ($model_size)，请耐心等待..."

    $DOWNLOAD_CMD "$model_url" -o "$model_path" || {
        log_error "下载失败: $model_file"
        return 1
    }

    log_success "模型下载完成: $model_file"
}

# 创建启动脚本
create_start_script() {
    log_step "创建启动脚本"

    local start_script="${BIN_DIR}/lingkong-start"
    cat > "$start_script" << 'SCRIPT'
#!/bin/bash
# LingKong AI 启动脚本

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
MODEL="${1:-$LINGKONG_HOME/models/gemma-3n-E2B-it-Q4_K_M.gguf}"
PORT="${LINGKONG_PORT:-5001}"

# 检查模型
if [[ ! -f "$MODEL" ]]; then
    echo "错误: 模型文件不存在: $MODEL"
    echo "请先运行: lingkong-download"
    exit 1
fi

# 检查 llama-server
LLAMA_SERVER=""
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER="llama-server"
elif [[ -f "$LINGKONG_HOME/bin/llama-server" ]]; then
    LLAMA_SERVER="$LINGKONG_HOME/bin/llama-server"
else
    echo "错误: 找不到 llama-server"
    echo "请安装: brew install llama.cpp"
    exit 1
fi

echo "🐉 启动 灵空 AI..."
echo "   模型: $MODEL"
echo "   端口: $PORT"
echo ""
echo "访问地址: http://localhost:$PORT"
echo "按 Ctrl+C 停止服务"
echo ""

$LLAMA_SERVER \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --flash-attn
SCRIPT
    chmod +x "$start_script"

    # 创建下载脚本
    local download_script="${BIN_DIR}/lingkong-download"
    cat > "$download_script" << DLSCRIPT
#!/bin/bash
# LingKong AI 模型下载脚本

LINGKONG_HOME="\${LINGKONG_HOME:-\$HOME/.lingkong}"
HF_URL="https://huggingface.co/jiaqiwang969/gemma3n-gguf/resolve/main"

echo "🐉 灵空 AI 模型下载"
echo ""

download_model() {
    local name="\$1"
    local size="\$2"
    local path="\$LINGKONG_HOME/models/\$name"

    if [[ -f "\$path" ]]; then
        echo "✓ \$name 已存在"
        return 0
    fi

    echo "下载 \$name (\$size)..."
    curl -fSL --progress-bar "\$HF_URL/\$name" -o "\$path"
    echo "✓ \$name 下载完成"
}

mkdir -p "\$LINGKONG_HOME/models"

case "\${1:-text}" in
    text|main)
        download_model "gemma-3n-E2B-it-Q4_K_M.gguf" "2.8GB"
        ;;
    vision)
        download_model "gemma-3n-vision-mmproj-f16.gguf" "600MB"
        ;;
    audio)
        download_model "gemma-3n-audio-mmproj-f16.gguf" "1.4GB"
        ;;
    all)
        download_model "gemma-3n-E2B-it-Q4_K_M.gguf" "2.8GB"
        download_model "gemma-3n-vision-mmproj-f16.gguf" "600MB"
        download_model "gemma-3n-audio-mmproj-f16.gguf" "1.4GB"
        ;;
    *)
        echo "用法: lingkong-download [text|vision|audio|all]"
        ;;
esac

echo ""
echo "模型存放位置: \$LINGKONG_HOME/models/"
DLSCRIPT
    chmod +x "$download_script"

    log_success "启动脚本创建完成"
}

# 配置 PATH
setup_path() {
    log_step "配置环境变量"

    local path_export="export PATH=\"\$PATH:${BIN_DIR}\""
    local home_export="export LINGKONG_HOME=\"${INSTALL_DIR}\""
    local shell_rc=""

    if [[ -f "$HOME/.zshrc" ]]; then
        shell_rc="$HOME/.zshrc"
    elif [[ -f "$HOME/.bashrc" ]]; then
        shell_rc="$HOME/.bashrc"
    elif [[ -f "$HOME/.profile" ]]; then
        shell_rc="$HOME/.profile"
    fi

    if [[ -n "$shell_rc" ]]; then
        if ! grep -q "LINGKONG_HOME" "$shell_rc" 2>/dev/null; then
            echo "" >> "$shell_rc"
            echo "# LingKong AI" >> "$shell_rc"
            echo "$home_export" >> "$shell_rc"
            echo "$path_export" >> "$shell_rc"
            log_success "已添加到 $shell_rc"
        else
            log_info "环境变量已配置"
        fi
    fi

    export LINGKONG_HOME="$INSTALL_DIR"
    export PATH="$PATH:$BIN_DIR"
}

# 打印成功信息
print_success() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "  ╔═══════════════════════════════════════════════════════════╗"
    echo "  ║                                                           ║"
    echo "  ║   ✅  安装完成！                                          ║"
    echo "  ║                                                           ║"
    echo "  ╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "  ${BOLD}快速开始:${NC}"
    echo ""
    echo -e "    ${CYAN}1. 重新加载 shell:${NC}"
    echo -e "       source ~/.zshrc  ${YELLOW}# 或 ~/.bashrc${NC}"
    echo ""
    echo -e "    ${CYAN}2. 下载模型:${NC}"
    echo -e "       lingkong-download        # 下载主模型 (2.8GB)"
    echo -e "       lingkong-download all    # 下载全部模型 (5GB)"
    echo ""
    echo -e "    ${CYAN}3. 启动服务:${NC}"
    echo -e "       lingkong-start"
    echo ""
    echo -e "    ${CYAN}4. 访问:${NC}"
    echo -e "       http://localhost:5001"
    echo ""
    echo -e "  ${BOLD}文档:${NC} https://lingkong.xyz/docs"
    echo -e "  ${BOLD}GitHub:${NC} https://github.com/jiaqiwang969/gemma"
    echo ""
}

# 主程序
main() {
    print_banner
    detect_platform
    check_dependencies
    create_directories
    install_llama_cpp
    create_start_script
    setup_path

    # 询问是否下载模型
    echo ""
    read -p "是否现在下载主模型 (2.8GB)? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        download_model "text"
    fi

    print_success
}

main "$@"

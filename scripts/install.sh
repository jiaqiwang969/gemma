#!/bin/bash
# ============================================================================
# LingKong AI - One-Click Installer
# ============================================================================
# 你的 AI. 你的数据. 你的掌控.
#
# 使用方法:
#   curl -fsSL https://lingkong.xyz/install.sh | bash
#
# 或者指定版本:
#   curl -fsSL https://lingkong.xyz/install.sh | bash -s -- --version 0.1.0
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 配置
VERSION="${LINGKONG_VERSION:-latest}"
INSTALL_DIR="${LINGKONG_HOME:-$HOME/.lingkong}"
BIN_DIR="${INSTALL_DIR}/bin"
MODELS_DIR="${INSTALL_DIR}/models"
BASE_URL="${LINGKONG_MIRROR:-https://lingkong.xyz}"
HF_BASE_URL="https://huggingface.co/jiaqiwang969/gemma3n-gguf/resolve/main"

# 默认模型
DEFAULT_MODEL="gemma-3n-E2B-it-Q4_K_M.gguf"
DEFAULT_MODEL_SIZE="2.6GB"

# ============================================================================
# 辅助函数
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

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}▶ $1${NC}"
}

# 检测系统
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        darwin)
            OS="darwin"
            ;;
        linux)
            OS="linux"
            ;;
        mingw*|msys*|cygwin*)
            OS="windows"
            ;;
        *)
            log_error "不支持的操作系统: $os"
            exit 1
            ;;
    esac

    case "$arch" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            log_error "不支持的架构: $arch"
            exit 1
            ;;
    esac

    PLATFORM="${OS}-${ARCH}"
    log_info "检测到系统: $PLATFORM"
}

# 检查依赖
check_dependencies() {
    log_step "检查系统依赖"

    local missing=()

    # 检查 curl 或 wget
    if command -v curl &> /dev/null; then
        DOWNLOADER="curl"
        DOWNLOAD_CMD="curl -fsSL"
    elif command -v wget &> /dev/null; then
        DOWNLOADER="wget"
        DOWNLOAD_CMD="wget -qO-"
    else
        missing+=("curl 或 wget")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "缺少依赖: ${missing[*]}"
        log_info "请先安装: brew install curl (macOS) 或 apt install curl (Linux)"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 检查磁盘空间
check_disk_space() {
    log_step "检查磁盘空间"

    local required_gb=5  # 需要至少 5GB
    local available_gb

    if [[ "$OS" == "darwin" ]]; then
        available_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
    else
        available_gb=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    fi

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "磁盘空间不足: 需要 ${required_gb}GB, 可用 ${available_gb}GB"
        exit 1
    fi

    log_success "磁盘空间充足 (可用: ${available_gb}GB)"
}

# 创建目录
create_directories() {
    log_step "创建安装目录"

    mkdir -p "$BIN_DIR"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$INSTALL_DIR/config"
    mkdir -p "$INSTALL_DIR/logs"

    log_success "目录创建完成: $INSTALL_DIR"
}

# 下载 CLI
download_cli() {
    log_step "下载 LingKong CLI"

    local cli_url="${BASE_URL}/bin/lingkong-${PLATFORM}"
    local cli_path="${BIN_DIR}/lingkong"

    log_info "下载地址: $cli_url"

    if [[ "$DOWNLOADER" == "curl" ]]; then
        curl -fSL --progress-bar "$cli_url" -o "$cli_path"
    else
        wget --show-progress -qO "$cli_path" "$cli_url"
    fi

    chmod +x "$cli_path"

    # 验证
    if "$cli_path" --version &> /dev/null; then
        log_success "CLI 下载完成"
    else
        log_error "CLI 验证失败"
        exit 1
    fi
}

# 下载默认模型
download_model() {
    log_step "下载 AI 模型 ($DEFAULT_MODEL_SIZE)"

    local model_url="${HF_BASE_URL}/${DEFAULT_MODEL}"
    local model_path="${MODELS_DIR}/${DEFAULT_MODEL}"

    if [ -f "$model_path" ]; then
        log_info "模型已存在，跳过下载"
        return 0
    fi

    log_info "下载地址: $model_url"
    log_warning "模型较大 ($DEFAULT_MODEL_SIZE)，请耐心等待..."

    if [[ "$DOWNLOADER" == "curl" ]]; then
        curl -fSL --progress-bar "$model_url" -o "$model_path"
    else
        wget --show-progress -qO "$model_path" "$model_url"
    fi

    log_success "模型下载完成"
}

# 配置环境变量
setup_path() {
    log_step "配置环境变量"

    local shell_rc=""
    local path_export="export PATH=\"\$PATH:${BIN_DIR}\""
    local home_export="export LINGKONG_HOME=\"${INSTALL_DIR}\""

    # 检测 shell
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        shell_rc="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
        shell_rc="$HOME/.bashrc"
    elif [ -f "$HOME/.profile" ]; then
        shell_rc="$HOME/.profile"
    fi

    if [ -n "$shell_rc" ]; then
        # 检查是否已添加
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

    # 当前会话也设置
    export LINGKONG_HOME="$INSTALL_DIR"
    export PATH="$PATH:$BIN_DIR"
}

# 创建默认配置
create_config() {
    log_step "创建默认配置"

    local config_file="${INSTALL_DIR}/config/config.toml"

    if [ ! -f "$config_file" ]; then
        cat > "$config_file" << EOF
# LingKong AI 配置文件
# 文档: https://lingkong.xyz/docs

[server]
host = "127.0.0.1"
port = 5000

[model]
default = "${DEFAULT_MODEL}"
path = "${MODELS_DIR}"

[inference]
# 推理后端: llama.cpp (推荐) 或 pytorch
backend = "llama.cpp"
# 上下文长度
context_length = 8192
# GPU 层数 (Metal/CUDA)
gpu_layers = 99

[logging]
level = "info"
file = "${INSTALL_DIR}/logs/lingkong.log"
EOF
        log_success "配置文件创建完成"
    else
        log_info "配置文件已存在"
    fi
}

# 验证安装
verify_installation() {
    log_step "验证安装"

    local cli_path="${BIN_DIR}/lingkong"

    if [ ! -f "$cli_path" ]; then
        log_error "CLI 未找到"
        return 1
    fi

    if [ ! -f "${MODELS_DIR}/${DEFAULT_MODEL}" ]; then
        log_warning "模型文件未找到 (可稍后使用 lingkong model pull 下载)"
    fi

    # 运行 doctor 命令
    "$cli_path" doctor 2>/dev/null || true

    log_success "安装验证完成"
}

# 打印完成信息
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
    echo -e "    ${CYAN}1. 重新加载 shell 或运行:${NC}"
    echo -e "       source ~/.zshrc  ${YELLOW}# 或 ~/.bashrc${NC}"
    echo ""
    echo -e "    ${CYAN}2. 启动服务:${NC}"
    echo -e "       lingkong serve start"
    echo ""
    echo -e "    ${CYAN}3. 打开浏览器:${NC}"
    echo -e "       http://localhost:5000"
    echo ""
    echo -e "  ${BOLD}更多命令:${NC}"
    echo ""
    echo -e "    lingkong model list       # 查看可用模型"
    echo -e "    lingkong model pull <名称> # 下载其他模型"
    echo -e "    lingkong config edit      # 编辑配置"
    echo -e "    lingkong doctor           # 诊断问题"
    echo ""
    echo -e "  ${BOLD}文档:${NC} https://lingkong.xyz/docs"
    echo -e "  ${BOLD}GitHub:${NC} https://github.com/jiaqiwang969/gemma"
    echo ""
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --no-model)
                SKIP_MODEL=true
                shift
                ;;
            --help)
                echo "LingKong AI 安装脚本"
                echo ""
                echo "用法: install.sh [选项]"
                echo ""
                echo "选项:"
                echo "  --version <版本>  指定版本 (默认: latest)"
                echo "  --no-model        跳过模型下载"
                echo "  --help            显示帮助"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    parse_args "$@"

    print_banner

    detect_platform
    check_dependencies
    check_disk_space
    create_directories
    download_cli

    if [ "$SKIP_MODEL" != "true" ]; then
        download_model
    fi

    setup_path
    create_config
    verify_installation

    print_success
}

main "$@"

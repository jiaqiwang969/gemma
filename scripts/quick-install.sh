#!/bin/bash
# =============================================================================
# 灵空 AI - 一键安装脚本 (Sandbox 版)
# =============================================================================
# 使用方法:
#   curl -fsSL http://115.159.223.227/install.sh | bash           # 默认模式
#   curl -fsSL http://115.159.223.227/install.sh | bash -s sandbox # Sandbox 模式
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
BIN_DIR="$LINGKONG_HOME/bin"
LIB_DIR="$LINGKONG_HOME/lib"
MODELS_DIR="$LINGKONG_HOME/models"
SANDBOX_DIR="$LINGKONG_HOME/sandbox"

# 下载地址 (支持环境变量覆盖)
LINGKONG_SERVER="${LINGKONG_SERVER:-http://115.159.223.227}"
BASE_URL="$LINGKONG_SERVER"
BINARY_URL_MACOS="$BASE_URL/bin/llama-lingkong-macos-arm64.tar.gz"
BINARY_URL_LINUX="$BASE_URL/bin/llama-lingkong-linux-x86_64.tar.gz"
WEBUI_URL="$BASE_URL/webui.tar.gz"
SANDBOX_URL="$BASE_URL/sandbox.tar.gz"

# 模型下载地址 (优先国内镜像，备用 HuggingFace)
MODELS_BASE="$BASE_URL/models"
MODEL_URL="$MODELS_BASE/gemma-3n-E2B-it-Q4_K_M.gguf"
VISION_URL="$MODELS_BASE/gemma-3n-vision-mmproj-f16.gguf"
AUDIO_URL="$MODELS_BASE/gemma-3n-audio-mmproj-f16.gguf"

# HuggingFace 备用地址
HF_BASE="https://huggingface.co/nicepkg/gemma-3n-gguf/resolve/main"
MODEL_URL_HF="$HF_BASE/gemma-3n-E2B-it-Q4_K_M.gguf"
VISION_URL_HF="$HF_BASE/gemma-3n-vision-mmproj-f16.gguf"
AUDIO_URL_HF="$HF_BASE/gemma-3n-audio-mmproj-f16.gguf"

# Python 依赖
PYTHON_DEPS="flask flask-cors pillow psutil librosa soundfile requests"

# 安装模式
INSTALL_MODE="${1:-auto}"  # auto | native | sandbox

# 日志函数
log_info() { echo -e "${BLUE}[信息]${NC} $1"; }
log_success() { echo -e "${GREEN}[成功]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[警告]${NC} $1"; }
log_error() { echo -e "${RED}[错误]${NC} $1"; }
log_step() { echo -e "${PURPLE}[步骤]${NC} $1"; }

# 欢迎信息
show_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}     🐉 ${PURPLE}灵空 AI${NC} - 本地多模态人工智能                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}     你的 AI. 你的数据. 你的掌控.                             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 检测系统
detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
        PLATFORM="macos-arm64"
        BINARY_URL="$BINARY_URL_MACOS"
        log_success "检测到 macOS Apple Silicon"
    elif [[ "$os" == "Darwin" && "$arch" == "x86_64" ]]; then
        PLATFORM="macos-x64"
        log_warn "macOS Intel - 将使用 Sandbox 模式"
        INSTALL_MODE="sandbox"
    elif [[ "$os" == "Linux" && "$arch" == "x86_64" ]]; then
        PLATFORM="linux-x64"
        BINARY_URL="$BINARY_URL_LINUX"
        log_success "检测到 Linux x86_64"
    elif [[ "$os" == "Linux" && "$arch" == "aarch64" ]]; then
        PLATFORM="linux-arm64"
        log_warn "Linux ARM64 - 将使用 Sandbox 模式"
        INSTALL_MODE="sandbox"
    else
        log_error "不支持的系统: $os $arch"
        exit 1
    fi
}

# 检测下载工具
detect_downloader() {
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -fsSL"
        DOWNLOAD_TO="curl -fL --progress-bar -o"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -qO-"
        DOWNLOAD_TO="wget -q --show-progress -O"
    else
        log_error "请先安装 curl 或 wget"
        exit 1
    fi
}

# 检测 Docker
detect_docker() {
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            DOCKER_AVAILABLE=true
            log_success "Docker 已就绪"
        else
            DOCKER_AVAILABLE=false
            log_warn "Docker 已安装但未运行"
        fi
    else
        DOCKER_AVAILABLE=false
        log_warn "Docker 未安装"
    fi
}

# 创建目录
create_directories() {
    log_step "创建安装目录..."
    mkdir -p "$BIN_DIR" "$LIB_DIR" "$MODELS_DIR" "$SANDBOX_DIR" "$LINGKONG_HOME/apps" "$LINGKONG_HOME/logs" "$LINGKONG_HOME/run"
    log_success "目录创建完成: $LINGKONG_HOME"
}

# ================== Python 环境配置 ==================

# 检测 Python
detect_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_success "Python 已安装: $PYTHON_VERSION"
        return 0
    elif command -v python &> /dev/null; then
        local ver=$(python --version 2>&1 | cut -d' ' -f2)
        if [[ "$ver" == 3.* ]]; then
            PYTHON_CMD="python"
            PYTHON_VERSION="$ver"
            log_success "Python 已安装: $PYTHON_VERSION"
            return 0
        fi
    fi
    PYTHON_CMD=""
    log_warn "Python3 未安装"
    return 1
}

# 安装 Python (如果需要)
install_python() {
    log_step "安装 Python3..."

    if [[ "$PLATFORM" == "macos"* ]]; then
        if command -v brew &> /dev/null; then
            log_info "使用 Homebrew 安装 Python..."
            brew install python3 || true
        else
            log_error "请先安装 Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            log_error "或手动安装 Python3: https://www.python.org/downloads/"
            return 1
        fi
    elif command -v apt-get &> /dev/null; then
        log_info "使用 apt 安装 Python..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    elif command -v yum &> /dev/null; then
        log_info "使用 yum 安装 Python..."
        sudo yum install -y python3 python3-pip
    elif command -v dnf &> /dev/null; then
        log_info "使用 dnf 安装 Python..."
        sudo dnf install -y python3 python3-pip
    else
        log_error "无法自动安装 Python，请手动安装"
        return 1
    fi

    detect_python
}

# 安装 Python 依赖
install_python_deps() {
    log_step "安装 Python 依赖..."

    if [[ -z "$PYTHON_CMD" ]]; then
        if ! detect_python; then
            install_python || return 1
        fi
    fi

    # 使用 pip 安装依赖
    log_info "安装: $PYTHON_DEPS"

    # 国内镜像源 (解决网络超时问题)
    local PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

    # 创建虚拟环境或使用 --break-system-packages
    local venv_dir="$LINGKONG_HOME/venv"

    # 方法1: 尝试创建虚拟环境
    if $PYTHON_CMD -m venv "$venv_dir" 2>/dev/null; then
        log_info "使用虚拟环境: $venv_dir"
        source "$venv_dir/bin/activate"
        # 先升级 pip
        pip install --upgrade pip $PIP_MIRROR --quiet 2>/dev/null || true
        # 安装依赖 (使用国内镜像)
        pip install $PIP_MIRROR --timeout 60 $PYTHON_DEPS
        deactivate
        # 创建激活脚本链接
        echo "source \"$venv_dir/bin/activate\"" > "$LINGKONG_HOME/activate.sh"
        log_success "Python 依赖安装完成 (虚拟环境)"
        return 0
    fi

    # 方法2: 使用 --break-system-packages (macOS Homebrew Python 3.12+)
    log_info "使用系统 pip 安装..."
    if $PYTHON_CMD -m pip install --user --break-system-packages $PIP_MIRROR --timeout 60 $PYTHON_DEPS 2>/dev/null; then
        log_success "Python 依赖安装完成"
        return 0
    fi

    # 方法3: 传统方式
    if $PYTHON_CMD -m pip install --user $PIP_MIRROR --timeout 60 $PYTHON_DEPS 2>/dev/null; then
        log_success "Python 依赖安装完成"
        return 0
    fi

    log_warn "Python 依赖安装失败，请手动安装: pip3 install $PYTHON_DEPS"
    return 1
}

# ================== 原生安装 (macOS/Linux) ==================

install_native_binaries() {
    log_step "下载灵空 AI 引擎..."

    local tmp_dir=$(mktemp -d)
    local archive_name="llama-lingkong-${PLATFORM}.tar.gz"

    $DOWNLOAD_TO "$tmp_dir/$archive_name" "$BINARY_URL"

    log_info "解压文件..."
    tar -xzf "$tmp_dir/$archive_name" -C "$tmp_dir"

    # 复制文件 (适配不同平台)
    local extract_dir=$(ls -d "$tmp_dir"/llama-lingkong-* 2>/dev/null | head -1)
    if [[ -d "$extract_dir" ]]; then
        cp "$extract_dir"/llama-server "$BIN_DIR/" 2>/dev/null || true
        cp "$extract_dir"/llama-mtmd-cli "$BIN_DIR/" 2>/dev/null || true

        # macOS 动态库 - 同时复制到 lib/ 和 bin/ 目录
        # 因为 llama.cpp 二进制的 @rpath 会查找 @executable_path (bin目录)
        if [[ -d "$extract_dir/lib" ]]; then
            cp "$extract_dir"/lib/*.dylib "$LIB_DIR/" 2>/dev/null || true
            cp "$extract_dir"/lib/*.so "$LIB_DIR/" 2>/dev/null || true
            # 同时复制到 bin/ 目录确保 @rpath 能找到
            cp "$extract_dir"/lib/*.dylib "$BIN_DIR/" 2>/dev/null || true
            cp "$extract_dir"/lib/*.so "$BIN_DIR/" 2>/dev/null || true
        fi
    fi

    chmod +x "$BIN_DIR"/* 2>/dev/null || true

    rm -rf "$tmp_dir"
    log_success "引擎安装完成"
}

# 下载 WebUI
download_webui() {
    log_step "下载 WebUI 和 Gemini API..."

    local webui_dir="$LINGKONG_HOME/apps/webui"
    local gemini_dir="$LINGKONG_HOME/apps/gemini_api"
    mkdir -p "$webui_dir/static" "$gemini_dir"

    # 尝试从服务器下载打包好的 WebUI (总是更新)
    if curl -fsSL "$WEBUI_URL" -o "/tmp/webui.tar.gz" 2>/dev/null; then
        log_info "解压 WebUI..."
        tar -xzf "/tmp/webui.tar.gz" -C "$webui_dir" 2>/dev/null || true
        rm -f "/tmp/webui.tar.gz"
        if [[ -f "$webui_dir/server.py" ]]; then
            log_success "WebUI 下载完成"
        fi
    fi

    # 下载 Gemini API 服务器 (带 thoughtSignature)
    if curl -fsSL "$BASE_URL/gemini_api.tar.gz" -o "/tmp/gemini_api.tar.gz" 2>/dev/null; then
        log_info "解压 Gemini API..."
        tar -xzf "/tmp/gemini_api.tar.gz" -C "$gemini_dir" 2>/dev/null || true
        rm -f "/tmp/gemini_api.tar.gz"
        if [[ -f "$gemini_dir/server.py" ]]; then
            log_success "Gemini API 下载完成"
        fi
    fi
}

# 下载模型
# 下载单个文件 (带断点续传和备用地址)
download_file() {
    local dest="$1"
    local url_primary="$2"
    local url_fallback="$3"
    local name="$4"
    local expected_size="$5"  # 预期大小 (字节)

    # 检查文件是否已完整存在
    if [[ -f "$dest" ]]; then
        local size=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null || echo 0)
        local size_mb=$(($size/1024/1024))

        # 如果有预期大小，检查是否达到 95%
        if [[ -n "$expected_size" && $expected_size -gt 0 ]]; then
            local threshold=$(($expected_size * 95 / 100))
            if [[ $size -ge $threshold ]]; then
                log_info "$name 已完整 (${size_mb}MB)，跳过"
                return 0
            else
                log_info "$name 不完整 (${size_mb}MB)，继续下载..."
            fi
        else
            # 没有预期大小，直接尝试续传让服务器判断
            log_info "$name 已存在 (${size_mb}MB)，验证完整性..."
        fi
    fi

    log_info "下载 $name..."
    log_info "  来源: 国内镜像 (支持断点续传)"

    # 尝试主地址 (国内镜像) - 使用 -C - 支持断点续传
    if command -v curl &> /dev/null; then
        if curl -fL -C - --progress-bar --retry 3 --retry-delay 5 -o "$dest" "$url_primary"; then
            log_success "$name 下载完成"
            return 0
        fi
    elif command -v wget &> /dev/null; then
        if wget -c --show-progress --tries=3 -O "$dest" "$url_primary"; then
            log_success "$name 下载完成"
            return 0
        fi
    fi

    # 主地址失败，尝试备用地址 (HuggingFace)
    if [[ -n "$url_fallback" ]]; then
        log_warn "国内镜像下载失败，尝试 HuggingFace..."
        if command -v curl &> /dev/null; then
            if curl -fL -C - --progress-bar --retry 3 -o "$dest" "$url_fallback"; then
                log_success "$name 下载完成 (HuggingFace)"
                return 0
            fi
        elif command -v wget &> /dev/null; then
            if wget -c --show-progress --tries=3 -O "$dest" "$url_fallback"; then
                log_success "$name 下载完成 (HuggingFace)"
                return 0
            fi
        fi
    fi

    log_error "$name 下载失败，请尝试手动下载:"
    log_error "  curl -C - -o $dest $url_primary"
    return 1
}

download_models() {
    log_step "下载 AI 模型 (国内镜像优先)..."

    # 文本模型 (必需) - 2.6GB = 2789350528 bytes
    download_file \
        "$MODELS_DIR/gemma-3n-E2B-it-Q4_K_M.gguf" \
        "$MODEL_URL" \
        "$MODEL_URL_HF" \
        "文本模型 (2.6GB)" \
        2789350528

    # 视觉模型 - 570MB = 598999040 bytes
    download_file \
        "$MODELS_DIR/gemma-3n-vision-mmproj-f16.gguf" \
        "$VISION_URL" \
        "$VISION_URL_HF" \
        "视觉模型 (570MB)" \
        598999040

    # 音频模型 - 1.3GB = 1395864576 bytes
    download_file \
        "$MODELS_DIR/gemma-3n-audio-mmproj-f16.gguf" \
        "$AUDIO_URL" \
        "$AUDIO_URL_HF" \
        "音频模型 (1.3GB)" \
        1395864576
}

# 创建原生启动脚本
create_native_scripts() {
    log_step "创建启动脚本..."

    # lingkong 主命令 (支持 start/stop/status)
    cat > "$BIN_DIR/lingkong" << 'SCRIPT'
#!/bin/bash
# 灵空 AI 启动脚本 (多模态: 文本 + 视觉 + 音频)

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
MODEL="$LINGKONG_HOME/models/gemma-3n-E2B-it-Q4_K_M.gguf"
VISION="$LINGKONG_HOME/models/gemma-3n-vision-mmproj-f16.gguf"
AUDIO="$LINGKONG_HOME/models/gemma-3n-audio-mmproj-f16.gguf"
LLAMA_PORT="${LLAMA_PORT:-8081}"
WEBUI_PORT="${WEBUI_PORT:-8080}"
PID_DIR="$LINGKONG_HOME/run"
LOG_DIR="$LINGKONG_HOME/logs"

export DYLD_LIBRARY_PATH="$LINGKONG_HOME/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LINGKONG_HOME/lib:${LD_LIBRARY_PATH:-}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$PID_DIR" "$LOG_DIR"

# 快速清理端口 (并行执行，无等待)
cleanup_ports() {
    local ports_to_clean=""

    # 快速检测需要清理的端口
    for port in 5001 8080 8090; do
        if lsof -i :$port -t > /dev/null 2>&1; then
            ports_to_clean="$ports_to_clean $port"
        fi
    done

    # 如果有端口需要清理，并行清理
    if [[ -n "$ports_to_clean" ]]; then
        echo -e "${YELLOW}[清理]${NC} 端口$ports_to_clean"
        for port in $ports_to_clean; do
            lsof -i :$port -t 2>/dev/null | xargs kill -9 2>/dev/null &
        done
        wait  # 等待所有清理完成
        sleep 0.3  # 短暂等待端口释放
    fi
}

# macOS: 签名二进制
sign_binaries() {
    if [[ "$(uname)" == "Darwin" ]]; then
        codesign -s - --force "$LINGKONG_HOME/bin/llama-server" 2>/dev/null || true
        codesign -s - --force "$LINGKONG_HOME/bin/llama-mtmd-cli" 2>/dev/null || true
    fi
}

# 启动 Gemini API (带 thoughtSignature)
start_gemini_api() {
    # 检查服务是否已在运行且健康
    if [[ -f "$PID_DIR/gemini.pid" ]] && kill -0 "$(cat "$PID_DIR/gemini.pid")" 2>/dev/null; then
        if curl -s --connect-timeout 1 "http://localhost:5001/health" > /dev/null 2>&1; then
            echo -e "${GREEN}[运行中]${NC} Gemini API (PID: $(cat "$PID_DIR/gemini.pid"))"
            return 0
        fi
    fi

    # 清理端口
    cleanup_ports

    if [[ ! -f "$LINGKONG_HOME/apps/gemini_api/server.py" ]]; then
        echo -e "${YELLOW}[警告]${NC} Gemini API 未安装，跳过"
        return 1
    fi

    sign_binaries

    # 设置环境变量
    export LLAMA_SERVER_BIN="$LINGKONG_HOME/bin/llama-server"
    export LLAMA_MTMD_BIN="$LINGKONG_HOME/bin/llama-mtmd-cli"
    export LLAMA_MODEL="$MODEL"
    export LLAMA_MODEL_AUDIO="$MODEL"
    export MMPROJ_IMAGE="$VISION"
    export MMPROJ_AUDIO="$AUDIO"
    export GEMINI_API_LLAMA_PORT="8090"
    export DYLD_LIBRARY_PATH="$LINGKONG_HOME/lib:${DYLD_LIBRARY_PATH:-}"

    cd "$LINGKONG_HOME/apps/gemini_api"

    if [[ -f "$LINGKONG_HOME/venv/bin/python" ]]; then
        nohup "$LINGKONG_HOME/venv/bin/python" server.py > "$LOG_DIR/gemini.log" 2>&1 &
    else
        nohup python3 server.py > "$LOG_DIR/gemini.log" 2>&1 &
    fi
    echo $! > "$PID_DIR/gemini.pid"
    echo -e "${GREEN}[启动]${NC} Gemini API (PID: $(cat "$PID_DIR/gemini.pid"))"

    # 快速等待就绪 (最多 30 秒，间隔 0.5 秒)
    for i in {1..60}; do
        if curl -s --connect-timeout 1 "http://localhost:5001/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo -e "${YELLOW}[提示]${NC} Gemini API 后台启动中..."
}

# 启动 WebUI (聊天界面)
start_webui() {
    # 检查服务是否已在运行且健康
    if [[ -f "$PID_DIR/webui.pid" ]] && kill -0 "$(cat "$PID_DIR/webui.pid")" 2>/dev/null; then
        if curl -s --connect-timeout 1 "http://localhost:$WEBUI_PORT/" > /dev/null 2>&1; then
            echo -e "${GREEN}[运行中]${NC} WebUI (PID: $(cat "$PID_DIR/webui.pid"))"
            return 0
        fi
    fi

    if [[ ! -f "$LINGKONG_HOME/apps/webui/server.py" ]]; then
        echo -e "${YELLOW}[警告]${NC} WebUI 未安装，跳过"
        return 1
    fi

    # 设置环境变量 - WebUI 使用 Gemini API 的 llama-server (8090)
    export LLAMA_SERVER_PORT="8090"
    export LLAMA_MM_MODEL="$MODEL"
    export LLAMA_MM_PROJ_IMAGE="$VISION"
    export LLAMA_MM_PROJ_AUDIO="$AUDIO"
    export LLAMA_MTMD_BIN="$LINGKONG_HOME/bin/llama-mtmd-cli"
    export WEBUI_PORT="$WEBUI_PORT"

    cd "$LINGKONG_HOME/apps/webui"

    if [[ -f "$LINGKONG_HOME/venv/bin/python" ]]; then
        nohup "$LINGKONG_HOME/venv/bin/python" server.py > "$LOG_DIR/webui.log" 2>&1 &
    else
        nohup python3 server.py > "$LOG_DIR/webui.log" 2>&1 &
    fi
    echo $! > "$PID_DIR/webui.pid"
    echo -e "${GREEN}[启动]${NC} WebUI (PID: $(cat "$PID_DIR/webui.pid"))"

    # 快速等待就绪 (最多 5 秒，间隔 0.3 秒)
    for i in {1..15}; do
        if curl -s --connect-timeout 1 "http://localhost:$WEBUI_PORT/" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.3
    done
}

# 停止服务
stop_all() {
    for name in webui gemini llama; do
        local pid_file="$PID_DIR/$name.pid"
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
                echo -e "${GREEN}[成功]${NC} 已停止 $name (PID: $pid)"
            fi
            rm -f "$pid_file"
        fi
    done
    pkill -f "llama-server.*8090" 2>/dev/null || true
    pkill -f "gemini_api/server.py" 2>/dev/null || true
    pkill -f "webui/server.py" 2>/dev/null || true
}

# 检查更新
check_update() {
    local SERVER_URL="$LINGKONG_SERVER"
    local CHECKSUMS_URL="$SERVER_URL/checksums.sha256"
    local LOCAL_CHECKSUMS="$LINGKONG_HOME/checksums.sha256"
    local NEEDS_UPDATE=false

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  灵空 AI 更新检查${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # 下载服务器校验和
    local TMP_CHECKSUMS="/tmp/lingkong_checksums_$$.sha256"
    if ! curl -fsSL "$CHECKSUMS_URL" -o "$TMP_CHECKSUMS" 2>/dev/null; then
        echo -e "${RED}[错误]${NC} 无法连接服务器，请检查网络"
        rm -f "$TMP_CHECKSUMS"
        return 1
    fi

    echo -e "${CYAN}检查组件更新:${NC}"
    echo ""

    # 检查 gemini_api
    local SERVER_GEMINI_HASH=$(grep "gemini_api.tar.gz" "$TMP_CHECKSUMS" | awk '{print $1}')
    local LOCAL_GEMINI_HASH=""
    if [[ -f "$LINGKONG_HOME/apps/gemini_api/server.py" ]]; then
        # 计算本地 gemini_api 目录的哈希 (简化: 只检查 server.py)
        LOCAL_GEMINI_HASH=$(shasum -a 256 "$LINGKONG_HOME/apps/gemini_api/server.py" 2>/dev/null | awk '{print $1}')
    fi

    # 检查二进制
    local SERVER_BIN_HASH=$(grep "llama-lingkong-macos-arm64.tar.gz" "$TMP_CHECKSUMS" | awk '{print $1}')
    local LOCAL_BIN_HASH=""
    if [[ -f "$LINGKONG_HOME/bin/llama-mtmd-cli" ]]; then
        LOCAL_BIN_HASH=$(shasum -a 256 "$LINGKONG_HOME/bin/llama-mtmd-cli" 2>/dev/null | awk '{print $1}')
    fi

    # 检查 webui
    local SERVER_WEBUI_HASH=$(grep "webui.tar.gz" "$TMP_CHECKSUMS" | awk '{print $1}')
    local LOCAL_WEBUI_HASH=""
    if [[ -f "$LINGKONG_HOME/apps/webui/server.py" ]]; then
        LOCAL_WEBUI_HASH=$(shasum -a 256 "$LINGKONG_HOME/apps/webui/server.py" 2>/dev/null | awk '{print $1}')
    fi

    # 显示状态 (简化比较: 服务器有新包就提示更新)
    # 实际比较需要下载包并解压，这里用时间戳或版本号更简单
    # 我们用一个本地记录文件来跟踪上次更新
    local LAST_UPDATE_FILE="$LINGKONG_HOME/.last_update"
    local SERVER_CHECKSUM_HASH=$(shasum -a 256 "$TMP_CHECKSUMS" | awk '{print $1}')
    local LOCAL_CHECKSUM_HASH=""
    if [[ -f "$LOCAL_CHECKSUMS" ]]; then
        LOCAL_CHECKSUM_HASH=$(shasum -a 256 "$LOCAL_CHECKSUMS" | awk '{print $1}')
    fi

    if [[ "$SERVER_CHECKSUM_HASH" != "$LOCAL_CHECKSUM_HASH" ]]; then
        NEEDS_UPDATE=true
        echo -e "  ${YELLOW}● 发现新版本${NC}"
        echo ""
        echo -e "  服务器校验和已更新，建议重新安装以获取最新修复。"
    else
        echo -e "  ${GREEN}● 已是最新版本${NC}"
    fi

    echo ""

    if [[ "$NEEDS_UPDATE" == "true" ]]; then
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  发现更新! 运行以下命令更新:${NC}"
        echo ""
        echo -e "  ${CYAN}curl -fsSL $LINGKONG_SERVER/install.sh | bash${NC}"
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"

        # 询问是否立即更新
        read -p "是否立即更新? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${CYAN}开始更新...${NC}"
            curl -fsSL "$LINGKONG_SERVER/install.sh" | bash
        fi
    else
        # 保存当前校验和
        cp "$TMP_CHECKSUMS" "$LOCAL_CHECKSUMS" 2>/dev/null || true
    fi

    rm -f "$TMP_CHECKSUMS"
}

# 显示状态
show_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  灵空 AI 服务状态${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

    if [[ -f "$PID_DIR/gemini.pid" ]] && kill -0 "$(cat "$PID_DIR/gemini.pid")" 2>/dev/null; then
        echo -e "  Gemini API: ${GREEN}● 运行中${NC} (PID: $(cat "$PID_DIR/gemini.pid"))"
    else
        echo -e "  Gemini API: ${RED}○ 已停止${NC}"
    fi

    if [[ -f "$PID_DIR/webui.pid" ]] && kill -0 "$(cat "$PID_DIR/webui.pid")" 2>/dev/null; then
        echo -e "  WebUI:      ${GREEN}● 运行中${NC} (PID: $(cat "$PID_DIR/webui.pid"))"
    else
        echo -e "  WebUI:      ${RED}○ 已停止${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}WebUI:${NC}      http://localhost:$WEBUI_PORT"
    echo -e "  ${CYAN}Playground:${NC} http://localhost:$WEBUI_PORT/static/playground.html"
    echo ""
}

# 主函数
case "${1:-start}" in
    start|up)
        echo ""
        echo -e "${CYAN}🐉 启动 灵空 AI...${NC}"
        echo ""
        start_gemini_api
        start_webui
        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✅ 灵空 AI 已启动!${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "  🌐 ${CYAN}WebUI:${NC}      ${YELLOW}http://localhost:$WEBUI_PORT${NC}"
        echo -e "  🧪 ${CYAN}Playground:${NC} ${YELLOW}http://localhost:$WEBUI_PORT/static/playground.html${NC}"
        echo ""
        echo -e "  ${CYAN}停止:${NC}   lingkong stop"
        echo -e "  ${CYAN}日志:${NC}   lingkong logs"
        echo ""
        # 打开浏览器
        if [[ "$(uname)" == "Darwin" ]]; then
            open "http://localhost:$WEBUI_PORT" 2>/dev/null || true
        fi
        ;;
    stop|down)
        echo -e "${CYAN}停止 灵空 AI...${NC}"
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        exec "$0" start
        ;;
    status|ps)
        show_status
        ;;
    logs)
        tail -f "$LOG_DIR/webui.log" "$LOG_DIR/gemini.log"
        ;;
    update|upgrade)
        echo -e "${CYAN}🔄 检查更新...${NC}"
        check_update
        ;;
    *)
        echo "使用方法: lingkong [start|stop|restart|status|logs|update]"
        ;;
esac
SCRIPT

    chmod +x "$BIN_DIR/lingkong"
    log_success "启动脚本创建完成"
}

# ================== Sandbox 安装 (Docker) ==================

install_docker() {
    log_step "安装 Docker..."

    if [[ "$PLATFORM" == "macos"* ]]; then
        log_info "请手动安装 Docker Desktop: https://www.docker.com/products/docker-desktop/"
        log_info "安装完成后重新运行此脚本"
        exit 1
    fi

    # Linux 自动安装 Docker
    if command -v apt-get &> /dev/null; then
        log_info "使用 apt 安装 Docker..."
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose-plugin
        sudo systemctl enable docker
        sudo systemctl start docker
        sudo usermod -aG docker "$USER"
        log_success "Docker 安装完成"
        log_warn "请重新登录以使 Docker 组权限生效"
    elif command -v yum &> /dev/null; then
        log_info "使用 yum 安装 Docker..."
        sudo yum install -y docker docker-compose-plugin
        sudo systemctl enable docker
        sudo systemctl start docker
        sudo usermod -aG docker "$USER"
        log_success "Docker 安装完成"
    else
        log_error "无法自动安装 Docker，请手动安装"
        exit 1
    fi
}

install_sandbox() {
    log_step "安装 Sandbox 环境..."

    # 检查 Docker
    if [[ "$DOCKER_AVAILABLE" != "true" ]]; then
        log_warn "Docker 未就绪，尝试安装..."
        install_docker
        detect_docker
    fi

    # 下载 Sandbox 配置
    log_info "下载 Sandbox 配置..."

    # 创建 docker-compose.yml
    cat > "$SANDBOX_DIR/docker-compose.yml" << 'COMPOSE'
# 灵空 AI Sandbox - Docker Compose (多模态: 文本 + 视觉 + 音频)

services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server
    container_name: lingkong-llama
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ${LINGKONG_HOME:-~/.lingkong}/models:/models:ro
    command: >
      --model /models/gemma-3n-E2B-it-Q4_K_M.gguf
      --mmproj /models/gemma-3n-vision-mmproj-f16.gguf,/models/gemma-3n-audio-mmproj-f16.gguf
      --host 0.0.0.0
      --port 8080
      -ngl 99
      --flash-attn on
      -c 8192
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  gemini-api:
    image: python:3.11-slim
    container_name: lingkong-gemini
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ${LINGKONG_HOME:-~/.lingkong}/apps:/app:ro
    working_dir: /app/gemini_api
    environment:
      - LLAMA_SERVER_HOST=llama-server
      - LLAMA_SERVER_PORT=8080
    command: >
      bash -c "pip install flask flask-cors requests -q && python server.py --port 8080"
    depends_on:
      llama-server:
        condition: service_healthy

networks:
  default:
    name: lingkong-network
COMPOSE

    # 创建 Sandbox 启动脚本
    cat > "$BIN_DIR/lingkong" << 'SCRIPT'
#!/bin/bash
# 灵空 AI Sandbox 启动脚本

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
SANDBOX_DIR="$LINGKONG_HOME/sandbox"

export LINGKONG_HOME

cd "$SANDBOX_DIR"

case "${1:-start}" in
    start|up)
        echo "🐉 启动 灵空 AI Sandbox..."
        docker compose up -d
        echo ""
        echo "  WebUI:     http://localhost:8080"
        echo "  Gemini API: http://localhost:8080"
        echo ""
        echo "  查看日志: lingkong logs"
        echo "  停止服务: lingkong stop"
        ;;
    stop|down)
        echo "停止 灵空 AI..."
        docker compose down
        ;;
    logs)
        docker compose logs -f "${@:2}"
        ;;
    status|ps)
        docker compose ps
        ;;
    restart)
        docker compose restart
        ;;
    *)
        echo "使用方法: lingkong [start|stop|logs|status|restart]"
        ;;
esac
SCRIPT

    chmod +x "$BIN_DIR/lingkong"
    log_success "Sandbox 环境安装完成"
}

# 添加到 PATH
setup_path() {
    log_step "配置环境..."

    local shell_rc=""
    if [[ -n "$ZSH_VERSION" ]] || [[ "$SHELL" == *"zsh"* ]]; then
        shell_rc="$HOME/.zshrc"
    elif [[ -f "$HOME/.bashrc" ]]; then
        shell_rc="$HOME/.bashrc"
    else
        shell_rc="$HOME/.profile"
    fi

    local path_line="export PATH=\"\$HOME/.lingkong/bin:\$PATH\""

    if [[ -f "$shell_rc" ]] && grep -q ".lingkong/bin" "$shell_rc" 2>/dev/null; then
        log_info "PATH 已配置"
    elif [[ -w "$shell_rc" ]] || [[ ! -f "$shell_rc" ]]; then
        echo "" >> "$shell_rc"
        echo "# 灵空 AI" >> "$shell_rc"
        echo "$path_line" >> "$shell_rc"
        log_success "已添加到 $shell_rc"
    else
        log_warn "无法修改 $shell_rc"
        log_info "请手动添加到 shell 配置:"
        echo "  $path_line"
    fi

    # 立即生效
    export PATH="$HOME/.lingkong/bin:$PATH"
}

# 完成提示
show_completion() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                                                              ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}     ✅ ${PURPLE}灵空 AI${NC} 安装完成!                                    ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}                                                              ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        echo -e "  安装模式: ${CYAN}Sandbox (Docker)${NC}"
    else
        echo -e "  安装模式: ${CYAN}原生${NC}"
    fi
    echo -e "  功能: 文本对话 + 图像理解 + 音频转录 + 会话记忆 + Gemini API"
    echo ""
    echo -e "  ${CYAN}安装目录:${NC}"
    echo -e "    程序: ${YELLOW}~/.lingkong/bin/${NC}"
    echo -e "    模型: ${YELLOW}~/.lingkong/models/${NC}"
    echo -e "    日志: ${YELLOW}~/.lingkong/logs/${NC}"
    echo ""
}

# 启动服务
start_service() {
    log_step "启动灵空 AI..."

    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        # Sandbox 模式
        cd "$SANDBOX_DIR"
        docker compose up -d

        # 等待服务就绪
        log_info "等待服务启动..."
        local count=0
        while ! curl -s http://localhost:8080/health > /dev/null 2>&1; do
            sleep 2
            count=$((count + 1))
            if [[ $count -gt 60 ]]; then
                log_warn "服务启动超时，请检查 Docker 日志"
                break
            fi
        done
    else
        # 原生模式
        export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"

        # 后台启动
        "$BIN_DIR/lingkong" &
        local pid=$!

        # 等待服务启动
        log_info "等待服务启动..."
        local count=0
        while ! curl -s http://localhost:8080/health > /dev/null 2>&1; do
            sleep 1
            count=$((count + 1))
            if [[ $count -gt 30 ]]; then
                log_warn "服务启动超时"
                break
            fi
        done
    fi

    log_success "服务已启动"

    # 打开浏览器
    if [[ "$PLATFORM" == "macos"* ]]; then
        log_info "打开浏览器..."
        open "http://localhost:8080" 2>/dev/null || true
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:8080" 2>/dev/null || true
    fi

    echo ""
    echo -e "  ${CYAN}浏览器已打开: ${YELLOW}http://localhost:8080${NC}"
    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        echo -e "  ${CYAN}Gemini API: ${YELLOW}http://localhost:8080${NC}"
        echo ""
        echo -e "  ${CYAN}查看日志: ${YELLOW}lingkong logs${NC}"
        echo -e "  ${CYAN}停止服务: ${YELLOW}lingkong stop${NC}"
    else
        echo -e "  ${CYAN}按 Ctrl+C 停止服务${NC}"
        echo ""
        # 原生模式前台等待
        wait $pid 2>/dev/null || true
    fi
}

# 主函数
main() {
    show_banner
    detect_platform
    detect_downloader
    detect_docker

    # 自动选择模式
    if [[ "$INSTALL_MODE" == "auto" ]]; then
        if [[ "$DOCKER_AVAILABLE" == "true" ]]; then
            # 有 Docker 优先使用 Sandbox
            log_info "检测到 Docker，使用 Sandbox 模式"
            INSTALL_MODE="sandbox"
        elif [[ "$PLATFORM" == "macos-arm64" ]] || [[ "$PLATFORM" == "linux-x64" ]]; then
            # 支持原生安装的平台
            log_info "使用原生安装模式"
            INSTALL_MODE="native"
        else
            # 其他平台必须用 Docker
            log_info "此平台需要 Sandbox 模式"
            INSTALL_MODE="sandbox"
        fi
    fi

    create_directories
    download_models

    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        install_sandbox
    else
        install_native_binaries
        download_webui
        detect_python || install_python
        install_python_deps
        create_native_scripts
    fi

    setup_path
    show_completion
    start_service
}

main "$@"

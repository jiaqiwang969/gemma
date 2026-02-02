#!/bin/bash
# =============================================================================
# 灵空 AI - 一键安装脚本 (Sandbox 版)
# =============================================================================
# 使用方法:
#   curl -fsSL https://lingkong.xyz/install.sh | bash           # 默认模式
#   curl -fsSL https://lingkong.xyz/install.sh | bash -s sandbox # Sandbox 模式
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
OPENCLAW_APP_DIR="$LINGKONG_HOME/apps/openclaw"
OPENCLAW_STATE_ROOT="$LINGKONG_HOME/openclaw"

# 下载地址 (支持环境变量覆盖)
LINGKONG_SERVER="${LINGKONG_SERVER:-https://lingkong.xyz}"
BASE_URL="$LINGKONG_SERVER"
BINARY_URL_MACOS="$BASE_URL/bin/llama-lingkong-macos-arm64.tar.gz"
BINARY_URL_LINUX_CUDA="$BASE_URL/bin/llama-lingkong-linux-x86_64-cuda.tar.gz"
BINARY_URL_LINUX_CPU="$BASE_URL/bin/llama-lingkong-linux-x86_64.tar.gz"
WEBUI_URL="$BASE_URL/webui.tar.gz"
SANDBOX_URL="$BASE_URL/sandbox.tar.gz"
# OpenClaw (WhatsApp agent gateway) bundle.
# Note: today we ship macOS arm64 only; keep URLs overridable for local testing.
OPENCLAW_BUNDLE_URL_MACOS="${OPENCLAW_BUNDLE_URL_MACOS:-$BASE_URL/bin/openclaw-macos-arm64.tar.gz}"

# Whisper.cpp (offline STT) bundle (macOS arm64).
WHISPER_BUNDLE_URL_MACOS="${WHISPER_BUNDLE_URL_MACOS:-$BASE_URL/bin/whisper-cli-macos-arm64.tar.gz}"

# Optional Node runtime (used when system Node is missing).
NODE_VERSION_DEFAULT="${NODE_VERSION_DEFAULT:-22.20.0}"
NODE_TARBALL_URL_MACOS="${NODE_TARBALL_URL_MACOS:-$BASE_URL/bin/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz}"
NODE_TARBALL_URL_MACOS_FALLBACK="https://nodejs.org/dist/v${NODE_VERSION_DEFAULT}/node-v${NODE_VERSION_DEFAULT}-darwin-arm64.tar.gz"

# 模型下载地址 (优先国内镜像，备用 HuggingFace)
MODELS_BASE="$BASE_URL/models"
MODEL_URL="$MODELS_BASE/gemma-3n-E2B-it-Q4_K_M.gguf"
VISION_URL="$MODELS_BASE/gemma-3n-vision-mmproj-f16.gguf"
AUDIO_URL="$MODELS_BASE/gemma-3n-audio-mmproj-f16.gguf"
WHISPER_URL="$MODELS_BASE/whisper/ggml-small.bin"

# HuggingFace 备用地址
HF_BASE="https://huggingface.co/nicepkg/gemma-3n-gguf/resolve/main"
MODEL_URL_HF="$HF_BASE/gemma-3n-E2B-it-Q4_K_M.gguf"
VISION_URL_HF="$HF_BASE/gemma-3n-vision-mmproj-f16.gguf"
AUDIO_URL_HF="$HF_BASE/gemma-3n-audio-mmproj-f16.gguf"
WHISPER_URL_HF="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"

# 默认不下载 Gemma audio mmproj（1.3GB）；WhatsApp 语音默认由 whisper.cpp 负责转写。
# 如需开启“音频理解/音频摘要”等能力：
#   LINGKONG_DOWNLOAD_AUDIO_MMPROJ=1 curl -fsSL https://lingkong.xyz/install.sh | bash
DOWNLOAD_AUDIO_MMPROJ="${LINGKONG_DOWNLOAD_AUDIO_MMPROJ:-0}"

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
        log_success "检测到 macOS Apple Silicon (Metal GPU 加速)"
    elif [[ "$os" == "Darwin" && "$arch" == "x86_64" ]]; then
        PLATFORM="macos-x64"
        log_warn "macOS Intel - 将使用 Sandbox 模式"
        INSTALL_MODE="sandbox"
    elif [[ "$os" == "Linux" && "$arch" == "x86_64" ]]; then
        PLATFORM="linux-x64"
        # 检测 NVIDIA GPU
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            HAS_NVIDIA=true
            BINARY_URL="$BINARY_URL_LINUX_CUDA"
            local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            log_success "检测到 Linux x86_64 + NVIDIA GPU: $gpu_name"
            log_info "将使用 CUDA 加速版本"
        else
            HAS_NVIDIA=false
            BINARY_URL="$BINARY_URL_LINUX_CPU"
            log_success "检测到 Linux x86_64 (CPU 模式)"
            log_warn "未检测到 NVIDIA GPU，将使用 CPU 版本"
        fi
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
    mkdir -p "$BIN_DIR" "$LIB_DIR" "$MODELS_DIR" "$MODELS_DIR/whisper" "$SANDBOX_DIR" "$LINGKONG_HOME/apps" "$LINGKONG_HOME/logs" "$LINGKONG_HOME/run" "$OPENCLAW_APP_DIR" "$OPENCLAW_STATE_ROOT"
    # OpenClaw state holds auth + tokens; keep it private by default.
    chmod 700 "$OPENCLAW_STATE_ROOT" 2>/dev/null || true
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

    # 根据平台选择正确的文件名
    local archive_name
    if [[ "$PLATFORM" == "macos-arm64" ]]; then
        archive_name="llama-lingkong-macos-arm64.tar.gz"
    elif [[ "$PLATFORM" == "linux-x64" && "$HAS_NVIDIA" == "true" ]]; then
        archive_name="llama-lingkong-linux-x86_64-cuda.tar.gz"
    else
        archive_name="llama-lingkong-linux-x86_64.tar.gz"
    fi

    log_info "下载: $BINARY_URL"
    $DOWNLOAD_TO "$tmp_dir/$archive_name" "$BINARY_URL"

    log_info "解压文件..."
    tar -xzf "$tmp_dir/$archive_name" -C "$tmp_dir"

    # 复制文件 (适配不同平台)
    local extract_dir=$(ls -d "$tmp_dir"/llama-lingkong-* 2>/dev/null | head -1)
    if [[ -d "$extract_dir" ]]; then
        cp "$extract_dir"/llama-server "$BIN_DIR/" 2>/dev/null || true
        cp "$extract_dir"/llama-mtmd-cli "$BIN_DIR/" 2>/dev/null || true

        # 动态库 - 同时复制到 lib/ 和 bin/ 目录
        if [[ -d "$extract_dir/lib" ]]; then
            # macOS dylib
            cp "$extract_dir"/lib/*.dylib "$LIB_DIR/" 2>/dev/null || true
            cp "$extract_dir"/lib/*.dylib "$BIN_DIR/" 2>/dev/null || true
            # Linux so
            cp "$extract_dir"/lib/*.so* "$LIB_DIR/" 2>/dev/null || true
            cp "$extract_dir"/lib/*.so* "$BIN_DIR/" 2>/dev/null || true
        fi
    fi

    chmod +x "$BIN_DIR"/* 2>/dev/null || true

    rm -rf "$tmp_dir"

    if [[ "$HAS_NVIDIA" == "true" ]]; then
        log_success "引擎安装完成 (CUDA 加速)"
    else
        log_success "引擎安装完成"
    fi
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

# 安装 Node.js (仅在缺少系统 node 时，用于运行 OpenClaw)
install_node_runtime() {
    if command -v node &> /dev/null; then
        return 0
    fi

    if [[ "$PLATFORM" != "macos-arm64" ]]; then
        log_warn "未检测到 node；OpenClaw 需要 Node.js。请手动安装 Node 后再试。"
        return 1
    fi

    log_step "安装 Node.js (用于 OpenClaw)..."

    local node_dir="$LINGKONG_HOME/node"
    local tmp_dir
    tmp_dir=$(mktemp -d)

    local tar_path="$tmp_dir/node.tar.gz"
    local url="$NODE_TARBALL_URL_MACOS"

    log_info "下载 Node: $url"
    if ! curl -fL --progress-bar "$url" -o "$tar_path" 2>/dev/null; then
        log_warn "镜像下载失败，尝试官方源..."
        url="$NODE_TARBALL_URL_MACOS_FALLBACK"
        log_info "下载 Node: $url"
        curl -fL --progress-bar "$url" -o "$tar_path"
    fi

    rm -rf "$node_dir"
    mkdir -p "$node_dir"

    tar -xzf "$tar_path" -C "$tmp_dir"
    local extracted
    extracted=$(ls -d "$tmp_dir"/node-v*-darwin-arm64 2>/dev/null | head -1)
    if [[ ! -d "$extracted" ]]; then
        log_error "Node 解压失败"
        rm -rf "$tmp_dir"
        return 1
    fi

    # Flatten into ~/.lingkong/node/{bin,lib,...}
    cp -R "$extracted"/* "$node_dir"/
    rm -rf "$tmp_dir"

    if [[ -x "$node_dir/bin/node" ]]; then
        log_success "Node 已安装: $node_dir/bin/node"
    else
        log_error "Node 安装失败: node 不可执行"
        return 1
    fi
}

# 下载 OpenClaw bundle (WhatsApp agent gateway)
download_openclaw() {
    if [[ "$PLATFORM" != "macos-arm64" ]]; then
        log_info "OpenClaw 暂仅提供 macOS arm64 bundle，跳过 ($PLATFORM)"
        return 0
    fi

    log_step "下载 OpenClaw (WhatsApp agent)..."

    local bundle_url="${OPENCLAW_BUNDLE_URL:-$OPENCLAW_BUNDLE_URL_MACOS}"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local tar_path="$tmp_dir/openclaw.tar.gz"
    local extract_dir="$tmp_dir/extract"
    mkdir -p "$extract_dir"

    log_info "下载: $bundle_url"
    if ! curl -fL --progress-bar "$bundle_url" -o "$tar_path" 2>/dev/null; then
        log_warn "OpenClaw bundle 下载失败，跳过"
        rm -rf "$tmp_dir"
        return 1
    fi

    log_info "解压 OpenClaw..."
    tar -xzf "$tar_path" -C "$extract_dir"

    if [[ ! -d "$extract_dir/openclaw" ]]; then
        log_error "OpenClaw bundle 格式错误: 缺少 openclaw/ 目录"
        rm -rf "$tmp_dir"
        return 1
    fi

    rm -rf "$OPENCLAW_APP_DIR"
    mkdir -p "$(dirname "$OPENCLAW_APP_DIR")"
    mv "$extract_dir/openclaw" "$OPENCLAW_APP_DIR"
    if [[ -f "$extract_dir/BUILD_INFO.txt" ]]; then
        mv "$extract_dir/BUILD_INFO.txt" "$OPENCLAW_APP_DIR/BUILD_INFO.txt" 2>/dev/null || true
    fi

    rm -rf "$tmp_dir"
    log_success "OpenClaw 安装完成: $OPENCLAW_APP_DIR"
}

# 下载 whisper.cpp CLI (whisper-cli) - 用于离线语音转写 (STT)
download_whisper_cli() {
    if [[ "$PLATFORM" != "macos-arm64" ]]; then
        # 目前发行目标只覆盖 macOS arm64
        return 0
    fi

    # 已安装则跳过
    if [[ -x "$BIN_DIR/whisper-cli" ]]; then
        log_info "whisper-cli 已存在，跳过: $BIN_DIR/whisper-cli"
        return 0
    fi

    log_step "安装 whisper-cli (离线 STT)..."

    local bundle_url="${WHISPER_BUNDLE_URL:-$WHISPER_BUNDLE_URL_MACOS}"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local tar_path="$tmp_dir/whisper-cli.tar.gz"
    local extract_dir="$tmp_dir/extract"
    mkdir -p "$extract_dir"

    log_info "下载: $bundle_url"
    if curl -fL --progress-bar "$bundle_url" -o "$tar_path" 2>/dev/null; then
        log_info "解压 whisper-cli..."
        tar -xzf "$tar_path" -C "$extract_dir"

        local candidate=""
        if [[ -x "$extract_dir/whisper-cli" ]]; then
            candidate="$extract_dir/whisper-cli"
        elif [[ -x "$extract_dir/bin/whisper-cli" ]]; then
            candidate="$extract_dir/bin/whisper-cli"
        fi

        if [[ -n "$candidate" ]]; then
            cp "$candidate" "$BIN_DIR/whisper-cli"
            chmod +x "$BIN_DIR/whisper-cli"
            rm -rf "$tmp_dir"
            log_success "whisper-cli 安装完成: $BIN_DIR/whisper-cli"
            return 0
        fi
        log_warn "whisper-cli bundle 格式不符合预期，尝试系统 whisper-cli 作为兜底"
    else
        log_warn "whisper-cli bundle 下载失败，尝试系统 whisper-cli 作为兜底"
    fi

    # Fallback: copy system whisper-cli if present (Homebrew, etc.)
    local sys_bin
    sys_bin="$(command -v whisper-cli || true)"
    if [[ -n "$sys_bin" && -x "$sys_bin" ]]; then
        cp "$sys_bin" "$BIN_DIR/whisper-cli"
        chmod +x "$BIN_DIR/whisper-cli"
        rm -rf "$tmp_dir"
        log_success "已使用系统 whisper-cli: $sys_bin"
        return 0
    fi

    rm -rf "$tmp_dir"
    log_warn "未能安装 whisper-cli：语音消息将无法离线转写 (可稍后重新运行 installer)"
    return 1
}

# 写入 OpenClaw 离线默认配置 (不覆盖用户已有配置)
write_openclaw_config_to() {
    local dest_path="$1"
    local dm_policy="$2"
    local allow_from_block="$3"

    cat >"$dest_path" <<EOF
// OpenClaw offline-first profile for LingKong (macOS arm64).
//
// Goal: run fully offline *except* WhatsApp transport.
// Runtime env (recommended):
//   OPENCLAW_STATE_DIR=$OPENCLAW_STATE_ROOT
//   OPENCLAW_CONFIG_PATH=$dest_path
//   OPENCLAW_OFFLINE=1
{
  gateway: {
    mode: "local",
    bind: "loopback",
    controlUi: { enabled: false },
  },

  // WhatsApp is the primary interface; keep auxiliary HTTP surfaces off.
  canvasHost: { enabled: false },

  update: {
    checkOnStart: false,
  },

  models: {
    mode: "replace",
    providers: {
      google: {
        baseUrl: "http://127.0.0.1:5001/v1beta",
        apiKey: "local-noauth",
        api: "google-generative-ai",
        models: [
          {
            id: "gemini-3-pro-preview",
            name: "LingKong (Gemma 3n) via local Gemini API",
            reasoning: false,
            // Enable offline image understanding (requires vision mmproj). In offline mode, OpenClaw
            // uses media-understanding to inject a text description and avoids inlining base64 media.
            input: ["text", "image"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },

  agents: {
    defaults: {
      model: { primary: "google/gemini-3-pro-preview" },
      memorySearch: { enabled: false },
      // Keep OpenClaw workspace co-located with LingKong state under ~/.lingkong.
      workspace: "$OPENCLAW_STATE_ROOT/workspace",
      thinkingDefault: "off",
      verboseDefault: "off",
    },
  },

  tools: {
    // Default to minimal tool surface for lower latency/token usage; power users can
    // switch profiles later.
    profile: "minimal",
    deny: ["group:web", "browser"],
    web: {
      search: { enabled: false },
      fetch: { enabled: false },
    },
    media: {
      image: {
        enabled: true,
        // Keep the injected vision description short and in Chinese so the main reply stays Chinese.
        maxChars: 400,
        prompt: "请用中文一句话描述图片的主要内容；不要推测；不要解释。",
      },
      video: { enabled: false },
      audio: { enabled: true, language: "zh" },
    },
    links: { enabled: false },
    agentToAgent: { enabled: true },
  },

  messages: {
    responsePrefix: "",
    // When busy, process WhatsApp backlog one-by-one (avoid "collected" prompts).
    queue: {
      byChannel: { whatsapp: "followup" },
      debounceMsByChannel: { whatsapp: 250 },
    },
    tts: {
      auto: "inbound",
      provider: "macos-say",
      macosSay: { enabled: true, preferOpus: true },
    },
  },

  channels: {
    whatsapp: {
      dmPolicy: "$dm_policy",
      $allow_from_block
    },
  },
}
EOF
}

write_openclaw_config() {
    local state_dir="$OPENCLAW_STATE_ROOT"
    local config_path="$state_dir/openclaw.json"

    mkdir -p "$state_dir"

    if [[ -f "$config_path" ]]; then
        log_info "OpenClaw 配置已存在，保留: $config_path"
        local template_path="$state_dir/openclaw.offline.template.json5"
        write_openclaw_config_to "$template_path" "pairing" ""
        log_info "已写入最新离线模板: $template_path (如需升级配置，可自行对比/替换)"
        return 0
    fi

    log_step "写入 OpenClaw 离线配置..."

    local allow_from="${OPENCLAW_WHATSAPP_ALLOW_FROM:-}"
    if [[ -z "$allow_from" ]]; then
        echo ""
        echo -e "${CYAN}WhatsApp 访问控制:${NC}"
        echo -e "  - 推荐填写你自己的手机号 (E.164 格式，例如 +8613800138000)"
        echo -e "  - 留空则使用 pairing 模式 (更安全，稍后在 WhatsApp 内配对)"
        read -r -p "请输入 allowFrom (可留空): " allow_from
    fi

    local dm_policy="pairing"
    local allow_from_block=""
    if [[ -n "$allow_from" ]]; then
        dm_policy="allowlist"
        allow_from_block="allowFrom: [\"$allow_from\"],"
    fi

    write_openclaw_config_to "$config_path" "$dm_policy" "$allow_from_block"

    log_success "OpenClaw 配置已写入: $config_path"
}

# 如果用户机器上已经有旧版/不兼容的 openclaw.json，OpenClaw 会进入 best-effort 模式，
# 导致 WhatsApp/voice 等行为不可预期。这里做一次“自动备份 + 回退到最新离线模板”。
repair_openclaw_config_if_invalid() {
    local state_dir="$OPENCLAW_STATE_ROOT"
    local config_path="$state_dir/openclaw.json"
    local template_path="$state_dir/openclaw.offline.template.json5"

    if [[ ! -f "$config_path" ]]; then
        return 0
    fi
    if [[ ! -x "$BIN_DIR/openclaw" ]]; then
        return 0
    fi
    if [[ ! -f "$template_path" ]]; then
        write_openclaw_config_to "$template_path" "pairing" ""
    fi

    # Detect invalid config via the CLI startup banner.
    # Note: OpenClaw prints this message even when running `--help`.
    local out
    out="$(
        OPENCLAW_STATE_DIR="$state_dir" \
        OPENCLAW_CONFIG_PATH="$config_path" \
        OPENCLAW_OFFLINE=1 \
        LINGKONG_OFFLINE=1 \
        "$BIN_DIR/openclaw" --no-color --help 2>&1 || true
    )"
    if echo "$out" | grep -q "Invalid config at"; then
        local ts
        ts="$(date +%Y%m%d-%H%M%S)"
        cp "$config_path" "$config_path.bak.$ts" 2>/dev/null || true
        cp "$template_path" "$config_path"
        chmod 600 "$config_path" 2>/dev/null || true
        log_warn "检测到旧 OpenClaw 配置不兼容，已备份并替换为最新离线模板:"
        log_warn "  - 备份: $config_path.bak.$ts"
        log_warn "  - 新配置: $config_path"
    fi
}

# 迁移旧版 Moltbot/OpenClaw 的 WhatsApp 登录凭证到 LingKong 的 OpenClaw state，
# 避免升级后需要重新扫码绑定（best-effort，不会覆盖已有 creds）。
migrate_openclaw_whatsapp_creds() {
    local state_dir="$OPENCLAW_STATE_ROOT"
    local dst_root="$state_dir/credentials/whatsapp"
    local dst_creds="$dst_root/default/creds.json"

    if [[ -f "$dst_creds" ]]; then
        return 0
    fi

    is_valid_creds() {
        local creds_path="$1"
        if [[ ! -f "$creds_path" ]]; then
            return 1
        fi
        local size
        size="$(stat -f%z "$creds_path" 2>/dev/null || stat -c%s "$creds_path" 2>/dev/null || echo 0)"
        if [[ "$size" -lt 200 ]]; then
            return 1
        fi
        if command -v python3 >/dev/null 2>&1; then
            python3 - <<PY >/dev/null 2>&1 || return 1
import json, pathlib
json.loads(pathlib.Path("$creds_path").read_text(encoding="utf-8"))
PY
        fi
        return 0
    }

    local candidates=(
        "$HOME/.moltbot/credentials/whatsapp"
        "$HOME/.clawdbot/credentials/whatsapp"
        "$HOME/.openclaw/credentials/whatsapp"
        "$HOME/.moldbot/credentials/whatsapp"
    )

    for src_root in "${candidates[@]}"; do
        local src_creds="$src_root/default/creds.json"
        if ! is_valid_creds "$src_creds"; then
            continue
        fi
        log_step "迁移 WhatsApp 登录凭证（避免重新扫码）..."
        mkdir -p "$dst_root"
        # Copy contents of credentials/whatsapp (supports multi-account).
        cp -R "$src_root"/. "$dst_root"/
        chmod 700 "$state_dir" "$state_dir/credentials" "$dst_root" 2>/dev/null || true
        chmod 700 "$dst_root/default" 2>/dev/null || true
        chmod 600 "$dst_creds" 2>/dev/null || true
        log_success "已迁移 WhatsApp creds: $src_root -> $dst_root"
        return 0
    done
    return 0
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

    # Sandbox 模式当前仅保证文本能力：docker 里使用的上游 llama.cpp server
    # 不包含我们对 Gemma-3n 多模态 projector 的补丁，加载 mmproj 会导致容器退出并变为 unhealthy。
    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        log_info "Sandbox 模式当前仅下载文本模型 (视觉/音频请使用原生模式)"
        return 0
    fi

    # 视觉模型 - 570MB = 598999040 bytes
    download_file \
        "$MODELS_DIR/gemma-3n-vision-mmproj-f16.gguf" \
        "$VISION_URL" \
        "$VISION_URL_HF" \
        "视觉模型 (570MB)" \
        598999040

    # 音频模型 (可选) - 1.3GB = 1395864576 bytes
    if [[ "$DOWNLOAD_AUDIO_MMPROJ" == "1" ]]; then
        download_file \
            "$MODELS_DIR/gemma-3n-audio-mmproj-f16.gguf" \
            "$AUDIO_URL" \
            "$AUDIO_URL_HF" \
            "音频模型 (1.3GB)" \
            1395864576
    else
        log_info "跳过音频模型 (Gemma audio mmproj)。如需开启音频理解：LINGKONG_DOWNLOAD_AUDIO_MMPROJ=1 重新运行安装脚本"
    fi

    # 离线语音转写 (STT) 模型 - Whisper small ~465MB
    download_file \
        "$MODELS_DIR/whisper/ggml-small.bin" \
        "$WHISPER_URL" \
        "$WHISPER_URL_HF" \
        "语音转写模型 (Whisper small ~465MB)" \
        487601967
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
# 可选多模态开关（提高启动/推理速度时可关闭；0=关闭，1=开启）
# Default to vision-enabled so offline image understanding works out-of-the-box.
# Users can opt out for best latency by exporting LINGKONG_ENABLE_VISION_MMPROJ=0.
ENABLE_VISION_MMPROJ="${LINGKONG_ENABLE_VISION_MMPROJ:-1}"
ENABLE_AUDIO_MMPROJ="${LINGKONG_ENABLE_AUDIO_MMPROJ:-0}"
LLAMA_PORT="${LLAMA_PORT:-8081}"
WEBUI_PORT="${WEBUI_PORT:-8080}"
OPENCLAW_PORT="${OPENCLAW_PORT:-18789}"
PID_DIR="$LINGKONG_HOME/run"
LOG_DIR="$LINGKONG_HOME/logs"
OPENCLAW_APP_DIR="$LINGKONG_HOME/apps/openclaw"
OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR:-$LINGKONG_HOME/openclaw}"
OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$OPENCLAW_STATE_DIR/openclaw.json}"

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
    export LINGKONG_OFFLINE="${LINGKONG_OFFLINE:-1}"
    export LLAMA_SERVER_BIN="$LINGKONG_HOME/bin/llama-server"
    export LLAMA_MTMD_BIN="$LINGKONG_HOME/bin/llama-mtmd-cli"
    export LLAMA_MODEL="$MODEL"
    export LLAMA_MODEL_AUDIO="$MODEL"
    # 多模态支持: llama-server 只支持单个 mmproj，优先视觉
    # 音频通过 llama-mtmd-cli 单独处理
    if [[ "${ENABLE_VISION_MMPROJ:-1}" == "1" && -f "$VISION" ]]; then
        export LLAMA_MMPROJ_VISION="$VISION"
    else
        unset LLAMA_MMPROJ_VISION
    fi
    if [[ "${ENABLE_AUDIO_MMPROJ:-0}" == "1" && -f "$AUDIO" ]]; then
        export LLAMA_MMPROJ_AUDIO="$AUDIO"
    else
        unset LLAMA_MMPROJ_AUDIO
    fi
    export GEMINI_API_LLAMA_PORT="8090"
    # GPU 配置 (可通过环境变量覆盖)
    # LLAMA_GPU_DEVICES: 指定 GPU (例如 "0" 或 "0,1")，空=自动选择最大显存的 GPU
    # LLAMA_SPLIT_MODE: 多 GPU 分割模式 (none, layer, row)
    export LLAMA_GPU_DEVICES="${LLAMA_GPU_DEVICES:-}"
    export LLAMA_SPLIT_MODE="${LLAMA_SPLIT_MODE:-layer}"
    export DYLD_LIBRARY_PATH="$LINGKONG_HOME/lib:${DYLD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$LINGKONG_HOME/lib:${LD_LIBRARY_PATH:-}"

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
    export LLAMA_MMPROJ_SERVER_PORT="8090"
    export LLAMA_MM_MODEL="$MODEL"
    if [[ "${ENABLE_VISION_MMPROJ:-1}" == "1" && -f "$VISION" ]]; then
        export LLAMA_MM_PROJ_IMAGE="$VISION"
    else
        unset LLAMA_MM_PROJ_IMAGE
    fi
    if [[ "${ENABLE_AUDIO_MMPROJ:-0}" == "1" && -f "$AUDIO" ]]; then
        export LLAMA_MM_PROJ_AUDIO="$AUDIO"
    else
        unset LLAMA_MM_PROJ_AUDIO
    fi
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

# 启动 OpenClaw (WhatsApp agent gateway)
start_openclaw() {
    if [[ ! -f "$OPENCLAW_APP_DIR/openclaw.mjs" ]]; then
        echo -e "${YELLOW}[警告]${NC} OpenClaw 未安装，跳过"
        return 1
    fi

    if [[ ! -f "$OPENCLAW_CONFIG_PATH" ]]; then
        echo -e "${YELLOW}[警告]${NC} OpenClaw 配置不存在: $OPENCLAW_CONFIG_PATH"
        echo -e "${YELLOW}[提示]${NC} 可重新运行安装脚本，或手动创建 openclaw.json"
    fi

    if [[ -f "$PID_DIR/openclaw.pid" ]] && kill -0 "$(cat "$PID_DIR/openclaw.pid")" 2>/dev/null; then
        echo -e "${GREEN}[运行中]${NC} OpenClaw (PID: $(cat "$PID_DIR/openclaw.pid"))"
        return 0
    fi

    # If a previous moltbot/openclaw gateway is still bound, kill it to avoid port conflicts.
    if command -v lsof >/dev/null 2>&1; then
        if lsof -i :"$OPENCLAW_PORT" -t > /dev/null 2>&1; then
            echo -e "${YELLOW}[清理]${NC} OpenClaw 端口 $OPENCLAW_PORT"
            lsof -i :"$OPENCLAW_PORT" -t 2>/dev/null | xargs kill -9 2>/dev/null || true
            sleep 0.3
        fi
    fi

    # Offline-first guardrails: allow WhatsApp transport, block other outbound networking where possible.
    export OPENCLAW_STATE_DIR="$OPENCLAW_STATE_DIR"
    export OPENCLAW_CONFIG_PATH="$OPENCLAW_CONFIG_PATH"
    export OPENCLAW_OFFLINE="${OPENCLAW_OFFLINE:-1}"
    export LINGKONG_OFFLINE="${LINGKONG_OFFLINE:-1}"
    # Ensure bundled binaries (whisper-cli, ffmpeg, etc) are discoverable by OpenClaw sub-processes.
    export PATH="$LINGKONG_HOME/bin:${PATH:-}"
    # Offline STT: point OpenClaw media-understanding (whisper-cli) at our bundled model.
    # Default to zh for short Chinese voice notes; users can override via WHISPER_CPP_LANG.
    if [[ -f "$LINGKONG_HOME/models/whisper/ggml-small.bin" ]]; then
        export WHISPER_CPP_MODEL="$LINGKONG_HOME/models/whisper/ggml-small.bin"
        export WHISPER_CPP_LANG="${WHISPER_CPP_LANG:-zh}"
    fi
    # Default to the smallest prompt for low-latency voice assistant UX.
    # Users can override (e.g. export OPENCLAW_PROMPT_MODE=full).
    export OPENCLAW_PROMPT_MODE="${OPENCLAW_PROMPT_MODE:-none}"

    nohup "$LINGKONG_HOME/bin/openclaw" gateway run --port "$OPENCLAW_PORT" --force --allow-unconfigured > "$LOG_DIR/openclaw.log" 2>&1 &
    echo $! > "$PID_DIR/openclaw.pid"
    echo -e "${GREEN}[启动]${NC} OpenClaw (PID: $(cat "$PID_DIR/openclaw.pid"))"
}

stop_openclaw() {
    if [[ -f "$PID_DIR/openclaw.pid" ]]; then
        local pid
        pid=$(cat "$PID_DIR/openclaw.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo -e "${GREEN}[成功]${NC} 已停止 openclaw (PID: $pid)"
        fi
        rm -f "$PID_DIR/openclaw.pid"
    fi
}

# 停止服务
stop_all() {
    stop_openclaw
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

    if [[ -f "$PID_DIR/openclaw.pid" ]] && kill -0 "$(cat "$PID_DIR/openclaw.pid")" 2>/dev/null; then
        echo -e "  OpenClaw:   ${GREEN}● 运行中${NC} (PID: $(cat "$PID_DIR/openclaw.pid"))"
    else
        echo -e "  OpenClaw:   ${RED}○ 已停止${NC}"
    fi

    # WhatsApp link status (best-effort; requires OpenClaw gateway to be reachable).
    if [[ -x "$LINGKONG_HOME/bin/openclaw" ]]; then
        local wa_json
        wa_json="$("$LINGKONG_HOME/bin/openclaw" channels status --probe --timeout 5000 --json 2>/dev/null || true)"
        if [[ -n "$wa_json" ]] && command -v python3 >/dev/null 2>&1; then
            local wa_line
            wa_line="$(WA_JSON="$wa_json" python3 - <<'PY'
import json
import os
import sys

raw = os.environ.get("WA_JSON", "")
try:
    data = json.loads(raw)
except Exception:
    sys.exit(1)

wa = (data.get("channels") or {}).get("whatsapp") or {}
linked = bool(wa.get("linked"))
connected = bool(wa.get("connected"))
running = bool(wa.get("running"))
self_id = wa.get("self") or {}
who = self_id.get("e164") or self_id.get("jid") or ""
last = wa.get("lastError") or ""

status = "not linked"
if linked and connected:
    status = "connected"
elif linked and running:
    status = "linked"
elif linked:
    status = "linked (not running)"

suffix = f" ({who})" if who else ""
if not linked:
    print(f"○ 未绑定（运行: lingkong agent login）")
elif status == "connected":
    print(f"● 已连接{suffix}")
else:
    extra = f" ({last})" if last else ""
    print(f"○ 已绑定但未连接{suffix}{extra}")
PY
            )" || true
            if [[ -n "$wa_line" ]]; then
                echo -e "  WhatsApp:   $wa_line"
            fi
        elif [[ -n "$wa_json" ]]; then
            echo -e "  WhatsApp:   (运行: lingkong agent login)"
        fi
    fi

    echo ""
    echo -e "  ${CYAN}WebUI:${NC}      http://localhost:$WEBUI_PORT"
    echo -e "  ${CYAN}Playground:${NC} http://localhost:$WEBUI_PORT/static/playground.html"
    echo ""
}

# ================== macOS launchd (长期运行) ==================

LAUNCHD_LABEL_GEMINI="ai.lingkong.gemini-api"
LAUNCHD_LABEL_OPENCLAW="ai.lingkong.openclaw"

service_install() {
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "launchd 仅支持 macOS"
        return 1
    fi

    local plist_dir="$HOME/Library/LaunchAgents"
    mkdir -p "$plist_dir" "$LOG_DIR"

    local python_bin="$LINGKONG_HOME/venv/bin/python"
    if [[ ! -x "$python_bin" ]]; then
        python_bin="$(command -v python3 || true)"
    fi
    if [[ -z "$python_bin" ]]; then
        echo "未找到 python3（Gemini API 需要 Python 运行）"
        return 1
    fi

    local ffmpeg_bin
    ffmpeg_bin="$(command -v ffmpeg 2>/dev/null || true)"
    if [[ -z "$ffmpeg_bin" ]]; then
        ffmpeg_bin="$LINGKONG_HOME/bin/ffmpeg"
    fi

    # Helper: wait for Gemini API before starting OpenClaw.
    local openclaw_launchd="$LINGKONG_HOME/bin/openclaw-launchd"
    cat >"$openclaw_launchd" <<'OPENCLAW_LAUNCHD_SCRIPT'
#!/bin/bash
set -e

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
export PATH="$LINGKONG_HOME/bin:${PATH:-}"
if [[ -f "$LINGKONG_HOME/models/whisper/ggml-small.bin" ]]; then
  export WHISPER_CPP_MODEL="$LINGKONG_HOME/models/whisper/ggml-small.bin"
  export WHISPER_CPP_LANG="${WHISPER_CPP_LANG:-zh}"
fi
export OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR:-$LINGKONG_HOME/openclaw}"
export OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$OPENCLAW_STATE_DIR/openclaw.json}"
export OPENCLAW_OFFLINE="${OPENCLAW_OFFLINE:-1}"
export LINGKONG_OFFLINE="${LINGKONG_OFFLINE:-1}"
export OPENCLAW_PROMPT_MODE="${OPENCLAW_PROMPT_MODE:-none}"

for i in {1..120}; do
  if curl -s --connect-timeout 1 http://127.0.0.1:5001/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

exec "$LINGKONG_HOME/bin/openclaw" gateway run --force --allow-unconfigured
OPENCLAW_LAUNCHD_SCRIPT
    chmod +x "$openclaw_launchd"

    local vision_mmproj="disabled"
    if [[ "${ENABLE_VISION_MMPROJ:-1}" == "1" && -f "$VISION" ]]; then
        vision_mmproj="$VISION"
    fi
    local audio_mmproj="disabled"
    if [[ "${ENABLE_AUDIO_MMPROJ:-0}" == "1" && -f "$AUDIO" ]]; then
        audio_mmproj="$AUDIO"
    fi

    local gemini_plist="$plist_dir/$LAUNCHD_LABEL_GEMINI.plist"
    cat >"$gemini_plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>$LAUNCHD_LABEL_GEMINI</string>
    <key>ProgramArguments</key>
    <array>
      <string>$python_bin</string>
      <string>server.py</string>
    </array>
    <key>WorkingDirectory</key><string>$LINGKONG_HOME/apps/gemini_api</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>LINGKONG_OFFLINE</key><string>1</string>
      <key>OPENCLAW_OFFLINE</key><string>1</string>
      <key>GEMINI_API_PORT</key><string>5001</string>
      <key>GEMINI_API_LLAMA_PORT</key><string>8090</string>
      <key>LLAMA_SERVER_BIN</key><string>$LINGKONG_HOME/bin/llama-server</string>
      <key>LLAMA_MTMD_BIN</key><string>$LINGKONG_HOME/bin/llama-mtmd-cli</string>
      <key>LLAMA_MODEL</key><string>$MODEL</string>
      <key>LLAMA_MODEL_AUDIO</key><string>$MODEL</string>
      <key>LLAMA_MMPROJ_VISION</key><string>$vision_mmproj</string>
      <key>LLAMA_MMPROJ_AUDIO</key><string>$audio_mmproj</string>
      <key>DYLD_LIBRARY_PATH</key><string>$LINGKONG_HOME/lib</string>
      <key>FFMPEG_BIN</key><string>$ffmpeg_bin</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>5</integer>
    <key>StandardOutPath</key><string>$LOG_DIR/gemini.launchd.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/gemini.launchd.err.log</string>
  </dict>
</plist>
EOF

    local openclaw_plist="$plist_dir/$LAUNCHD_LABEL_OPENCLAW.plist"
    cat >"$openclaw_plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>$LAUNCHD_LABEL_OPENCLAW</string>
    <key>ProgramArguments</key>
    <array>
      <string>$openclaw_launchd</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>5</integer>
    <key>StandardOutPath</key><string>$LOG_DIR/openclaw.launchd.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/openclaw.launchd.err.log</string>
  </dict>
</plist>
EOF

    local domain="gui/$UID"
    launchctl bootout "$domain" "$gemini_plist" 2>/dev/null || true
    launchctl bootout "$domain" "$openclaw_plist" 2>/dev/null || true
    launchctl bootstrap "$domain" "$gemini_plist"
    launchctl bootstrap "$domain" "$openclaw_plist"
    launchctl enable "$domain/$LAUNCHD_LABEL_GEMINI" 2>/dev/null || true
    launchctl enable "$domain/$LAUNCHD_LABEL_OPENCLAW" 2>/dev/null || true
    launchctl kickstart -k "$domain/$LAUNCHD_LABEL_GEMINI" 2>/dev/null || true
    launchctl kickstart -k "$domain/$LAUNCHD_LABEL_OPENCLAW" 2>/dev/null || true

    echo -e "${GREEN}[成功]${NC} launchd 已安装并启动:"
    echo "  - $LAUNCHD_LABEL_GEMINI"
    echo "  - $LAUNCHD_LABEL_OPENCLAW"
}

service_uninstall() {
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "launchd 仅支持 macOS"
        return 1
    fi

    local plist_dir="$HOME/Library/LaunchAgents"
    local gemini_plist="$plist_dir/$LAUNCHD_LABEL_GEMINI.plist"
    local openclaw_plist="$plist_dir/$LAUNCHD_LABEL_OPENCLAW.plist"
    local domain="gui/$UID"

    launchctl bootout "$domain" "$gemini_plist" 2>/dev/null || true
    launchctl bootout "$domain" "$openclaw_plist" 2>/dev/null || true

    rm -f "$gemini_plist" "$openclaw_plist" "$LINGKONG_HOME/bin/openclaw-launchd" 2>/dev/null || true

    echo -e "${GREEN}[成功]${NC} launchd 已卸载"
}

service_status() {
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "launchd 仅支持 macOS"
        return 1
    fi

    local domain="gui/$UID"
    for label in "$LAUNCHD_LABEL_GEMINI" "$LAUNCHD_LABEL_OPENCLAW"; do
        if launchctl print "$domain/$label" >/dev/null 2>&1; then
            echo -e "  ${GREEN}●${NC} $label"
        else
            echo -e "  ${RED}○${NC} $label"
        fi
    done

    echo ""
    echo "日志:"
    echo "  $LOG_DIR/gemini.launchd.log"
    echo "  $LOG_DIR/openclaw.launchd.log"
}

# 主函数
case "${1:-start}" in
    start|up)
        echo ""
        echo -e "${CYAN}🐉 启动 灵空 AI...${NC}"
        echo ""
        start_gemini_api
        start_webui
        start_openclaw || true
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
        tail -f "$LOG_DIR/webui.log" "$LOG_DIR/gemini.log" "$LOG_DIR/openclaw.log" 2>/dev/null || \
          tail -f "$LOG_DIR/webui.log" "$LOG_DIR/gemini.log"
        ;;
    agent)
        shift || true
        case "${1:-status}" in
            start|up)
                start_openclaw
                ;;
            stop|down)
                stop_openclaw
                ;;
            status|ps)
                if [[ -f "$PID_DIR/openclaw.pid" ]] && kill -0 "$(cat "$PID_DIR/openclaw.pid")" 2>/dev/null; then
                    echo -e "${GREEN}[运行中]${NC} OpenClaw (PID: $(cat "$PID_DIR/openclaw.pid"))"
                else
                    echo -e "${RED}[已停止]${NC} OpenClaw"
                fi
                ;;
            logs)
                tail -f "$LOG_DIR/openclaw.log"
                ;;
            login)
                if [[ -x "$LINGKONG_HOME/bin/lingkong-whatsapp-login" ]]; then
                    "$LINGKONG_HOME/bin/lingkong-whatsapp-login"
                else
                    "$LINGKONG_HOME/bin/openclaw" channels login --channel whatsapp --verbose
                fi
                ;;
            *)
                echo "使用方法: lingkong agent [start|stop|status|logs|login]"
                ;;
        esac
        ;;
    service)
        shift || true
        case "${1:-status}" in
            install)
                service_install
                ;;
            uninstall)
                service_uninstall
                ;;
            status)
                service_status
                ;;
            *)
                echo "使用方法: lingkong service [install|uninstall|status]"
                ;;
        esac
        ;;
    update|upgrade)
        echo -e "${CYAN}🔄 检查更新...${NC}"
        check_update
        ;;
    *)
        echo "使用方法: lingkong [start|stop|restart|status|logs|update|agent|service]"
        ;;
esac
SCRIPT

    chmod +x "$BIN_DIR/lingkong"

    # OpenClaw wrapper (runs the bundled OpenClaw CLI with LingKong defaults)
    cat > "$BIN_DIR/openclaw" << 'SCRIPT'
#!/bin/bash
set -e

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
OPENCLAW_APP_DIR="${OPENCLAW_APP_DIR:-$LINGKONG_HOME/apps/openclaw}"
STATE_DIR="${OPENCLAW_STATE_DIR:-$LINGKONG_HOME/openclaw}"

export OPENCLAW_STATE_DIR="$STATE_DIR"
export OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$STATE_DIR/openclaw.json}"
export OPENCLAW_OFFLINE="${OPENCLAW_OFFLINE:-1}"
export LINGKONG_OFFLINE="${LINGKONG_OFFLINE:-1}"
export OPENCLAW_PROMPT_MODE="${OPENCLAW_PROMPT_MODE:-none}"

# Gateway token is required even for loopback-only deployments.
# Keep it on disk so launchd/background runs can reconnect consistently.
token_file="$STATE_DIR/gateway.token"
chmod 700 "$STATE_DIR" 2>/dev/null || true
if [[ -z "${OPENCLAW_GATEWAY_TOKEN:-}" ]]; then
  if [[ -f "$token_file" ]]; then
    export OPENCLAW_GATEWAY_TOKEN="$(tr -d '\r\n' < "$token_file" | head -c 200)"
  else
    token=""
    if command -v openssl >/dev/null 2>&1; then
      umask 077
      token="$(openssl rand -hex 16)"
    elif command -v uuidgen >/dev/null 2>&1; then
      umask 077
      token="$(uuidgen | tr -d '-' | tr '[:upper:]' '[:lower:]')"
    fi
    if [[ -n "$token" ]]; then
      printf "%s" "$token" > "$token_file"
      chmod 600 "$token_file" 2>/dev/null || true
      export OPENCLAW_GATEWAY_TOKEN="$token"
    fi
  fi
fi
export WHISPER_CPP_MODEL="${WHISPER_CPP_MODEL:-$LINGKONG_HOME/models/whisper/ggml-small.bin}"
export WHISPER_CPP_LANG="${WHISPER_CPP_LANG:-zh}"

# Whisper performance tuning (safe default): cap threads to avoid CPU saturation.
if [[ -z "${WHISPER_CPP_THREADS:-}" ]]; then
  ncpu=""
  if command -v sysctl >/dev/null 2>&1; then
    ncpu="$(sysctl -n hw.ncpu 2>/dev/null || true)"
  elif command -v nproc >/dev/null 2>&1; then
    ncpu="$(nproc 2>/dev/null || true)"
  fi
  if [[ "$ncpu" =~ ^[0-9]+$ ]]; then
    if [[ "$ncpu" -gt 8 ]]; then ncpu=8; fi
    if [[ "$ncpu" -lt 1 ]]; then ncpu=1; fi
    export WHISPER_CPP_THREADS="$ncpu"
  fi
fi
export PATH="$LINGKONG_HOME/bin:${PATH:-}"

if [[ ! -f "$OPENCLAW_APP_DIR/openclaw.mjs" ]]; then
  echo "[openclaw] missing runtime at: $OPENCLAW_APP_DIR/openclaw.mjs" >&2
  echo "[openclaw] re-run installer to fetch the OpenClaw bundle." >&2
  exit 1
fi

NODE_BIN="$(command -v node || true)"
if [[ -z "$NODE_BIN" && -x "$LINGKONG_HOME/node/bin/node" ]]; then
  NODE_BIN="$LINGKONG_HOME/node/bin/node"
fi
if [[ -z "$NODE_BIN" ]]; then
  echo "[openclaw] node not found. Install Node.js, or re-run installer to fetch a runtime." >&2
  exit 1
fi

exec "$NODE_BIN" "$OPENCLAW_APP_DIR/openclaw.mjs" "$@"
SCRIPT

    chmod +x "$BIN_DIR/openclaw"

    # WhatsApp login helper: generate a QR PNG via the local Gateway method (web.login.start),
    # then wait for the scan (web.login.wait). Falls back to ASCII QR via `openclaw channels login`.
    cat > "$BIN_DIR/lingkong-whatsapp-login" << 'SCRIPT'
#!/bin/bash
set -e

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
export PATH="$LINGKONG_HOME/bin:${PATH:-}"

# Ensure OpenClaw gateway is running (required for gateway call).
if ! "$LINGKONG_HOME/bin/openclaw" gateway health --json >/dev/null 2>&1; then
  if [[ -x "$LINGKONG_HOME/bin/lingkong" ]]; then
    "$LINGKONG_HOME/bin/lingkong" agent start >/dev/null 2>&1 || true
  fi
  # Give the gateway a moment to bind before we try web.login.* calls.
  for _ in {1..20}; do
    if "$LINGKONG_HOME/bin/openclaw" gateway health --json >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

python_bin=""
if [[ -x "$LINGKONG_HOME/venv/bin/python" ]]; then
  python_bin="$LINGKONG_HOME/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
fi

tmp_dir="$LINGKONG_HOME/tmp"
mkdir -p "$tmp_dir"
qr_json="$tmp_dir/whatsapp-qr.json"
qr_png="$tmp_dir/whatsapp-qr.png"
wait_json="$tmp_dir/whatsapp-wait.json"

if [[ -z "$python_bin" ]]; then
  echo "[lingkong] python3 not found; falling back to terminal QR login." >&2
  exec "$LINGKONG_HOME/bin/openclaw" channels login --channel whatsapp --verbose
fi

if ! "$LINGKONG_HOME/bin/openclaw" gateway call web.login.start --timeout 70000 --params "{\"timeoutMs\":60000,\"force\":true}" --json > "$qr_json" 2>/dev/null; then
  echo "[lingkong] gateway QR login failed; falling back to terminal QR login." >&2
  exec "$LINGKONG_HOME/bin/openclaw" channels login --channel whatsapp --verbose
fi

if ! QR_JSON="$qr_json" QR_PNG="$qr_png" "$python_bin" - <<'PY'
import base64
import json
import os
import pathlib
import sys

qr_json = os.environ["QR_JSON"]
qr_png = os.environ["QR_PNG"]

with open(qr_json, "r", encoding="utf-8") as f:
    data = json.load(f)

url = data.get("qrDataUrl")
if not url and isinstance(data.get("result"), dict):
    url = data["result"].get("qrDataUrl")

prefix = "data:image/png;base64,"
if not isinstance(url, str) or not url.startswith(prefix):
    print("missing qrDataUrl in gateway response", file=sys.stderr)
    sys.exit(2)

payload = base64.b64decode(url[len(prefix) :])
pathlib.Path(qr_png).write_bytes(payload)
print(qr_png)
PY
then
  echo "[lingkong] failed to write QR PNG; falling back to terminal QR login." >&2
  exec "$LINGKONG_HOME/bin/openclaw" channels login --channel whatsapp --verbose
fi

echo ""
echo "[lingkong] WhatsApp QR saved to: $qr_png"
echo "[lingkong] Scan it in WhatsApp → Settings → Linked Devices."
echo ""

if [[ "$(uname)" == "Darwin" ]]; then
  open "$qr_png" 2>/dev/null || true
fi

if "$LINGKONG_HOME/bin/openclaw" gateway call web.login.wait --timeout 130000 --params "{\"timeoutMs\":120000}" --json > "$wait_json" 2>/dev/null; then
  WAIT_JSON="$wait_json" "$python_bin" - <<'PY'
import json
import os

with open(os.environ["WAIT_JSON"], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("message", ""))
PY
else
  echo "[lingkong] timed out waiting for scan. Re-run: lingkong agent login" >&2
fi

# Print a quick status hint (non-fatal).
"$LINGKONG_HOME/bin/openclaw" channels status --probe >/dev/null 2>&1 || true
SCRIPT

    chmod +x "$BIN_DIR/lingkong-whatsapp-login"

    # Voice benchmark helper: local STT (whisper-cli) -> local Gemini -> local TTS (say)
    cat > "$BIN_DIR/lingkong-voice-bench" << 'SCRIPT'
#!/bin/bash
set -euo pipefail

LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"
GEMINI_BASE_URL="${GEMINI_BASE_URL:-http://127.0.0.1:5001}"
MODEL_ID="${MODEL_ID:-gemini-3-pro-preview}"

WHISPER_CPP_MODEL="${WHISPER_CPP_MODEL:-$LINGKONG_HOME/models/whisper/ggml-small.bin}"
WHISPER_CPP_LANG="${WHISPER_CPP_LANG:-zh}"
WHISPER_CPP_THREADS="${WHISPER_CPP_THREADS:-}"

TEXT="${1:-What day is it today?}"
NOW="$(date '+%F %A %T')"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$LINGKONG_HOME/venv/bin/python" ]]; then
    PYTHON_BIN="$LINGKONG_HOME/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  fi
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[voice-bench] python3 is required for JSON parsing (Gemini API response)" >&2
  exit 1
fi

now_ms() {
  "$PYTHON_BIN" - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

WHISPER_BIN="${WHISPER_CLI_BIN:-}"
if [[ -z "$WHISPER_BIN" ]]; then
  if [[ -x "$LINGKONG_HOME/bin/whisper-cli" ]]; then
    WHISPER_BIN="$LINGKONG_HOME/bin/whisper-cli"
  else
    WHISPER_BIN="$(command -v whisper-cli || true)"
  fi
fi
if [[ -z "$WHISPER_BIN" || ! -x "$WHISPER_BIN" ]]; then
  echo "[voice-bench] whisper-cli not found. Re-run installer to install it." >&2
  exit 1
fi
if [[ ! -f "$WHISPER_CPP_MODEL" ]]; then
  echo "[voice-bench] missing WHISPER_CPP_MODEL: $WHISPER_CPP_MODEL" >&2
  echo "[voice-bench] Re-run installer to download whisper model, or set WHISPER_CPP_MODEL." >&2
  exit 1
fi

if [[ ! -x /usr/bin/say || ! -x /usr/bin/afconvert ]]; then
  echo "[voice-bench] macOS TTS prerequisites missing: /usr/bin/say or /usr/bin/afconvert" >&2
  exit 1
fi

if ! curl -fsS "$GEMINI_BASE_URL/health" >/dev/null 2>&1; then
  echo "[voice-bench] Gemini API not running at: $GEMINI_BASE_URL" >&2
  echo "[voice-bench] Try: $LINGKONG_HOME/bin/lingkong start" >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/lingkong-voice-bench.XXXXXX")"
cleanup() { rm -rf "$tmp_dir" >/dev/null 2>&1 || true; }
trap cleanup EXIT

in_aiff="$tmp_dir/in.aiff"
in_wav="$tmp_dir/in.wav"
stt_out="$tmp_dir/stt"
req_json="$tmp_dir/req.json"
resp_json="$tmp_dir/resp.json"
out_aiff="$tmp_dir/out.aiff"
out_ogg="$tmp_dir/out.ogg"
out_m4a="$tmp_dir/out.m4a"

echo "[voice-bench] input: $TEXT"

t0="$(now_ms)"
say -o "$in_aiff" "$TEXT"
t1="$(now_ms)"
afconvert -f WAVE -d LEI16 -c 1 -r 16000 "$in_aiff" "$in_wav"
t2="$(now_ms)"

stt_args=()
if [[ "${WHISPER_CPP_THREADS:-}" =~ ^[0-9]+$ ]]; then
  stt_args+=("-t" "$WHISPER_CPP_THREADS")
fi
stt_args+=("-m" "$WHISPER_CPP_MODEL" "-l" "$WHISPER_CPP_LANG" "-otxt" "-of" "$stt_out" "-np" "-nt" "$in_wav")

"$WHISPER_BIN" "${stt_args[@]}" >/dev/null
t3="$(now_ms)"

transcript=""
if [[ -f "${stt_out}.txt" ]]; then
  transcript="$(cat "${stt_out}.txt" | tr -d '\r' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
fi
if [[ -z "$transcript" ]]; then
  echo "[voice-bench] STT produced empty transcript" >&2
  exit 1
fi

echo "[voice-bench] transcript: $transcript"

cat >"$req_json" <<EOF
{
  "contents": [
    {
      "role": "user",
      "parts": [
        { "text": "你是语音助手。只输出最终答案：言简意赅，不要解释，不要带前缀，不要换行。当前系统时间：${NOW}。用户说：${transcript}" }
      ]
    }
  ],
  "generationConfig": {
    "maxOutputTokens": 96,
    "temperature": 0.2,
    "thinkingConfig": { "thinkingLevel": "none", "includeThoughts": false }
  }
}
EOF

curl -fsS "$GEMINI_BASE_URL/v1beta/models/$MODEL_ID:generateContent" \
  -H 'Content-Type: application/json' \
  -d "@$req_json" >"$resp_json"
t4="$(now_ms)"

reply="$("$PYTHON_BIN" - "$resp_json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    j = json.load(f)
cands = j.get("candidates") or []
content = (cands[0].get("content") if cands else {}) or {}
parts = content.get("parts") or []
out = []
for p in parts:
    if isinstance(p, dict) and isinstance(p.get("text"), str):
        if p.get("thought") is True:
            continue
        out.append(p["text"])
print("".join(out).strip())
PY
)"
if [[ -z "$reply" ]]; then
  echo "[voice-bench] empty model reply" >&2
  exit 1
fi

echo "[voice-bench] reply: $reply"

say -o "$out_aiff" "$reply"
t5="$(now_ms)"

out_audio="$out_m4a"
if command -v ffmpeg >/dev/null 2>&1; then
  if ffmpeg -hide_banner -loglevel error -y -i "$out_aiff" -ac 1 -c:a libopus -b:a 24k "$out_ogg" >/dev/null 2>&1; then
    out_audio="$out_ogg"
  fi
fi
if [[ "$out_audio" == "$out_m4a" ]]; then
  afconvert -f m4af -d aac "$out_aiff" "$out_m4a" >/dev/null 2>&1 || true
fi
t6="$(now_ms)"

echo "[voice-bench] output audio: $out_audio"
echo "[voice-bench] timing(ms): say_in=$((t1-t0)) convert_in=$((t2-t1)) stt=$((t3-t2)) llm=$((t4-t3)) say_out=$((t5-t4)) encode_out=$((t6-t5)) total=$((t6-t0))"
SCRIPT

    chmod +x "$BIN_DIR/lingkong-voice-bench"
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
# 灵空 AI Sandbox - Docker Compose (文本 + Gemini API)
#
# 说明:
# - Docker Sandbox 当前仅保证文本能力；图像/音频多模态请使用原生模式（macOS 可用 Metal 加速）。
# - Gemini API 容器默认映射到宿主机 8080（容器内是 5001）。

services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server
    container_name: lingkong-llama
    restart: unless-stopped
    ports:
      # 可选：暴露原始 llama-server，便于调试
      - "8081:8080"
    volumes:
      - ${LINGKONG_HOME:-~/.lingkong}/models:/models:ro
    command: >
      --model /models/gemma-3n-E2B-it-Q4_K_M.gguf
      --host 0.0.0.0
      --port 8080
      -c 8192
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 300s

  gemini-api:
    image: python:3.11-slim
    container_name: lingkong-gemini
    restart: unless-stopped
    ports:
      # Gemini API 服务器固定监听 5001，通过端口映射对外提供 8080
      - "8080:5001"
    volumes:
      - ${LINGKONG_HOME:-~/.lingkong}/apps:/app:ro
      - ${LINGKONG_HOME:-~/.lingkong}/models:/models:ro
    working_dir: /app/gemini_api
    environment:
      - LLAMA_SERVER_HOST=llama-server
      - LLAMA_SERVER_PORT=8080
      - GEMINI_API_LLAMA_PORT=8080
      - LLAMA_MODEL=/models/gemma-3n-E2B-it-Q4_K_M.gguf
      - LLAMA_MODEL_AUDIO=/models/gemma-3n-E2B-it-Q4_K_M.gguf
    command: >
      bash -c "pip install flask flask-cors requests psutil -q && python server.py"
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
        echo "  Gemini API:  http://localhost:8080"
        echo "  llama-server: http://localhost:8081 (可选)"
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
    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        echo -e "  功能: 文本对话 + Gemini API (图像/音频/会话记忆请使用原生模式)"
    else
        echo -e "  功能: 文本对话 + 图像理解 + 音频转录 + 会话记忆 + Gemini API"
    fi
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
        # 注意: 安装/更新会覆盖 ~/.lingkong/apps 下的代码文件，但运行中的进程不会自动加载新代码。
        # 这里使用 restart，避免“已更新但仍按旧版本(如 max_tokens=128)运行”的困惑。
        "$BIN_DIR/lingkong" restart &
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
        # 支持原生安装的平台：优先原生（更快/支持多模态；macOS 可用 Metal）
        if [[ "$PLATFORM" == "macos-arm64" ]] || [[ "$PLATFORM" == "linux-x64" ]]; then
            log_info "本平台支持原生模式，默认使用原生安装 (推荐)"
            log_info "如需 Docker Sandbox: curl -fsSL $LINGKONG_SERVER/install.sh | bash -s sandbox"
            INSTALL_MODE="native"
        elif [[ "$DOCKER_AVAILABLE" == "true" ]]; then
            log_info "检测到 Docker，使用 Sandbox 模式"
            INSTALL_MODE="sandbox"
        else
            log_info "此平台需要 Sandbox 模式，请先安装并启动 Docker"
            INSTALL_MODE="sandbox"
        fi
    fi

    create_directories
    download_models

    if [[ "$INSTALL_MODE" == "sandbox" ]]; then
        # Sandbox 仍需要 Gemini API 代码（运行在容器里，通过挂载 /app 提供）
        download_webui
        install_sandbox
    else
        install_native_binaries
        download_webui
        download_openclaw
        download_whisper_cli || true
        install_node_runtime || true
        write_openclaw_config
        migrate_openclaw_whatsapp_creds || true
        detect_python || install_python
        install_python_deps
        create_native_scripts
        repair_openclaw_config_if_invalid || true
    fi

    setup_path
    show_completion
    start_service
}

main "$@"

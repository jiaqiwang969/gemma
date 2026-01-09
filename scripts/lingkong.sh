#!/bin/bash
# =============================================================================
# 灵空 AI - 一键启动脚本
# =============================================================================
# 使用方法:
#   ./scripts/lingkong.sh           # 启动所有服务
#   ./scripts/lingkong.sh stop      # 停止所有服务
#   ./scripts/lingkong.sh status    # 查看状态
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LINGKONG_HOME="${LINGKONG_HOME:-$HOME/.lingkong}"

# 模型路径
MODEL="$LINGKONG_HOME/models/gemma-3n-E2B-it-Q4_K_M.gguf"
VISION="$LINGKONG_HOME/models/gemma-3n-vision-mmproj-f16.gguf"
AUDIO="$LINGKONG_HOME/models/gemma-3n-audio-mmproj-f16.gguf"

# 二进制路径
LLAMA_SERVER="$LINGKONG_HOME/bin/llama-server"

# 端口
LLAMA_PORT=8081
WEBUI_PORT=5001

# PID 文件
PID_DIR="$LINGKONG_HOME/run"
LLAMA_PID_FILE="$PID_DIR/llama-server.pid"
WEBUI_PID_FILE="$PID_DIR/webui.pid"

# 日志函数
log_info() { echo -e "${BLUE}[信息]${NC} $1"; }
log_success() { echo -e "${GREEN}[成功]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[警告]${NC} $1"; }
log_error() { echo -e "${RED}[错误]${NC} $1"; }

# 显示 Banner
show_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}     🐉 ${PURPLE}灵空 AI${NC} - 本地三模态人工智能                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}     文本对话 • 图像理解 • 音频转录                           ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 检查依赖
check_requirements() {
    # 检查模型
    if [[ ! -f "$MODEL" ]]; then
        log_error "模型不存在: $MODEL"
        log_info "请先运行安装脚本: curl -fsSL http://115.159.223.227/install.sh | bash"
        exit 1
    fi

    # 检查 llama-server
    if [[ ! -f "$LLAMA_SERVER" ]]; then
        log_error "llama-server 不存在: $LLAMA_SERVER"
        exit 1
    fi

    # macOS: 检查并签名二进制 (防止被 Gatekeeper 杀掉)
    if [[ "$(uname)" == "Darwin" ]]; then
        if ! codesign -v "$LLAMA_SERVER" 2>/dev/null; then
            log_info "签名 llama-server..."
            codesign -s - --force "$LLAMA_SERVER" 2>/dev/null || true
            codesign -s - --force "$LINGKONG_HOME/bin/llama-mtmd-cli" 2>/dev/null || true
        fi
    fi

    # 创建 PID 和日志目录
    mkdir -p "$PID_DIR" "$LINGKONG_HOME/logs"
}

# 启动 llama-server
start_llama_server() {
    log_info "启动推理引擎..."

    # 检查是否已运行
    if [[ -f "$LLAMA_PID_FILE" ]] && kill -0 "$(cat "$LLAMA_PID_FILE")" 2>/dev/null; then
        log_warn "llama-server 已在运行 (PID: $(cat "$LLAMA_PID_FILE"))"
        return 0
    fi

    # 构建 mmproj 列表
    MMPROJ_LIST=""
    if [[ -f "$VISION" ]]; then
        MMPROJ_LIST="$VISION"
        log_info "  视觉模型: ✓"
    fi
    if [[ -f "$AUDIO" ]]; then
        if [[ -n "$MMPROJ_LIST" ]]; then
            MMPROJ_LIST="$MMPROJ_LIST,$AUDIO"
        else
            MMPROJ_LIST="$AUDIO"
        fi
        log_info "  音频模型: ✓"
    fi

    # 设置动态库路径
    export DYLD_LIBRARY_PATH="$LINGKONG_HOME/lib:${DYLD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$LINGKONG_HOME/lib:${LD_LIBRARY_PATH:-}"

    # 启动命令
    LLAMA_CMD="$LLAMA_SERVER --model $MODEL --port $LLAMA_PORT --host 0.0.0.0 -ngl 99 --flash-attn on -c 8192"
    if [[ -n "$MMPROJ_LIST" ]]; then
        LLAMA_CMD="$LLAMA_CMD --mmproj $MMPROJ_LIST"
    fi

    # 后台启动
    nohup $LLAMA_CMD > "$LINGKONG_HOME/logs/llama-server.log" 2>&1 &
    echo $! > "$LLAMA_PID_FILE"

    log_success "推理引擎已启动 (PID: $(cat "$LLAMA_PID_FILE"))"

    # 等待就绪
    log_info "等待引擎就绪..."
    for i in {1..30}; do
        if curl -s "http://localhost:$LLAMA_PORT/health" > /dev/null 2>&1; then
            log_success "引擎就绪!"
            return 0
        fi
        sleep 1
    done
    log_warn "引擎启动超时，但可能仍在加载模型..."
}

# 启动 WebUI
start_webui() {
    log_info "启动 WebUI..."

    # 检查是否已运行
    if [[ -f "$WEBUI_PID_FILE" ]] && kill -0 "$(cat "$WEBUI_PID_FILE")" 2>/dev/null; then
        log_warn "WebUI 已在运行 (PID: $(cat "$WEBUI_PID_FILE"))"
        return 0
    fi

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi

    # 设置环境变量
    export LLAMA_SERVER_URL="http://localhost:$LLAMA_PORT"
    export LLAMA_MM_MODEL="$MODEL"
    export LLAMA_MM_PROJ_IMAGE="$VISION"
    export LLAMA_MM_PROJ_AUDIO="$AUDIO"
    export LLAMA_MTMD_BIN="$LINGKONG_HOME/bin/llama-mtmd-cli"
    export DYLD_LIBRARY_PATH="$LINGKONG_HOME/lib:${DYLD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$LINGKONG_HOME/lib:${LD_LIBRARY_PATH:-}"
    export WEBUI_PORT="$WEBUI_PORT"

    # 启动 WebUI
    cd "$PROJECT_DIR/apps/webui"
    nohup python3 server.py > "$LINGKONG_HOME/logs/webui.log" 2>&1 &
    echo $! > "$WEBUI_PID_FILE"

    log_success "WebUI 已启动 (PID: $(cat "$WEBUI_PID_FILE"))"

    # 等待就绪
    for i in {1..10}; do
        if curl -s "http://localhost:$WEBUI_PORT/api/status" > /dev/null 2>&1; then
            log_success "WebUI 就绪!"
            return 0
        fi
        sleep 1
    done
}

# 停止服务
stop_services() {
    log_info "停止所有服务..."

    # 停止 WebUI
    if [[ -f "$WEBUI_PID_FILE" ]]; then
        if kill -0 "$(cat "$WEBUI_PID_FILE")" 2>/dev/null; then
            kill "$(cat "$WEBUI_PID_FILE")" 2>/dev/null || true
            log_success "WebUI 已停止"
        fi
        rm -f "$WEBUI_PID_FILE"
    fi

    # 停止 llama-server
    if [[ -f "$LLAMA_PID_FILE" ]]; then
        if kill -0 "$(cat "$LLAMA_PID_FILE")" 2>/dev/null; then
            kill "$(cat "$LLAMA_PID_FILE")" 2>/dev/null || true
            log_success "推理引擎已停止"
        fi
        rm -f "$LLAMA_PID_FILE"
    fi

    # 额外清理
    pkill -f "llama-server.*$LLAMA_PORT" 2>/dev/null || true
    pkill -f "python.*server.py.*$WEBUI_PORT" 2>/dev/null || true

    log_success "所有服务已停止"
}

# 查看状态
show_status() {
    echo ""
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  灵空 AI 服务状态${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════${NC}"

    # llama-server
    if [[ -f "$LLAMA_PID_FILE" ]] && kill -0 "$(cat "$LLAMA_PID_FILE")" 2>/dev/null; then
        echo -e "  推理引擎:  ${GREEN}● 运行中${NC} (PID: $(cat "$LLAMA_PID_FILE"))"
    else
        echo -e "  推理引擎:  ${RED}○ 已停止${NC}"
    fi

    # WebUI
    if [[ -f "$WEBUI_PID_FILE" ]] && kill -0 "$(cat "$WEBUI_PID_FILE")" 2>/dev/null; then
        echo -e "  WebUI:     ${GREEN}● 运行中${NC} (PID: $(cat "$WEBUI_PID_FILE"))"
    else
        echo -e "  WebUI:     ${RED}○ 已停止${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}WebUI 地址:${NC} http://localhost:$WEBUI_PORT"
    echo -e "  ${CYAN}API 地址:${NC}   http://localhost:$LLAMA_PORT"
    echo ""
}

# 显示完成信息
show_completion() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ 灵空 AI 已启动!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  🌐 ${CYAN}WebUI:${NC}     ${YELLOW}http://localhost:$WEBUI_PORT${NC}"
    echo -e "  🔌 ${CYAN}API:${NC}       ${YELLOW}http://localhost:$LLAMA_PORT/v1${NC}"
    echo ""
    echo -e "  ${CYAN}功能:${NC} 文本对话 • 图像理解 • 音频转录"
    echo ""
    echo -e "  ${CYAN}停止服务:${NC} ./scripts/lingkong.sh stop"
    echo -e "  ${CYAN}查看日志:${NC} tail -f ~/.lingkong/logs/webui.log"
    echo ""
}

# 打开浏览器
open_browser() {
    if [[ "$(uname)" == "Darwin" ]]; then
        open "http://localhost:$WEBUI_PORT" 2>/dev/null || true
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$WEBUI_PORT" 2>/dev/null || true
    fi
}

# 主函数
main() {
    case "${1:-start}" in
        start|up)
            show_banner
            check_requirements
            start_llama_server
            start_webui
            show_completion
            open_browser
            ;;
        stop|down)
            show_banner
            stop_services
            ;;
        restart)
            show_banner
            stop_services
            sleep 2
            check_requirements
            start_llama_server
            start_webui
            show_completion
            ;;
        status|ps)
            show_banner
            show_status
            ;;
        logs)
            tail -f "$LINGKONG_HOME/logs/webui.log" "$LINGKONG_HOME/logs/llama-server.log"
            ;;
        *)
            echo "使用方法: $0 [start|stop|restart|status|logs]"
            ;;
    esac
}

main "$@"

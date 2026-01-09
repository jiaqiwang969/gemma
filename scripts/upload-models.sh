#!/bin/bash
# =============================================================================
# 灵空 AI - 模型上传脚本 (支持断点续传)
# =============================================================================
# 使用方法:
#   ./upload-models.sh              # 上传所有模型
#   ./upload-models.sh text         # 只上传文本模型
#   ./upload-models.sh vision       # 只上传视觉模型
#   ./upload-models.sh audio        # 只上传音频模型
# =============================================================================

set -e

SERVER="ubuntu@115.159.223.227"
REMOTE_DIR="/var/www/html/models"
MODELS_DIR="${LINGKONG_HOME:-$HOME/.lingkong}/models"

# 模型文件列表
declare -A MODELS=(
    ["text"]="gemma-3n-E2B-it-Q4_K_M.gguf"
    ["vision"]="gemma-3n-vision-mmproj-f16.gguf"
    ["audio"]="gemma-3n-audio-mmproj-f16.gguf"
)

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# 最大重试次数
MAX_RETRIES=5
RETRY_DELAY=10

log_info() { echo -e "${CYAN}[信息]${NC} $1"; }
log_success() { echo -e "${GREEN}[成功]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[警告]${NC} $1"; }
log_error() { echo -e "${RED}[错误]${NC} $1"; }

# 获取文件大小 (人类可读)
get_file_size() {
    if [[ -f "$1" ]]; then
        du -h "$1" | cut -f1
    else
        echo "N/A"
    fi
}

# 获取远程文件大小
get_remote_size() {
    ssh -o ConnectTimeout=10 $SERVER "stat -c%s $REMOTE_DIR/$1 2>/dev/null || echo 0"
}

# 上传单个文件 (带断点续传和重试)
upload_file() {
    local model_type="$1"
    local filename="${MODELS[$model_type]}"
    local local_path="$MODELS_DIR/$filename"

    if [[ ! -f "$local_path" ]]; then
        log_error "本地文件不存在: $local_path"
        return 1
    fi

    local local_size=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path")
    local human_size=$(get_file_size "$local_path")

    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  上传: $filename ($human_size)${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"

    local retry=0
    while [[ $retry -lt $MAX_RETRIES ]]; do
        # 检查远程文件大小
        local remote_size=$(get_remote_size "$filename")

        if [[ "$remote_size" -eq "$local_size" ]]; then
            log_success "$filename 已完整上传，跳过"
            return 0
        elif [[ "$remote_size" -gt 0 ]]; then
            local percent=$((remote_size * 100 / local_size))
            log_info "检测到未完成的上传 ($percent%)，继续..."
        fi

        log_info "开始上传 (尝试 $((retry + 1))/$MAX_RETRIES)..."

        # 使用 rsync 断点续传
        if rsync -avz --progress --partial --timeout=300 \
            -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=5 -o ConnectTimeout=30" \
            "$local_path" "$SERVER:$REMOTE_DIR/$filename"; then

            # 验证上传
            remote_size=$(get_remote_size "$filename")
            if [[ "$remote_size" -eq "$local_size" ]]; then
                log_success "$filename 上传完成!"

                # 设置权限
                ssh $SERVER "chmod 644 $REMOTE_DIR/$filename"
                return 0
            else
                log_warn "文件大小不匹配，将重试"
            fi
        else
            log_warn "上传中断，等待 ${RETRY_DELAY} 秒后重试..."
        fi

        retry=$((retry + 1))
        if [[ $retry -lt $MAX_RETRIES ]]; then
            sleep $RETRY_DELAY
        fi
    done

    log_error "$filename 上传失败 (已重试 $MAX_RETRIES 次)"
    return 1
}

# 检查服务器连接
check_server() {
    log_info "检查服务器连接..."
    if ! ssh -o ConnectTimeout=10 $SERVER "echo 'ok'" &>/dev/null; then
        log_error "无法连接服务器: $SERVER"
        exit 1
    fi
    log_success "服务器连接正常"

    # 确保目录存在并有权限
    ssh $SERVER "sudo mkdir -p $REMOTE_DIR && sudo chown -R ubuntu:ubuntu $REMOTE_DIR && sudo chmod 755 $REMOTE_DIR"
}

# 显示状态
show_status() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  模型文件状态${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"

    for model_type in text vision audio; do
        local filename="${MODELS[$model_type]}"
        local local_path="$MODELS_DIR/$filename"

        if [[ -f "$local_path" ]]; then
            local local_size=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path")
            local human_size=$(get_file_size "$local_path")
            local remote_size=$(get_remote_size "$filename" 2>/dev/null || echo "0")

            if [[ "$remote_size" -eq "$local_size" ]]; then
                echo -e "  ${GREEN}✓${NC} $model_type: $filename ($human_size) - ${GREEN}已上传${NC}"
            elif [[ "$remote_size" -gt 0 ]]; then
                local percent=$((remote_size * 100 / local_size))
                echo -e "  ${YELLOW}◐${NC} $model_type: $filename ($human_size) - ${YELLOW}${percent}%${NC}"
            else
                echo -e "  ${RED}○${NC} $model_type: $filename ($human_size) - ${RED}未上传${NC}"
            fi
        else
            echo -e "  ${RED}✗${NC} $model_type: $filename - ${RED}本地不存在${NC}"
        fi
    done
    echo ""
}

# 主函数
main() {
    echo ""
    echo -e "${CYAN}🐉 灵空 AI - 模型上传工具 (支持断点续传)${NC}"
    echo ""

    check_server

    local target="${1:-all}"

    case "$target" in
        status)
            show_status
            ;;
        text|vision|audio)
            upload_file "$target"
            ;;
        all|"")
            show_status

            local failed=0
            for model_type in text vision audio; do
                if ! upload_file "$model_type"; then
                    failed=$((failed + 1))
                fi
            done

            echo ""
            if [[ $failed -eq 0 ]]; then
                log_success "所有模型上传完成!"
                echo ""
                echo -e "  📥 下载地址: ${YELLOW}http://115.159.223.227/models/${NC}"
                echo ""
            else
                log_error "$failed 个模型上传失败，请重新运行脚本继续"
            fi
            ;;
        *)
            echo "使用方法: $0 [text|vision|audio|all|status]"
            echo ""
            echo "  text    - 上传文本模型 (2.6GB)"
            echo "  vision  - 上传视觉模型 (570MB)"
            echo "  audio   - 上传音频模型 (1.3GB)"
            echo "  all     - 上传所有模型 (默认)"
            echo "  status  - 查看上传状态"
            ;;
    esac
}

main "$@"

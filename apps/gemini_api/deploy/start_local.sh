#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LingKong AI - 本地端启动脚本
# ═══════════════════════════════════════════════════════════════════════════════
#
# 在本地 Mac 上运行，启动 Gemma 3N 推理服务并建立到公网服务器的隧道
#
# 功能:
#   1. 启动本地 Gemma 3N 推理服务 (llama.cpp)
#   2. 建立 SSH 反向隧道到公网服务器
#
# 使用:
#   ./start_local.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 配置
REMOTE_SERVER="ubuntu@115.159.223.227"
LOCAL_API_PORT=5001
TUNNEL_PORT=5001

echo "═══════════════════════════════════════════════════════════════════════"
echo "LingKong AI - 本地推理服务"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "仓库根目录: $REPO_ROOT"
echo "本地 API 端口: $LOCAL_API_PORT"
echo "远程服务器: $REMOTE_SERVER"
echo ""

# ========== 1. 检查本地服务是否已运行 ==========
if curl -s "http://127.0.0.1:$LOCAL_API_PORT/health" > /dev/null 2>&1; then
    echo "[✓] 本地推理服务已在运行"
else
    echo "[!] 本地推理服务未运行，正在启动..."

    # 启动本地 Gemini API 服务
    cd "$REPO_ROOT"

    # 检查 Python 环境
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # 后台启动服务
    nohup python apps/gemini_api/server.py > /tmp/gemini-api.log 2>&1 &
    LOCAL_PID=$!
    echo "    启动进程 PID: $LOCAL_PID"

    # 等待服务就绪
    echo "    等待服务就绪..."
    for i in {1..30}; do
        if curl -s "http://127.0.0.1:$LOCAL_API_PORT/health" > /dev/null 2>&1; then
            echo "[✓] 本地推理服务已就绪"
            break
        fi
        sleep 2
        if [ $i -eq 30 ]; then
            echo "[✗] 服务启动超时，请检查日志: /tmp/gemini-api.log"
            exit 1
        fi
    done
fi

# ========== 2. 建立 SSH 隧道 ==========
echo ""
echo "正在建立 SSH 反向隧道..."
echo "  本地 127.0.0.1:$LOCAL_API_PORT -> 远程 127.0.0.1:$TUNNEL_PORT"
echo ""
echo "提示: 保持此终端运行以维持隧道连接"
echo "     按 Ctrl+C 断开隧道"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# 建立 SSH 反向隧道
# -N: 不执行远程命令
# -R: 反向隧道
# -o ServerAliveInterval: 保持连接
# -o ExitOnForwardFailure: 端口转发失败时退出
ssh -N \
    -R $TUNNEL_PORT:127.0.0.1:$LOCAL_API_PORT \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    $REMOTE_SERVER

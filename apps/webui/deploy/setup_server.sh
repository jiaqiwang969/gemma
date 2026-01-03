#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LingKong AI WebUI - 服务器部署脚本
# ═══════════════════════════════════════════════════════════════════════════════
#
# 功能: 在 Ubuntu 服务器上部署多模态聊天 WebUI
#
# 使用方式:
#   ssh ubuntu@115.159.223.227 'bash -s' < apps/webui/deploy/setup_server.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════════════"
echo "LingKong AI WebUI - 服务器部署"
echo "═══════════════════════════════════════════════════════════════════════"

# ========== 配置变量 ==========
INSTALL_DIR="/opt/lingkong-webui"
LOG_DIR="/var/log/lingkong-webui"
DATA_DIR="/var/lib/lingkong-webui"
USER="ubuntu"
GROUP="ubuntu"

# ========== 1. 安装系统依赖 ==========
echo "[1/7] 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    ffmpeg libsndfile1

# ========== 2. 创建目录结构 ==========
echo "[2/7] 创建目录结构..."
sudo mkdir -p $INSTALL_DIR/{static,sessions}
sudo mkdir -p $LOG_DIR
sudo mkdir -p $DATA_DIR/sessions
sudo chown -R $USER:$GROUP $INSTALL_DIR $LOG_DIR $DATA_DIR

# ========== 3. 设置 Python 虚拟环境 ==========
echo "[3/7] 设置 Python 环境..."
cd $INSTALL_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install \
    flask flask-cors \
    requests \
    gunicorn \
    python-dotenv \
    pillow \
    soundfile \
    numpy

# ========== 4. 创建配置文件 ==========
echo "[4/7] 创建配置文件..."

# 生成随机密钥
SECRET_KEY=$(openssl rand -hex 32)

cat > $INSTALL_DIR/.env << EOF
# LingKong WebUI 配置
SECRET_KEY=${SECRET_KEY}
FLASK_ENV=production

# 后端配置 (SSH 隧道连接到本地)
LLAMA_CPP_URL=http://127.0.0.1:8081
MMPROJ_URL=http://127.0.0.1:5001

# 存储路径
SESSION_DIR=${DATA_DIR}/sessions
LOG_DIR=${LOG_DIR}

# 安全设置
MAX_CONTENT_LENGTH=52428800
ALLOWED_EXTENSIONS=png,jpg,jpeg,gif,webp,mp3,wav,m4a,webm
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "配置信息:"
echo "  SECRET_KEY: ${SECRET_KEY}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ========== 5. 创建 WSGI 入口 ==========
echo "[5/7] 创建 WSGI 入口..."

cat > $INSTALL_DIR/wsgi.py << 'EOF'
#!/usr/bin/env python3
"""WSGI 入口点 - 用于 Gunicorn 部署"""
import os
import sys

# 设置环境变量
os.environ.setdefault('GEMMA_SESSION_DIR', '/var/lib/lingkong-webui/sessions')

# 导入 Flask 应用
from server import app

if __name__ == "__main__":
    app.run()
EOF

# ========== 6. 创建 Systemd 服务 ==========
echo "[6/7] 创建 Systemd 服务..."

sudo tee /etc/systemd/system/lingkong-webui.service > /dev/null << EOF
[Unit]
Description=LingKong AI WebUI - Multimodal Chat Interface
After=network.target

[Service]
Type=simple
User=$USER
Group=$GROUP
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/gunicorn \\
    --workers 2 \\
    --threads 4 \\
    --bind 127.0.0.1:5000 \\
    --timeout 600 \\
    --access-logfile $LOG_DIR/access.log \\
    --error-logfile $LOG_DIR/error.log \\
    --capture-output \\
    wsgi:app
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ========== 7. 配置 Nginx ==========
echo "[7/7] 配置 Nginx..."

sudo tee /etc/nginx/sites-available/lingkong-webui > /dev/null << 'EOF'
# LingKong AI WebUI - Nginx 配置
# 监听端口 80 (HTTP) 和 443 (HTTPS)

upstream webui_backend {
    server 127.0.0.1:5000;
    keepalive 32;
}

server {
    listen 80;
    server_name lingkong.xyz www.lingkong.xyz;

    # Let's Encrypt 验证
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # 重定向到 HTTPS (在申请证书后启用)
    # return 301 https://$host$request_uri;

    # 静态文件
    location /static/ {
        alias /opt/lingkong-webui/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # WebSocket 支持
    location /socket.io/ {
        proxy_pass http://webui_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }

    # API 和页面
    location / {
        proxy_pass http://webui_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 长连接超时 (用于流式响应)
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;

        # 禁用缓冲 (用于 SSE)
        proxy_buffering off;
        proxy_cache off;

        # 文件上传大小限制
        client_max_body_size 50M;

        # CORS
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS' always;
        add_header Access-Control-Allow-Headers 'Content-Type, Authorization' always;

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
}

# HTTPS 配置 (在 certbot 申请证书后自动添加)
# server {
#     listen 443 ssl http2;
#     server_name lingkong.xyz www.lingkong.xyz;
#
#     ssl_certificate /etc/letsencrypt/live/lingkong.xyz/fullchain.pem;
#     ssl_certificate_key /etc/letsencrypt/live/lingkong.xyz/privkey.pem;
#
#     ... (其他配置同上)
# }
EOF

# 启用站点
sudo ln -sf /etc/nginx/sites-available/lingkong-webui /etc/nginx/sites-enabled/

# 测试 Nginx 配置
sudo nginx -t

# 重载 Nginx
sudo systemctl reload nginx

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "部署完成!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "下一步:"
echo ""
echo "1. 上传 WebUI 文件到服务器:"
echo "   ./apps/webui/deploy/deploy.sh"
echo ""
echo "2. 启动服务:"
echo "   ssh ubuntu@115.159.223.227 'sudo systemctl daemon-reload && sudo systemctl enable lingkong-webui && sudo systemctl start lingkong-webui'"
echo ""
echo "3. 申请 SSL 证书:"
echo "   ssh ubuntu@115.159.223.227 'sudo certbot --nginx -d lingkong.xyz'"
echo ""
echo "4. 建立 SSH 隧道 (在本地 Mac 运行):"
echo "   ssh -R 5001:127.0.0.1:5001 -R 8081:127.0.0.1:8081 ubuntu@115.159.223.227"
echo ""
echo "5. 访问:"
echo "   http://lingkong.xyz (或 https://lingkong.xyz)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

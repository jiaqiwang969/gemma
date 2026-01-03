#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LingKong AI - 完整服务器部署脚本
# ═══════════════════════════════════════════════════════════════════════════════
#
# 功能: 一键部署 WebUI + Gateway 到远程服务器
#
# 包含:
#   - Gemini API Gateway (兼容 Google Gemini API)
#   - 多模态聊天 WebUI
#   - Nginx 反向代理
#   - SSL 证书
#
# 使用方式:
#   ssh ubuntu@115.159.223.227 'bash -s' < apps/deploy_all.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  _     _       _  __                       _    ___ "
echo " | |   (_)_ __ | |/ /___  _ __   __ _      / \  |_ _|"
echo " | |   | | '_ \| ' // _ \| '_ \ / _\` |    / _ \  | | "
echo " | |___| | | | | . \ (_) | | | | (_| |   / ___ \ | | "
echo " |_____|_|_| |_|_|\_\___/|_| |_|\__, |  /_/   \_\___|"
echo "                                |___/                "
echo ""
echo " 灵空 AI - 完整服务器部署"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ========== 配置 ==========
USER="ubuntu"
GROUP="ubuntu"

# ========== 1. 安装系统依赖 ==========
echo "[1/8] 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    ffmpeg libsndfile1 \
    git curl wget

# ========== 2. 创建目录结构 ==========
echo "[2/8] 创建目录结构..."

# Gateway
sudo mkdir -p /opt/lingkong-gateway
sudo mkdir -p /var/lib/lingkong-gateway
sudo mkdir -p /var/log/lingkong-gateway

# WebUI
sudo mkdir -p /opt/lingkong-webui/{static,sessions}
sudo mkdir -p /var/lib/lingkong-webui/sessions
sudo mkdir -p /var/log/lingkong-webui

# 设置权限
sudo chown -R $USER:$GROUP /opt/lingkong-gateway /var/lib/lingkong-gateway /var/log/lingkong-gateway
sudo chown -R $USER:$GROUP /opt/lingkong-webui /var/lib/lingkong-webui /var/log/lingkong-webui

# ========== 3. 设置 Gateway Python 环境 ==========
echo "[3/8] 设置 Gateway Python 环境..."
cd /opt/lingkong-gateway
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flask flask-cors requests gunicorn python-dotenv
deactivate

# ========== 4. 设置 WebUI Python 环境 ==========
echo "[4/8] 设置 WebUI Python 环境..."
cd /opt/lingkong-webui
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flask flask-cors requests gunicorn python-dotenv pillow soundfile numpy
deactivate

# ========== 5. 创建配置文件 ==========
echo "[5/8] 创建配置文件..."

# Gateway 配置
ADMIN_PASSWORD=$(openssl rand -hex 16)
INITIAL_API_KEY="sk-$(openssl rand -hex 24)"

cat > /opt/lingkong-gateway/.env << EOF
ADMIN_PASSWORD=${ADMIN_PASSWORD}
PRESET_API_KEYS=${INITIAL_API_KEY}
LOCAL_INFERENCE_URL=http://127.0.0.1:5001
GATEWAY_PORT=8080
DB_PATH=/var/lib/lingkong-gateway/api_keys.db
LOG_REQUESTS=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_TOKENS=100000
EOF

# WebUI 配置
WEBUI_SECRET=$(openssl rand -hex 32)

cat > /opt/lingkong-webui/.env << EOF
SECRET_KEY=${WEBUI_SECRET}
FLASK_ENV=production
LLAMA_CPP_URL=http://127.0.0.1:8081
MMPROJ_URL=http://127.0.0.1:5001
SESSION_DIR=/var/lib/lingkong-webui/sessions
LOG_DIR=/var/log/lingkong-webui
MAX_CONTENT_LENGTH=52428800
EOF

# 创建 WebUI WSGI 入口
cat > /opt/lingkong-webui/wsgi.py << 'EOF'
import os
os.environ.setdefault('GEMMA_SESSION_DIR', '/var/lib/lingkong-webui/sessions')
from server import app
if __name__ == "__main__":
    app.run()
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "重要信息 (请保存!):"
echo "═══════════════════════════════════════════════════════════════════════"
echo "Gateway Admin Password: ${ADMIN_PASSWORD}"
echo "Initial API Key:        ${INITIAL_API_KEY}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ========== 6. 创建 Systemd 服务 ==========
echo "[6/8] 创建 Systemd 服务..."

# Gateway 服务
sudo tee /etc/systemd/system/lingkong-gateway.service > /dev/null << 'EOF'
[Unit]
Description=LingKong AI Gateway - Gemini API Compatible
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/lingkong-gateway
EnvironmentFile=/opt/lingkong-gateway/.env
ExecStart=/opt/lingkong-gateway/venv/bin/gunicorn \
    --workers 4 \
    --threads 2 \
    --bind 127.0.0.1:8080 \
    --timeout 300 \
    --access-logfile /var/log/lingkong-gateway/access.log \
    --error-logfile /var/log/lingkong-gateway/error.log \
    gateway:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# WebUI 服务
sudo tee /etc/systemd/system/lingkong-webui.service > /dev/null << 'EOF'
[Unit]
Description=LingKong AI WebUI - Multimodal Chat Interface
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/lingkong-webui
EnvironmentFile=/opt/lingkong-webui/.env
ExecStart=/opt/lingkong-webui/venv/bin/gunicorn \
    --workers 2 \
    --threads 4 \
    --bind 127.0.0.1:5000 \
    --timeout 600 \
    --access-logfile /var/log/lingkong-webui/access.log \
    --error-logfile /var/log/lingkong-webui/error.log \
    --capture-output \
    wsgi:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ========== 7. 配置 Nginx ==========
echo "[7/8] 配置 Nginx..."

sudo tee /etc/nginx/sites-available/lingkong > /dev/null << 'EOF'
# LingKong AI - 统一 Nginx 配置
# WebUI: / (根路径)
# API Gateway: /v1beta/, /admin/

upstream webui {
    server 127.0.0.1:5000;
    keepalive 32;
}

upstream gateway {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name lingkong.xyz www.lingkong.xyz;

    # Let's Encrypt 验证
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # API Gateway 路由
    location /v1beta/ {
        proxy_pass http://gateway;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;

        # CORS
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS' always;
        add_header Access-Control-Allow-Headers 'Content-Type, X-API-Key, Authorization' always;

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }

    # Gateway 管理接口
    location /admin/ {
        proxy_pass http://gateway;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Gateway 健康检查
    location /health {
        proxy_pass http://gateway;
    }

    # WebUI 静态文件
    location /static/ {
        alias /opt/lingkong-webui/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # WebUI API
    location /api/ {
        proxy_pass http://webui;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        proxy_buffering off;
        client_max_body_size 50M;
    }

    # WebUI 主页面
    location / {
        proxy_pass http://webui;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        proxy_buffering off;
    }
}
EOF

# 启用站点
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f /etc/nginx/sites-enabled/lingkong-ai
sudo rm -f /etc/nginx/sites-enabled/lingkong-webui
sudo ln -sf /etc/nginx/sites-available/lingkong /etc/nginx/sites-enabled/

# 测试并重载 Nginx
sudo nginx -t
sudo systemctl reload nginx

# ========== 8. 启用服务 ==========
echo "[8/8] 启用服务..."
sudo systemctl daemon-reload
# 暂不启动服务，等待上传代码后再启动

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "服务器环境配置完成!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "下一步:"
echo ""
echo "1. 上传代码到服务器:"
echo "   scp apps/gemini_api/gateway.py ubuntu@115.159.223.227:/opt/lingkong-gateway/"
echo "   scp apps/webui/server.py ubuntu@115.159.223.227:/opt/lingkong-webui/"
echo "   scp -r apps/webui/static/* ubuntu@115.159.223.227:/opt/lingkong-webui/static/"
echo ""
echo "2. 启动服务:"
echo "   ssh ubuntu@115.159.223.227 'sudo systemctl enable --now lingkong-gateway lingkong-webui'"
echo ""
echo "3. 申请 SSL 证书:"
echo "   ssh ubuntu@115.159.223.227 'sudo certbot --nginx -d lingkong.xyz'"
echo ""
echo "4. 建立 SSH 隧道 (本地运行):"
echo "   ssh -R 5001:127.0.0.1:5001 -R 8081:127.0.0.1:8081 ubuntu@115.159.223.227"
echo ""
echo "5. 测试:"
echo "   - WebUI:   http://lingkong.xyz"
echo "   - API:     http://lingkong.xyz/v1beta/models"
echo "   - Health:  http://lingkong.xyz/health"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "重要凭据 (请保存!):"
echo "  Admin Password: ${ADMIN_PASSWORD}"
echo "  API Key:        ${INITIAL_API_KEY}"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

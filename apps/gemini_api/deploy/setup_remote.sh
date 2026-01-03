#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LingKong AI Gateway - 远程服务器部署脚本
# ═══════════════════════════════════════════════════════════════════════════════
#
# 在公网服务器 (115.159.223.227) 上运行此脚本
#
# 前置条件:
#   1. 已配置 SSH 无密码登录
#   2. 域名 lingkong.xyz 已解析到服务器
#   3. 服务器有 Python 3.8+
#
# 使用:
#   ssh ubuntu@115.159.223.227 'bash -s' < setup_remote.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════════════"
echo "LingKong AI Gateway - 服务器部署"
echo "═══════════════════════════════════════════════════════════════════════"

# ========== 1. 安装依赖 ==========
echo "[1/6] 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

# ========== 2. 创建项目目录 ==========
echo "[2/6] 创建项目目录..."
sudo mkdir -p /opt/lingkong-ai
sudo mkdir -p /var/lib/gemini-gateway
sudo mkdir -p /var/log/gemini-gateway
sudo chown -R $USER:$USER /opt/lingkong-ai /var/lib/gemini-gateway /var/log/gemini-gateway

# ========== 3. 设置 Python 虚拟环境 ==========
echo "[3/6] 设置 Python 环境..."
cd /opt/lingkong-ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flask flask-cors requests gunicorn

# ========== 4. 创建配置文件 ==========
echo "[4/6] 创建配置文件..."

# 生成随机管理员密码
ADMIN_PASSWORD=$(openssl rand -hex 16)
echo "ADMIN_PASSWORD=${ADMIN_PASSWORD}" > /opt/lingkong-ai/.env

# 生成初始 API Key
INITIAL_API_KEY="sk-$(openssl rand -hex 24)"
echo "PRESET_API_KEYS=${INITIAL_API_KEY}" >> /opt/lingkong-ai/.env

cat >> /opt/lingkong-ai/.env << 'EOF'
LOCAL_INFERENCE_URL=http://127.0.0.1:5001
GATEWAY_PORT=8080
DB_PATH=/var/lib/gemini-gateway/api_keys.db
LOG_REQUESTS=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_TOKENS=100000
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "重要信息 (请保存!):"
echo "═══════════════════════════════════════════════════════════════════════"
echo "ADMIN_PASSWORD: ${ADMIN_PASSWORD}"
echo "INITIAL_API_KEY: ${INITIAL_API_KEY}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ========== 5. 创建 Systemd 服务 ==========
echo "[5/6] 创建 Systemd 服务..."

sudo tee /etc/systemd/system/lingkong-gateway.service > /dev/null << 'EOF'
[Unit]
Description=LingKong AI Gateway - Gemini API Compatible
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/lingkong-ai
EnvironmentFile=/opt/lingkong-ai/.env
ExecStart=/opt/lingkong-ai/venv/bin/gunicorn \
    --workers 4 \
    --threads 2 \
    --bind 127.0.0.1:8080 \
    --timeout 300 \
    --access-logfile /var/log/gemini-gateway/access.log \
    --error-logfile /var/log/gemini-gateway/error.log \
    gateway:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ========== 6. 配置 Nginx ==========
echo "[6/6] 配置 Nginx..."

sudo tee /etc/nginx/sites-available/lingkong-ai > /dev/null << 'EOF'
server {
    listen 80;
    server_name lingkong.xyz www.lingkong.xyz;

    # 用于 Let's Encrypt 验证
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # API 反向代理
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;

        # CORS
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
        add_header Access-Control-Allow-Headers 'Content-Type, X-API-Key';

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/lingkong-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "部署完成!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "下一步:"
echo ""
echo "1. 上传 gateway.py 到服务器:"
echo "   scp apps/gemini_api/gateway.py ubuntu@115.159.223.227:/opt/lingkong-ai/"
echo ""
echo "2. 启动服务:"
echo "   ssh ubuntu@115.159.223.227 'sudo systemctl daemon-reload && sudo systemctl enable lingkong-gateway && sudo systemctl start lingkong-gateway'"
echo ""
echo "3. 申请 SSL 证书 (推荐):"
echo "   ssh ubuntu@115.159.223.227 'sudo certbot --nginx -d lingkong.xyz -d www.lingkong.xyz'"
echo ""
echo "4. 建立 SSH 隧道 (本地运行):"
echo "   ssh -R 5001:127.0.0.1:5001 ubuntu@115.159.223.227"
echo ""
echo "5. 测试 API:"
echo "   curl 'https://lingkong.xyz/v1beta/models/gemini-3-pro-preview:generateContent?key=${INITIAL_API_KEY}' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"contents\": [{\"parts\": [{\"text\": \"hi\"}]}]}'"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

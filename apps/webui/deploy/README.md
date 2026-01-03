# LingKong AI WebUI - 部署文档

## 架构概览

```
用户 (浏览器)
    │
    ▼
┌─────────────────────────────────────────┐
│  公网服务器 (lingkong.xyz)              │
│  115.159.223.227                        │
│  ├─ Nginx (SSL/反向代理/静态文件)       │
│  └─ Gunicorn + Flask (WebUI 后端)       │
└─────────────────────────────────────────┘
    │
    │ SSH 反向隧道
    │ ├─ 5001: 推理 API
    │ └─ 8081: llama.cpp
    ▼
┌─────────────────────────────────────────┐
│  本地 Mac                               │
│  ├─ llama.cpp 推理服务                  │
│  └─ Gemma 3N 模型 (2B/4B)               │
└─────────────────────────────────────────┘
```

## 快速开始

### 1. 首次部署 (在服务器上执行)

```bash
# 从本地运行初始化脚本
ssh ubuntu@115.159.223.227 'bash -s' < apps/webui/deploy/setup_server.sh
```

### 2. 上传 WebUI 文件

```bash
# 设置执行权限
chmod +x apps/webui/deploy/*.sh

# 部署文件
./apps/webui/deploy/deploy.sh
```

### 3. 启动服务

```bash
# 在服务器上启动 WebUI 服务
ssh ubuntu@115.159.223.227 'sudo systemctl daemon-reload && sudo systemctl enable lingkong-webui && sudo systemctl start lingkong-webui'
```

### 4. 申请 SSL 证书 (推荐)

```bash
ssh ubuntu@115.159.223.227 'sudo certbot --nginx -d lingkong.xyz'
```

### 5. 本地启动推理服务

```bash
# 使用一键启动脚本
./apps/webui/deploy/start_local.sh

# 或手动操作:
# 终端 1: 启动 llama.cpp
~/.lingkong/llama.cpp/build/bin/llama-server \
    -m ~/.lingkong/models/gemma3n-2b-text.gguf \
    --port 8081 \
    -ngl 99

# 终端 2: 建立 SSH 隧道
ssh -R 5001:127.0.0.1:5001 -R 8081:127.0.0.1:8081 ubuntu@115.159.223.227
```

## 目录结构

### 服务器端

```
/opt/lingkong-webui/
├── venv/              # Python 虚拟环境
├── server.py          # Flask 主程序
├── wsgi.py            # Gunicorn WSGI 入口
├── .env               # 配置文件
└── static/            # 静态文件
    └── index.html     # 前端页面

/var/lib/lingkong-webui/
└── sessions/          # 会话存储

/var/log/lingkong-webui/
├── access.log         # 访问日志
└── error.log          # 错误日志
```

### 本地端

```
apps/webui/
├── server.py          # Flask 服务器
├── static/
│   └── index.html     # 前端页面
├── deploy/
│   ├── setup_server.sh   # 服务器初始化
│   ├── deploy.sh         # 文件部署
│   ├── start_local.sh    # 本地启动
│   └── README.md         # 本文档
└── run.sh             # 本地开发脚本
```

## 服务管理

### 查看状态

```bash
ssh ubuntu@115.159.223.227 'sudo systemctl status lingkong-webui'
```

### 查看日志

```bash
# 实时日志
ssh ubuntu@115.159.223.227 'sudo journalctl -u lingkong-webui -f'

# 访问日志
ssh ubuntu@115.159.223.227 'tail -f /var/log/lingkong-webui/access.log'

# 错误日志
ssh ubuntu@115.159.223.227 'tail -f /var/log/lingkong-webui/error.log'
```

### 重启服务

```bash
ssh ubuntu@115.159.223.227 'sudo systemctl restart lingkong-webui'
```

### 停止服务

```bash
ssh ubuntu@115.159.223.227 'sudo systemctl stop lingkong-webui'
```

## 更新部署

当修改了代码后，运行:

```bash
./apps/webui/deploy/deploy.sh
```

这会自动:
1. 上传更新的文件
2. 重启 WebUI 服务

## 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 80 | Nginx | HTTP 入口 |
| 443 | Nginx | HTTPS 入口 |
| 5000 | Gunicorn | WebUI 后端 (内部) |
| 5001 | SSH 隧道 | 推理 API (远程 → 本地) |
| 8081 | SSH 隧道 | llama.cpp (远程 → 本地) |

## 故障排查

### WebUI 无法访问

```bash
# 检查 Nginx
ssh ubuntu@115.159.223.227 'sudo nginx -t && sudo systemctl status nginx'

# 检查 WebUI 服务
ssh ubuntu@115.159.223.227 'sudo systemctl status lingkong-webui'

# 测试本地连接
ssh ubuntu@115.159.223.227 'curl http://127.0.0.1:5000/api/status'
```

### 推理失败

```bash
# 检查 SSH 隧道是否建立
ssh ubuntu@115.159.223.227 'curl http://127.0.0.1:5001/health'

# 检查本地 llama.cpp 是否运行
curl http://127.0.0.1:8081/health
```

### SSL 证书问题

```bash
# 重新申请证书
ssh ubuntu@115.159.223.227 'sudo certbot renew --force-renewal'

# 检查证书状态
ssh ubuntu@115.159.223.227 'sudo certbot certificates'
```

## 安全建议

1. **使用 HTTPS**: 务必申请 SSL 证书
2. **防火墙**: 只开放 80、443 端口
3. **SSH 隧道**: 不要暴露内部端口 5000、5001、8081
4. **定期更新**: `sudo apt update && sudo apt upgrade`

## 性能优化

### Gunicorn 配置

编辑 `/etc/systemd/system/lingkong-webui.service`:

```ini
# 根据 CPU 核心数调整 workers
--workers 4
--threads 4
```

### Nginx 缓存

静态文件已配置 7 天缓存。如需调整，编辑:
`/etc/nginx/sites-available/lingkong-webui`

## 联系方式

- 项目: https://github.com/jiaqiwang969/LingKong-AI
- 问题反馈: GitHub Issues

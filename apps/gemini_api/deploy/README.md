# LingKong AI Gateway - 公网 Gemini API 服务

## 架构概览

```
用户请求
    │
    ▼
┌─────────────────────────────────────┐
│  公网服务器 (lingkong.xyz)          │
│  115.159.223.227                    │
│  ├─ Nginx (SSL/反向代理)            │
│  └─ Gateway (认证/限流/转发)        │
└─────────────────────────────────────┘
    │
    │ SSH 反向隧道 (端口 5001)
    ▼
┌─────────────────────────────────────┐
│  本地 Mac                           │
│  ├─ Gemini API Server (server.py)  │
│  └─ llama.cpp + Gemma 3N 模型      │
└─────────────────────────────────────┘
```

## 快速开始

### 1. 部署公网网关 (首次设置)

```bash
# 在远程服务器上执行初始化脚本
ssh ubuntu@115.159.223.227 'bash -s' < apps/gemini_api/deploy/setup_remote.sh

# 上传 gateway.py
scp apps/gemini_api/gateway.py ubuntu@115.159.223.227:/opt/lingkong-ai/

# 启动服务
ssh ubuntu@115.159.223.227 'sudo systemctl daemon-reload && sudo systemctl enable lingkong-gateway && sudo systemctl start lingkong-gateway'

# 申请 SSL 证书 (可选但推荐)
ssh ubuntu@115.159.223.227 'sudo certbot --nginx -d lingkong.xyz'
```

### 2. 启动本地推理服务

```bash
# 方式一: 使用启动脚本 (推荐)
chmod +x apps/gemini_api/deploy/start_local.sh
./apps/gemini_api/deploy/start_local.sh

# 方式二: 手动启动
# 终端 1: 启动本地 API
python apps/gemini_api/server.py

# 终端 2: 建立 SSH 隧道
ssh -R 5001:127.0.0.1:5001 ubuntu@115.159.223.227
```

### 3. 测试 API

```bash
# 替换 YOUR_API_KEY 为实际的 key
API_KEY="sk-xxx"

curl -X POST "https://lingkong.xyz/v1beta/models/gemini-3-pro-preview:generateContent?key=${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "hi"}]}]
  }'
```

## 用户使用方式

给用户的接入信息:

```
BASE_URL: https://lingkong.xyz
API_KEY: sk-xxx (由管理员分配)
```

### Python 示例

```python
import requests

BASE_URL = "https://lingkong.xyz"
API_KEY = "sk-xxx"

response = requests.post(
    f"{BASE_URL}/v1beta/models/gemini-3-pro-preview:generateContent?key={API_KEY}",
    json={
        "contents": [{"parts": [{"text": "你好，介绍一下你自己"}]}]
    }
)

result = response.json()
text = result["candidates"][0]["content"]["parts"][0]["text"]
print(text)
```

### JavaScript 示例

```javascript
const BASE_URL = "https://lingkong.xyz";
const API_KEY = "sk-xxx";

const response = await fetch(
  `${BASE_URL}/v1beta/models/gemini-3-pro-preview:generateContent?key=${API_KEY}`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: "你好" }] }]
    })
  }
);

const result = await response.json();
console.log(result.candidates[0].content.parts[0].text);
```

### curl 示例

```bash
curl -X POST 'https://lingkong.xyz/v1beta/models/gemini-3-pro-preview:generateContent?key=sk-xxx' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{"parts": [{"text": "hi"}]}],
    "generationConfig": {
      "thinkingConfig": {"thinkingLevel": "medium"},
      "maxOutputTokens": 1024
    }
  }'
```

## 支持的功能

| 功能 | 状态 | 说明 |
|------|------|------|
| generateContent | ✅ | 文本生成 |
| thinkingConfig | ✅ | 思考等级 (low/medium/high) |
| systemInstruction | ✅ | 系统指令 |
| responseSchema | ✅ | JSON 结构化输出 |
| safetySettings | ✅ | 安全设置 |
| functionCall | ✅ | 工具调用 |
| thoughtSignature | ✅ | 思维签名 (多轮对话状态) |
| googleSearch | ❌ | 需要外部 API |
| 图像生成 | ❌ | 本地模型不支持 |

## 管理 API

### 创建新 API Key

```bash
curl -X POST "https://lingkong.xyz/admin/keys" \
  -H "Authorization: Bearer YOUR_ADMIN_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"name": "用户名"}'
```

### 查看使用统计

```bash
curl "https://lingkong.xyz/admin/stats" \
  -H "Authorization: Bearer YOUR_ADMIN_PASSWORD"
```

## 故障排查

### 服务器端

```bash
# 查看网关日志
ssh ubuntu@115.159.223.227 'sudo journalctl -u lingkong-gateway -f'

# 查看 Nginx 日志
ssh ubuntu@115.159.223.227 'sudo tail -f /var/log/nginx/access.log'

# 重启服务
ssh ubuntu@115.159.223.227 'sudo systemctl restart lingkong-gateway'
```

### 本地端

```bash
# 查看本地 API 日志
tail -f /tmp/gemini-api.log

# 测试本地服务
curl http://127.0.0.1:5001/health

# 检查 SSH 隧道
ssh ubuntu@115.159.223.227 'curl http://127.0.0.1:5001/health'
```

## 安全注意事项

1. **API Key 保护**: 不要在客户端代码中硬编码 API Key
2. **HTTPS**: 生产环境务必启用 SSL
3. **限流**: 默认每分钟 60 请求 / 100k tokens
4. **日志**: 所有请求都会被记录，用于监控和审计

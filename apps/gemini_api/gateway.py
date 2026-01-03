"""
Gemini API 公网网关服务
═══════════════════════════════════════════════════════════════════════════════

部署在公网服务器 (115.159.223.227 / lingkong.xyz) 上，
作为反向代理将用户请求转发到本地 Gemma 3N 推理服务。

架构图:
─────────
用户 → https://lingkong.xyz/v1beta/models/gemini-3-pro-preview:generateContent
                                    ↓
         [公网服务器: 115.159.223.227]
         [Nginx + SSL + 本服务]
                                    ↓  SSH 隧道 / 内网穿透
         [本地 Mac: Gemma 3N + llama.cpp]
                                    ↓
                            返回推理结果

功能:
─────
1. API Key 认证 (sk-xxx 格式)
2. 请求转发到本地推理服务
3. 请求日志 & 监控
4. 限流 & 防护

使用方式:
─────────
用户只需:
  BASE_URL = "https://lingkong.xyz"
  API_KEY = "sk-xxx"

  curl -X POST "${BASE_URL}/v1beta/models/gemini-3-pro-preview:generateContent?key=${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"contents": [{"parts": [{"text": "hi"}]}]}'
"""

import os
import sys
import json
import time
import uuid
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, List

import requests
from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS

# ========== 配置 ==========

# 本地推理服务地址 (通过 SSH 隧道或内网穿透)
LOCAL_INFERENCE_URL = os.environ.get("LOCAL_INFERENCE_URL", "http://127.0.0.1:5001")

# 网关端口
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))

# API Keys 数据库
DB_PATH = Path(os.environ.get("DB_PATH", "/var/lib/gemini-gateway/api_keys.db"))

# 预置的 API Keys (用于初始化)
PRESET_API_KEYS = os.environ.get("PRESET_API_KEYS", "").split(",")

# 限流配置
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))  # 每分钟
RATE_LIMIT_TOKENS = int(os.environ.get("RATE_LIMIT_TOKENS", "100000"))  # 每分钟

# 日志配置
LOG_REQUESTS = os.environ.get("LOG_REQUESTS", "true").lower() == "true"

# ========== 安全配置 ==========

# IP 黑名单 (内存缓存)
IP_BLACKLIST: Dict[str, float] = {}  # IP -> 解封时间戳
IP_BLACKLIST_DURATION = int(os.environ.get("IP_BLACKLIST_DURATION", "3600"))  # 默认封禁1小时

# 失败计数器
IP_FAIL_COUNT: Dict[str, List[float]] = {}  # IP -> [失败时间戳列表]
MAX_FAIL_COUNT = int(os.environ.get("MAX_FAIL_COUNT", "10"))  # 10次失败后封禁
FAIL_WINDOW = int(os.environ.get("FAIL_WINDOW", "300"))  # 5分钟内

# 可疑请求模式
SUSPICIOUS_PATTERNS = [
    "../", "..\\",  # 路径遍历
    "<script", "javascript:",  # XSS
    "SELECT ", "UNION ", "DROP ",  # SQL 注入
    "${", "#{",  # 模板注入
    "/etc/passwd", "/etc/shadow",  # 敏感文件
    "cmd=", "exec=", "system(",  # 命令注入
]

# 可疑 User-Agent
SUSPICIOUS_USER_AGENTS = [
    "sqlmap", "nikto", "nmap", "masscan", "nessus",
    "acunetix", "burpsuite", "dirbuster", "gobuster",
    "hydra", "medusa", "wfuzz", "ffuf",
]

# ========== Flask 应用 ==========

app = Flask(__name__)
CORS(app)

# 注册加密 API Blueprint
try:
    from crypto_api import crypto_bp
    app.register_blueprint(crypto_bp)
    print("[Gateway] Crypto API endpoints registered")
except ImportError:
    print("[Gateway] Crypto API not available (crypto_api module not found)")

# ========== 数据库管理 ==========

def get_db():
    """获取数据库连接"""
    if 'db' not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """关闭数据库连接"""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """初始化数据库"""
    db = sqlite3.connect(str(DB_PATH))
    db.executescript('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            total_requests INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_prefix TEXT,
            endpoint TEXT,
            method TEXT,
            status_code INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT
        );

        CREATE TABLE IF NOT EXISTS rate_limits (
            key_prefix TEXT PRIMARY KEY,
            minute_requests INTEGER DEFAULT 0,
            minute_tokens INTEGER DEFAULT 0,
            minute_start TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_logs_created_at ON request_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_key_prefix ON request_logs(key_prefix);
    ''')
    db.commit()
    db.close()


def hash_api_key(key: str) -> str:
    """对 API Key 进行哈希"""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key(name: str = "") -> str:
    """生成新的 API Key"""
    # 生成 sk-xxx 格式的 key
    random_part = uuid.uuid4().hex + uuid.uuid4().hex[:16]
    key = f"sk-{random_part}"

    # 存储到数据库
    db = sqlite3.connect(str(DB_PATH))
    try:
        db.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, name) VALUES (?, ?, ?)",
            (hash_api_key(key), key[:15], name)
        )
        db.commit()
    finally:
        db.close()

    return key


def validate_api_key(key: str) -> bool:
    """验证 API Key"""
    if not key or not key.startswith("sk-"):
        return False

    db = get_db()
    result = db.execute(
        "SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1",
        (hash_api_key(key),)
    ).fetchone()

    if result:
        # 更新最后使用时间
        db.execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE key_hash = ?",
            (hash_api_key(key),)
        )
        db.commit()
        return True

    return False


def check_rate_limit(key: str) -> tuple:
    """检查限流，返回 (是否通过, 剩余请求数, 剩余token数)"""
    key_prefix = key[:15] if key else "anonymous"
    now = datetime.now()
    minute_start = now.strftime("%Y-%m-%d %H:%M:00")

    db = get_db()

    # 获取或创建限流记录
    result = db.execute(
        "SELECT * FROM rate_limits WHERE key_prefix = ?",
        (key_prefix,)
    ).fetchone()

    if not result or result["minute_start"] != minute_start:
        # 新的一分钟，重置计数
        db.execute(
            """INSERT OR REPLACE INTO rate_limits
               (key_prefix, minute_requests, minute_tokens, minute_start)
               VALUES (?, 0, 0, ?)""",
            (key_prefix, minute_start)
        )
        db.commit()
        return True, RATE_LIMIT_REQUESTS, RATE_LIMIT_TOKENS

    requests_left = RATE_LIMIT_REQUESTS - result["minute_requests"]
    tokens_left = RATE_LIMIT_TOKENS - result["minute_tokens"]

    return requests_left > 0, requests_left, tokens_left


def update_rate_limit(key: str, tokens: int):
    """更新限流计数"""
    key_prefix = key[:15] if key else "anonymous"

    db = get_db()
    db.execute(
        """UPDATE rate_limits
           SET minute_requests = minute_requests + 1,
               minute_tokens = minute_tokens + ?
           WHERE key_prefix = ?""",
        (tokens, key_prefix)
    )
    db.commit()


def log_request(key: str, endpoint: str, method: str, status_code: int,
                prompt_tokens: int = 0, completion_tokens: int = 0,
                latency_ms: int = 0):
    """记录请求日志"""
    if not LOG_REQUESTS:
        return

    key_prefix = key[:15] if key else "anonymous"

    db = get_db()
    db.execute(
        """INSERT INTO request_logs
           (key_prefix, endpoint, method, status_code, prompt_tokens,
            completion_tokens, latency_ms, ip_address, user_agent)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (key_prefix, endpoint, method, status_code, prompt_tokens,
         completion_tokens, latency_ms,
         request.remote_addr, request.user_agent.string[:200] if request.user_agent else "")
    )
    db.commit()


# ========== 安全函数 ==========

def get_client_ip() -> str:
    """获取客户端真实 IP (考虑代理)"""
    # 优先从 X-Real-IP 获取 (Nginx 设置)
    if request.headers.get("X-Real-IP"):
        return request.headers.get("X-Real-IP")
    # 其次从 X-Forwarded-For 获取
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    return request.remote_addr


def is_ip_blacklisted(ip: str) -> bool:
    """检查 IP 是否在黑名单中"""
    if ip in IP_BLACKLIST:
        if time.time() < IP_BLACKLIST[ip]:
            return True
        else:
            # 已过期，移除
            del IP_BLACKLIST[ip]
    return False


def add_to_blacklist(ip: str, reason: str = ""):
    """将 IP 加入黑名单"""
    IP_BLACKLIST[ip] = time.time() + IP_BLACKLIST_DURATION
    print(f"[SECURITY] IP {ip} 已被封禁 {IP_BLACKLIST_DURATION}s, 原因: {reason}")


def record_fail(ip: str) -> bool:
    """记录失败尝试，返回是否应该封禁"""
    now = time.time()

    if ip not in IP_FAIL_COUNT:
        IP_FAIL_COUNT[ip] = []

    # 清理过期记录
    IP_FAIL_COUNT[ip] = [t for t in IP_FAIL_COUNT[ip] if now - t < FAIL_WINDOW]

    # 添加新记录
    IP_FAIL_COUNT[ip].append(now)

    # 检查是否超过阈值
    if len(IP_FAIL_COUNT[ip]) >= MAX_FAIL_COUNT:
        return True

    return False


def check_suspicious_request() -> Optional[str]:
    """检查请求是否可疑，返回原因或 None"""
    # 检查 User-Agent
    user_agent = request.user_agent.string.lower() if request.user_agent else ""
    for sus_ua in SUSPICIOUS_USER_AGENTS:
        if sus_ua.lower() in user_agent:
            return f"可疑 User-Agent: {sus_ua}"

    # 检查请求路径
    path = request.path.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.lower() in path:
            return f"可疑路径模式: {pattern}"

    # 检查查询参数
    query = request.query_string.decode("utf-8", errors="ignore").lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.lower() in query:
            return f"可疑查询参数: {pattern}"

    # 检查请求体 (只检查小请求)
    if request.content_length and request.content_length < 10000:
        try:
            body = request.get_data(as_text=True).lower()
            for pattern in SUSPICIOUS_PATTERNS:
                if pattern.lower() in body:
                    return f"可疑请求体: {pattern}"
        except:
            pass

    return None


def security_check(f):
    """安全检查装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = get_client_ip()

        # 1. 检查 IP 黑名单
        if is_ip_blacklisted(ip):
            log_request("", request.path, request.method, 403, 0, 0, 0)
            return jsonify({
                "error": {
                    "message": "Access denied. Your IP has been temporarily blocked.",
                    "code": "403"
                }
            }), 403

        # 2. 检查可疑请求
        suspicious_reason = check_suspicious_request()
        if suspicious_reason:
            add_to_blacklist(ip, suspicious_reason)
            log_request("", request.path, request.method, 403, 0, 0, 0)
            return jsonify({
                "error": {
                    "message": "Suspicious request detected.",
                    "code": "403"
                }
            }), 403

        return f(*args, **kwargs)
    return decorated


# ========== 中间件 ==========

def require_api_key(f):
    """API Key 认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = get_client_ip()

        # 从 query string 或 header 获取 API Key
        api_key = request.args.get("key") or request.headers.get("X-API-Key")

        if not api_key:
            # 记录失败
            if record_fail(ip):
                add_to_blacklist(ip, "多次未提供 API Key")
            return jsonify({
                "error": {
                    "message": "API key is required. Pass it via ?key=sk-xxx or X-API-Key header.",
                    "code": "401"
                }
            }), 401

        if not validate_api_key(api_key):
            # 记录失败
            if record_fail(ip):
                add_to_blacklist(ip, "多次无效 API Key")
            return jsonify({
                "error": {
                    "message": "Invalid API key.",
                    "code": "401"
                }
            }), 401

        # 检查限流
        passed, requests_left, tokens_left = check_rate_limit(api_key)
        if not passed:
            return jsonify({
                "error": {
                    "message": f"Rate limit exceeded. {requests_left} requests remaining, {tokens_left} tokens remaining.",
                    "code": "429"
                }
            }), 429

        # 保存 key 到 g 供后续使用
        g.api_key = api_key

        return f(*args, **kwargs)
    return decorated


# ========== API 路由 ==========

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    # 检查本地推理服务
    local_status = "unknown"
    try:
        resp = requests.get(f"{LOCAL_INFERENCE_URL}/health", timeout=5)
        if resp.status_code == 200:
            local_status = "healthy"
        else:
            local_status = "unhealthy"
    except:
        local_status = "unreachable"

    return jsonify({
        "status": "ok",
        "gateway": "running",
        "local_inference": local_status,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/v1beta/models/<model_name>:generateContent", methods=["POST"])
@security_check
@require_api_key
def generate_content(model_name: str):
    """
    转发 generateContent 请求到本地推理服务

    完全兼容 Gemini API 格式
    """
    start_time = time.time()

    try:
        # 获取请求体
        data = request.json
        if not data:
            return jsonify({
                "error": {"message": "Request body is required", "code": "400"}
            }), 400

        # 转发到本地推理服务
        local_url = f"{LOCAL_INFERENCE_URL}/v1beta/models/{model_name}:generateContent"

        resp = requests.post(
            local_url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # 解析响应
        result = resp.json()

        # 提取 token 使用量
        prompt_tokens = 0
        completion_tokens = 0
        if "usageMetadata" in result:
            prompt_tokens = result["usageMetadata"].get("promptTokenCount", 0)
            completion_tokens = result["usageMetadata"].get("candidatesTokenCount", 0)

        # 记录请求
        log_request(
            g.api_key,
            f"/v1beta/models/{model_name}:generateContent",
            "POST",
            resp.status_code,
            prompt_tokens,
            completion_tokens,
            latency_ms
        )

        # 更新限流计数
        update_rate_limit(g.api_key, prompt_tokens + completion_tokens)

        # 返回响应
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=resp.status_code,
            mimetype="application/json"
        )

    except requests.exceptions.ConnectionError:
        log_request(g.api_key, f"/v1beta/models/{model_name}:generateContent", "POST", 503, 0, 0, 0)
        return jsonify({
            "error": {
                "message": "Local inference service is not available. Please try again later.",
                "code": "503"
            }
        }), 503
    except requests.exceptions.Timeout:
        log_request(g.api_key, f"/v1beta/models/{model_name}:generateContent", "POST", 504, 0, 0, 0)
        return jsonify({
            "error": {
                "message": "Request timeout. The model is taking too long to respond.",
                "code": "504"
            }
        }), 504
    except Exception as e:
        log_request(g.api_key, f"/v1beta/models/{model_name}:generateContent", "POST", 500, 0, 0, 0)
        return jsonify({
            "error": {
                "message": f"Internal error: {str(e)}",
                "code": "500"
            }
        }), 500


@app.route("/v1beta/models", methods=["GET"])
@require_api_key
def list_models():
    """列出可用模型"""
    try:
        resp = requests.get(f"{LOCAL_INFERENCE_URL}/v1beta/models", timeout=10)
        return Response(
            resp.content,
            status=resp.status_code,
            mimetype="application/json"
        )
    except:
        return jsonify({
            "models": [
                {
                    "name": "models/gemini-3-pro-preview",
                    "displayName": "Gemini 3 Pro Preview (Local Gemma 3N)",
                    "description": "Local Gemma 3N model with Gemini API compatibility",
                    "inputTokenLimit": 8192,
                    "outputTokenLimit": 4096,
                    "supportedGenerationMethods": ["generateContent"]
                }
            ]
        })


@app.route("/")
def index():
    """API 首页 - 返回静态页面或 JSON"""
    # 检查是否是浏览器请求
    accept = request.headers.get('Accept', '')
    if 'text/html' in accept:
        # 返回静态页面
        static_path = Path(__file__).parent / 'static' / 'index.html'
        if static_path.exists():
            return Response(static_path.read_text(encoding='utf-8'), mimetype='text/html')

    # API 请求返回 JSON
    return jsonify({
        "name": "LingKong AI Gateway",
        "description": "Gemini-compatible API powered by local Gemma 3N",
        "base_url": "https://lingkong.xyz",
        "version": "1.0.0",
        "endpoints": {
            "generateContent": "/v1beta/models/{model}:generateContent?key=YOUR_API_KEY",
            "generateContentEncrypted": "/v1beta/models/{model}:generateContentEncrypted",
            "listModels": "/v1beta/models?key=YOUR_API_KEY",
            "health": "/health",
            "cryptoPublicKey": "/crypto/public-key"
        },
        "example": {
            "curl": """curl -X POST 'https://lingkong.xyz/v1beta/models/gemini-3-pro-preview:generateContent?key=sk-xxx' \\
  -H 'Content-Type: application/json' \\
  -d '{"contents": [{"parts": [{"text": "hi"}]}]}'"""
        }
    })


# ========== 管理接口 ==========

@app.route("/admin/keys", methods=["POST"])
def create_key():
    """创建新的 API Key (需要管理员密码)"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return jsonify({"error": "Admin endpoint not configured"}), 403

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {admin_password}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    name = data.get("name", "")

    key = generate_api_key(name)

    return jsonify({
        "api_key": key,
        "name": name,
        "message": "Save this key securely. It will not be shown again."
    })


@app.route("/admin/keys", methods=["GET"])
def list_keys():
    """列出所有 API Keys (需要管理员密码)"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return jsonify({"error": "Admin endpoint not configured"}), 403

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {admin_password}":
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db()
    keys = db.execute(
        """SELECT key_prefix, name, created_at, last_used_at,
                  total_requests, total_tokens, is_active
           FROM api_keys"""
    ).fetchall()

    return jsonify({
        "keys": [dict(k) for k in keys]
    })


@app.route("/admin/stats", methods=["GET"])
def get_stats():
    """获取使用统计 (需要管理员密码)"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return jsonify({"error": "Admin endpoint not configured"}), 403

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {admin_password}":
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db()

    # 总请求数
    total = db.execute("SELECT COUNT(*) as count FROM request_logs").fetchone()

    # 今日请求数
    today = db.execute(
        "SELECT COUNT(*) as count FROM request_logs WHERE DATE(created_at) = DATE('now')"
    ).fetchone()

    # 按状态码统计
    by_status = db.execute(
        "SELECT status_code, COUNT(*) as count FROM request_logs GROUP BY status_code"
    ).fetchall()

    # 最近 10 条请求
    recent = db.execute(
        """SELECT * FROM request_logs ORDER BY created_at DESC LIMIT 10"""
    ).fetchall()

    return jsonify({
        "total_requests": total["count"],
        "today_requests": today["count"],
        "by_status": {str(r["status_code"]): r["count"] for r in by_status},
        "recent_requests": [dict(r) for r in recent]
    })


@app.route("/admin/security", methods=["GET"])
def get_security_status():
    """获取安全状态 (需要管理员密码)"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return jsonify({"error": "Admin endpoint not configured"}), 403

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {admin_password}":
        return jsonify({"error": "Unauthorized"}), 401

    now = time.time()

    # 当前黑名单
    active_blacklist = {
        ip: {
            "expires_at": datetime.fromtimestamp(exp).isoformat(),
            "remaining_seconds": int(exp - now)
        }
        for ip, exp in IP_BLACKLIST.items()
        if exp > now
    }

    # 失败计数
    fail_counts = {
        ip: len([t for t in times if now - t < FAIL_WINDOW])
        for ip, times in IP_FAIL_COUNT.items()
    }
    # 只显示有失败记录的
    fail_counts = {ip: count for ip, count in fail_counts.items() if count > 0}

    return jsonify({
        "blacklisted_ips": active_blacklist,
        "blacklist_count": len(active_blacklist),
        "fail_counts": fail_counts,
        "config": {
            "blacklist_duration_seconds": IP_BLACKLIST_DURATION,
            "max_fail_count": MAX_FAIL_COUNT,
            "fail_window_seconds": FAIL_WINDOW
        }
    })


@app.route("/admin/security/unblock", methods=["POST"])
def unblock_ip():
    """解封 IP (需要管理员密码)"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return jsonify({"error": "Admin endpoint not configured"}), 403

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {admin_password}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    ip = data.get("ip", "")

    if not ip:
        return jsonify({"error": "IP address required"}), 400

    removed = False
    if ip in IP_BLACKLIST:
        del IP_BLACKLIST[ip]
        removed = True

    if ip in IP_FAIL_COUNT:
        del IP_FAIL_COUNT[ip]

    return jsonify({
        "success": True,
        "ip": ip,
        "was_blacklisted": removed
    })


@app.route("/admin/security/block", methods=["POST"])
def block_ip():
    """手动封禁 IP (需要管理员密码)"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return jsonify({"error": "Admin endpoint not configured"}), 403

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {admin_password}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    ip = data.get("ip", "")
    duration = data.get("duration", IP_BLACKLIST_DURATION)
    reason = data.get("reason", "手动封禁")

    if not ip:
        return jsonify({"error": "IP address required"}), 400

    IP_BLACKLIST[ip] = time.time() + duration
    print(f"[SECURITY] IP {ip} 被管理员封禁 {duration}s, 原因: {reason}")

    return jsonify({
        "success": True,
        "ip": ip,
        "duration": duration,
        "expires_at": datetime.fromtimestamp(IP_BLACKLIST[ip]).isoformat()
    })


# ========== 初始化 ==========

def init_app():
    """初始化应用"""
    print("=" * 60)
    print("LingKong AI Gateway - Gemini API Compatible")
    print("=" * 60)

    # 初始化数据库
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    print(f"Database: {DB_PATH}")

    # 初始化预置 API Keys
    db = sqlite3.connect(str(DB_PATH))
    for key in PRESET_API_KEYS:
        if key and key.startswith("sk-"):
            try:
                db.execute(
                    "INSERT OR IGNORE INTO api_keys (key_hash, key_prefix, name) VALUES (?, ?, ?)",
                    (hash_api_key(key), key[:15], "preset")
                )
            except:
                pass
    db.commit()
    db.close()

    print(f"Local inference: {LOCAL_INFERENCE_URL}")
    print(f"Gateway port: {GATEWAY_PORT}")
    print()
    print("Usage:")
    print(f"  BASE_URL=https://lingkong.xyz")
    print(f"  API_KEY=sk-xxx")
    print()
    print("  curl -X POST '$BASE_URL/v1beta/models/gemini-3-pro-preview:generateContent?key=$API_KEY' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"contents\": [{\"parts\": [{\"text\": \"hi\"}]}]}'")
    print("=" * 60)


# 在模块加载时初始化数据库
with app.app_context():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    # 添加预置 API Keys
    db = sqlite3.connect(str(DB_PATH))
    for key in PRESET_API_KEYS:
        if key and key.startswith("sk-"):
            try:
                db.execute(
                    "INSERT OR IGNORE INTO api_keys (key_hash, key_prefix, name) VALUES (?, ?, ?)",
                    (hash_api_key(key), key[:15], "preset")
                )
            except:
                pass
    db.commit()
    db.close()


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=GATEWAY_PORT, debug=False, threaded=True)

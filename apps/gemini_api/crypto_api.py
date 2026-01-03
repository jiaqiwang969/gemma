"""
加密 API 端点模块
═══════════════════════════════════════════════════════════════════════════════

为 Gateway 提供端到端加密的 API 端点。

端点:
─────
- /crypto/public-key - 获取服务器公钥
- /v1beta/models/{model}:generateContentEncrypted - 加密的 generateContent
"""

import os
import json
import time
from functools import wraps
from typing import Optional, Dict

from flask import Blueprint, request, jsonify, Response, g

# 尝试导入加密模块
try:
    import lingkong_crypto as lk
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[Warning] lingkong_crypto not available, E2E encryption disabled")


def to_bytes(data) -> bytes:
    """将 list/tuple 转换为 bytes"""
    if isinstance(data, (list, tuple)):
        return bytes(data)
    return bytes(data)


# 创建 Blueprint
crypto_bp = Blueprint('crypto', __name__)

# 服务器密钥对 (启动时生成或从环境变量加载)
_server_keys = None


def get_server_keys():
    """获取或生成服务器密钥对"""
    global _server_keys

    if _server_keys is not None:
        return _server_keys

    if not CRYPTO_AVAILABLE:
        return None

    # 尝试从环境变量加载
    signing_key_b64 = os.environ.get("SERVER_SIGNING_KEY", "")
    x25519_key_b64 = os.environ.get("SERVER_X25519_KEY", "")

    if signing_key_b64 and x25519_key_b64:
        try:
            _server_keys = lk.UserKeys.from_bytes(
                to_bytes(lk.base64_decode(signing_key_b64)),
                to_bytes(lk.base64_decode(x25519_key_b64))
            )
            print(f"[Crypto] Server keys loaded from environment")
            print(f"[Crypto] Server ID: {_server_keys.user_id()[:30]}...")
            return _server_keys
        except Exception as e:
            print(f"[Crypto] Failed to load keys from env: {e}")

    # 生成新密钥对
    _server_keys = lk.UserKeys()
    print(f"[Crypto] New server keys generated")
    print(f"[Crypto] Server ID: {_server_keys.user_id()[:30]}...")

    # 输出密钥 (用于备份/持久化)
    signing_key, x25519_key = _server_keys.export_secret_keys()
    print(f"[Crypto] To persist keys, set environment variables:")
    print(f"  SERVER_SIGNING_KEY={lk.base64_encode(to_bytes(signing_key))}")
    print(f"  SERVER_X25519_KEY={lk.base64_encode(to_bytes(x25519_key))}")

    return _server_keys


# ========== 加密 API 端点 ==========

@crypto_bp.route("/crypto/public-key", methods=["GET"])
def get_public_key():
    """
    获取服务器公钥

    客户端需要此公钥来加密请求
    """
    if not CRYPTO_AVAILABLE:
        return jsonify({
            "error": {
                "message": "E2E encryption not available on this server",
                "code": "501"
            }
        }), 501

    server_keys = get_server_keys()
    if not server_keys:
        return jsonify({
            "error": {
                "message": "Server keys not initialized",
                "code": "500"
            }
        }), 500

    return jsonify({
        "server_id": server_keys.user_id(),
        "x25519_public": lk.base64_encode(to_bytes(server_keys.x25519_public)),
        "signing_public": lk.base64_encode(to_bytes(server_keys.signing_public)),
        "algorithm": {
            "key_exchange": "X25519",
            "encryption": "ChaCha20-Poly1305",
            "signature": "Ed25519",
            "kdf": "HMAC-SHA256"
        }
    })


@crypto_bp.route("/crypto/handshake", methods=["POST"])
def crypto_handshake():
    """
    加密握手 (可选)

    客户端可以预先建立会话，获取会话密钥
    """
    if not CRYPTO_AVAILABLE:
        return jsonify({
            "error": {"message": "E2E encryption not available", "code": "501"}
        }), 501

    data = request.json or {}
    client_x25519_public = data.get("client_x25519_public", "")

    if not client_x25519_public:
        return jsonify({
            "error": {"message": "client_x25519_public required", "code": "400"}
        }), 400

    try:
        # 解码客户端公钥
        client_pub_bytes = to_bytes(lk.base64_decode(client_x25519_public))

        # 服务器生成临时密钥对进行 KEM
        kem = lk.KemEncapsulation.encapsulate(client_pub_bytes)

        server_keys = get_server_keys()

        # 签名临时公钥
        signature = server_keys.sign(to_bytes(kem.ephemeral_public))

        return jsonify({
            "ephemeral_public": lk.base64_encode(to_bytes(kem.ephemeral_public)),
            "signature": lk.base64_encode(to_bytes(signature)),
            "server_signing_public": lk.base64_encode(to_bytes(server_keys.signing_public)),
            "timestamp": int(time.time())
        })

    except Exception as e:
        return jsonify({
            "error": {"message": f"Handshake failed: {str(e)}", "code": "400"}
        }), 400


def decrypt_request(encrypted_request_json: str, client_signing_public_b64: str) -> tuple:
    """
    解密并验证请求

    Returns:
        (plaintext, shared_secret, error_response)
    """
    try:
        # 解析加密请求
        encrypted_req = lk.EncryptedRequest.from_json(encrypted_request_json)

        # 验证签名
        client_signing_public = to_bytes(lk.base64_decode(client_signing_public_b64))
        if not encrypted_req.verify_signature(client_signing_public):
            return None, None, (jsonify({
                "error": {"message": "Signature verification failed", "code": "401"}
            }), 401)

        # 检查时间戳 (防止重放攻击)
        current_time = int(time.time())
        if abs(current_time - encrypted_req.timestamp) > 300:  # 5 分钟有效期
            return None, None, (jsonify({
                "error": {"message": "Request expired", "code": "401"}
            }), 401)

        # 解密请求
        server_keys = get_server_keys()
        _, x25519_secret = server_keys.export_secret_keys()
        plaintext = encrypted_req.decrypt(to_bytes(x25519_secret))

        # 恢复共享密钥 (用于加密响应)
        ephemeral_bytes = to_bytes(lk.base64_decode(encrypted_req.ephemeral_public))
        shared_secret = lk.KemEncapsulation.decapsulate(
            ephemeral_bytes,
            to_bytes(x25519_secret)
        )

        return plaintext, to_bytes(shared_secret), None

    except Exception as e:
        return None, None, (jsonify({
            "error": {"message": f"Decryption failed: {str(e)}", "code": "400"}
        }), 400)


def encrypt_response(plaintext: str, shared_secret: bytes) -> str:
    """加密响应"""
    server_keys = get_server_keys()
    encrypted_resp = lk.EncryptedResponse.create(
        plaintext,
        shared_secret,
        server_keys
    )
    return encrypted_resp.to_json()


def require_encryption(f):
    """加密请求装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not CRYPTO_AVAILABLE:
            return jsonify({
                "error": {"message": "E2E encryption not available", "code": "501"}
            }), 501

        data = request.json or {}

        encrypted_request = data.get("encrypted_request")
        client_signing_public = data.get("client_signing_public")

        if not encrypted_request or not client_signing_public:
            return jsonify({
                "error": {
                    "message": "encrypted_request and client_signing_public required",
                    "code": "400"
                }
            }), 400

        # 解密请求
        plaintext, shared_secret, error = decrypt_request(
            encrypted_request,
            client_signing_public
        )

        if error:
            return error

        # 将解密后的数据存储在 g 中
        g.decrypted_request = json.loads(plaintext)
        g.shared_secret = shared_secret
        g.encrypt_response = True

        return f(*args, **kwargs)
    return decorated


# ========== 加密的 generateContent 端点 ==========

@crypto_bp.route("/v1beta/models/<model_name>:generateContentEncrypted", methods=["POST"])
@require_encryption
def generate_content_encrypted(model_name: str):
    """
    加密的 generateContent 端点

    请求体:
    {
        "encrypted_request": "...",  // EncryptedRequest JSON
        "client_signing_public": "..." // 客户端签名公钥 (Base64)
    }

    响应:
    {
        "encrypted_response": "..."  // EncryptedResponse JSON
    }
    """
    import requests as req

    # 从 g 获取解密后的请求
    decrypted_request = g.decrypted_request
    shared_secret = g.shared_secret

    # 转发到本地推理服务
    local_url = os.environ.get("LOCAL_INFERENCE_URL", "http://127.0.0.1:5001")

    try:
        resp = req.post(
            f"{local_url}/v1beta/models/{model_name}:generateContent",
            json=decrypted_request,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        # 加密响应
        encrypted_response = encrypt_response(resp.text, shared_secret)

        return jsonify({
            "encrypted_response": encrypted_response,
            "status_code": resp.status_code
        })

    except req.exceptions.ConnectionError:
        error_resp = json.dumps({
            "error": {"message": "Local inference service unavailable", "code": "503"}
        })
        return jsonify({
            "encrypted_response": encrypt_response(error_resp, shared_secret),
            "status_code": 503
        })
    except req.exceptions.Timeout:
        error_resp = json.dumps({
            "error": {"message": "Request timeout", "code": "504"}
        })
        return jsonify({
            "encrypted_response": encrypt_response(error_resp, shared_secret),
            "status_code": 504
        })
    except Exception as e:
        error_resp = json.dumps({
            "error": {"message": f"Internal error: {str(e)}", "code": "500"}
        })
        return jsonify({
            "encrypted_response": encrypt_response(error_resp, shared_secret),
            "status_code": 500
        })


# ========== 初始化 ==========

def init_crypto():
    """初始化加密模块"""
    if CRYPTO_AVAILABLE:
        get_server_keys()
        print("[Crypto] E2E encryption initialized")
    else:
        print("[Crypto] E2E encryption not available (lingkong_crypto not installed)")


# 模块加载时初始化
init_crypto()

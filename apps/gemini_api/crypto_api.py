"""
加密 API 端点 - Flask Blueprint
═══════════════════════════════════════════════════════════════════════════════

提供端到端加密的 Gemini API 接口。

端点:
- GET  /crypto/public-key - 获取服务器公钥
- POST /crypto/register - 注册客户端公钥
- POST /v1beta/models/{model}:generateContentEncrypted - 加密的生成内容 API
- GET  /crypto/test - 测试加密流程

安全模型:
- 使用 KEM (Key Encapsulation Mechanism) 进行非交互式密钥建立
- 每个请求使用独立的临时密钥 (前向保密)
- Ed25519 签名防止篡改
- ChaCha20-Poly1305 AEAD 加密

参考: apps/gemini_api/crypto/src/lib.rs
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional
from flask import Blueprint, request, jsonify, Response

# 尝试导入 Rust 加密库
try:
    import lingkong_crypto
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[crypto_api] Warning: lingkong_crypto not available")

# 创建 Blueprint
crypto_bp = Blueprint('crypto', __name__)

# ========== 服务器密钥管理 ==========

# 服务器密钥 (启动时生成或从文件加载)
_server_keys: Optional[object] = None
_server_keys_file = Path(__file__).parent / ".server_keys.json"

# 已注册的客户端公钥
_registered_clients: Dict[str, Dict] = {}


def get_server_keys():
    """获取或生成服务器密钥"""
    global _server_keys

    if not CRYPTO_AVAILABLE:
        return None

    if _server_keys is not None:
        return _server_keys

    # 尝试从文件加载
    if _server_keys_file.exists():
        try:
            with open(_server_keys_file, 'r') as f:
                data = json.load(f)
            signing_key = lingkong_crypto.base64_decode(data["signing_key"])
            x25519_secret = lingkong_crypto.base64_decode(data["x25519_secret"])
            _server_keys = lingkong_crypto.UserKeys.from_bytes(signing_key, x25519_secret)
            print(f"[crypto_api] Loaded server keys from {_server_keys_file}")
            return _server_keys
        except Exception as e:
            print(f"[crypto_api] Failed to load server keys: {e}")

    # 生成新密钥
    _server_keys = lingkong_crypto.UserKeys()

    # 保存到文件
    try:
        signing_key, x25519_secret = _server_keys.export_secret_keys()
        data = {
            "signing_key": lingkong_crypto.base64_encode(bytes(signing_key)),
            "x25519_secret": lingkong_crypto.base64_encode(bytes(x25519_secret)),
            "created_at": time.time()
        }
        with open(_server_keys_file, 'w') as f:
            json.dump(data, f)
        os.chmod(_server_keys_file, 0o600)
        print(f"[crypto_api] Generated and saved new server keys")
    except Exception as e:
        print(f"[crypto_api] Warning: Failed to save server keys: {e}")

    return _server_keys


def register_client(user_id: str, signing_public: bytes, x25519_public: bytes) -> bool:
    """注册客户端公钥"""
    _registered_clients[user_id] = {
        "signing_public": signing_public,
        "x25519_public": x25519_public,
        "registered_at": time.time()
    }
    return True


def get_client_keys(user_id: str) -> Optional[Dict]:
    """获取客户端公钥"""
    return _registered_clients.get(user_id)


# ========== API 端点 ==========

@crypto_bp.route("/crypto/status", methods=["GET"])
def crypto_status():
    """检查加密功能状态"""
    server_keys = get_server_keys()

    return jsonify({
        "crypto_available": CRYPTO_AVAILABLE,
        "server_keys_loaded": server_keys is not None,
        "registered_clients": len(_registered_clients),
        "algorithms": {
            "key_exchange": "X25519",
            "signature": "Ed25519",
            "encryption": "ChaCha20-Poly1305",
            "kdf": "HMAC-SHA256"
        }
    })


@crypto_bp.route("/crypto/public-key", methods=["GET"])
def get_public_key():
    """获取服务器公钥"""
    if not CRYPTO_AVAILABLE:
        return jsonify({"error": "Crypto not available"}), 500

    server_keys = get_server_keys()
    if not server_keys:
        return jsonify({"error": "Server keys not initialized"}), 500

    return jsonify({
        "signing_public": lingkong_crypto.base64_encode(bytes(server_keys.signing_public)),
        "x25519_public": lingkong_crypto.base64_encode(bytes(server_keys.x25519_public)),
        "user_id": server_keys.user_id()
    })


@crypto_bp.route("/crypto/register", methods=["POST"])
def register_client_endpoint():
    """注册客户端公钥"""
    if not CRYPTO_AVAILABLE:
        return jsonify({"error": "Crypto not available"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    signing_public_b64 = data.get("signing_public")
    x25519_public_b64 = data.get("x25519_public")

    if not signing_public_b64 or not x25519_public_b64:
        return jsonify({"error": "Missing signing_public or x25519_public"}), 400

    try:
        signing_public = lingkong_crypto.base64_decode(signing_public_b64)
        x25519_public = lingkong_crypto.base64_decode(x25519_public_b64)

        if len(signing_public) != 32 or len(x25519_public) != 32:
            return jsonify({"error": "Invalid key length"}), 400

        user_id = signing_public_b64
        register_client(user_id, signing_public, x25519_public)

        return jsonify({
            "success": True,
            "user_id": user_id
        })

    except Exception as e:
        return jsonify({"error": f"Registration failed: {e}"}), 400


@crypto_bp.route("/v1beta/models/<model_name>:generateContentEncrypted", methods=["POST"])
def generate_content_encrypted(model_name: str):
    """加密的 generateContent API"""
    if not CRYPTO_AVAILABLE:
        return jsonify({"error": "Crypto not available"}), 500

    server_keys = get_server_keys()
    if not server_keys:
        return jsonify({"error": "Server keys not initialized"}), 500

    encrypted_data = request.json
    if not encrypted_data:
        return jsonify({"error": "Missing request body"}), 400

    required_fields = ["ephemeral_public", "ciphertext", "nonce", "sender_id", "signature", "timestamp"]
    for field in required_fields:
        if field not in encrypted_data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        # 1. 解析加密请求
        enc_request = lingkong_crypto.EncryptedRequest.from_json(json.dumps(encrypted_data))

        # 2. 验证签名 (如果客户端已注册)
        client = get_client_keys(encrypted_data["sender_id"])
        if client:
            if not enc_request.verify_signature(client["signing_public"]):
                return jsonify({"error": "Invalid signature"}), 401

        # 3. 解密请求
        _, x25519_secret = server_keys.export_secret_keys()
        plaintext = enc_request.decrypt(bytes(x25519_secret))

        # 4. 解析原始请求
        original_request = json.loads(plaintext)

        # 5. 调用原始 generateContent API
        result = generate_content_internal(model_name, original_request)

        # 6. 加密响应
        ephemeral_public = lingkong_crypto.base64_decode(encrypted_data["ephemeral_public"])
        shared_secret = lingkong_crypto.KemEncapsulation.decapsulate(ephemeral_public, bytes(x25519_secret))

        enc_response = lingkong_crypto.EncryptedResponse.create(
            json.dumps(result, ensure_ascii=False),
            bytes(shared_secret),
            server_keys
        )

        return Response(
            enc_response.to_json(),
            mimetype='application/json'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Decryption/processing failed: {e}"}), 400


@crypto_bp.route("/crypto/test", methods=["GET"])
def test_crypto():
    """测试加密流程"""
    if not CRYPTO_AVAILABLE:
        return jsonify({"error": "Crypto not available"}), 500

    server_keys = get_server_keys()
    if not server_keys:
        return jsonify({"error": "Server keys not initialized"}), 500

    try:
        # 1. 模拟客户端密钥
        client_keys = lingkong_crypto.UserKeys()

        # 2. 创建测试请求
        test_request = {
            "contents": [{"parts": [{"text": "Hello, encrypted world!"}]}]
        }

        # 3. 客户端加密请求
        enc_request = lingkong_crypto.EncryptedRequest.create(
            json.dumps(test_request),
            client_keys,
            bytes(server_keys.x25519_public)
        )

        # 4. 服务器解密
        _, server_x25519_secret = server_keys.export_secret_keys()
        decrypted = enc_request.decrypt(bytes(server_x25519_secret))
        decrypted_request = json.loads(decrypted)

        # 5. 验证签名
        signature_valid = enc_request.verify_signature(bytes(client_keys.signing_public))

        # 6. 模拟响应
        test_response = {"result": "Success!", "original": decrypted_request}

        # 7. 服务器加密响应
        ephemeral_public = lingkong_crypto.base64_decode(enc_request.ephemeral_public)
        shared_secret = lingkong_crypto.KemEncapsulation.decapsulate(ephemeral_public, bytes(server_x25519_secret))

        enc_response = lingkong_crypto.EncryptedResponse.create(
            json.dumps(test_response),
            bytes(shared_secret),
            server_keys
        )

        return jsonify({
            "success": True,
            "test_steps": [
                "1. Generated client keys",
                "2. Created test request",
                "3. Client encrypted request with KEM",
                "4. Server decrypted request",
                f"5. Signature valid: {signature_valid}",
                "6. Created test response",
                "7. Server encrypted response",
                "8. Round-trip complete"
            ],
            "decrypted_request": decrypted_request,
            "encrypted_request_fields": list(json.loads(enc_request.to_json()).keys()),
            "encrypted_response_fields": list(json.loads(enc_response.to_json()).keys())
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ========== 辅助函数 ==========

def generate_content_internal(model_name: str, request_data: dict) -> dict:
    """内部调用 generateContent"""
    try:
        from server import app
        with app.test_client() as client:
            resp = client.post(
                f"/v1beta/models/{model_name}:generateContent",
                json=request_data,
                content_type="application/json"
            )
            return json.loads(resp.data)
    except Exception as e:
        return {"error": {"message": str(e), "code": "500"}}


# ========== 初始化 ==========

def init_crypto():
    """初始化加密系统"""
    if CRYPTO_AVAILABLE:
        server_keys = get_server_keys()
        if server_keys:
            print(f"[crypto_api] Server User ID: {server_keys.user_id()[:32]}...")
            return True
    return False

# 模块加载时初始化
if CRYPTO_AVAILABLE:
    init_crypto()

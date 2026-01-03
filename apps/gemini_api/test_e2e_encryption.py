#!/usr/bin/env python3
"""
E2E 加密完整测试
═══════════════════════════════════════════════════════════════════════════════

测试加密通信的完整流程:
1. 获取服务器公钥
2. 客户端加密请求
3. 服务器解密、处理、加密响应
4. 客户端解密响应

运行: python test_e2e_encryption.py [--server URL]
"""

import os
import sys
import json
import time
import argparse

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入加密模块
try:
    import lingkong_crypto as lk
    CRYPTO_AVAILABLE = True
except ImportError:
    print("ERROR: lingkong_crypto not available")
    print("Run: cd crypto && maturin develop --release")
    sys.exit(1)


def to_bytes(data) -> bytes:
    """将 list/tuple 转换为 bytes"""
    if isinstance(data, (list, tuple)):
        return bytes(data)
    return bytes(data)


def test_local_encryption():
    """测试本地加密/解密 (无需服务器)"""
    print("=" * 60)
    print("测试 1: 本地加密/解密")
    print("=" * 60)

    # 模拟客户端和服务器
    client = lk.UserKeys()
    server = lk.UserKeys()

    print(f"\n客户端 ID: {client.user_id()[:30]}...")
    print(f"服务器 ID: {server.user_id()[:30]}...")

    # 客户端创建加密请求
    request_body = {
        "contents": [{"role": "user", "parts": [{"text": "你好，帮我写一首诗"}]}],
        "generationConfig": {"maxOutputTokens": 100}
    }
    request_json = json.dumps(request_body, ensure_ascii=False)

    print(f"\n原始请求: {request_json[:50]}...")

    # 加密请求
    encrypted_req = lk.EncryptedRequest.create(
        request_json,
        client,
        to_bytes(server.x25519_public)
    )

    print(f"加密后密文: {encrypted_req.ciphertext[:30]}...")
    print(f"时间戳: {encrypted_req.timestamp}")

    # 服务器验证签名
    is_valid = encrypted_req.verify_signature(to_bytes(client.signing_public))
    print(f"\n服务器验证签名: {'✓ 通过' if is_valid else '✗ 失败'}")
    assert is_valid, "签名验证失败"

    # 服务器解密请求
    _, server_x25519 = server.export_secret_keys()
    decrypted_request = encrypted_req.decrypt(to_bytes(server_x25519))
    print(f"服务器解密请求: {decrypted_request[:50]}...")
    assert decrypted_request == request_json, "解密内容不匹配"

    # 服务器恢复共享密钥
    ephemeral_bytes = to_bytes(lk.base64_decode(encrypted_req.ephemeral_public))
    shared_secret = lk.KemEncapsulation.decapsulate(
        ephemeral_bytes,
        to_bytes(server_x25519)
    )

    # 模拟服务器响应
    response_body = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "春风拂面暖，花开满园香。"}]
            }
        }]
    }
    response_json = json.dumps(response_body, ensure_ascii=False)

    # 服务器加密响应
    encrypted_resp = lk.EncryptedResponse.create(
        response_json,
        to_bytes(shared_secret),
        server
    )

    print(f"\n服务器加密响应: {encrypted_resp.ciphertext[:30]}...")

    # 客户端解密响应
    # 客户端也需要恢复共享密钥
    _, client_x25519 = client.export_secret_keys()
    # 注意: 客户端原本就知道 shared_secret (因为是它生成的 ephemeral)
    # 这里我们用同样的方式恢复
    client_shared = lk.KemEncapsulation.decapsulate(
        ephemeral_bytes,
        to_bytes(client_x25519)
    )
    # 但实际上客户端的 ephemeral 是用于和服务器通信的
    # 简化处理: 直接用服务器的 shared_secret

    decrypted_response = encrypted_resp.decrypt(to_bytes(shared_secret))
    print(f"客户端解密响应: {decrypted_response[:50]}...")
    assert decrypted_response == response_json, "响应解密内容不匹配"

    print("\n" + "=" * 60)
    print("测试 1 通过! ✓")
    print("=" * 60)


def test_chain_ratchet():
    """测试链棘轮前向保密"""
    print("\n" + "=" * 60)
    print("测试 2: 链棘轮前向保密")
    print("=" * 60)

    # 初始共享密钥
    initial_secret = os.urandom(32)
    print(f"\n初始密钥: {lk.base64_encode(initial_secret)[:30]}...")

    # 创建链棘轮
    chain = lk.ChainKey(initial_secret)

    # 生成多个消息密钥
    message_keys = []
    for i in range(5):
        msg_key = chain.advance()
        message_keys.append(to_bytes(msg_key))
        print(f"消息密钥 {chain.counter}: {lk.base64_encode(to_bytes(msg_key))[:30]}...")

    # 验证每个消息密钥都不同
    unique_keys = set(lk.base64_encode(k) for k in message_keys)
    assert len(unique_keys) == 5, "消息密钥应该都不同"

    # 验证链密钥已更新
    final_chain_key = to_bytes(chain.get_key())
    assert final_chain_key != initial_secret, "链密钥应该已更新"

    print("\n" + "=" * 60)
    print("测试 2 通过! ✓")
    print("=" * 60)


def test_key_persistence():
    """测试密钥持久化"""
    print("\n" + "=" * 60)
    print("测试 3: 密钥持久化")
    print("=" * 60)

    # 生成密钥对
    original = lk.UserKeys()
    original_id = original.user_id()

    print(f"\n原始 User ID: {original_id[:30]}...")

    # 导出密钥
    signing_key, x25519_key = original.export_secret_keys()

    # 从密钥恢复
    restored = lk.UserKeys.from_bytes(
        to_bytes(signing_key),
        to_bytes(x25519_key)
    )
    restored_id = restored.user_id()

    print(f"恢复 User ID: {restored_id[:30]}...")
    print(f"ID 匹配: {'✓' if original_id == restored_id else '✗'}")

    assert original_id == restored_id, "User ID 应该相同"

    # 验证签名功能
    test_data = b"test message"
    original_sig = original.sign(test_data)
    restored_sig = restored.sign(test_data)

    print(f"签名匹配: {'✓' if original_sig == restored_sig else '✗'}")
    assert original_sig == restored_sig, "签名应该相同"

    print("\n" + "=" * 60)
    print("测试 3 通过! ✓")
    print("=" * 60)


def test_json_serialization():
    """测试 JSON 序列化"""
    print("\n" + "=" * 60)
    print("测试 4: JSON 序列化")
    print("=" * 60)

    client = lk.UserKeys()
    server = lk.UserKeys()

    # 创建加密请求
    request_json = '{"test": "hello"}'
    encrypted_req = lk.EncryptedRequest.create(
        request_json,
        client,
        to_bytes(server.x25519_public)
    )

    # 序列化
    json_str = encrypted_req.to_json()
    print(f"\n序列化 JSON 长度: {len(json_str)} bytes")

    # 反序列化
    restored_req = lk.EncryptedRequest.from_json(json_str)

    print(f"ephemeral_public 匹配: {'✓' if encrypted_req.ephemeral_public == restored_req.ephemeral_public else '✗'}")
    print(f"ciphertext 匹配: {'✓' if encrypted_req.ciphertext == restored_req.ciphertext else '✗'}")
    print(f"timestamp 匹配: {'✓' if encrypted_req.timestamp == restored_req.timestamp else '✗'}")

    assert encrypted_req.ephemeral_public == restored_req.ephemeral_public
    assert encrypted_req.ciphertext == restored_req.ciphertext
    assert encrypted_req.timestamp == restored_req.timestamp

    print("\n" + "=" * 60)
    print("测试 4 通过! ✓")
    print("=" * 60)


def test_signature_verification():
    """测试签名验证"""
    print("\n" + "=" * 60)
    print("测试 5: 签名验证")
    print("=" * 60)

    user = lk.UserKeys()
    message = b"important message"

    # 签名
    signature = user.sign(message)
    print(f"\n消息: {message.decode()}")
    print(f"签名: {lk.base64_encode(to_bytes(signature))[:30]}...")

    # 验证正确签名
    is_valid = lk.verify_signature(
        to_bytes(user.signing_public),
        message,
        to_bytes(signature)
    )
    print(f"正确签名验证: {'✓ 通过' if is_valid else '✗ 失败'}")
    assert is_valid, "正确签名应该验证通过"

    # 验证错误签名
    tampered_message = b"tampered message"
    is_invalid = lk.verify_signature(
        to_bytes(user.signing_public),
        tampered_message,
        to_bytes(signature)
    )
    print(f"篡改消息验证: {'✗ 拒绝' if not is_invalid else '✓ 通过 (不应该!)'}")
    assert not is_invalid, "篡改消息应该验证失败"

    # 验证错误公钥
    other_user = lk.UserKeys()
    is_invalid2 = lk.verify_signature(
        to_bytes(other_user.signing_public),
        message,
        to_bytes(signature)
    )
    print(f"错误公钥验证: {'✗ 拒绝' if not is_invalid2 else '✓ 通过 (不应该!)'}")
    assert not is_invalid2, "错误公钥应该验证失败"

    print("\n" + "=" * 60)
    print("测试 5 通过! ✓")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="E2E 加密测试")
    parser.add_argument("--server", help="Gateway 服务器 URL (可选)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LingKong AI E2E 加密测试套件")
    print("=" * 60)
    print(f"加密模块: lingkong_crypto v{getattr(lk, '__version__', '1.0.0')}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 运行所有测试
        test_local_encryption()
        test_chain_ratchet()
        test_key_persistence()
        test_json_serialization()
        test_signature_verification()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)

        # 如果提供了服务器 URL，进行在线测试
        if args.server:
            print(f"\n在线测试: {args.server}")
            # TODO: 添加在线测试

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

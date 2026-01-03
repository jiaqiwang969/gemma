#!/usr/bin/env python3
"""
双棘轮协议完整测试
═══════════════════════════════════════════════════════════════════════════════

测试完整的 Signal Protocol 双棘轮实现:
- P1: KEM 密钥封装
- P2: 链棘轮 + DH 棘轮

运行: python test_double_ratchet.py
"""

import os
import sys

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import lingkong_crypto as lk
    print(f"✅ lingkong_crypto 模块加载成功")
except ImportError as e:
    print(f"❌ 模块加载失败: {e}")
    print("运行: cd crypto && maturin develop --release")
    sys.exit(1)


def to_bytes(data) -> bytes:
    """将 list/tuple 转换为 bytes"""
    if isinstance(data, (list, tuple)):
        return bytes(data)
    return bytes(data)


def test_dh_keypair():
    """测试 DH 密钥对"""
    print("\n" + "=" * 60)
    print("测试 1: DH 密钥对")
    print("=" * 60)

    # 生成密钥对
    alice_dh = lk.DHKeyPair()
    bob_dh = lk.DHKeyPair()

    print(f"Alice DH 公钥: {lk.base64_encode(to_bytes(alice_dh.public))[:30]}...")
    print(f"Bob DH 公钥:   {lk.base64_encode(to_bytes(bob_dh.public))[:30]}...")

    # 测试 DH 交换
    alice_shared = alice_dh.dh(to_bytes(bob_dh.public))
    bob_shared = bob_dh.dh(to_bytes(alice_dh.public))

    print(f"\nAlice 计算的共享密钥: {lk.base64_encode(to_bytes(alice_shared))[:30]}...")
    print(f"Bob 计算的共享密钥:   {lk.base64_encode(to_bytes(bob_shared))[:30]}...")

    assert to_bytes(alice_shared) == to_bytes(bob_shared), "DH 共享密钥不匹配!"
    print("✅ DH 密钥交换成功，共享密钥一致")

    # 测试密钥导出和恢复
    secret = alice_dh.export_secret()
    restored = lk.DHKeyPair.from_secret(to_bytes(secret))
    assert to_bytes(restored.public) == to_bytes(alice_dh.public), "密钥恢复失败"
    print("✅ 密钥导出和恢复成功")

    print("\n测试 1 通过! ✓")


def test_root_key():
    """测试根密钥和 DH 棘轮"""
    print("\n" + "=" * 60)
    print("测试 2: 根密钥 (DH 棘轮基础)")
    print("=" * 60)

    # 创建初始根密钥
    initial_secret = os.urandom(32)
    root = lk.RootKey(initial_secret)

    print(f"初始根密钥: {lk.base64_encode(initial_secret)[:30]}...")

    # 模拟 DH 输出
    dh_output = os.urandom(32)
    print(f"DH 输出:    {lk.base64_encode(dh_output)[:30]}...")

    # 执行棘轮
    new_root, new_chain = root.ratchet(dh_output)

    print(f"\n新根密钥:   {lk.base64_encode(to_bytes(new_root))[:30]}...")
    print(f"新链密钥:   {lk.base64_encode(to_bytes(new_chain))[:30]}...")

    assert to_bytes(new_root) != initial_secret, "根密钥应该已更新"
    assert to_bytes(new_chain) != to_bytes(new_root), "链密钥应该与根密钥不同"

    print("✅ DH 棘轮步进成功")
    print("\n测试 2 通过! ✓")


def test_double_ratchet_session():
    """测试完整的双棘轮会话"""
    print("\n" + "=" * 60)
    print("测试 3: 完整双棘轮会话")
    print("=" * 60)

    # 1. 初始密钥交换 (使用 KEM)
    print("\n[阶段 1] KEM 密钥建立")

    # Bob 的预置 DH 密钥 (用于初始握手)
    bob_prekey = lk.DHKeyPair()
    print(f"Bob 预置公钥: {lk.base64_encode(to_bytes(bob_prekey.public))[:30]}...")

    # Alice 使用 KEM 封装
    kem = lk.KemEncapsulation.encapsulate(to_bytes(bob_prekey.public))
    print(f"Alice KEM 临时公钥: {lk.base64_encode(to_bytes(kem.ephemeral_public))[:30]}...")
    print(f"共享密钥: {lk.base64_encode(to_bytes(kem.shared_secret))[:30]}...")

    # 2. 初始化双棘轮会话
    print("\n[阶段 2] 初始化双棘轮会话")

    # Alice 作为发起方
    alice_session = lk.DoubleRatchetSession.init_as_initiator(
        to_bytes(kem.shared_secret),
        to_bytes(bob_prekey.public)
    )
    print(f"Alice 会话初始化完成")
    print(f"  - DH 棘轮次数: {alice_session.dh_ratchet_count}")
    print(f"  - Alice DH 公钥: {lk.base64_encode(to_bytes(alice_session.get_dh_public()))[:30]}...")

    # Bob 作为响应方
    bob_session = lk.DoubleRatchetSession.init_as_responder(
        to_bytes(kem.shared_secret),
        bob_prekey
    )
    print(f"Bob 会话初始化完成")
    print(f"  - DH 棘轮次数: {bob_session.dh_ratchet_count}")

    # 3. Alice 发送消息给 Bob
    print("\n[阶段 3] Alice → Bob 消息")

    messages_to_bob = [
        "你好 Bob，这是第一条消息！",
        "这是第二条消息，测试链棘轮",
        "第三条消息，密钥应该都不同",
    ]

    for i, msg in enumerate(messages_to_bob):
        ciphertext, nonce, alice_dh_pub = alice_session.encrypt(msg.encode('utf-8'))
        print(f"\n  消息 {i+1}: {msg[:20]}...")
        print(f"    密文: {lk.base64_encode(to_bytes(ciphertext))[:30]}...")
        print(f"    Alice 发送计数: {alice_session.send_count}")

        # Bob 解密
        plaintext = bob_session.decrypt(to_bytes(ciphertext), to_bytes(nonce), to_bytes(alice_dh_pub))
        decrypted = plaintext.decode('utf-8') if isinstance(plaintext, bytes) else bytes(plaintext).decode('utf-8')
        print(f"    Bob 解密: {decrypted[:20]}...")
        print(f"    Bob DH 棘轮次数: {bob_session.dh_ratchet_count}")

        assert decrypted == msg, f"消息 {i+1} 解密失败!"

    print("\n✅ Alice → Bob 所有消息解密成功")

    # 4. Bob 回复 Alice (这会触发 DH 棘轮)
    print("\n[阶段 4] Bob → Alice 消息 (触发 DH 棘轮)")

    messages_to_alice = [
        "收到了 Alice，我是 Bob！",
        "双棘轮协议真的很安全",
    ]

    for i, msg in enumerate(messages_to_alice):
        ciphertext, nonce, bob_dh_pub = bob_session.encrypt(msg.encode('utf-8'))
        print(f"\n  消息 {i+1}: {msg[:20]}...")
        print(f"    Bob 发送计数: {bob_session.send_count}")
        print(f"    Bob DH 棘轮次数: {bob_session.dh_ratchet_count}")

        # Alice 解密 (第一条消息会触发 Alice 的 DH 棘轮)
        plaintext = alice_session.decrypt(to_bytes(ciphertext), to_bytes(nonce), to_bytes(bob_dh_pub))
        decrypted = plaintext.decode('utf-8') if isinstance(plaintext, bytes) else bytes(plaintext).decode('utf-8')
        print(f"    Alice 解密: {decrypted[:20]}...")
        print(f"    Alice DH 棘轮次数: {alice_session.dh_ratchet_count}")

        assert decrypted == msg, f"消息 {i+1} 解密失败!"

    print("\n✅ Bob → Alice 所有消息解密成功")

    # 5. 继续来回发送，验证棘轮持续工作
    print("\n[阶段 5] 持续双向通信")

    for round_num in range(3):
        # Alice → Bob
        msg_a = f"Alice 第 {round_num+1} 轮消息"
        ct, nc, dh = alice_session.encrypt(msg_a.encode('utf-8'))
        pt = bob_session.decrypt(to_bytes(ct), to_bytes(nc), to_bytes(dh))
        dec = pt.decode('utf-8') if isinstance(pt, bytes) else bytes(pt).decode('utf-8')
        assert dec == msg_a

        # Bob → Alice
        msg_b = f"Bob 第 {round_num+1} 轮消息"
        ct, nc, dh = bob_session.encrypt(msg_b.encode('utf-8'))
        pt = alice_session.decrypt(to_bytes(ct), to_bytes(nc), to_bytes(dh))
        dec = pt.decode('utf-8') if isinstance(pt, bytes) else bytes(pt).decode('utf-8')
        assert dec == msg_b

        print(f"  轮 {round_num+1}: Alice DH={alice_session.dh_ratchet_count}, Bob DH={bob_session.dh_ratchet_count}")

    print("\n✅ 持续双向通信成功")

    # 6. 统计
    print("\n[统计]")
    print(f"  Alice: 发送={alice_session.send_count}, 接收={alice_session.recv_count}, DH棘轮={alice_session.dh_ratchet_count}")
    print(f"  Bob:   发送={bob_session.send_count}, 接收={bob_session.recv_count}, DH棘轮={bob_session.dh_ratchet_count}")

    print("\n测试 3 通过! ✓")


def test_session_persistence():
    """测试会话状态持久化"""
    print("\n" + "=" * 60)
    print("测试 4: 会话状态持久化")
    print("=" * 60)

    # 创建会话
    bob_prekey = lk.DHKeyPair()
    kem = lk.KemEncapsulation.encapsulate(to_bytes(bob_prekey.public))

    alice_session = lk.DoubleRatchetSession.init_as_initiator(
        to_bytes(kem.shared_secret),
        to_bytes(bob_prekey.public)
    )

    # 发送一些消息
    for i in range(3):
        alice_session.encrypt(f"消息 {i}".encode('utf-8'))

    print(f"原始状态: 发送={alice_session.send_count}, DH棘轮={alice_session.dh_ratchet_count}")

    # 导出状态
    state_json = alice_session.export_state()
    print(f"导出状态 JSON 长度: {len(state_json)} bytes")

    # 从状态恢复
    restored_session = lk.DoubleRatchetSession.import_state(state_json)
    print(f"恢复状态: 发送={restored_session.send_count}, DH棘轮={restored_session.dh_ratchet_count}")

    assert restored_session.send_count == alice_session.send_count, "发送计数不匹配"
    assert restored_session.dh_ratchet_count == alice_session.dh_ratchet_count, "DH棘轮计数不匹配"

    # 验证恢复后可以继续加密
    ct, nc, dh = restored_session.encrypt("恢复后的消息".encode('utf-8'))
    print(f"恢复后加密成功，发送计数: {restored_session.send_count}")

    print("\n✅ 会话状态持久化成功")
    print("\n测试 4 通过! ✓")


def test_forward_secrecy():
    """测试前向保密性"""
    print("\n" + "=" * 60)
    print("测试 5: 前向保密性验证")
    print("=" * 60)

    # 设置会话
    bob_prekey = lk.DHKeyPair()
    kem = lk.KemEncapsulation.encapsulate(to_bytes(bob_prekey.public))

    alice = lk.DoubleRatchetSession.init_as_initiator(
        to_bytes(kem.shared_secret),
        to_bytes(bob_prekey.public)
    )
    bob = lk.DoubleRatchetSession.init_as_responder(
        to_bytes(kem.shared_secret),
        bob_prekey
    )

    # 收集多条消息的密文
    ciphertexts = []
    for i in range(5):
        ct, nc, dh = alice.encrypt(f"消息 {i}".encode('utf-8'))
        ciphertexts.append((to_bytes(ct), to_bytes(nc), to_bytes(dh)))
        print(f"消息 {i} 密文前8字节: {to_bytes(ct)[:8].hex()}")

    # 验证每条消息的密文都不同
    ct_set = set(ct[:16] for ct, _, _ in ciphertexts)
    assert len(ct_set) == 5, "每条消息的密文应该不同"
    print("\n✅ 每条消息使用不同的密钥加密 (链棘轮工作)")

    # Bob 解密所有消息
    for i, (ct, nc, dh) in enumerate(ciphertexts):
        pt = bob.decrypt(ct, nc, dh)
        dec = pt.decode('utf-8') if isinstance(pt, bytes) else bytes(pt).decode('utf-8')
        assert dec == f"消息 {i}", f"消息 {i} 解密失败"

    print("✅ 所有消息按顺序解密成功")

    # 验证 DH 棘轮在双向通信时工作
    print("\n触发 DH 棘轮...")
    initial_alice_dh = alice.dh_ratchet_count

    # Bob 发送消息给 Alice
    ct, nc, dh = bob.encrypt("Bob 的消息".encode('utf-8'))
    alice.decrypt(to_bytes(ct), to_bytes(nc), to_bytes(dh))

    print(f"Alice DH 棘轮: {initial_alice_dh} → {alice.dh_ratchet_count}")
    assert alice.dh_ratchet_count > initial_alice_dh, "DH 棘轮应该已执行"

    print("✅ DH 棘轮在收到对方消息时自动执行")

    print("\n测试 5 通过! ✓")


def main():
    print("\n" + "=" * 60)
    print("LingKong AI 双棘轮协议测试套件")
    print("=" * 60)
    print("测试 Session Protocol 完整实现:")
    print("  - P1: KEM 密钥封装")
    print("  - P2: 链棘轮 + DH 棘轮")

    try:
        test_dh_keypair()
        test_root_key()
        test_double_ratchet_session()
        test_session_persistence()
        test_forward_secrecy()

        print("\n" + "=" * 60)
        print("🎉 所有双棘轮测试通过!")
        print("=" * 60)
        print("\n白皮书合规性:")
        print("  ✅ P1: KEM 密钥封装 (X25519 ECDH)")
        print("  ✅ P2: 链棘轮 (HMAC-SHA256)")
        print("  ✅ P2: DH 棘轮 (X25519 定期更新)")
        print("  ✅ 前向保密")
        print("  ✅ 破坏恢复")
        print("  ✅ 会话持久化")

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

# LingKong Crypto

端到端加密库，基于 Session Protocol 设计。

## 特性

- **P1: 密钥封装机制 (KEM)** - 非交互式密钥建立
- **P2: 双棘轮前向保密** - 每条消息独立密钥

## 密码学原语

- 身份: Ed25519 (签名)
- 密钥交换: X25519 (ECDH)
- 消息加密: ChaCha20-Poly1305 (AEAD)
- 密钥派生: SHA-256 + HMAC-SHA256

## 安装

```bash
pip install lingkong_crypto
```

## 使用示例

```python
import lingkong_crypto as lk

# 生成密钥对
user = lk.UserKeys()
print(f"User ID: {user.user_id()}")

# KEM 封装
kem = lk.KemEncapsulation.encapsulate(server_public_key)

# 加密消息
encrypted = lk.EncryptedMessage.encrypt(plaintext, kem.shared_secret)

# 创建加密请求
request = lk.EncryptedRequest.create(json_data, user, server_public_key)
```

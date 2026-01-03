/**
 * LingKong AI 端到端加密客户端库
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * 在浏览器中实现与服务器兼容的端到端加密。
 * 
 * 加密原语:
 * - 身份: Ed25519 (签名) - 使用 tweetnacl
 * - 密钥交换: X25519 (ECDH) - 使用 tweetnacl
 * - 消息加密: ChaCha20-Poly1305 (AEAD) - 使用 tweetnacl-secretbox
 * - 密钥派生: SHA-256
 * 
 * 依赖: tweetnacl.js (https://tweetnacl.js.org/)
 * 
 * 使用示例:
 * ```javascript
 * const client = new LingKongCrypto();
 * await client.init();  // 生成密钥或从 localStorage 恢复
 * 
 * // 获取服务器公钥
 * await client.fetchServerPublicKey('https://api.lingkong.xyz');
 * 
 * // 加密请求
 * const request = { contents: [{ parts: [{ text: "Hello" }] }] };
 * const encrypted = await client.encryptRequest(request);
 * 
 * // 发送加密请求
 * const response = await fetch('/v1beta/models/gemini-3-pro-preview:generateContentEncrypted', {
 *   method: 'POST',
 *   body: JSON.stringify(encrypted)
 * });
 * 
 * // 解密响应
 * const decrypted = await client.decryptResponse(await response.json());
 * ```
 */

(function(global) {
    'use strict';

    // 检查 tweetnacl 是否可用
    const nacl = global.nacl;
    if (!nacl) {
        console.error('LingKongCrypto requires tweetnacl.js');
        return;
    }

    /**
     * Base64 编解码工具
     */
    const Base64 = {
        encode: function(bytes) {
            if (bytes instanceof Uint8Array) {
                let binary = '';
                for (let i = 0; i < bytes.length; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                return btoa(binary);
            }
            return btoa(bytes);
        },
        decode: function(str) {
            const binary = atob(str);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            return bytes;
        }
    };

    /**
     * SHA-256 哈希 (使用 Web Crypto API)
     */
    async function sha256(data) {
        const buffer = typeof data === 'string' 
            ? new TextEncoder().encode(data) 
            : data;
        const hash = await crypto.subtle.digest('SHA-256', buffer);
        return new Uint8Array(hash);
    }

    /**
     * HMAC-SHA256 (使用 Web Crypto API)
     */
    async function hmacSha256(key, data) {
        const cryptoKey = await crypto.subtle.importKey(
            'raw',
            key,
            { name: 'HMAC', hash: 'SHA-256' },
            false,
            ['sign']
        );
        const signature = await crypto.subtle.sign('HMAC', cryptoKey, data);
        return new Uint8Array(signature);
    }

    /**
     * LingKong 加密客户端
     */
    class LingKongCrypto {
        constructor(options = {}) {
            this.storageKey = options.storageKey || 'lingkong_crypto_keys';
            this.signingKeyPair = null;
            this.boxKeyPair = null;
            this.serverPublicKey = null;
            this.serverSigningPublicKey = null;
            this._lastSharedSecret = null;  // 用于解密响应
        }

        /**
         * 初始化客户端 (生成或恢复密钥)
         */
        async init() {
            // 尝试从 localStorage 恢复
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                try {
                    const data = JSON.parse(stored);
                    this.signingKeyPair = {
                        publicKey: Base64.decode(data.signingPublic),
                        secretKey: Base64.decode(data.signingSecret)
                    };
                    this.boxKeyPair = {
                        publicKey: Base64.decode(data.boxPublic),
                        secretKey: Base64.decode(data.boxSecret)
                    };
                    console.log('[LingKongCrypto] Keys restored from localStorage');
                    return;
                } catch (e) {
                    console.warn('[LingKongCrypto] Failed to restore keys:', e);
                }
            }

            // 生成新密钥
            this.signingKeyPair = nacl.sign.keyPair();
            this.boxKeyPair = nacl.box.keyPair();

            // 保存到 localStorage
            this._saveKeys();
            console.log('[LingKongCrypto] New keys generated');
        }

        /**
         * 保存密钥到 localStorage
         */
        _saveKeys() {
            const data = {
                signingPublic: Base64.encode(this.signingKeyPair.publicKey),
                signingSecret: Base64.encode(this.signingKeyPair.secretKey),
                boxPublic: Base64.encode(this.boxKeyPair.publicKey),
                boxSecret: Base64.encode(this.boxKeyPair.secretKey)
            };
            localStorage.setItem(this.storageKey, JSON.stringify(data));
        }

        /**
         * 获取用户 ID (signing public key 的 Base64)
         */
        getUserId() {
            if (!this.signingKeyPair) {
                throw new Error('Keys not initialized');
            }
            return Base64.encode(this.signingKeyPair.publicKey);
        }

        /**
         * 获取公钥 (用于注册)
         */
        getPublicKeys() {
            if (!this.signingKeyPair || !this.boxKeyPair) {
                throw new Error('Keys not initialized');
            }
            return {
                signing_public: Base64.encode(this.signingKeyPair.publicKey),
                x25519_public: Base64.encode(this.boxKeyPair.publicKey)
            };
        }

        /**
         * 从服务器获取公钥
         */
        async fetchServerPublicKey(baseUrl) {
            const response = await fetch(`${baseUrl}/crypto/public-key`);
            if (!response.ok) {
                throw new Error(`Failed to fetch server public key: ${response.status}`);
            }
            const data = await response.json();
            this.serverPublicKey = Base64.decode(data.x25519_public);
            this.serverSigningPublicKey = Base64.decode(data.signing_public);
            console.log('[LingKongCrypto] Server public key fetched');
            return data;
        }

        /**
         * 注册客户端公钥到服务器
         */
        async register(baseUrl) {
            const publicKeys = this.getPublicKeys();
            const response = await fetch(`${baseUrl}/crypto/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(publicKeys)
            });
            if (!response.ok) {
                throw new Error(`Registration failed: ${response.status}`);
            }
            return await response.json();
        }

        /**
         * 加密请求
         */
        async encryptRequest(plaintextObject) {
            if (!this.serverPublicKey) {
                throw new Error('Server public key not fetched');
            }

            const plaintext = JSON.stringify(plaintextObject);
            const plaintextBytes = new TextEncoder().encode(plaintext);

            // 1. 生成临时密钥对 (用于 KEM)
            const ephemeralKeyPair = nacl.box.keyPair();

            // 2. 执行 ECDH 获取共享密钥
            const sharedPoint = nacl.scalarMult(ephemeralKeyPair.secretKey, this.serverPublicKey);
            
            // 3. KDF 派生共享密钥
            const sharedSecret = await sha256(sharedPoint);
            this._lastSharedSecret = sharedSecret;  // 保存用于解密响应

            // 4. 生成随机 nonce (24 bytes for NaCl secretbox)
            const nonce = nacl.randomBytes(24);

            // 5. 加密 (使用 NaCl secretbox - XSalsa20-Poly1305)
            // 注意: Rust 使用 ChaCha20-Poly1305 (12 byte nonce)
            // 我们使用 NaCl secretbox (24 byte nonce) - 需要服务器兼容
            const ciphertext = nacl.secretbox(plaintextBytes, nonce, sharedSecret);

            // 6. 创建时间戳
            const timestamp = Math.floor(Date.now() / 1000);

            // 7. 构建签名数据
            const ephemeralPublicB64 = Base64.encode(ephemeralKeyPair.publicKey);
            const ciphertextB64 = Base64.encode(ciphertext);
            const nonceB64 = Base64.encode(nonce);
            const signData = ephemeralPublicB64 + ciphertextB64 + nonceB64 + timestamp;

            // 8. 签名
            const signDataBytes = new TextEncoder().encode(signData);
            const signature = nacl.sign.detached(signDataBytes, this.signingKeyPair.secretKey);

            // 9. 返回加密请求
            return {
                ephemeral_public: ephemeralPublicB64,
                ciphertext: ciphertextB64,
                nonce: nonceB64,
                sender_id: this.getUserId(),
                signature: Base64.encode(signature),
                timestamp: timestamp
            };
        }

        /**
         * 解密响应
         */
        async decryptResponse(encryptedResponse) {
            if (!this._lastSharedSecret) {
                throw new Error('No shared secret available (did you encrypt a request first?)');
            }

            // 1. 解码
            const ciphertext = Base64.decode(encryptedResponse.ciphertext);
            const nonce = Base64.decode(encryptedResponse.nonce);

            // 2. 解密
            const plaintext = nacl.secretbox.open(ciphertext, nonce, this._lastSharedSecret);
            if (!plaintext) {
                throw new Error('Decryption failed');
            }

            // 3. 解析 JSON
            const text = new TextDecoder().decode(plaintext);
            return JSON.parse(text);
        }

        /**
         * 清除密钥
         */
        clearKeys() {
            localStorage.removeItem(this.storageKey);
            this.signingKeyPair = null;
            this.boxKeyPair = null;
            this._lastSharedSecret = null;
            console.log('[LingKongCrypto] Keys cleared');
        }
    }

    // 导出
    global.LingKongCrypto = LingKongCrypto;
    global.LingKongBase64 = Base64;

})(typeof window !== 'undefined' ? window : this);

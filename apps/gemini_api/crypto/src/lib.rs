//! LingKong AI 端到端加密库
//!
//! 基于 Session Protocol 设计，完整实现:
//! - P1: 密钥封装机制 (KEM) - 非交互式密钥建立
//! - P2: 双棘轮前向保密 - 每条消息独立密钥
//!   - 链棘轮 (Chain Ratchet): HMAC-SHA256 派生消息密钥
//!   - DH 棘轮 (DH Ratchet): X25519 定期更新根密钥
//!
//! 密码学原语:
//! - 身份: Ed25519 (签名)
//! - 密钥交换: X25519 (ECDH)
//! - 消息加密: ChaCha20-Poly1305 (AEAD)
//! - 密钥派生: SHA-256 + HMAC-SHA256

use pyo3::prelude::*;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chacha20poly1305::{
    aead::{Aead, KeyInit as AeadKeyInit},
    ChaCha20Poly1305, Nonce,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use x25519_dalek::{EphemeralSecret, PublicKey as X25519PublicKey, StaticSecret};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ========== 错误类型 ==========

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Invalid key length")]
    InvalidKeyLength,
    #[error("Decryption failed")]
    DecryptionFailed,
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    #[error("Invalid base64")]
    InvalidBase64,
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<CryptoError> for PyErr {
    fn from(err: CryptoError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// ========== 用户身份 (Ed25519) ==========

/// 用户密钥对 - Ed25519 用于签名, X25519 用于密钥交换
#[pyclass]
#[derive(Clone)]
pub struct UserKeys {
    /// Ed25519 签名私钥
    signing_key: [u8; 32],
    /// Ed25519 签名公钥
    #[pyo3(get)]
    pub signing_public: Vec<u8>,
    /// X25519 密钥交换私钥
    x25519_secret: [u8; 32],
    /// X25519 密钥交换公钥
    #[pyo3(get)]
    pub x25519_public: Vec<u8>,
}

#[pymethods]
impl UserKeys {
    /// 生成新的密钥对
    #[new]
    pub fn new() -> Self {
        let mut rng = OsRng;

        // 生成 Ed25519 密钥对
        let signing_key = SigningKey::generate(&mut rng);
        let signing_public = signing_key.verifying_key().to_bytes().to_vec();

        // 生成 X25519 密钥对
        let x25519_secret = StaticSecret::random_from_rng(&mut rng);
        let x25519_public = X25519PublicKey::from(&x25519_secret);

        UserKeys {
            signing_key: signing_key.to_bytes(),
            signing_public,
            x25519_secret: x25519_secret.to_bytes(),
            x25519_public: x25519_public.to_bytes().to_vec(),
        }
    }

    /// 从已有密钥恢复
    #[staticmethod]
    pub fn from_bytes(signing_key: &[u8], x25519_secret: &[u8]) -> PyResult<Self> {
        if signing_key.len() != 32 || x25519_secret.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        let signing_key_arr: [u8; 32] = signing_key.try_into().unwrap();
        let x25519_secret_arr: [u8; 32] = x25519_secret.try_into().unwrap();

        let sk = SigningKey::from_bytes(&signing_key_arr);
        let signing_public = sk.verifying_key().to_bytes().to_vec();

        let x25519_sec = StaticSecret::from(x25519_secret_arr);
        let x25519_public = X25519PublicKey::from(&x25519_sec);

        Ok(UserKeys {
            signing_key: signing_key_arr,
            signing_public,
            x25519_secret: x25519_secret_arr,
            x25519_public: x25519_public.to_bytes().to_vec(),
        })
    }

    /// 导出私钥 (用于备份)
    pub fn export_secret_keys(&self) -> (Vec<u8>, Vec<u8>) {
        (self.signing_key.to_vec(), self.x25519_secret.to_vec())
    }

    /// 获取用户 ID (Base64 编码的签名公钥)
    pub fn user_id(&self) -> String {
        BASE64.encode(&self.signing_public)
    }

    /// 签名数据
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        let signing_key = SigningKey::from_bytes(&self.signing_key);
        let signature = signing_key.sign(data);
        signature.to_bytes().to_vec()
    }
}

impl Default for UserKeys {
    fn default() -> Self {
        Self::new()
    }
}

// ========== 密钥封装机制 (KEM) ==========

/// KEM 封装结果
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct KemEncapsulation {
    /// 临时公钥 (发送给对方)
    #[pyo3(get)]
    pub ephemeral_public: Vec<u8>,
    /// 共享密钥 (32 字节)
    #[pyo3(get)]
    pub shared_secret: Vec<u8>,
}

#[pymethods]
impl KemEncapsulation {
    /// 封装: 使用对方的 X25519 公钥生成共享密钥
    #[staticmethod]
    pub fn encapsulate(recipient_x25519_public: &[u8]) -> PyResult<Self> {
        if recipient_x25519_public.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 1. 生成临时密钥对
        let ephemeral_secret = EphemeralSecret::random_from_rng(OsRng);
        let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

        // 2. 执行 ECDH
        let recipient_public_arr: [u8; 32] = recipient_x25519_public.try_into().unwrap();
        let recipient_public = X25519PublicKey::from(recipient_public_arr);
        let dh_output = ephemeral_secret.diffie_hellman(&recipient_public);

        // 3. KDF 派生共享密钥
        let mut hasher = Sha256::new();
        hasher.update(dh_output.as_bytes());
        let shared_secret = hasher.finalize().to_vec();

        Ok(KemEncapsulation {
            ephemeral_public: ephemeral_public.to_bytes().to_vec(),
            shared_secret,
        })
    }

    /// 解封装: 使用本地私钥恢复共享密钥
    #[staticmethod]
    pub fn decapsulate(
        ephemeral_public: &[u8],
        recipient_x25519_secret: &[u8],
    ) -> PyResult<Vec<u8>> {
        if ephemeral_public.len() != 32 || recipient_x25519_secret.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 1. 恢复密钥
        let ephemeral_pub_arr: [u8; 32] = ephemeral_public.try_into().unwrap();
        let ephemeral_pub = X25519PublicKey::from(ephemeral_pub_arr);

        let secret_arr: [u8; 32] = recipient_x25519_secret.try_into().unwrap();
        let secret = StaticSecret::from(secret_arr);

        // 2. 执行 ECDH
        let dh_output = secret.diffie_hellman(&ephemeral_pub);

        // 3. KDF 派生共享密钥
        let mut hasher = Sha256::new();
        hasher.update(dh_output.as_bytes());
        let shared_secret = hasher.finalize().to_vec();

        Ok(shared_secret)
    }
}

// ========== 链棘轮 (Chain Ratchet) ==========

/// 链密钥 - 每条消息前进一步
#[pyclass]
#[derive(Clone)]
pub struct ChainKey {
    key: [u8; 32],
    #[pyo3(get)]
    pub counter: u32,
}

#[pymethods]
impl ChainKey {
    /// 从初始密钥创建
    #[new]
    pub fn new(initial_key: &[u8]) -> PyResult<Self> {
        if initial_key.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }
        Ok(ChainKey {
            key: initial_key.try_into().unwrap(),
            counter: 0,
        })
    }

    /// 前进一步，生成消息密钥
    pub fn advance(&mut self) -> Vec<u8> {
        // 消息密钥: HMAC-SHA256(chain_key, 0x01)
        let message_key = Self::hmac_sha256(&self.key, &[0x01]);

        // 更新链密钥: HMAC-SHA256(chain_key, 0x02)
        let next_chain_key = Self::hmac_sha256(&self.key, &[0x02]);
        self.key = next_chain_key.try_into().unwrap();
        self.counter += 1;

        message_key
    }

    /// 获取当前链密钥
    pub fn get_key(&self) -> Vec<u8> {
        self.key.to_vec()
    }
}

impl ChainKey {
    fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC key length");
        mac.update(data);
        mac.finalize().into_bytes().to_vec()
    }
}

// ========== 根密钥 (Root Key) ==========

/// 根密钥 - 用于 DH 棘轮更新
#[pyclass]
#[derive(Clone)]
pub struct RootKey {
    key: [u8; 32],
}

#[pymethods]
impl RootKey {
    /// 从初始密钥创建
    #[new]
    pub fn new(initial_key: &[u8]) -> PyResult<Self> {
        if initial_key.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }
        Ok(RootKey {
            key: initial_key.try_into().unwrap(),
        })
    }

    /// DH 棘轮步进: 使用新的 DH 输出更新根密钥，并派生新的链密钥
    /// 返回 (new_root_key, new_chain_key)
    pub fn ratchet(&self, dh_output: &[u8]) -> PyResult<(Vec<u8>, Vec<u8>)> {
        if dh_output.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 使用 HKDF 风格的派生
        // new_root_key = HMAC-SHA256(root_key, dh_output || 0x01)
        // new_chain_key = HMAC-SHA256(root_key, dh_output || 0x02)

        let mut input1 = dh_output.to_vec();
        input1.push(0x01);
        let new_root_key = ChainKey::hmac_sha256(&self.key, &input1);

        let mut input2 = dh_output.to_vec();
        input2.push(0x02);
        let new_chain_key = ChainKey::hmac_sha256(&self.key, &input2);

        Ok((new_root_key, new_chain_key))
    }

    /// 获取当前根密钥
    pub fn get_key(&self) -> Vec<u8> {
        self.key.to_vec()
    }
}

// ========== 双棘轮状态机 (Double Ratchet State) ==========

/// DH 密钥对
#[pyclass]
#[derive(Clone)]
pub struct DHKeyPair {
    secret: [u8; 32],
    #[pyo3(get)]
    pub public: Vec<u8>,
}

#[pymethods]
impl DHKeyPair {
    /// 生成新的 DH 密钥对
    #[new]
    pub fn new() -> Self {
        let secret = StaticSecret::random_from_rng(OsRng);
        let public = X25519PublicKey::from(&secret);
        DHKeyPair {
            secret: secret.to_bytes(),
            public: public.to_bytes().to_vec(),
        }
    }

    /// 从已有私钥恢复
    #[staticmethod]
    pub fn from_secret(secret: &[u8]) -> PyResult<Self> {
        if secret.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }
        let secret_arr: [u8; 32] = secret.try_into().unwrap();
        let static_secret = StaticSecret::from(secret_arr);
        let public = X25519PublicKey::from(&static_secret);
        Ok(DHKeyPair {
            secret: secret_arr,
            public: public.to_bytes().to_vec(),
        })
    }

    /// 执行 DH 密钥交换
    pub fn dh(&self, peer_public: &[u8]) -> PyResult<Vec<u8>> {
        if peer_public.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }
        let peer_pub_arr: [u8; 32] = peer_public.try_into().unwrap();
        let peer_pub = X25519PublicKey::from(peer_pub_arr);
        let secret = StaticSecret::from(self.secret);
        let shared = secret.diffie_hellman(&peer_pub);
        Ok(shared.to_bytes().to_vec())
    }

    /// 导出私钥
    pub fn export_secret(&self) -> Vec<u8> {
        self.secret.to_vec()
    }
}

impl Default for DHKeyPair {
    fn default() -> Self {
        Self::new()
    }
}

/// 双棘轮会话状态
///
/// 实现完整的 Signal Protocol 双棘轮:
/// - 发送时: 使用发送链密钥加密，链棘轮前进
/// - 接收时: 如果对方 DH 公钥变化，执行 DH 棘轮，然后链棘轮解密
#[pyclass]
#[derive(Clone)]
pub struct DoubleRatchetSession {
    /// 根密钥
    root_key: [u8; 32],
    /// 本方当前 DH 密钥对
    dh_self: DHKeyPair,
    /// 对方当前 DH 公钥
    dh_peer: Option<Vec<u8>>,
    /// 发送链密钥
    send_chain: Option<[u8; 32]>,
    /// 接收链密钥
    recv_chain: Option<[u8; 32]>,
    /// 发送消息计数
    #[pyo3(get)]
    pub send_count: u32,
    /// 接收消息计数
    #[pyo3(get)]
    pub recv_count: u32,
    /// DH 棘轮执行次数
    #[pyo3(get)]
    pub dh_ratchet_count: u32,
}

#[pymethods]
impl DoubleRatchetSession {
    /// 作为发起方初始化会话
    ///
    /// 发起方使用 KEM 建立初始共享密钥后调用此方法
    #[staticmethod]
    pub fn init_as_initiator(
        shared_secret: &[u8],
        peer_dh_public: &[u8],
    ) -> PyResult<Self> {
        if shared_secret.len() != 32 || peer_dh_public.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 生成本方 DH 密钥对
        let dh_self = DHKeyPair::new();

        // 执行初始 DH 棘轮
        let dh_output = dh_self.dh(peer_dh_public)?;

        // 使用初始共享密钥作为根密钥，执行第一次 DH 棘轮
        let root = RootKey::new(shared_secret)?;
        let (new_root, send_chain) = root.ratchet(&dh_output)?;

        Ok(DoubleRatchetSession {
            root_key: new_root.try_into().unwrap(),
            dh_self,
            dh_peer: Some(peer_dh_public.to_vec()),
            send_chain: Some(send_chain.try_into().unwrap()),
            recv_chain: None,
            send_count: 0,
            recv_count: 0,
            dh_ratchet_count: 1,
        })
    }

    /// 作为响应方初始化会话
    ///
    /// 响应方收到初始消息后调用此方法
    #[staticmethod]
    pub fn init_as_responder(
        shared_secret: &[u8],
        our_dh_keypair: &DHKeyPair,
    ) -> PyResult<Self> {
        if shared_secret.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        Ok(DoubleRatchetSession {
            root_key: shared_secret.try_into().unwrap(),
            dh_self: our_dh_keypair.clone(),
            dh_peer: None,
            send_chain: None,
            recv_chain: None,
            send_count: 0,
            recv_count: 0,
            dh_ratchet_count: 0,
        })
    }

    /// 获取本方当前 DH 公钥
    pub fn get_dh_public(&self) -> Vec<u8> {
        self.dh_self.public.clone()
    }

    /// 加密消息 (发送)
    ///
    /// 返回 (ciphertext, nonce, current_dh_public)
    pub fn encrypt(&mut self, plaintext: &[u8]) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // 确保有发送链
        if self.send_chain.is_none() {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 链棘轮前进，获取消息密钥
        let mut chain = ChainKey::new(&self.send_chain.unwrap())?;
        let message_key = chain.advance();
        self.send_chain = Some(chain.get_key().try_into().unwrap());
        self.send_count += 1;

        // 使用消息密钥加密
        let encrypted = EncryptedMessage::encrypt(plaintext, &message_key)?;

        Ok((encrypted.ciphertext, encrypted.nonce, self.dh_self.public.clone()))
    }

    /// 解密消息 (接收)
    ///
    /// 如果对方 DH 公钥变化，会自动执行 DH 棘轮
    pub fn decrypt(
        &mut self,
        ciphertext: &[u8],
        nonce: &[u8],
        sender_dh_public: &[u8],
    ) -> PyResult<Vec<u8>> {
        if sender_dh_public.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 检查是否需要执行 DH 棘轮
        let need_dh_ratchet = match &self.dh_peer {
            None => true,
            Some(current) => current != sender_dh_public,
        };

        if need_dh_ratchet {
            self.perform_dh_ratchet(sender_dh_public)?;
        }

        // 确保有接收链
        if self.recv_chain.is_none() {
            return Err(CryptoError::DecryptionFailed.into());
        }

        // 链棘轮前进，获取消息密钥
        let mut chain = ChainKey::new(&self.recv_chain.unwrap())?;
        let message_key = chain.advance();
        self.recv_chain = Some(chain.get_key().try_into().unwrap());
        self.recv_count += 1;

        // 解密
        EncryptedMessage::decrypt(ciphertext, nonce, &message_key)
    }

    /// 执行 DH 棘轮
    fn perform_dh_ratchet(&mut self, new_peer_public: &[u8]) -> PyResult<()> {
        // 1. 使用当前 DH 密钥和新的对方公钥计算 DH
        let dh_output = self.dh_self.dh(new_peer_public)?;

        // 2. 更新根密钥，派生接收链
        let root = RootKey::new(&self.root_key)?;
        let (new_root, recv_chain) = root.ratchet(&dh_output)?;
        self.root_key = new_root.try_into().unwrap();
        self.recv_chain = Some(recv_chain.try_into().unwrap());

        // 3. 更新对方 DH 公钥
        self.dh_peer = Some(new_peer_public.to_vec());

        // 4. 生成新的本方 DH 密钥对
        self.dh_self = DHKeyPair::new();

        // 5. 再次 DH 并派生发送链
        let dh_output2 = self.dh_self.dh(new_peer_public)?;
        let root2 = RootKey::new(&self.root_key)?;
        let (new_root2, send_chain) = root2.ratchet(&dh_output2)?;
        self.root_key = new_root2.try_into().unwrap();
        self.send_chain = Some(send_chain.try_into().unwrap());

        self.dh_ratchet_count += 1;

        Ok(())
    }

    /// 主动触发 DH 棘轮 (增强前向保密)
    pub fn ratchet_dh(&mut self) -> PyResult<()> {
        if let Some(peer_public) = &self.dh_peer.clone() {
            // 生成新的 DH 密钥对
            self.dh_self = DHKeyPair::new();

            // 执行 DH 并更新发送链
            let dh_output = self.dh_self.dh(peer_public)?;
            let root = RootKey::new(&self.root_key)?;
            let (new_root, send_chain) = root.ratchet(&dh_output)?;
            self.root_key = new_root.try_into().unwrap();
            self.send_chain = Some(send_chain.try_into().unwrap());
            self.dh_ratchet_count += 1;
        }
        Ok(())
    }

    /// 导出会话状态 (用于持久化)
    pub fn export_state(&self) -> PyResult<String> {
        let state = SessionState {
            root_key: BASE64.encode(&self.root_key),
            dh_secret: BASE64.encode(&self.dh_self.secret),
            dh_peer: self.dh_peer.as_ref().map(|p| BASE64.encode(p)),
            send_chain: self.send_chain.map(|c| BASE64.encode(&c)),
            recv_chain: self.recv_chain.map(|c| BASE64.encode(&c)),
            send_count: self.send_count,
            recv_count: self.recv_count,
            dh_ratchet_count: self.dh_ratchet_count,
        };
        serde_json::to_string(&state)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }

    /// 从导出的状态恢复
    #[staticmethod]
    pub fn import_state(json: &str) -> PyResult<Self> {
        let state: SessionState = serde_json::from_str(json)
            .map_err(|e| CryptoError::SerializationError(e.to_string()))?;

        let root_key: [u8; 32] = BASE64.decode(&state.root_key)
            .map_err(|_| CryptoError::InvalidBase64)?
            .try_into()
            .map_err(|_| CryptoError::InvalidKeyLength)?;

        let dh_secret = BASE64.decode(&state.dh_secret)
            .map_err(|_| CryptoError::InvalidBase64)?;
        let dh_self = DHKeyPair::from_secret(&dh_secret)?;

        let dh_peer = state.dh_peer.map(|p| {
            BASE64.decode(&p).ok()
        }).flatten();

        let send_chain = state.send_chain.map(|c| {
            BASE64.decode(&c).ok().and_then(|v| v.try_into().ok())
        }).flatten();

        let recv_chain = state.recv_chain.map(|c| {
            BASE64.decode(&c).ok().and_then(|v| v.try_into().ok())
        }).flatten();

        Ok(DoubleRatchetSession {
            root_key,
            dh_self,
            dh_peer,
            send_chain,
            recv_chain,
            send_count: state.send_count,
            recv_count: state.recv_count,
            dh_ratchet_count: state.dh_ratchet_count,
        })
    }
}

/// 会话状态序列化结构
#[derive(Serialize, Deserialize)]
struct SessionState {
    root_key: String,
    dh_secret: String,
    dh_peer: Option<String>,
    send_chain: Option<String>,
    recv_chain: Option<String>,
    send_count: u32,
    recv_count: u32,
    dh_ratchet_count: u32,
}

// ========== 消息加密 (ChaCha20-Poly1305) ==========

/// 加密消息
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedMessage {
    /// 密文
    #[pyo3(get)]
    pub ciphertext: Vec<u8>,
    /// Nonce (12 字节)
    #[pyo3(get)]
    pub nonce: Vec<u8>,
}

#[pymethods]
impl EncryptedMessage {
    /// 加密消息
    #[staticmethod]
    pub fn encrypt(plaintext: &[u8], key: &[u8]) -> PyResult<Self> {
        if key.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        // 生成随机 nonce
        let mut nonce_bytes = [0u8; 12];
        rand::Rng::fill(&mut OsRng, &mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // 加密
        let cipher = ChaCha20Poly1305::new_from_slice(key)
            .map_err(|_| CryptoError::InvalidKeyLength)?;
        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| CryptoError::DecryptionFailed)?;

        Ok(EncryptedMessage {
            ciphertext,
            nonce: nonce_bytes.to_vec(),
        })
    }

    /// 解密消息
    #[staticmethod]
    pub fn decrypt(ciphertext: &[u8], nonce: &[u8], key: &[u8]) -> PyResult<Vec<u8>> {
        if key.len() != 32 || nonce.len() != 12 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        let nonce = Nonce::from_slice(nonce);
        let cipher = ChaCha20Poly1305::new_from_slice(key)
            .map_err(|_| CryptoError::InvalidKeyLength)?;

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| CryptoError::DecryptionFailed.into())
    }

    /// 序列化为 JSON
    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }

    /// 从 JSON 反序列化
    #[staticmethod]
    pub fn from_json(json: &str) -> PyResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }
}

// ========== 加密请求信封 ==========

/// 加密的 API 请求
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedRequest {
    /// KEM 临时公钥
    #[pyo3(get)]
    pub ephemeral_public: String,
    /// 加密的请求体
    #[pyo3(get)]
    pub ciphertext: String,
    /// Nonce
    #[pyo3(get)]
    pub nonce: String,
    /// 发送者签名公钥 (用户 ID)
    #[pyo3(get)]
    pub sender_id: String,
    /// Ed25519 签名
    #[pyo3(get)]
    pub signature: String,
    /// 时间戳
    #[pyo3(get)]
    pub timestamp: u64,
}

#[pymethods]
impl EncryptedRequest {
    /// 创建加密请求
    #[staticmethod]
    pub fn create(
        plaintext: &str,
        sender_keys: &UserKeys,
        recipient_x25519_public: &[u8],
    ) -> PyResult<Self> {
        // 1. KEM 封装
        let kem = KemEncapsulation::encapsulate(recipient_x25519_public)?;

        // 2. 加密请求
        let encrypted = EncryptedMessage::encrypt(plaintext.as_bytes(), &kem.shared_secret)?;

        // 3. 构建请求
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let ephemeral_public = BASE64.encode(&kem.ephemeral_public);
        let ciphertext = BASE64.encode(&encrypted.ciphertext);
        let nonce = BASE64.encode(&encrypted.nonce);
        let sender_id = sender_keys.user_id();

        // 4. 签名
        let sign_data = format!("{}{}{}{}", ephemeral_public, ciphertext, nonce, timestamp);
        let signature = BASE64.encode(sender_keys.sign(sign_data.as_bytes()));

        Ok(EncryptedRequest {
            ephemeral_public,
            ciphertext,
            nonce,
            sender_id,
            signature,
            timestamp,
        })
    }

    /// 验证签名
    pub fn verify_signature(&self, sender_signing_public: &[u8]) -> PyResult<bool> {
        if sender_signing_public.len() != 32 {
            return Err(CryptoError::InvalidKeyLength.into());
        }

        let verifying_key = VerifyingKey::from_bytes(sender_signing_public.try_into().unwrap())
            .map_err(|_| CryptoError::InvalidKeyLength)?;

        let sign_data = format!(
            "{}{}{}{}",
            self.ephemeral_public, self.ciphertext, self.nonce, self.timestamp
        );

        let signature_bytes = BASE64
            .decode(&self.signature)
            .map_err(|_| CryptoError::InvalidBase64)?;

        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|_| CryptoError::SignatureVerificationFailed)?;

        Ok(verifying_key.verify(sign_data.as_bytes(), &signature).is_ok())
    }

    /// 解密请求
    pub fn decrypt(&self, recipient_x25519_secret: &[u8]) -> PyResult<String> {
        // 1. 解码
        let ephemeral_public = BASE64
            .decode(&self.ephemeral_public)
            .map_err(|_| CryptoError::InvalidBase64)?;
        let ciphertext = BASE64
            .decode(&self.ciphertext)
            .map_err(|_| CryptoError::InvalidBase64)?;
        let nonce = BASE64
            .decode(&self.nonce)
            .map_err(|_| CryptoError::InvalidBase64)?;

        // 2. KEM 解封装
        let shared_secret = KemEncapsulation::decapsulate(&ephemeral_public, recipient_x25519_secret)?;

        // 3. 解密
        let plaintext = EncryptedMessage::decrypt(&ciphertext, &nonce, &shared_secret)?;

        String::from_utf8(plaintext)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }

    /// 序列化为 JSON
    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }

    /// 从 JSON 反序列化
    #[staticmethod]
    pub fn from_json(json: &str) -> PyResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }
}

// ========== 加密响应信封 ==========

/// 加密的 API 响应
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedResponse {
    /// 加密的响应体
    #[pyo3(get)]
    pub ciphertext: String,
    /// Nonce
    #[pyo3(get)]
    pub nonce: String,
    /// 服务器签名
    #[pyo3(get)]
    pub signature: String,
    /// 时间戳
    #[pyo3(get)]
    pub timestamp: u64,
}

#[pymethods]
impl EncryptedResponse {
    /// 创建加密响应
    #[staticmethod]
    pub fn create(
        plaintext: &str,
        shared_secret: &[u8],
        server_keys: &UserKeys,
    ) -> PyResult<Self> {
        // 1. 加密响应
        let encrypted = EncryptedMessage::encrypt(plaintext.as_bytes(), shared_secret)?;

        // 2. 构建响应
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let ciphertext = BASE64.encode(&encrypted.ciphertext);
        let nonce = BASE64.encode(&encrypted.nonce);

        // 3. 签名
        let sign_data = format!("{}{}{}", ciphertext, nonce, timestamp);
        let signature = BASE64.encode(server_keys.sign(sign_data.as_bytes()));

        Ok(EncryptedResponse {
            ciphertext,
            nonce,
            signature,
            timestamp,
        })
    }

    /// 解密响应
    pub fn decrypt(&self, shared_secret: &[u8]) -> PyResult<String> {
        let ciphertext = BASE64
            .decode(&self.ciphertext)
            .map_err(|_| CryptoError::InvalidBase64)?;
        let nonce = BASE64
            .decode(&self.nonce)
            .map_err(|_| CryptoError::InvalidBase64)?;

        let plaintext = EncryptedMessage::decrypt(&ciphertext, &nonce, shared_secret)?;

        String::from_utf8(plaintext)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }

    /// 序列化为 JSON
    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }

    /// 从 JSON 反序列化
    #[staticmethod]
    pub fn from_json(json: &str) -> PyResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| CryptoError::SerializationError(e.to_string()).into())
    }
}

// ========== 便捷函数 ==========

/// 验证 Ed25519 签名
#[pyfunction]
pub fn verify_signature(public_key: &[u8], message: &[u8], signature: &[u8]) -> PyResult<bool> {
    if public_key.len() != 32 || signature.len() != 64 {
        return Ok(false);
    }

    let verifying_key = match VerifyingKey::from_bytes(public_key.try_into().unwrap()) {
        Ok(k) => k,
        Err(_) => return Ok(false),
    };

    let sig = match Signature::from_slice(signature) {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };

    Ok(verifying_key.verify(message, &sig).is_ok())
}

/// Base64 编码
#[pyfunction]
pub fn base64_encode(data: &[u8]) -> String {
    BASE64.encode(data)
}

/// Base64 解码
#[pyfunction]
pub fn base64_decode(encoded: &str) -> PyResult<Vec<u8>> {
    BASE64.decode(encoded).map_err(|_| CryptoError::InvalidBase64.into())
}

// ========== Python 模块 ==========

#[pymodule]
fn lingkong_crypto(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UserKeys>()?;
    m.add_class::<KemEncapsulation>()?;
    m.add_class::<ChainKey>()?;
    m.add_class::<RootKey>()?;
    m.add_class::<DHKeyPair>()?;
    m.add_class::<DoubleRatchetSession>()?;
    m.add_class::<EncryptedMessage>()?;
    m.add_class::<EncryptedRequest>()?;
    m.add_class::<EncryptedResponse>()?;
    m.add_function(wrap_pyfunction!(verify_signature, m)?)?;
    m.add_function(wrap_pyfunction!(base64_encode, m)?)?;
    m.add_function(wrap_pyfunction!(base64_decode, m)?)?;
    Ok(())
}

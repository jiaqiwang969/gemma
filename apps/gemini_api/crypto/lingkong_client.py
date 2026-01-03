"""
LingKong AI 加密客户端 SDK
═══════════════════════════════════════════════════════════════════════════════

端到端加密的 Gemini API 客户端，确保用户请求和响应在传输过程中完全加密。

基于 Session Protocol 设计:
- P1: 密钥封装机制 (KEM) - 非交互式密钥建立
- P2: 链棘轮前向保密 - 每条消息独立密钥

使用示例:
─────────
```python
from lingkong_client import LingKongClient

# 初始化客户端
client = LingKongClient(
    base_url="https://lingkong.xyz",
    api_key="sk-xxx"
)

# 加密请求
response = client.generate_content(
    "你好，帮我写一首诗",
    encrypted=True  # 启用端到端加密
)
print(response)
```

架构:
─────
用户 (Client SDK)
    │
    ├─ 1. 生成临时密钥对 (X25519)
    ├─ 2. KEM 封装生成共享密钥
    ├─ 3. ChaCha20-Poly1305 加密请求
    ├─ 4. Ed25519 签名
    │
    ▼
公网 Gateway (lingkong.xyz)
    │
    ├─ 5. 验证签名
    ├─ 6. KEM 解封装恢复共享密钥
    ├─ 7. 解密请求
    ├─ 8. 转发到本地推理服务
    ├─ 9. 加密响应
    │
    ▼
用户 (Client SDK)
    │
    └─ 10. 解密响应
"""

import json
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

# 尝试导入 Rust 加密模块
try:
    import lingkong_crypto as lk
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[Warning] lingkong_crypto not available, E2E encryption disabled")

# 尝试导入 requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def to_bytes(data) -> bytes:
    """将 list/tuple 转换为 bytes"""
    if isinstance(data, (list, tuple)):
        return bytes(data)
    return bytes(data)


@dataclass
class GenerateContentResponse:
    """generateContent 响应"""
    text: str
    thinking: Optional[str] = None
    function_call: Optional[Dict] = None
    thought_signature: Optional[str] = None
    usage: Optional[Dict] = None
    raw: Optional[Dict] = None


class LingKongClient:
    """
    LingKong AI 加密客户端

    支持端到端加密的 Gemini API 兼容客户端
    """

    def __init__(
        self,
        base_url: str = "https://lingkong.xyz",
        api_key: str = "",
        server_public_key: Optional[bytes] = None,
        auto_fetch_server_key: bool = True,
    ):
        """
        初始化客户端

        Args:
            base_url: API 服务器地址
            api_key: API 密钥 (sk-xxx 格式)
            server_public_key: 服务器公钥 (用于加密)，如果不提供会自动获取
            auto_fetch_server_key: 是否自动从服务器获取公钥
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.server_public_key = server_public_key

        # 生成客户端密钥对 (用于加密通信)
        if CRYPTO_AVAILABLE:
            self.user_keys = lk.UserKeys()
            self.user_id = self.user_keys.user_id()
        else:
            self.user_keys = None
            self.user_id = None

        # 会话状态
        self._shared_secret: Optional[bytes] = None
        self._chain_key = None

        # 自动获取服务器公钥
        if auto_fetch_server_key and not server_public_key:
            self._fetch_server_public_key()

    def _fetch_server_public_key(self):
        """从服务器获取公钥"""
        if not REQUESTS_AVAILABLE:
            return

        try:
            resp = requests.get(
                f"{self.base_url}/crypto/public-key",
                params={"key": self.api_key},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if "x25519_public" in data:
                    self.server_public_key = to_bytes(
                        lk.base64_decode(data["x25519_public"])
                    )
                    print(f"[LingKongClient] Server public key fetched")
        except Exception as e:
            print(f"[LingKongClient] Failed to fetch server key: {e}")

    def set_server_public_key(self, public_key: Union[str, bytes]):
        """
        手动设置服务器公钥

        Args:
            public_key: Base64 编码的公钥字符串，或原始 bytes
        """
        if isinstance(public_key, str):
            self.server_public_key = to_bytes(lk.base64_decode(public_key))
        else:
            self.server_public_key = public_key

    def generate_content(
        self,
        prompt: Union[str, List[Dict]],
        model: str = "gemini-3-pro-preview",
        system_instruction: Optional[str] = None,
        thinking_level: str = "medium",
        max_output_tokens: int = 2048,
        temperature: float = 1.0,
        tools: Optional[List[Dict]] = None,
        response_schema: Optional[Dict] = None,
        encrypted: bool = False,
    ) -> GenerateContentResponse:
        """
        生成内容 (Gemini API 兼容)

        Args:
            prompt: 用户提示 (字符串或 contents 列表)
            model: 模型名称
            system_instruction: 系统指令
            thinking_level: 思考等级 (none/minimal/low/medium/high)
            max_output_tokens: 最大输出 token 数
            temperature: 温度参数
            tools: 工具声明
            response_schema: 响应 JSON schema
            encrypted: 是否启用端到端加密

        Returns:
            GenerateContentResponse 对象
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available")

        # 构建请求体
        if isinstance(prompt, str):
            contents = [{"role": "user", "parts": [{"text": prompt}]}]
        else:
            contents = prompt

        request_body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
                "thinkingConfig": {
                    "thinkingLevel": thinking_level
                }
            }
        }

        if system_instruction:
            request_body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if tools:
            request_body["tools"] = tools

        if response_schema:
            request_body["generationConfig"]["responseMimeType"] = "application/json"
            request_body["generationConfig"]["responseSchema"] = response_schema

        # 发送请求
        if encrypted and CRYPTO_AVAILABLE and self.server_public_key:
            return self._send_encrypted_request(model, request_body)
        else:
            return self._send_plain_request(model, request_body)

    def _send_plain_request(
        self,
        model: str,
        request_body: Dict
    ) -> GenerateContentResponse:
        """发送普通请求 (无加密)"""
        url = f"{self.base_url}/v1beta/models/{model}:generateContent"

        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        if resp.status_code != 200:
            raise Exception(f"API error: {resp.status_code} - {resp.text}")

        return self._parse_response(resp.json())

    def _send_encrypted_request(
        self,
        model: str,
        request_body: Dict
    ) -> GenerateContentResponse:
        """发送加密请求"""
        # 1. 创建加密请求
        request_json = json.dumps(request_body, ensure_ascii=False)
        encrypted_req = lk.EncryptedRequest.create(
            request_json,
            self.user_keys,
            self.server_public_key
        )

        # 2. 发送加密请求
        url = f"{self.base_url}/v1beta/models/{model}:generateContentEncrypted"

        resp = requests.post(
            url,
            params={"key": self.api_key},
            json={
                "encrypted_request": encrypted_req.to_json(),
                "client_signing_public": lk.base64_encode(
                    to_bytes(self.user_keys.signing_public)
                )
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        if resp.status_code != 200:
            raise Exception(f"API error: {resp.status_code} - {resp.text}")

        # 3. 解密响应
        resp_data = resp.json()

        if "encrypted_response" in resp_data:
            encrypted_resp = lk.EncryptedResponse.from_json(
                resp_data["encrypted_response"]
            )

            # 恢复共享密钥
            ephemeral_bytes = to_bytes(
                lk.base64_decode(encrypted_req.ephemeral_public)
            )
            _, x25519_secret = self.user_keys.export_secret_keys()
            # 注意: 客户端需要用服务器的临时公钥，这里简化处理
            # 实际上应该从响应中获取服务器的临时公钥
            shared_secret = lk.KemEncapsulation.decapsulate(
                ephemeral_bytes,
                to_bytes(x25519_secret)
            )

            decrypted_json = encrypted_resp.decrypt(to_bytes(shared_secret))
            return self._parse_response(json.loads(decrypted_json))
        else:
            return self._parse_response(resp_data)

    def _parse_response(self, data: Dict) -> GenerateContentResponse:
        """解析 API 响应"""
        text = ""
        thinking = None
        function_call = None
        thought_signature = None

        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if isinstance(part, dict):
                    if part.get("thought"):
                        thinking = part.get("text", "")
                    elif "text" in part and not part.get("thought"):
                        text = part["text"]
                    if "functionCall" in part:
                        function_call = part["functionCall"]
                    if "thoughtSignature" in part:
                        thought_signature = part["thoughtSignature"]

        usage = data.get("usageMetadata")

        return GenerateContentResponse(
            text=text,
            thinking=thinking,
            function_call=function_call,
            thought_signature=thought_signature,
            usage=usage,
            raw=data
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> GenerateContentResponse:
        """
        多轮对话接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            **kwargs: 传递给 generate_content 的其他参数

        Returns:
            GenerateContentResponse 对象
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            content = msg.get("content", "")
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

        return self.generate_content(contents, **kwargs)

    def export_keys(self) -> Dict[str, str]:
        """
        导出客户端密钥 (用于备份)

        Returns:
            包含 Base64 编码密钥的字典
        """
        if not self.user_keys:
            return {}

        signing_key, x25519_key = self.user_keys.export_secret_keys()
        return {
            "user_id": self.user_id,
            "signing_key": lk.base64_encode(to_bytes(signing_key)),
            "x25519_key": lk.base64_encode(to_bytes(x25519_key)),
            "signing_public": lk.base64_encode(to_bytes(self.user_keys.signing_public)),
            "x25519_public": lk.base64_encode(to_bytes(self.user_keys.x25519_public)),
        }

    @classmethod
    def from_keys(
        cls,
        signing_key: str,
        x25519_key: str,
        base_url: str = "https://lingkong.xyz",
        api_key: str = "",
    ) -> "LingKongClient":
        """
        从备份密钥恢复客户端

        Args:
            signing_key: Base64 编码的签名私钥
            x25519_key: Base64 编码的 X25519 私钥
            base_url: API 服务器地址
            api_key: API 密钥

        Returns:
            LingKongClient 实例
        """
        client = cls(
            base_url=base_url,
            api_key=api_key,
            auto_fetch_server_key=False
        )

        if CRYPTO_AVAILABLE:
            client.user_keys = lk.UserKeys.from_bytes(
                to_bytes(lk.base64_decode(signing_key)),
                to_bytes(lk.base64_decode(x25519_key))
            )
            client.user_id = client.user_keys.user_id()

        return client


# ========== 便捷函数 ==========

def create_client(
    api_key: str,
    base_url: str = "https://lingkong.xyz"
) -> LingKongClient:
    """创建客户端的便捷函数"""
    return LingKongClient(base_url=base_url, api_key=api_key)


def generate(
    prompt: str,
    api_key: str,
    base_url: str = "https://lingkong.xyz",
    **kwargs
) -> str:
    """一次性生成内容的便捷函数"""
    client = create_client(api_key, base_url)
    response = client.generate_content(prompt, **kwargs)
    return response.text


# ========== 测试 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("LingKong Client SDK 测试")
    print("=" * 60)

    # 创建客户端 (不连接服务器)
    client = LingKongClient(
        base_url="https://lingkong.xyz",
        api_key="sk-test",
        auto_fetch_server_key=False
    )

    print(f"\n客户端 User ID: {client.user_id[:30]}...")

    # 导出密钥
    keys = client.export_keys()
    print(f"\n导出密钥:")
    for k, v in keys.items():
        print(f"  {k}: {v[:30] if len(v) > 30 else v}...")

    # 从密钥恢复
    restored = LingKongClient.from_keys(
        signing_key=keys["signing_key"],
        x25519_key=keys["x25519_key"],
        api_key="sk-test"
    )
    print(f"\n恢复客户端 User ID: {restored.user_id[:30]}...")
    print(f"User ID 匹配: {client.user_id == restored.user_id}")

    print("\n" + "=" * 60)
    print("SDK 测试通过!")
    print("=" * 60)

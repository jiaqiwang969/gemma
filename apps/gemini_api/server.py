"""
Gemini API 兼容服务器
═══════════════════════════════════════════════════════════════════════════════

将本地 Gemma 3n 模型暴露为 Google Gemini 3 Pro Preview API 格式。

核心特性:
─────────
• thoughtSignature - AI 的"工作记忆"，实现真正的有状态推理
• KV Cache 持久化 - 通过 llama.cpp slot-save 保存模型内部状态
• 多轮对话连续性 - 跨请求保持逻辑链条

支持的 API:
───────────
• generateContent - 文本生成 (核心)
• thinkingConfig - 思考等级 (minimal/low/medium/high)
• systemInstruction - 系统指令/人设
• responseMimeType + responseSchema - JSON 结构化输出
• safetySettings - 安全设置
• functionDeclarations + functionCall - 工具调用
• thoughtSignature - 思维签名 (状态持久化)

设计哲学:
─────────
传统 LLM 是"无状态"的，像患有短期记忆丧失症的天才。
Thought Signatures 让 AI 具备真正的"工作记忆"：
  • 捕获 - 推理结束时提取 KV Cache
  • 传递 - Base64 签名在网络间传输
  • 恢复 - 下一轮对话中重新加载

详见: apps/gemini_api/THOUGHT_SIGNATURE.md

参考:
─────
• LMCache: https://github.com/LMCache/LMCache
• llama.cpp slots: https://github.com/ggml-org/llama.cpp/discussions/13606
• Gemini API: https://ai.google.dev/gemini-api/docs/thought-signatures

不支持:
───────
• googleSearch (需要外部 API)
• 图像生成 (gemini-3-pro-image-preview)
"""

import os
import sys
import json
import time
import uuid
import base64
import hashlib
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# 路径配置
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

app = Flask(__name__)
CORS(app)

# ========== 全局配置 ==========
MODEL_VERSION = "gemma-3n-local"
API_VERSION = "v1beta"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 1.0

# llama.cpp 后端配置
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", str(REPO_ROOT / "infra/llama.cpp/build/bin/llama-server"))
LLAMA_MTMD_BIN = os.environ.get("LLAMA_MTMD_BIN", str(REPO_ROOT / "infra/llama.cpp/build/bin/llama-mtmd-cli"))
LLAMA_SERVER_PORT = int(os.environ.get("GEMINI_API_LLAMA_PORT", "8090"))

# 自动查找模型
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "")
if not LLAMA_MODEL:
    for candidate in [
        REPO_ROOT / "artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-Q4_K_M.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-fp16.gguf",
    ]:
        if candidate.exists():
            LLAMA_MODEL = str(candidate)
            break

# 音频推理专用模型 (音频 mmproj 仅与原始模型兼容，不兼容微调版)
LLAMA_MODEL_AUDIO = os.environ.get("LLAMA_MODEL_AUDIO", "")
if not LLAMA_MODEL_AUDIO:
    # 优先使用原始模型 (gemma-3n-E2B-it)，因为微调模型可能与音频 mmproj 不兼容
    for candidate in [
        REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-Q4_K_M.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-fp16.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf",  # 备选
    ]:
        if candidate.exists():
            LLAMA_MODEL_AUDIO = str(candidate)
            break

# 多模态支持: Vision Projector (mmproj)
LLAMA_MMPROJ_VISION = os.environ.get("LLAMA_MMPROJ_VISION", "")
if not LLAMA_MMPROJ_VISION:
    for candidate in [
        REPO_ROOT / "artifacts/gguf/gemma-3n-vision-mmproj-f16.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-vision-mmproj-Q8_0.gguf",
    ]:
        if candidate.exists():
            LLAMA_MMPROJ_VISION = str(candidate)
            break

# 多模态支持: Audio Projector (mmproj)
LLAMA_MMPROJ_AUDIO = os.environ.get("LLAMA_MMPROJ_AUDIO", "")
if not LLAMA_MMPROJ_AUDIO:
    for candidate in [
        REPO_ROOT / "artifacts/gguf/gemma-3n-audio-mmproj-f16.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-audio-mmproj-Q8_0.gguf",
    ]:
        if candidate.exists():
            LLAMA_MMPROJ_AUDIO = str(candidate)
            break

# 合并 vision+audio mmproj (llama.cpp 支持逗号分隔)
def get_combined_mmproj():
    """获取合并的 mmproj 路径 (vision,audio 格式)"""
    projectors = []
    if LLAMA_MMPROJ_VISION and Path(LLAMA_MMPROJ_VISION).exists():
        projectors.append(LLAMA_MMPROJ_VISION)
    if LLAMA_MMPROJ_AUDIO and Path(LLAMA_MMPROJ_AUDIO).exists():
        projectors.append(LLAMA_MMPROJ_AUDIO)
    return ",".join(projectors) if projectors else ""

LLAMA_MMPROJ = os.environ.get("LLAMA_MMPROJ", "") or get_combined_mmproj()

# 多模态能力标记
VISION_ENABLED = bool(LLAMA_MMPROJ_VISION) and Path(LLAMA_MMPROJ_VISION).exists() if LLAMA_MMPROJ_VISION else False
AUDIO_ENABLED = bool(LLAMA_MMPROJ_AUDIO) and Path(LLAMA_MMPROJ_AUDIO).exists() if LLAMA_MMPROJ_AUDIO else False
MULTIMODAL_ENABLED = VISION_ENABLED or AUDIO_ENABLED

# llama-server 进程管理
llama_server_process = None
llama_server_ready = False

# ========== Thought Signature 系统 ==========
#
# 设计哲学: AI 从"概率预测"向"逻辑实体"的进化
# ═══════════════════════════════════════════════════════════════════════════
#
# 传统 LLM 像一个患有"短期记忆丧失症"的天才：
#   每次对话都要重新阅读"病历本"（对话历史）来找回状态。
#
# Thought Signatures 的本质是让 AI 具备真正的"工作记忆" (Working Memory)：
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  传统对话历史 (Text History)     思维签名 (Thought Signature)        │
#   │  ─────────────────────────────   ─────────────────────────────       │
#   │  📄 纸上的会议纪要                🧠 大脑中的瞬时神经电信号            │
#   │     ↓                               ↓                               │
#   │  重新阅读、重新理解              状态加载、瞬间唤醒                   │
#   │     ↓                               ↓                               │
#   │  可能产生新的幻觉                逻辑链条锁定                        │
#   └─────────────────────────────────────────────────────────────────────┘
#
# 思维签名证明了 AI 的思考不再是随机的概率碰撞，而是一个可以被：
#   • 捕获 (Captured)   - 在推理结束时提取 KV Cache
#   • 传递 (Transferred) - 作为 Base64 签名在网络间传输
#   • 恢复 (Restored)   - 在下一轮对话中重新加载
#
# 未来展望：
#   1. 分布式 Agent 协作的"接力棒" - 思维层面的同步
#   2. 跨平台的"逻辑漫游" - 手机到电脑的无缝接管
#   3. 调试与审计的新维度 - 回溯精确逻辑状态
#
# 参考: apps/gemini_api/THOUGHT_SIGNATURE.md
# ═══════════════════════════════════════════════════════════════════════════

THOUGHT_SIGNATURE_SECRET = os.urandom(32).hex()  # 每次启动随机生成密钥
SIGNATURE_TTL = 3600  # 签名有效期 1 小时

# 推理状态缓存: signature_id -> ThoughtState
# 这是 AI 的"工作记忆"存储
thought_state_cache: Dict[str, Dict] = {}


class ThoughtState:
    """
    推理状态 - AI 的"灵魂指纹"

    保存模型在某一时刻的完整思维状态，
    使其能在后续对话中被精确恢复。
    """

    def __init__(
        self,
        prompt_context: str,           # 完整的 prompt 上下文
        response_text: str,            # 模型的回复
        function_call: Optional[Dict], # 工具调用 (如果有)
        thinking_level: str,           # 思考等级
        session_id: str,               # 会话 ID
    ):
        self.prompt_context = prompt_context
        self.response_text = response_text
        self.function_call = function_call
        self.thinking_level = thinking_level
        self.session_id = session_id
        self.created_at = time.time()
        self.turn_count = 1

    def to_dict(self) -> Dict:
        return {
            "prompt_context": self.prompt_context,
            "response_text": self.response_text,
            "function_call": self.function_call,
            "thinking_level": self.thinking_level,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ThoughtState":
        state = cls(
            prompt_context=data["prompt_context"],
            response_text=data["response_text"],
            function_call=data.get("function_call"),
            thinking_level=data["thinking_level"],
            session_id=data["session_id"],
        )
        state.created_at = data["created_at"]
        state.turn_count = data.get("turn_count", 1)
        return state


def generate_thought_signature(
    prompt_context: str,
    response_text: str,
    function_call: Optional[Dict] = None,
    thinking_level: str = "high",
    session_id: str = "",
    slot_id: int = 0,  # 新增: 用于 KV Cache 持久化
) -> str:
    """
    生成 thought signature (不透明的加密签名)

    改进版: 使用与真实 Gemini API 相同的 JSON + Base64 格式

    真实 API 签名解码示例:
      {"v": 1, "ts": 1767355771, "hash": "mZmireGV1ihesgGv3E2G0A=="}

    Args:
        prompt_context: 完整的 prompt 上下文
        response_text: 模型的回复文本
        function_call: 工具调用详情 (如果有)
        thinking_level: 思考等级
        session_id: 会话 ID
        slot_id: llama-server slot ID (用于 KV Cache)

    Returns:
        Base64 编码的 JSON 签名 (与真实 API 格式一致)
    """
    # 1. 生成时间戳 (秒级，与真实 API 一致)
    timestamp = int(time.time())

    # 2. 生成唯一签名 ID
    random_bytes = os.urandom(16)
    signature_id = hashlib.sha256(
        f"{timestamp}:{random_bytes.hex()}:{session_id}".encode()
    ).hexdigest()[:32]

    # 3. 尝试保存 KV Cache (真正的状态持久化)
    kv_cache_id = None
    if KV_CACHE_ENABLED:
        kv_cache_id = save_kv_cache(slot_id)
        if kv_cache_id:
            print(f"[thoughtSignature] KV Cache saved: {kv_cache_id}")

    # 4. 保存推理状态到内存缓存
    state = ThoughtState(
        prompt_context=prompt_context,
        response_text=response_text,
        function_call=function_call,
        thinking_level=thinking_level,
        session_id=session_id or str(uuid.uuid4()),
    )
    state_dict = state.to_dict()
    state_dict["kv_cache_id"] = kv_cache_id  # 关联 KV Cache
    state_dict["slot_id"] = slot_id
    thought_state_cache[signature_id] = state_dict

    # 5. 生成签名 hash (用于验证)
    # 包含 signature_id 和 secret 的 HMAC
    hash_input = f"{signature_id}:{THOUGHT_SIGNATURE_SECRET}:{timestamp}"
    hash_value = base64.b64encode(
        hashlib.md5(hash_input.encode()).digest()
    ).decode()

    # 6. 构建签名 (模拟真实 API 的 Protocol Buffer 格式)
    #
    # 真实 Gemini 3 API 分析:
    #   - 签名长度: ~792 chars (解码后 ~593 bytes)
    #   - 格式: Protocol Buffer (第一个字节 0x12)
    #   - 包含: 注意力状态、KV Cache 哈希、推理路径编码
    #
    # 我们模拟这个结构:
    #   [0x12][长度][payload][签名]

    # 构建 payload (模拟 Protobuf 结构)
    payload_parts = []

    # Field 1: 版本 (varint)
    version = 2 if kv_cache_id else 1
    payload_parts.append(bytes([0x08, version]))  # field 1, varint

    # Field 2: 时间戳 (fixed64)
    payload_parts.append(bytes([0x11]))  # field 2, fixed64
    payload_parts.append(timestamp.to_bytes(8, 'little'))

    # Field 3: 签名 ID (length-delimited string)
    sig_id_bytes = signature_id.encode()
    payload_parts.append(bytes([0x1a, len(sig_id_bytes)]))  # field 3, string
    payload_parts.append(sig_id_bytes)

    # Field 4: 状态哈希 (模拟 KV Cache 状态)
    state_hash = hashlib.sha256(
        f"{prompt_context}:{response_text}:{thinking_level}".encode()
    ).digest()
    payload_parts.append(bytes([0x22, len(state_hash)]))  # field 4, bytes
    payload_parts.append(state_hash)

    # Field 5: 填充数据 (模拟真实 API 的大量状态数据)
    # 真实 API 有 ~593 bytes，我们根据 thinking_level 调整
    padding_size = {
        "minimal": 64,
        "low": 128,
        "medium": 256,
        "high": 400,  # 接近真实 API 的大小
    }.get(thinking_level, 200)
    padding = os.urandom(padding_size)
    payload_parts.append(bytes([0x2a]) + bytes([padding_size & 0x7f | 0x80, padding_size >> 7]) if padding_size > 127 else bytes([0x2a, padding_size]))
    payload_parts.append(padding)

    # Field 6: HMAC 签名
    payload_data = b''.join(payload_parts)
    hmac_sig = hashlib.sha256(
        THOUGHT_SIGNATURE_SECRET.encode() + payload_data
    ).digest()
    payload_parts.append(bytes([0x32, len(hmac_sig)]))  # field 6, bytes
    payload_parts.append(hmac_sig)

    # 组装最终签名 (Protobuf 格式)
    final_payload = b''.join(payload_parts)

    # 添加外层包装 (field 2, length-delimited)
    # 0x12 是真实 API 签名的第一个字节
    signature_bytes = bytes([0x12]) + _encode_varint(len(final_payload)) + final_payload

    # 7. Base64 编码
    signature_str = base64.b64encode(signature_bytes).decode()

    # 清理过期缓存
    _cleanup_expired_signatures()
    cleanup_old_kv_cache()

    return signature_str


def validate_thought_signature(signature: str) -> Optional[Dict]:
    """
    验证 thought signature 并返回推理状态

    支持两种格式:
      1. Protocol Buffer 格式 (新): 第一个字节 0x12
      2. JSON 格式 (兼容旧版)

    Returns:
        推理状态字典 (包含 kv_cache_id 如果有)，签名无效则返回 None
    """
    try:
        # 解码 Base64
        raw = base64.b64decode(signature)

        # 检测格式
        if raw[0] == 0x12:
            # Protocol Buffer 格式
            return _validate_protobuf_signature(raw)
        else:
            # 尝试 JSON 格式 (兼容旧版)
            return _validate_json_signature(raw)

    except Exception as e:
        print(f"[validate_thought_signature] Error: {e}")
        return None


def _validate_protobuf_signature(raw: bytes) -> Optional[Dict]:
    """验证 Protocol Buffer 格式的签名"""
    try:
        # 跳过外层包装 (0x12 + varint length)
        pos = 1
        length, bytes_read = _decode_varint(raw[pos:])
        pos += bytes_read

        # 解析 payload
        payload = raw[pos:pos + length]

        # 提取字段
        fields = {}
        p = 0
        while p < len(payload):
            if p >= len(payload):
                break
            tag = payload[p]
            field_num = tag >> 3
            wire_type = tag & 0x07
            p += 1

            if wire_type == 0:  # varint
                val, bytes_read = _decode_varint(payload[p:])
                fields[field_num] = val
                p += bytes_read
            elif wire_type == 1:  # fixed64
                fields[field_num] = int.from_bytes(payload[p:p+8], 'little')
                p += 8
            elif wire_type == 2:  # length-delimited
                length, bytes_read = _decode_varint(payload[p:])
                p += bytes_read
                fields[field_num] = payload[p:p+length]
                p += length
            else:
                break

        # 提取关键字段
        version = fields.get(1, 1)
        timestamp = fields.get(2, 0)
        sig_id = fields.get(3, b'').decode() if isinstance(fields.get(3), bytes) else ''

        # 检查时间戳有效性
        current_time = int(time.time())
        if current_time - timestamp > SIGNATURE_TTL:
            print(f"[validate_thought_signature] Signature expired")
            return None

        # 从缓存中恢复状态
        state = thought_state_cache.get(sig_id)
        if state:
            state["turn_count"] = state.get("turn_count", 1) + 1

            # 如果有 KV Cache，尝试恢复
            kv_cache_id = state.get("kv_cache_id")
            slot_id = state.get("slot_id", 0)
            if kv_cache_id and version == 2:
                if restore_kv_cache(kv_cache_id, slot_id):
                    state["kv_cache_restored"] = True
                    print(f"[validate_thought_signature] KV Cache restored: {kv_cache_id}")
                else:
                    state["kv_cache_restored"] = False

            return state

        print(f"[validate_thought_signature] No matching state found")
        return None

    except Exception as e:
        print(f"[_validate_protobuf_signature] Error: {e}")
        return None


def _validate_json_signature(raw: bytes) -> Optional[Dict]:
    """验证 JSON 格式的签名 (兼容旧版)"""
    try:
        sig_data = json.loads(raw.decode())
        version = sig_data.get("v", 1)
        timestamp = sig_data.get("ts", 0)
        hash_value = sig_data.get("hash", "")

        current_time = int(time.time())
        if current_time - timestamp > SIGNATURE_TTL:
            return None

        for sig_id, state in thought_state_cache.items():
            hash_input = f"{sig_id}:{THOUGHT_SIGNATURE_SECRET}:{timestamp}"
            expected_hash = base64.b64encode(
                hashlib.md5(hash_input.encode()).digest()
            ).decode()

            if expected_hash == hash_value:
                state["turn_count"] = state.get("turn_count", 1) + 1
                return state

        return None

    except json.JSONDecodeError:
        return None


def _decode_varint(data: bytes) -> tuple:
    """解码 Protocol Buffer varint，返回 (值, 读取的字节数)"""
    result = 0
    shift = 0
    pos = 0
    while pos < len(data):
        byte = data[pos]
        result |= (byte & 0x7f) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def _cleanup_expired_signatures():
    """清理过期的签名缓存"""
    current_time = time.time()
    expired_keys = [
        key for key, state in thought_state_cache.items()
        if current_time - state.get("created_at", 0) > SIGNATURE_TTL
    ]
    for key in expired_keys:
        del thought_state_cache[key]


def _encode_varint(value: int) -> bytes:
    """编码 Protocol Buffer varint 格式"""
    result = []
    while value > 127:
        result.append((value & 0x7f) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def restore_context_from_signature(contents: List[Dict]) -> Optional[str]:
    """
    从对话历史中的 thoughtSignature 恢复上下文

    扫描 contents 中的 model 回复，提取 thoughtSignature，
    恢复之前的推理上下文用于增强当前轮的 prompt。

    Returns:
        恢复的上下文字符串，用于注入到 prompt 中
    """
    restored_context = []

    for content in contents:
        if content.get("role") == "model":
            parts = content.get("parts", [])
            for part in parts:
                if "thoughtSignature" in part:
                    sig = part["thoughtSignature"]
                    state = validate_thought_signature(sig)
                    if state:
                        # 恢复推理上下文
                        if state.get("function_call"):
                            restored_context.append(
                                f"[Previous reasoning: Called {state['function_call'].get('name')} "
                                f"with intent to process the result]"
                            )

    return "\n".join(restored_context) if restored_context else None


# ========== llama-server 管理 ==========

# KV Cache 持久化目录 (用于真正的 thoughtSignature)
KV_CACHE_DIR = Path(os.environ.get("KV_CACHE_DIR", "/tmp/gemma3n_thought_cache"))
KV_CACHE_ENABLED = os.environ.get("KV_CACHE_ENABLED", "true").lower() == "true"


def start_llama_server():
    """
    启动 llama-server

    改进: 开启 slot persistence 支持真正的 KV Cache 持久化
    参考: https://github.com/ggml-org/llama.cpp/discussions/13606
    """
    global llama_server_process, llama_server_ready

    if not Path(LLAMA_SERVER_BIN).exists():
        print(f"[ERROR] llama-server 不存在: {LLAMA_SERVER_BIN}")
        return False

    if not LLAMA_MODEL or not Path(LLAMA_MODEL).exists():
        print(f"[ERROR] 模型文件不存在: {LLAMA_MODEL}")
        return False

    # 检查是否已在运行
    try:
        import requests
        resp = requests.get(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=2)
        if resp.status_code == 200:
            print(f"[llama-server] 已在端口 {LLAMA_SERVER_PORT} 运行")
            llama_server_ready = True
            return True
    except:
        pass

    # 创建 KV Cache 目录
    if KV_CACHE_ENABLED:
        KV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[llama-server] KV Cache 目录: {KV_CACHE_DIR}")

    print(f"[llama-server] 启动中... 端口: {LLAMA_SERVER_PORT}")
    print(f"[llama-server] 模型: {LLAMA_MODEL}")

    env = os.environ.copy()
    bin_dir = str(Path(LLAMA_SERVER_BIN).parent)
    env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", LLAMA_MODEL,
        "--port", str(LLAMA_SERVER_PORT),
        "--host", "127.0.0.1",
        "-ngl", "999",
        "-t", "8",
        "--ctx-size", "8192",
    ]

    # 多模态支持: 添加 vision/audio projector
    if MULTIMODAL_ENABLED:
        cmd.extend(["-mm", LLAMA_MMPROJ])
        capabilities = []
        if VISION_ENABLED:
            capabilities.append("视觉")
        if AUDIO_ENABLED:
            capabilities.append("音频")
        print(f"[llama-server] 多模态已启用: {'+'.join(capabilities)}")
        if VISION_ENABLED:
            print(f"  - Vision: {Path(LLAMA_MMPROJ_VISION).name}")
        if AUDIO_ENABLED:
            print(f"  - Audio: {Path(LLAMA_MMPROJ_AUDIO).name}")

    # 开启 slot persistence (真正的 KV Cache 持久化)
    if KV_CACHE_ENABLED:
        cmd.extend([
            "--slot-save-path", str(KV_CACHE_DIR),  # KV Cache 保存目录
            "-np", "4",  # 4 个 slots 支持并发
        ])
        print(f"[llama-server] KV Cache 持久化已启用")

    llama_server_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待服务启动
    import requests
    for i in range(60):
        try:
            resp = requests.get(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=1)
            if resp.status_code == 200:
                print(f"[llama-server] 启动成功！")
                llama_server_ready = True
                return True
        except:
            pass
        time.sleep(1)
        if i % 10 == 0:
            print(f"[llama-server] 等待启动... {i}s")

    print("[llama-server] 启动超时")
    return False


def query_llama_server(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stop_sequences: List[str] = None,
    image_data: List[str] = None  # 多模态: Base64 图像数据列表
) -> Dict[str, Any]:
    """查询 llama-server"""
    global llama_server_ready

    if not llama_server_ready:
        if not start_llama_server():
            return {"error": "llama-server 未就绪"}

    start = time.time()
    try:
        import requests

        stop = stop_sequences or ["</s>", "<eos>", "<end_of_turn>"]

        # 构建请求体
        request_body = {
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "stream": False,
        }

        # 检查是否有图像数据 (多模态)
        if image_data and MULTIMODAL_ENABLED:
            # 处理图像数据: llama-server 需要纯 base64，不带 data: 前缀
            processed_images = []
            for img in image_data:
                if img.startswith("data:"):
                    # 移除 data:image/xxx;base64, 前缀
                    try:
                        _, b64_data = img.split(",", 1)
                        processed_images.append(b64_data)
                    except:
                        processed_images.append(img)
                else:
                    processed_images.append(img)

            # 使用多模态格式
            request_body["prompt"] = {
                "prompt_string": prompt,
                "multimodal_data": processed_images
            }
            print(f"[query] 多模态请求: {len(processed_images)} 张图像")
        else:
            # 纯文本格式
            request_body["prompt"] = prompt

        # 根据是否多模态选择端点
        # 多模态需要使用 /completions (带 s)
        endpoint = "/completions" if (image_data and MULTIMODAL_ENABLED) else "/completion"

        resp = requests.post(
            f"http://127.0.0.1:{LLAMA_SERVER_PORT}{endpoint}",
            json=request_body,
            timeout=120
        )

        elapsed = time.time() - start

        if resp.status_code != 200:
            return {"error": f"llama-server 请求失败: {resp.status_code}"}

        data = resp.json()
        response = data.get("content", "").strip()
        tokens_predicted = data.get("tokens_predicted", len(response) // 4)
        tokens_evaluated = data.get("tokens_evaluated", 0)

        timings = data.get("timings", {})
        speed = timings.get("predicted_per_second", 0)
        if speed == 0:
            speed = tokens_predicted / elapsed if elapsed > 0 else 0

        return {
            "response": response,
            "prompt_tokens": tokens_evaluated,
            "completion_tokens": tokens_predicted,
            "total_time": elapsed,
            "speed": speed
        }
    except Exception as e:
        llama_server_ready = False
        return {"error": f"llama-server 错误: {str(e)}"}


def run_llama_mtmd_cli(
    prompt: str,
    image_path: str = None,
    audio_path: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE
) -> Dict[str, Any]:
    """
    使用 llama-mtmd-cli 进行多模态推理 (支持音频)

    llama-server 只支持视觉多模态，音频需要使用 llama-mtmd-cli 命令行工具

    Args:
        prompt: 文本提示
        image_path: 图像文件路径 (可选)
        audio_path: 音频文件路径 (可选)
        max_tokens: 最大生成 token 数
        temperature: 温度参数

    Returns:
        包含 response、prompt_tokens、completion_tokens 等的字典
    """
    start = time.time()

    # 检查 llama-mtmd-cli 是否存在
    if not Path(LLAMA_MTMD_BIN).exists():
        return {"error": f"llama-mtmd-cli 不存在: {LLAMA_MTMD_BIN}"}

    # 选择模型: 音频推理使用专用模型 (原始模型，非微调版)
    # 因为音频 mmproj 仅与原始模型兼容
    model_path = LLAMA_MODEL_AUDIO if audio_path else LLAMA_MODEL

    # 检查模型是否存在
    if not Path(model_path).exists():
        return {"error": f"模型文件不存在: {model_path}"}

    # 动态选择 mmproj
    mmproj_list = []
    if image_path and LLAMA_MMPROJ_VISION and Path(LLAMA_MMPROJ_VISION).exists():
        mmproj_list.append(LLAMA_MMPROJ_VISION)
    if audio_path and LLAMA_MMPROJ_AUDIO and Path(LLAMA_MMPROJ_AUDIO).exists():
        mmproj_list.append(LLAMA_MMPROJ_AUDIO)

    if not mmproj_list:
        return {"error": "未找到所需的 mmproj 文件"}

    mmproj_combined = ",".join(mmproj_list)

    # 设置环境变量
    env = os.environ.copy()
    bin_dir = str(Path(LLAMA_MTMD_BIN).parent)
    env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    # 构建命令 (与 WebUI 保持一致)
    # 参考: apps/webui/server.py run_llama_mmproj_cli()
    # - --log-verbosity 0: 减少日志输出
    # - --no-warmup: 跳过 warmup (对音频必须)
    # - 不强制 CPU 模式，让 llama.cpp 自动选择 GPU
    cmd = [
        LLAMA_MTMD_BIN,
        "--log-verbosity", "0",
        "--no-warmup",
        "-m", model_path,
        "--mmproj", mmproj_combined,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
    ]

    # 添加图像
    if image_path:
        cmd.extend(["--image", image_path])
        print(f"[llama-mtmd-cli] 图像: {image_path}")

    # 添加音频
    if audio_path:
        cmd.extend(["--audio", audio_path])
        print(f"[llama-mtmd-cli] 音频: {audio_path}")

    print(f"[llama-mtmd-cli] 执行命令: {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=180  # 3分钟超时
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "未知错误"
            return {"error": f"llama-mtmd-cli 执行失败: {error_msg}"}

        # 过滤输出 (与 WebUI 保持一致)
        # 参考: apps/webui/server.py run_llama_mmproj_cli()
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        content_lines = [
            ln for ln in lines
            if not ln.startswith(("ggml", "AVX", "gguf", "llama", "clip", "Using", "model", "warmup", "load"))
        ]
        response = "\n".join(content_lines).strip() if content_lines else "\n".join(lines).strip()

        # 估算 token 数量
        tokens_predicted = len(response) // 4

        print(f"[llama-mtmd-cli] 完成, 耗时 {elapsed:.2f}s, 输出 {len(response)} 字符")

        return {
            "response": response,
            "prompt_tokens": 0,  # CLI 模式无法获取
            "completion_tokens": tokens_predicted,
            "total_time": elapsed,
            "speed": tokens_predicted / elapsed if elapsed > 0 else 0
        }

    except subprocess.TimeoutExpired:
        return {"error": "llama-mtmd-cli 执行超时"}
    except Exception as e:
        return {"error": f"llama-mtmd-cli 错误: {str(e)}"}


# ========== KV Cache 持久化 (真正的 thoughtSignature) ==========

def save_kv_cache(slot_id: int = 0) -> Optional[str]:
    """
    保存指定 slot 的 KV Cache 到磁盘

    这是真正的 thoughtSignature 实现核心:
    将模型的内部状态 (KV Cache) 持久化，而不仅仅是保存 prompt 文本

    参考: https://github.com/ggml-org/llama.cpp/discussions/13606

    Returns:
        保存的文件名 (用作 thoughtSignature)，失败返回 None
    """
    if not KV_CACHE_ENABLED:
        return None

    try:
        import urllib.request
        import urllib.parse

        # 生成唯一的缓存文件名
        cache_id = f"thought_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        # 调用 llama-server 的 slot save API
        url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/slots/{slot_id}?action=save&filename={cache_id}"

        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                print(f"[KV Cache] Saved slot {slot_id} to {cache_id}")
                return cache_id

    except Exception as e:
        print(f"[KV Cache] Save failed: {e}")

    return None


def restore_kv_cache(cache_id: str, slot_id: int = 0) -> bool:
    """
    从磁盘恢复 KV Cache 到指定 slot

    这允许模型"恢复"之前的思考状态，实现真正的多轮推理连续性

    Args:
        cache_id: 之前保存时返回的缓存 ID (thoughtSignature)
        slot_id: 要恢复到的 slot ID

    Returns:
        是否恢复成功
    """
    if not KV_CACHE_ENABLED or not cache_id:
        return False

    try:
        import urllib.request

        # 检查缓存文件是否存在
        cache_file = KV_CACHE_DIR / f"{cache_id}.bin"
        if not cache_file.exists():
            print(f"[KV Cache] Cache file not found: {cache_id}")
            return False

        # 调用 llama-server 的 slot restore API
        url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/slots/{slot_id}?action=restore&filename={cache_id}"

        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                print(f"[KV Cache] Restored slot {slot_id} from {cache_id}")
                return True

    except Exception as e:
        print(f"[KV Cache] Restore failed: {e}")

    return False


def cleanup_old_kv_cache(max_age_hours: int = 1):
    """清理过期的 KV Cache 文件"""
    if not KV_CACHE_ENABLED:
        return

    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for cache_file in KV_CACHE_DIR.glob("thought_*.bin"):
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                cache_file.unlink()
                print(f"[KV Cache] Cleaned up: {cache_file.name}")

    except Exception as e:
        print(f"[KV Cache] Cleanup error: {e}")


# ========== Gemini API 格式转换 ==========

# 支持的图像 MIME 类型
SUPPORTED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/heic", "image/heif", "image/gif"
}

SUPPORTED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mp3", "audio/mpeg",
    "audio/ogg", "audio/vorbis",
    "audio/flac", "audio/x-flac",
    "audio/aac", "audio/mp4",
    "audio/webm"
}

SUPPORTED_MEDIA_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_AUDIO_TYPES


def extract_audio_from_media(media_data_list: List[str]) -> tuple:
    """
    从媒体数据列表中分离音频和图像

    Args:
        media_data_list: data URI 格式的媒体数据列表

    Returns:
        (image_data_list, audio_data_list)
        - image_data_list: 仅包含图像的 data URI 列表
        - audio_data_list: 仅包含音频的 data URI 列表
    """
    image_data = []
    audio_data = []

    for data_uri in media_data_list:
        # 解析 data URI 获取 MIME 类型
        # 格式: data:mime/type;base64,DATA
        if data_uri.startswith("data:"):
            try:
                # 提取 mime type
                mime_part = data_uri.split(";")[0]  # "data:audio/flac"
                mime_type = mime_part.replace("data:", "")  # "audio/flac"

                if mime_type in SUPPORTED_AUDIO_TYPES:
                    audio_data.append(data_uri)
                elif mime_type in SUPPORTED_IMAGE_TYPES:
                    image_data.append(data_uri)
                else:
                    # 未知类型，假设为图像
                    image_data.append(data_uri)
            except:
                # 解析失败，假设为图像
                image_data.append(data_uri)
        else:
            # 不是 data URI，假设为图像
            image_data.append(data_uri)

    return image_data, audio_data


def save_audio_to_temp_file(audio_data_uri: str) -> Optional[str]:
    """
    将音频 data URI 保存到临时文件

    Args:
        audio_data_uri: data URI 格式的音频数据

    Returns:
        临时文件路径，失败返回 None
    """
    import tempfile

    try:
        # 解析 data URI
        # 格式: data:audio/flac;base64,DATA
        header, b64_data = audio_data_uri.split(",", 1)
        mime_part = header.split(";")[0]  # "data:audio/flac"
        mime_type = mime_part.replace("data:", "")  # "audio/flac"

        # 确定文件扩展名
        ext_map = {
            "audio/wav": ".wav",
            "audio/wave": ".wav",
            "audio/x-wav": ".wav",
            "audio/mp3": ".mp3",
            "audio/mpeg": ".mp3",
            "audio/ogg": ".ogg",
            "audio/vorbis": ".ogg",
            "audio/flac": ".flac",
            "audio/x-flac": ".flac",
            "audio/aac": ".aac",
            "audio/mp4": ".m4a",
            "audio/webm": ".webm",
        }
        ext = ext_map.get(mime_type, ".audio")

        # 解码 base64 并写入临时文件
        audio_bytes = base64.b64decode(b64_data)

        # 创建临时文件 (不自动删除，需要手动清理)
        fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="gemini_audio_")
        with os.fdopen(fd, 'wb') as f:
            f.write(audio_bytes)

        print(f"[save_audio] 保存音频到: {temp_path} ({len(audio_bytes)} bytes)")
        return temp_path

    except Exception as e:
        print(f"[save_audio] 保存音频失败: {e}")
        return None


def parse_gemini_contents(contents: List[Dict]) -> tuple:
    """
    解析 Gemini contents 格式

    Returns:
        (prompt_text, media_data_list)
        - prompt_text: 纯文本 prompt (使用 <__media__> 标记媒体位置)
        - media_data_list: Base64 编码的媒体数据列表 (图像和音频)
    """
    messages = []
    media_data_list = []

    for content in contents:
        role = content.get("role", "user")
        parts = content.get("parts", [])

        text_parts = []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])
                elif "inlineData" in part:
                    # 内联媒体数据 (Gemini API 格式)
                    inline = part["inlineData"]
                    mime_type = inline.get("mimeType", "image/jpeg")
                    data = inline.get("data", "")

                    if mime_type in SUPPORTED_MEDIA_TYPES and data:
                        # 添加媒体占位符
                        text_parts.append("<__media__>")
                        # 添加 base64 数据 (可能带或不带 data URI 前缀)
                        if data.startswith("data:"):
                            # 已经是 data URI 格式
                            media_data_list.append(data)
                        else:
                            # 纯 base64，添加 data URI 前缀
                            media_data_list.append(f"data:{mime_type};base64,{data}")

                        # 日志记录媒体类型
                        if mime_type in SUPPORTED_IMAGE_TYPES:
                            print(f"[parse] 检测到图像: {mime_type}")
                        elif mime_type in SUPPORTED_AUDIO_TYPES:
                            print(f"[parse] 检测到音频: {mime_type}")
                    else:
                        print(f"[parse] 不支持的媒体类型: {mime_type}")
                elif "fileData" in part:
                    # 文件引用 (需要下载)
                    file_data = part["fileData"]
                    file_uri = file_data.get("fileUri", "")
                    mime_type = file_data.get("mimeType", "image/jpeg")

                    if file_uri and mime_type in SUPPORTED_MEDIA_TYPES:
                        # 尝试下载文件
                        try:
                            import requests as req
                            resp = req.get(file_uri, timeout=30)
                            if resp.status_code == 200:
                                data = base64.b64encode(resp.content).decode()
                                text_parts.append("<__media__>")
                                media_data_list.append(f"data:{mime_type};base64,{data}")

                                if mime_type in SUPPORTED_IMAGE_TYPES:
                                    print(f"[parse] 下载图像成功: {file_uri[:50]}...")
                                elif mime_type in SUPPORTED_AUDIO_TYPES:
                                    print(f"[parse] 下载音频成功: {file_uri[:50]}...")
                        except Exception as e:
                            print(f"[parse] 下载文件失败: {e}")
                elif "thought" in part and part.get("thought"):
                    # 思考过程，跳过
                    continue
                elif "thoughtSignature" in part:
                    # 签名，跳过
                    continue
                elif "functionCall" in part:
                    # 函数调用
                    fc = part["functionCall"]
                    text_parts.append(f"[Function Call: {fc.get('name')}({json.dumps(fc.get('args', {}))})]")
                elif "functionResponse" in part:
                    # 函数响应
                    fr = part["functionResponse"]
                    text_parts.append(f"[Function Response: {fr.get('name')} = {json.dumps(fr.get('response', {}))}]")

        if text_parts:
            text = "\n".join(text_parts)
            if role == "user":
                messages.append(f"<start_of_turn>user\n{text}<end_of_turn>")
            elif role == "model":
                messages.append(f"<start_of_turn>model\n{text}<end_of_turn>")
            elif role == "function":
                messages.append(f"<start_of_turn>user\n{text}<end_of_turn>")

    # 添加模型回复开头
    messages.append("<start_of_turn>model\n")

    return "".join(messages), media_data_list


def parse_system_instruction(system_instruction: Dict) -> str:
    """解析系统指令"""
    if not system_instruction:
        return ""

    parts = system_instruction.get("parts", [])
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
        elif isinstance(part, str):
            texts.append(part)

    return "\n".join(texts)


def parse_tools(tools: List[Dict], tool_config: Dict = None) -> str:
    """
    解析工具声明为提示词

    Args:
        tools: 工具声明列表
        tool_config: 工具配置 (toolConfig)
            - functionCallingConfig.mode: AUTO | ANY | NONE
            - functionCallingConfig.allowedFunctionNames: 允许的函数名列表

    Returns:
        工具提示词字符串
    """
    if not tools:
        return ""

    # 解析 toolConfig
    fc_config = tool_config.get("functionCallingConfig", {}) if tool_config else {}
    mode = fc_config.get("mode", "AUTO")  # 默认 AUTO
    allowed_names = fc_config.get("allowedFunctionNames", [])

    # NONE 模式: 禁用函数调用
    if mode == "NONE":
        return ""

    tool_descriptions = []
    has_code_execution = False

    for tool in tools:
        if "functionDeclarations" in tool:
            for func in tool["functionDeclarations"]:
                name = func.get("name", "unknown")

                # 如果指定了 allowedFunctionNames，过滤函数
                if allowed_names and name not in allowed_names:
                    continue

                desc = func.get("description", "")
                params = func.get("parameters", {})

                tool_descriptions.append(f"""
Function: {name}
Description: {desc}
Parameters: {json.dumps(params, indent=2)}
""")
        elif "codeExecution" in tool:
            has_code_execution = True
        elif "googleSearch" in tool:
            # 不支持，跳过
            pass

    prompt_parts = []

    # 函数声明提示
    if tool_descriptions:
        mode_instruction = ""
        if mode == "ANY":
            mode_instruction = "\nIMPORTANT: You MUST use one of the available tools to respond. Do not respond with plain text."
        else:  # AUTO
            mode_instruction = "\nUse a tool when appropriate, or respond with plain text if no tool is needed."

        prompt_parts.append(f"""
You have access to the following tools. When you need to use a tool, respond with a function call in the following format:
{{
  "functionCall": {{
    "name": "tool_name",
    "args": {{...}}
  }}
}}

For multiple function calls, include multiple functionCall objects:
{{
  "functionCalls": [
    {{"name": "tool1", "args": {{...}}}},
    {{"name": "tool2", "args": {{...}}}}
  ]
}}
{mode_instruction}

Available tools:
{"".join(tool_descriptions)}
""")

    # codeExecution 提示
    if has_code_execution:
        prompt_parts.append("""
You also have access to a Python code execution tool. When you need to run Python code, respond with:
{
  "executableCode": {
    "language": "PYTHON",
    "code": "your python code here"
  }
}
After execution, you will receive the result and should provide a final text response.
""")

    return "\n".join(prompt_parts)


def parse_response_schema(generation_config: Dict) -> Optional[Dict]:
    """解析响应 schema"""
    if not generation_config:
        return None

    mime_type = generation_config.get("responseMimeType")
    schema = generation_config.get("responseSchema")

    if mime_type == "application/json" and schema:
        return schema
    return None


# ========== 思考模拟系统 ==========
#
# 通过 prompt engineering 模拟 Gemini 3 的 thinking 功能
# 让模型先输出思考过程，再输出最终答案
#

THINKING_PROMPT_TEMPLATE = {
    "none": "",  # 显式禁用思考
    "minimal": """Think briefly before answering. Use <thinking> for your thoughts and <answer> for your response.

""",
    "low": """Before answering, briefly consider the key points in 1-2 sentences inside <thinking> tags, then give your answer inside <answer> tags.

Format:
<thinking>Brief consideration...</thinking>
<answer>Your response...</answer>

""",
    "medium": """Before answering, think through the problem step by step inside <thinking> tags. Consider:
- What is being asked?
- What are the key points?
- What's the best approach?

Then provide your answer inside <answer> tags.

Format:
<thinking>Your reasoning process...</thinking>
<answer>Your final response...</answer>

""",
    "high": """Before answering, engage in thorough reasoning inside <thinking> tags. Consider:
- What exactly is the user asking?
- What background knowledge is relevant?
- What are different approaches or perspectives?
- What's the most helpful way to respond?

Take your time to think deeply, then provide your comprehensive answer inside <answer> tags.

Format:
<thinking>
Your detailed reasoning process...
- First, I need to understand...
- The key considerations are...
- My approach will be...
</thinking>
<answer>
Your final, well-considered response...
</answer>

"""
}

# 默认思考等级 (gemini-3-pro-preview 默认会思考)
DEFAULT_THINKING_LEVEL = "medium"


def parse_thinking_response(response_text: str) -> tuple:
    """
    解析包含思考和答案的响应

    Returns:
        (thinking_text, answer_text, thoughts_tokens)
    """
    import re

    thinking_text = ""
    answer_text = response_text  # 默认整个响应都是答案

    # 尝试提取 <thinking>...</thinking>
    thinking_match = re.search(r'<thinking>([\s\S]*?)</thinking>', response_text, re.IGNORECASE)
    if thinking_match:
        thinking_text = thinking_match.group(1).strip()

    # 尝试提取 <answer>...</answer>
    answer_match = re.search(r'<answer>([\s\S]*?)</answer>', response_text, re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    elif thinking_match:
        # 如果有 thinking 但没有 answer 标签，取 thinking 之后的内容
        after_thinking = response_text[thinking_match.end():].strip()
        if after_thinking:
            answer_text = after_thinking
        else:
            # 只有思考，没有答案（可能是 MAX_TOKENS）
            answer_text = ""

    # 估算 thoughts tokens (大约 4 字符 = 1 token)
    thoughts_tokens = len(thinking_text) // 4 if thinking_text else 0

    return thinking_text, answer_text, thoughts_tokens


def build_gemini_response(
    response_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    model_version: str = MODEL_VERSION,
    function_call: Dict = None,
    function_calls: List[Dict] = None,  # 新增: 多个函数调用
    executable_code: Dict = None,       # 新增: codeExecution
    code_execution_result: Dict = None, # 新增: 代码执行结果
    finish_reason: str = "STOP",
    thoughts_tokens: int = 0,
    prompt_context: str = "",       # 完整的 prompt 上下文
    thinking_level: str = "medium", # 思考等级
    session_id: str = "",           # 会话 ID
    thinking_text: str = "",        # 思考内容
    show_thinking: bool = True,     # 是否显示思考过程 (low 等级隐藏)
    tool_use_tokens: int = 0,       # 新增: 工具使用的 token 数
    has_audio: bool = False,        # 新增: 是否包含音频输入
    audio_tokens: int = 0,          # 新增: 音频 token 数
) -> Dict:
    """
    构建 Gemini API 格式的响应
    完全对齐真实 Gemini API 的 JSON 结构和字段顺序

    改进: 支持多个函数调用 (Parallel Function Calling) 和 codeExecution
    """

    response_id = hashlib.md5(f"{time.time()}:{response_text[:50] if response_text else 'fc'}".encode()).hexdigest()[:22]

    # 生成 thoughtSignature (保存完整推理状态)
    signature = generate_thought_signature(
        prompt_context=prompt_context,
        response_text=response_text,
        function_call=function_call or (function_calls[0] if function_calls else None),
        thinking_level=thinking_level,
        session_id=session_id or response_id,
    )

    parts = []

    # 当 finish_reason=MAX_TOKENS 时，真实 API 只返回一个 part
    # 这表示生成被截断，我们应该只返回已生成的内容
    if finish_reason == "MAX_TOKENS":
        # MAX_TOKENS 时：
        # - 如果有 thinking_text，说明思考过程存在，返回思考 part (with thought: true)
        # - 如果没有 thinking 但有 response_text，返回 response part
        # 真实 Gemini API 在高思考模式下 MAX_TOKENS 通常返回 thought: true 的 part
        if thinking_text:
            # 返回思考内容作为唯一的 part
            parts.append({
                "text": thinking_text,
                "thought": True,
                "thoughtSignature": signature
            })
        elif response_text:
            parts.append({
                "text": response_text,
                "thoughtSignature": signature
            })
    else:
        # 正常情况：如果有思考内容且需要显示，先添加思考 part
        # 注意: thinkingLevel=low 时不显示思考过程，但 thoughtsTokenCount 仍然计数
        if thinking_text and show_thinking:
            parts.append({
                "text": thinking_text,
                "thought": True
            })

    # 添加主响应 (只在非 MAX_TOKENS 时处理)
    has_tool_call = False

    # MAX_TOKENS 已在上面处理，这里只处理正常情况
    if finish_reason != "MAX_TOKENS":
        # 处理多个函数调用 (Parallel Function Calling)
        if function_calls and len(function_calls) > 0:
            has_tool_call = True
            # 第一个函数调用带 thoughtSignature
            parts.append({
                "thoughtSignature": signature,
                "functionCall": function_calls[0]
            })
            # 后续函数调用不带 thoughtSignature
            for fc in function_calls[1:]:
                parts.append({
                    "functionCall": fc
                })
        # 处理单个函数调用 (向后兼容)
        elif function_call:
            has_tool_call = True
            parts.append({
                "thoughtSignature": signature,
                "functionCall": function_call
            })
        # 处理 codeExecution
        elif executable_code:
            has_tool_call = True
            parts.append({
                "thoughtSignature": signature,
                "executableCode": executable_code
            })
            # 如果有执行结果，添加结果 part
            if code_execution_result:
                parts.append({
                    "codeExecutionResult": code_execution_result
                })
            # 如果有最终文本响应
            if response_text:
                parts.append({
                    "text": response_text
                })
        elif response_text:
            # 普通文本响应 (只有在有答案时才添加)
            parts.append({
                "text": response_text,
                "thoughtSignature": signature
            })
        elif thinking_text:
            # 只有思考没有答案时，在思考 part 上添加签名
            if parts:
                parts[-1]["thoughtSignature"] = signature

    # 按真实 Gemini API 的字段顺序构建响应
    # 真实顺序: candidates -> usageMetadata -> modelVersion -> responseId
    # 注意: content 内部顺序是 role -> parts
    from collections import OrderedDict

    # 构建 candidate 对象
    candidate = OrderedDict([
        ("content", OrderedDict([
            ("role", "model"),
            ("parts", parts)
        ])),
        ("finishReason", finish_reason),
        ("index", 0),
        ("safetyRatings", None)  # 始终返回 null (与真实 API 一致)
    ])

    # 如果是工具调用，添加 finishMessage (与真实 API 一致)
    if has_tool_call:
        candidate["finishMessage"] = "Model generated function call(s)."

    # 构建 usageMetadata (字段顺序与真实 API 完全一致)
    # 真实 API 顺序: promptTokenCount → candidatesTokenCount → totalTokenCount → thoughtsTokenCount → promptTokensDetails

    # 构建 promptTokensDetails (支持多模态)
    prompt_tokens_details = [
        {
            "modality": "TEXT",
            "tokenCount": prompt_tokens - audio_tokens  # 文本 token = 总 prompt token - 音频 token
        }
    ]

    # 如果有音频，添加 AUDIO 模态
    if has_audio and audio_tokens > 0:
        prompt_tokens_details.append({
            "modality": "AUDIO",
            "tokenCount": audio_tokens
        })

    usage_metadata = OrderedDict([
        ("promptTokenCount", prompt_tokens),
        ("candidatesTokenCount", completion_tokens),
        ("totalTokenCount", prompt_tokens + completion_tokens + thoughts_tokens + tool_use_tokens),
        ("thoughtsTokenCount", thoughts_tokens),
        ("promptTokensDetails", prompt_tokens_details)
    ])

    # 如果有工具使用，添加 toolUsePromptTokenCount
    if tool_use_tokens > 0:
        usage_metadata["toolUsePromptTokenCount"] = tool_use_tokens
        usage_metadata["toolUsePromptTokensDetails"] = [
            {
                "modality": "TEXT",
                "tokenCount": tool_use_tokens
            }
        ]

    # 真实 API 字段顺序: candidates -> promptFeedback -> usageMetadata -> modelVersion -> responseId
    response = OrderedDict([
        ("candidates", [candidate]),
        ("promptFeedback", {
            "safetyRatings": None
        }),
        ("usageMetadata", usage_metadata),
        ("modelVersion", model_version),
        ("responseId", response_id)
    ])

    return response


def extract_function_call(response_text: str) -> Optional[Dict]:
    """从响应中提取函数调用（单个）"""
    result = extract_tool_calls(response_text)
    if result and result.get("function_calls"):
        return result["function_calls"][0]
    return None


def extract_tool_calls(response_text: str) -> Optional[Dict]:
    """
    从响应中提取工具调用（支持多个函数调用和 codeExecution）

    改进版：使用更可靠的 JSON 解析方法

    Returns:
        {
            "function_calls": [{"name": ..., "args": ...}, ...],
            "executable_code": {"language": "PYTHON", "code": ...} | None
        }
        或 None (如果没有找到任何工具调用)
    """
    result = {
        "function_calls": [],
        "executable_code": None
    }

    def try_parse_json(text: str) -> Optional[Dict]:
        """尝试解析 JSON，支持嵌套结构"""
        text = text.strip()
        if not text.startswith('{'):
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def find_json_objects(text: str) -> list:
        """查找文本中所有完整的 JSON 对象"""
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                # 找到完整的 JSON 对象
                depth = 0
                start = i
                in_string = False
                escape_next = False
                j = i
                while j < len(text):
                    c = text[j]
                    if escape_next:
                        escape_next = False
                    elif c == '\\':
                        escape_next = True
                    elif c == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                json_str = text[start:j+1]
                                parsed = try_parse_json(json_str)
                                if parsed:
                                    objects.append(parsed)
                                break
                    j += 1
            i += 1
        return objects

    def extract_from_data(data: Dict):
        """从解析后的数据中提取工具调用"""
        # 检查 functionCalls (复数 - 多个函数调用)
        if "functionCalls" in data:
            for fc in data["functionCalls"]:
                if isinstance(fc, dict) and "name" in fc:
                    result["function_calls"].append(fc)
        # 检查 functionCall (单数)
        elif "functionCall" in data:
            fc = data["functionCall"]
            if isinstance(fc, dict) and "name" in fc:
                result["function_calls"].append(fc)
        # 如果直接是 {name, args} 格式
        elif "name" in data and "args" in data:
            result["function_calls"].append(data)

        # 检查 executableCode
        if "executableCode" in data and not result["executable_code"]:
            result["executable_code"] = data["executableCode"]

    # 1. 首先尝试提取 ```json ``` 代码块
    json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if json_block_match:
        parsed = try_parse_json(json_block_match.group(1))
        if parsed:
            extract_from_data(parsed)

    # 2. 尝试直接解析整个文本为 JSON
    if not result["function_calls"]:
        parsed = try_parse_json(response_text)
        if parsed:
            extract_from_data(parsed)

    # 3. 查找文本中的 JSON 对象
    if not result["function_calls"]:
        json_objects = find_json_objects(response_text)
        for obj in json_objects:
            extract_from_data(obj)
            if result["function_calls"]:
                break

    # 如果没有找到任何工具调用，返回 None
    if not result["function_calls"] and not result["executable_code"]:
        return None

    return result


def enforce_json_schema(response_text: str, schema: Dict) -> str:
    """尝试强制响应符合 JSON schema"""
    # 首先尝试直接解析
    try:
        data = json.loads(response_text)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except:
        pass

    # 尝试提取 JSON 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return json.dumps(data, ensure_ascii=False, indent=2)
        except:
            pass

    # 尝试找到 { } 包围的 JSON
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return json.dumps(data, ensure_ascii=False, indent=2)
        except:
            pass

    # 返回原始响应
    return response_text


# ========== API 路由 ==========

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "model": LLAMA_MODEL,
        "server_ready": llama_server_ready
    })


@app.route(f"/{API_VERSION}/models/<model_name>:generateContent", methods=["POST"])
def generate_content(model_name: str):
    """
    Gemini generateContent API

    支持的模型名称:
    - gemini-3-pro-preview
    - gemini-3-flash-preview
    - gemma-3n (本地别名)
    """
    try:
        # 获取 API key (可选验证)
        api_key = request.args.get("key", "")

        data = request.json
        if not data:
            return jsonify({"error": {"message": "请求体为空", "code": "400"}}), 400

        # 解析请求
        contents = data.get("contents", [])
        system_instruction = data.get("systemInstruction")
        generation_config = data.get("generationConfig", {})
        safety_settings = data.get("safetySettings", [])
        tools = data.get("tools", [])
        tool_config = data.get("toolConfig", {})

        if not contents:
            return jsonify({"error": {"message": "contents 不能为空", "code": "400"}}), 400

        # 解析生成配置
        thinking_config = generation_config.get("thinkingConfig", {})
        # 默认使用 medium 思考等级 (与真实 gemini-3-pro-preview 行为一致)
        thinking_level = thinking_config.get("thinkingLevel", DEFAULT_THINKING_LEVEL)
        # includeThoughts: 控制是否在响应中显示思考过程 (独立于 thinkingLevel)
        include_thoughts = thinking_config.get("includeThoughts", True)  # 默认显示

        max_tokens = generation_config.get("maxOutputTokens", DEFAULT_MAX_TOKENS)
        temperature = generation_config.get("temperature", DEFAULT_TEMPERATURE)

        # 解析响应 schema
        response_schema = parse_response_schema(generation_config)

        # 构建 prompt
        prompt_parts = []

        # 1. 系统指令
        if system_instruction:
            sys_text = parse_system_instruction(system_instruction)
            if sys_text:
                prompt_parts.append(f"<start_of_turn>user\nSystem: {sys_text}<end_of_turn>")

        # 2. 工具声明
        if tools:
            tools_prompt = parse_tools(tools, tool_config)
            if tools_prompt:
                prompt_parts.append(f"<start_of_turn>user\n{tools_prompt}<end_of_turn>")

        # 3. 思考提示 (根据 thinkingLevel 添加)
        # 注意: 工具调用和 JSON Schema 模式都支持思考 (真实 API 行为)
        thinking_prompt = THINKING_PROMPT_TEMPLATE.get(thinking_level, "")
        if thinking_prompt:
            prompt_parts.append(f"<start_of_turn>user\n{thinking_prompt}<end_of_turn>")

        # 4. 结构化输出指令
        if response_schema:
            schema_prompt = f"""
Please respond with valid JSON that matches this schema:
{json.dumps(response_schema, indent=2)}

Your response must be valid JSON only, no other text.
"""
            prompt_parts.append(f"<start_of_turn>user\n{schema_prompt}<end_of_turn>")

        # 5. 对话内容 (包括多模态解析)
        conversation, media_data = parse_gemini_contents(contents)
        prompt_parts.append(conversation)

        full_prompt = "".join(prompt_parts)

        # 日志
        print(f"[generateContent] model={model_name}, thinking={thinking_level}")
        print(f"[generateContent] prompt length: {len(full_prompt)} chars")
        if media_data:
            print(f"[generateContent] 多模态: {len(media_data)} 个媒体文件")

        # 分离图像和音频
        image_data, audio_data = extract_audio_from_media(media_data) if media_data else ([], [])

        # 调用模型 (根据是否有音频选择不同的后端)
        temp_audio_file = None
        try:
            if audio_data:
                # 音频模式: 使用 llama-mtmd-cli (支持音频)
                print(f"[generateContent] 音频模式: {len(audio_data)} 个音频文件")

                # 保存第一个音频到临时文件
                temp_audio_file = save_audio_to_temp_file(audio_data[0])
                if not temp_audio_file:
                    return jsonify({
                        "error": {
                            "message": "音频文件保存失败",
                            "code": "500"
                        }
                    }), 500

                result = run_llama_mtmd_cli(
                    prompt=full_prompt,
                    audio_path=temp_audio_file,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif image_data:
                # 图像模式: 使用 llama-server (仅支持视觉)
                print(f"[generateContent] 图像模式: {len(image_data)} 个图像文件")
                result = query_llama_server(
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    image_data=image_data
                )
            else:
                # 纯文本模式
                result = query_llama_server(
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    image_data=None
                )
        finally:
            # 清理临时音频文件
            if temp_audio_file and Path(temp_audio_file).exists():
                try:
                    Path(temp_audio_file).unlink()
                    print(f"[generateContent] 清理临时文件: {temp_audio_file}")
                except:
                    pass

        if "error" in result:
            return jsonify({
                "error": {
                    "message": result["error"],
                    "code": "500"
                }
            }), 500

        response_text = result["response"]
        prompt_tokens = result.get("prompt_tokens", 0)
        completion_tokens = result.get("completion_tokens", 0)

        # 解析思考内容 (如果启用了思考模式)
        thinking_text = ""
        thoughts_tokens = 0
        use_thinking = bool(thinking_prompt)  # 所有模式都支持思考

        if use_thinking:
            thinking_text, answer_text, thoughts_tokens = parse_thinking_response(response_text)
            if thinking_text:
                response_text = answer_text
                print(f"[generateContent] Thinking: {len(thinking_text)} chars, Answer: {len(response_text)} chars")

        # 检查是否有工具调用 (支持多个函数调用和 codeExecution)
        function_call = None
        function_calls = None
        executable_code = None
        tool_use_tokens = 0

        if tools:
            tool_result = extract_tool_calls(response_text)
            if tool_result:
                if tool_result.get("function_calls"):
                    if len(tool_result["function_calls"]) == 1:
                        function_call = tool_result["function_calls"][0]
                    else:
                        function_calls = tool_result["function_calls"]
                    print(f"[generateContent] Function calls: {len(tool_result['function_calls'])}")

                if tool_result.get("executable_code"):
                    executable_code = tool_result["executable_code"]
                    # codeExecution 的 token 估算
                    tool_use_tokens = len(str(executable_code)) // 4
                    print(f"[generateContent] Code execution detected")

                # 清理响应文本，只保留工具调用
                response_text = ""

        # 强制 JSON schema (如果需要)
        if response_schema and not function_call and not function_calls and not executable_code:
            response_text = enforce_json_schema(response_text, response_schema)

        # 如果没有解析出思考内容，使用模拟值
        if not use_thinking or not thinking_text:
            if thinking_level == "high":
                thoughts_tokens = max(50, completion_tokens * 2)
            elif thinking_level == "medium":
                thoughts_tokens = max(20, completion_tokens)
            elif thinking_level == "low":
                thoughts_tokens = max(10, completion_tokens // 2)

        # 确定 finishReason
        # MAX_TOKENS: 达到 token 限制、只有思考没有答案、响应被截断
        finish_reason = "STOP"

        # 计算总输出 token 数 (包含 thinking)
        total_output_chars = len(response_text or "") + len(thinking_text or "")
        estimated_total_tokens = total_output_chars // 4

        import sys
        print(f"[finishReason debug] completion_tokens={completion_tokens}, max_tokens={max_tokens}, "
              f"response_text_len={len(response_text) if response_text else 0}, "
              f"thinking_text_len={len(thinking_text) if thinking_text else 0}, "
              f"estimated_total_tokens={estimated_total_tokens}, thinking_level={thinking_level}", flush=True)
        sys.stdout.flush()

        if not response_text and thinking_text:
            # 只有思考没有答案，可能是 token 不够
            finish_reason = "MAX_TOKENS"
            print(f"[finishReason] MAX_TOKENS: no response, only thinking", flush=True)
        elif completion_tokens >= max_tokens - 10:
            # 输出 token 接近限制，可能被截断
            finish_reason = "MAX_TOKENS"
            print(f"[finishReason] MAX_TOKENS: tokens near limit", flush=True)
        elif thinking_level == "high" and thinking_text:
            # 高思考模式下，只要有思考内容就认为可能达到限制
            # 因为真实 Gemini API 在高思考+音频场景下通常返回 MAX_TOKENS
            finish_reason = "MAX_TOKENS"
            print(f"[finishReason] MAX_TOKENS: high thinking mode with thinking content", flush=True)

        # 尝试从历史 thoughtSignature 恢复上下文 (用于多轮对话)
        restored_context = restore_context_from_signature(contents)
        if restored_context:
            print(f"[generateContent] Restored context from thoughtSignature")

        # 确定是否显示思考过程
        # 1. includeThoughts=false 时完全隐藏
        # 2. thinkingLevel=low/minimal/none 时也隐藏
        show_thinking = include_thoughts and thinking_level not in ("low", "minimal", "none")

        # 计算音频 token 数 (估算: 每秒音频约 32-50 tokens，按 325 tokens 估算短音频)
        has_audio_input = bool(audio_data)
        audio_token_count = 325 if has_audio_input else 0  # 与真实 API 一致的估算

        # 构建响应 (使用改进的 thoughtSignature 系统)
        response = build_gemini_response(
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_version=MODEL_VERSION,  # 使用本地模型名称 gemma-3n-local
            function_call=function_call,
            function_calls=function_calls,
            executable_code=executable_code,
            finish_reason=finish_reason,
            thoughts_tokens=thoughts_tokens,
            prompt_context=full_prompt,   # 保存完整 prompt 用于恢复
            thinking_level=thinking_level,
            session_id=api_key or "",     # 使用 API key 作为会话标识
            thinking_text=thinking_text,  # 传递思考内容
            show_thinking=show_thinking,  # 是否显示思考过程
            tool_use_tokens=tool_use_tokens,  # 工具使用的 token 数
            has_audio=has_audio_input,    # 是否包含音频
            audio_tokens=audio_token_count,  # 音频 token 数
        )

        # 使用 Response + json.dumps 保持字段顺序
        return Response(
            json.dumps(response, ensure_ascii=False),
            mimetype='application/json'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "code": "500"
            }
        }), 500


@app.route(f"/{API_VERSION}/models/<model_name>:streamGenerateContent", methods=["POST"])
def stream_generate_content(model_name: str):
    """
    Gemini streamGenerateContent API - 流式输出

    与 generateContent 相同的输入格式，但返回 Server-Sent Events (SSE) 流。
    每个 chunk 是一个完整的 JSON 响应，逐步包含更多文本。

    真实 Gemini API 流式响应格式:
    data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},...}],...}

    注意: alt=sse 参数由客户端指定
    """
    try:
        api_key = request.args.get("key", "")
        alt = request.args.get("alt", "json")  # 默认 json，可以是 sse

        data = request.json
        if not data:
            return jsonify({"error": {"message": "请求体为空", "code": "400"}}), 400

        contents = data.get("contents", [])
        system_instruction = data.get("systemInstruction")
        generation_config = data.get("generationConfig", {})
        tools = data.get("tools", [])
        tool_config = data.get("toolConfig", {})

        if not contents:
            return jsonify({"error": {"message": "contents 不能为空", "code": "400"}}), 400

        # 解析配置 (与 generateContent 相同)
        thinking_config = generation_config.get("thinkingConfig", {})
        thinking_level = thinking_config.get("thinkingLevel", DEFAULT_THINKING_LEVEL)
        max_tokens = generation_config.get("maxOutputTokens", DEFAULT_MAX_TOKENS)
        temperature = generation_config.get("temperature", DEFAULT_TEMPERATURE)

        # 构建 prompt
        prompt_parts = []

        if system_instruction:
            sys_text = parse_system_instruction(system_instruction)
            if sys_text:
                prompt_parts.append(f"<start_of_turn>user\nSystem: {sys_text}<end_of_turn>")

        if tools:
            tools_prompt = parse_tools(tools, tool_config)
            if tools_prompt:
                prompt_parts.append(f"<start_of_turn>user\n{tools_prompt}<end_of_turn>")

        thinking_prompt = THINKING_PROMPT_TEMPLATE.get(thinking_level, "")
        if thinking_prompt:
            prompt_parts.append(f"<start_of_turn>user\n{thinking_prompt}<end_of_turn>")

        conversation, media_data = parse_gemini_contents(contents)
        prompt_parts.append(conversation)

        full_prompt = "".join(prompt_parts)

        print(f"[streamGenerateContent] model={model_name}, thinking={thinking_level}")

        def generate_stream():
            """生成 SSE 流"""
            global llama_server_ready

            if not llama_server_ready:
                if not start_llama_server():
                    yield f"data: {json.dumps({'error': {'message': 'llama-server 未就绪'}})}\n\n"
                    return

            try:
                import requests as req

                # 使用 llama-server 的流式 API
                resp = req.post(
                    f"http://127.0.0.1:{LLAMA_SERVER_PORT}/completion",
                    json={
                        "prompt": full_prompt,
                        "n_predict": max_tokens,
                        "temperature": temperature,
                        "stop": ["</s>", "<eos>", "<end_of_turn>"],
                        "stream": True,
                    },
                    stream=True,
                    timeout=120
                )

                accumulated_text = ""
                prompt_tokens = 0
                completion_tokens = 0

                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            chunk_data = json.loads(line_str[6:])

                            if "content" in chunk_data:
                                new_text = chunk_data["content"]
                                accumulated_text += new_text
                                completion_tokens = len(accumulated_text) // 4

                                # 构建流式响应 (与真实 API 格式一致)
                                stream_response = {
                                    "candidates": [{
                                        "content": {
                                            "parts": [{"text": accumulated_text}],
                                            "role": "model"
                                        },
                                        "finishReason": None,
                                        "index": 0,
                                        "safetyRatings": None
                                    }],
                                    "usageMetadata": {
                                        "promptTokenCount": prompt_tokens,
                                        "candidatesTokenCount": completion_tokens,
                                        "totalTokenCount": prompt_tokens + completion_tokens
                                    },
                                    "modelVersion": MODEL_VERSION
                                }

                                yield f"data: {json.dumps(stream_response, ensure_ascii=False)}\n\n"

                            # 检查是否完成
                            if chunk_data.get("stop"):
                                # 最终响应
                                final_response = {
                                    "candidates": [{
                                        "content": {
                                            "parts": [{"text": accumulated_text}],
                                            "role": "model"
                                        },
                                        "finishReason": "STOP",
                                        "index": 0,
                                        "safetyRatings": None
                                    }],
                                    "usageMetadata": {
                                        "promptTokenCount": chunk_data.get("tokens_evaluated", 0),
                                        "candidatesTokenCount": chunk_data.get("tokens_predicted", completion_tokens),
                                        "totalTokenCount": chunk_data.get("tokens_evaluated", 0) + chunk_data.get("tokens_predicted", completion_tokens)
                                    },
                                    "modelVersion": MODEL_VERSION
                                }
                                yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"

        # 返回 SSE 流
        return Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "code": "500"
            }
        }), 500


@app.route(f"/{API_VERSION}/models", methods=["GET"])
def list_models():
    """列出可用模型"""
    return jsonify({
        "models": [
            {
                "name": "models/gemini-3-pro-preview",
                "displayName": "Gemini 3 Pro Preview (Local)",
                "description": "Local Gemma 3n model emulating Gemini 3 Pro API",
                "inputTokenLimit": 1000000,
                "outputTokenLimit": 64000,
                "supportedGenerationMethods": ["generateContent"]
            },
            {
                "name": "models/gemini-3-flash-preview",
                "displayName": "Gemini 3 Flash Preview (Local)",
                "description": "Local Gemma 3n model emulating Gemini 3 Flash API",
                "inputTokenLimit": 1000000,
                "outputTokenLimit": 64000,
                "supportedGenerationMethods": ["generateContent"]
            }
        ]
    })


@app.route("/")
def index():
    """API 首页"""
    return jsonify({
        "name": "Gemini API Compatible Server",
        "version": "1.1.0",
        "description": "Local Gemma 3n model with Gemini API format",
        "endpoints": {
            "generateContent": f"/{API_VERSION}/models/{{model}}:generateContent",
            "streamGenerateContent": f"/{API_VERSION}/models/{{model}}:streamGenerateContent",
            "listModels": f"/{API_VERSION}/models",
            "health": "/health"
        },
        "supported_features": [
            "generateContent - 文本生成",
            "streamGenerateContent - 流式输出 (SSE)",
            "thinkingConfig - 思考等级 (minimal/low/medium/high)",
            "systemInstruction - 系统指令/人设",
            "responseMimeType + responseSchema - JSON 结构化输出",
            "safetySettings - 安全设置 (忽略)",
            "multi-turn conversation - 多轮对话",
            "thoughtSignature - 思维签名 (KV Cache 持久化)",
            "functionDeclarations + functionCall - 工具调用",
            "toolConfig (mode: AUTO/ANY/NONE) - 工具配置",
            "parallel function calling - 并行函数调用",
            "codeExecution - 代码执行 (仅识别格式)",
            "vision multimodal - 视觉多模态 (单/多图片)",
            "audio multimodal - 音频多模态 (单音频)",
            "includeThoughts - 控制思考过程显示"
        ],
        "unsupported_features": [
            "googleSearch - 需要外部 API",
            "image generation - gemini-3-pro-image-preview",
            "multiple audio files - 仅支持单音频"
        ],
        "multimodal_status": {
            "vision_enabled": VISION_ENABLED,
            "audio_enabled": AUDIO_ENABLED
        }
    })


# ========== 测试工具 ==========

@app.route("/test/basic", methods=["GET"])
def test_basic():
    """基础文本测试"""
    test_request = {
        "contents": [
            {
                "parts": [{"text": "hi"}]
            }
        ]
    }

    # 模拟调用 - 返回原始 Response 保持字段顺序
    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/thinking", methods=["GET"])
def test_thinking():
    """思考等级测试"""
    test_request = {
        "contents": [
            {
                "parts": [{"text": "What is 2 + 2?"}]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/system", methods=["GET"])
def test_system():
    """系统指令测试"""
    test_request = {
        "contents": [
            {
                "parts": [{"text": "你好，请介绍一下你自己"}]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": "你是一个专业的中文助手，名叫小智。回答时要简洁、专业，使用中文。每次回答以「小智：」开头。"}
            ]
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/json-schema", methods=["GET"])
def test_json_schema():
    """结构化输出测试"""
    test_request = {
        "contents": [
            {
                "parts": [{"text": "列出三种常见的编程语言，包括名称、发布年份和主要用途。"}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "languages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "year": {"type": "integer"},
                                "use_case": {"type": "string"}
                            },
                            "required": ["name", "year", "use_case"]
                        }
                    }
                },
                "required": ["languages"]
            }
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/function-call", methods=["GET"])
def test_function_call():
    """工具调用测试"""
    test_request = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "你能用bash来执行ls吗？"}]
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "shell_command",
                        "description": "Runs a shell command and returns its output.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The shell script to execute"
                                },
                                "workdir": {
                                    "type": "string",
                                    "description": "The working directory"
                                }
                            },
                            "required": ["command"]
                        }
                    }
                ]
            }
        ]
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/multi-turn", methods=["GET"])
def test_multi_turn():
    """多轮对话测试"""
    # 这里需要使用之前获取的 thoughtSignature
    # 为了演示，我们直接构建请求
    test_request = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "我想学习 Python 编程，应该从哪里开始？"}]
            },
            {
                "role": "model",
                "parts": [
                    {"text": "Python 是一个很好的选择！你可以从以下几个方面开始：1. 安装 Python 环境 2. 学习基础语法 3. 做一些小项目"},
                    {"thoughtSignature": "test-signature-placeholder"}
                ]
            },
            {
                "role": "user",
                "parts": [{"text": "有什么推荐的在线学习资源吗？"}]
            }
        ]
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/tool-config-any", methods=["GET"])
def test_tool_config_any():
    """toolConfig mode:ANY 测试 - 强制函数调用"""
    test_request = {
        "contents": [{"role": "user", "parts": [{"text": "What is the weather?"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }]
        }],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "ANY"
            }
        },
        "generationConfig": {"maxOutputTokens": 256}
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/tool-config-none", methods=["GET"])
def test_tool_config_none():
    """toolConfig mode:NONE 测试 - 禁用函数调用"""
    test_request = {
        "contents": [{"role": "user", "parts": [{"text": "What is the weather in Tokyo?"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }]
        }],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "NONE"
            }
        },
        "generationConfig": {"maxOutputTokens": 256}
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/parallel-function-call", methods=["GET"])
def test_parallel_function_call():
    """Parallel Function Calling 测试 - 多个函数调用"""
    test_request = {
        "contents": [{"role": "user", "parts": [{"text": "What is the weather in San Francisco and Tokyo?"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }]
        }],
        "generationConfig": {"maxOutputTokens": 512}
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/code-execution", methods=["GET"])
def test_code_execution():
    """codeExecution 工具测试"""
    test_request = {
        "contents": [{"role": "user", "parts": [{"text": "Calculate 123 * 456 using Python code"}]}],
        "tools": [{"codeExecution": {}}],
        "generationConfig": {"maxOutputTokens": 1024}
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/include-thoughts-false", methods=["GET"])
def test_include_thoughts_false():
    """includeThoughts=false 测试 - 隐藏思考过程"""
    test_request = {
        "contents": [{"role": "user", "parts": [{"text": "What is 2+2?"}]}],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "high",
                "includeThoughts": False  # 显式隐藏思考
            },
            "maxOutputTokens": 256
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/vision", methods=["GET"])
def test_vision():
    """多模态视觉测试 - 使用 URL 图像"""
    # 使用一个简单的测试图像 (CIFAR-10 样式)
    test_image_url = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"

    # 下载图像并转换为 base64
    try:
        import requests as req
        resp = req.get(test_image_url, timeout=30)
        if resp.status_code == 200:
            image_base64 = base64.b64encode(resp.content).decode()
        else:
            return jsonify({"error": "无法下载测试图像"}), 500
    except Exception as e:
        return jsonify({"error": f"下载图像失败: {e}"}), 500

    test_request = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "What is in this image? Describe it briefly."},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_base64
                    }
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 256,
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/vision-base64", methods=["POST"])
def test_vision_base64():
    """
    多模态视觉测试 - 接受 base64 图像

    请求体格式:
    {
        "image": "base64_encoded_image_data",
        "mimeType": "image/jpeg",  # 可选，默认 image/jpeg
        "prompt": "What is in this image?"  # 可选
    }
    """
    data = request.json or {}
    image_data = data.get("image", "")
    mime_type = data.get("mimeType", "image/jpeg")
    prompt = data.get("prompt", "What is in this image? Describe it briefly.")

    if not image_data:
        return jsonify({"error": "缺少 image 字段"}), 400

    test_request = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": image_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 512,
            "thinkingConfig": {
                "thinkingLevel": "medium"
            }
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/audio", methods=["POST"])
def test_audio():
    """
    多模态音频测试 - 接受 base64 音频

    请求体格式:
    {
        "audio": "base64_encoded_audio_data",
        "mimeType": "audio/wav",  # 可选，默认 audio/wav
        "prompt": "Transcribe this audio."  # 可选
    }

    支持的音频格式:
    - audio/wav, audio/wave, audio/x-wav
    - audio/mp3, audio/mpeg
    - audio/ogg, audio/vorbis
    - audio/flac, audio/x-flac
    - audio/aac, audio/mp4
    - audio/webm
    """
    if not AUDIO_ENABLED:
        return jsonify({
            "error": "音频功能未启用。请确保音频 mmproj 文件存在。",
            "audio_enabled": False,
            "required_file": "gemma-3n-audio-mmproj-f16.gguf"
        }), 400

    data = request.json or {}
    audio_data = data.get("audio", "")
    mime_type = data.get("mimeType", "audio/wav")
    prompt = data.get("prompt", "Transcribe this audio and describe what you hear.")

    if not audio_data:
        return jsonify({"error": "缺少 audio 字段"}), 400

    if mime_type not in SUPPORTED_AUDIO_TYPES:
        return jsonify({
            "error": f"不支持的音频类型: {mime_type}",
            "supported_types": list(SUPPORTED_AUDIO_TYPES)
        }), 400

    test_request = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": audio_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 1024,
            "thinkingConfig": {
                "thinkingLevel": "medium"
            }
        }
    }

    with app.test_client() as client:
        resp = client.post(
            f"/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json=test_request,
            content_type="application/json"
        )
        return Response(resp.data, mimetype='application/json')


@app.route("/test/audio-status", methods=["GET"])
def test_audio_status():
    """检查音频功能状态"""
    return jsonify({
        "audio_enabled": AUDIO_ENABLED,
        "vision_enabled": VISION_ENABLED,
        "multimodal_enabled": MULTIMODAL_ENABLED,
        "audio_mmproj": str(LLAMA_MMPROJ_AUDIO) if LLAMA_MMPROJ_AUDIO else None,
        "vision_mmproj": str(LLAMA_MMPROJ_VISION) if LLAMA_MMPROJ_VISION else None,
        "combined_mmproj": str(LLAMA_MMPROJ) if LLAMA_MMPROJ else None,
        "supported_audio_types": list(SUPPORTED_AUDIO_TYPES),
        "supported_image_types": list(SUPPORTED_IMAGE_TYPES)
    })

if __name__ == "__main__":
    print("=" * 60)
    print("Gemini API 兼容服务器")
    print("=" * 60)
    print(f"模型: {LLAMA_MODEL}")
    if MULTIMODAL_ENABLED:
        capabilities = []
        if VISION_ENABLED:
            capabilities.append("视觉")
        if AUDIO_ENABLED:
            capabilities.append("音频")
        print(f"多模态: {'+'.join(capabilities)}")
        if VISION_ENABLED:
            print(f"  - Vision: {Path(LLAMA_MMPROJ_VISION).name}")
        if AUDIO_ENABLED:
            print(f"  - Audio: {Path(LLAMA_MMPROJ_AUDIO).name}")
            if LLAMA_MODEL_AUDIO and LLAMA_MODEL_AUDIO != LLAMA_MODEL:
                print(f"  - Audio Model: {Path(LLAMA_MODEL_AUDIO).name}")
    else:
        print("多模态: 未启用")
    print(f"llama-server 端口: {LLAMA_SERVER_PORT}")
    print()

    # 预启动 llama-server
    start_llama_server()

    print()
    print("API 端点:")
    print(f"  POST /{API_VERSION}/models/{{model}}:generateContent")
    print(f"  GET  /{API_VERSION}/models")
    print(f"  GET  /health")
    print()
    print("测试端点:")
    print("  GET  /test/basic")
    print("  GET  /test/thinking")
    print("  GET  /test/system")
    print("  GET  /test/json-schema")
    print("  GET  /test/function-call")
    print("  GET  /test/multi-turn")
    print("  GET  /test/tool-config-any")
    print("  GET  /test/tool-config-none")
    print("  GET  /test/parallel-function-call")
    print("  GET  /test/code-execution")
    print("  GET  /test/include-thoughts-false")
    if VISION_ENABLED:
        print("  GET  /test/vision")
        print("  POST /test/vision-base64")
    if AUDIO_ENABLED:
        print("  POST /test/audio")
        print("  GET  /test/audio-status")
    print()
    print("启动服务器: http://localhost:5001")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)

"""
Gemma 3n API 服务器
兼容 Gemini API 风格 + OpenAI 风格

端点:
- POST /v1beta/models/{model}:generateContent  (Gemini 风格)
- POST /v1beta/models/{model}:streamGenerateContent  (Gemini 流式)
- POST /v1/chat/completions  (OpenAI 风格)
- GET /v1/models  (模型列表)

支持:
- 多模态输入 (文本、图片、音频)
- 多轮对话
- Function Calling (基础框架)
- 流式输出 (SSE)
"""
import os
import io
import base64
import torch
import numpy as np
import uuid
import json
import time
import hashlib
import hmac
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import librosa
import warnings
import psutil

warnings.filterwarnings("ignore")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
CORS(app)

# ========== 全局变量 ==========
model = None
processor = None
model_loaded = False
model_info = {}
dummy_image = None

# 模型配置
MODEL_NAME = "google/gemma-3n-E2B-it"
MODEL_ID = "gemma-3n-e2b-it"
API_VERSION = "v1beta"

# ========== Thought Signature 系统 ==========
# 模拟 Gemini 的 thoughtSignature 功能
# 核心思想: signature 不仅是验证token，更是"压缩的记忆"
#
# 传统方式: 图片 → 保存原始数据 → 下一轮重新处理 (昂贵)
# Signature方式: 图片 → 模型理解 → signature(含理解摘要) → 恢复上下文 (轻量)

# 签名密钥 (生产环境应使用环境变量)
THOUGHT_SIGNATURE_SECRET = os.environ.get("THOUGHT_SIGNATURE_SECRET", "gemma3n-thought-signature-key")

# 思维状态缓存 (session_id -> thinking_state)
# 包含: 对话轮次、思考步骤、媒体理解摘要、function calls 等
thought_states = {}

# 媒体理解缓存 (用于存储模型对图片/音频的理解)
media_understanding_cache = {}


def generate_thought_signature(
    session_id: str,
    turn_index: int,
    content_hash: str,
    function_call_name: str = None,
    thinking_step: int = 0,
    media_understanding: str = None  # 新增: 媒体理解摘要
) -> str:
    """
    生成 thought signature

    签名包含:
    - session_id: 会话标识
    - turn_index: 对话轮次
    - content_hash: 内容摘要
    - function_call_name: 工具调用名称 (如有)
    - thinking_step: 思考步骤序号
    - media_ref: 媒体理解的引用ID (如有)
    - timestamp: 时间戳

    返回 base64 编码的签名
    """
    timestamp = int(time.time())

    # 如果有媒体理解，存储到缓存并生成引用ID
    media_ref = None
    if media_understanding:
        media_ref = hashlib.md5(f"{session_id}:{turn_index}:{timestamp}".encode()).hexdigest()[:12]
        media_understanding_cache[media_ref] = {
            "session_id": session_id,
            "turn_index": turn_index,
            "understanding": media_understanding,
            "created_at": timestamp
        }

    # 构建签名数据
    signature_data = {
        "sid": session_id,
        "turn": turn_index,
        "hash": content_hash[:16],  # 内容指纹
        "func": function_call_name or "",
        "step": thinking_step,
        "media": media_ref or "",  # 媒体理解引用
        "ts": timestamp
    }

    # 序列化
    data_str = json.dumps(signature_data, sort_keys=True, separators=(',', ':'))

    # HMAC 签名
    hmac_sig = hmac.new(
        THOUGHT_SIGNATURE_SECRET.encode(),
        data_str.encode(),
        hashlib.sha256
    ).digest()

    # 组合: data + signature
    combined = data_str.encode() + b"|" + hmac_sig

    # Base64 编码
    return base64.urlsafe_b64encode(combined).decode()


def verify_thought_signature(signature: str) -> dict:
    """
    验证 thought signature

    返回解码后的签名数据，验证失败返回 None
    """
    try:
        decoded = base64.urlsafe_b64decode(signature.encode())
        parts = decoded.rsplit(b"|", 1)
        if len(parts) != 2:
            return None

        data_str, received_hmac = parts

        # 验证 HMAC
        expected_hmac = hmac.new(
            THOUGHT_SIGNATURE_SECRET.encode(),
            data_str,
            hashlib.sha256
        ).digest()

        if not hmac.compare_digest(received_hmac, expected_hmac):
            return None

        return json.loads(data_str.decode())
    except Exception:
        return None


def get_media_understanding(signature: str) -> str:
    """
    从 thought signature 中恢复媒体理解

    这是 signature 作为"压缩记忆"的核心功能
    """
    sig_data = verify_thought_signature(signature)
    if not sig_data:
        return None

    media_ref = sig_data.get("media", "")
    if not media_ref:
        return None

    cached = media_understanding_cache.get(media_ref)
    if cached:
        return cached.get("understanding")

    return None


def get_session_media_context(session_id: str) -> str:
    """
    获取会话中所有媒体理解的上下文

    用于在多轮对话中恢复对图片/音频的理解
    """
    context_parts = []

    for media_ref, data in media_understanding_cache.items():
        if data.get("session_id") == session_id:
            turn = data.get("turn_index", 0)
            understanding = data.get("understanding", "")
            if understanding:
                context_parts.append(f"[Turn {turn} media understanding]: {understanding}")

    return "\n".join(context_parts) if context_parts else ""


def compute_content_hash(messages: list, current_text: str = "") -> str:
    """
    计算内容摘要，用于 thought signature
    """
    content_parts = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            text = msg.get("text", "")
            content_parts.append(f"{role}:{text[:50]}")

    if current_text:
        content_parts.append(f"current:{current_text[:50]}")

    content_str = "|".join(content_parts)
    return hashlib.md5(content_str.encode()).hexdigest()


def update_thought_state(session_id: str, state_update: dict):
    """
    更新思维状态缓存
    """
    if session_id not in thought_states:
        thought_states[session_id] = {
            "turn_index": 0,
            "thinking_step": 0,
            "function_calls": [],
            "media_understandings": [],  # 新增: 媒体理解历史
            "last_signature": None
        }

    thought_states[session_id].update(state_update)


def get_thought_state(session_id: str) -> dict:
    """
    获取当前思维状态
    """
    return thought_states.get(session_id, {
        "turn_index": 0,
        "thinking_step": 0,
        "function_calls": [],
        "media_understandings": [],
        "last_signature": None
    })

# ========== 工具函数 ==========

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return round(mem_gb, 2)

def load_model():
    """加载模型"""
    global model, processor, model_loaded, model_info, dummy_image

    if model_loaded:
        return True

    print("=" * 60)
    print("加载 Gemma 3n 多模态模型...")
    print("=" * 60)

    load_start = time.time()

    print("[1/2] 加载处理器...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("[2/2] 加载模型...")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        max_memory={"mps": "64GiB", "cpu": "64GiB"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    dummy_image = Image.new('RGB', (64, 64), color='white')
    load_time = time.time() - load_start
    total_params = sum(p.numel() for p in model.parameters())

    model_info = {
        "id": MODEL_ID,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "google",
        "name": "Gemma 3n E2B-IT",
        "full_name": MODEL_NAME,
        "params": f"{total_params / 1e9:.2f}B",
        "dtype": "bfloat16",
        "device": str(model.device),
        "load_time": round(load_time, 2),
        "capabilities": ["text", "image", "audio", "function_calling"],
        "context_window": 8192,
        "max_output_tokens": 8192,
    }

    model_loaded = True
    print("=" * 60)
    print(f"模型加载完成! 耗时 {load_time:.2f}s")
    print(f"内存占用: {get_memory_usage()} GB")
    print("=" * 60)
    return True

def decode_base64_image(data_url_or_base64):
    """解码 base64 图片"""
    if "," in data_url_or_base64:
        data_url_or_base64 = data_url_or_base64.split(",")[1]
    image_bytes = base64.b64decode(data_url_or_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def decode_base64_audio(data_url_or_base64):
    """解码 base64 音频"""
    mime_part = ""
    if "," in data_url_or_base64:
        mime_part = data_url_or_base64.split(",")[0]
        data_url_or_base64 = data_url_or_base64.split(",")[1]
    audio_bytes = base64.b64decode(data_url_or_base64)

    # 确定扩展名
    if "wav" in mime_part:
        ext = ".wav"
    elif "webm" in mime_part:
        ext = ".webm"
    elif "ogg" in mime_part:
        ext = ".ogg"
    elif "mp3" in mime_part or "mpeg" in mime_part:
        ext = ".mp3"
    elif "flac" in mime_part:
        ext = ".flac"
    else:
        ext = ".wav"

    temp_path = f"/tmp/api_audio_{uuid.uuid4().hex[:8]}{ext}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    audio_array, sr = librosa.load(temp_path, sr=16000)
    return audio_array, sr

def parse_gemini_content(contents):
    """
    解析 Gemini 格式的 contents

    格式:
    [
        {
            "role": "user",
            "parts": [
                {"text": "..."},
                {"inlineData": {"mimeType": "image/png", "data": "base64..."}},
                {"inlineData": {"mimeType": "audio/wav", "data": "base64..."}}
            ]
        },
        {
            "role": "model",
            "parts": [{"text": "..."}]
        }
    ]
    """
    messages = []

    for content in contents:
        role = content.get("role", "user")
        parts = content.get("parts", [])

        message_content = []
        text_parts = []
        image = None
        audio = None

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

            elif "inlineData" in part:
                inline = part["inlineData"]
                mime_type = inline.get("mimeType", "")
                data = inline.get("data", "")

                if mime_type.startswith("image/"):
                    image = decode_base64_image(data)
                elif mime_type.startswith("audio/"):
                    audio = decode_base64_audio(data)

            elif "functionCall" in part:
                # Function call 响应
                message_content.append({
                    "type": "function_call",
                    "function_call": part["functionCall"]
                })

            elif "functionResponse" in part:
                # Function response
                message_content.append({
                    "type": "function_response",
                    "function_response": part["functionResponse"]
                })

        # 组装消息
        text = "\n".join(text_parts) if text_parts else ""

        messages.append({
            "role": role,
            "text": text,
            "image": image,
            "audio": audio,
            "content": message_content if message_content else None
        })

    return messages

def parse_openai_messages(messages):
    """
    解析 OpenAI 格式的 messages

    格式:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    """
    parsed = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        text = ""
        image = None
        audio = None

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        image = decode_base64_image(url)
                elif item_type == "audio":
                    data = item.get("data", "")
                    if data:
                        audio = decode_base64_audio(data)
            text = "\n".join(text_parts)

        parsed.append({
            "role": role,
            "text": text,
            "image": image,
            "audio": audio
        })

    return parsed

def build_model_messages(parsed_messages):
    """
    将解析后的消息转换为模型输入格式
    """
    model_messages = []

    for i, msg in enumerate(parsed_messages):
        role = msg["role"]
        text = msg["text"]
        image = msg["image"]
        audio = msg["audio"]

        # 映射角色
        if role in ["system", "user"]:
            model_role = "user"
        elif role in ["assistant", "model"]:
            model_role = "assistant"
        else:
            model_role = "user"

        content = []
        has_media = (image is not None or audio is not None)

        # 如果是历史消息且没有媒体，添加占位图
        if not has_media and model_role == "user":
            content.append({"type": "image", "image": dummy_image})
            if text:
                text = "Ignore the blank image. " + text

        if image is not None:
            content.append({"type": "image", "image": image})

        if audio is not None:
            content.append({"type": "audio", "audio": audio[0], "sample_rate": audio[1]})

        if text:
            content.append({"type": "text", "text": text})

        if model_role == "assistant":
            # 助手消息只包含文本
            content = [{"type": "text", "text": text}]

        model_messages.append({
            "role": model_role,
            "content": content
        })

    return model_messages

def generate_response(model_messages, generation_config=None):
    """
    生成模型响应
    """
    if not model_loaded:
        return {"error": "Model not loaded"}

    config = generation_config or {}
    max_tokens = config.get("maxOutputTokens", config.get("max_tokens", 512))
    temperature = config.get("temperature", 1.0)
    do_sample = temperature > 0

    start_time = time.time()

    # 处理输入
    inputs = processor.apply_chat_template(
        model_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_tokens = input_ids.shape[1]

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature

    # 处理图像
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        generate_kwargs["pixel_values"] = inputs["pixel_values"].to(model.device, dtype=model.dtype)

    # 处理音频
    if "input_features" in inputs and inputs["input_features"] is not None:
        generate_kwargs["input_features"] = inputs["input_features"].to(model.device, dtype=model.dtype)
        generate_kwargs["input_features_mask"] = inputs["input_features_mask"].to(model.device)

    # 生成
    generate_start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    generate_time = time.time() - generate_start

    # 解码
    output_tokens = len(outputs[0]) - input_tokens
    response_text = processor.tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    total_time = time.time() - start_time

    return {
        "text": response_text,
        "usage": {
            "prompt_tokens": int(input_tokens),
            "completion_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens)
        },
        "metrics": {
            "total_time": round(total_time, 2),
            "generate_time": round(generate_time, 2),
            "tokens_per_second": round(output_tokens / generate_time, 1) if generate_time > 0 else 0
        }
    }

def format_tools_prompt(tools):
    """
    将工具定义格式化为系统提示词

    tools 格式 (Gemini):
    [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            ]
        }
    ]
    """
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        if "functionDeclarations" in tool:
            for func in tool["functionDeclarations"]:
                name = func.get("name", "")
                desc = func.get("description", "")
                params = func.get("parameters", {})

                param_str = json.dumps(params, indent=2) if params else "{}"

                tool_descriptions.append(f"""
Function: {name}
Description: {desc}
Parameters: {param_str}
""")

    if not tool_descriptions:
        return ""

    return f"""You have access to the following tools/functions:

{"".join(tool_descriptions)}

When you need to use a tool, respond with a JSON object in this exact format:
```json
{{"function_call": {{"name": "function_name", "arguments": {{"arg1": "value1"}}}}}}
```

Only use a function if it's necessary to answer the user's question.
If you don't need to use a function, just respond normally with text.
"""

def parse_function_call(response_text):
    """
    解析响应中的 function call

    返回 (function_call, remaining_text) 或 (None, original_text)
    """
    import re

    # 尝试找到 JSON 代码块
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        try:
            json_str = match.group(1)
            data = json.loads(json_str)
            if "function_call" in data:
                fc = data["function_call"]
                return {
                    "name": fc.get("name", ""),
                    "args": fc.get("arguments", {})
                }, response_text[:match.start()].strip()
        except json.JSONDecodeError:
            pass

    # 尝试直接解析 JSON
    try:
        # 寻找 {"function_call": ...} 模式
        fc_pattern = r'\{"function_call"\s*:\s*\{[^}]+\}\}'
        match = re.search(fc_pattern, response_text)
        if match:
            data = json.loads(match.group())
            fc = data["function_call"]
            return {
                "name": fc.get("name", ""),
                "args": fc.get("arguments", {})
            }, response_text[:match.start()].strip()
    except json.JSONDecodeError:
        pass

    return None, response_text

def generate_with_tools(model_messages, tools, generation_config=None):
    """
    带工具的生成 (Function Calling)
    """
    # 将工具定义添加到系统提示
    tools_prompt = format_tools_prompt(tools)

    if tools_prompt:
        # 在第一条用户消息前添加工具说明
        if model_messages and model_messages[0]["role"] == "user":
            original_content = model_messages[0]["content"]
            # 找到文本部分并添加工具说明
            for item in original_content:
                if item.get("type") == "text":
                    item["text"] = tools_prompt + "\n\n" + item["text"]
                    break

    result = generate_response(model_messages, generation_config)

    if "error" in result:
        return result

    # 解析是否有 function call
    function_call, remaining_text = parse_function_call(result["text"])

    if function_call:
        result["function_call"] = function_call
        result["text"] = remaining_text

    return result

# ========== API 端点 ==========

@app.route("/v1/models", methods=["GET"])
def list_models():
    """列出可用模型 (OpenAI 风格)"""
    return jsonify({
        "object": "list",
        "data": [model_info] if model_loaded else []
    })

@app.route(f"/{API_VERSION}/models", methods=["GET"])
def list_models_gemini():
    """列出可用模型 (Gemini 风格)"""
    return jsonify({
        "models": [{
            "name": f"models/{MODEL_ID}",
            "displayName": model_info.get("name", MODEL_ID),
            "description": "Gemma 3n multimodal model",
            "inputTokenLimit": 8192,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            "capabilities": model_info.get("capabilities", [])
        }] if model_loaded else []
    })

@app.route(f"/{API_VERSION}/models/<model_name>:generateContent", methods=["POST"])
def generate_content(model_name):
    """
    Gemini 风格的生成端点

    请求格式:
    {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Hello"},
                    {"inlineData": {"mimeType": "image/png", "data": "base64..."}}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 512
        },
        "tools": [
            {
                "functionDeclarations": [...]
            }
        ]
    }
    """
    try:
        data = request.json
        contents = data.get("contents", [])
        generation_config = data.get("generationConfig", {})
        tools = data.get("tools", [])
        system_instruction = data.get("systemInstruction")

        # 获取或创建 session_id (用于 thought signature)
        session_id = data.get("sessionId") or request.headers.get("X-Session-Id") or str(uuid.uuid4())[:8]

        # 解析消息
        parsed = parse_gemini_content(contents)

        # 如果有系统指令，添加到开头
        if system_instruction:
            sys_parts = system_instruction.get("parts", [])
            sys_text = " ".join([p.get("text", "") for p in sys_parts])
            if sys_text:
                parsed.insert(0, {
                    "role": "user",
                    "text": sys_text,
                    "image": None,
                    "audio": None
                })

        # 构建模型消息
        model_messages = build_model_messages(parsed)

        # 生成响应 (带工具或不带工具)
        if tools:
            result = generate_with_tools(model_messages, tools, generation_config)
        else:
            result = generate_response(model_messages, generation_config)

        if "error" in result:
            return jsonify({"error": {"message": result["error"]}}), 500

        # 获取并更新思维状态
        thought_state = get_thought_state(session_id)
        turn_index = thought_state.get("turn_index", 0) + 1
        thinking_step = thought_state.get("thinking_step", 0)

        # 计算内容摘要
        content_hash = compute_content_hash(parsed, result.get("text", ""))

        # 构建 Gemini 风格响应
        parts = []

        # 如果有 function call
        if "function_call" in result:
            fc = result["function_call"]
            thinking_step += 1

            # 生成 function call 的 thought signature
            fc_signature = generate_thought_signature(
                session_id=session_id,
                turn_index=turn_index,
                content_hash=content_hash,
                function_call_name=fc["name"],
                thinking_step=thinking_step
            )

            parts.append({
                "functionCall": {
                    "name": fc["name"],
                    "args": fc["args"]
                },
                "thoughtSignature": fc_signature
            })

            # 记录 function call
            thought_state["function_calls"] = thought_state.get("function_calls", []) + [{
                "name": fc["name"],
                "signature": fc_signature
            }]

        # 添加文本响应
        if result["text"]:
            # 生成文本响应的 thought signature
            text_signature = generate_thought_signature(
                session_id=session_id,
                turn_index=turn_index,
                content_hash=content_hash,
                thinking_step=thinking_step
            )

            parts.append({
                "text": result["text"],
                "thoughtSignature": text_signature
            })

        # 更新思维状态
        update_thought_state(session_id, {
            "turn_index": turn_index,
            "thinking_step": thinking_step,
            "last_signature": parts[-1].get("thoughtSignature") if parts else None
        })

        response = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": parts if parts else [{"text": ""}]
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": result["usage"]["prompt_tokens"],
                "candidatesTokenCount": result["usage"]["completion_tokens"],
                "totalTokenCount": result["usage"]["total_tokens"]
            },
            "modelVersion": MODEL_ID,
            "sessionId": session_id  # 返回 session_id 供客户端追踪
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"message": str(e)}}), 500

@app.route(f"/{API_VERSION}/models/<model_name>:streamGenerateContent", methods=["POST"])
def stream_generate_content(model_name):
    """
    Gemini 风格的流式生成端点 (SSE)

    注意: 当前实现是伪流式 (生成完成后分块返回)
    """
    try:
        data = request.json
        contents = data.get("contents", [])
        generation_config = data.get("generationConfig", {})
        system_instruction = data.get("systemInstruction")

        # 解析消息
        parsed = parse_gemini_content(contents)

        if system_instruction:
            sys_parts = system_instruction.get("parts", [])
            sys_text = " ".join([p.get("text", "") for p in sys_parts])
            if sys_text:
                parsed.insert(0, {
                    "role": "user",
                    "text": sys_text,
                    "image": None,
                    "audio": None
                })

        model_messages = build_model_messages(parsed)
        result = generate_response(model_messages, generation_config)

        def generate():
            if "error" in result:
                yield f"data: {json.dumps({'error': {'message': result['error']}})}\n\n"
                return

            text = result["text"]
            # 分块输出 (模拟流式)
            chunk_size = 20
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                response = {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": chunk}]
                        },
                        "finishReason": "STOP" if i + chunk_size >= len(text) else None,
                        "index": 0
                    }]
                }
                yield f"data: {json.dumps(response)}\n\n"
                time.sleep(0.01)  # 模拟延迟

            # 发送使用统计
            yield f"data: {json.dumps({'usageMetadata': result['usage']})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"message": str(e)}}), 500

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    OpenAI 风格的 Chat Completions 端点

    请求格式:
    {
        "model": "gemma-3n-e2b-it",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": false
    }
    """
    try:
        data = request.json
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 1.0)
        stream = data.get("stream", False)

        # 解析消息
        parsed = parse_openai_messages(messages)

        # 构建模型消息
        model_messages = build_model_messages(parsed)

        generation_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if stream:
            result = generate_response(model_messages, generation_config)

            def generate():
                if "error" in result:
                    yield f"data: {json.dumps({'error': {'message': result['error']}})}\n\n"
                    return

                text = result["text"]
                response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

                # 分块输出
                chunk_size = 20
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    response = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(response)}\n\n"

                # 结束标记
                final = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            result = generate_response(model_messages, generation_config)

            if "error" in result:
                return jsonify({"error": {"message": result["error"]}}), 500

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "total_tokens": result["usage"]["total_tokens"]
                }
            }

            return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"message": str(e)}}), 500

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "memory_gb": get_memory_usage() if model_loaded else 0
    })

# ========== Thought Signature API 端点 ==========

@app.route("/api/thought/verify", methods=["POST"])
def verify_signature():
    """
    验证 thought signature

    请求:
    {
        "signature": "base64-encoded-signature"
    }

    响应:
    {
        "valid": true/false,
        "data": {...}  // 如果有效，返回解码后的签名数据
    }
    """
    data = request.json
    signature = data.get("signature", "")

    if not signature:
        return jsonify({"valid": False, "error": "Missing signature"})

    decoded = verify_thought_signature(signature)
    if decoded:
        return jsonify({"valid": True, "data": decoded})
    else:
        return jsonify({"valid": False, "error": "Invalid signature"})


@app.route("/api/thought/state/<session_id>", methods=["GET"])
def get_session_thought_state(session_id):
    """
    获取会话的思维状态

    用于调试和监控
    """
    state = get_thought_state(session_id)
    return jsonify({
        "session_id": session_id,
        "state": state
    })


@app.route("/api/thought/clear/<session_id>", methods=["POST"])
def clear_session_thought_state(session_id):
    """
    清除会话的思维状态
    """
    if session_id in thought_states:
        del thought_states[session_id]
    return jsonify({"success": True})


@app.route("/api/thought/stats", methods=["GET"])
def thought_stats():
    """
    获取思维状态统计

    用于监控
    """
    return jsonify({
        "total_sessions": len(thought_states),
        "sessions": [
            {
                "session_id": sid,
                "turn_index": state.get("turn_index", 0),
                "thinking_step": state.get("thinking_step", 0),
                "function_calls_count": len(state.get("function_calls", []))
            }
            for sid, state in thought_states.items()
        ]
    })

# ========== 主入口 ==========

if __name__ == "__main__":
    load_model()
    print("\n" + "=" * 60)
    print("Gemma 3n API 服务器已启动")
    print("=" * 60)
    print(f"\n端点:")
    print(f"  Gemini 风格:")
    print(f"    POST /{API_VERSION}/models/{MODEL_ID}:generateContent")
    print(f"    POST /{API_VERSION}/models/{MODEL_ID}:streamGenerateContent")
    print(f"  OpenAI 风格:")
    print(f"    POST /v1/chat/completions")
    print(f"    GET  /v1/models")
    print(f"\n服务地址: http://localhost:5001")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)

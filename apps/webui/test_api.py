"""
测试 Gemma 3n API 服务器

用法:
1. 先启动 api_server.py: python apps/webui/api_server.py
2. 运行测试: python apps/webui/test_api.py
"""
import requests
import json
import base64
from pathlib import Path

API_BASE = "http://localhost:5001"

def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("测试: 健康检查")
    print("=" * 60)

    resp = requests.get(f"{API_BASE}/health")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()

def test_list_models_openai():
    """测试 OpenAI 风格模型列表"""
    print("=" * 60)
    print("测试: OpenAI 风格 - 列出模型")
    print("=" * 60)

    resp = requests.get(f"{API_BASE}/v1/models")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()

def test_list_models_gemini():
    """测试 Gemini 风格模型列表"""
    print("=" * 60)
    print("测试: Gemini 风格 - 列出模型")
    print("=" * 60)

    resp = requests.get(f"{API_BASE}/v1beta/models")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()

def test_openai_chat_simple():
    """测试 OpenAI 风格简单对话"""
    print("=" * 60)
    print("测试: OpenAI 风格 - 简单对话")
    print("=" * 60)

    payload = {
        "model": "gemma-3n-e2b-it",
        "messages": [
            {"role": "user", "content": "Hello! What can you do?"}
        ],
        "max_tokens": 100
    }

    resp = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {resp.status_code}")
    data = resp.json()
    if "choices" in data:
        print(f"回复: {data['choices'][0]['message']['content']}")
        print(f"Token 使用: {data.get('usage', {})}")
    else:
        print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()

def test_openai_chat_stream():
    """测试 OpenAI 风格流式对话"""
    print("=" * 60)
    print("测试: OpenAI 风格 - 流式对话")
    print("=" * 60)

    payload = {
        "model": "gemma-3n-e2b-it",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        "max_tokens": 50,
        "stream": True
    }

    resp = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    )

    print(f"状态码: {resp.status_code}")
    print("流式响应:")

    full_text = ""
    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_text += content
                except json.JSONDecodeError:
                    pass

    print(f"\n\n完整文本: {full_text}")
    print()

def test_gemini_generate_simple():
    """测试 Gemini 风格简单生成"""
    print("=" * 60)
    print("测试: Gemini 风格 - 简单生成")
    print("=" * 60)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "What is 2 + 2?"}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 50
        }
    }

    resp = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:generateContent",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {resp.status_code}")
    data = resp.json()
    if "candidates" in data:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"回复: {text}")
        print(f"Token 使用: {data.get('usageMetadata', {})}")
    else:
        print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()

def test_gemini_generate_multiturn():
    """测试 Gemini 风格多轮对话"""
    print("=" * 60)
    print("测试: Gemini 风格 - 多轮对话")
    print("=" * 60)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "My name is Alice."}]
            },
            {
                "role": "model",
                "parts": [{"text": "Nice to meet you, Alice!"}]
            },
            {
                "role": "user",
                "parts": [{"text": "What's my name?"}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 50
        }
    }

    resp = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:generateContent",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {resp.status_code}")
    data = resp.json()
    if "candidates" in data:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"回复: {text}")
    else:
        print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()

def test_gemini_system_instruction():
    """测试 Gemini 风格系统指令"""
    print("=" * 60)
    print("测试: Gemini 风格 - 系统指令")
    print("=" * 60)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello"}]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": "You are a pirate. Always speak like a pirate."}
            ]
        },
        "generationConfig": {
            "maxOutputTokens": 100
        }
    }

    resp = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:generateContent",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {resp.status_code}")
    data = resp.json()
    if "candidates" in data:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"回复: {text}")
    else:
        print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()

def test_gemini_stream():
    """测试 Gemini 风格流式生成"""
    print("=" * 60)
    print("测试: Gemini 风格 - 流式生成")
    print("=" * 60)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Write a haiku about coding."}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 100
        }
    }

    resp = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:streamGenerateContent",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    )

    print(f"状态码: {resp.status_code}")
    print("流式响应:")

    full_text = ""
    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if "candidates" in data:
                        parts = data["candidates"][0]["content"]["parts"]
                        if parts:
                            text = parts[0].get("text", "")
                            print(text, end="", flush=True)
                            full_text += text
                except json.JSONDecodeError:
                    pass

    print(f"\n\n完整文本: {full_text}")
    print()

def test_gemini_function_calling():
    """测试 Gemini 风格 Function Calling"""
    print("=" * 60)
    print("测试: Gemini 风格 - Function Calling")
    print("=" * 60)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "What's the weather in Tokyo?"}]
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name, e.g. 'Tokyo'"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 200
        }
    }

    resp = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:generateContent",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {resp.status_code}")
    data = resp.json()
    if "candidates" in data:
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            if "functionCall" in part:
                print(f"Function Call: {json.dumps(part['functionCall'], indent=2)}")
            if "text" in part:
                print(f"Text: {part['text']}")
    else:
        print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print()

def test_thought_signature():
    """测试 Thought Signature 功能"""
    print("=" * 60)
    print("测试: Thought Signature - 多轮对话")
    print("=" * 60)

    session_id = None

    # 第一轮对话
    payload1 = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "My name is Alice. Remember this."}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 100
        }
    }

    resp1 = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:generateContent",
        json=payload1,
        headers={"Content-Type": "application/json"}
    )

    print(f"第一轮状态码: {resp1.status_code}")
    data1 = resp1.json()

    if "sessionId" in data1:
        session_id = data1["sessionId"]
        print(f"会话 ID: {session_id}")

    if "candidates" in data1:
        parts = data1["candidates"][0]["content"]["parts"]
        for part in parts:
            if "text" in part:
                print(f"回复: {part['text'][:100]}...")
            if "thoughtSignature" in part:
                print(f"Thought Signature: {part['thoughtSignature'][:50]}...")

    # 第二轮对话 (使用相同的 session_id)
    payload2 = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "My name is Alice. Remember this."}]
            },
            {
                "role": "model",
                "parts": [{"text": data1.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")}]
            },
            {
                "role": "user",
                "parts": [{"text": "What is my name?"}]
            }
        ],
        "sessionId": session_id,
        "generationConfig": {
            "maxOutputTokens": 100
        }
    }

    resp2 = requests.post(
        f"{API_BASE}/v1beta/models/gemma-3n-e2b-it:generateContent",
        json=payload2,
        headers={"Content-Type": "application/json"}
    )

    print(f"\n第二轮状态码: {resp2.status_code}")
    data2 = resp2.json()

    if "candidates" in data2:
        parts = data2["candidates"][0]["content"]["parts"]
        for part in parts:
            if "text" in part:
                print(f"回复: {part['text']}")
            if "thoughtSignature" in part:
                print(f"Thought Signature: {part['thoughtSignature'][:50]}...")

    # 获取思维状态
    if session_id:
        print(f"\n获取会话思维状态:")
        state_resp = requests.get(f"{API_BASE}/api/thought/state/{session_id}")
        print(f"思维状态: {json.dumps(state_resp.json(), indent=2)}")

    print()

def test_thought_signature_stats():
    """测试 Thought Signature 统计"""
    print("=" * 60)
    print("测试: Thought Signature - 统计")
    print("=" * 60)

    resp = requests.get(f"{API_BASE}/api/thought/stats")
    print(f"状态码: {resp.status_code}")
    print(f"统计: {json.dumps(resp.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Gemma 3n API 测试")
    print("=" * 60 + "\n")

    try:
        # 基础测试
        test_health()
        test_list_models_openai()
        test_list_models_gemini()

        # OpenAI 风格
        test_openai_chat_simple()
        test_openai_chat_stream()

        # Gemini 风格
        test_gemini_generate_simple()
        test_gemini_generate_multiturn()
        test_gemini_system_instruction()
        test_gemini_stream()

        # Function Calling
        test_gemini_function_calling()

        # Thought Signature 测试
        test_thought_signature()
        test_thought_signature_stats()

        print("=" * 60)
        print("所有测试完成!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
        print("请先启动服务器: python apps/webui/api_server.py")

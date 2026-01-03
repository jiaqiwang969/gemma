#!/usr/bin/env python3
"""
Gemini API 兼容性测试脚本
测试本地服务器是否完全兼容 Gemini API 格式

使用方法:
    python test_api.py [base_url]

默认 base_url: http://localhost:5001
"""

import sys
import json
import requests
from typing import Dict, Any

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"
API_VERSION = "v1beta"


def print_result(name: str, success: bool, response: Dict = None, expected_keys: list = None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n{status} - {name}")

    if response and expected_keys:
        missing = [k for k in expected_keys if k not in response]
        if missing:
            print(f"  ⚠️  Missing keys: {missing}")

    if not success and response:
        if "error" in response:
            print(f"  Error: {response['error']}")
        else:
            print(f"  Response: {json.dumps(response, ensure_ascii=False)[:200]}...")


def validate_response_format(response: Dict) -> bool:
    """验证响应格式是否符合 Gemini API 规范"""
    required_keys = ["candidates", "usageMetadata", "modelVersion", "responseId"]
    for key in required_keys:
        if key not in response:
            return False

    # 验证 candidates 结构
    if not response.get("candidates") or len(response["candidates"]) == 0:
        return False

    candidate = response["candidates"][0]
    if "content" not in candidate or "finishReason" not in candidate:
        return False

    content = candidate["content"]
    if "role" not in content or "parts" not in content:
        return False

    # 验证 usageMetadata 结构
    usage = response.get("usageMetadata", {})
    usage_keys = ["promptTokenCount", "candidatesTokenCount", "totalTokenCount"]
    for key in usage_keys:
        if key not in usage:
            return False

    return True


def test_health():
    """测试健康检查端点"""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        success = resp.status_code == 200 and data.get("status") == "ok"
        print_result("Health Check", success, data, ["status", "model", "server_ready"])
        return success
    except Exception as e:
        print_result("Health Check", False, {"error": str(e)})
        return False


def test_basic_text():
    """测试基础文本生成"""
    try:
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [{"parts": [{"text": "hi"}]}]
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)
        print_result("Basic Text Generation", success, data)

        # 验证 parts 结构
        if success:
            parts = data["candidates"][0]["content"]["parts"]
            has_text = any("text" in p for p in parts)
            has_signature = any("thoughtSignature" in p for p in parts)
            print(f"  📝 Has text: {has_text}, Has signature: {has_signature}")

        return success
    except Exception as e:
        print_result("Basic Text Generation", False, {"error": str(e)})
        return False


def test_thinking_level():
    """测试思考等级配置"""
    try:
        # 测试 low thinking level
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [{"parts": [{"text": "What is 2+2?"}]}],
                "generationConfig": {
                    "thinkingConfig": {"thinkingLevel": "low"}
                }
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)
        print_result("Thinking Level (low)", success, data)
        return success
    except Exception as e:
        print_result("Thinking Level (low)", False, {"error": str(e)})
        return False


def test_system_instruction():
    """测试系统指令"""
    try:
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [{"parts": [{"text": "你好"}]}],
                "systemInstruction": {
                    "parts": [{"text": "你是小智，回答以「小智：」开头"}]
                }
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)

        # 验证回复是否遵循系统指令
        if success:
            text = data["candidates"][0]["content"]["parts"][-1].get("text", "")
            follows_instruction = "小智" in text
            print(f"  📝 Follows instruction: {follows_instruction}")
            success = success and follows_instruction

        print_result("System Instruction", success, data)
        return success
    except Exception as e:
        print_result("System Instruction", False, {"error": str(e)})
        return False


def test_json_schema():
    """测试 JSON Schema 结构化输出"""
    try:
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [{"parts": [{"text": "列出2种编程语言"}]}],
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
                                        "year": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)

        # 验证输出是否为有效 JSON
        if success:
            text = data["candidates"][0]["content"]["parts"][-1].get("text", "")
            try:
                parsed = json.loads(text)
                is_valid_json = "languages" in parsed
                print(f"  📝 Valid JSON: {is_valid_json}")
                success = success and is_valid_json
            except:
                print(f"  ⚠️  Output is not valid JSON")
                success = False

        print_result("JSON Schema Output", success, data)
        return success
    except Exception as e:
        print_result("JSON Schema Output", False, {"error": str(e)})
        return False


def test_function_call_step1():
    """测试工具调用第一步"""
    try:
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "执行 ls 命令"}]}],
                "tools": [{
                    "functionDeclarations": [{
                        "name": "shell_command",
                        "description": "Run shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"]
                        }
                    }]
                }]
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)

        # 验证是否返回了 functionCall
        if success:
            parts = data["candidates"][0]["content"]["parts"]
            has_function_call = any("functionCall" in p for p in parts)
            has_signature = any("thoughtSignature" in p for p in parts)
            print(f"  📝 Has functionCall: {has_function_call}, Has signature: {has_signature}")
            success = success and has_function_call and has_signature

        print_result("Function Call (Step 1)", success, data)
        return success
    except Exception as e:
        print_result("Function Call (Step 1)", False, {"error": str(e)})
        return False


def test_function_call_step2():
    """测试工具调用第二步 (回放)"""
    try:
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "执行 ls 命令"}]},
                    {"role": "model", "parts": [{
                        "functionCall": {"name": "shell_command", "args": {"command": "ls"}},
                        "thoughtSignature": "test-signature"
                    }]},
                    {"role": "user", "parts": [{
                        "functionResponse": {
                            "name": "shell_command",
                            "response": {"output": "file1.txt\nfile2.txt", "success": True}
                        }
                    }]}
                ],
                "tools": [{
                    "functionDeclarations": [{
                        "name": "shell_command",
                        "description": "Run shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"]
                        }
                    }]
                }]
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)

        # 验证是否返回了文本响应（而不是又一个 functionCall）
        if success:
            parts = data["candidates"][0]["content"]["parts"]
            has_text = any("text" in p and not p.get("thought") for p in parts)
            print(f"  📝 Has text response: {has_text}")
            success = success and has_text

        print_result("Function Call (Step 2 - Replay)", success, data)
        return success
    except Exception as e:
        print_result("Function Call (Step 2 - Replay)", False, {"error": str(e)})
        return False


def test_multi_turn():
    """测试多轮对话"""
    try:
        resp = requests.post(
            f"{BASE_URL}/{API_VERSION}/models/gemini-3-pro-preview:generateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "我叫小明"}]},
                    {"role": "model", "parts": [
                        {"text": "你好小明！"},
                        {"thoughtSignature": "test-sig"}
                    ]},
                    {"role": "user", "parts": [{"text": "我叫什么名字？"}]}
                ]
            },
            timeout=60
        )
        data = resp.json()
        success = validate_response_format(data)

        # 验证是否记住了上下文
        if success:
            text = data["candidates"][0]["content"]["parts"][-1].get("text", "")
            remembers_context = "小明" in text or "Ming" in text.lower()
            print(f"  📝 Remembers context: {remembers_context}")

        print_result("Multi-turn Conversation", success, data)
        return success
    except Exception as e:
        print_result("Multi-turn Conversation", False, {"error": str(e)})
        return False


def main():
    print("=" * 60)
    print("Gemini API 兼容性测试")
    print(f"Base URL: {BASE_URL}")
    print("=" * 60)

    tests = [
        ("Health Check", test_health),
        ("Basic Text", test_basic_text),
        ("Thinking Level", test_thinking_level),
        ("System Instruction", test_system_instruction),
        ("JSON Schema", test_json_schema),
        ("Function Call Step 1", test_function_call_step1),
        ("Function Call Step 2", test_function_call_step2),
        ("Multi-turn", test_multi_turn),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ FAIL - {name}")
            print(f"  Unexpected error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！API 格式完全兼容 Gemini API")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试未通过")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Gemini API 全面对比测试脚本
═══════════════════════════════════════════════════════════════════════════════

对比真实 Gemini API 与本地 Gemma 3n 服务器的响应格式和功能。

测试用例:
  1.1 纯文本生成 (hi)
  1.2 思考等级 (thinkingLevel: low/medium/high)
  1.3 系统指令 (systemInstruction)
  1.4 JSON Schema 结构化输出
  1.5 多轮对话 (multi-turn)
  2.1 工具调用 Step1 (functionCall)
  2.2 工具调用 Step2 (functionResponse)

使用方法:
  python apps/gemini_api/test_comparison.py
"""

import json
import time
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from collections import OrderedDict


# ========== 配置 ==========
REAL_API_URL = "https://api.vectorengine.ai"
REAL_API_KEY = "sk-jSjH0gc2EZNkKSJdBUcVgZN0RXdiOWzzAhWvS4X2goJakiNK"
REAL_MODEL = "gemini-3-pro-preview"

LOCAL_API_URL = "http://localhost:5001"
LOCAL_MODEL = "gemini-3-pro-preview"  # 使用相同的路由


# ========== 测试用例 ==========
TEST_CASES = {
    "1.1_basic_text": {
        "name": "1.1 纯文本生成",
        "body": {
            "contents": [{"parts": [{"text": "hi"}]}]
        }
    },
    "1.2_thinking_low": {
        "name": "1.2 思考等级 low",
        "body": {
            "contents": [{"parts": [{"text": "What is 2+2?"}]}],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": "low"},
                "maxOutputTokens": 128
            }
        }
    },
    "1.3_thinking_high": {
        "name": "1.3 思考等级 high",
        "body": {
            "contents": [{"parts": [{"text": "Explain quantum entanglement briefly."}]}],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": "high"},
                "maxOutputTokens": 256
            }
        }
    },
    "1.4_system_instruction": {
        "name": "1.4 系统指令",
        "body": {
            "contents": [{"parts": [{"text": "Hello, introduce yourself"}]}],
            "systemInstruction": {
                "parts": [{"text": "You are a helpful assistant named Gemma. Always start your response with 'Gemma:'."}]
            },
            "generationConfig": {"maxOutputTokens": 128}
        }
    },
    "1.5_json_schema": {
        "name": "1.5 JSON Schema",
        "body": {
            "contents": [{"parts": [{"text": "List 2 programming languages with their release year"}]}],
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
                },
                "maxOutputTokens": 256
            }
        }
    },
    "1.6_multi_turn": {
        "name": "1.6 多轮对话",
        "body": {
            "contents": [
                {"role": "user", "parts": [{"text": "My name is Alice"}]},
                {"role": "model", "parts": [{"text": "Nice to meet you, Alice!"}]},
                {"role": "user", "parts": [{"text": "What is my name?"}]}
            ],
            "generationConfig": {"maxOutputTokens": 64}
        }
    },
    "2.1_function_call": {
        "name": "2.1 工具调用 Step1",
        "body": {
            "contents": [{"role": "user", "parts": [{"text": "Run the ls command"}]}],
            "tools": [{
                "functionDeclarations": [{
                    "name": "shell_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The command to run"}
                        },
                        "required": ["command"]
                    }
                }]
            }],
            "generationConfig": {"maxOutputTokens": 128}
        }
    },
    "2.2_function_response": {
        "name": "2.2 工具调用 Step2",
        "skip_real_api": True,  # 真实 API 需要有效签名，跳过对比
        "body": {
            "contents": [
                {"role": "user", "parts": [{"text": "Run the ls command"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "shell_command", "args": {"command": "ls"}}},
                    {"thoughtSignature": "skip_thought_signature_validator"}  # 使用官方跳过验证的签名
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {
                        "name": "shell_command",
                        "response": {"output": "file1.txt\nfile2.txt\nfolder1", "exit_code": 0}
                    }}
                ]}
            ],
            "tools": [{
                "functionDeclarations": [{
                    "name": "shell_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }]
            }],
            "generationConfig": {"maxOutputTokens": 128}
        }
    }
}


# ========== API 调用函数 ==========
def call_api(base_url: str, model: str, body: Dict, api_key: str = "") -> Dict:
    """调用 API 并返回响应"""
    if api_key:
        url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"
    else:
        url = f"{base_url}/v1beta/models/{model}:generateContent"

    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            elapsed = int((time.time() - start_time) * 1000)
            return {"success": True, "data": result, "time_ms": elapsed}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_response_text(response: Dict) -> str:
    """从响应中提取文本"""
    try:
        parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part and not part.get("thought"):
                return part["text"][:200] + ("..." if len(part.get("text", "")) > 200 else "")
        return ""
    except:
        return ""


def extract_function_call(response: Dict) -> Optional[Dict]:
    """从响应中提取函数调用"""
    try:
        parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            if "functionCall" in part:
                return part["functionCall"]
        return None
    except:
        return None


def analyze_signature(sig: str) -> Dict:
    """分析 thoughtSignature"""
    try:
        decoded = base64.b64decode(sig)
        first_byte = decoded[0]
        return {
            "length": len(sig),
            "decoded_bytes": len(decoded),
            "format": "Protobuf" if first_byte == 0x12 else "JSON" if first_byte == 0x7b else "Unknown",
            "first_byte": f"0x{first_byte:02x}"
        }
    except:
        return {"error": "Failed to decode"}


def compare_field_order(real: Dict, local: Dict) -> List[str]:
    """比较字段顺序"""
    issues = []

    # 顶层字段顺序
    expected_order = ["candidates", "usageMetadata", "modelVersion", "responseId"]
    real_keys = list(real.keys())
    local_keys = list(local.keys())

    for i, key in enumerate(expected_order):
        if i < len(real_keys) and real_keys[i] != key:
            issues.append(f"Real API: expected '{key}' at position {i}, got '{real_keys[i]}'")
        if i < len(local_keys) and local_keys[i] != key:
            issues.append(f"Local: expected '{key}' at position {i}, got '{local_keys[i]}'")

    # usageMetadata 内部顺序
    expected_usage_order = ["promptTokenCount", "candidatesTokenCount", "totalTokenCount", "promptTokensDetails", "thoughtsTokenCount"]
    real_usage = real.get("usageMetadata", {})
    local_usage = local.get("usageMetadata", {})

    real_usage_keys = list(real_usage.keys())
    local_usage_keys = list(local_usage.keys())

    for i, key in enumerate(expected_usage_order):
        if key in real_usage_keys:
            real_pos = real_usage_keys.index(key)
            if real_pos != i:
                pass  # Real API 顺序作为参考
        if key in local_usage_keys:
            local_pos = local_usage_keys.index(key)
            # 检查是否与 real 一致
            if key in real_usage_keys:
                expected_pos = real_usage_keys.index(key)
                if local_pos != expected_pos:
                    issues.append(f"usageMetadata.{key}: Real at {expected_pos}, Local at {local_pos}")

    return issues


def compare_responses(real: Dict, local: Dict, test_name: str) -> Dict:
    """全面对比两个响应"""
    result = {
        "test_name": test_name,
        "checks": [],
        "passed": 0,
        "failed": 0
    }

    def check(name: str, condition: bool, detail: str = ""):
        status = "✅" if condition else "❌"
        result["checks"].append({"name": name, "passed": condition, "detail": detail})
        if condition:
            result["passed"] += 1
        else:
            result["failed"] += 1

    # 1. 顶层字段存在性
    for field in ["candidates", "usageMetadata", "modelVersion", "responseId"]:
        check(f"顶层字段 {field}", field in local, f"Real: {field in real}, Local: {field in local}")

    # 2. 字段顺序
    real_keys = list(real.keys())
    local_keys = list(local.keys())
    expected = ["candidates", "usageMetadata", "modelVersion", "responseId"]
    order_match = local_keys[:4] == expected[:len(local_keys[:4])]
    check("顶层字段顺序", order_match, f"Expected: {expected}, Got: {local_keys[:4]}")

    # 3. candidates 结构
    if "candidates" in real and "candidates" in local:
        real_cand = real["candidates"][0] if real["candidates"] else {}
        local_cand = local["candidates"][0] if local["candidates"] else {}

        check("candidates[0].content 存在", "content" in local_cand)
        check("candidates[0].finishReason 存在", "finishReason" in local_cand)
        check("candidates[0].index 存在", "index" in local_cand)

        # 检查是否有多余字段
        extra_fields = set(local_cand.keys()) - set(real_cand.keys())
        check("candidates 无多余字段", len(extra_fields) == 0, f"Extra: {extra_fields}" if extra_fields else "")

    # 4. usageMetadata 字段
    if "usageMetadata" in real and "usageMetadata" in local:
        real_usage = real["usageMetadata"]
        local_usage = local["usageMetadata"]

        for field in ["promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount"]:
            check(f"usageMetadata.{field}", field in local_usage)

        check("usageMetadata.promptTokensDetails", "promptTokensDetails" in local_usage)

        # 顺序检查
        expected_usage = ["promptTokenCount", "candidatesTokenCount", "totalTokenCount", "promptTokensDetails", "thoughtsTokenCount"]
        local_usage_keys = list(local_usage.keys())
        usage_order_match = all(
            local_usage_keys.index(k) < local_usage_keys.index(expected_usage[i+1])
            for i, k in enumerate(expected_usage[:-1])
            if k in local_usage_keys and expected_usage[i+1] in local_usage_keys
        )
        check("usageMetadata 字段顺序", usage_order_match)

    # 5. thoughtSignature 格式
    real_sig = None
    local_sig = None
    try:
        for part in real.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "thoughtSignature" in part:
                real_sig = part["thoughtSignature"]
                break
        for part in local.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "thoughtSignature" in part:
                local_sig = part["thoughtSignature"]
                break
    except:
        pass

    if real_sig and local_sig:
        real_analysis = analyze_signature(real_sig)
        local_analysis = analyze_signature(local_sig)

        check("thoughtSignature 格式 (Protobuf)",
              local_analysis.get("format") == "Protobuf",
              f"Real: {real_analysis.get('format')}, Local: {local_analysis.get('format')}")

        check("thoughtSignature 首字节 (0x12)",
              local_analysis.get("first_byte") == "0x12",
              f"Real: {real_analysis.get('first_byte')}, Local: {local_analysis.get('first_byte')}")

        # 长度比较 (允许 50% 差异)
        if "length" in real_analysis and "length" in local_analysis:
            ratio = local_analysis["length"] / real_analysis["length"]
            check("thoughtSignature 长度合理 (>50%)",
                  ratio > 0.5,
                  f"Real: {real_analysis['length']} chars, Local: {local_analysis['length']} chars ({ratio:.0%})")

    return result


# ========== 主测试函数 ==========
def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("Gemini API 全面对比测试")
    print("=" * 80)
    print(f"真实 API: {REAL_API_URL}")
    print(f"本地服务器: {LOCAL_API_URL}")
    print("=" * 80)
    print()

    all_results = []

    for test_id, test_config in TEST_CASES.items():
        test_name = test_config["name"]
        body = test_config["body"]
        skip_real_api = test_config.get("skip_real_api", False)

        print(f"\n{'─' * 60}")
        print(f"测试: {test_name}")
        print(f"{'─' * 60}")

        real_result = None

        if skip_real_api:
            print("  [1/2] 跳过真实 API (需要有效 thoughtSignature)")
        else:
            # 调用真实 API
            print("  [1/2] 调用真实 Gemini API...")
            real_result = call_api(REAL_API_URL, REAL_MODEL, body, REAL_API_KEY)

            if not real_result["success"]:
                print(f"    ❌ 失败: {real_result['error'][:100]}")
                continue

            print(f"    ✅ 成功 ({real_result['time_ms']}ms)")
            real_text = extract_response_text(real_result["data"])
            if real_text:
                print(f"    回复: {real_text[:80]}...")

            # 短暂延迟避免 rate limit
            time.sleep(1)

        # 调用本地 API
        print("  [2/2] 调用本地服务器...")
        local_result = call_api(LOCAL_API_URL, LOCAL_MODEL, body)

        if not local_result["success"]:
            print(f"    ❌ 失败: {local_result['error'][:100]}")
            continue

        print(f"    ✅ 成功 ({local_result['time_ms']}ms)")
        local_text = extract_response_text(local_result["data"])
        if local_text:
            print(f"    回复: {local_text[:80]}...")

        # 对比分析 (如果有真实 API 结果)
        if real_result and real_result["success"]:
            print("\n  对比分析:")
            comparison = compare_responses(real_result["data"], local_result["data"], test_name)
            all_results.append(comparison)

            for check in comparison["checks"]:
                status = "✅" if check["passed"] else "❌"
                detail = f" ({check['detail']})" if check["detail"] else ""
                print(f"    {status} {check['name']}{detail}")

            print(f"\n  结果: {comparison['passed']}/{comparison['passed'] + comparison['failed']} 通过")

            # 保存详细响应
            with open(f"/tmp/test_{test_id}_real.json", "w") as f:
                json.dump(real_result["data"], f, indent=2, ensure_ascii=False)
        else:
            # 只验证本地服务器格式正确性
            print("\n  本地服务器格式验证:")
            local_data = local_result["data"]
            checks_passed = 0
            checks_total = 0

            for field in ["candidates", "usageMetadata", "modelVersion", "responseId"]:
                checks_total += 1
                if field in local_data:
                    print(f"    ✅ 顶层字段 {field}")
                    checks_passed += 1
                else:
                    print(f"    ❌ 顶层字段 {field} 缺失")

            # 检查 thoughtSignature
            try:
                parts = local_data["candidates"][0]["content"]["parts"]
                has_sig = any("thoughtSignature" in p for p in parts)
                checks_total += 1
                if has_sig:
                    print("    ✅ thoughtSignature 存在")
                    checks_passed += 1
                else:
                    print("    ❌ thoughtSignature 缺失")
            except:
                pass

            print(f"\n  结果: {checks_passed}/{checks_total} 通过")
            all_results.append({"test_name": test_name, "passed": checks_passed, "failed": checks_total - checks_passed, "checks": []})

        with open(f"/tmp/test_{test_id}_local.json", "w") as f:
            json.dump(local_result["data"], f, indent=2, ensure_ascii=False)

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)

    for result in all_results:
        status = "✅" if result["failed"] == 0 else "⚠️" if result["failed"] < 3 else "❌"
        print(f"  {status} {result['test_name']}: {result['passed']}/{result['passed'] + result['failed']} 通过")

    print(f"\n总计: {total_passed}/{total_passed + total_failed} 检查通过")
    print(f"通过率: {total_passed / (total_passed + total_failed) * 100:.1f}%")

    # 找出所有失败的检查
    print("\n失败项目:")
    for result in all_results:
        for check in result["checks"]:
            if not check["passed"]:
                print(f"  ❌ [{result['test_name']}] {check['name']}: {check['detail']}")

    return all_results


if __name__ == "__main__":
    run_all_tests()

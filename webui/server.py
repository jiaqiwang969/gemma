"""
AI 多模态聊天服务器
支持: 文本 + 图片 + 音频 + 多轮对话历史
存储: ~/.gemma3n/ (参考 Codex 架构)
"""
import os
import io
import base64
import torch
import numpy as np
import uuid
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import librosa
import warnings
import time
import psutil
import subprocess
import platform
import threading

warnings.filterwarnings("ignore")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__, static_folder="static")
CORS(app)

# 全局变量
model = None
processor = None
model_loaded = False
model_info = {}
dummy_image = None

# 存储路径 (~/.gemma3n/)
GEMMA3N_HOME = Path.home() / ".gemma3n"
SESSIONS_DIR = GEMMA3N_HOME / "sessions"
HISTORY_FILE = GEMMA3N_HOME / "history.jsonl"

# 会话管理 (内存缓存)
sessions = {}  # session_id -> {"messages": [...], "created_at": timestamp, "file_path": ...}
MAX_SESSIONS = 100
MAX_HISTORY_TURNS = 10  # 保留最近N轮对话

# sudo 权限状态
sudo_authorized = False
sudo_refresh_thread = None

# ========== Thought Signature 系统 (压缩记忆) ==========
# 核心思想: 图片/音频 → 模型理解 → 存入 signature → 后续轮次恢复
import hashlib
import hmac

THOUGHT_SIGNATURE_SECRET = "gemma3n-thought-signature-key"
media_understanding_cache = {}  # media_ref -> {"understanding": "...", "session_id": "..."}
thought_states = {}  # session_id -> {"turn_index": 0, "media_refs": [...]}


def generate_media_signature(session_id: str, turn_index: int, understanding: str) -> str:
    """
    为媒体理解生成签名引用

    Returns: media_ref (用于后续恢复)
    """
    import base64
    timestamp = int(time.time())
    media_ref = hashlib.md5(f"{session_id}:{turn_index}:{timestamp}".encode()).hexdigest()[:12]

    # 存储理解到缓存
    media_understanding_cache[media_ref] = {
        "session_id": session_id,
        "turn_index": turn_index,
        "understanding": understanding,
        "created_at": timestamp
    }

    # 更新会话的思维状态
    if session_id not in thought_states:
        thought_states[session_id] = {"turn_index": 0, "media_refs": []}
    thought_states[session_id]["media_refs"].append(media_ref)
    thought_states[session_id]["turn_index"] = turn_index

    return media_ref


def get_session_media_context(session_id: str) -> str:
    """
    获取会话中所有媒体理解的上下文

    这是 thought signature 作为"压缩记忆"的核心：
    - 第1轮: 用户上传图片 → 模型生成理解 → 存入 signature
    - 第2轮: 从 signature 恢复理解 → 注入到上下文中
    """
    context_parts = []

    state = thought_states.get(session_id, {})
    media_refs = state.get("media_refs", [])

    for media_ref in media_refs:
        cached = media_understanding_cache.get(media_ref)
        if cached:
            turn = cached.get("turn_index", 0)
            understanding = cached.get("understanding", "")
            if understanding:
                context_parts.append(f"[Turn {turn} - Media Understanding]: {understanding}")

    return "\n".join(context_parts) if context_parts else ""

stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_time": 0,
    "avg_speed": 0
}

def init_storage():
    """初始化存储目录结构"""
    GEMMA3N_HOME.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.touch()
    print(f"[Storage] 初始化完成: {GEMMA3N_HOME}")

def get_session_dir():
    """获取当天的会话目录 sessions/YYYY/MM/DD/"""
    now = datetime.now()
    day_dir = SESSIONS_DIR / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir

def save_session_to_disk(session_id, session_data):
    """将会话保存到 JSONL 文件"""
    if "file_path" not in session_data:
        # 创建新文件
        now = datetime.now()
        timestamp = now.strftime("%H%M%S")
        filename = f"session-{timestamp}-{session_id}.jsonl"
        session_data["file_path"] = str(get_session_dir() / filename)

    file_path = Path(session_data["file_path"])

    # 写入会话数据 (完整覆盖)
    with open(file_path, "w", encoding="utf-8") as f:
        # 第一行: 会话元数据
        meta = {
            "type": "meta",
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "title": session_data.get("title", "新对话"),
            "updated_at": time.time()
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 后续行: 消息
        for msg in session_data["messages"]:
            item = {"type": "message", **msg}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_session_from_disk(file_path):
    """从 JSONL 文件加载会话"""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    session_data = {"messages": [], "file_path": str(file_path)}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "meta":
                session_data["created_at"] = item.get("created_at", time.time())
                session_data["title"] = item.get("title", "新对话")
                session_data["session_id"] = item.get("session_id")
            elif item.get("type") == "message":
                del item["type"]
                session_data["messages"].append(item)

    return session_data

def append_to_history(session_id, text):
    """追加到全局历史记录"""
    entry = {
        "session_id": session_id,
        "ts": int(time.time()),
        "text": text[:100]  # 只保存前100字符作为摘要
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def list_all_sessions():
    """列出所有会话文件"""
    session_list = []
    for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    meta = json.loads(first_line)
                    if meta.get("type") == "meta":
                        session_list.append({
                            "session_id": meta.get("session_id"),
                            "title": meta.get("title", "新对话"),
                            "created_at": meta.get("created_at", 0),
                            "updated_at": meta.get("updated_at", 0),
                            "file_path": str(jsonl_file)
                        })
        except Exception as e:
            print(f"[Warning] 无法读取会话文件 {jsonl_file}: {e}")

    # 按更新时间倒序
    session_list.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return session_list[:50]  # 最多返回50个

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return round(mem_gb, 2)


def request_sudo_permission():
    """
    启动时请求 sudo 权限（用户输入一次密码）
    并启动后台线程定期刷新凭证
    """
    global sudo_authorized, sudo_refresh_thread

    if platform.system() != "Darwin":
        return False

    print("\n" + "=" * 60)
    print("🔐 请求硬件监控权限 (用于获取 GPU 温度)")
    print("   请输入您的 macOS 密码（可选，按 Ctrl+C 跳过）")
    print("=" * 60)

    try:
        # 请求 sudo 权限
        result = subprocess.run(
            ["sudo", "-v"],
            timeout=60  # 给用户60秒输入密码
        )
        if result.returncode == 0:
            sudo_authorized = True
            print("✅ 权限授权成功！GPU 温度监控已启用")

            # 启动后台线程定期刷新 sudo 凭证
            def refresh_sudo():
                while sudo_authorized:
                    time.sleep(240)  # 每4分钟刷新一次（sudo 默认5分钟超时）
                    try:
                        subprocess.run(["sudo", "-v"], capture_output=True, timeout=5)
                    except:
                        pass

            sudo_refresh_thread = threading.Thread(target=refresh_sudo, daemon=True)
            sudo_refresh_thread.start()
            return True
        else:
            print("⚠️  权限未授权，GPU 温度监控将不可用")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  授权超时，GPU 温度监控将不可用")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  已跳过权限授权，GPU 温度监控将不可用")
        return False
    except Exception as e:
        print(f"⚠️  授权失败: {e}")
        return False


def get_hardware_stats():
    """获取硬件监控信息 (GPU使用率、显存、温度等)"""
    hw_stats = {
        "gpu_usage": "-",
        "vram_usage": "-",
        "gpu_temp": "-",
        "memory_usage": "-",
        "cpu_usage": "-"
    }

    try:
        # CPU 使用率 (不使用 interval 避免阻塞)
        hw_stats["cpu_usage"] = f"{psutil.cpu_percent():.1f}%"

        # 内存使用
        mem = psutil.virtual_memory()
        hw_stats["memory_usage"] = f"{mem.used / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB"

        system = platform.system()

        if system == "Darwin":  # macOS
            try:
                # 检测 MPS 状态
                if torch.backends.mps.is_available():
                    hw_stats["gpu_usage"] = "MPS 运行中"
                else:
                    hw_stats["gpu_usage"] = "CPU 模式"

                # macOS 统一内存架构，显存和内存共享
                hw_stats["vram_usage"] = "统一内存"

                # 尝试通过 powermetrics 获取 GPU 温度 (需要 sudo 权限)
                if sudo_authorized:
                    try:
                        result = subprocess.run(
                            ["sudo", "powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
                            capture_output=True, text=True, timeout=3
                        )
                        if result.returncode == 0:
                            output = result.stdout
                            # 查找 GPU 温度相关行
                            for line in output.split("\n"):
                                if "GPU" in line and "die" in line.lower():
                                    parts = line.split(":")
                                    if len(parts) >= 2:
                                        temp_str = parts[1].strip().replace("C", "").strip()
                                        try:
                                            temp = float(temp_str)
                                            hw_stats["gpu_temp"] = f"{temp:.0f}°C"
                                            break
                                        except:
                                            pass
                                elif "GPU" in line and "temp" in line.lower():
                                    parts = line.split(":")
                                    if len(parts) >= 2:
                                        temp_str = parts[1].strip().replace("C", "").strip()
                                        try:
                                            temp = float(temp_str)
                                            hw_stats["gpu_temp"] = f"{temp:.0f}°C"
                                            break
                                        except:
                                            pass

                            # 如果没找到 GPU 温度，尝试找 SOC 温度
                            if hw_stats["gpu_temp"] == "-":
                                for line in output.split("\n"):
                                    if "SOC" in line and "temp" in line.lower():
                                        parts = line.split(":")
                                        if len(parts) >= 2:
                                            temp_str = parts[1].strip().replace("C", "").strip()
                                            try:
                                                temp = float(temp_str)
                                                hw_stats["gpu_temp"] = f"{temp:.0f}°C"
                                                break
                                            except:
                                                pass
                    except subprocess.TimeoutExpired:
                        hw_stats["gpu_temp"] = "超时"
                    except Exception:
                        hw_stats["gpu_temp"] = "获取失败"
                else:
                    hw_stats["gpu_temp"] = "未授权"

            except Exception as e:
                print(f"[DEBUG] macOS GPU info error: {e}")

        elif system == "Linux":
            # NVIDIA GPU (使用 nvidia-smi)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        # 可能有多个 GPU，取第一个
                        parts = lines[0].split(",")
                        if len(parts) >= 4:
                            gpu_util = parts[0].strip()
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())
                            temp = parts[3].strip()

                            hw_stats["gpu_usage"] = f"{gpu_util}%"
                            hw_stats["vram_usage"] = f"{mem_used/1024:.1f} GB / {mem_total/1024:.1f} GB"
                            hw_stats["gpu_temp"] = f"{temp}°C"
            except FileNotFoundError:
                # nvidia-smi 不存在，可能是 AMD 或无 GPU
                try:
                    # 尝试 AMD GPU (rocm-smi)
                    result = subprocess.run(
                        ["rocm-smi", "--showuse", "--showtemp", "--showmeminfo", "vram"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        hw_stats["gpu_usage"] = "AMD GPU"
                except:
                    pass
            except Exception as e:
                print(f"[DEBUG] Linux GPU info error: {e}")

        elif system == "Windows":
            # Windows NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, shell=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        parts = lines[0].split(",")
                        if len(parts) >= 4:
                            gpu_util = parts[0].strip()
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())
                            temp = parts[3].strip()

                            hw_stats["gpu_usage"] = f"{gpu_util}%"
                            hw_stats["vram_usage"] = f"{mem_used/1024:.1f} GB / {mem_total/1024:.1f} GB"
                            hw_stats["gpu_temp"] = f"{temp}°C"
            except Exception as e:
                print(f"[DEBUG] Windows GPU info error: {e}")

    except Exception as e:
        print(f"[DEBUG] Hardware stats error: {e}")

    return hw_stats

def cleanup_old_sessions():
    """清理内存中的旧会话缓存 (磁盘文件保留)"""
    if len(sessions) > MAX_SESSIONS:
        # 按创建时间排序，从内存中移除最老的
        sorted_sessions = sorted(sessions.items(), key=lambda x: x[1]["created_at"])
        for sid, _ in sorted_sessions[:len(sessions) - MAX_SESSIONS]:
            del sessions[sid]

def load_model():
    global model, processor, model_loaded, model_info, dummy_image

    if model_loaded:
        return True

    print("=" * 60)
    print("加载 AI 多模态模型...")
    print("=" * 60)

    model_name = "google/gemma-3n-E2B-it"
    load_start = time.time()

    print("[1/2] 加载处理器...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print("[2/2] 加载模型...")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_name,
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
        "name": "AI Multimodal",
        "params": f"{total_params / 1e9:.2f}B",
        "dtype": "bfloat16",
        "device": str(model.device),
        "load_time": round(load_time, 2),
        "memory_gb": get_memory_usage(),
        "capabilities": ["文本对话", "图像理解", "音频转录", "多轮对话"],
        "max_tokens": 8192,
    }

    model_loaded = True
    print("=" * 60)
    print(f"模型加载完成! 耗时 {load_time:.2f}s")
    print(f"内存占用: {model_info['memory_gb']} GB")
    print("=" * 60)
    return True

def generate_response(messages_history, current_content, session_id=None, has_media=False, media_type=None):
    """
    生成回复
    messages_history: 历史消息列表 [{"role": "user/assistant", "text": "..."}]
    current_content: 当前消息的content列表
    session_id: 会话ID (用于 thought signature)
    has_media: 当前消息是否包含媒体 (图片/音频)
    media_type: 当前媒体类型 ("image", "audio", None)

    注意: Gemma 3n 处理器要求每条消息都有图片，否则会报批次大小不一致错误。
    解决方案: 将历史对话合并成上下文文本，只发送一条包含图片的当前消息。
    """
    global stats

    if not model_loaded:
        return {"error": "模型未加载"}

    start_time = time.time()
    history_turns = len(messages_history) // 2  # 记录实际的历史轮次

    # 获取历史媒体理解上下文 (thought signature 压缩记忆)
    # 策略：
    # - 如果当前消息有新媒体：不注入历史媒体理解（避免干扰）
    # - 如果当前消息没有新媒体：注入历史媒体理解，让模型能够回答关于之前媒体的问题
    media_context = ""
    if session_id and not has_media:
        # 只在没有新媒体时才注入历史媒体理解
        media_context = get_session_media_context(session_id)
        if media_context:
            print(f"[DEBUG] 注入媒体理解上下文: {len(media_context)} 字符")
    elif has_media:
        print(f"[DEBUG] 当前有新{media_type or '媒体'}，跳过历史媒体理解注入")

    # 方案: 将历史对话合并成上下文文本
    # 这样就只有一条 user 消息，避免批次不一致问题
    # 注意: 当有新媒体时，不注入历史对话，避免干扰模型对当前媒体的理解
    history_context = ""
    if messages_history and not has_media:
        history_parts = []
        for msg in messages_history[-MAX_HISTORY_TURNS * 2:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['text']}")
        history_context = "\n".join(history_parts)
    elif has_media:
        print(f"[DEBUG] 当前有新媒体，跳过历史对话注入")

    # 构建单条消息 (包含历史上下文 + 当前输入)
    messages = []

    # 修改当前消息内容，添加历史上下文
    modified_content = []
    current_text = ""

    for item in current_content:
        if item.get("type") == "text":
            current_text = item.get("text", "")
        else:
            modified_content.append(item)

    # 构建完整的上下文提示
    # 包含: 媒体理解 + 历史对话 + 当前消息
    full_context_parts = []

    # 1. 媒体理解上下文 (thought signature 压缩记忆) - 只在无新媒体时注入
    if media_context:
        full_context_parts.append(f"[Previous Media Understanding]\n{media_context}")

    # 2. 历史对话上下文
    if history_context:
        full_context_parts.append(f"[Previous Conversation]\n{history_context}")

    # 3. 当前消息
    if full_context_parts:
        # 有上下文，构建完整提示
        context_str = "\n\n".join(full_context_parts)
        context_prompt = f"""{context_str}

[Current Message]: {current_text}

Please respond to the current message, taking into account the context above."""
        modified_content.append({"type": "text", "text": context_prompt})
    else:
        # 无上下文，直接使用当前文本
        modified_content.append({"type": "text", "text": current_text})

    messages.append({"role": "user", "content": modified_content})

    print(f"[DEBUG] 消息数量: {len(messages)}, 历史轮次: {history_turns}, 有新媒体: {has_media}")

    # 处理输入
    tokenize_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    tokenize_time = time.time() - tokenize_start

    # 准备生成参数
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_tokens = input_ids.shape[1]

    print(f"[DEBUG] input_tokens: {input_tokens}")

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 512,
        "do_sample": False,
    }

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
    response = processor.tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    total_time = time.time() - start_time
    speed = output_tokens / generate_time if generate_time > 0 else 0

    # 更新统计
    stats["total_requests"] += 1
    stats["total_tokens"] += output_tokens
    stats["total_time"] += total_time
    stats["avg_speed"] = stats["total_tokens"] / stats["total_time"] if stats["total_time"] > 0 else 0

    return {
        "response": response,
        "metrics": {
            "total_time": round(total_time, 2),
            "tokenize_time": round(tokenize_time, 3),
            "generate_time": round(generate_time, 2),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "speed": round(speed, 1),
            "history_turns": history_turns  # 使用实际的历史轮次
        }
    }

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    hw_stats = get_hardware_stats() if model_loaded else {}
    return jsonify({
        "loaded": model_loaded,
        "stats": stats,
        "memory_gb": get_memory_usage() if model_loaded else 0,
        "hardware": hw_stats,
        "active_sessions": len(sessions)
    })

@app.route("/api/session/new", methods=["POST"])
def new_session():
    """创建新会话"""
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "messages": [],
        "created_at": time.time(),
        "title": "新对话"
    }
    sessions[session_id] = session_data
    # 立即保存到磁盘
    save_session_to_disk(session_id, session_data)
    cleanup_old_sessions()
    return jsonify({"session_id": session_id})

@app.route("/api/session/list", methods=["GET"])
def list_sessions():
    """列出所有会话"""
    session_list = list_all_sessions()
    return jsonify({"sessions": session_list})

@app.route("/api/session/<session_id>/load", methods=["GET"])
def load_session(session_id):
    """加载指定会话"""
    # 先检查内存缓存
    if session_id in sessions:
        session = sessions[session_id]
        return jsonify({
            "session_id": session_id,
            "messages": session["messages"],
            "title": session.get("title", "新对话")
        })

    # 从磁盘查找
    for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
        if session_id in jsonl_file.name:
            session_data = load_session_from_disk(jsonl_file)
            if session_data:
                sessions[session_id] = session_data
                return jsonify({
                    "session_id": session_id,
                    "messages": session_data["messages"],
                    "title": session_data.get("title", "新对话")
                })

    return jsonify({"error": "会话不存在"}), 404

@app.route("/api/session/<session_id>/delete", methods=["POST"])
def delete_session(session_id):
    """删除会话"""
    # 从内存移除
    if session_id in sessions:
        file_path = sessions[session_id].get("file_path")
        del sessions[session_id]
        # 删除磁盘文件
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
            return jsonify({"success": True})

    # 从磁盘查找并删除
    for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
        if session_id in jsonl_file.name:
            jsonl_file.unlink()
            return jsonify({"success": True})

    return jsonify({"success": True})

@app.route("/api/session/<session_id>/clear", methods=["POST"])
def clear_session(session_id):
    """清空会话历史"""
    if session_id in sessions:
        sessions[session_id]["messages"] = []
        sessions[session_id]["title"] = "新对话"
        # 保存到磁盘
        save_session_to_disk(session_id, sessions[session_id])
    return jsonify({"success": True})

@app.route("/api/session/<session_id>/history", methods=["GET"])
def get_history(session_id):
    """获取会话历史"""
    if session_id not in sessions:
        return jsonify({"messages": []})
    return jsonify({"messages": sessions[session_id]["messages"]})

@app.route("/api/thought/state/<session_id>", methods=["GET"])
def get_thought_state(session_id):
    """获取会话的 thought signature 状态"""
    state = thought_states.get(session_id, {"turn_index": 0, "media_refs": []})
    media_understandings = []

    for media_ref in state.get("media_refs", []):
        cached = media_understanding_cache.get(media_ref)
        if cached:
            media_understandings.append({
                "media_ref": media_ref,
                "turn_index": cached.get("turn_index", 0),
                "understanding_preview": cached.get("understanding", "")[:100] + "...",
                "created_at": cached.get("created_at", 0)
            })

    return jsonify({
        "session_id": session_id,
        "turn_index": state.get("turn_index", 0),
        "media_count": len(state.get("media_refs", [])),
        "media_understandings": media_understandings
    })

@app.route("/api/thought/stats", methods=["GET"])
def thought_stats():
    """获取 thought signature 全局统计"""
    return jsonify({
        "total_sessions": len(thought_states),
        "total_media_cached": len(media_understanding_cache),
        "sessions": [
            {
                "session_id": sid,
                "turn_index": state.get("turn_index", 0),
                "media_count": len(state.get("media_refs", []))
            }
            for sid, state in thought_states.items()
        ]
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        text = data.get("text", "")
        image_data = data.get("image")
        audio_data = data.get("audio")
        session_id = data.get("session_id")

        # 获取或创建会话
        if not session_id or session_id not in sessions:
            # 尝试从磁盘加载
            loaded = False
            if session_id:
                for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
                    if session_id in jsonl_file.name:
                        session_data = load_session_from_disk(jsonl_file)
                        if session_data:
                            sessions[session_id] = session_data
                            loaded = True
                            break
            if not loaded:
                session_id = str(uuid.uuid4())[:8]
                sessions[session_id] = {"messages": [], "created_at": time.time(), "title": "新对话"}

        session = sessions[session_id]
        image = None
        audio = None

        # 处理图片
        if image_data:
            if "," in image_data:
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 处理音频
        if audio_data:
            mime_part = ""
            if "," in audio_data:
                mime_part = audio_data.split(",")[0]
                audio_data = audio_data.split(",")[1]
            audio_bytes = base64.b64decode(audio_data)

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

            temp_path = f"/tmp/audio_{session_id}{ext}"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            audio_array, sr = librosa.load(temp_path, sr=16000)
            audio = (audio_array, sr)
            print(f"[DEBUG] 音频: {len(audio_array)/sr:.2f}秒")

        # 构建当前消息内容
        content = []
        has_media = (image is not None or audio is not None)

        if not has_media:
            # 纯文本消息：添加 dummy_image
            content.append({"type": "image", "image": dummy_image})
            display_text = text
            text = "Ignore the blank image. " + text
        else:
            display_text = text

        # Gemma 3n 要求每条消息都有图片
        # 如果只有音频没有图片，也需要添加 dummy_image
        if image is not None:
            content.append({"type": "image", "image": image})
        elif audio is not None:
            # 只有音频，添加 dummy_image
            content.append({"type": "image", "image": dummy_image})
            text = "Ignore the blank image. " + text

        if audio is not None:
            content.append({"type": "audio", "audio": audio[0], "sample_rate": audio[1]})

        content.append({"type": "text", "text": text})

        # 计算当前轮次
        turn_index = len(session["messages"]) // 2 + 1

        # 生成回复 (传入 session_id 和 has_media)
        # 确定当前媒体类型
        current_media_type = None
        if image is not None:
            current_media_type = "image"
        elif audio is not None:
            current_media_type = "audio"

        # 生成回复 (传入 session_id, has_media, media_type)
        result = generate_response(
            session["messages"],
            content,
            session_id=session_id,
            has_media=has_media,
            media_type=current_media_type
        )

        if "error" not in result:
            # 如果有媒体输入，从模型回复中提取理解并存储到 thought signature
            if has_media:
                # 模型的回复就是对媒体的理解
                # 将其存储为 "压缩记忆"
                understanding = result["response"]

                # 生成媒体理解签名
                media_ref = generate_media_signature(
                    session_id=session_id,
                    turn_index=turn_index,
                    understanding=understanding[:500]  # 限制长度，只保存摘要
                )
                print(f"[Thought Signature] 存储 {current_media_type} 理解: {media_ref}")

            # 保存到历史（只保存文本摘要）
            user_summary = display_text
            if image is not None:
                user_summary = "[图片] " + user_summary
            if audio is not None:
                user_summary = "[音频] " + user_summary

            session["messages"].append({
                "role": "user",
                "text": user_summary,
                "has_image": image is not None,
                "has_audio": audio is not None,
                "timestamp": time.time()
            })
            session["messages"].append({
                "role": "assistant",
                "text": result["response"],
                "timestamp": time.time()
            })

            # 限制历史长度
            if len(session["messages"]) > MAX_HISTORY_TURNS * 2:
                session["messages"] = session["messages"][-MAX_HISTORY_TURNS * 2:]

            # 更新标题 (用第一条用户消息)
            if session.get("title") == "新对话" and len(session["messages"]) >= 1:
                first_user_msg = next((m for m in session["messages"] if m["role"] == "user"), None)
                if first_user_msg:
                    session["title"] = first_user_msg["text"][:30] + ("..." if len(first_user_msg["text"]) > 30 else "")

            # 保存到磁盘
            save_session_to_disk(session_id, session)
            # 追加到全局历史
            append_to_history(session_id, display_text)

        result["session_id"] = session_id
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_storage()

    # macOS: 请求 sudo 权限用于硬件监控
    if platform.system() == "Darwin":
        request_sudo_permission()

    load_model()
    print("\n" + "=" * 60)
    print("AI 多模态聊天服务器启动: http://localhost:5000")
    print(f"会话存储: {GEMMA3N_HOME}")
    print("支持多轮对话历史记忆")
    if sudo_authorized:
        print("GPU 温度监控: ✅ 已启用")
    else:
        print("GPU 温度监控: ❌ 未启用 (可重启服务并授权)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

"""
LingKong AI WebUI - 轻量服务器版本
用于公网服务器部署，通过 SSH 隧道连接本地推理

功能:
- 静态页面服务
- API 代理 (转发到本地推理服务)
- 会话管理
"""
import os
import json
import uuid
import time
import requests
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# 配置
INFERENCE_URL = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8081")
MMPROJ_URL = os.environ.get("MMPROJ_URL", "http://127.0.0.1:5001")
SESSION_DIR = Path(os.environ.get("SESSION_DIR", "/var/lib/lingkong-webui/sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# 会话缓存
sessions = {}

def get_session(session_id):
    """获取或创建会话"""
    if session_id not in sessions:
        session_file = SESSION_DIR / f"{session_id}.json"
        if session_file.exists():
            with open(session_file) as f:
                sessions[session_id] = json.load(f)
        else:
            sessions[session_id] = {
                "id": session_id,
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
    return sessions[session_id]

def save_session(session_id):
    """保存会话到文件"""
    if session_id in sessions:
        session_file = SESSION_DIR / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(sessions[session_id], f, ensure_ascii=False, indent=2)

# ========== 静态文件 ==========
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "home.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# ========== API 端点 ==========
@app.route("/api/status", methods=["GET"])
def api_status():
    """获取服务状态"""
    # 检查本地推理服务
    inference_ok = False
    try:
        resp = requests.get(f"{INFERENCE_URL}/health", timeout=2)
        inference_ok = resp.status_code == 200
    except:
        pass

    return jsonify({
        "status": "ok",
        "server": "lingkong-webui-lite",
        "inference_connected": inference_ok,
        "inference_url": INFERENCE_URL,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/session/new", methods=["POST"])
def session_new():
    """创建新会话"""
    session_id = str(uuid.uuid4())
    session = get_session(session_id)
    save_session(session_id)
    return jsonify({
        "session_id": session_id,
        "created_at": session["created_at"]
    })

@app.route("/api/session/<session_id>", methods=["GET"])
def session_get(session_id):
    """获取会话信息"""
    session = get_session(session_id)
    return jsonify(session)

@app.route("/api/session/list", methods=["GET"])
def session_list():
    """列出所有会话"""
    session_files = list(SESSION_DIR.glob("*.json"))
    session_list = []
    for sf in sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
        try:
            with open(sf) as f:
                data = json.load(f)
                session_list.append({
                    "id": data.get("id", sf.stem),
                    "created_at": data.get("created_at", ""),
                    "message_count": len(data.get("messages", []))
                })
        except:
            pass
    return jsonify({"sessions": session_list})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """处理聊天请求 - 代理到本地推理服务"""
    data = request.json or {}
    message = data.get("message", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # 获取会话
    session = get_session(session_id)

    # 添加用户消息
    session["messages"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })

    # 构建对话上下文
    context = ""
    for msg in session["messages"][-10:]:  # 最近10轮
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            context += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        else:
            context += f"<start_of_turn>model\n{content}<end_of_turn>\n"

    context += "<start_of_turn>model\n"

    # 调用本地推理服务
    try:
        resp = requests.post(
            f"{INFERENCE_URL}/completion",
            json={
                "prompt": context,
                "n_predict": 512,
                "temperature": 0.7,
                "stop": ["<end_of_turn>", "<start_of_turn>"]
            },
            timeout=120
        )

        if resp.status_code == 200:
            result = resp.json()
            assistant_message = result.get("content", "").strip()
        else:
            assistant_message = f"[推理服务错误: {resp.status_code}]"
    except requests.exceptions.ConnectionError:
        assistant_message = "[无法连接到推理服务，请确保 SSH 隧道已建立]"
    except Exception as e:
        assistant_message = f"[错误: {str(e)}]"

    # 添加助手回复
    session["messages"].append({
        "role": "assistant",
        "content": assistant_message,
        "timestamp": datetime.now().isoformat()
    })

    # 保存会话
    save_session(session_id)

    return jsonify({
        "response": assistant_message,
        "session_id": session_id
    })

@app.route("/api/completions", methods=["POST"])
def api_completions():
    """OpenAI 兼容的 completions API - 代理到本地"""
    data = request.json or {}

    try:
        resp = requests.post(
            f"{INFERENCE_URL}/completion",
            json=data,
            timeout=120
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("=" * 60)
    print("LingKong AI WebUI - Server Lite")
    print("=" * 60)
    print(f"Inference URL: {INFERENCE_URL}")
    print(f"Session Dir: {SESSION_DIR}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)

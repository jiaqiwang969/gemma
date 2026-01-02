"""
AI 眼镜 - Python AI 服务

提供:
1. V-JEPA2 视频编码
2. Gemma 3n 多模态理解
3. 变化分数计算

启动: python server.py --port 8080
"""

import os
import sys
import json
import time
import base64
import argparse
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Flask
from flask import Flask, request, jsonify

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__)

# ============================================================
# 全局模型
# ============================================================

vjepa_encoder = None
gemma_model = None
gemma_processor = None
device = None


def get_device():
    global device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device


# ============================================================
# V-JEPA2 编码器
# ============================================================

def load_vjepa():
    """加载 V-JEPA2 编码器"""
    global vjepa_encoder

    if vjepa_encoder is not None:
        return vjepa_encoder

    print("加载 V-JEPA2 编码器...")

    try:
        from vjepa2.vjepa2_encoder import VJEPA2Encoder
        vjepa_encoder = VJEPA2Encoder(model_size="L", device=get_device())
        print("V-JEPA2 加载成功!")
    except Exception as e:
        print(f"V-JEPA2 加载失败: {e}")
        print("使用 Mock 编码器")
        vjepa_encoder = MockVJEPA2Encoder()

    return vjepa_encoder


class MockVJEPA2Encoder:
    """Mock V-JEPA2 编码器 (用于测试)"""
    def __init__(self):
        self.embed_dim = 1024
        self.num_frames = 16

    def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """生成 mock embedding"""
        T = len(frames)
        embeddings = []

        for i, frame in enumerate(frames):
            # 基于帧内容生成伪 embedding
            mean_val = np.mean(frame) / 255.0
            std_val = np.std(frame) / 255.0

            # 创建基础向量
            base = np.random.RandomState(int(mean_val * 1000)).randn(self.embed_dim)
            # 添加帧特定的变化
            noise = np.random.RandomState(i).randn(self.embed_dim) * 0.1
            emb = base + noise + std_val * 0.5

            # 归一化
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)

        return torch.tensor(np.stack(embeddings), dtype=torch.float32)


# ============================================================
# Gemma 3n 模型
# ============================================================

def load_gemma():
    """加载 Gemma 3n 模型"""
    global gemma_model, gemma_processor

    if gemma_model is not None:
        return gemma_model, gemma_processor

    print("加载 Gemma 3n 模型...")

    try:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        model_name = "google/gemma-3n-E4B-it"
        gemma_processor = AutoProcessor.from_pretrained(model_name)
        gemma_model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if get_device() != "cpu" else torch.float32,
            device_map=get_device()
        )
        print("Gemma 3n 加载成功!")
    except Exception as e:
        print(f"Gemma 3n 加载失败: {e}")
        print("使用 Mock 模型")
        gemma_model = MockGemmaModel()
        gemma_processor = MockGemmaProcessor()

    return gemma_model, gemma_processor


class MockGemmaModel:
    """Mock Gemma 模型"""
    def generate(self, **kwargs):
        # 返回 mock token ids
        return torch.tensor([[1, 2, 3, 4, 5]])


class MockGemmaProcessor:
    """Mock Gemma 处理器"""
    def apply_chat_template(self, messages, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    class tokenizer:
        @staticmethod
        def decode(ids, **kwargs):
            return "这是一个测试响应。视频显示了一个场景。"


# ============================================================
# API 端点
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": get_device()})


@app.route("/encode", methods=["POST"])
def encode():
    """
    V-JEPA2 编码

    Request:
        {
            "frames": ["base64_image1", "base64_image2", ...],
            "timestamps": [0.0, 0.5, ...]
        }

    Response:
        {
            "embeddings": [[...], [...], ...],
            "dim": 1024,
            "change_scores": [0.0, 0.05, ...]
        }
    """
    start_time = time.time()

    data = request.json
    frames_b64 = data.get("frames", [])
    timestamps = data.get("timestamps", [])

    if not frames_b64:
        return jsonify({"error": "No frames provided"}), 400

    # 解码图像
    frames = []
    for b64 in frames_b64:
        try:
            img_data = base64.b64decode(b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            frames.append(np.array(img))
        except Exception as e:
            return jsonify({"error": f"Failed to decode image: {e}"}), 400

    # 加载编码器并编码
    encoder = load_vjepa()
    embeddings = encoder.encode_frames(frames)

    # 计算变化分数
    change_scores = [0.0]
    for i in range(1, len(embeddings)):
        sim = F.cosine_similarity(
            embeddings[i-1].unsqueeze(0),
            embeddings[i].unsqueeze(0)
        ).item()
        change_scores.append(max(0.0, 1.0 - sim))

    processing_time = int((time.time() - start_time) * 1000)

    return jsonify({
        "embeddings": embeddings.tolist(),
        "dim": embeddings.shape[1],
        "change_scores": change_scores,
        "processing_time_ms": processing_time
    })


@app.route("/understand", methods=["POST"])
def understand():
    """
    Gemma 3n 多模态理解

    Request:
        {
            "keyframes": ["base64_image1", ...],
            "keyframe_times": [0.0, 2.5, ...],
            "prompt": "描述这个视频",
            "context": "可选的上下文信息"
        }

    Response:
        {
            "response": "视频显示...",
            "processing_time_ms": 1234
        }
    """
    start_time = time.time()

    data = request.json
    keyframes_b64 = data.get("keyframes", [])
    keyframe_times = data.get("keyframe_times", [])
    prompt = data.get("prompt", "描述这个视频")
    context = data.get("context", "")

    if not keyframes_b64:
        return jsonify({"error": "No keyframes provided"}), 400

    # 解码图像
    images = []
    for b64 in keyframes_b64:
        try:
            img_data = base64.b64decode(b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            images.append(img)
        except Exception as e:
            return jsonify({"error": f"Failed to decode image: {e}"}), 400

    # 构建消息
    content = []
    for i, (img, t) in enumerate(zip(images, keyframe_times)):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[{t:.1f}s]"})

    # 添加上下文和提示
    full_prompt = ""
    if context:
        full_prompt += f"{context}\n\n"
    full_prompt += f"用户问题: {prompt}"
    content.append({"type": "text", "text": full_prompt})

    messages = [{"role": "user", "content": content}]

    # 加载模型并推理
    model, processor = load_gemma()

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 移动到设备
        input_ids = inputs["input_ids"].to(get_device())

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": False,
        }

        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            generate_kwargs["pixel_values"] = inputs["pixel_values"].to(get_device())

        with torch.inference_mode():
            outputs = model.generate(**generate_kwargs)

        response = processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

    except Exception as e:
        # 如果是 Mock 模型，返回 mock 响应
        response = f"[Mock 响应] 视频显示了{len(images)}个关键帧。"
        if "什么时候" in prompt or "when" in prompt.lower():
            response += f" 可能在 {keyframe_times[len(keyframe_times)//2]:.1f}s 处有相关内容。"
        elif "变化" in prompt or "change" in prompt.lower():
            response += " 视频开始和结束之间有一些变化。"

    processing_time = int((time.time() - start_time) * 1000)

    return jsonify({
        "response": response,
        "processing_time_ms": processing_time
    })


@app.route("/analyze_intent", methods=["POST"])
def analyze_intent():
    """
    分析 embedding 以增强意图理解

    Request:
        {
            "embeddings": [[...], [...], ...],
            "change_scores": [0.0, 0.05, ...],
            "current_intent": "describe"
        }

    Response:
        {
            "suggested_intents": ["compare", "locate"],
            "activity_level": "medium",
            "has_significant_change": true,
            "change_peaks": [2.5, 5.0]
        }
    """
    data = request.json
    embeddings = data.get("embeddings", [])
    change_scores = data.get("change_scores", [])
    current_intent = data.get("current_intent", "describe")

    if not embeddings or not change_scores:
        return jsonify({"error": "No data provided"}), 400

    # 分析
    max_change = max(change_scores) if change_scores else 0.0

    # 活动级别
    if max_change < 0.03:
        activity_level = "static"
    elif max_change < 0.08:
        activity_level = "low"
    elif max_change < 0.2:
        activity_level = "medium"
    else:
        activity_level = "high"

    # 检测变化峰值
    threshold = np.mean(change_scores) + np.std(change_scores) if len(change_scores) > 1 else 0.05
    change_peaks = [
        i * 0.5  # 假设每帧 0.5s
        for i, score in enumerate(change_scores)
        if score > threshold
    ]

    # 建议的意图
    suggested_intents = []
    if max_change > 0.15 and current_intent != "compare":
        suggested_intents.append("compare")
    if len(change_peaks) >= 2 and current_intent != "locate":
        suggested_intents.append("locate")

    return jsonify({
        "suggested_intents": suggested_intents,
        "activity_level": activity_level,
        "has_significant_change": max_change > 0.1,
        "change_peaks": change_peaks[:5],
        "max_change": max_change
    })


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Glasses Python Service")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--preload", action="store_true", help="Preload models")
    args = parser.parse_args()

    if args.preload:
        print("预加载模型...")
        load_vjepa()
        load_gemma()

    print(f"\n启动 AI 服务: http://{args.host}:{args.port}")
    print("端点:")
    print("  GET  /health         - 健康检查")
    print("  POST /encode         - V-JEPA2 编码")
    print("  POST /understand     - Gemma 3n 理解")
    print("  POST /analyze_intent - 意图增强分析")
    print()

    app.run(host=args.host, port=args.port, debug=False)

#!/usr/bin/env python3
"""
测试 Cambrian-S 输出惊讶曲线

方法：通过 prompt 让模型对每一帧进行"惊讶程度"评分
"""
import os
import sys
import time
import torch
import json
from pathlib import Path
from PIL import Image
import cv2

# 添加 cambrian-s 到路径
cambrian_path = Path(__file__).parent.parent.parent / "cambrian-s"
sys.path.insert(0, str(cambrian_path))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def extract_frames_from_video(video_path: str, fps: float = 1.0):
    """从视频提取帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / fps)

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            timestamps.append(frame_idx / video_fps)

        frame_idx += 1

    cap.release()
    return frames, timestamps


def load_cambrian_model(model_path: str, device: str = "mps"):
    """加载 Cambrian-S 模型"""
    from cambrian.model.builder import load_pretrained_model

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="cambrian-s",
        device_map="auto",
        device=device,
    )
    return tokenizer, model, image_processor, context_len


def analyze_surprise(model, tokenizer, image_processor, frames, timestamps, device):
    """分析每一帧的惊讶程度"""
    from cambrian.conversation import conv_templates
    from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from cambrian.mm_utils import process_images, tokenizer_image_token

    # 构建 prompt - 要求模型输出 JSON 格式的惊讶分数
    prompt = """你正在观看一个视频的连续帧序列。请分析这些帧，识别场景中的变化和"惊讶"时刻。

对于每一帧，请给出一个 0-100 的"惊讶分数"：
- 0-20: 与上一帧几乎相同，没有变化
- 21-40: 轻微变化（如轻微移动）
- 41-60: 中等变化（如物体移动、场景变化）
- 61-80: 显著变化（如新物体出现、场景切换）
- 81-100: 剧烈变化（如突然事件、完全不同的场景）

第一帧的惊讶分数设为 50（作为基准）。

请用以下 JSON 格式输出每帧的分析：
```json
{
  "frames": [
    {"time": "0.0s", "surprise_score": 50, "description": "基准帧，..."},
    {"time": "2.0s", "surprise_score": XX, "description": "..."},
    ...
  ],
  "peak_surprise": {"time": "X.Xs", "score": XX, "reason": "..."}
}
```

请严格按照 JSON 格式输出。"""

    print(f"分析 {len(frames)} 帧的惊讶曲线...")

    # 处理图像
    model_cfg = model.config
    image_tensors, image_sizes = process_images(frames, image_processor, model_cfg)

    # 构建对话
    conv = conv_templates["qwen_2"].copy()
    image_tokens = DEFAULT_IMAGE_TOKEN * len(frames)
    full_prompt = f"{image_tokens}\n{prompt}"

    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    image_tensors = [t.to(device, dtype=torch.float16) for t in image_tensors]

    # 生成
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )
    t_generate = time.time() - t0

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"生成时间: {t_generate:.2f}s")

    return response


def parse_surprise_scores(response: str):
    """解析模型输出的惊讶分数"""
    import re

    # 尝试提取 JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data
        except json.JSONDecodeError:
            pass

    # 如果没有 JSON，尝试提取数字
    scores = []
    lines = response.split('\n')
    for line in lines:
        # 查找类似 "帧X: 惊讶分数 XX" 的模式
        match = re.search(r'(\d+\.?\d*)\s*[秒s].*?(\d+)', line)
        if match:
            time_val = float(match.group(1))
            score = int(match.group(2))
            if 0 <= score <= 100:
                scores.append({"time": f"{time_val}s", "surprise_score": score})

    return {"frames": scores} if scores else None


def plot_surprise_curve(data, timestamps):
    """绘制惊讶曲线（ASCII 形式）"""
    if not data or "frames" not in data:
        print("无法解析惊讶分数")
        return

    frames_data = data["frames"]

    print("\n" + "=" * 60)
    print("惊讶曲线 (Surprise Curve)")
    print("=" * 60)

    max_width = 50

    for frame in frames_data:
        time_str = frame.get("time", "?")
        score = frame.get("surprise_score", 0)
        desc = frame.get("description", "")[:30]

        bar_len = int(score * max_width / 100)
        bar = "█" * bar_len + "░" * (max_width - bar_len)

        print(f"{time_str:>6} | {bar} | {score:3d} | {desc}")

    # 显示峰值
    if "peak_surprise" in data:
        peak = data["peak_surprise"]
        print("\n" + "-" * 60)
        print(f"峰值惊讶: {peak.get('time', '?')} - 分数 {peak.get('score', '?')}")
        print(f"原因: {peak.get('reason', '?')}")


def main():
    print("=" * 70)
    print("  Cambrian-S 惊讶曲线测试")
    print("=" * 70)

    # 检测设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # 视频路径
    video_path = Path(__file__).parent.parent.parent / "data" / "test-video" / "识别笔记本电脑台数.mp4"

    if not video_path.exists():
        print(f"视频不存在: {video_path}")
        return

    # 提取帧
    print("\n提取视频帧 (1 fps)...")
    frames, timestamps = extract_frames_from_video(str(video_path), fps=1.0)
    print(f"提取了 {len(frames)} 帧")

    # 限制帧数
    max_frames = 8
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
        timestamps = timestamps[::step][:max_frames]
        print(f"使用 {len(frames)} 帧进行分析")

    # 加载模型
    print("\n加载 Cambrian-S 3B...")
    model_id = "nyu-visionx/Cambrian-S-3B"
    tokenizer, model, image_processor, _ = load_cambrian_model(model_id, device)
    print("模型加载完成!")

    # 分析惊讶曲线
    print("\n" + "=" * 70)
    print("分析惊讶曲线")
    print("=" * 70)

    response = analyze_surprise(model, tokenizer, image_processor, frames, timestamps, device)

    print("\n模型原始输出:")
    print("-" * 60)
    print(response)
    print("-" * 60)

    # 解析并绘制
    data = parse_surprise_scores(response)
    plot_surprise_curve(data, timestamps)


if __name__ == "__main__":
    main()

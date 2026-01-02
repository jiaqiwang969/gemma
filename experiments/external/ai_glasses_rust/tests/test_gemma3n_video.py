#!/usr/bin/env python3
"""
Gemma 3n 视频测试 - 与 Cambrian-S 对比

使用相同的输入（5帧视频帧）和相同的 prompt 进行测试
"""
import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image
import cv2

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def extract_frames_from_video(video_path: str, fps: float = 1.0):
    """从视频提取帧，按指定 fps"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"视频信息:")
    print(f"  路径: {video_path}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f}s")
    print(f"  采样率: {fps} fps")

    # 计算采样间隔
    frame_interval = int(video_fps / fps)

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            timestamps.append(frame_idx / video_fps)

        frame_idx += 1

    cap.release()

    print(f"  提取帧数: {len(frames)}")
    for i, t in enumerate(timestamps):
        print(f"    帧 {i+1}: {t:.1f}s")

    return frames, timestamps


def main():
    print("=" * 70)
    print("  Gemma 3n 视频理解测试 (与 Cambrian-S 对比)")
    print("=" * 70)

    # 1. 检测设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # 2. 视频路径
    video_path = Path(__file__).parent.parent.parent / "data" / "test-video" / "识别笔记本电脑台数.mp4"

    if not video_path.exists():
        print(f"视频不存在: {video_path}")
        return

    # 3. 提取帧 (1 fps)
    print("\n" + "=" * 70)
    print("提取视频帧")
    print("=" * 70)
    frames, timestamps = extract_frames_from_video(str(video_path), fps=1.0)

    if not frames:
        print("没有提取到帧")
        return

    # 限制帧数（与 Cambrian-S 测试相同：5帧）
    max_frames = 5
    if len(frames) > max_frames:
        print(f"\n限制帧数: {len(frames)} -> {max_frames}")
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
        timestamps = timestamps[::step][:max_frames]
        print(f"使用帧:")
        for i, t in enumerate(timestamps):
            print(f"  帧 {i+1}: {t:.1f}s")

    # 4. 加载 Gemma 3n 模型
    print("\n" + "=" * 70)
    print("加载 Gemma 3n 模型")
    print("=" * 70)

    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = "google/gemma-3n-E4B-it"  # 使用 4B 版本与 Cambrian-S 3B 对比

    print(f"加载模型: {model_name}")
    print(f"设备: {device}")

    t_load_start = time.time()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    t_load = time.time() - t_load_start
    print(f"模型加载完成! 耗时: {t_load:.2f}s")

    # 5. 构建输入 - 使用与 Cambrian-S 完全相同的 prompt
    print("\n" + "=" * 70)
    print("视频理解推理")
    print("=" * 70)

    # 改进的 prompt - 强调跨帧去重
    query = """这5张图片是从同一个视频中按时间顺序提取的帧（0秒、2秒、4秒、6秒、8秒）。
视频中可能出现多台笔记本电脑，同一台电脑可能在多个帧中重复出现。

请综合分析所有帧，识别视频中一共出现了几台【不同的】笔记本电脑？
注意：如果同一台电脑在多帧中出现，只计算一次。请先描述你在每帧中看到的内容，然后给出去重后的总数。"""

    print(f"运行推理...")
    print(f"  图像数量: {len(frames)}")
    print(f"  查询: {query}")

    # 构建多图像输入
    content = []
    for i, img in enumerate(frames):
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": query})

    messages = [{"role": "user", "content": content}]

    # 预处理
    t_preprocess_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    t_preprocess = time.time() - t_preprocess_start

    print(f"  输入 token 数: {inputs['input_ids'].shape[1]}")
    print(f"  预处理时间: {t_preprocess:.2f}s")

    # 6. 生成
    model.eval()
    t_generate_start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
    t_generate = time.time() - t_generate_start

    # 计算生成的 token 数
    input_len = inputs["input_ids"].shape[1]
    num_new_tokens = outputs.shape[1] - input_len
    tokens_per_sec = num_new_tokens / t_generate

    print(f"\n  === 生成速度统计 ===")
    print(f"  生成 token 数: {num_new_tokens}")
    print(f"  生成时间: {t_generate:.2f}s")
    print(f"  速度: {tokens_per_sec:.2f} tokens/s")

    # 解码
    generated_ids = outputs[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 70)
    print("Gemma 3n 回答:")
    print("=" * 70)
    print(response)
    print("=" * 70)


if __name__ == "__main__":
    main()

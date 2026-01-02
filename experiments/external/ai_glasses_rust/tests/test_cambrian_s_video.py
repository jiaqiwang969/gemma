#!/usr/bin/env python3
"""
Cambrian-S 视频测试 - 1fps 采样

从原始视频提取帧，使用 Cambrian-S 进行视频理解
"""
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import cv2

# 添加 cambrian-s 到路径
cambrian_path = Path(__file__).parent.parent.parent / "cambrian-s"
sys.path.insert(0, str(cambrian_path))

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


def load_cambrian_model(model_path: str, device: str = "mps"):
    """加载 Cambrian-S 模型"""
    from cambrian.model.builder import load_pretrained_model

    print(f"\n加载模型: {model_path}")
    print(f"设备: {device}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="cambrian-s",
        device_map="auto",
        device=device,
    )

    return tokenizer, model, image_processor, context_len


def process_images_official(images, image_processor, model):
    """使用官方 process_images 函数"""
    from cambrian.mm_utils import process_images as cambrian_process_images
    model_cfg = model.config
    return cambrian_process_images(images, image_processor, model_cfg)


def run_inference(model, tokenizer, image_processor, images, query, device):
    """运行推理"""
    import time
    from cambrian.conversation import conv_templates
    from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from cambrian.mm_utils import tokenizer_image_token

    print(f"\n运行推理...")
    print(f"  图像数量: {len(images)}")
    print(f"  查询: {query}")

    # 处理图像 - 使用官方函数
    t0 = time.time()
    image_tensors, image_sizes = process_images_official(images, image_processor, model)
    t_preprocess = time.time() - t0
    print(f"  图像张量形状: {image_tensors[0].shape}")
    print(f"  图像预处理时间: {t_preprocess:.2f}s")

    # 构建对话 - Cambrian-S 使用 qwen_2 模板
    conv = conv_templates["qwen_2"].copy()

    # 添加图像 token - 每张图片一个 <image> token
    image_tokens = DEFAULT_IMAGE_TOKEN * len(images)
    prompt = f"{image_tokens}\n{query}"

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    full_prompt = conv.get_prompt()
    print(f"  完整 prompt: {full_prompt[:100]}...")

    # 使用 tokenizer_image_token 来正确处理 image token
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    print(f"  输入 token 数: {input_ids.shape[1]}")

    # 移动图像到设备
    image_tensors = [t.to(device, dtype=torch.float16) for t in image_tensors]

    # 生成并计时
    t1 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
    t_generate = time.time() - t1

    # 计算生成的 token 数
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = num_new_tokens / t_generate

    print(f"\n  === 生成速度统计 ===")
    print(f"  生成 token 数: {num_new_tokens}")
    print(f"  生成时间: {t_generate:.2f}s")
    print(f"  速度: {tokens_per_sec:.2f} tokens/s")

    # 解码 - 完整输出
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 直接返回完整响应（去掉系统和用户的部分）
    return full_response


def main():
    print("=" * 70)
    print("  Cambrian-S 视频理解测试 (1 fps)")
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

    # 限制帧数，避免模型处理过多图片
    max_frames = 5
    if len(frames) > max_frames:
        print(f"\n限制帧数: {len(frames)} -> {max_frames}")
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
        timestamps = timestamps[::step][:max_frames]
        print(f"使用帧:")
        for i, t in enumerate(timestamps):
            print(f"  帧 {i+1}: {t:.1f}s")

    # 4. 加载模型 - 使用 3B 版本
    print("\n" + "=" * 70)
    print("加载 Cambrian-S 模型")
    print("=" * 70)
    model_id = "nyu-visionx/Cambrian-S-3B"

    try:
        tokenizer, model, image_processor, context_len = load_cambrian_model(
            model_path=model_id,
            device=device
        )
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 运行推理
    print("\n" + "=" * 70)
    print("视频理解推理")
    print("=" * 70)

    # 改进的 prompt - 强调跨帧去重
    query = """这5张图片是从同一个视频中按时间顺序提取的帧（0秒、2秒、4秒、6秒、8秒）。
视频中可能出现多台笔记本电脑，同一台电脑可能在多个帧中重复出现。

请综合分析所有帧，识别视频中一共出现了几台【不同的】笔记本电脑？
注意：如果同一台电脑在多帧中出现，只计算一次。请先描述你在每帧中看到的内容，然后给出去重后的总数。"""

    try:
        response = run_inference(model, tokenizer, image_processor, frames, query, device)

        print("\n" + "=" * 70)
        print("Cambrian-S 回答:")
        print("=" * 70)
        print(response)
        print("=" * 70)

    except Exception as e:
        print(f"推理失败: {repr(e)}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to see full error


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
V-JEPA2 惊讶曲线测试

通过计算帧间预测误差来生成惊讶曲线
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 添加 vjepa2 到路径
vjepa_path = Path(__file__).parent.parent.parent.parent / "vjepa2"
sys.path.insert(0, str(vjepa_path))


def extract_frames_from_video(video_path: str, fps: float = 1.0):
    """从视频提取帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
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


def load_vjepa2_model(device="mps"):
    """加载 V-JEPA2 模型"""
    from vjepa2.model import VJEPA2Model

    print("加载 V-JEPA2 模型...")
    model = VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    model = model.to(device)
    model.eval()
    print("模型加载完成!")
    return model


def compute_frame_embeddings(model, frames, device="mps"):
    """计算每帧的 embedding"""
    from torchvision import transforms

    # V-JEPA2 的预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            # 预处理
            img_tensor = transform(frame).unsqueeze(0).to(device)  # [1, 3, 256, 256]

            # 添加时间维度: [B, C, T, H, W]
            img_tensor = img_tensor.unsqueeze(2)  # [1, 3, 1, 256, 256]

            # 获取 embedding
            output = model.encode_video(img_tensor)

            # 取平均 embedding
            if isinstance(output, dict):
                emb = output.get('video_embedding', output.get('cls_token', None))
                if emb is None:
                    emb = list(output.values())[0]
            else:
                emb = output

            # 展平
            emb = emb.mean(dim=1) if len(emb.shape) > 2 else emb
            emb = emb.flatten()

            embeddings.append(emb.cpu().numpy())
            print(f"  帧 {i+1}: embedding shape = {emb.shape}")

    return np.array(embeddings)


def compute_surprise_from_embeddings(embeddings):
    """从 embedding 序列计算惊讶分数"""
    n_frames = len(embeddings)
    surprise_scores = [50.0]  # 第一帧作为基准

    for i in range(1, n_frames):
        # 计算与前一帧的余弦距离
        prev_emb = embeddings[i-1]
        curr_emb = embeddings[i]

        # 余弦相似度
        cos_sim = np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb) + 1e-8)

        # 转换为惊讶分数 (0-100)
        # 相似度 1.0 -> 惊讶 0, 相似度 0.0 -> 惊讶 100
        surprise = (1 - cos_sim) * 100
        surprise = max(0, min(100, surprise))  # 限制范围

        surprise_scores.append(surprise)

    return surprise_scores


def compute_surprise_simple(frames, device="mps"):
    """使用简单的像素差异计算惊讶分数（不需要 V-JEPA2）"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tensors = [transform(f) for f in frames]

    surprise_scores = [50.0]  # 基准帧

    for i in range(1, len(tensors)):
        prev = tensors[i-1]
        curr = tensors[i]

        # 计算像素差异
        diff = torch.abs(curr - prev).mean().item()

        # 转换为 0-100 分数
        surprise = min(100, diff * 500)  # 缩放因子
        surprise_scores.append(surprise)

    return surprise_scores


def compute_surprise_with_clip(frames, device="mps"):
    """使用 CLIP 视觉编码器计算惊讶分数"""
    try:
        import open_clip

        print("使用 CLIP 视觉编码器...")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.to(device)
        model.eval()

        embeddings = []
        with torch.no_grad():
            for frame in frames:
                img = preprocess(frame).unsqueeze(0).to(device)
                emb = model.encode_image(img)
                embeddings.append(emb.cpu().numpy().flatten())

        embeddings = np.array(embeddings)
        return compute_surprise_from_embeddings(embeddings)

    except Exception as e:
        print(f"CLIP 加载失败: {e}")
        return None


def compute_surprise_with_siglip(frames, device="mps"):
    """使用 SigLIP 视觉编码器计算惊讶分数（与 Cambrian-S 相同的视觉编码器）"""
    try:
        from transformers import AutoProcessor, AutoModel

        print("使用 SigLIP 视觉编码器（与 Cambrian-S 相同）...")
        model_name = "google/siglip-base-patch16-224"

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        embeddings = []
        with torch.no_grad():
            for frame in frames:
                inputs = processor(images=frame, return_tensors="pt").to(device)
                outputs = model.get_image_features(**inputs)
                embeddings.append(outputs.cpu().numpy().flatten())

        embeddings = np.array(embeddings)
        return compute_surprise_from_embeddings(embeddings)

    except Exception as e:
        print(f"SigLIP 加载失败: {e}")
        return None


def plot_surprise_curve(timestamps, scores, title="惊讶曲线"):
    """绘制惊讶曲线（ASCII）"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print("=" * 60)

    max_width = 50

    for i, (t, score) in enumerate(zip(timestamps, scores)):
        bar_len = int(score * max_width / 100)
        bar = "█" * bar_len + "░" * (max_width - bar_len)
        print(f"{t:>5.1f}s | {bar} | {score:5.1f}")

    # 找峰值
    peak_idx = np.argmax(scores[1:]) + 1 if len(scores) > 1 else 0
    print("-" * 60)
    print(f"峰值惊讶: {timestamps[peak_idx]:.1f}s - 分数 {scores[peak_idx]:.1f}")


def main():
    print("=" * 70)
    print("  V-JEPA2 / 视觉编码器 惊讶曲线测试")
    print("=" * 70)

    # 设备
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

    # Cambrian-S 的参考结果
    cambrian_scores = [80, 32, 50, 32, 62, 32, 32, 32][:len(frames)]
    plot_surprise_curve(timestamps, cambrian_scores, "Cambrian-S 惊讶曲线 (参考)")

    # 方法 1: 像素差异
    print("\n" + "-" * 60)
    print("方法 1: 像素差异")
    pixel_scores = compute_surprise_simple(frames, device)
    plot_surprise_curve(timestamps, pixel_scores, "像素差异惊讶曲线")

    # 方法 2: CLIP
    print("\n" + "-" * 60)
    print("方法 2: CLIP 视觉编码器")
    clip_scores = compute_surprise_with_clip(frames, device)
    if clip_scores:
        plot_surprise_curve(timestamps, clip_scores, "CLIP 惊讶曲线")

    # 方法 3: SigLIP（与 Cambrian-S 相同的视觉编码器）
    print("\n" + "-" * 60)
    print("方法 3: SigLIP 视觉编码器")
    siglip_scores = compute_surprise_with_siglip(frames, device)
    if siglip_scores:
        plot_surprise_curve(timestamps, siglip_scores, "SigLIP 惊讶曲线")

    # 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(f"{'时间':>6} | {'Cambrian-S':>12} | {'像素差异':>12} | {'CLIP':>12} | {'SigLIP':>12}")
    print("-" * 70)
    for i, t in enumerate(timestamps):
        c = cambrian_scores[i] if i < len(cambrian_scores) else 0
        p = pixel_scores[i] if pixel_scores and i < len(pixel_scores) else 0
        cl = clip_scores[i] if clip_scores and i < len(clip_scores) else 0
        s = siglip_scores[i] if siglip_scores and i < len(siglip_scores) else 0
        print(f"{t:>5.1f}s | {c:>12.1f} | {p:>12.1f} | {cl:>12.1f} | {s:>12.1f}")


if __name__ == "__main__":
    main()

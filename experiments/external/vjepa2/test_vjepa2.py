#!/usr/bin/env python3
"""
V-JEPA2 + Gemma 3n 视频理解测试脚本

测试内容:
1. V-JEPA2 编码器初始化
2. 语义变化检测
3. 关键帧提取
4. (可选) Gemma 3n 多模态理解
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_encoder():
    """测试 V-JEPA2 编码器"""
    print("\n" + "=" * 60)
    print("测试 1: V-JEPA2 编码器")
    print("=" * 60)

    from vjepa2 import VJEPA2Encoder

    encoder = VJEPA2Encoder(model_size="L", img_size=256, num_frames=16)

    # 创建随机测试帧
    print("\n生成随机测试帧...")
    frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)

    print("编码帧...")
    start = time.time()
    embeddings = encoder.encode_frames(frames)
    elapsed = time.time() - start

    print(f"  输入: {frames.shape}")
    print(f"  输出: {embeddings.shape}")
    print(f"  耗时: {elapsed:.3f}s")
    print(f"  Embedding 维度: {encoder.embedding_dim}")

    return encoder


def test_change_detection():
    """测试语义变化检测"""
    print("\n" + "=" * 60)
    print("测试 2: 语义变化检测")
    print("=" * 60)

    from vjepa2 import SemanticChangeDetector
    import torch

    detector = SemanticChangeDetector(
        base_threshold=0.15,
        min_keyframe_interval=2,
        max_keyframe_interval=10
    )

    # 模拟 embedding 序列: 静止 → 变化 → 静止
    print("\n模拟场景: 静止 (10帧) → 变化 (10帧) → 静止 (10帧)")

    embeddings = []
    D = 1024

    # 静止段
    base = torch.randn(D)
    for _ in range(10):
        embeddings.append(base + torch.randn(D) * 0.01)

    # 变化段
    for _ in range(10):
        embeddings.append(torch.randn(D))

    # 新静止段
    base2 = torch.randn(D)
    for _ in range(10):
        embeddings.append(base2 + torch.randn(D) * 0.01)

    embeddings = torch.stack(embeddings)

    # 检测
    frame_infos, keyframe_indices = detector.detect_changes(embeddings, fps=5.0)
    events = detector.detect_events(frame_infos)

    print(f"\n结果:")
    print(f"  总帧数: {len(frame_infos)}")
    print(f"  关键帧数: {len(keyframe_indices)}")
    print(f"  关键帧索引: {keyframe_indices}")
    print(f"  压缩比: {len(keyframe_indices)/len(frame_infos):.1%}")
    print(f"  检测事件数: {len(events)}")

    # 可视化变化分数
    print("\n变化分数分布:")
    print("帧  | 分数  | 状态")
    print("-" * 35)
    for info in frame_infos:
        marker = "★ 关键帧" if info.is_keyframe else ""
        bar = "█" * int(info.change_score * 20)
        print(f"{info.index:2d}  | {info.change_score:.3f} | {bar} {marker}")

    return detector


def test_pipeline(video_path=None, use_gemma=False):
    """测试完整管道"""
    print("\n" + "=" * 60)
    print("测试 3: 完整视频理解管道")
    print("=" * 60)

    from vjepa2 import VideoPipeline

    pipeline = VideoPipeline(
        vjepa_model_size="L",
        target_keyframes=5,
        load_gemma=use_gemma
    )

    if video_path and os.path.exists(video_path):
        # 分析真实视频
        print(f"\n分析视频: {video_path}")
        result = pipeline.analyze_video(
            video_path=video_path,
            sample_fps=5.0,
            prompt="What is happening in this video? Describe the main events.",
            generate_response=use_gemma
        )
    else:
        # 分析模拟视频帧
        print("\n分析模拟视频帧 (60帧)...")

        frames = []
        for i in range(60):
            # 模拟: 静止 → 运动 → 静止
            if i < 20:
                intensity = 100
            elif i < 40:
                intensity = int(100 + (i - 20) * 7)
            else:
                intensity = 240

            frame = np.full((256, 256, 3), intensity, dtype=np.uint8)
            # 添加一些变化
            if 20 <= i < 40:
                frame[50:100, 50:100] = 255 - intensity
            frames.append(frame)

        result = pipeline.analyze_frames(
            frames=frames,
            fps=10.0,
            prompt="Describe what changed in this video.",
            generate_response=use_gemma
        )

    print(f"\n分析结果:")
    print(f"  关键帧索引: {result.keyframe_indices}")
    print(f"  关键帧时间: {[f'{t:.1f}s' for t in result.keyframe_timestamps]}")
    print(f"  变化事件数: {len(result.events)}")
    print(f"  处理耗时: {result.processing_time:.2f}s")

    if result.response:
        print(f"\nGemma 3n 响应:")
        print("-" * 40)
        print(result.response)
        print("-" * 40)

    print(f"\n统计信息: {result.stats}")

    return result


def test_video_file(video_path):
    """测试真实视频文件"""
    print("\n" + "=" * 60)
    print(f"测试视频文件: {video_path}")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return

    from vjepa2 import VideoPipeline

    pipeline = VideoPipeline(
        vjepa_model_size="L",
        target_keyframes=5,
        load_gemma=True  # 加载 Gemma 进行多模态理解
    )

    result = pipeline.analyze_video(
        video_path=video_path,
        sample_fps=5.0,
        prompt="Please analyze this video and describe: 1) What objects/people are visible? 2) What actions or events occur? 3) What changes happen over time?",
        generate_response=True
    )

    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")

    print(f"\n关键帧: {len(result.keyframes)} 张")
    print(f"时间戳: {result.keyframe_timestamps}")

    if result.events:
        print(f"\n检测到 {len(result.events)} 个事件:")
        for event in result.events:
            print(f"  - {event.start_time:.1f}s-{event.end_time:.1f}s: {event.description}")

    if result.response:
        print(f"\n{'='*60}")
        print("Gemma 3n 视频理解:")
        print(f"{'='*60}")
        print(result.response)

    # 尝试可视化
    try:
        output_path = video_path.rsplit('.', 1)[0] + '_analysis.png'
        pipeline.visualize_analysis(result, output_path)
    except Exception as e:
        print(f"\n可视化失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA2 + Gemma 3n 视频理解测试")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--gemma", action="store_true", help="启用 Gemma 3n 多模态理解")
    parser.add_argument("--test", type=str, choices=["encoder", "detector", "pipeline", "all"],
                       default="all", help="测试内容")

    args = parser.parse_args()

    print("=" * 60)
    print("V-JEPA2 + Gemma 3n 视频理解系统测试")
    print("=" * 60)
    print(f"测试项: {args.test}")
    print(f"Gemma 3n: {'启用' if args.gemma else '禁用'}")
    if args.video:
        print(f"视频文件: {args.video}")
    print("=" * 60)

    if args.video:
        test_video_file(args.video)
    else:
        if args.test in ["encoder", "all"]:
            test_encoder()

        if args.test in ["detector", "all"]:
            test_change_detection()

        if args.test in ["pipeline", "all"]:
            test_pipeline(use_gemma=args.gemma)

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

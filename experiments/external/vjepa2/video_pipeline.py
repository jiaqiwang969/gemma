"""
V-JEPA2 + Gemma 3n 视频理解 Pipeline

完整流程:
1. 视频输入 (文件/摄像头/帧序列)
2. V-JEPA2 编码 → embedding
3. 语义变化检测 → 关键帧选择
4. Gemma 3n 多模态理解 (关键帧 + 音频 + 文本)
5. 输出响应

特点:
- 语义驱动抽帧，而非固定帧率
- 支持实时视频流
- 支持长视频分析
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa2.vjepa2_encoder import VJEPA2Encoder
from vjepa2.change_detector import SemanticChangeDetector, KeyframeExtractor, FrameInfo


@dataclass
class VideoAnalysisResult:
    """视频分析结果"""
    keyframes: List[np.ndarray]           # 关键帧图像
    keyframe_indices: List[int]           # 关键帧索引
    keyframe_timestamps: List[float]      # 关键帧时间戳
    frame_infos: List[FrameInfo]          # 所有帧信息
    events: List                          # 变化事件
    stats: Dict                           # 统计信息
    response: Optional[str] = None        # Gemma 3n 的响应
    processing_time: float = 0.0          # 处理耗时


class VideoPipeline:
    """
    V-JEPA2 + Gemma 3n 视频理解管道

    核心思想: "像人一样观察视频，只在需要时认真看"
    """

    def __init__(
        self,
        # V-JEPA2 参数
        vjepa_model_size: str = "L",
        vjepa_img_size: int = 256,
        vjepa_num_frames: int = 16,
        # 关键帧参数
        target_keyframes: int = 5,
        change_threshold: float = 0.15,
        min_keyframe_interval: int = 3,
        max_keyframe_interval: int = 30,
        # Gemma 3n 参数 (延迟加载)
        gemma_model_name: str = "google/gemma-3n-E2B-it",
        load_gemma: bool = False,  # 是否立即加载 Gemma
        # 设备
        device: Optional[str] = None,
    ):
        self.target_keyframes = target_keyframes
        self.gemma_model_name = gemma_model_name
        self.device = device

        print("=" * 60)
        print("初始化 V-JEPA2 + Gemma 3n 视频理解管道")
        print("=" * 60)

        # 初始化 V-JEPA2 编码器
        print("\n[1/3] 初始化 V-JEPA2 编码器...")
        self.encoder = VJEPA2Encoder(
            model_size=vjepa_model_size,
            img_size=vjepa_img_size,
            num_frames=vjepa_num_frames,
            device=device
        )

        # 初始化变化检测器
        print("\n[2/3] 初始化语义变化检测器...")
        self.detector = SemanticChangeDetector(
            base_threshold=change_threshold,
            min_keyframe_interval=min_keyframe_interval,
            max_keyframe_interval=max_keyframe_interval
        )

        # 初始化关键帧提取器
        self.keyframe_extractor = KeyframeExtractor(
            encoder=self.encoder,
            detector=self.detector,
            target_keyframes=target_keyframes
        )

        # Gemma 3n (延迟加载)
        self.gemma_model = None
        self.gemma_processor = None
        self.gemma_loaded = False

        if load_gemma:
            self._load_gemma()

        print("\n" + "=" * 60)
        print("管道初始化完成!")
        print("=" * 60)

    def _load_gemma(self):
        """加载 Gemma 3n 模型"""
        if self.gemma_loaded:
            return

        print("\n[3/3] 加载 Gemma 3n 多模态模型...")

        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration

            self.gemma_processor = AutoProcessor.from_pretrained(
                self.gemma_model_name,
                trust_remote_code=True,
                local_files_only=True
            )

            self.gemma_model = Gemma3nForConditionalGeneration.from_pretrained(
                self.gemma_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True
            )
            self.gemma_model.eval()
            self.gemma_loaded = True
            print("    Gemma 3n 加载完成!")

        except Exception as e:
            print(f"    警告: Gemma 3n 加载失败 ({e})")
            print("    将只进行关键帧提取，不进行多模态理解")

    def analyze_video(
        self,
        video_path: str,
        sample_fps: float = 5.0,
        prompt: Optional[str] = None,
        audio_path: Optional[str] = None,
        generate_response: bool = True
    ) -> VideoAnalysisResult:
        """
        分析视频文件

        Args:
            video_path: 视频文件路径
            sample_fps: 采样帧率
            prompt: 用户提示/问题
            audio_path: 音频文件路径 (可选)
            generate_response: 是否生成 Gemma 3n 响应

        Returns:
            VideoAnalysisResult
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"分析视频: {video_path}")
        print(f"{'='*60}")

        # 1. 加载视频帧
        print("\n[Step 1] 加载视频帧...")
        frames = self.encoder._load_video(video_path, int(sample_fps))
        print(f"    加载了 {len(frames)} 帧 (采样率: {sample_fps} FPS)")

        if not frames:
            return VideoAnalysisResult(
                keyframes=[], keyframe_indices=[], keyframe_timestamps=[],
                frame_infos=[], events=[], stats={}, response=None,
                processing_time=time.time() - start_time
            )

        # 2. V-JEPA2 编码 (分批处理长视频)
        print("\n[Step 2] V-JEPA2 视频编码...")
        encode_start = time.time()

        # 对长视频分批编码，记录每个 embedding 对应的原始帧范围
        batch_size = self.encoder.num_frames  # 每批处理的帧数
        tubelet_size = 2  # V-JEPA2 的时间压缩因子
        all_embeddings = []
        embedding_to_frame_map = []  # embedding_idx -> 原始帧索引

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            if len(batch_frames) >= 4:  # 至少4帧才处理
                batch_emb = self.encoder.encode_frames(batch_frames)
                all_embeddings.append(batch_emb)

                # 记录每个 embedding 对应的原始帧索引
                emb_per_batch = batch_emb.shape[0]
                frames_per_emb = max(1, len(batch_frames) // emb_per_batch)
                for i in range(emb_per_batch):
                    frame_idx = batch_start + i * frames_per_emb
                    embedding_to_frame_map.append(min(frame_idx, len(frames) - 1))

        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
        else:
            embeddings = self.encoder.encode_frames(frames)
            emb_count = embeddings.shape[0]
            frames_per_emb = max(1, len(frames) // emb_count)
            embedding_to_frame_map = [i * frames_per_emb for i in range(emb_count)]

        encode_time = time.time() - encode_start
        print(f"    编码完成: {embeddings.shape} ({encode_time:.2f}s)")
        print(f"    Embedding 数量: {len(embedding_to_frame_map)}")

        # 3. 语义变化检测
        print("\n[Step 3] 语义变化检测...")
        detect_start = time.time()

        # 计算实际的 embedding fps (用于时间戳计算)
        embedding_fps = len(embedding_to_frame_map) / (len(frames) / sample_fps)
        frame_infos, keyframe_emb_indices = self.detector.detect_changes(embeddings, embedding_fps)
        events = self.detector.detect_events(frame_infos)
        detect_time = time.time() - detect_start

        # 4. 关键帧选择 (将 embedding 索引映射回原始帧索引)
        print("\n[Step 4] 关键帧选择...")

        # 映射 embedding 索引到原始帧索引
        keyframe_indices = []
        for emb_idx in keyframe_emb_indices:
            if emb_idx < len(embedding_to_frame_map):
                keyframe_indices.append(embedding_to_frame_map[emb_idx])

        # 如果关键帧太多，选择变化分数最高的
        if len(keyframe_indices) > self.target_keyframes:
            scored = [(keyframe_indices[i], frame_infos[keyframe_emb_indices[i]].change_score)
                     for i in range(len(keyframe_indices)) if keyframe_emb_indices[i] < len(frame_infos)]
            scored.sort(key=lambda x: x[1], reverse=True)

            # 保证时间分布：将视频分成 target_keyframes 段，每段选一帧
            selected = set()
            segment_size = len(frames) // self.target_keyframes

            # 先按时间段选择
            for seg in range(self.target_keyframes):
                seg_start = seg * segment_size
                seg_end = (seg + 1) * segment_size
                # 在这个时间段内找变化分数最高的关键帧
                seg_frames = [(idx, score) for idx, score in scored if seg_start <= idx < seg_end]
                if seg_frames:
                    best_in_seg = max(seg_frames, key=lambda x: x[1])
                    selected.add(best_in_seg[0])

            # 如果还不够，从剩余的高分帧中补充
            for idx, score in scored:
                if len(selected) >= self.target_keyframes:
                    break
                selected.add(idx)

            keyframe_indices = sorted(list(selected))

        # 确保不超过原始帧数
        keyframe_indices = [i for i in keyframe_indices if i < len(frames)]

        keyframes = [frames[i] for i in keyframe_indices]
        keyframe_timestamps = [i / sample_fps for i in keyframe_indices]

        stats = self.detector.get_stats()
        stats["encode_time"] = encode_time
        stats["detect_time"] = detect_time
        stats["sample_fps"] = sample_fps
        stats["total_frames"] = len(frames)
        stats["selected_keyframes"] = len(keyframes)

        print(f"    原始帧数: {len(frames)}")
        print(f"    关键帧数: {len(keyframes)}")
        print(f"    压缩比: {len(keyframes)/len(frames):.1%}")
        print(f"    关键帧时间: {[f'{t:.1f}s' for t in keyframe_timestamps]}")

        # 5. 变化事件
        if events:
            print(f"\n[Step 5] 检测到 {len(events)} 个变化事件:")
            for event in events[:5]:  # 最多显示5个
                print(f"    - {event.start_time:.1f}s-{event.end_time:.1f}s: {event.description}")

        # 6. Gemma 3n 多模态理解
        response = None
        if generate_response and keyframes:
            response = self._generate_response(
                keyframes=keyframes,
                keyframe_timestamps=keyframe_timestamps,
                events=events,
                prompt=prompt,
                audio_path=audio_path
            )

        processing_time = time.time() - start_time
        stats["total_processing_time"] = processing_time

        print(f"\n{'='*60}")
        print(f"分析完成! 总耗时: {processing_time:.2f}s")
        print(f"{'='*60}")

        return VideoAnalysisResult(
            keyframes=keyframes,
            keyframe_indices=keyframe_indices,
            keyframe_timestamps=keyframe_timestamps,
            frame_infos=frame_infos,
            events=events,
            stats=stats,
            response=response,
            processing_time=processing_time
        )

    def analyze_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 5.0,
        prompt: Optional[str] = None,
        generate_response: bool = True
    ) -> VideoAnalysisResult:
        """
        分析帧序列 (用于实时视频流)

        Args:
            frames: 帧列表 [H, W, C] uint8
            fps: 帧率
            prompt: 用户提示
            generate_response: 是否生成响应

        Returns:
            VideoAnalysisResult
        """
        start_time = time.time()

        # 编码
        embeddings = self.encoder.encode_frames(frames)

        # 变化检测
        frame_infos, keyframe_indices = self.detector.detect_changes(embeddings, fps)
        events = self.detector.detect_events(frame_infos)

        # 限制关键帧数量
        if len(keyframe_indices) > self.target_keyframes:
            scored = [(i, frame_infos[i].change_score) for i in keyframe_indices]
            scored.sort(key=lambda x: x[1], reverse=True)
            keyframe_indices = sorted([idx for idx, _ in scored[:self.target_keyframes]])

        keyframes = [frames[i] for i in keyframe_indices]
        keyframe_timestamps = [i / fps for i in keyframe_indices]

        # 生成响应
        response = None
        if generate_response and keyframes:
            response = self._generate_response(
                keyframes=keyframes,
                keyframe_timestamps=keyframe_timestamps,
                events=events,
                prompt=prompt
            )

        return VideoAnalysisResult(
            keyframes=keyframes,
            keyframe_indices=keyframe_indices,
            keyframe_timestamps=keyframe_timestamps,
            frame_infos=frame_infos,
            events=events,
            stats=self.detector.get_stats(),
            response=response,
            processing_time=time.time() - start_time
        )

    def _generate_response(
        self,
        keyframes: List[np.ndarray],
        keyframe_timestamps: List[float],
        events: List,
        prompt: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Optional[str]:
        """
        使用 Gemma 3n 生成多模态响应

        Args:
            keyframes: 关键帧列表
            keyframe_timestamps: 关键帧时间戳
            events: 变化事件列表
            prompt: 用户提示
            audio_path: 音频文件路径

        Returns:
            response: 模型响应
        """
        if not self.gemma_loaded:
            self._load_gemma()

        if not self.gemma_loaded or not keyframes:
            return None

        print("\n[Step 6] Gemma 3n 多模态理解...")

        # 构建上下文
        context_parts = []

        # 添加关键帧时间信息
        time_info = ", ".join([f"Frame {i+1}: {t:.1f}s" for i, t in enumerate(keyframe_timestamps)])
        context_parts.append(f"[Video keyframes at timestamps: {time_info}]")

        # 添加事件信息
        if events:
            event_info = "; ".join([
                f"{e.start_time:.1f}s-{e.end_time:.1f}s: {e.description}"
                for e in events[:3]
            ])
            context_parts.append(f"[Detected events: {event_info}]")

        # 默认提示
        if not prompt:
            prompt = "Please analyze these video keyframes and describe what is happening over time."

        # 构建消息
        content = []

        # 添加关键帧图像 (转换为 PIL Image)
        for i, frame in enumerate(keyframes[:5]):  # 最多5张图
            img = Image.fromarray(frame)
            content.append({"type": "image", "image": img})

        # 添加音频 (如果有)
        if audio_path and os.path.exists(audio_path):
            try:
                import librosa
                audio_array, sr = librosa.load(audio_path, sr=16000)
                content.append({"type": "audio", "audio": audio_array, "sample_rate": sr})
                context_parts.append("[Audio track included]")
            except Exception as e:
                print(f"    警告: 音频加载失败 ({e})")

        # 构建完整提示
        full_prompt = "\n".join(context_parts) + "\n\n" + prompt
        content.append({"type": "text", "text": full_prompt})

        messages = [{"role": "user", "content": content}]

        # 生成响应
        try:
            inputs = self.gemma_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(self.gemma_model.device)
            attention_mask = inputs["attention_mask"].to(self.gemma_model.device)

            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": 512,
                "do_sample": False,
            }

            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                generate_kwargs["pixel_values"] = inputs["pixel_values"].to(
                    self.gemma_model.device, dtype=self.gemma_model.dtype
                )

            if "input_features" in inputs and inputs["input_features"] is not None:
                generate_kwargs["input_features"] = inputs["input_features"].to(
                    self.gemma_model.device, dtype=self.gemma_model.dtype
                )
                generate_kwargs["input_features_mask"] = inputs["input_features_mask"].to(
                    self.gemma_model.device
                )

            with torch.inference_mode():
                outputs = self.gemma_model.generate(**generate_kwargs)

            response = self.gemma_processor.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            print(f"    响应生成完成 ({len(response)} 字符)")
            return response

        except Exception as e:
            print(f"    错误: 响应生成失败 ({e})")
            return None

    def visualize_analysis(
        self,
        result: VideoAnalysisResult,
        output_path: Optional[str] = None
    ):
        """
        可视化分析结果

        Args:
            result: 分析结果
            output_path: 输出路径 (可选)
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(14, 8))

            # 上图: 变化分数曲线
            ax1 = axes[0]
            scores = [f.change_score for f in result.frame_infos]
            timestamps = [f.timestamp for f in result.frame_infos]

            ax1.plot(timestamps, scores, 'b-', label='Change Score', alpha=0.7)
            ax1.axhline(y=self.detector.get_adaptive_threshold(),
                       color='r', linestyle='--', label='Threshold')

            # 标记关键帧
            for idx in result.keyframe_indices:
                ax1.axvline(x=result.frame_infos[idx].timestamp,
                           color='g', alpha=0.3)

            # 标记事件
            for event in result.events:
                ax1.axvspan(event.start_time, event.end_time,
                           alpha=0.2, color='orange')

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Change Score')
            ax1.set_title('Semantic Change Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 下图: 关键帧展示
            ax2 = axes[1]
            n_keyframes = min(5, len(result.keyframes))

            for i in range(n_keyframes):
                ax_sub = fig.add_axes([
                    0.1 + i * 0.18, 0.05, 0.15, 0.3
                ])
                ax_sub.imshow(result.keyframes[i])
                ax_sub.set_title(f'{result.keyframe_timestamps[i]:.1f}s')
                ax_sub.axis('off')

            ax2.axis('off')
            ax2.set_title('Selected Keyframes')

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"可视化保存至: {output_path}")
            else:
                plt.show()

        except ImportError:
            print("警告: matplotlib 未安装，无法可视化")


# ============================================================
# 便捷函数
# ============================================================

def analyze_video(
    video_path: str,
    prompt: Optional[str] = None,
    target_keyframes: int = 5,
    sample_fps: float = 5.0,
    load_gemma: bool = True
) -> VideoAnalysisResult:
    """
    便捷函数: 分析视频文件

    Args:
        video_path: 视频文件路径
        prompt: 用户提示
        target_keyframes: 目标关键帧数
        sample_fps: 采样帧率
        load_gemma: 是否加载 Gemma 3n

    Returns:
        VideoAnalysisResult
    """
    pipeline = VideoPipeline(
        target_keyframes=target_keyframes,
        load_gemma=load_gemma
    )
    return pipeline.analyze_video(
        video_path=video_path,
        sample_fps=sample_fps,
        prompt=prompt,
        generate_response=load_gemma
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("V-JEPA2 + Gemma 3n 视频理解管道测试")
    print("=" * 60)

    # 初始化管道 (不加载 Gemma，仅测试 V-JEPA2)
    pipeline = VideoPipeline(
        vjepa_model_size="L",
        target_keyframes=5,
        load_gemma=False  # 先不加载 Gemma，只测试关键帧提取
    )

    # 创建模拟视频帧
    print("\n创建模拟视频帧 (60帧)...")

    frames = []
    for i in range(60):
        # 模拟场景: 静止 → 运动 → 静止
        if i < 20:
            # 静止场景
            frame = np.full((256, 256, 3), 100, dtype=np.uint8)
        elif i < 40:
            # 运动场景 (渐变)
            intensity = int(100 + (i - 20) * 7)
            frame = np.full((256, 256, 3), intensity, dtype=np.uint8)
        else:
            # 新的静止场景
            frame = np.full((256, 256, 3), 240, dtype=np.uint8)

        frames.append(frame)

    # 分析帧序列
    print("\n分析帧序列...")
    result = pipeline.analyze_frames(
        frames=frames,
        fps=10.0,
        generate_response=False
    )

    print(f"\n分析结果:")
    print(f"  关键帧索引: {result.keyframe_indices}")
    print(f"  关键帧时间: {result.keyframe_timestamps}")
    print(f"  检测事件数: {len(result.events)}")
    print(f"  处理耗时: {result.processing_time:.2f}s")
    print(f"  统计信息: {result.stats}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

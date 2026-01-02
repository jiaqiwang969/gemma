"""
V-JEPA2 + Gemma 3n 增强版视频理解管道

改进:
1. 自适应阈值 - 根据视频内容动态调整
2. 时序上下文增强 - 向 Gemma 3n 传递更丰富的时序信息
3. Embedding 投影 - 将 V-JEPA2 语义信息注入提示
4. 变化事件描述 - 提供结构化的变化事件信息
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from PIL import Image

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa2.vjepa2_encoder import VJEPA2Encoder
from vjepa2.change_detector import SemanticChangeDetector, FrameInfo, ChangeEvent


@dataclass
class EnhancedVideoAnalysisResult:
    """增强版视频分析结果"""
    # 基础信息
    keyframes: List[np.ndarray]
    keyframe_indices: List[int]
    keyframe_timestamps: List[float]
    keyframe_scores: List[float]  # 每个关键帧的变化分数

    # V-JEPA2 语义信息
    keyframe_embeddings: torch.Tensor  # 关键帧的 embedding
    temporal_context: str  # 时序上下文描述
    semantic_summary: str  # 语义变化摘要

    # 变化分析
    frame_infos: List[FrameInfo]
    events: List[ChangeEvent]
    activity_level: str  # 活动级别: static/low/medium/high

    # 统计
    stats: Dict

    # Gemma 3n 响应
    response: Optional[str] = None
    processing_time: float = 0.0


class AdaptiveThresholdCalculator:
    """
    自适应阈值计算器

    根据视频内容特征动态调整变化检测阈值
    """

    def __init__(self):
        self.min_threshold = 0.02
        self.max_threshold = 0.3

    def calculate(self, change_scores: List[float]) -> Tuple[float, str]:
        """
        计算自适应阈值

        Args:
            change_scores: 所有帧的变化分数列表

        Returns:
            threshold: 自适应阈值
            activity_level: 活动级别描述
        """
        if not change_scores:
            return 0.1, "unknown"

        scores = np.array(change_scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)

        # 计算百分位数
        p50 = np.percentile(scores, 50)
        p75 = np.percentile(scores, 75)
        p90 = np.percentile(scores, 90)

        # 判断活动级别
        if max_score < 0.03:
            activity_level = "static"  # 几乎静止
            threshold = max(self.min_threshold, p75 * 1.5)
        elif max_score < 0.08:
            activity_level = "low"  # 低活动
            threshold = max(self.min_threshold, p75)
        elif max_score < 0.2:
            activity_level = "medium"  # 中等活动
            threshold = mean_score + std_score
        else:
            activity_level = "high"  # 高活动
            threshold = mean_score + 1.5 * std_score

        # 限制阈值范围
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))

        return threshold, activity_level


class TemporalContextBuilder:
    """
    时序上下文构建器

    将 V-JEPA2 分析结果转换为结构化的文本描述
    """

    def build_context(
        self,
        frame_infos: List[FrameInfo],
        keyframe_indices: List[int],
        keyframe_timestamps: List[float],
        events: List[ChangeEvent],
        activity_level: str,
        video_duration: float
    ) -> Tuple[str, str]:
        """
        构建时序上下文

        Returns:
            temporal_context: 详细的时序上下文
            semantic_summary: 简短的语义摘要
        """
        # 1. 构建时序上下文
        context_parts = []

        # 视频基本信息
        context_parts.append(f"[Video Duration: {video_duration:.1f}s | Activity Level: {activity_level}]")

        # 关键帧时间线
        keyframe_desc = []
        for i, (idx, ts) in enumerate(zip(keyframe_indices, keyframe_timestamps)):
            if idx < len(frame_infos):
                score = frame_infos[idx].change_score
                if score < 0.03:
                    change_type = "stable"
                elif score < 0.08:
                    change_type = "minor_change"
                elif score < 0.15:
                    change_type = "moderate_change"
                else:
                    change_type = "major_change"
                keyframe_desc.append(f"Frame{i+1}({ts:.1f}s): {change_type}")

        context_parts.append("[Timeline: " + " → ".join(keyframe_desc) + "]")

        # 变化事件
        if events:
            event_desc = []
            for e in events[:5]:  # 最多5个事件
                intensity_label = "high" if e.intensity > 0.5 else "medium" if e.intensity > 0.2 else "low"
                event_desc.append(f"{e.start_time:.1f}s-{e.end_time:.1f}s({intensity_label})")
            context_parts.append("[Change Events: " + ", ".join(event_desc) + "]")
        else:
            context_parts.append("[No significant change events detected]")

        temporal_context = "\n".join(context_parts)

        # 2. 构建语义摘要
        summary_parts = []

        # 活动描述
        activity_desc = {
            "static": "This appears to be a mostly static scene with minimal motion.",
            "low": "The video shows subtle movements or gradual changes.",
            "medium": "The video contains noticeable activity and scene changes.",
            "high": "The video is dynamic with significant motion and scene transitions."
        }
        summary_parts.append(activity_desc.get(activity_level, ""))

        # 时间分布描述
        if len(keyframe_timestamps) >= 2:
            intervals = np.diff(keyframe_timestamps)
            avg_interval = np.mean(intervals)
            if avg_interval < 1.0:
                summary_parts.append("Keyframes are densely distributed, indicating rapid changes.")
            elif avg_interval > 3.0:
                summary_parts.append("Keyframes are sparsely distributed, with long stable periods.")
            else:
                summary_parts.append("Keyframes are evenly distributed throughout the video.")

        # 事件描述
        if events:
            summary_parts.append(f"Detected {len(events)} distinct change event(s).")

        semantic_summary = " ".join(summary_parts)

        return temporal_context, semantic_summary


class EnhancedVideoPipeline:
    """
    V-JEPA2 + Gemma 3n 增强版视频理解管道

    核心改进:
    1. 自适应阈值 - 对静态视频更敏感，对动态视频更宽松
    2. 时序上下文 - 向 LLM 提供结构化的时间信息
    3. 语义摘要 - 基于 V-JEPA2 分析生成的初步理解
    """

    def __init__(
        self,
        # V-JEPA2 参数
        vjepa_model_size: str = "L",
        vjepa_img_size: int = 256,
        vjepa_num_frames: int = 16,
        # 关键帧参数
        target_keyframes: int = 8,
        min_keyframe_interval: int = 2,
        max_keyframe_interval: int = 15,
        # Gemma 3n 参数
        gemma_model_name: str = "google/gemma-3n-E2B-it",
        load_gemma: bool = False,
        # 设备
        device: Optional[str] = None,
    ):
        self.target_keyframes = target_keyframes
        self.gemma_model_name = gemma_model_name
        self.device = device

        print("=" * 60)
        print("初始化 V-JEPA2 + Gemma 3n 增强版管道")
        print("=" * 60)

        # 初始化 V-JEPA2 编码器
        print("\n[1/4] 初始化 V-JEPA2 编码器...")
        self.encoder = VJEPA2Encoder(
            model_size=vjepa_model_size,
            img_size=vjepa_img_size,
            num_frames=vjepa_num_frames,
            device=device
        )

        # 初始化变化检测器 (使用较低的基础阈值)
        print("\n[2/4] 初始化变化检测器...")
        self.detector = SemanticChangeDetector(
            base_threshold=0.05,  # 较低的基础阈值
            min_keyframe_interval=min_keyframe_interval,
            max_keyframe_interval=max_keyframe_interval,
            adaptive_factor=1.0
        )

        # 初始化辅助组件
        print("\n[3/4] 初始化增强组件...")
        self.threshold_calculator = AdaptiveThresholdCalculator()
        self.context_builder = TemporalContextBuilder()

        # Gemma 3n (延迟加载)
        self.gemma_model = None
        self.gemma_processor = None
        self.gemma_loaded = False

        if load_gemma:
            self._load_gemma()

        print("\n" + "=" * 60)
        print("增强版管道初始化完成!")
        print("=" * 60)

    def _load_gemma(self):
        """加载 Gemma 3n 模型"""
        if self.gemma_loaded:
            return

        print("\n[4/4] 加载 Gemma 3n 多模态模型...")

        try:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

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

    def analyze_video(
        self,
        video_path: str,
        sample_fps: float = 5.0,
        prompt: Optional[str] = None,
        audio_path: Optional[str] = None,
        generate_response: bool = True
    ) -> EnhancedVideoAnalysisResult:
        """
        分析视频文件 (增强版)
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"分析视频: {video_path}")
        print(f"{'='*60}")

        # 1. 加载视频帧
        print("\n[Step 1] 加载视频帧...")
        frames = self.encoder._load_video(video_path, int(sample_fps))
        video_duration = len(frames) / sample_fps
        print(f"    加载了 {len(frames)} 帧 ({video_duration:.1f}s, {sample_fps} FPS)")

        if not frames:
            return self._empty_result(start_time)

        # 2. V-JEPA2 编码
        print("\n[Step 2] V-JEPA2 视频编码...")
        embeddings, embedding_to_frame_map = self._batch_encode(frames)
        print(f"    编码完成: {embeddings.shape}")

        # 3. 计算变化分数
        print("\n[Step 3] 计算语义变化...")
        raw_scores = []
        for i in range(len(embeddings)):
            if i == 0:
                score = 0.0
            else:
                sim = F.cosine_similarity(
                    embeddings[i-1].unsqueeze(0),
                    embeddings[i].unsqueeze(0)
                )
                score = max(0.0, 1.0 - sim.item())
            raw_scores.append(score)

        # 4. 自适应阈值
        print("\n[Step 4] 自适应阈值计算...")
        threshold, activity_level = self.threshold_calculator.calculate(raw_scores)
        print(f"    活动级别: {activity_level}")
        print(f"    自适应阈值: {threshold:.4f}")
        print(f"    变化范围: {min(raw_scores):.4f} - {max(raw_scores):.4f}")

        # 5. 使用自适应阈值检测变化
        print("\n[Step 5] 关键帧选择...")
        self.detector.base_threshold = threshold

        # 根据活动级别调整间隔
        if activity_level == "static":
            self.detector.max_keyframe_interval = 8
        elif activity_level == "low":
            self.detector.max_keyframe_interval = 10
        else:
            self.detector.max_keyframe_interval = 15

        frame_infos, keyframe_emb_indices = self.detector.detect_changes(embeddings, len(embeddings) / video_duration)
        events = self.detector.detect_events(frame_infos, event_threshold=threshold * 2)

        # 6. 映射回原始帧索引
        keyframe_indices = []
        for emb_idx in keyframe_emb_indices:
            if emb_idx < len(embedding_to_frame_map):
                keyframe_indices.append(embedding_to_frame_map[emb_idx])

        # 7. 优化关键帧选择
        keyframe_indices, keyframe_scores = self._optimize_keyframes(
            keyframe_indices, frame_infos, keyframe_emb_indices,
            len(frames), self.target_keyframes
        )

        # 确保索引有效
        keyframe_indices = [i for i in keyframe_indices if i < len(frames)]
        keyframes = [frames[i] for i in keyframe_indices]
        keyframe_timestamps = [i / sample_fps for i in keyframe_indices]

        # 8. 获取关键帧 embedding
        keyframe_embeddings = torch.stack([
            embeddings[min(idx // max(1, len(frames) // len(embeddings)), len(embeddings)-1)]
            for idx in keyframe_indices
        ]) if keyframe_indices else torch.zeros(0, self.encoder.embedding_dim)

        # 9. 构建时序上下文
        print("\n[Step 6] 构建时序上下文...")
        temporal_context, semantic_summary = self.context_builder.build_context(
            frame_infos, keyframe_indices, keyframe_timestamps,
            events, activity_level, video_duration
        )
        print(f"    活动级别: {activity_level}")
        print(f"    关键帧数: {len(keyframes)}")
        print(f"    变化事件: {len(events)}")

        stats = {
            "total_frames": len(frames),
            "embeddings": len(embeddings),
            "selected_keyframes": len(keyframes),
            "activity_level": activity_level,
            "adaptive_threshold": threshold,
            "sample_fps": sample_fps,
            "video_duration": video_duration,
            "events_count": len(events),
            "compression_ratio": f"{len(keyframes)/len(frames):.1%}"
        }

        # 10. Gemma 3n 响应
        response = None
        if generate_response and keyframes:
            response = self._generate_enhanced_response(
                keyframes=keyframes,
                keyframe_timestamps=keyframe_timestamps,
                temporal_context=temporal_context,
                semantic_summary=semantic_summary,
                events=events,
                prompt=prompt,
                audio_path=audio_path
            )

        processing_time = time.time() - start_time
        stats["processing_time"] = processing_time

        print(f"\n{'='*60}")
        print(f"分析完成! 总耗时: {processing_time:.2f}s")
        print(f"{'='*60}")

        return EnhancedVideoAnalysisResult(
            keyframes=keyframes,
            keyframe_indices=keyframe_indices,
            keyframe_timestamps=keyframe_timestamps,
            keyframe_scores=keyframe_scores,
            keyframe_embeddings=keyframe_embeddings,
            temporal_context=temporal_context,
            semantic_summary=semantic_summary,
            frame_infos=frame_infos,
            events=events,
            activity_level=activity_level,
            stats=stats,
            response=response,
            processing_time=processing_time
        )

    def _batch_encode(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, List[int]]:
        """分批编码帧"""
        batch_size = self.encoder.num_frames
        all_embeddings = []
        embedding_to_frame_map = []

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            if len(batch_frames) >= 4:
                batch_emb = self.encoder.encode_frames(batch_frames)
                all_embeddings.append(batch_emb)

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

        return embeddings, embedding_to_frame_map

    def _optimize_keyframes(
        self,
        keyframe_indices: List[int],
        frame_infos: List[FrameInfo],
        keyframe_emb_indices: List[int],
        total_frames: int,
        target_count: int
    ) -> Tuple[List[int], List[float]]:
        """
        优化关键帧选择 - 确保时间均匀分布

        策略:
        1. 将视频分成 target_count 个时间段
        2. 每个时间段选择变化分数最高的帧
        3. 如果某时间段没有检测到的关键帧，选择该段中间帧
        """
        # 收集所有帧的变化分数
        all_scores = {}
        for i, info in enumerate(frame_infos):
            # 映射 embedding 索引到帧索引
            if i < len(keyframe_emb_indices):
                frame_idx = keyframe_indices[min(i, len(keyframe_indices)-1)] if keyframe_indices else i
            else:
                frame_idx = i * (total_frames // len(frame_infos)) if frame_infos else i
            all_scores[frame_idx] = info.change_score

        # 确保有足够的候选帧
        segment_size = max(1, total_frames // target_count)
        selected = []

        for seg in range(target_count):
            seg_start = seg * segment_size
            seg_end = min((seg + 1) * segment_size, total_frames)
            if seg == target_count - 1:
                seg_end = total_frames

            # 在这个时间段内找最佳帧
            best_idx = None
            best_score = -1

            # 优先从已检测的关键帧中选择
            for idx in keyframe_indices:
                if seg_start <= idx < seg_end:
                    score = all_scores.get(idx, 0.0)
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            # 如果没有检测到的关键帧，从所有帧中选择变化最大的
            if best_idx is None:
                for idx in range(seg_start, seg_end):
                    score = all_scores.get(idx, 0.0)
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            # 如果仍然没有，选择时间段中点
            if best_idx is None:
                best_idx = (seg_start + seg_end) // 2
                best_score = 0.0

            selected.append((best_idx, best_score))

        # 确保首帧
        if selected and selected[0][0] != 0:
            selected[0] = (0, all_scores.get(0, 0.0))

        # 确保尾帧 (如果需要)
        if selected and selected[-1][0] < total_frames - 1:
            selected[-1] = (total_frames - 1, all_scores.get(total_frames - 1, 0.0))

        # 去重并排序
        seen = set()
        unique = []
        for idx, score in selected:
            if idx not in seen and idx < total_frames:
                seen.add(idx)
                unique.append((idx, score))

        unique.sort(key=lambda x: x[0])

        indices = [x[0] for x in unique]
        scores = [x[1] for x in unique]

        return indices, scores

    def _generate_enhanced_response(
        self,
        keyframes: List[np.ndarray],
        keyframe_timestamps: List[float],
        temporal_context: str,
        semantic_summary: str,
        events: List[ChangeEvent],
        prompt: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Optional[str]:
        """生成增强版多模态响应"""
        if not self.gemma_loaded:
            self._load_gemma()

        if not self.gemma_loaded or not keyframes:
            return None

        print("\n[Step 7] Gemma 3n 多模态理解...")

        # 构建增强提示
        enhanced_prompt_parts = [
            "## V-JEPA2 Video Analysis Context",
            "",
            temporal_context,
            "",
            f"**Semantic Summary:** {semantic_summary}",
            "",
        ]

        # 添加变化事件
        if events:
            enhanced_prompt_parts.append("**Detected Events:**")
            for i, e in enumerate(events[:5], 1):
                enhanced_prompt_parts.append(
                    f"  {i}. [{e.start_time:.1f}s-{e.end_time:.1f}s] {e.description}"
                )
            enhanced_prompt_parts.append("")

        # 添加用户提示
        enhanced_prompt_parts.append("## User Query")
        if prompt:
            enhanced_prompt_parts.append(prompt)
        else:
            enhanced_prompt_parts.append(
                "Please analyze the video keyframes above. Describe:\n"
                "1. What objects, people, or scenes are visible?\n"
                "2. What actions or events occur?\n"
                "3. How does the content change over time?\n"
                "4. Provide a concise summary of the video."
            )

        full_prompt = "\n".join(enhanced_prompt_parts)

        # 构建消息
        content = []

        # 添加关键帧 (最多5张)
        for frame in keyframes[:5]:
            img = Image.fromarray(frame)
            content.append({"type": "image", "image": img})

        # 添加音频
        if audio_path and os.path.exists(audio_path):
            try:
                import librosa
                audio_array, sr = librosa.load(audio_path, sr=16000)
                content.append({"type": "audio", "audio": audio_array, "sample_rate": sr})
            except Exception as e:
                print(f"    警告: 音频加载失败 ({e})")

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
            import traceback
            traceback.print_exc()
            return None

    def _empty_result(self, start_time: float) -> EnhancedVideoAnalysisResult:
        """返回空结果"""
        return EnhancedVideoAnalysisResult(
            keyframes=[], keyframe_indices=[], keyframe_timestamps=[],
            keyframe_scores=[], keyframe_embeddings=torch.zeros(0, 1024),
            temporal_context="", semantic_summary="",
            frame_infos=[], events=[], activity_level="unknown",
            stats={}, response=None, processing_time=time.time() - start_time
        )


# ============================================================
# 便捷函数
# ============================================================

def analyze_video_enhanced(
    video_path: str,
    prompt: Optional[str] = None,
    target_keyframes: int = 8,
    sample_fps: float = 5.0,
    load_gemma: bool = True
) -> EnhancedVideoAnalysisResult:
    """
    便捷函数: 使用增强版管道分析视频
    """
    pipeline = EnhancedVideoPipeline(
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
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("V-JEPA2 + Gemma 3n 增强版管道测试")
    print("=" * 60)

    # 测试自适应阈值计算器
    print("\n测试自适应阈值计算器:")
    calc = AdaptiveThresholdCalculator()

    # 静态视频场景
    static_scores = [0.01, 0.02, 0.01, 0.015, 0.02, 0.01, 0.02, 0.015]
    threshold, level = calc.calculate(static_scores)
    print(f"  静态视频: threshold={threshold:.4f}, level={level}")

    # 动态视频场景
    dynamic_scores = [0.05, 0.15, 0.25, 0.1, 0.3, 0.2, 0.15, 0.1]
    threshold, level = calc.calculate(dynamic_scores)
    print(f"  动态视频: threshold={threshold:.4f}, level={level}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

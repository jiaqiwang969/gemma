"""
视频 Thought Signature 系统

核心思想: V-JEPA2 和 Gemma 3n 的迭代交互
- 不是一次性处理，而是不断深化的理解过程
- Signature 是动态演化的"压缩记忆"

流程:
1. 初始扫描: V-JEPA2 全局分析 → 生成初始 Signature
2. 模型理解: Gemma 3n 基于 Signature 生成理解
3. 深化分析: 基于理解更新 V-JEPA2 关注点
4. 迭代: 重复 2-3 直到理解充分或用户满意

这个过程模拟了人观看视频的方式:
- 先快速浏览全局
- 发现感兴趣的内容
- 聚焦特定区域深入分析
- 不断更新对视频的整体理解
"""

import os
import sys
import time
import json
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 视频 Thought Signature
# ============================================================

@dataclass
class VideoThoughtSignature:
    """
    视频思维签名 - V-JEPA2 和 Gemma 3n 交互的压缩记忆

    这不是静态的，而是随着交互不断演化的
    """
    # 基础信息
    video_id: str
    video_duration: float
    created_at: float = field(default_factory=time.time)

    # V-JEPA2 分析状态
    global_embedding: Optional[torch.Tensor] = None  # 全局语义
    keyframe_embeddings: Optional[torch.Tensor] = None  # 关键帧 embeddings
    keyframe_times: List[float] = field(default_factory=list)
    change_profile: List[float] = field(default_factory=list)  # 变化曲线
    activity_level: str = "unknown"

    # 当前关注状态 (动态更新)
    current_focus: str = "global"  # global / temporal / spatial / audio
    focus_regions: List[Tuple[float, float]] = field(default_factory=list)  # 时间区域
    attention_weights: Optional[torch.Tensor] = None  # 帧注意力权重

    # Gemma 3n 理解状态 (累积)
    visual_understanding: str = ""  # 视觉理解
    audio_understanding: str = ""   # 音频理解
    temporal_understanding: str = ""  # 时序理解
    semantic_summary: str = ""  # 语义摘要

    # 用户意图 (动态)
    inferred_intents: List[str] = field(default_factory=list)  # 推断的意图
    user_questions: List[str] = field(default_factory=list)  # 用户问题历史

    # 迭代状态
    iteration: int = 0
    understanding_depth: float = 0.0  # 理解深度 (0-1)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "video_id": self.video_id,
            "video_duration": self.video_duration,
            "created_at": self.created_at,
            "keyframe_times": self.keyframe_times,
            "activity_level": self.activity_level,
            "current_focus": self.current_focus,
            "focus_regions": self.focus_regions,
            "visual_understanding": self.visual_understanding,
            "audio_understanding": self.audio_understanding,
            "temporal_understanding": self.temporal_understanding,
            "semantic_summary": self.semantic_summary,
            "inferred_intents": self.inferred_intents,
            "user_questions": self.user_questions,
            "iteration": self.iteration,
            "understanding_depth": self.understanding_depth,
        }

    def to_context_string(self) -> str:
        """生成上下文字符串供 LLM 使用"""
        parts = []

        parts.append(f"## Video Context (Iteration {self.iteration})")
        parts.append(f"- Duration: {self.video_duration:.1f}s")
        parts.append(f"- Activity: {self.activity_level}")
        parts.append(f"- Current Focus: {self.current_focus}")
        parts.append(f"- Understanding Depth: {self.understanding_depth:.0%}")

        if self.focus_regions:
            regions = ", ".join([f"{s:.1f}s-{e:.1f}s" for s, e in self.focus_regions])
            parts.append(f"- Focus Regions: {regions}")

        if self.visual_understanding:
            parts.append(f"\n### Visual Understanding\n{self.visual_understanding}")

        if self.audio_understanding:
            parts.append(f"\n### Audio Understanding\n{self.audio_understanding}")

        if self.temporal_understanding:
            parts.append(f"\n### Temporal Understanding\n{self.temporal_understanding}")

        if self.semantic_summary:
            parts.append(f"\n### Current Summary\n{self.semantic_summary}")

        if self.inferred_intents:
            parts.append(f"\n### Inferred User Intents\n- " + "\n- ".join(self.inferred_intents))

        return "\n".join(parts)


class FocusType(Enum):
    """关注类型"""
    GLOBAL = "global"      # 全局扫描
    TEMPORAL = "temporal"  # 时间定位
    SPATIAL = "spatial"    # 空间区域
    AUDIO = "audio"        # 音频内容
    DETAIL = "detail"      # 细节分析


# ============================================================
# 迭代式视频理解器
# ============================================================

class IterativeVideoUnderstanding:
    """
    迭代式视频理解系统

    核心: V-JEPA2 和 Gemma 3n 的双向交互
    - V-JEPA2: 提供视觉感知、时序分析、embedding
    - Gemma 3n: 提供语义理解、意图推断、问答
    - Signature: 两者交互的"共享记忆"
    """

    def __init__(
        self,
        vjepa_model_size: str = "L",
        device: Optional[str] = None
    ):
        print("=" * 60)
        print("初始化迭代式视频理解系统")
        print("=" * 60)

        from vjepa2.vjepa2_encoder import VJEPA2Encoder

        self.encoder = VJEPA2Encoder(
            model_size=vjepa_model_size,
            device=device
        )

        self.signatures: Dict[str, VideoThoughtSignature] = {}
        self.gemma_model = None
        self.gemma_processor = None

        print("初始化完成!")

    def create_signature(
        self,
        video_path: str,
        frames: List[np.ndarray],
        video_duration: float
    ) -> VideoThoughtSignature:
        """
        创建初始 Thought Signature

        这是 V-JEPA2 的初始分析，相当于"快速浏览"
        """
        video_id = hashlib.md5(video_path.encode()).hexdigest()[:12]

        print(f"\n[Iteration 0] 创建初始 Signature...")

        # V-JEPA2 编码
        embeddings = self._encode_frames(frames)
        T = embeddings.shape[0]

        # 全局语义
        global_embedding = embeddings.mean(dim=0)

        # 变化曲线
        change_profile = [0.0]
        for i in range(1, T):
            sim = F.cosine_similarity(embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0))
            change_profile.append(max(0.0, 1.0 - sim.item()))

        # 活动级别
        max_change = max(change_profile)
        if max_change < 0.03:
            activity_level = "static"
        elif max_change < 0.08:
            activity_level = "low"
        elif max_change < 0.2:
            activity_level = "medium"
        else:
            activity_level = "high"

        # 初始关键帧选择 (均匀采样)
        num_keyframes = 5
        step = max(1, T // num_keyframes)
        keyframe_indices = [min(i * step, T - 1) for i in range(num_keyframes)]
        keyframe_times = [idx / T * video_duration for idx in keyframe_indices]
        keyframe_embeddings = embeddings[keyframe_indices]

        # 创建 Signature
        signature = VideoThoughtSignature(
            video_id=video_id,
            video_duration=video_duration,
            global_embedding=global_embedding,
            keyframe_embeddings=keyframe_embeddings,
            keyframe_times=keyframe_times,
            change_profile=change_profile,
            activity_level=activity_level,
            current_focus="global",
            iteration=0,
            understanding_depth=0.1,  # 初始理解深度 10%
        )

        # 初始时序理解 (基于变化曲线)
        signature.temporal_understanding = self._generate_temporal_summary(
            change_profile, video_duration
        )

        self.signatures[video_id] = signature
        print(f"    创建完成: video_id={video_id}, activity={activity_level}")

        return signature

    def update_with_gemma_understanding(
        self,
        signature: VideoThoughtSignature,
        gemma_response: str,
        response_type: str = "visual"
    ) -> VideoThoughtSignature:
        """
        用 Gemma 3n 的理解更新 Signature

        这是 Gemma 3n → V-JEPA2 的反馈
        """
        signature.iteration += 1
        print(f"\n[Iteration {signature.iteration}] 更新 Gemma 理解...")

        if response_type == "visual":
            # 累积视觉理解
            if signature.visual_understanding:
                signature.visual_understanding += f"\n[Iter {signature.iteration}]: {gemma_response}"
            else:
                signature.visual_understanding = gemma_response

        elif response_type == "audio":
            if signature.audio_understanding:
                signature.audio_understanding += f"\n[Iter {signature.iteration}]: {gemma_response}"
            else:
                signature.audio_understanding = gemma_response

        elif response_type == "summary":
            signature.semantic_summary = gemma_response

        # 从响应中推断意图/关注点
        inferred_focus, new_intents = self._infer_focus_from_response(gemma_response)

        if inferred_focus:
            signature.current_focus = inferred_focus

        for intent in new_intents:
            if intent not in signature.inferred_intents:
                signature.inferred_intents.append(intent)

        # 更新理解深度
        signature.understanding_depth = min(1.0, signature.understanding_depth + 0.15)

        print(f"    Focus: {signature.current_focus}")
        print(f"    Depth: {signature.understanding_depth:.0%}")
        print(f"    Intents: {signature.inferred_intents}")

        return signature

    def refocus_vjepa2(
        self,
        signature: VideoThoughtSignature,
        frames: List[np.ndarray],
        focus_type: FocusType,
        focus_params: Optional[Dict] = None
    ) -> Tuple[List[int], torch.Tensor]:
        """
        基于当前 Signature 重新聚焦 V-JEPA2

        这是 Signature → V-JEPA2 的反馈
        根据理解状态调整关键帧选择策略
        """
        signature.iteration += 1
        print(f"\n[Iteration {signature.iteration}] V-JEPA2 重新聚焦: {focus_type.value}")

        embeddings = self._encode_frames(frames)
        T = embeddings.shape[0]

        focus_params = focus_params or {}

        if focus_type == FocusType.GLOBAL:
            # 全局: 均匀采样
            indices = self._select_uniform(T, 5)

        elif focus_type == FocusType.TEMPORAL:
            # 时间定位: 聚焦变化区域
            indices = self._select_by_change(signature.change_profile, 5)

            # 更新关注区域
            signature.focus_regions = self._find_active_regions(signature.change_profile, signature.video_duration)

        elif focus_type == FocusType.SPATIAL:
            # 空间: 基于特定时间范围
            time_range = focus_params.get("time_range", (0, signature.video_duration))
            indices = self._select_in_time_range(T, time_range, signature.video_duration, 5)

        elif focus_type == FocusType.AUDIO:
            # 音频: 选择视觉稳定的帧
            indices = self._select_stable_frames(signature.change_profile, 3)

        elif focus_type == FocusType.DETAIL:
            # 细节: 基于与全局语义的差异
            indices = self._select_unique_frames(embeddings, signature.global_embedding, 5)

        else:
            indices = self._select_uniform(T, 5)

        # 更新 Signature
        signature.current_focus = focus_type.value
        signature.keyframe_times = [idx / T * signature.video_duration for idx in indices]
        signature.keyframe_embeddings = embeddings[indices]

        # 计算新的注意力权重
        signature.attention_weights = self._compute_attention_weights(embeddings, indices)

        print(f"    选择帧: {indices}")
        print(f"    时间: {[f'{t:.1f}s' for t in signature.keyframe_times]}")

        return indices, embeddings[indices]

    def process_user_query(
        self,
        signature: VideoThoughtSignature,
        query: str,
        frames: List[np.ndarray]
    ) -> Tuple[VideoThoughtSignature, List[int], str]:
        """
        处理用户查询 - 核心交互循环

        1. 分析查询 → 更新意图
        2. 基于意图 → 调整 V-JEPA2 策略
        3. 重新选择关键帧
        4. 生成上下文供 Gemma 3n 使用
        """
        signature.iteration += 1
        print(f"\n[Iteration {signature.iteration}] 处理用户查询: {query}")

        # 记录问题
        signature.user_questions.append(query)

        # 分析查询意图
        focus_type, new_intents = self._analyze_query(query)
        print(f"    推断 Focus: {focus_type.value}")
        print(f"    新意图: {new_intents}")

        # 更新意图
        for intent in new_intents:
            if intent not in signature.inferred_intents:
                signature.inferred_intents.append(intent)

        # 根据意图调整 V-JEPA2
        focus_params = self._get_focus_params(query, signature)
        indices, keyframe_embs = self.refocus_vjepa2(signature, frames, focus_type, focus_params)

        # 生成增强上下文
        context = self._build_enhanced_context(signature, query)

        return signature, indices, context

    # ============================================================
    # 内部方法
    # ============================================================

    def _encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """分批编码帧"""
        batch_size = self.encoder.num_frames
        all_embeddings = []

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            if len(batch_frames) >= 4:
                batch_emb = self.encoder.encode_frames(batch_frames)
                all_embeddings.append(batch_emb)

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return self.encoder.encode_frames(frames)

    def _generate_temporal_summary(self, change_profile: List[float], duration: float) -> str:
        """生成时序摘要"""
        T = len(change_profile)
        if T == 0:
            return "No temporal data."

        # 找变化峰值
        peaks = []
        for i in range(1, T - 1):
            if change_profile[i] > change_profile[i-1] and change_profile[i] > change_profile[i+1]:
                if change_profile[i] > np.mean(change_profile) * 1.5:
                    time = i / T * duration
                    peaks.append((time, change_profile[i]))

        if not peaks:
            return f"Video is mostly stable throughout its {duration:.1f}s duration."

        peak_desc = ", ".join([f"{t:.1f}s" for t, _ in peaks[:3]])
        return f"Significant changes detected at: {peak_desc}. Total duration: {duration:.1f}s."

    def _infer_focus_from_response(self, response: str) -> Tuple[Optional[str], List[str]]:
        """从 Gemma 响应推断关注点"""
        response_lower = response.lower()
        new_focus = None
        intents = []

        # 时间相关
        if any(kw in response_lower for kw in ["时间", "when", "moment", "开始", "结束"]):
            new_focus = "temporal"
            intents.append("temporal_localization")

        # 空间/物体相关
        if any(kw in response_lower for kw in ["位置", "where", "left", "right", "区域"]):
            new_focus = "spatial"
            intents.append("spatial_localization")

        # 音频相关
        if any(kw in response_lower for kw in ["说", "声音", "音乐", "audio", "speech"]):
            new_focus = "audio"
            intents.append("audio_understanding")

        # 细节相关
        if any(kw in response_lower for kw in ["细节", "detail", "仔细", "具体"]):
            new_focus = "detail"
            intents.append("detail_analysis")

        return new_focus, intents

    def _analyze_query(self, query: str) -> Tuple[FocusType, List[str]]:
        """分析查询确定 focus 类型"""
        query_lower = query.lower()

        # 时间定位
        if any(kw in query_lower for kw in ["什么时候", "when", "何时", "出现", "发生"]):
            return FocusType.TEMPORAL, ["temporal_query"]

        # 对比
        if any(kw in query_lower for kw in ["变化", "区别", "不同", "对比", "compare"]):
            return FocusType.TEMPORAL, ["comparison_query"]

        # 音频
        if any(kw in query_lower for kw in ["说了", "声音", "音频", "听到", "audio"]):
            return FocusType.AUDIO, ["audio_query"]

        # 总结
        if any(kw in query_lower for kw in ["总结", "概括", "summary", "主要"]):
            return FocusType.GLOBAL, ["summary_query"]

        # 细节
        if any(kw in query_lower for kw in ["细节", "具体", "详细", "detail"]):
            return FocusType.DETAIL, ["detail_query"]

        return FocusType.GLOBAL, ["general_query"]

    def _get_focus_params(self, query: str, signature: VideoThoughtSignature) -> Dict:
        """获取 focus 参数"""
        params = {}

        # 如果查询提到特定时间
        import re
        time_match = re.search(r'(\d+(?:\.\d+)?)\s*[秒s]', query)
        if time_match:
            target_time = float(time_match.group(1))
            params["time_range"] = (max(0, target_time - 2), min(signature.video_duration, target_time + 2))

        return params

    def _build_enhanced_context(self, signature: VideoThoughtSignature, query: str) -> str:
        """构建增强上下文"""
        parts = [signature.to_context_string()]

        parts.append(f"\n## Current Query")
        parts.append(query)

        if signature.iteration > 1:
            parts.append(f"\n## Analysis History")
            parts.append(f"This is iteration {signature.iteration} of our analysis.")
            parts.append(f"Previous questions: {signature.user_questions[:-1]}")

        return "\n".join(parts)

    # 关键帧选择策略
    def _select_uniform(self, T: int, count: int) -> List[int]:
        """均匀选择"""
        step = max(1, T // count)
        return [min(i * step, T - 1) for i in range(count)]

    def _select_by_change(self, change_profile: List[float], count: int) -> List[int]:
        """按变化分数选择"""
        scored = [(i, s) for i, s in enumerate(change_profile)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 确保时间分散
        selected = []
        min_gap = len(change_profile) // (count * 2)

        for idx, score in scored:
            if len(selected) >= count:
                break
            if not selected or all(abs(idx - s) >= min_gap for s in selected):
                selected.append(idx)

        return sorted(selected)

    def _select_in_time_range(self, T: int, time_range: Tuple[float, float], duration: float, count: int) -> List[int]:
        """在时间范围内选择"""
        start_idx = int(time_range[0] / duration * T)
        end_idx = int(time_range[1] / duration * T)
        range_size = end_idx - start_idx

        if range_size <= 0:
            return self._select_uniform(T, count)

        step = max(1, range_size // count)
        return [min(start_idx + i * step, end_idx - 1) for i in range(count)]

    def _select_stable_frames(self, change_profile: List[float], count: int) -> List[int]:
        """选择稳定帧 (变化小)"""
        scored = [(i, s) for i, s in enumerate(change_profile)]
        scored.sort(key=lambda x: x[1])  # 升序，变化最小的在前

        return sorted([s[0] for s in scored[:count]])

    def _select_unique_frames(self, embeddings: torch.Tensor, global_emb: torch.Tensor, count: int) -> List[int]:
        """选择独特帧 (与全局差异大)"""
        T = embeddings.shape[0]
        distances = []

        for i in range(T):
            dist = 1 - F.cosine_similarity(embeddings[i].unsqueeze(0), global_emb.unsqueeze(0)).item()
            distances.append((i, dist))

        distances.sort(key=lambda x: x[1], reverse=True)
        return sorted([d[0] for d in distances[:count]])

    def _find_active_regions(self, change_profile: List[float], duration: float) -> List[Tuple[float, float]]:
        """找到活跃区域"""
        T = len(change_profile)
        threshold = np.mean(change_profile) + np.std(change_profile)

        regions = []
        in_region = False
        start = 0

        for i, score in enumerate(change_profile):
            if score > threshold and not in_region:
                start = i
                in_region = True
            elif score <= threshold and in_region:
                start_time = start / T * duration
                end_time = i / T * duration
                regions.append((start_time, end_time))
                in_region = False

        if in_region:
            regions.append((start / T * duration, duration))

        return regions

    def _compute_attention_weights(self, embeddings: torch.Tensor, selected_indices: List[int]) -> torch.Tensor:
        """计算注意力权重"""
        T = embeddings.shape[0]
        weights = torch.zeros(T)

        for idx in selected_indices:
            # 高斯分布权重
            for i in range(T):
                dist = abs(i - idx)
                weights[i] += np.exp(-(dist ** 2) / (2 * 3 ** 2))  # sigma=3

        weights = weights / weights.sum()
        return weights


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("视频 Thought Signature 系统测试")
    print("=" * 60)

    # 模拟帧数据
    import numpy as np
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(30)]

    # 初始化
    system = IterativeVideoUnderstanding.__new__(IterativeVideoUnderstanding)

    # 模拟 encoder
    class MockEncoder:
        num_frames = 16
        def encode_frames(self, frames):
            T = len(frames)
            # 模拟有变化的 embedding
            embs = []
            base = torch.randn(1024)
            for i in range(T):
                if i < 10:
                    embs.append(base + torch.randn(1024) * 0.1)
                elif i < 20:
                    embs.append(torch.randn(1024))  # 变化区域
                else:
                    embs.append(base + torch.randn(1024) * 0.1)
            return torch.stack(embs)

    system.encoder = MockEncoder()
    system.signatures = {}

    # 创建初始 Signature
    sig = system.create_signature("test_video.mp4", frames, 6.0)
    print(f"\n初始 Signature:")
    print(f"  Activity: {sig.activity_level}")
    print(f"  Temporal: {sig.temporal_understanding}")

    # 模拟 Gemma 响应
    sig = system.update_with_gemma_understanding(
        sig,
        "视频显示了一个工作场景，有人在使用电脑。中间部分出现了明显的动作。",
        "visual"
    )

    # 用户查询
    sig, indices, context = system.process_user_query(
        sig,
        "什么时候出现了动作变化？",
        frames
    )

    print(f"\n处理查询后:")
    print(f"  Focus: {sig.current_focus}")
    print(f"  Keyframe times: {sig.keyframe_times}")
    print(f"  Intents: {sig.inferred_intents}")
    print(f"  Depth: {sig.understanding_depth:.0%}")

    print("\n" + "=" * 60)
    print("上下文预览:")
    print("=" * 60)
    print(context[:500] + "..." if len(context) > 500 else context)

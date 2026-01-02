"""
意图驱动的 V-JEPA2 视频理解

核心思想: 用户意图决定 V-JEPA2 的使用策略
- 不同意图 → 不同的 embedding 处理方式
- 不同意图 → 不同的关键帧选择策略
- 不同意图 → 不同的时序分析粒度
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa2.vjepa2_encoder import VJEPA2Encoder
from vjepa2.smart_processor import UserIntent, IntentClassifier


# ============================================================
# 意图驱动的 V-JEPA2 策略
# ============================================================

class VJEPA2Strategy(ABC):
    """V-JEPA2 策略基类"""

    @abstractmethod
    def process_embeddings(
        self,
        embeddings: torch.Tensor,
        frame_count: int,
        video_duration: float
    ) -> Dict:
        """处理 embeddings，返回策略特定的结果"""
        pass

    @abstractmethod
    def select_keyframes(
        self,
        embeddings: torch.Tensor,
        target_count: int,
        video_duration: float
    ) -> Tuple[List[int], List[float], Dict]:
        """选择关键帧，返回 (indices, scores, metadata)"""
        pass


class DescribeStrategy(VJEPA2Strategy):
    """
    描述策略: 均匀覆盖 + 变化检测

    目标: 捕捉视频的完整时间线
    """

    def process_embeddings(self, embeddings: torch.Tensor, frame_count: int, video_duration: float) -> Dict:
        T, D = embeddings.shape

        # 计算变化分数
        change_scores = [0.0]
        for i in range(1, T):
            sim = F.cosine_similarity(embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0))
            change_scores.append(max(0.0, 1.0 - sim.item()))

        # 计算全局语义 (平均池化)
        global_embedding = embeddings.mean(dim=0)

        # 找到语义中心 (最接近平均的帧)
        distances = []
        for i in range(T):
            dist = 1 - F.cosine_similarity(embeddings[i].unsqueeze(0), global_embedding.unsqueeze(0)).item()
            distances.append(dist)
        semantic_center_idx = int(np.argmin(distances))

        return {
            "change_scores": change_scores,
            "global_embedding": global_embedding,
            "semantic_center_idx": semantic_center_idx,
            "avg_change": np.mean(change_scores),
            "max_change": np.max(change_scores),
        }

    def select_keyframes(self, embeddings: torch.Tensor, target_count: int, video_duration: float) -> Tuple[List[int], List[float], Dict]:
        T = embeddings.shape[0]
        result = self.process_embeddings(embeddings, T, video_duration)
        change_scores = result["change_scores"]

        # 策略: 均匀时间分段 + 每段选变化最大的
        segment_size = max(1, T // target_count)
        selected = []

        for seg in range(target_count):
            start = seg * segment_size
            end = min((seg + 1) * segment_size, T)

            # 找这个时间段内变化最大的
            best_idx = start
            best_score = 0
            for i in range(start, end):
                if change_scores[i] > best_score:
                    best_score = change_scores[i]
                    best_idx = i

            selected.append((best_idx, best_score))

        # 确保首帧
        if selected and selected[0][0] != 0:
            selected[0] = (0, change_scores[0])

        indices = [x[0] for x in selected]
        scores = [x[1] for x in selected]

        return indices, scores, {"strategy": "describe", "coverage": "uniform_with_change"}


class LocateStrategy(VJEPA2Strategy):
    """
    定位策略: 密集采样 + 变化峰值检测

    目标: 找到特定内容出现的时刻
    """

    def process_embeddings(self, embeddings: torch.Tensor, frame_count: int, video_duration: float) -> Dict:
        T, D = embeddings.shape

        # 密集的变化分数
        change_scores = [0.0]
        for i in range(1, T):
            sim = F.cosine_similarity(embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0))
            change_scores.append(max(0.0, 1.0 - sim.item()))

        # 检测变化峰值
        peaks = []
        for i in range(1, T - 1):
            if change_scores[i] > change_scores[i-1] and change_scores[i] > change_scores[i+1]:
                if change_scores[i] > np.mean(change_scores) * 0.5:  # 超过平均值一半
                    peaks.append((i, change_scores[i]))

        peaks.sort(key=lambda x: x[1], reverse=True)

        # 计算每帧与其他帧的差异度 (用于定位"独特"内容)
        uniqueness = []
        for i in range(T):
            avg_sim = 0
            for j in range(T):
                if i != j:
                    avg_sim += F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            avg_sim /= (T - 1)
            uniqueness.append(1 - avg_sim)  # 独特性 = 1 - 平均相似度

        return {
            "change_scores": change_scores,
            "peaks": peaks,
            "uniqueness": uniqueness,
            "peak_times": [p[0] / T * video_duration for p in peaks[:5]],
        }

    def select_keyframes(self, embeddings: torch.Tensor, target_count: int, video_duration: float) -> Tuple[List[int], List[float], Dict]:
        T = embeddings.shape[0]
        result = self.process_embeddings(embeddings, T, video_duration)

        # 策略: 变化峰值 + 独特性高的帧
        peaks = result["peaks"][:target_count // 2]
        uniqueness = result["uniqueness"]

        # 结合峰值和独特性
        combined_scores = []
        for i in range(T):
            peak_score = result["change_scores"][i]
            unique_score = uniqueness[i]
            combined_scores.append(0.6 * peak_score + 0.4 * unique_score)

        # 选择得分最高的帧 (确保时间分散)
        min_interval = max(1, T // (target_count * 2))
        selected = []
        used = set()

        scored = [(i, combined_scores[i]) for i in range(T)]
        scored.sort(key=lambda x: x[1], reverse=True)

        for idx, score in scored:
            if len(selected) >= target_count:
                break

            # 检查最小间隔
            too_close = False
            for prev_idx, _ in selected:
                if abs(idx - prev_idx) < min_interval:
                    too_close = True
                    break

            if not too_close:
                selected.append((idx, score))

        selected.sort(key=lambda x: x[0])
        indices = [x[0] for x in selected]
        scores = [x[1] for x in selected]

        return indices, scores, {"strategy": "locate", "peaks_found": len(peaks)}


class CompareStrategy(VJEPA2Strategy):
    """
    对比策略: 首尾对比 + 差异分析

    目标: 分析视频前后的变化
    """

    def process_embeddings(self, embeddings: torch.Tensor, frame_count: int, video_duration: float) -> Dict:
        T, D = embeddings.shape

        # 分成三段: 开始、中间、结束
        third = T // 3
        start_emb = embeddings[:third].mean(dim=0)
        mid_emb = embeddings[third:2*third].mean(dim=0)
        end_emb = embeddings[2*third:].mean(dim=0)

        # 计算段间差异
        start_mid_diff = 1 - F.cosine_similarity(start_emb.unsqueeze(0), mid_emb.unsqueeze(0)).item()
        mid_end_diff = 1 - F.cosine_similarity(mid_emb.unsqueeze(0), end_emb.unsqueeze(0)).item()
        start_end_diff = 1 - F.cosine_similarity(start_emb.unsqueeze(0), end_emb.unsqueeze(0)).item()

        # 找到变化最大的时间点
        change_scores = []
        for i in range(1, T):
            sim = F.cosine_similarity(embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0))
            change_scores.append((i, max(0.0, 1.0 - sim.item())))

        change_scores.sort(key=lambda x: x[1], reverse=True)
        transition_point = change_scores[0][0] if change_scores else T // 2

        return {
            "start_embedding": start_emb,
            "mid_embedding": mid_emb,
            "end_embedding": end_emb,
            "start_mid_diff": start_mid_diff,
            "mid_end_diff": mid_end_diff,
            "start_end_diff": start_end_diff,
            "transition_point": transition_point,
            "total_change": start_end_diff,
        }

    def select_keyframes(self, embeddings: torch.Tensor, target_count: int, video_duration: float) -> Tuple[List[int], List[float], Dict]:
        T = embeddings.shape[0]
        result = self.process_embeddings(embeddings, T, video_duration)

        # 策略: 开始 + 转折点 + 结束
        transition = result["transition_point"]

        if target_count >= 4:
            # 开始、1/4处、转折、3/4处、结束
            indices = [
                0,
                T // 4,
                transition,
                3 * T // 4,
                T - 1
            ][:target_count]
        else:
            # 简化: 开始、转折、结束
            indices = [0, transition, T - 1][:target_count]

        # 去重
        indices = list(dict.fromkeys(indices))

        # 计算每个点的"对比重要性"
        scores = []
        for idx in indices:
            if idx <= T // 3:
                score = result["start_mid_diff"]
            elif idx >= 2 * T // 3:
                score = result["mid_end_diff"]
            else:
                score = max(result["start_mid_diff"], result["mid_end_diff"])
            scores.append(score)

        return indices, scores, {
            "strategy": "compare",
            "total_change": result["total_change"],
            "transition_time": transition / T * video_duration
        }


class SummarizeStrategy(VJEPA2Strategy):
    """
    总结策略: 全局语义 + 代表帧

    目标: 找到最能代表视频整体的帧
    """

    def process_embeddings(self, embeddings: torch.Tensor, frame_count: int, video_duration: float) -> Dict:
        T, D = embeddings.shape

        # 全局语义 (加权平均，中间帧权重更高)
        weights = torch.zeros(T)
        for i in range(T):
            # 高斯权重，中心权重最高
            weights[i] = np.exp(-((i - T/2) / (T/4)) ** 2)
        weights = weights / weights.sum()

        global_embedding = (embeddings * weights.unsqueeze(1).to(embeddings.device)).sum(dim=0)

        # 找到最具代表性的帧 (最接近全局语义)
        similarities = []
        for i in range(T):
            sim = F.cosine_similarity(embeddings[i].unsqueeze(0), global_embedding.unsqueeze(0)).item()
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        most_representative = [s[0] for s in similarities[:5]]

        # 计算语义多样性 (用于确保摘要帧覆盖不同内容)
        diversity_clusters = self._cluster_embeddings(embeddings, n_clusters=min(5, T // 3))

        return {
            "global_embedding": global_embedding,
            "representative_frames": most_representative,
            "similarities": [s[1] for s in similarities],
            "diversity_clusters": diversity_clusters,
        }

    def _cluster_embeddings(self, embeddings: torch.Tensor, n_clusters: int) -> List[int]:
        """简单的聚类，返回每个簇的代表帧"""
        T = embeddings.shape[0]
        if n_clusters >= T:
            return list(range(T))

        # 简单的 K-means 风格聚类
        cluster_size = T // n_clusters
        representatives = []

        for i in range(n_clusters):
            start = i * cluster_size
            end = (i + 1) * cluster_size if i < n_clusters - 1 else T
            cluster_embs = embeddings[start:end]

            # 找簇内中心
            cluster_center = cluster_embs.mean(dim=0)
            best_idx = start
            best_sim = -1
            for j in range(start, end):
                sim = F.cosine_similarity(embeddings[j].unsqueeze(0), cluster_center.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            representatives.append(best_idx)

        return representatives

    def select_keyframes(self, embeddings: torch.Tensor, target_count: int, video_duration: float) -> Tuple[List[int], List[float], Dict]:
        T = embeddings.shape[0]
        result = self.process_embeddings(embeddings, T, video_duration)

        # 策略: 代表性 + 多样性
        representatives = result["representative_frames"][:target_count // 2]
        clusters = result["diversity_clusters"][:target_count // 2]

        # 合并并去重
        all_candidates = list(dict.fromkeys(representatives + clusters))

        if len(all_candidates) < target_count:
            # 补充: 均匀采样
            step = T // (target_count - len(all_candidates) + 1)
            for i in range(0, T, step):
                if i not in all_candidates:
                    all_candidates.append(i)
                if len(all_candidates) >= target_count:
                    break

        indices = sorted(all_candidates[:target_count])
        scores = [result["similarities"][i] if i < len(result["similarities"]) else 0.5 for i in indices]

        return indices, scores, {
            "strategy": "summarize",
            "representativeness": np.mean(scores)
        }


class AudioFocusStrategy(VJEPA2Strategy):
    """
    音频聚焦策略: 最少视觉帧 + 标记音频活动区域

    目标: 关注音频内容，视觉作为辅助
    """

    def process_embeddings(self, embeddings: torch.Tensor, frame_count: int, video_duration: float) -> Dict:
        T, D = embeddings.shape

        # 计算视觉稳定区域 (变化小的区域更适合关注音频)
        change_scores = [0.0]
        for i in range(1, T):
            sim = F.cosine_similarity(embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0))
            change_scores.append(max(0.0, 1.0 - sim.item()))

        # 找到稳定区域 (变化分数低于平均值)
        avg_change = np.mean(change_scores)
        stable_regions = []
        in_stable = False
        start = 0

        for i, score in enumerate(change_scores):
            if score < avg_change * 0.5:
                if not in_stable:
                    start = i
                    in_stable = True
            else:
                if in_stable:
                    stable_regions.append((start, i - 1))
                    in_stable = False

        if in_stable:
            stable_regions.append((start, T - 1))

        # 全局视觉摘要
        global_embedding = embeddings.mean(dim=0)

        return {
            "change_scores": change_scores,
            "stable_regions": stable_regions,
            "global_embedding": global_embedding,
            "visual_stability": 1 - np.mean(change_scores),
        }

    def select_keyframes(self, embeddings: torch.Tensor, target_count: int, video_duration: float) -> Tuple[List[int], List[float], Dict]:
        T = embeddings.shape[0]
        result = self.process_embeddings(embeddings, T, video_duration)

        # 策略: 最少帧，覆盖稳定区域
        stable_regions = result["stable_regions"]

        # 每个稳定区域选一帧 (区域中心)
        indices = []
        for start, end in stable_regions[:target_count]:
            center = (start + end) // 2
            indices.append(center)

        # 如果稳定区域不够，均匀采样补充
        if len(indices) < target_count:
            step = T // (target_count - len(indices) + 1)
            for i in range(0, T, step):
                if i not in indices:
                    indices.append(i)
                if len(indices) >= target_count:
                    break

        indices = sorted(set(indices))[:target_count]
        scores = [1 - result["change_scores"][i] for i in indices]  # 稳定性分数

        return indices, scores, {
            "strategy": "audio_focus",
            "visual_stability": result["visual_stability"],
            "stable_regions_count": len(stable_regions)
        }


# ============================================================
# 意图到策略的映射
# ============================================================

class IntentDrivenVJEPA2:
    """
    意图驱动的 V-JEPA2 处理器

    根据用户意图选择最优的 V-JEPA2 使用策略
    """

    STRATEGY_MAP = {
        UserIntent.DESCRIBE: DescribeStrategy(),
        UserIntent.SUMMARIZE: SummarizeStrategy(),
        UserIntent.LOCATE: LocateStrategy(),
        UserIntent.COMPARE: CompareStrategy(),
        UserIntent.COUNT: LocateStrategy(),  # 计数需要找到所有实例
        UserIntent.EXPLAIN: DescribeStrategy(),  # 解释需要完整上下文
        UserIntent.AUDIO_FOCUS: AudioFocusStrategy(),
        UserIntent.TRANSCRIBE: AudioFocusStrategy(),
        UserIntent.GENERAL: DescribeStrategy(),
    }

    def __init__(
        self,
        vjepa_model_size: str = "L",
        device: Optional[str] = None
    ):
        print("初始化意图驱动的 V-JEPA2 处理器...")
        self.encoder = VJEPA2Encoder(
            model_size=vjepa_model_size,
            device=device
        )
        print("初始化完成!")

    def process(
        self,
        frames: List[np.ndarray],
        user_query: str,
        video_duration: float,
        target_keyframes: int = 5
    ) -> Dict:
        """
        根据用户意图处理视频帧

        Returns:
            {
                "intent": UserIntent,
                "strategy": str,
                "keyframe_indices": List[int],
                "keyframe_scores": List[float],
                "embeddings": torch.Tensor,
                "analysis": Dict,  # 策略特定的分析结果
            }
        """
        # 1. 意图分类
        intent, confidence = IntentClassifier.classify(user_query)
        print(f"  意图: {intent.value} (置信度: {confidence:.2f})")

        # 2. 选择策略
        strategy = self.STRATEGY_MAP.get(intent, DescribeStrategy())
        print(f"  策略: {strategy.__class__.__name__}")

        # 3. V-JEPA2 编码
        embeddings = self._batch_encode(frames)
        print(f"  Embeddings: {embeddings.shape}")

        # 4. 使用策略处理
        analysis = strategy.process_embeddings(embeddings, len(frames), video_duration)
        indices, scores, metadata = strategy.select_keyframes(embeddings, target_keyframes, video_duration)

        # 5. 映射回原始帧索引
        frame_indices = self._map_to_frame_indices(indices, len(embeddings), len(frames))

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "strategy": metadata.get("strategy", "unknown"),
            "keyframe_indices": frame_indices,
            "keyframe_scores": scores,
            "embeddings": embeddings,
            "analysis": {**analysis, **metadata},
        }

    def _batch_encode(self, frames: List[np.ndarray]) -> torch.Tensor:
        """分批编码"""
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

    def _map_to_frame_indices(
        self,
        emb_indices: List[int],
        emb_count: int,
        frame_count: int
    ) -> List[int]:
        """将 embedding 索引映射回帧索引"""
        ratio = frame_count / emb_count
        frame_indices = [min(int(idx * ratio), frame_count - 1) for idx in emb_indices]
        return frame_indices

    def get_strategy_for_intent(self, intent: UserIntent) -> VJEPA2Strategy:
        """获取指定意图的策略"""
        return self.STRATEGY_MAP.get(intent, DescribeStrategy())


# ============================================================
# 使用示例
# ============================================================

def demo():
    """演示不同意图下的策略差异"""
    print("=" * 60)
    print("意图驱动的 V-JEPA2 演示")
    print("=" * 60)

    # 模拟 embeddings
    T, D = 30, 1024
    torch.manual_seed(42)

    # 模拟一个有三个阶段的视频
    embeddings = torch.zeros(T, D)
    base1 = torch.randn(D)
    base2 = torch.randn(D)
    base3 = torch.randn(D)

    for i in range(T):
        if i < 10:
            embeddings[i] = base1 + torch.randn(D) * 0.1
        elif i < 20:
            # 渐变过渡
            alpha = (i - 10) / 10
            embeddings[i] = base1 * (1 - alpha) + base2 * alpha + torch.randn(D) * 0.1
        else:
            embeddings[i] = base3 + torch.randn(D) * 0.1

    print("\n模拟视频: 30 embeddings, 3个阶段")

    # 测试不同策略
    strategies = {
        "describe": DescribeStrategy(),
        "summarize": SummarizeStrategy(),
        "locate": LocateStrategy(),
        "compare": CompareStrategy(),
        "audio_focus": AudioFocusStrategy(),
    }

    for name, strategy in strategies.items():
        print(f"\n--- {name.upper()} 策略 ---")
        indices, scores, metadata = strategy.select_keyframes(embeddings, target_count=5, video_duration=6.0)
        print(f"  关键帧索引: {indices}")
        print(f"  关键帧分数: {[f'{s:.3f}' for s in scores]}")
        print(f"  元数据: {metadata}")


if __name__ == "__main__":
    demo()

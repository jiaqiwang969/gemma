"""
语义变化检测模块

功能:
1. 基于 V-JEPA2 embedding 计算帧间变化
2. 自适应阈值调整
3. 关键帧选择策略

核心思想:
- embedding 相似度高 → 画面无变化 → 跳过
- embedding 相似度低 → 画面有变化 → 标记为关键帧
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class FrameInfo:
    """帧信息"""
    index: int              # 原始帧索引
    timestamp: float        # 时间戳 (秒)
    embedding: torch.Tensor # V-JEPA2 embedding
    change_score: float     # 变化分数 (0-1)
    is_keyframe: bool       # 是否为关键帧


@dataclass
class ChangeEvent:
    """变化事件"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    intensity: float  # 变化强度
    description: str  # 变化描述


class SemanticChangeDetector:
    """
    语义变化检测器

    使用 V-JEPA2 embedding 计算帧间语义差异，
    实现"语义驱动抽帧"。
    """

    def __init__(
        self,
        # 阈值参数
        base_threshold: float = 0.15,       # 基础变化阈值
        adaptive_window: int = 30,          # 自适应窗口大小
        adaptive_factor: float = 1.5,       # 自适应倍数
        # 关键帧参数
        min_keyframe_interval: int = 3,     # 最小关键帧间隔
        max_keyframe_interval: int = 30,    # 最大关键帧间隔 (强制抽帧)
        max_keyframes_per_window: int = 5,  # 每窗口最大关键帧数
        # 平滑参数
        smooth_window: int = 3,             # 变化分数平滑窗口
        # 距离度量
        distance_metric: str = "cosine",    # "cosine" 或 "l2"
    ):
        self.base_threshold = base_threshold
        self.adaptive_window = adaptive_window
        self.adaptive_factor = adaptive_factor
        self.min_keyframe_interval = min_keyframe_interval
        self.max_keyframe_interval = max_keyframe_interval
        self.max_keyframes_per_window = max_keyframes_per_window
        self.smooth_window = smooth_window
        self.distance_metric = distance_metric

        # 历史记录 (用于自适应阈值)
        self.change_history = deque(maxlen=adaptive_window)

        # 统计信息
        self.stats = {
            "total_frames": 0,
            "keyframes": 0,
            "avg_change_score": 0,
            "max_change_score": 0,
        }

    def compute_change_score(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> float:
        """
        计算两个 embedding 之间的变化分数

        Args:
            emb1: [D] 前一帧 embedding
            emb2: [D] 当前帧 embedding

        Returns:
            change_score: 0-1 之间的变化分数
        """
        if self.distance_metric == "cosine":
            # Cosine distance: 1 - cosine_similarity
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            change = 1.0 - similarity.item()
        else:  # l2
            # L2 distance (归一化)
            dist = torch.norm(emb1 - emb2, p=2)
            # 使用 sigmoid 归一化到 0-1
            change = torch.sigmoid(dist / 10).item()

        return max(0.0, min(1.0, change))

    def get_adaptive_threshold(self) -> float:
        """
        获取自适应阈值

        基于历史变化分数动态调整阈值，
        使得在静止场景中更敏感，在动态场景中更宽松。
        """
        if len(self.change_history) < 5:
            return self.base_threshold

        history = list(self.change_history)
        mean_change = np.mean(history)
        std_change = np.std(history)

        # 自适应阈值 = 均值 + 标准差 * 系数
        adaptive_threshold = mean_change + std_change * self.adaptive_factor

        # 限制在合理范围内
        return max(
            self.base_threshold * 0.5,
            min(self.base_threshold * 2.0, adaptive_threshold)
        )

    def detect_changes(
        self,
        embeddings: torch.Tensor,
        fps: float = 5.0
    ) -> Tuple[List[FrameInfo], List[int]]:
        """
        检测视频帧中的变化，返回帧信息和关键帧索引

        Args:
            embeddings: [T, D] 帧 embedding 序列
            fps: 帧率 (用于计算时间戳)

        Returns:
            frame_infos: 所有帧的信息列表
            keyframe_indices: 关键帧的索引列表
        """
        T, D = embeddings.shape
        frame_infos = []
        keyframe_indices = []

        # 变化分数序列
        change_scores = []

        # 计算每帧的变化分数
        for i in range(T):
            if i == 0:
                change_score = 0.0
            else:
                change_score = self.compute_change_score(
                    embeddings[i - 1], embeddings[i]
                )

            change_scores.append(change_score)
            self.change_history.append(change_score)

        # 平滑变化分数
        smoothed_scores = self._smooth_scores(change_scores)

        # 获取自适应阈值
        threshold = self.get_adaptive_threshold()

        # 选择关键帧
        last_keyframe_idx = -self.max_keyframe_interval

        for i in range(T):
            timestamp = i / fps
            is_keyframe = False

            # 判断是否为关键帧
            frames_since_last = i - last_keyframe_idx

            if i == 0:
                # 第一帧总是关键帧
                is_keyframe = True
            elif frames_since_last >= self.max_keyframe_interval:
                # 超过最大间隔，强制抽帧
                is_keyframe = True
            elif (frames_since_last >= self.min_keyframe_interval and
                  smoothed_scores[i] > threshold):
                # 变化超过阈值且满足最小间隔
                is_keyframe = True

            if is_keyframe:
                keyframe_indices.append(i)
                last_keyframe_idx = i

            frame_info = FrameInfo(
                index=i,
                timestamp=timestamp,
                embedding=embeddings[i],
                change_score=smoothed_scores[i],
                is_keyframe=is_keyframe
            )
            frame_infos.append(frame_info)

        # 更新统计
        self.stats["total_frames"] = T
        self.stats["keyframes"] = len(keyframe_indices)
        self.stats["avg_change_score"] = np.mean(smoothed_scores)
        self.stats["max_change_score"] = np.max(smoothed_scores)

        return frame_infos, keyframe_indices

    def _smooth_scores(self, scores: List[float]) -> List[float]:
        """平滑变化分数"""
        if self.smooth_window <= 1:
            return scores

        smoothed = []
        half_window = self.smooth_window // 2

        for i in range(len(scores)):
            start = max(0, i - half_window)
            end = min(len(scores), i + half_window + 1)
            smoothed.append(np.mean(scores[start:end]))

        return smoothed

    def detect_events(
        self,
        frame_infos: List[FrameInfo],
        event_threshold: float = 0.3,
        min_event_duration: int = 3
    ) -> List[ChangeEvent]:
        """
        检测变化事件 (连续高变化的时间段)

        Args:
            frame_infos: 帧信息列表
            event_threshold: 事件检测阈值
            min_event_duration: 最小事件持续帧数

        Returns:
            events: 变化事件列表
        """
        events = []
        in_event = False
        event_start = 0
        event_scores = []

        for i, frame in enumerate(frame_infos):
            if frame.change_score > event_threshold:
                if not in_event:
                    in_event = True
                    event_start = i
                    event_scores = []
                event_scores.append(frame.change_score)
            else:
                if in_event:
                    # 事件结束
                    event_duration = i - event_start
                    if event_duration >= min_event_duration:
                        events.append(ChangeEvent(
                            start_frame=event_start,
                            end_frame=i - 1,
                            start_time=frame_infos[event_start].timestamp,
                            end_time=frame_infos[i - 1].timestamp,
                            intensity=np.mean(event_scores),
                            description=self._describe_event(np.mean(event_scores))
                        ))
                    in_event = False

        # 处理末尾的事件
        if in_event and len(event_scores) >= min_event_duration:
            events.append(ChangeEvent(
                start_frame=event_start,
                end_frame=len(frame_infos) - 1,
                start_time=frame_infos[event_start].timestamp,
                end_time=frame_infos[-1].timestamp,
                intensity=np.mean(event_scores),
                description=self._describe_event(np.mean(event_scores))
            ))

        return events

    def _describe_event(self, intensity: float) -> str:
        """根据变化强度描述事件"""
        if intensity > 0.7:
            return "剧烈变化 (场景切换/突发动作)"
        elif intensity > 0.5:
            return "显著变化 (主体运动)"
        elif intensity > 0.3:
            return "中等变化 (局部运动)"
        else:
            return "轻微变化"

    def get_stats(self) -> Dict:
        """获取检测统计信息"""
        compression_ratio = (
            self.stats["keyframes"] / self.stats["total_frames"]
            if self.stats["total_frames"] > 0 else 0
        )
        return {
            **self.stats,
            "compression_ratio": f"{compression_ratio:.1%}",
            "current_threshold": self.get_adaptive_threshold()
        }


class KeyframeExtractor:
    """
    关键帧提取器

    结合 V-JEPA2 编码器和变化检测器，
    从视频中提取关键帧。
    """

    def __init__(
        self,
        encoder,  # VJEPA2Encoder 实例
        detector: Optional[SemanticChangeDetector] = None,
        target_keyframes: int = 5,  # 目标关键帧数量
    ):
        self.encoder = encoder
        self.detector = detector or SemanticChangeDetector()
        self.target_keyframes = target_keyframes

    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 5.0
    ) -> Tuple[List[np.ndarray], List[FrameInfo]]:
        """
        从帧列表中提取关键帧

        Args:
            frames: 原始帧列表 [H, W, C] uint8
            fps: 帧率

        Returns:
            keyframes: 关键帧列表
            frame_infos: 所有帧的信息
        """
        # 编码所有帧
        embeddings = self.encoder.encode_frames(frames)

        # 检测变化
        frame_infos, keyframe_indices = self.detector.detect_changes(embeddings, fps)

        # 如果关键帧太多，进行二次筛选
        if len(keyframe_indices) > self.target_keyframes:
            keyframe_indices = self._select_top_keyframes(
                frame_infos, keyframe_indices, self.target_keyframes
            )

        # 提取关键帧
        keyframes = [frames[i] for i in keyframe_indices]

        return keyframes, frame_infos

    def extract_from_video(
        self,
        video_path: str,
        sample_fps: float = 5.0
    ) -> Tuple[List[np.ndarray], List[FrameInfo], Dict]:
        """
        从视频文件中提取关键帧

        Args:
            video_path: 视频文件路径
            sample_fps: 采样帧率

        Returns:
            keyframes: 关键帧列表
            frame_infos: 所有帧的信息
            stats: 统计信息
        """
        # 加载视频帧
        frames = self.encoder._load_video(video_path, int(sample_fps))

        if not frames:
            return [], [], {}

        # 提取关键帧
        keyframes, frame_infos = self.extract_from_frames(frames, sample_fps)

        # 获取统计信息
        stats = self.detector.get_stats()
        stats["video_path"] = video_path
        stats["sample_fps"] = sample_fps
        stats["original_frames"] = len(frames)
        stats["extracted_keyframes"] = len(keyframes)

        return keyframes, frame_infos, stats

    def _select_top_keyframes(
        self,
        frame_infos: List[FrameInfo],
        candidates: List[int],
        target_count: int
    ) -> List[int]:
        """
        从候选关键帧中选择 top-k

        策略: 优先选择变化分数最高的帧，
        同时保证时间分布均匀。
        """
        if len(candidates) <= target_count:
            return candidates

        # 按变化分数排序
        scored = [(i, frame_infos[i].change_score) for i in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 选择 top-k，但保证第一帧和最后一帧
        selected = set()

        # 确保第一帧
        if candidates[0] not in selected:
            selected.add(candidates[0])

        # 确保最后一帧
        if candidates[-1] not in selected:
            selected.add(candidates[-1])

        # 按分数填充剩余
        for idx, score in scored:
            if len(selected) >= target_count:
                break
            selected.add(idx)

        return sorted(list(selected))


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("语义变化检测测试")
    print("=" * 60)

    # 创建检测器
    detector = SemanticChangeDetector(
        base_threshold=0.15,
        min_keyframe_interval=2,
        max_keyframe_interval=10
    )

    # 模拟 embedding 序列 (假设有静止段和变化段)
    print("\n模拟场景: 静止 → 变化 → 静止")

    embeddings = []
    D = 1024

    # 静止段 (帧 0-9): embedding 几乎不变
    base = torch.randn(D)
    for i in range(10):
        embeddings.append(base + torch.randn(D) * 0.01)

    # 变化段 (帧 10-19): embedding 显著变化
    for i in range(10):
        embeddings.append(torch.randn(D))

    # 静止段 (帧 20-29): embedding 又稳定
    base2 = torch.randn(D)
    for i in range(10):
        embeddings.append(base2 + torch.randn(D) * 0.01)

    embeddings = torch.stack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # 检测变化
    frame_infos, keyframe_indices = detector.detect_changes(embeddings, fps=5.0)

    print(f"\n总帧数: {len(frame_infos)}")
    print(f"关键帧数: {len(keyframe_indices)}")
    print(f"关键帧索引: {keyframe_indices}")
    print(f"压缩比: {len(keyframe_indices)/len(frame_infos):.1%}")

    # 显示每帧的变化分数
    print("\n变化分数分布:")
    print("帧     | 分数  | 关键帧")
    print("-" * 30)
    for info in frame_infos:
        marker = "★" if info.is_keyframe else " "
        bar = "█" * int(info.change_score * 20)
        print(f"{info.index:3d}    | {info.change_score:.3f} | {marker} {bar}")

    # 检测事件
    events = detector.detect_events(frame_infos)
    print(f"\n检测到 {len(events)} 个变化事件:")
    for event in events:
        print(f"  帧 {event.start_frame}-{event.end_frame}: "
              f"{event.start_time:.1f}s-{event.end_time:.1f}s | "
              f"强度 {event.intensity:.2f} | {event.description}")

    # 统计信息
    print(f"\n统计信息: {detector.get_stats()}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

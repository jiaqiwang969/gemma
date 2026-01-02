#!/usr/bin/env python3
"""
AI 眼镜系统 - V-JEPA2 + Gemma 3n 协调引擎

正确实现意图-嵌入双向循环:
1. 用户意图 → V-JEPA2 提取策略
2. V-JEPA2 embedding → 分析反馈 → 增强意图
3. 意图 + embedding 分析 → 智能帧选择
4. 选中帧 + 上下文 → Gemma 回答

用法: python tests/coordinated_engine.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ============================================================
# 意图和策略定义
# ============================================================

class IntentType(Enum):
    DESCRIBE = "describe"
    LOCATE = "locate"
    COMPARE = "compare"
    COUNT = "count"
    SUMMARIZE = "summarize"
    TRACK = "track"
    GENERAL = "general"


class VJEPAStrategy(Enum):
    """V-JEPA2 提取策略"""
    FULL_ENCODE_CLUSTER = "full_encode_cluster"      # COUNT: 全帧编码+聚类
    DENSE_PEAK_DETECT = "dense_peak_detect"          # LOCATE: 密集编码+峰值检测
    START_END_CHANGE = "start_end_change"            # COMPARE: 首尾+变化点
    UNIFORM_REPRESENTATIVE = "uniform_representative" # DESCRIBE: 均匀采样+代表性
    HIGH_FREQ_CONTINUOUS = "high_freq_continuous"    # TRACK: 高频连续


# 意图 → V-JEPA2 策略映射
INTENT_TO_VJEPA_STRATEGY = {
    IntentType.COUNT: VJEPAStrategy.FULL_ENCODE_CLUSTER,
    IntentType.LOCATE: VJEPAStrategy.DENSE_PEAK_DETECT,
    IntentType.COMPARE: VJEPAStrategy.START_END_CHANGE,
    IntentType.DESCRIBE: VJEPAStrategy.UNIFORM_REPRESENTATIVE,
    IntentType.TRACK: VJEPAStrategy.HIGH_FREQ_CONTINUOUS,
    IntentType.SUMMARIZE: VJEPAStrategy.UNIFORM_REPRESENTATIVE,
    IntentType.GENERAL: VJEPAStrategy.UNIFORM_REPRESENTATIVE,
}


# ============================================================
# V-JEPA2 编码器
# ============================================================

@dataclass
class VJEPAAnalysis:
    """V-JEPA2 分析结果"""
    embeddings: List[np.ndarray]           # 所有帧的 embedding
    timestamps: List[float]                 # 时间戳
    change_scores: List[float]              # 变化分数序列
    peak_indices: List[int]                 # 变化峰值的帧索引
    peak_times: List[float]                 # 变化峰值时间
    cluster_labels: Optional[List[int]]     # 聚类标签
    cluster_centers: Optional[List[int]]    # 聚类中心帧索引
    activity_level: str                     # 活动级别
    max_change: float                       # 最大变化分数
    has_significant_change: bool            # 是否有显著变化
    semantic_context: str                   # 语义上下文描述


class VJEPAEncoder:
    """V-JEPA2 编码器 (模拟)"""

    def __init__(self, embed_dim: int = 1024):
        self.embed_dim = embed_dim

    def encode_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """编码单帧为 embedding"""
        # 基于帧内容生成 embedding
        mean_val = np.mean(frame) / 255.0
        std_val = np.std(frame) / 255.0

        # 使用帧特征作为随机种子
        seed = int((mean_val * 1000 + timestamp * 100) % 10000)
        rng = np.random.RandomState(seed)

        # 生成并归一化
        embedding = rng.randn(self.embed_dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def compute_change_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算两个 embedding 之间的变化分数"""
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return float(max(0.0, 1.0 - sim))

    def analyze(
        self,
        frames: List[Tuple[np.ndarray, float]],
        strategy: VJEPAStrategy
    ) -> VJEPAAnalysis:
        """根据策略分析视频帧"""

        # 1. 编码所有帧
        embeddings = []
        timestamps = []
        for frame, ts in frames:
            emb = self.encode_frame(frame, ts)
            embeddings.append(emb)
            timestamps.append(ts)

        # 2. 计算变化分数
        change_scores = [0.0]
        for i in range(1, len(embeddings)):
            score = self.compute_change_score(embeddings[i-1], embeddings[i])
            change_scores.append(score)

        # 3. 检测变化峰值
        mean_change = np.mean(change_scores)
        std_change = np.std(change_scores)
        threshold = mean_change + std_change

        peak_indices = []
        peak_times = []
        for i, score in enumerate(change_scores):
            if score > threshold and score > 0.1:
                peak_indices.append(i)
                peak_times.append(timestamps[i])

        # 4. 聚类分析 (COUNT 策略需要)
        cluster_labels = None
        cluster_centers = None

        if strategy == VJEPAStrategy.FULL_ENCODE_CLUSTER and len(embeddings) > 3:
            n_clusters = min(3, len(embeddings) // 3)
            if n_clusters >= 2:
                emb_array = np.array(embeddings)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(emb_array).tolist()

                # 找每个聚类的中心帧
                cluster_centers = []
                for c in range(n_clusters):
                    indices = [i for i, l in enumerate(cluster_labels) if l == c]
                    if indices:
                        center_idx = indices[len(indices) // 2]
                        cluster_centers.append(center_idx)

        # 5. 活动级别
        max_change = max(change_scores) if change_scores else 0.0
        if max_change < 0.05:
            activity_level = "static"
        elif max_change < 0.1:
            activity_level = "low"
        elif max_change < 0.2:
            activity_level = "medium"
        else:
            activity_level = "high"

        # 6. 生成语义上下文
        semantic_context = self._generate_context(
            len(frames), peak_times, cluster_labels, activity_level, timestamps
        )

        return VJEPAAnalysis(
            embeddings=embeddings,
            timestamps=timestamps,
            change_scores=change_scores,
            peak_indices=peak_indices,
            peak_times=peak_times,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            activity_level=activity_level,
            max_change=max_change,
            has_significant_change=max_change > 0.15,
            semantic_context=semantic_context
        )

    def _generate_context(
        self,
        frame_count: int,
        peak_times: List[float],
        cluster_labels: Optional[List[int]],
        activity_level: str,
        timestamps: List[float]
    ) -> str:
        """生成语义上下文描述 (给 Gemma 的提示)"""
        parts = []

        # 视频长度
        if timestamps:
            duration = timestamps[-1] - timestamps[0]
            parts.append(f"视频时长约 {duration:.0f} 秒，共 {frame_count} 帧")

        # 活动级别
        activity_desc = {
            "static": "内容非常静态，几乎没有变化",
            "low": "内容变化较小",
            "medium": "有一定程度的内容变化",
            "high": "内容变化显著"
        }
        parts.append(activity_desc.get(activity_level, ""))

        # 变化峰值
        if peak_times:
            times_str = ", ".join([f"{t:.1f}s" for t in peak_times[:3]])
            parts.append(f"在 {times_str} 处检测到显著变化")

        # 聚类信息
        if cluster_labels:
            n_clusters = len(set(cluster_labels))
            parts.append(f"内容可分为 {n_clusters} 个不同的场景片段")

        return "。".join(parts) + "。"


# ============================================================
# 意图分析器 (带反馈)
# ============================================================

@dataclass
class IntentState:
    """意图状态"""
    primary: IntentType
    confidence: float
    secondary: List[Tuple[IntentType, float]] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    iteration: int = 0


def analyze_intent(query: str) -> IntentState:
    """从查询分析意图"""
    q = query.lower()

    patterns = [
        (IntentType.COUNT, ["几", "多少", "数量", "count", "几个", "几台", "有多少"]),
        (IntentType.COMPARE, ["变化", "区别", "对比", "不同", "前后", "compare", "改变"]),
        (IntentType.LOCATE, ["哪里", "在哪", "什么时候", "where", "when", "找", "出现"]),
        (IntentType.DESCRIBE, ["描述", "是什么", "看到", "describe", "内容", "说明"]),
        (IntentType.SUMMARIZE, ["总结", "概括", "summary", "主要"]),
        (IntentType.TRACK, ["跟踪", "追踪", "一直", "track"]),
    ]

    for intent, keywords in patterns:
        matched = [kw for kw in keywords if kw in q]
        if matched:
            confidence = min(len(matched) * 0.3 + 0.5, 0.95)
            return IntentState(
                primary=intent,
                confidence=confidence,
                matched_keywords=matched
            )

    return IntentState(
        primary=IntentType.GENERAL,
        confidence=0.5,
        matched_keywords=[]
    )


def update_intent_from_analysis(
    intent: IntentState,
    analysis: VJEPAAnalysis
) -> IntentState:
    """根据 V-JEPA2 分析结果更新意图"""
    intent.iteration += 1

    # 如果检测到显著变化，建议 COMPARE
    if analysis.has_significant_change and intent.primary != IntentType.COMPARE:
        intent.secondary.append((IntentType.COMPARE, 0.4))

    # 如果有多个聚类，可能有多个物体/场景
    if analysis.cluster_labels and len(set(analysis.cluster_labels)) > 2:
        if intent.primary not in [IntentType.COUNT, IntentType.COMPARE]:
            intent.secondary.append((IntentType.COUNT, 0.3))

    # 如果有变化峰值，建议 LOCATE
    if len(analysis.peak_times) > 0 and intent.primary != IntentType.LOCATE:
        intent.secondary.append((IntentType.LOCATE, 0.3))

    # 去重并排序
    seen = {intent.primary}
    unique_secondary = []
    for i, c in sorted(intent.secondary, key=lambda x: -x[1]):
        if i not in seen:
            seen.add(i)
            unique_secondary.append((i, c))
    intent.secondary = unique_secondary[:3]

    return intent


# ============================================================
# 智能帧选择器
# ============================================================

def select_frames_for_gemma(
    frames: List[Tuple[np.ndarray, float]],
    analysis: VJEPAAnalysis,
    intent: IntentState,
    max_frames: int = 6
) -> Tuple[List[Tuple[np.ndarray, float]], str]:
    """
    根据意图和 V-JEPA2 分析智能选择帧

    Returns:
        - 选中的帧列表
        - 选择理由
    """
    strategy = INTENT_TO_VJEPA_STRATEGY[intent.primary]
    reasons = []

    if strategy == VJEPAStrategy.FULL_ENCODE_CLUSTER:
        # COUNT: 选择聚类中心 + 变化峰值
        selected_indices = set()

        if analysis.cluster_centers:
            for idx in analysis.cluster_centers:
                selected_indices.add(idx)
            reasons.append(f"选择了 {len(analysis.cluster_centers)} 个聚类中心帧")

        for idx in analysis.peak_indices[:2]:
            selected_indices.add(idx)
        if analysis.peak_indices:
            reasons.append(f"选择了 {min(2, len(analysis.peak_indices))} 个变化峰值帧")

        # 补充均匀采样
        while len(selected_indices) < max_frames and len(selected_indices) < len(frames):
            step = len(frames) // (max_frames - len(selected_indices) + 1)
            for i in range(0, len(frames), max(1, step)):
                if i not in selected_indices:
                    selected_indices.add(i)
                    if len(selected_indices) >= max_frames:
                        break

        indices = sorted(selected_indices)[:max_frames]

    elif strategy == VJEPAStrategy.START_END_CHANGE:
        # COMPARE: 首帧 + 最大变化点 + 尾帧
        indices = [0]
        reasons.append("选择首帧")

        if analysis.peak_indices:
            max_change_idx = max(analysis.peak_indices, key=lambda i: analysis.change_scores[i])
            if max_change_idx not in indices:
                indices.append(max_change_idx)
                reasons.append(f"选择最大变化点 ({analysis.timestamps[max_change_idx]:.1f}s)")

        if len(frames) > 1 and (len(frames) - 1) not in indices:
            indices.append(len(frames) - 1)
            reasons.append("选择尾帧")

        indices = sorted(indices)[:max_frames]

    elif strategy == VJEPAStrategy.DENSE_PEAK_DETECT:
        # LOCATE: 变化峰值 + 周围帧
        selected_indices = set()

        for idx in analysis.peak_indices[:3]:
            selected_indices.add(idx)
            # 添加前后帧
            if idx > 0:
                selected_indices.add(idx - 1)
            if idx < len(frames) - 1:
                selected_indices.add(idx + 1)

        if analysis.peak_indices:
            reasons.append(f"选择了 {len(analysis.peak_indices)} 个变化峰值及周围帧")

        # 补充
        if not selected_indices:
            step = max(1, len(frames) // max_frames)
            selected_indices = set(range(0, len(frames), step))
            reasons.append("无明显峰值，均匀采样")

        indices = sorted(selected_indices)[:max_frames]

    else:
        # DESCRIBE/SUMMARIZE/GENERAL: 均匀采样
        step = max(1, len(frames) // max_frames)
        indices = list(range(0, len(frames), step))[:max_frames]
        reasons.append(f"均匀采样 {len(indices)} 帧")

    selected = [frames[i] for i in indices]
    reason = "；".join(reasons)

    return selected, reason


# ============================================================
# 协调引擎
# ============================================================

class CoordinatedEngine:
    """V-JEPA2 + Gemma 协调引擎"""

    def __init__(self, step_by_step: bool = False):
        self.vjepa = VJEPAEncoder()
        self.step_by_step = step_by_step

    def process(
        self,
        frames: List[Tuple[np.ndarray, float]],
        query: str
    ) -> Dict[str, Any]:
        """执行完整的协调流程"""

        results = {
            "query": query,
            "steps": []
        }

        # ======== 步骤 1: 意图分析 ========
        self._log_step(1, "意图分析", results)
        intent = analyze_intent(query)

        results["steps"].append({
            "name": "意图分析",
            "intent": intent.primary.value,
            "confidence": intent.confidence,
            "keywords": intent.matched_keywords
        })

        self._print_step(
            1, "意图分析",
            f"意图: {intent.primary.value} (置信度: {intent.confidence:.0%})",
            f"关键词: {intent.matched_keywords}"
        )

        # ======== 步骤 2: V-JEPA2 策略选择 ========
        self._log_step(2, "V-JEPA2 策略选择", results)
        strategy = INTENT_TO_VJEPA_STRATEGY[intent.primary]

        results["steps"].append({
            "name": "V-JEPA2 策略",
            "strategy": strategy.value
        })

        self._print_step(
            2, "V-JEPA2 策略选择",
            f"策略: {strategy.value}"
        )

        # ======== 步骤 3: V-JEPA2 编码与分析 ========
        self._log_step(3, "V-JEPA2 编码与分析", results)
        analysis = self.vjepa.analyze(frames, strategy)

        results["steps"].append({
            "name": "V-JEPA2 分析",
            "frame_count": len(frames),
            "change_scores": analysis.change_scores,
            "peak_times": analysis.peak_times,
            "cluster_count": len(set(analysis.cluster_labels)) if analysis.cluster_labels else 0,
            "activity_level": analysis.activity_level,
            "max_change": analysis.max_change
        })

        self._print_step(
            3, "V-JEPA2 编码与分析",
            f"编码帧数: {len(frames)}",
            f"活动级别: {analysis.activity_level}",
            f"最大变化: {analysis.max_change:.3f}",
            f"变化峰值: {analysis.peak_times}" if analysis.peak_times else "无显著峰值",
            f"聚类数: {len(set(analysis.cluster_labels))}" if analysis.cluster_labels else "未聚类"
        )

        # ======== 步骤 4: 意图反馈更新 ========
        self._log_step(4, "意图反馈更新", results)
        intent = update_intent_from_analysis(intent, analysis)

        feedback_info = []
        if analysis.has_significant_change:
            feedback_info.append("检测到显著变化")
        if analysis.cluster_labels and len(set(analysis.cluster_labels)) > 2:
            feedback_info.append(f"发现 {len(set(analysis.cluster_labels))} 个场景聚类")
        if analysis.peak_times:
            feedback_info.append(f"发现 {len(analysis.peak_times)} 个变化峰值")

        results["steps"].append({
            "name": "意图反馈",
            "feedback": feedback_info,
            "secondary_intents": [(i.value, c) for i, c in intent.secondary],
            "iteration": intent.iteration
        })

        self._print_step(
            4, "意图反馈更新",
            f"V-JEPA2 反馈: {feedback_info if feedback_info else '无特殊发现'}",
            f"次要意图: {[(i.value, f'{c:.0%}') for i, c in intent.secondary]}" if intent.secondary else "无次要意图"
        )

        # ======== 步骤 5: 智能帧选择 ========
        self._log_step(5, "智能帧选择", results)
        selected_frames, selection_reason = select_frames_for_gemma(
            frames, analysis, intent, max_frames=6
        )

        selected_times = [f[1] for f in selected_frames]

        results["steps"].append({
            "name": "帧选择",
            "selected_count": len(selected_frames),
            "selected_times": selected_times,
            "reason": selection_reason
        })

        self._print_step(
            5, "智能帧选择",
            f"选中 {len(selected_frames)} 帧",
            f"时间点: {[f'{t:.1f}s' for t in selected_times]}",
            f"选择理由: {selection_reason}"
        )

        # ======== 步骤 6: 生成 Gemma 上下文 ========
        self._log_step(6, "生成 Gemma 上下文", results)

        gemma_context = {
            "semantic_context": analysis.semantic_context,
            "intent_hint": f"用户想要{intent.primary.value}",
            "frame_info": f"提供了 {len(selected_frames)} 张关键帧进行分析"
        }

        results["steps"].append({
            "name": "Gemma 上下文",
            "context": gemma_context
        })

        self._print_step(
            6, "生成 Gemma 上下文",
            f"语义上下文: {analysis.semantic_context}",
            f"意图提示: 用户想要{intent.primary.value}"
        )

        # ======== 步骤 7: Gemma 处理 (模拟) ========
        self._log_step(7, "Gemma 多模态分析", results)

        # 模拟 Gemma 响应
        response = self._mock_gemma_response(query, intent, analysis, selected_frames)

        results["steps"].append({
            "name": "Gemma 响应",
            "response": response
        })

        self._print_step(
            7, "Gemma 多模态分析",
            "=" * 50,
            response,
            "=" * 50
        )

        results["final_response"] = response
        results["intent"] = intent.primary.value
        results["selected_frames"] = len(selected_frames)

        return results

    def _log_step(self, num: int, name: str, results: dict):
        """记录步骤"""
        if self.step_by_step:
            input(f"\n按 Enter 继续步骤 {num}...")

    def _print_step(self, num: int, title: str, *lines: str):
        """打印步骤"""
        print(f"\n┌─ 步骤 {num}: {title} " + "─" * (50 - len(title)))
        for line in lines:
            print(f"│  {line}")
        print("└" + "─" * 60)

    def _mock_gemma_response(
        self,
        query: str,
        intent: IntentState,
        analysis: VJEPAAnalysis,
        frames: List[Tuple[np.ndarray, float]]
    ) -> str:
        """模拟 Gemma 响应"""
        q = query.lower()

        if intent.primary == IntentType.COUNT:
            if "笔记本" in q or "电脑" in q:
                cluster_info = ""
                if analysis.cluster_labels:
                    n_clusters = len(set(analysis.cluster_labels))
                    cluster_info = f"\n\nV-JEPA2 将视频分为 {n_clusters} 个场景片段，经分析这些片段显示的是同一台电脑的不同操作界面。"

                return f"""根据对 {len(frames)} 张关键帧的分析：

**笔记本电脑数量: 1 台**

判断依据：
- 所有帧显示的是同一个 macOS 界面
- 左上角的红黄绿按钮表明这是 Mac 系统
- 界面信息显示 "MPS 运行中"，表明是 Apple Silicon Mac{cluster_info}

因此，视频中只有 **1 台 MacBook 笔记本电脑**。"""

        elif intent.primary == IntentType.COMPARE:
            if analysis.peak_times:
                peak_str = ", ".join([f"{t:.1f}s" for t in analysis.peak_times[:3]])
                return f"""对比分析结果：

V-JEPA2 检测到在 {peak_str} 处有显著变化。

变化内容：
- 界面从首页切换到对话界面
- 用户进行了多轮 AI 对话
- 展示了图片分析、音频处理等功能

活动级别: {analysis.activity_level}
最大变化分数: {analysis.max_change:.3f}"""
            else:
                return f"视频整体比较静态，没有检测到显著变化。活动级别: {analysis.activity_level}"

        elif intent.primary == IntentType.DESCRIBE:
            return f"""视频内容描述：

{analysis.semantic_context}

这是一个 AI 多模态聊天系统的演示视频，展示了：
1. 本地运行的 AI 助手界面
2. 支持多轮对话
3. 图像理解能力
4. 音频处理能力

基于 {len(frames)} 张关键帧分析完成。"""

        elif intent.primary == IntentType.LOCATE:
            if analysis.peak_times:
                peak_str = ", ".join([f"{t:.1f}s" for t in analysis.peak_times[:3]])
                return f"检测到在 {peak_str} 时间点有显著变化，可能是您要找的内容出现的时刻。"
            return "未检测到明显的变化点。"

        return f"分析了 {len(frames)} 张关键帧。{analysis.semantic_context}"


# ============================================================
# 测试
# ============================================================

def generate_test_frames(scenario: str = "normal") -> List[Tuple[np.ndarray, float]]:
    """生成测试帧"""
    frames = []

    if scenario == "transition":
        # 有明显变化的场景
        for i in range(20):
            if i < 10:
                brightness = 50
            else:
                brightness = 200
            frame = np.ones((100, 100, 3), dtype=np.uint8) * brightness
            # 添加一些变化
            if 9 <= i <= 11:
                frame[:, :50, :] = 255 - brightness
            frames.append((frame, i * 0.5))
    else:
        # 正常场景
        for i in range(20):
            brightness = 100 + i * 5
            frame = np.ones((100, 100, 3), dtype=np.uint8) * brightness
            frames.append((frame, i * 0.5))

    return frames


def main():
    print("\n" + "=" * 60)
    print("  V-JEPA2 + Gemma 协调引擎测试")
    print("=" * 60)

    engine = CoordinatedEngine(step_by_step=False)

    # 测试 1: 计数查询
    print("\n" + "=" * 60)
    print("  测试 1: 计数查询")
    print("=" * 60)

    frames = generate_test_frames("normal")
    result = engine.process(frames, "这里面有几台笔记本电脑？")

    # 测试 2: 对比查询
    print("\n" + "=" * 60)
    print("  测试 2: 对比查询")
    print("=" * 60)

    frames = generate_test_frames("transition")
    result = engine.process(frames, "视频前后有什么变化？")

    # 测试 3: 描述查询
    print("\n" + "=" * 60)
    print("  测试 3: 描述查询")
    print("=" * 60)

    frames = generate_test_frames("normal")
    result = engine.process(frames, "描述一下这个视频的内容")


if __name__ == "__main__":
    main()

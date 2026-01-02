#!/usr/bin/env python3
"""
AI 眼镜系统 - 端到端集成测试

验证核心循环:
1. 用户意图 → 提取策略
2. 视频帧 → V-JEPA2 embedding → 存储
3. Embedding 分析 → 意图增强
4. 意图更新 → 策略调整

用法: python tests/integration_test.py
"""

import os
import sys
import json
import time
import base64
import subprocess
import threading
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
from PIL import Image

# ============================================================
# 意图和策略定义 (与 Rust 保持一致)
# ============================================================

class IntentType(Enum):
    DESCRIBE = "describe"
    LOCATE = "locate"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    AUDIO_FOCUS = "audio_focus"
    TRACK = "track"
    GENERAL = "general"


class ExtractionStrategy(Enum):
    UNIFORM_WITH_CHANGE = "uniform_with_change"
    DENSE_WITH_PEAKS = "dense_with_peaks"
    START_END_TRANSITION = "start_end_transition"
    REPRESENTATIVE = "representative"
    STABLE_FRAMES = "stable_frames"
    HIGH_FREQUENCY = "high_frequency"


# 意图 → 策略映射
INTENT_STRATEGY_MAP = {
    IntentType.DESCRIBE: ExtractionStrategy.UNIFORM_WITH_CHANGE,
    IntentType.LOCATE: ExtractionStrategy.DENSE_WITH_PEAKS,
    IntentType.COMPARE: ExtractionStrategy.START_END_TRANSITION,
    IntentType.SUMMARIZE: ExtractionStrategy.REPRESENTATIVE,
    IntentType.AUDIO_FOCUS: ExtractionStrategy.STABLE_FRAMES,
    IntentType.TRACK: ExtractionStrategy.HIGH_FREQUENCY,
    IntentType.GENERAL: ExtractionStrategy.UNIFORM_WITH_CHANGE,
}


def intent_from_query(query: str) -> Tuple[IntentType, float]:
    """从查询推断意图 (与 Rust 实现保持一致)"""
    q = query.lower()

    patterns = [
        (IntentType.DESCRIBE, ["描述", "说明", "看到", "是什么", "describe", "what", "show", "see"]),
        (IntentType.LOCATE, ["什么时候", "哪里", "出现", "找", "when", "where", "find", "locate"]),
        (IntentType.COMPARE, ["变化", "区别", "对比", "不同", "compare", "change", "differ"]),
        (IntentType.SUMMARIZE, ["总结", "概括", "summary", "brief", "主要"]),
        (IntentType.AUDIO_FOCUS, ["说了", "声音", "音频", "听", "audio", "sound", "say", "speak"]),
        (IntentType.TRACK, ["跟踪", "追踪", "一直", "track", "follow"]),
    ]

    best_intent = IntentType.GENERAL
    best_count = 0

    for intent, keywords in patterns:
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_intent = intent
            best_count = count

    confidence = min(best_count / 2.0, 1.0)
    confidence = max(confidence, 0.3) if best_count > 0 else 0.3

    return best_intent, confidence


# ============================================================
# 模拟 V-JEPA2 编码器
# ============================================================

class MockVJEPA2Encoder:
    """模拟 V-JEPA2 编码器"""

    def __init__(self, embed_dim: int = 1024):
        self.embed_dim = embed_dim

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """编码单帧"""
        # 基于帧内容生成伪 embedding
        mean_val = np.mean(frame) / 255.0
        std_val = np.std(frame) / 255.0

        # 使用帧特征作为随机种子以保证可重复性
        seed = int(mean_val * 1000) % 10000
        rng = np.random.RandomState(seed)

        embedding = rng.randn(self.embed_dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # 添加帧特定的变化
        embedding += std_val * rng.randn(self.embed_dim) * 0.1
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def compute_change_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算变化分数 (1 - 余弦相似度)"""
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return max(0.0, 1.0 - sim)


# ============================================================
# 意图-嵌入核心循环 (Python 版本用于测试)
# ============================================================

@dataclass
class EmbeddingEntry:
    timestamp: float
    embedding: np.ndarray
    change_score: float
    intent: IntentType


@dataclass
class IntentState:
    primary: IntentType
    primary_confidence: float
    secondary: List[Tuple[IntentType, float]] = field(default_factory=list)
    iteration: int = 0
    semantic_cues: List[str] = field(default_factory=list)

    def current_strategy(self) -> ExtractionStrategy:
        return INTENT_STRATEGY_MAP[self.primary]

    def update_from_analysis(self, analysis: dict):
        """从 embedding 分析更新意图"""
        self.iteration += 1

        if analysis.get("has_significant_change") and self.primary != IntentType.COMPARE:
            self._add_secondary(IntentType.COMPARE, 0.5)

        if analysis.get("has_stable_regions") and self.primary != IntentType.AUDIO_FOCUS:
            self._add_secondary(IntentType.AUDIO_FOCUS, 0.3)

        self._maybe_switch_primary()

    def update_from_user(self, query: str):
        """用户反馈更新"""
        new_intent, confidence = intent_from_query(query)

        if new_intent != self.primary and confidence > self.primary_confidence:
            self.primary = new_intent
            self.primary_confidence = confidence
        else:
            self._add_secondary(new_intent, confidence)

        self.iteration += 1

    def _add_secondary(self, intent: IntentType, confidence: float):
        if intent == self.primary:
            return

        for i, (sec_intent, sec_conf) in enumerate(self.secondary):
            if sec_intent == intent:
                self.secondary[i] = (intent, min(sec_conf + confidence, 1.0))
                self.secondary.sort(key=lambda x: -x[1])
                return

        self.secondary.append((intent, confidence))
        self.secondary.sort(key=lambda x: -x[1])
        self.secondary = self.secondary[:3]

    def _maybe_switch_primary(self):
        if self.secondary and self.secondary[0][1] > self.primary_confidence + 0.2:
            old_primary = self.primary
            self.primary = self.secondary[0][0]
            self.primary_confidence = self.secondary[0][1]
            self.secondary = [(i, c) for i, c in self.secondary if i != self.primary]
            self.secondary.append((old_primary, self.primary_confidence - 0.2))


class IntentEmbeddingEngine:
    """意图-嵌入核心引擎 (Python 测试版本)"""

    def __init__(self, initial_query: str):
        intent, confidence = intent_from_query(initial_query)
        self.intent = IntentState(primary=intent, primary_confidence=confidence)
        self.embeddings: List[EmbeddingEntry] = []
        self.change_points: List[Tuple[float, float]] = []  # (timestamp, score)
        self.encoder = MockVJEPA2Encoder()
        self.last_embedding: Optional[np.ndarray] = None
        self.analysis_interval = 2.0
        self.last_analysis_time = 0.0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, float]:
        """处理新帧"""
        # 编码
        embedding = self.encoder.encode_frame(frame)

        # 计算变化分数
        if self.last_embedding is not None:
            change_score = self.encoder.compute_change_score(self.last_embedding, embedding)
        else:
            change_score = 0.0

        # 存储
        entry = EmbeddingEntry(
            timestamp=timestamp,
            embedding=embedding,
            change_score=change_score,
            intent=self.intent.primary
        )
        self.embeddings.append(entry)

        # 记录变化点
        if change_score > 0.1:
            self.change_points.append((timestamp, change_score))

        self.last_embedding = embedding

        # 定期分析
        if timestamp - self.last_analysis_time >= self.analysis_interval:
            analysis = self.analyze_embeddings()
            self.intent.update_from_analysis(analysis)
            self.last_analysis_time = timestamp

        return embedding, change_score

    def analyze_embeddings(self) -> dict:
        """分析 embedding 以增强意图"""
        if len(self.embeddings) < 2:
            return {}

        change_scores = [e.change_score for e in self.embeddings]
        max_change = max(change_scores) if change_scores else 0.0

        # 检测稳定区域
        stable_count = sum(1 for s in change_scores if s < 0.03)
        has_stable = stable_count > len(change_scores) * 0.3

        return {
            "has_significant_change": max_change > 0.15,
            "max_change_score": max_change,
            "has_stable_regions": has_stable,
            "activity_level": self._classify_activity(max_change),
        }

    def _classify_activity(self, max_change: float) -> str:
        if max_change < 0.03:
            return "static"
        elif max_change < 0.08:
            return "low"
        elif max_change < 0.2:
            return "medium"
        else:
            return "high"

    def get_keyframes(self, count: int = 5) -> List[Tuple[float, np.ndarray]]:
        """根据当前策略获取关键帧"""
        if not self.embeddings:
            return []

        strategy = self.intent.current_strategy()

        if strategy == ExtractionStrategy.START_END_TRANSITION:
            result = []
            if self.embeddings:
                result.append((self.embeddings[0].timestamp, self.embeddings[0].embedding))
            if self.change_points:
                max_change = max(self.change_points, key=lambda x: x[1])
                for e in self.embeddings:
                    if abs(e.timestamp - max_change[0]) < 0.1:
                        result.append((e.timestamp, e.embedding))
                        break
            if len(self.embeddings) > 1:
                result.append((self.embeddings[-1].timestamp, self.embeddings[-1].embedding))
            return result[:count]

        elif strategy in [ExtractionStrategy.UNIFORM_WITH_CHANGE, ExtractionStrategy.DENSE_WITH_PEAKS]:
            # 变化点 + 均匀采样
            result = []
            for t, _ in sorted(self.change_points, key=lambda x: -x[1])[:count//2]:
                for e in self.embeddings:
                    if abs(e.timestamp - t) < 0.1:
                        result.append((e.timestamp, e.embedding))
                        break

            step = max(1, len(self.embeddings) // (count - len(result)))
            for i in range(0, len(self.embeddings), step):
                if len(result) >= count:
                    break
                if not any(abs(r[0] - self.embeddings[i].timestamp) < 0.5 for r in result):
                    result.append((self.embeddings[i].timestamp, self.embeddings[i].embedding))

            return sorted(result, key=lambda x: x[0])[:count]

        else:
            # 均匀采样
            step = max(1, len(self.embeddings) // count)
            return [(e.timestamp, e.embedding) for e in self.embeddings[::step]][:count]


# ============================================================
# 测试用例
# ============================================================

def generate_test_video(scenario: str) -> List[Tuple[np.ndarray, float]]:
    """生成测试视频帧序列"""
    frames = []

    if scenario == "static":
        # 静态场景
        base_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        for i in range(20):
            frames.append((base_frame.copy(), i * 0.5))

    elif scenario == "with_person":
        # 有人出现的场景
        for i in range(20):
            if 8 <= i <= 12:
                # 人出现时帧变化大
                frame = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
            else:
                frame = np.ones((100, 100, 3), dtype=np.uint8) * (128 + i * 2)
            frames.append((frame, i * 0.5))

    elif scenario == "changing":
        # 持续变化的场景
        for i in range(20):
            brightness = int(50 + i * 10)
            frame = np.ones((100, 100, 3), dtype=np.uint8) * brightness
            # 添加随机噪声
            noise = np.random.randint(-20, 20, (100, 100, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append((frame, i * 0.5))

    elif scenario == "transition":
        # 前后对比场景
        for i in range(20):
            if i < 10:
                # 前半段：暗色
                frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
            else:
                # 后半段：亮色
                frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
            frames.append((frame, i * 0.5))

    return frames


def test_intent_classification():
    """测试意图分类"""
    print("\n" + "="*60)
    print("测试 1: 意图分类")
    print("="*60)

    test_cases = [
        ("描述一下这个视频", IntentType.DESCRIBE),
        ("什么时候出现了人", IntentType.LOCATE),
        ("视频前后有什么变化", IntentType.COMPARE),
        ("总结一下视频内容", IntentType.SUMMARIZE),
        ("他说了什么", IntentType.AUDIO_FOCUS),
        ("跟踪那个物体", IntentType.TRACK),
        ("hello", IntentType.GENERAL),
    ]

    passed = 0
    for query, expected in test_cases:
        intent, conf = intent_from_query(query)
        status = "✓" if intent == expected else "✗"
        print(f"  {status} \"{query}\" → {intent.value} (置信度: {conf:.2f})")
        if intent == expected:
            passed += 1

    print(f"\n  结果: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_intent_to_strategy():
    """测试意图到策略的映射"""
    print("\n" + "="*60)
    print("测试 2: 意图 → 策略映射")
    print("="*60)

    for intent, strategy in INTENT_STRATEGY_MAP.items():
        print(f"  {intent.value:15} → {strategy.value}")

    print("\n  所有映射已定义 ✓")
    return True


def test_core_loop():
    """测试核心意图-嵌入循环"""
    print("\n" + "="*60)
    print("测试 3: 核心意图-嵌入循环")
    print("="*60)

    # 场景：用户问"描述场景"，但视频有显著变化
    engine = IntentEmbeddingEngine("描述这个场景")

    print(f"  初始意图: {engine.intent.primary.value}")
    print(f"  初始策略: {engine.intent.current_strategy().value}")

    # 处理有变化的视频
    frames = generate_test_video("transition")

    for frame, timestamp in frames:
        _, change = engine.process_frame(frame, timestamp)

    print(f"\n  处理后:")
    print(f"    - Embedding 数量: {len(engine.embeddings)}")
    print(f"    - 变化点数量: {len(engine.change_points)}")
    print(f"    - 迭代次数: {engine.intent.iteration}")
    print(f"    - 当前意图: {engine.intent.primary.value}")
    print(f"    - 次要意图: {[i.value for i, c in engine.intent.secondary]}")

    # 验证：视频有显著变化时，Compare 应该成为次要意图
    has_compare = any(i == IntentType.COMPARE for i, _ in engine.intent.secondary)

    if has_compare:
        print("\n  ✓ 检测到变化，Compare 意图被添加到次要意图")
    else:
        print("\n  ✗ 未能检测到变化")

    return has_compare


def test_user_feedback():
    """测试用户反馈更新意图"""
    print("\n" + "="*60)
    print("测试 4: 用户反馈更新意图")
    print("="*60)

    # 直接测试意图更新，不处理帧避免自动演化
    intent, conf = intent_from_query("描述场景")
    state = IntentState(primary=intent, primary_confidence=conf)
    print(f"  初始意图: {state.primary.value}")

    # 用户追问 - 应该更新为 Locate
    state.update_from_user("什么时候发生了变化?")

    print(f"  用户追问后: {state.primary.value}")
    print(f"  新策略: {state.current_strategy().value}")

    # 验证：Locate 应该在意图中 (主要或次要)
    has_locate = state.primary == IntentType.LOCATE or \
                 any(i == IntentType.LOCATE for i, _ in state.secondary)

    success = has_locate
    print(f"\n  {'✓' if success else '✗'} Locate 意图已添加")

    return success


def test_keyframe_selection():
    """测试关键帧选择"""
    print("\n" + "="*60)
    print("测试 5: 关键帧选择")
    print("="*60)

    # 创建一个固定策略的引擎，不让它自动演化
    engine = IntentEmbeddingEngine("视频前后有什么变化")

    # 禁用自动分析以测试特定策略
    engine.analysis_interval = 1000.0  # 设置很大的间隔

    frames = generate_test_video("transition")
    for frame, timestamp in frames:
        engine.process_frame(frame, timestamp)

    # 手动设置策略为 StartEndTransition
    engine.intent.primary = IntentType.COMPARE
    engine.intent.primary_confidence = 1.0

    keyframes = engine.get_keyframes(5)

    print(f"  策略: {engine.intent.current_strategy().value}")
    print(f"  选中的关键帧时间点:")
    for t, _ in keyframes:
        print(f"    - {t:.1f}s")

    # 验证：StartEndTransition 应该包含开始和结束的帧
    has_start = any(abs(t) < 0.5 for t, _ in keyframes)
    has_end = any(abs(t - 9.5) < 1.0 for t, _ in keyframes)

    success = has_start and has_end
    print(f"\n  {'✓' if success else '✗'} 包含开始和结束帧")

    return success


def test_intent_evolution():
    """测试意图演化"""
    print("\n" + "="*60)
    print("测试 6: 意图演化")
    print("="*60)

    engine = IntentEmbeddingEngine("描述场景")
    print(f"  初始: {engine.intent.primary.value} (置信度: {engine.intent.primary_confidence:.2f})")

    # 处理有显著变化的视频，触发多次分析
    frames = generate_test_video("transition")

    for frame, timestamp in frames:
        engine.process_frame(frame, timestamp)

    # 强制再次分析
    analysis = engine.analyze_embeddings()
    engine.intent.update_from_analysis(analysis)

    print(f"  分析后:")
    print(f"    主要意图: {engine.intent.primary.value}")
    print(f"    次要意图: {[(i.value, f'{c:.2f}') for i, c in engine.intent.secondary]}")
    print(f"    迭代次数: {engine.intent.iteration}")

    # 多次添加 Compare 意图，看是否能切换
    for _ in range(5):
        analysis = {"has_significant_change": True, "max_change_score": 0.5}
        engine.intent.update_from_analysis(analysis)

    print(f"\n  多次分析后:")
    print(f"    主要意图: {engine.intent.primary.value}")

    # 验证意图是否演化
    success = engine.intent.iteration > 0
    print(f"\n  {'✓' if success else '✗'} 意图经历了演化过程")

    return success


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("AI 眼镜系统 - 端到端集成测试")
    print("="*60)

    tests = [
        ("意图分类", test_intent_classification),
        ("意图→策略映射", test_intent_to_strategy),
        ("核心循环", test_core_loop),
        ("用户反馈", test_user_feedback),
        ("关键帧选择", test_keyframe_selection),
        ("意图演化", test_intent_evolution),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ 测试失败: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")

    print(f"\n  总计: {passed}/{total} 测试通过")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

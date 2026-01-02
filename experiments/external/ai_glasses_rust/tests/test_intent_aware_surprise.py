#!/usr/bin/env python3
"""
意图感知的惊讶度计算 (V2)

核心改进:
1. 首帧作为"第一个场景"必选
2. 惊讶度与用户意图关联
3. 不同意图使用不同的惊讶度策略
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class Intent(Enum):
    """用户意图类型"""
    COUNT = "count"      # 计数: 有几个X?
    LOCATE = "locate"    # 定位: X在哪里?
    COMPARE = "compare"  # 对比: 前后有什么变化?
    DESCRIBE = "describe"  # 描述: 这是什么?
    TRACK = "track"      # 跟踪: X去哪了?


@dataclass
class IntentConfig:
    """意图配置"""
    surprise_threshold: float  # 惊讶度阈值
    focus: str                 # 关注点
    strategy: str              # 策略名称


# 意图 → 惊讶度策略映射
INTENT_STRATEGIES = {
    Intent.COUNT: IntentConfig(
        surprise_threshold=0.10,
        focus="新物体出现",
        strategy="OBJECT_APPEARANCE"
    ),
    Intent.LOCATE: IntentConfig(
        surprise_threshold=0.15,
        focus="位置变化",
        strategy="POSITION_CHANGE"
    ),
    Intent.COMPARE: IntentConfig(
        surprise_threshold=0.25,
        focus="场景差异",
        strategy="SCENE_DIFFERENCE"
    ),
    Intent.DESCRIBE: IntentConfig(
        surprise_threshold=0.08,
        focus="代表性内容",
        strategy="REPRESENTATIVE"
    ),
    Intent.TRACK: IntentConfig(
        surprise_threshold=0.05,
        focus="连续运动",
        strategy="CONTINUOUS_MOTION"
    ),
}


class IntentAnalyzer:
    """意图分析器"""

    INTENT_KEYWORDS = {
        Intent.COUNT: ["几", "多少", "几个", "几台", "数量", "count", "how many"],
        Intent.LOCATE: ["在哪", "哪里", "位置", "where", "location"],
        Intent.COMPARE: ["变化", "区别", "不同", "对比", "change", "difference"],
        Intent.DESCRIBE: ["是什么", "描述", "介绍", "what", "describe"],
        Intent.TRACK: ["去哪", "跟踪", "移动", "track", "where did"],
    }

    def analyze(self, query: str) -> tuple[Intent, float]:
        """分析用户意图"""
        query_lower = query.lower()

        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[intent] = score

        # 找到最高分的意图
        best_intent = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_intent] / 2)  # 归一化置信度

        # 如果没有匹配，默认 DESCRIBE
        if scores[best_intent] == 0:
            best_intent = Intent.DESCRIBE
            confidence = 0.5

        return best_intent, confidence


class IntentAwareSurpriseCalculator:
    """
    意图感知的惊讶度计算器

    核心思想:
    - COUNT: 关注"新物体"的出现 → 检测 embedding 聚类变化
    - LOCATE: 关注"位置"变化 → 检测空间特征变化
    - COMPARE: 关注"场景"差异 → 检测整体变化
    """

    def __init__(self, intent: Intent = Intent.COUNT):
        self.intent = intent
        self.config = INTENT_STRATEGIES[intent]
        self.embeddings = []
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # 加载特征提取器
        print(f"[Calculator] 意图: {intent.value} ({self.config.focus})")
        print(f"[Calculator] 策略: {self.config.strategy}")
        print(f"[Calculator] 阈值: {self.config.surprise_threshold}")
        print(f"[Calculator] 加载特征提取器...")

        from torchvision import models, transforms

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        print("[Calculator] 初始化完成")

    def encode_frame(self, image: Image.Image) -> np.ndarray:
        """编码单帧"""
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        return embedding

    def compute_intent_aware_surprise(self,
                                       current_emb: np.ndarray,
                                       prev_emb: np.ndarray = None,
                                       all_prev_embs: List[np.ndarray] = None) -> Dict:
        """
        计算意图感知的惊讶度

        不同意图使用不同的计算方式:
        - COUNT: 与所有历史 embedding 的最小距离 (检测新物体)
        - LOCATE: 与前一帧的距离 (检测位置变化)
        - COMPARE: 与首帧的距离 (检测场景变化)
        """
        result = {
            "raw_surprise": 0.0,
            "intent_surprise": 0.0,
            "is_new_object": False,
            "reason": ""
        }

        if prev_emb is None:
            # 首帧: 作为第一个场景，惊讶度设为基准值
            result["raw_surprise"] = 0.5  # 中等惊讶度（新场景）
            result["intent_surprise"] = 0.5
            result["is_new_object"] = True
            result["reason"] = "首帧 (第一个场景)"
            return result

        # 计算与前一帧的余弦距离
        def cosine_distance(a, b):
            a_norm = a / (np.linalg.norm(a) + 1e-8)
            b_norm = b / (np.linalg.norm(b) + 1e-8)
            return 1 - np.dot(a_norm, b_norm)

        raw_surprise = cosine_distance(current_emb, prev_emb)
        result["raw_surprise"] = float(raw_surprise)

        # 根据意图计算不同的惊讶度
        if self.intent == Intent.COUNT:
            # COUNT: 检测是否出现了"新物体"
            # 策略: 与所有历史 embedding 比较，如果与所有都差异大，说明是新物体
            if all_prev_embs and len(all_prev_embs) > 0:
                min_dist = min(cosine_distance(current_emb, e) for e in all_prev_embs)
                max_dist = max(cosine_distance(current_emb, e) for e in all_prev_embs)

                # 如果与所有历史都不相似 (min_dist 大)，可能是新物体
                if min_dist > 0.3:
                    result["intent_surprise"] = min_dist
                    result["is_new_object"] = True
                    result["reason"] = f"可能是新物体 (与历史最小距离={min_dist:.3f})"
                else:
                    # 与某个历史帧相似，可能是同一物体
                    result["intent_surprise"] = min_dist * 0.5
                    result["is_new_object"] = False
                    result["reason"] = f"可能是已有物体 (与历史最小距离={min_dist:.3f})"
            else:
                result["intent_surprise"] = raw_surprise
                result["is_new_object"] = raw_surprise > self.config.surprise_threshold
                result["reason"] = "与前一帧比较"

        elif self.intent == Intent.COMPARE:
            # COMPARE: 与首帧比较
            if all_prev_embs and len(all_prev_embs) > 0:
                first_frame_dist = cosine_distance(current_emb, all_prev_embs[0])
                result["intent_surprise"] = first_frame_dist
                result["reason"] = f"与首帧距离={first_frame_dist:.3f}"
            else:
                result["intent_surprise"] = raw_surprise
                result["reason"] = "与前一帧比较"

        else:
            # 其他意图: 使用原始惊讶度
            result["intent_surprise"] = raw_surprise
            result["reason"] = "与前一帧比较"

        return result

    def analyze_frames(self, images: List[Image.Image], timestamps: List[float]) -> List[Dict]:
        """分析所有帧"""
        print(f"\n[分析] 处理 {len(images)} 帧 (意图: {self.intent.value})...")

        results = []
        all_embeddings = []

        for i, (img, ts) in enumerate(zip(images, timestamps)):
            # 编码
            current_emb = self.encode_frame(img)

            # 计算意图感知惊讶度
            prev_emb = all_embeddings[-1] if all_embeddings else None
            surprise_info = self.compute_intent_aware_surprise(
                current_emb,
                prev_emb,
                all_embeddings if all_embeddings else None
            )

            all_embeddings.append(current_emb)

            # 判断是否为关键帧
            is_key_frame = (i == 0) or (surprise_info["intent_surprise"] > self.config.surprise_threshold)

            result = {
                "frame_id": i + 1,
                "timestamp": ts,
                "raw_surprise": surprise_info["raw_surprise"],
                "intent_surprise": surprise_info["intent_surprise"],
                "is_new_object": surprise_info["is_new_object"],
                "is_key_frame": is_key_frame,
                "reason": surprise_info["reason"]
            }
            results.append(result)

            # 打印
            status = ""
            if is_key_frame:
                status = " ★ 关键帧"
            if surprise_info["is_new_object"]:
                status += " [新物体!]"

            print(f"  帧{i+1} ({ts:.1f}s): "
                  f"原始={surprise_info['raw_surprise']:.4f}, "
                  f"意图={surprise_info['intent_surprise']:.4f} "
                  f"| {surprise_info['reason']}{status}")

        return results

    def select_frames(self, results: List[Dict], max_frames: int = 3) -> List[int]:
        """选择帧"""
        selected = []

        # 1. 首帧必选
        selected.append(0)

        # 2. 根据意图选择
        if self.intent == Intent.COUNT:
            # COUNT: 选择"新物体"的帧
            new_object_frames = [r for r in results[1:] if r["is_new_object"]]
            new_object_frames.sort(key=lambda x: x["intent_surprise"], reverse=True)
            for r in new_object_frames:
                if len(selected) >= max_frames:
                    break
                if r["frame_id"] - 1 not in selected:
                    selected.append(r["frame_id"] - 1)
        else:
            # 其他意图: 按惊讶度选择
            key_frames = [r for r in results[1:] if r["is_key_frame"]]
            key_frames.sort(key=lambda x: x["intent_surprise"], reverse=True)
            for r in key_frames:
                if len(selected) >= max_frames:
                    break
                if r["frame_id"] - 1 not in selected:
                    selected.append(r["frame_id"] - 1)

        return sorted(selected)


def main():
    print("=" * 70)
    print("  意图感知的惊讶度计算验证 (V2)")
    print("=" * 70)

    # 1. 用户查询
    user_query = "这里面有几台笔记本电脑？"
    print(f"\n用户查询: {user_query}")

    # 2. 分析意图
    intent_analyzer = IntentAnalyzer()
    intent, confidence = intent_analyzer.analyze(user_query)
    print(f"识别意图: {intent.value} (置信度: {confidence:.0%})")
    print(f"策略配置: {INTENT_STRATEGIES[intent]}")

    # 3. 加载关键帧
    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes"
    images = []
    timestamps = []

    print(f"\n加载关键帧: {keyframes_dir}")
    for f in sorted(keyframes_dir.glob("frame_*.jpg")):
        img = Image.open(f).convert("RGB")
        images.append(img)
        ts = float(f.stem.split("_")[-1].replace("s", ""))
        timestamps.append(ts)
        print(f"  {f.name}")

    # 4. 意图感知的惊讶度分析
    print("\n" + "=" * 70)
    print(f"意图感知惊讶度分析 (意图: {intent.value})")
    print("=" * 70)

    calculator = IntentAwareSurpriseCalculator(intent=intent)
    results = calculator.analyze_frames(images, timestamps)

    # 5. 帧选择
    selected_indices = calculator.select_frames(results, max_frames=3)
    print(f"\n选中帧: {[i+1 for i in selected_indices]}")

    # 6. 分析摘要
    print("\n" + "=" * 70)
    print("分析摘要")
    print("=" * 70)

    new_objects = [r for r in results if r["is_new_object"]]
    print(f"检测到 {len(new_objects)} 个可能的新物体出现点:")
    for r in new_objects:
        print(f"  - 帧{r['frame_id']} ({r['timestamp']:.1f}s): {r['reason']}")

    print(f"\n建议送入 Gemma 的帧: {[i+1 for i in selected_indices]}")
    print(f"  → 帧1: 第一个场景 (MacBook)")
    if 3 in selected_indices:
        print(f"  → 帧4: 新物体出现 (Dell 笔记本)")

    # 7. 对比旧方案
    print("\n" + "=" * 70)
    print("与旧方案对比")
    print("=" * 70)
    print("""
    旧方案 (无意图感知):
    - 首帧惊讶度 = 0 (因为没有预测对象)
    - 只看帧间变化，不考虑"新物体"

    新方案 (意图感知):
    - 首帧作为"第一个场景"必选
    - COUNT 意图: 检测与所有历史帧的最小距离
    - 如果与所有历史都不相似 → 可能是新物体
    """)


if __name__ == "__main__":
    main()

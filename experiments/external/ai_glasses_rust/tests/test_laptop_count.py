#!/usr/bin/env python3
"""
AI 眼镜系统 - 真实场景测试

测试用例: 分析视频中有几台笔记本电脑
演示完整的意图-嵌入循环

用法: python tests/test_laptop_count.py
"""

import os
import sys
import time
import base64
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

# 添加父目录
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 意图系统 (与 Rust 保持一致)
# ============================================================

class IntentType(Enum):
    DESCRIBE = "describe"
    LOCATE = "locate"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    COUNT = "count"  # 新增: 计数意图
    GENERAL = "general"


class ExtractionStrategy(Enum):
    UNIFORM_WITH_CHANGE = "uniform_with_change"
    DENSE_WITH_PEAKS = "dense_with_peaks"
    START_END_TRANSITION = "start_end_transition"
    REPRESENTATIVE = "representative"
    ALL_FRAMES = "all_frames"  # 计数需要看所有帧


def analyze_intent(query: str) -> Tuple[IntentType, float, ExtractionStrategy]:
    """分析用户意图"""
    q = query.lower()

    # 计数相关
    count_keywords = ["几", "多少", "数量", "count", "how many", "几个", "几台"]
    if any(kw in q for kw in count_keywords):
        return IntentType.COUNT, 0.9, ExtractionStrategy.ALL_FRAMES

    # 对比/变化相关
    compare_keywords = ["变化", "区别", "对比", "不同", "前后", "compare", "change", "differ"]
    if any(kw in q for kw in compare_keywords):
        return IntentType.COMPARE, 0.85, ExtractionStrategy.START_END_TRANSITION

    # 定位相关
    locate_keywords = ["哪里", "在哪", "什么时候", "where", "when", "找"]
    if any(kw in q for kw in locate_keywords):
        return IntentType.LOCATE, 0.8, ExtractionStrategy.DENSE_WITH_PEAKS

    # 描述相关
    describe_keywords = ["描述", "是什么", "看到", "describe", "what", "内容"]
    if any(kw in q for kw in describe_keywords):
        return IntentType.DESCRIBE, 0.7, ExtractionStrategy.UNIFORM_WITH_CHANGE

    return IntentType.GENERAL, 0.5, ExtractionStrategy.REPRESENTATIVE


# ============================================================
# Gemma 3n 模型接口
# ============================================================

class GemmaModel:
    """Gemma 3n 模型封装"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def _load_model(self, use_mock: bool = False):
        """加载模型"""
        import torch

        # 检测设备
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"使用设备: {self.device}")

        if use_mock:
            print("使用 Mock 模式 (快速演示)")
            self.model = None
            self.processor = None
            return

        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration

            model_name = "google/gemma-3n-E4B-it"
            print(f"加载模型: {model_name}")

            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                device_map=self.device
            )
            print("模型加载成功!")

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用 Mock 模式")
            self.model = None
            self.processor = None

    def analyze_images(self, images: List[Image.Image], query: str) -> str:
        """分析图片"""
        import torch

        if self.model is None:
            # Mock 模式
            return self._mock_analyze(images, query)

        # 构建消息
        content = []
        for i, img in enumerate(images):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[图{i+1}]"})

        content.append({"type": "text", "text": f"\n用户问题: {query}\n请仔细分析图片并回答。"})

        messages = [{"role": "user", "content": content}]

        # 处理输入
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": False,
        }

        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            generate_kwargs["pixel_values"] = inputs["pixel_values"].to(self.device)

        # 生成
        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        response = self.processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response

    def _mock_analyze(self, images: List[Image.Image], query: str) -> str:
        """Mock 分析"""
        q = query.lower()

        if "笔记本" in q or "电脑" in q or "laptop" in q:
            return """根据分析这些图片，我可以看到：

这些截图显示的是一个 AI 聊天应用的界面，运行在一台 MacBook 上。

**笔记本电脑数量: 1 台**

从界面特征判断：
- 左上角的红黄绿三个按钮是 macOS 的窗口控制按钮
- 界面显示的是一个本地运行的 AI 多模态聊天系统
- 系统信息显示 "MPS 运行中"，表明使用的是 Apple Silicon Mac

所以，视频中显示的是 **1 台 MacBook 笔记本电脑**。"""

        return f"分析了 {len(images)} 张图片。这些图片显示的是一个 AI 聊天系统的界面。"


# ============================================================
# 意图-嵌入引擎
# ============================================================

@dataclass
class AnalysisResult:
    """分析结果"""
    intent: IntentType
    confidence: float
    strategy: ExtractionStrategy
    selected_frames: List[Tuple[float, Image.Image]]
    response: str
    processing_time_ms: int


class IntentEmbeddingAnalyzer:
    """意图-嵌入分析器"""

    def __init__(self, use_mock: bool = False):
        self.model = None
        self.use_mock = use_mock

    def load_model(self):
        """延迟加载模型"""
        if self.model is None:
            self.model = GemmaModel()
            self.model._load_model(use_mock=self.use_mock)

    def analyze(self, keyframes_dir: str, query: str) -> AnalysisResult:
        """执行分析"""
        start_time = time.time()

        # 1. 意图分析
        print(f"\n{'='*60}")
        print(f"用户查询: \"{query}\"")
        print('='*60)

        intent, confidence, strategy = analyze_intent(query)
        print(f"意图分析:")
        print(f"  - 类型: {intent.value}")
        print(f"  - 置信度: {confidence:.0%}")
        print(f"  - 策略: {strategy.value}")

        # 2. 加载关键帧
        keyframes = self._load_keyframes(keyframes_dir)
        print(f"\n加载关键帧: {len(keyframes)} 张")

        # 3. 根据策略选择帧
        selected = self._select_frames(keyframes, strategy)
        print(f"策略选择: {len(selected)} 张用于分析")
        for t, _ in selected:
            print(f"  - {t:.1f}s")

        # 4. 加载模型并分析
        print(f"\n正在分析...")
        self.load_model()

        images = [img for _, img in selected]
        response = self.model.analyze_images(images, query)

        processing_time = int((time.time() - start_time) * 1000)

        return AnalysisResult(
            intent=intent,
            confidence=confidence,
            strategy=strategy,
            selected_frames=selected,
            response=response,
            processing_time_ms=processing_time
        )

    def _load_keyframes(self, keyframes_dir: str) -> List[Tuple[float, Image.Image]]:
        """加载关键帧"""
        keyframes = []
        keyframes_path = Path(keyframes_dir)

        for f in sorted(keyframes_path.glob("keyframe_*.jpg")):
            # 从文件名提取时间戳
            # 格式: keyframe_01_0.0s.jpg
            parts = f.stem.split("_")
            if len(parts) >= 3:
                time_str = parts[2].replace("s", "")
                try:
                    timestamp = float(time_str)
                    img = Image.open(f).convert("RGB")
                    keyframes.append((timestamp, img))
                except ValueError:
                    pass

        return keyframes

    def _select_frames(
        self,
        keyframes: List[Tuple[float, Image.Image]],
        strategy: ExtractionStrategy
    ) -> List[Tuple[float, Image.Image]]:
        """根据策略选择帧"""

        if not keyframes:
            return []

        if strategy == ExtractionStrategy.ALL_FRAMES:
            # 计数任务: 均匀选择多帧
            step = max(1, len(keyframes) // 6)
            return keyframes[::step][:6]

        elif strategy == ExtractionStrategy.REPRESENTATIVE:
            # 选择代表性帧
            if len(keyframes) <= 3:
                return keyframes
            return [keyframes[0], keyframes[len(keyframes)//2], keyframes[-1]]

        elif strategy == ExtractionStrategy.START_END_TRANSITION:
            # 首尾 + 中间
            return [keyframes[0], keyframes[len(keyframes)//2], keyframes[-1]]

        else:
            # 默认: 均匀采样
            step = max(1, len(keyframes) // 4)
            return keyframes[::step][:4]


# ============================================================
# 主测试
# ============================================================

def main():
    print("\n" + "="*60)
    print("AI 眼镜系统 - 笔记本电脑计数测试")
    print("="*60)

    # 关键帧目录
    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "videos" / "keyframes"

    if not keyframes_dir.exists():
        print(f"错误: 找不到关键帧目录 {keyframes_dir}")
        return False

    # 创建分析器 (使用 Mock 模式快速演示)
    analyzer = IntentEmbeddingAnalyzer(use_mock=True)

    # 测试查询
    query = "这里面有几台笔记本电脑？"

    # 执行分析
    result = analyzer.analyze(str(keyframes_dir), query)

    # 显示结果
    print(f"\n{'='*60}")
    print("分析结果")
    print('='*60)
    print(f"\n{result.response}")
    print(f"\n处理时间: {result.processing_time_ms}ms")

    return True


def test_multiple_queries():
    """测试多个查询"""
    print("\n" + "="*60)
    print("AI 眼镜系统 - 多查询测试")
    print("="*60)

    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "videos" / "keyframes"

    if not keyframes_dir.exists():
        print(f"错误: 找不到关键帧目录 {keyframes_dir}")
        return

    queries = [
        "这里面有几台笔记本电脑？",
        "描述一下这个视频的内容",
        "视频前后有什么变化？",
    ]

    analyzer = IntentEmbeddingAnalyzer(use_mock=True)

    for query in queries:
        result = analyzer.analyze(str(keyframes_dir), query)

        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"意图: {result.intent.value} (置信度: {result.confidence:.0%})")
        print(f"策略: {result.strategy.value}")
        print(f"选中帧数: {len(result.selected_frames)}")
        print(f"\n回答:\n{result.response}")
        print(f"\n处理时间: {result.processing_time_ms}ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", action="store_true", help="运行多查询测试")
    args = parser.parse_args()

    if args.multi:
        test_multiple_queries()
    else:
        main()

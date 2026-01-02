#!/usr/bin/env python3
"""
AI 眼镜系统 - 交互式演示

支持:
- 步骤展示 (每步暂停)
- 流程可视化
- 用户交互修改

用法: python tests/interactive_demo.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

# ============================================================
# 意图和策略定义
# ============================================================

class IntentType(Enum):
    DESCRIBE = "描述场景"
    LOCATE = "定位内容"
    COMPARE = "对比变化"
    COUNT = "计数统计"
    SUMMARIZE = "总结概括"
    GENERAL = "通用理解"


class ExtractionStrategy(Enum):
    UNIFORM_WITH_CHANGE = "均匀采样+变化检测"
    DENSE_WITH_PEAKS = "密集采样+峰值检测"
    START_END_TRANSITION = "首尾+转折点"
    ALL_FRAMES = "全帧分析"
    REPRESENTATIVE = "代表性帧"


# 意图 → 策略映射
INTENT_STRATEGY_MAP = {
    IntentType.DESCRIBE: ExtractionStrategy.UNIFORM_WITH_CHANGE,
    IntentType.LOCATE: ExtractionStrategy.DENSE_WITH_PEAKS,
    IntentType.COMPARE: ExtractionStrategy.START_END_TRANSITION,
    IntentType.COUNT: ExtractionStrategy.ALL_FRAMES,
    IntentType.SUMMARIZE: ExtractionStrategy.REPRESENTATIVE,
    IntentType.GENERAL: ExtractionStrategy.UNIFORM_WITH_CHANGE,
}

# 策略 → 帧选择参数
STRATEGY_PARAMS = {
    ExtractionStrategy.ALL_FRAMES: {"max_frames": 6, "focus": "全部关键帧"},
    ExtractionStrategy.START_END_TRANSITION: {"max_frames": 3, "focus": "开始、转折、结束"},
    ExtractionStrategy.UNIFORM_WITH_CHANGE: {"max_frames": 4, "focus": "均匀分布"},
    ExtractionStrategy.DENSE_WITH_PEAKS: {"max_frames": 5, "focus": "变化峰值"},
    ExtractionStrategy.REPRESENTATIVE: {"max_frames": 3, "focus": "代表性帧"},
}


def analyze_intent(query: str) -> Tuple[IntentType, float, List[str]]:
    """分析用户意图，返回 (意图类型, 置信度, 匹配的关键词)"""
    q = query.lower()

    patterns = [
        (IntentType.COUNT, ["几", "多少", "数量", "count", "几个", "几台"]),
        (IntentType.COMPARE, ["变化", "区别", "对比", "不同", "前后", "compare"]),
        (IntentType.LOCATE, ["哪里", "在哪", "什么时候", "where", "when", "找"]),
        (IntentType.DESCRIBE, ["描述", "是什么", "看到", "describe", "内容"]),
        (IntentType.SUMMARIZE, ["总结", "概括", "summary", "主要"]),
    ]

    for intent, keywords in patterns:
        matched = [kw for kw in keywords if kw in q]
        if matched:
            confidence = min(len(matched) * 0.3 + 0.5, 0.95)
            return intent, confidence, matched

    return IntentType.GENERAL, 0.5, []


# ============================================================
# 交互式演示器
# ============================================================

class InteractiveDemo:
    """交互式演示器"""

    def __init__(self, keyframes_dir: str, step_by_step: bool = True):
        self.keyframes_dir = Path(keyframes_dir)
        self.step_by_step = step_by_step
        self.keyframes = self._load_keyframes()

    def _load_keyframes(self) -> List[Tuple[float, str]]:
        """加载关键帧信息"""
        keyframes = []
        for f in sorted(self.keyframes_dir.glob("keyframe_*.jpg")):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                time_str = parts[2].replace("s", "")
                try:
                    timestamp = float(time_str)
                    keyframes.append((timestamp, str(f)))
                except ValueError:
                    pass
        return keyframes

    def wait_for_user(self, prompt: str = "按 Enter 继续..."):
        """等待用户输入"""
        if self.step_by_step:
            input(f"\n{prompt}")

    def print_header(self, text: str):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print('='*60)

    def print_step(self, step_num: int, title: str):
        """打印步骤"""
        print(f"\n┌─────────────────────────────────────────────────────────┐")
        print(f"│  步骤 {step_num}: {title:50}│")
        print(f"└─────────────────────────────────────────────────────────┘")

    def run(self, query: str):
        """运行演示"""
        self.print_header("AI 眼镜系统 - 意图-嵌入核心循环演示")
        print(f"\n用户查询: \"{query}\"")
        print(f"关键帧数量: {len(self.keyframes)} 张")

        self.wait_for_user()

        # ============ 步骤 1: 意图分析 ============
        self.print_step(1, "意图分析")
        print("\n分析用户查询，识别意图类型...")
        time.sleep(0.5)

        intent, confidence, matched_keywords = analyze_intent(query)

        print(f"\n  📝 查询文本: \"{query}\"")
        print(f"  🔍 匹配关键词: {matched_keywords if matched_keywords else '无特定关键词'}")
        print(f"\n  ✅ 识别结果:")
        print(f"     意图类型: {intent.value}")
        print(f"     置信度: {confidence:.0%}")

        self.wait_for_user()

        # ============ 步骤 2: 策略选择 ============
        self.print_step(2, "策略选择")
        print("\n根据意图类型，选择最优提取策略...")
        time.sleep(0.5)

        strategy = INTENT_STRATEGY_MAP[intent]
        params = STRATEGY_PARAMS[strategy]

        print(f"\n  📊 意图-策略映射:")
        for i, s in INTENT_STRATEGY_MAP.items():
            marker = "  →" if i == intent else "   "
            print(f"    {marker} {i.value:10} → {s.value}")

        print(f"\n  ✅ 选中策略: {strategy.value}")
        print(f"     最大帧数: {params['max_frames']}")
        print(f"     选择重点: {params['focus']}")

        self.wait_for_user()

        # ============ 步骤 3: 关键帧选择 ============
        self.print_step(3, "关键帧选择")
        print("\n根据策略从视频中选择关键帧...")
        time.sleep(0.5)

        selected = self._select_frames(strategy, params['max_frames'])

        print(f"\n  📹 可用关键帧: {len(self.keyframes)} 张")
        print(f"  🎯 策略要求: {params['focus']}")
        print(f"\n  ✅ 选中的关键帧:")
        for i, (t, path) in enumerate(selected):
            print(f"     [{i+1}] {t:6.1f}s - {Path(path).name}")

        self.wait_for_user()

        # ============ 步骤 4: 模型分析 ============
        self.print_step(4, "多模态分析")
        print("\n将选中的关键帧送入 Gemma 3n 进行分析...")
        time.sleep(0.5)

        print(f"\n  🖼️  输入图片: {len(selected)} 张关键帧")
        print(f"  💬 用户问题: {query}")
        print(f"\n  ⏳ 模型推理中...")
        time.sleep(1)

        # Mock 响应
        response = self._generate_mock_response(query, intent, selected)

        print(f"\n  ✅ 分析完成!")

        self.wait_for_user()

        # ============ 步骤 5: 结果展示 ============
        self.print_step(5, "结果输出")

        print(f"\n{'─'*60}")
        print(f"  AI 回答:")
        print(f"{'─'*60}")
        for line in response.split('\n'):
            print(f"  {line}")
        print(f"{'─'*60}")

        # ============ 流程总结 ============
        self.print_header("流程总结")

        print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │                    意图-嵌入核心循环                       │
  ├──────────────────────────────────────────────────────────┤
  │                                                          │
  │    用户查询: "{query[:20]}..."
  │         │                                                │
  │         ▼                                                │
  │    ┌─────────────┐                                       │
  │    │  意图分析   │ ──→ {intent.value:10}            │
  │    └─────────────┘                                       │
  │         │                                                │
  │         ▼                                                │
  │    ┌─────────────┐                                       │
  │    │  策略选择   │ ──→ {strategy.value[:15]:15}│
  │    └─────────────┘                                       │
  │         │                                                │
  │         ▼                                                │
  │    ┌─────────────┐                                       │
  │    │  帧选择    │ ──→ {len(selected)} 帧                        │
  │    └─────────────┘                                       │
  │         │                                                │
  │         ▼                                                │
  │    ┌─────────────┐                                       │
  │    │ Gemma 分析  │ ──→ 生成回答                        │
  │    └─────────────┘                                       │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
""")

    def _select_frames(
        self,
        strategy: ExtractionStrategy,
        max_frames: int
    ) -> List[Tuple[float, str]]:
        """根据策略选择帧"""
        if not self.keyframes:
            return []

        if strategy == ExtractionStrategy.ALL_FRAMES:
            step = max(1, len(self.keyframes) // max_frames)
            return self.keyframes[::step][:max_frames]

        elif strategy == ExtractionStrategy.START_END_TRANSITION:
            result = [self.keyframes[0]]
            if len(self.keyframes) > 2:
                result.append(self.keyframes[len(self.keyframes)//2])
            if len(self.keyframes) > 1:
                result.append(self.keyframes[-1])
            return result[:max_frames]

        else:
            step = max(1, len(self.keyframes) // max_frames)
            return self.keyframes[::step][:max_frames]

    def _generate_mock_response(
        self,
        query: str,
        intent: IntentType,
        frames: List[Tuple[float, str]]
    ) -> str:
        """生成模拟响应"""
        q = query.lower()

        if intent == IntentType.COUNT and ("笔记本" in q or "电脑" in q):
            return """根据分析这些关键帧，我可以看到：

这些截图显示的是一个 AI 聊天应用的界面，
运行在一台 MacBook 笔记本电脑上。

**笔记本电脑数量: 1 台**

判断依据：
- 左上角的红黄绿按钮是 macOS 窗口控制
- 界面显示 "MPS 运行中" (Apple Silicon)
- 整个视频只展示同一台电脑的屏幕"""

        elif intent == IntentType.COMPARE:
            return f"""对比分析 {len(frames)} 个时间点的变化：

开始 ({frames[0][0]:.1f}s): AI 聊天系统首页
中间 ({frames[len(frames)//2][0]:.1f}s): 用户正在进行对话
结束 ({frames[-1][0]:.1f}s): 多轮对话完成

主要变化：
- 对话内容增加
- 界面展示不同功能（图片分析、音频处理）"""

        elif intent == IntentType.DESCRIBE:
            return f"""视频内容描述：

这是一个 AI 多模态聊天系统的演示视频，
展示了以下功能：
1. 文本对话
2. 图片分析（分析4张图片：猫、狗、食物、花）
3. 音频处理

视频时长约 {frames[-1][0]:.0f} 秒"""

        return f"分析了 {len(frames)} 张关键帧。这些图片显示的是一个 AI 聊天系统的界面。"


# ============================================================
# 主程序
# ============================================================

def main():
    print("\n" + "="*60)
    print("  AI 眼镜系统 - 交互式演示")
    print("  (按 Enter 进行每一步)")
    print("="*60)

    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "videos" / "keyframes"

    if not keyframes_dir.exists():
        print(f"错误: 找不到关键帧目录 {keyframes_dir}")
        return

    demo = InteractiveDemo(str(keyframes_dir), step_by_step=True)

    # 测试查询
    queries = [
        "这里面有几台笔记本电脑？",
    ]

    for query in queries:
        demo.run(query)
        print("\n" + "="*60)

    # 询问是否继续
    print("\n是否继续测试其他查询？")
    print("  1. 描述视频内容")
    print("  2. 视频前后有什么变化")
    print("  3. 自定义查询")
    print("  0. 退出")

    while True:
        choice = input("\n请选择 (0-3): ").strip()

        if choice == "0":
            print("\n演示结束，感谢使用！")
            break
        elif choice == "1":
            demo.run("描述一下这个视频的内容")
        elif choice == "2":
            demo.run("视频前后有什么变化？")
        elif choice == "3":
            custom = input("请输入查询: ").strip()
            if custom:
                demo.run(custom)
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AI 眼镜系统 - 真实 Gemma 分析测试

使用真实的 Gemma 3n 模型分析视频中的笔记本电脑数量

用法: python tests/test_real_gemma.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

# ============================================================
# Gemma 3n 模型
# ============================================================

class RealGemmaAnalyzer:
    """真实 Gemma 3n 分析器"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def load(self):
        """加载模型"""
        if self.model is not None:
            return

        # 检测设备
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"设备: {self.device}")

        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        model_name = "google/gemma-3n-E4B-it"
        print(f"加载模型: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map=self.device
        )
        print("模型加载完成!")

    def analyze(self, images: List[Image.Image], query: str) -> str:
        """分析图片"""
        self.load()

        # 构建消息
        content = []
        for i, img in enumerate(images):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[图片{i+1}]"})

        content.append({"type": "text", "text": f"\n\n请仔细分析以上图片，然后回答问题：{query}"})

        messages = [{"role": "user", "content": content}]

        # 处理
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

        print("正在分析...")
        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        response = self.processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response


# ============================================================
# 主测试
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  笔记本电脑数量识别测试 (真实 Gemma 3n)")
    print("=" * 60)

    # 加载关键帧
    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes"

    images = []
    for f in sorted(keyframes_dir.glob("frame_*.jpg")):
        img = Image.open(f).convert("RGB")
        images.append(img)
        print(f"加载: {f.name}")

    print(f"\n共加载 {len(images)} 张关键帧")

    # 创建分析器
    analyzer = RealGemmaAnalyzer()

    # 查询
    query = "这些图片来自同一个视频，请仔细数一数视频中一共出现了几台笔记本电脑？请说明每台笔记本的特征（品牌、颜色、状态等）。"

    print(f"\n查询: {query}")
    print("-" * 60)

    # 分析
    start_time = time.time()
    response = analyzer.analyze(images, query)
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("分析结果:")
    print("=" * 60)
    print(response)
    print(f"\n耗时: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()

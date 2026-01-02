#!/usr/bin/env python3
"""
简单测试 - 单张图片分析
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from pathlib import Path

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("  单张图片测试")
print("=" * 60)

# 加载单张图片
img_path = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes" / "frame_01_0.0s.jpg"
image = Image.open(img_path).convert("RGB")
print(f"图片: {img_path.name} ({image.size})")

# 加载模型
model_name = "google/gemma-3n-E2B-it"
print(f"\n加载模型: {model_name}")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print("模型加载完成!")

# 简单问题
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "这张图片里有几台笔记本电脑？请简单回答。"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print(f"输入 tokens: {inputs['input_ids'].shape[1]}")

# 生成
print("\n分析中...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

# 解码
input_len = inputs["input_ids"].shape[1]
generated_ids = outputs[0][input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print(f"\n生成 tokens: {len(generated_ids)}")
print(f"\n回答: {response}")

# 调试
if not response.strip():
    print("\n[DEBUG] 完整输出:")
    full = processor.decode(outputs[0], skip_special_tokens=True)
    print(full[-500:])

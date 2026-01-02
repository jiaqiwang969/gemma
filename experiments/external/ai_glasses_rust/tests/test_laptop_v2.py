#!/usr/bin/env python3
"""
AI 眼镜系统 - 笔记本电脑数量识别

使用 Gemma 3n 分析视频关键帧
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
from pathlib import Path

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("  笔记本电脑数量识别")
print("=" * 60)

# 加载关键帧
keyframes_dir = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes"
images = []
for f in sorted(keyframes_dir.glob("frame_*.jpg")):
    img = Image.open(f).convert("RGB")
    images.append(img)
    print(f"加载: {f.name}")

print(f"\n共 {len(images)} 张关键帧")

# 加载模型
model_name = "google/gemma-3n-E2B-it"
print(f"\n[1] 加载模型: {model_name}")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print("模型加载完成!")

# 构建输入
print("\n[2] 构建输入...")
content = []
for i, img in enumerate(images):
    content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": f"[图{i+1}]"})

query = "以上图片来自同一个视频。请数一数视频中一共出现了几台笔记本电脑？描述每台笔记本的特征。"
content.append({"type": "text", "text": query})

messages = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# 移动到设备
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to(model.device)
pixel_values = inputs.get("pixel_values")
if pixel_values is not None:
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)

print(f"input_ids: {input_ids.shape}")
print(f"pixel_values: {pixel_values.shape if pixel_values is not None else 'None'}")

# 生成
print("\n[3] 分析中...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=300,
        do_sample=False,
    )

# 解码
generated_ids = outputs[0][input_ids.shape[1]:]
response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n生成 tokens: {len(generated_ids)}")
print("\n" + "=" * 60)
print("分析结果:")
print("=" * 60)
print(response)
print("=" * 60)

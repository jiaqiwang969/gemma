#!/usr/bin/env python3
"""
AI 眼镜系统 - 笔记本电脑数量识别测试

使用 Gemma 3n E2B 模型分析视频关键帧
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from pathlib import Path

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("  笔记本电脑数量识别测试")
print("=" * 60)

# 加载关键帧
keyframes_dir = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes"
images = []
for f in sorted(keyframes_dir.glob("frame_*.jpg")):
    img = Image.open(f).convert("RGB")
    images.append(img)
    print(f"加载: {f.name} ({img.size})")

print(f"\n共 {len(images)} 张关键帧")

# 加载模型
model_name = "google/gemma-3n-E2B-it"
print(f"\n[1] 加载处理器: {model_name}")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("[2] 加载模型...")
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print("模型加载完成!")

# 构建多模态输入
print("\n[3] 构建输入...")

content = []
for i, img in enumerate(images):
    content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": f"[图{i+1}]"})

query = """以上是同一个视频的关键帧。请仔细观察每一帧，数一数视频中一共出现了几台笔记本电脑？

请按以下格式回答：
1. 总共有 X 台笔记本电脑
2. 每台笔记本的描述（品牌、颜色、屏幕状态等）"""

content.append({"type": "text", "text": query})

messages = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print(f"输入 tokens: {inputs['input_ids'].shape[1]}")

# 生成
print("\n[4] 模型分析中...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
    )

# 解码
input_len = inputs["input_ids"].shape[1]
generated_ids = outputs[0][input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print(f"\n生成 tokens: {len(generated_ids)}")
print("\n" + "=" * 60)
print("分析结果:")
print("=" * 60)
print(response)
print("=" * 60)

# 如果为空，打印完整输出调试
if not response.strip():
    print("\n[DEBUG] 完整输出:")
    full = processor.decode(outputs[0], skip_special_tokens=True)
    print(full[-1000:] if len(full) > 1000 else full)

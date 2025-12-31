"""
测试 Gemma 3n 的图像理解能力 - 方法2
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("测试 Gemma 3n 多模态能力 - 图像理解 v2")
print("=" * 60)

model_name = "google/gemma-3n-E2B-it"

print("\n[1] 加载处理器和模型...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("[2] 下载测试图片...")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(f"    图片大小: {image.size}")

print("[3] 构建输入...")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# 移动到正确设备
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to(model.device)
pixel_values = inputs.get("pixel_values")
if pixel_values is not None:
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)

print(f"    input_ids shape: {input_ids.shape}")
print(f"    pixel_values: {pixel_values.shape if pixel_values is not None else 'None'}")

print("[4] 生成回答...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=256,
        do_sample=False,
    )

print("[5] 解码结果...")
# 只解码新生成的部分
generated_ids = outputs[0][input_ids.shape[1]:]
response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n" + "=" * 60)
print("图像描述结果:")
print("=" * 60)
print(response)
print("=" * 60)

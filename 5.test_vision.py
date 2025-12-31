"""
测试 Gemma 3n 的图像理解能力
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("测试 Gemma 3n 多模态能力 - 图像理解")
print("=" * 60)

# 使用专门的多模态处理器和模型
model_name = "google/gemma-3n-E2B-it"

print("\n[1] 加载处理器...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("[2] 加载多模态模型...")
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("[3] 加载测试图片...")
image_path = "/tmp/test_image.jpg"
image = Image.open(image_path)
print(f"    图片大小: {image.size}")

print("[4] 构建多模态输入...")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What do you see in this image? Describe it briefly."}
        ]
    }
]

# 使用处理器处理输入
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print("[5] 生成回答...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        use_cache=True,
    )

# 解码输出
input_len = inputs["input_ids"].shape[1]
generated_ids = outputs[0][input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print("\n" + "=" * 60)
print("图像描述结果:")
print("=" * 60)
print(f"生成了 {len(generated_ids)} 个 tokens")
print(f"回答: {response}")
print("=" * 60)

# 也打印完整输出用于调试
full_response = processor.decode(outputs[0], skip_special_tokens=True)
print("\n完整输出:")
print(full_response[-500:] if len(full_response) > 500 else full_response)

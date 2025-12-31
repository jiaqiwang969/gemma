"""
示例1: 文本推理 - 交互式对话
注意: Gemma 3n 需要图片输入，纯文本使用空白占位图
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("示例1: Gemma 3n 文本推理")
print("=" * 60)

model_name = "google/gemma-3n-E2B-it"

print("\n[1] 加载模型...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
print("    模型加载完成!")

# 空白占位图（Gemma 3n 多模态模型需要图片输入）
dummy_image = Image.new('RGB', (64, 64), color='white')

print("\n" + "=" * 60)
print("开始交互式问答 (输入 'quit' 退出)")
print("=" * 60)

while True:
    print()
    question = input("请输入问题: ").strip()
    if question.lower() in ['quit', 'exit', 'q']:
        print("再见!")
        break
    if not question:
        continue

    print(f"\n问题: {question}")
    print("-" * 40)

    # 使用占位图片 + 忽略指令
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": f"Ignore the blank image. {question}"}
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

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device, dtype=model.dtype)

    print("生成中...")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=200,
            do_sample=False,
        )

    response = processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n回答: {response}")
    print("-" * 40)

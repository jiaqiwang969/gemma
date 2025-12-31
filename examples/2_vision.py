"""
示例2: 图像理解 - 测试模型的视觉能力
支持：本地图片路径 或 URL
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("示例2: Gemma 3n 图像理解")
print("=" * 60)

model_name = "google/gemma-3n-E2B-it"

print("\n[1] 加载模型...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
print("    模型加载完成!")

def load_image(path_or_url):
    """加载本地图片或URL图片"""
    if path_or_url.startswith(('http://', 'https://')):
        return Image.open(requests.get(path_or_url, stream=True).raw)
    else:
        return Image.open(path_or_url)

def display_image_info(image, source):
    """显示图片信息"""
    print(f"\n图片来源: {source}")
    print(f"图片大小: {image.size[0]} x {image.size[1]}")
    print(f"图片模式: {image.mode}")

def ask_about_image(image, question):
    """对图片进行问答"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
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

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=300,
            do_sample=False,
        )

    return processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

# 默认测试图片
default_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"

print("\n" + "=" * 60)
print("图像理解交互模式")
print("=" * 60)
print("\n输入图片路径或URL (直接回车使用默认蜜蜂图片)")
print("输入 'quit' 退出")

current_image = None
current_source = None

while True:
    print()

    if current_image is None:
        path = input("图片路径/URL: ").strip()
        if path.lower() in ['quit', 'exit', 'q']:
            print("再见!")
            break

        if not path:
            path = default_url
            print(f"使用默认图片: {path}")

        try:
            current_image = load_image(path)
            current_source = path
            display_image_info(current_image, current_source)

            # 显示图片 (如果可能)
            try:
                current_image.show()
                print("(已在新窗口打开图片)")
            except:
                pass

        except Exception as e:
            print(f"加载图片失败: {e}")
            continue

    print("\n可选操作:")
    print("  1. 输入问题 - 询问关于图片的问题")
    print("  2. 输入 'new' - 加载新图片")
    print("  3. 输入 'quit' - 退出")

    action = input("\n请输入问题或命令: ").strip()

    if action.lower() in ['quit', 'exit', 'q']:
        print("再见!")
        break
    elif action.lower() == 'new':
        current_image = None
        current_source = None
        continue
    elif not action:
        action = "Describe this image in detail."
        print(f"使用默认问题: {action}")

    print(f"\n问题: {action}")
    print("-" * 40)
    print("分析图片中...")

    response = ask_about_image(current_image, action)

    print(f"\n回答:\n{response}")
    print("-" * 40)

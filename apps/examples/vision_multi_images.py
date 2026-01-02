"""
示例2b: 多图批量处理 - Gemma 3n 可以一次处理多张图片

关于图片数量限制：
  - Gemma 3n 没有硬性的"最大图片数量"限制
  - 限制取决于总体 context token 数量
  - Gemma 3n 上下文限制: 32K tokens (输入 + 输出)
  - 每张图片大约占用 256-512 tokens（取决于分辨率）
  - 理论上可以处理 50+ 张图片，但实际受内存限制

支持：
  1. 多张图片作为一个对话输入（比较/分析多张图）
  2. 批量独立处理多张图片
"""
import os
from pathlib import Path
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Gemma 3n 上下文限制
MAX_CONTEXT_TOKENS = 32768  # 32K tokens

print("=" * 60)
print("示例2b: Gemma 3n 多图批量处理")
print("=" * 60)
print(f"上下文限制: {MAX_CONTEXT_TOKENS:,} tokens (32K)")
print("图片数量: 无硬性限制，取决于总 token 数")
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

# 图片路径
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "assets" / "data" / "images"
IMAGE_FILES = ["cat.jpg", "dog.jpg", "food.jpg", "bee.jpg"]


def load_images(image_paths):
    """加载多张图片"""
    images = []
    for path in image_paths:
        full_path = Path(path) if os.path.isabs(path) else DATA_DIR / path
        if full_path.exists():
            img = Image.open(full_path).convert("RGB")
            images.append((path, img))
            print(f"    已加载: {path} ({img.size[0]}x{img.size[1]})")
        else:
            print(f"    跳过: {path} (文件不存在)")
    return images


# ============================================================
# 方法1: 多张图片在同一个对话中（用于比较/分析多张图）
# ============================================================
def analyze_multiple_images(images, question):
    """
    将多张图片放入同一个对话中进行分析
    适用于：比较图片、找不同、描述多个物体等
    """
    print(f"\n{'='*60}")
    print(f"多图联合分析 ({len(images)} 张图片)")
    print(f"{'='*60}")
    print(f"问题: {question}")

    # 构建包含多张图片的消息
    content = []
    for name, img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

    start_time = time.time()

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

    input_tokens = input_ids.shape[1]
    remaining_tokens = MAX_CONTEXT_TOKENS - input_tokens

    print(f"\n📊 Token 统计:")
    print(f"    输入 tokens: {input_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({input_tokens/MAX_CONTEXT_TOKENS*100:.1f}%)")
    print(f"    剩余空间: {remaining_tokens:,} tokens")
    print(f"    图片张量: {pixel_values.shape}")

    max_new_tokens = min(500, remaining_tokens - 100)  # 预留一些空间

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    elapsed = time.time() - start_time
    output_tokens = len(outputs[0]) - input_tokens
    total_tokens = input_tokens + output_tokens
    response = processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n回答:\n{'-'*40}")
    print(response)
    print(f"{'-'*40}")
    print(f"输出 tokens: {output_tokens:,}")
    print(f"总计 tokens: {total_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({total_tokens/MAX_CONTEXT_TOKENS*100:.1f}%)")
    print(f"耗时: {elapsed:.2f}s")

    return response


# ============================================================
# 方法2: 批量独立处理每张图片
# ============================================================
def batch_process_images(images, question):
    """
    独立处理每张图片，使用相同的问题
    适用于：批量描述、批量分类等
    """
    print(f"\n{'='*60}")
    print(f"批量独立处理 ({len(images)} 张图片)")
    print(f"{'='*60}")
    print(f"统一问题: {question}")

    results = []
    total_start = time.time()

    for i, (name, img) in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] 处理: {name}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question}
                ]
            }
        ]

        start_time = time.time()

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

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=200,
                do_sample=False,
            )

        elapsed = time.time() - start_time
        response = processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        results.append({
            "image": name,
            "response": response,
            "time": elapsed
        })

        print(f"    回答: {response[:100]}..." if len(response) > 100 else f"    回答: {response}")
        print(f"    耗时: {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"批量处理完成! 总耗时: {total_elapsed:.2f}s")
    print(f"平均每张: {total_elapsed/len(images):.2f}s")

    return results


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("\n[2] 加载测试图片...")
    images = load_images(IMAGE_FILES)

    if len(images) < 2:
        print("错误: 需要至少2张图片进行测试")
        exit(1)

    print(f"\n成功加载 {len(images)} 张图片")

    # 演示1: 多图联合分析（比较两张图）
    print("\n" + "=" * 60)
    print("演示1: 比较两张图片 (cat.jpg vs dog.jpg)")
    print("=" * 60)

    two_images = [img for img in images if img[0] in ["cat.jpg", "dog.jpg"]]
    if len(two_images) == 2:
        analyze_multiple_images(
            two_images,
            "Compare these two images. What animals do you see? What are the similarities and differences between them?"
        )

    # 演示2: 多图联合分析（分析所有图片）
    print("\n" + "=" * 60)
    print("演示2: 同时分析多张图片")
    print("=" * 60)

    # 只用前3张图避免内存问题
    subset = images[:3]
    analyze_multiple_images(
        subset,
        f"I'm showing you {len(subset)} images. Please describe each image briefly and tell me what they have in common."
    )

    # 演示3: 批量独立处理
    print("\n" + "=" * 60)
    print("演示3: 批量独立处理每张图片")
    print("=" * 60)

    batch_process_images(
        images,
        "What is the main subject of this image? Answer in one sentence."
    )

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)

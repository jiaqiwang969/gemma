#!/usr/bin/env python3
"""
Cambrian-S 模型测试 (使用官方加载方式)

测试 Cambrian-S 的视频理解能力
"""
import os
import sys
import torch
from pathlib import Path
from PIL import Image

# 添加 cambrian-s 到路径
cambrian_path = Path(__file__).parent.parent.parent / "cambrian-s"
sys.path.insert(0, str(cambrian_path))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def load_cambrian_model(model_path: str, device: str = "mps"):
    """使用官方方式加载 Cambrian-S 模型"""
    from cambrian.model.builder import load_pretrained_model

    print(f"加载模型: {model_path}")
    print(f"设备: {device}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="cambrian-s",
        device_map="auto",
        device=device,
    )

    return tokenizer, model, image_processor, context_len


def process_images(images, image_processor, model):
    """处理图像列表"""
    from cambrian.mm_utils import process_images as cambrian_process_images

    # 使用 model.config 来获取正确的配置
    model_cfg = model.config

    return cambrian_process_images(images, image_processor, model_cfg)


def main():
    print("=" * 70)
    print("  Cambrian-S 模型测试 (官方加载方式)")
    print("=" * 70)

    # 1. 检测设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # 2. 加载模型 (使用最小的 0.5B 模型)
    model_id = "nyu-visionx/Cambrian-S-0.5B"

    try:
        tokenizer, model, image_processor, context_len = load_cambrian_model(
            model_path=model_id,
            device=device
        )
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 加载测试图像
    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes"
    images = []

    print(f"\n加载关键帧: {keyframes_dir}")
    for f in sorted(keyframes_dir.glob("frame_*.jpg")):
        img = Image.open(f).convert("RGB")
        images.append(img)
        print(f"  {f.name}")

    if not images:
        print("没有找到关键帧图像")
        return

    # 4. 构建查询
    query = "这些图片来自同一个视频，请数一数视频中一共出现了几台笔记本电脑？"
    print(f"\n查询: {query}")

    # 5. 处理图像
    try:
        image_tensor = process_images(images, image_processor, model)
        print(f"图像处理完成: {image_tensor.shape if hasattr(image_tensor, 'shape') else type(image_tensor)}")
    except Exception as e:
        print(f"图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 生成回答
    try:
        from cambrian.conversation import conv_templates
        from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        # 构建对话
        conv = conv_templates["qwen"].copy()

        # 添加图像 token
        image_tokens = DEFAULT_IMAGE_TOKEN * len(images)
        prompt = f"{image_tokens}\n{query}"

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        full_prompt = conv.get_prompt()

        # tokenize
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

        # 生成
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(device, dtype=torch.float16),
                max_new_tokens=256,
                do_sample=False,
            )

        # 解码
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("\n" + "=" * 70)
        print("Cambrian-S 回答:")
        print("=" * 70)
        print(response)
        print("=" * 70)

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cambrian-S 模型测试

测试 Cambrian-S 的视频理解能力，特别是：
1. 惊讶度驱动的帧选择
2. 意图感知的视频理解
3. 笔记本电脑计数任务
"""
import os
import torch
from pathlib import Path
from PIL import Image

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def check_model_available():
    """检查模型是否可用"""
    from transformers import AutoConfig

    models_to_try = [
        "nyu-visionx/Cambrian-S-0.5B",  # 最小，0.9B
        "nyu-visionx/Cambrian-S-1.5B",  # 2B
        "nyu-visionx/Cambrian-S-3B",    # 3B
        "nyu-visionx/Cambrian-S-7B-LFP", # 带 LFP
    ]

    print("检查可用模型...")
    for model_id in models_to_try:
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            print(f"  ✓ {model_id} - 可用")
            return model_id
        except Exception as e:
            print(f"  ✗ {model_id} - {e}")

    return None


def load_cambrian_s(model_id: str):
    """加载 Cambrian-S 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    print(f"\n加载模型: {model_id}")

    # 尝试加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("  ✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        tokenizer = None

    # 尝试加载 processor (多模态)
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("  ✓ Processor 加载成功")
    except Exception as e:
        print(f"  ✗ Processor 加载失败: {e}")
        processor = None

    # 加载模型
    try:
        # 检测设备
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32

        print(f"  设备: {device}, dtype: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("  ✓ 模型加载成功")

        return model, tokenizer or processor, device

    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_video_understanding(model, processor, device, images, query):
    """测试视频理解"""
    print(f"\n测试查询: {query}")
    print(f"图像数量: {len(images)}")

    # 构建输入 (这个需要根据 Cambrian-S 的实际 API 调整)
    try:
        # 方式 1: 使用 processor
        if hasattr(processor, 'apply_chat_template'):
            content = []
            for i, img in enumerate(images):
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": f"[帧{i+1}]"})
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
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        else:
            # 方式 2: 简单的 tokenizer
            prompt = f"问题: {query}\n\n请分析这些视频帧并回答。"
            inputs = processor(prompt, return_tensors="pt").to(device)

        # 生成
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        # 解码
        if hasattr(processor, 'decode'):
            response = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("  Cambrian-S 模型测试")
    print("=" * 70)

    # 1. 检查模型
    model_id = check_model_available()
    if not model_id:
        print("\n没有可用的模型，尝试下载最小的模型...")
        model_id = "nyu-visionx/Cambrian-S-0.5B"

    # 2. 加载模型
    model, processor, device = load_cambrian_s(model_id)
    if model is None:
        print("\n模型加载失败，请检查网络或手动下载模型")
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

    # 4. 测试视频理解
    query = "这些图片来自同一个视频，请数一数视频中一共出现了几台笔记本电脑？"

    response = test_video_understanding(model, processor, device, images, query)

    if response:
        print("\n" + "=" * 70)
        print("Cambrian-S 回答:")
        print("=" * 70)
        print(response)
        print("=" * 70)


if __name__ == "__main__":
    main()

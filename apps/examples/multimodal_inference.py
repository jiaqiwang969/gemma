"""
示例2c: 多模态联合分析 - 同时处理 文字 + 图片 + 音频

Gemma 3n 支持在单次请求中同时处理：
  - 文本 (Text)
  - 图像 (Image) - 可多张
  - 音频 (Audio)

这使得模型可以：
  - 根据语音指令分析图片
  - 将图片内容与音频内容关联
  - 多模态上下文理解
"""
import os
from pathlib import Path
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import librosa
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Gemma 3n 上下文限制
MAX_CONTEXT_TOKENS = 32768  # 32K tokens

print("=" * 60)
print("示例2c: Gemma 3n 多模态联合分析")
print("=" * 60)
print("支持: 文字 + 图片 + 音频 同时输入")
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

# 数据路径
ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT_DIR / "assets" / "data" / "images"
AUDIO_DIR = ROOT_DIR / "assets" / "data" / "audio"


def load_image(filename):
    """加载图片"""
    path = IMAGE_DIR / filename
    if path.exists():
        img = Image.open(path).convert("RGB")
        print(f"    图片: {filename} ({img.size[0]}x{img.size[1]})")
        return img
    else:
        print(f"    图片不存在: {path}")
        return None


def load_audio(filename, target_sr=16000):
    """加载音频"""
    path = AUDIO_DIR / filename
    if path.exists():
        audio_array, sr = librosa.load(path, sr=target_sr)
        duration = len(audio_array) / target_sr
        print(f"    音频: {filename} ({duration:.2f}s, {target_sr}Hz)")
        return audio_array, target_sr
    else:
        print(f"    音频不存在: {path}")
        return None, None


def multimodal_analysis(text, images=None, audio=None):
    """
    多模态联合分析

    Args:
        text: 文本提示/问题
        images: [(name, PIL.Image), ...] 图片列表
        audio: (audio_array, sample_rate) 音频数据
    """
    print(f"\n{'='*60}")
    print("多模态联合分析")
    print(f"{'='*60}")

    # 统计输入
    has_text = bool(text)
    has_images = images and len(images) > 0
    has_audio = audio and audio[0] is not None

    modalities = []
    if has_text:
        modalities.append("文字")
    if has_images:
        modalities.append(f"图片x{len(images)}")
    if has_audio:
        modalities.append("音频")

    print(f"输入模态: {' + '.join(modalities)}")
    print(f"文本: {text[:100]}..." if len(text) > 100 else f"文本: {text}")

    # 构建消息内容
    content = []

    # 添加图片
    if has_images:
        for name, img in images:
            content.append({"type": "image", "image": img})

    # 添加音频
    if has_audio:
        audio_array, sr = audio
        content.append({"type": "audio", "audio": audio_array, "sample_rate": sr})

    # 添加文本
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]

    start_time = time.time()

    # 处理输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    input_tokens = input_ids.shape[1]
    remaining_tokens = MAX_CONTEXT_TOKENS - input_tokens

    print(f"\n📊 Token 统计:")
    print(f"    输入 tokens: {input_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({input_tokens/MAX_CONTEXT_TOKENS*100:.1f}%)")
    print(f"    剩余空间: {remaining_tokens:,} tokens")

    # 准备生成参数
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": min(500, remaining_tokens - 100),
        "do_sample": False,
    }

    # 添加图片数据
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        generate_kwargs["pixel_values"] = inputs["pixel_values"].to(model.device, dtype=model.dtype)
        print(f"    图片张量: {generate_kwargs['pixel_values'].shape}")

    # 添加音频数据
    if "input_features" in inputs and inputs["input_features"] is not None:
        generate_kwargs["input_features"] = inputs["input_features"].to(model.device, dtype=model.dtype)
        generate_kwargs["input_features_mask"] = inputs["input_features_mask"].to(model.device)
        print(f"    音频张量: {generate_kwargs['input_features'].shape}")

    print("\n生成回复中...")

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

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
# 主程序
# ============================================================
if __name__ == "__main__":
    print("\n[2] 加载测试数据...")

    # 加载图片
    cat_img = load_image("cat.jpg")
    dog_img = load_image("dog.jpg")
    food_img = load_image("food.jpg")

    # 加载音频
    audio_data = load_audio("mlk_speech.flac")

    # ============================================================
    # 演示1: 图片 + 文字
    # ============================================================
    print("\n" + "=" * 60)
    print("演示1: 图片 + 文字")
    print("=" * 60)

    if cat_img:
        multimodal_analysis(
            text="Describe this image in detail. What animal is it? What is it doing?",
            images=[("cat.jpg", cat_img)]
        )

    # ============================================================
    # 演示2: 音频 + 文字
    # ============================================================
    print("\n" + "=" * 60)
    print("演示2: 音频 + 文字")
    print("=" * 60)

    if audio_data[0] is not None:
        multimodal_analysis(
            text="Please transcribe this audio and summarize its main message.",
            audio=audio_data
        )

    # ============================================================
    # 演示3: 图片 + 音频 + 文字 (完整多模态)
    # ============================================================
    print("\n" + "=" * 60)
    print("演示3: 图片 + 音频 + 文字 (完整多模态)")
    print("=" * 60)

    if cat_img and audio_data[0] is not None:
        multimodal_analysis(
            text="I'm showing you an image and playing an audio clip. "
                 "First, describe what you see in the image. "
                 "Then, transcribe what you hear in the audio. "
                 "Finally, tell me if there's any connection between them.",
            images=[("cat.jpg", cat_img)],
            audio=audio_data
        )

    # ============================================================
    # 演示4: 多张图片 + 音频 + 文字
    # ============================================================
    print("\n" + "=" * 60)
    print("演示4: 多张图片 + 音频 + 文字")
    print("=" * 60)

    images_list = []
    if cat_img:
        images_list.append(("cat.jpg", cat_img))
    if dog_img:
        images_list.append(("dog.jpg", dog_img))

    if len(images_list) >= 2 and audio_data[0] is not None:
        multimodal_analysis(
            text="I'm showing you two images (a cat and a dog) and playing an audio clip. "
                 "Compare the two animals in the images. "
                 "Also transcribe the audio and tell me the topic.",
            images=images_list,
            audio=audio_data
        )

    # ============================================================
    # 演示5: 纯文字 (基线对比)
    # ============================================================
    print("\n" + "=" * 60)
    print("演示5: 纯文字 (基线对比)")
    print("=" * 60)

    # 纯文字需要 dummy image (Gemma 3n 的限制)
    from PIL import Image as PILImage
    dummy = PILImage.new('RGB', (64, 64), color='white')

    multimodal_analysis(
        text="Ignore the blank image. What are the key differences between cats and dogs as pets?",
        images=[("dummy", dummy)]
    )

    print("\n" + "=" * 60)
    print("所有多模态演示完成!")
    print("=" * 60)

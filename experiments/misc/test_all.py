"""
Gemma 3n 统一测试脚本
解决纯文本输入问题：使用空白图片作为占位符
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import librosa
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class Gemma3nChat:
    def __init__(self):
        self.model_name = "google/gemma-3n-E2B-it"
        self.model = None
        self.processor = None
        # 用于纯文本对话的占位图片
        self.dummy_image = Image.new('RGB', (64, 64), color='white')

    def load(self):
        print("加载模型...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
            max_memory={"mps": "64GiB", "cpu": "64GiB"},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        print("模型加载完成!")

    def chat(self, text, image=None, audio=None):
        """
        统一的聊天接口
        - text: 文本问题
        - image: PIL Image 或 None
        - audio: (audio_array, sample_rate) 或 None
        """
        content = []

        # 如果没有图片和音频，使用占位图片
        if image is None and audio is None:
            content.append({"type": "image", "image": self.dummy_image})
            # 告诉模型忽略图片
            text = "Ignore the blank image. " + text

        # 添加图片
        if image is not None:
            content.append({"type": "image", "image": image})

        # 添加音频
        if audio is not None:
            audio_array, sr = audio
            content.append({"type": "audio", "audio": audio_array, "sample_rate": sr})

        # 添加文本
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        # 处理输入
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 准备生成参数
        generate_kwargs = {
            "input_ids": inputs["input_ids"].to(self.model.device),
            "attention_mask": inputs["attention_mask"].to(self.model.device),
            "max_new_tokens": 512,
            "do_sample": False,
        }

        # 处理图片
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            generate_kwargs["pixel_values"] = inputs["pixel_values"].to(
                self.model.device, dtype=self.model.dtype
            )

        # 处理音频
        if "input_features" in inputs and inputs["input_features"] is not None:
            generate_kwargs["input_features"] = inputs["input_features"].to(
                self.model.device, dtype=self.model.dtype
            )
            generate_kwargs["input_features_mask"] = inputs["input_features_mask"].to(
                self.model.device
            )

        # 生成
        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        # 解码
        input_len = inputs["input_ids"].shape[1]
        response = self.processor.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return response


def main():
    print("=" * 60)
    print("Gemma 3n 多模态聊天测试")
    print("=" * 60)

    chat = Gemma3nChat()
    chat.load()

    # 测试1: 纯文本
    print("\n" + "=" * 60)
    print("测试1: 纯文本对话")
    print("=" * 60)
    response = chat.chat("法国的首都是哪里？请用中文简短回答。")
    print(f"回答: {response}")

    # 测试2: 图片理解
    print("\n" + "=" * 60)
    print("测试2: 图片理解")
    print("=" * 60)
    image_path = "assets/data/images/bee.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        print(f"图片: {image_path} ({image.size})")
        response = chat.chat("描述这张图片的内容。", image=image)
        print(f"回答: {response}")
    else:
        print(f"图片不存在: {image_path}")

    # 测试3: 音频转录
    print("\n" + "=" * 60)
    print("测试3: 音频转录")
    print("=" * 60)
    audio_path = "assets/data/audio/mlk_speech.flac"
    if os.path.exists(audio_path):
        audio_array, sr = librosa.load(audio_path, sr=16000)
        print(f"音频: {audio_path} ({len(audio_array)/sr:.1f}秒)")
        response = chat.chat("请转录这段音频。", audio=(audio_array, sr))
        print(f"回答: {response}")
    else:
        print(f"音频不存在: {audio_path}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
示例3: 音频理解 - 测试模型的语音转录能力
支持：本地音频文件 或 URL
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import requests
import librosa
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("示例3: Gemma 3n 音频理解")
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

def load_audio(path_or_url):
    """加载本地音频或URL音频，返回 (audio_array, sample_rate)"""
    if path_or_url.startswith(('http://', 'https://')):
        # 下载到临时文件
        temp_path = "/tmp/temp_audio_file"
        response = requests.get(path_or_url)
        with open(temp_path, "wb") as f:
            f.write(response.content)
        path_or_url = temp_path

    # 加载并重采样到 16kHz
    audio_array, sr = librosa.load(path_or_url, sr=16000)
    return audio_array, sr

def display_audio_info(audio_array, sr, source):
    """显示音频信息"""
    duration = len(audio_array) / sr
    print(f"\n音频来源: {source}")
    print(f"采样率: {sr} Hz")
    print(f"时长: {duration:.2f} 秒")
    print(f"样本数: {len(audio_array)}")

def ask_about_audio(audio_array, sr, question):
    """对音频进行问答"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array, "sample_rate": sr},
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
    input_features = inputs["input_features"].to(model.device, dtype=model.dtype)
    input_features_mask = inputs["input_features_mask"].to(model.device)

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            input_features_mask=input_features_mask,
            max_new_tokens=300,
            do_sample=False,
        )

    return processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

# 默认测试音频 (MLK "I Have a Dream" 片段)
default_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

print("\n" + "=" * 60)
print("音频理解交互模式")
print("=" * 60)
print("\n输入音频路径或URL (直接回车使用默认 MLK 演讲)")
print("支持格式: wav, mp3, flac, ogg 等")
print("输入 'quit' 退出")

current_audio = None
current_sr = None
current_source = None

while True:
    print()

    if current_audio is None:
        path = input("音频路径/URL: ").strip()
        if path.lower() in ['quit', 'exit', 'q']:
            print("再见!")
            break

        if not path:
            path = default_url
            print(f"使用默认音频: MLK 'I Have a Dream' 演讲片段")

        try:
            print("加载音频中...")
            current_audio, current_sr = load_audio(path)
            current_source = path
            display_audio_info(current_audio, current_sr, current_source)

        except Exception as e:
            print(f"加载音频失败: {e}")
            continue

    print("\n可选操作:")
    print("  1. 输入问题 - 询问关于音频的问题")
    print("  2. 输入 'new' - 加载新音频")
    print("  3. 输入 'quit' - 退出")
    print("\n常用问题示例:")
    print("  - Please transcribe this audio.")
    print("  - What language is spoken in this audio?")
    print("  - Summarize the content of this audio.")

    action = input("\n请输入问题或命令: ").strip()

    if action.lower() in ['quit', 'exit', 'q']:
        print("再见!")
        break
    elif action.lower() == 'new':
        current_audio = None
        current_sr = None
        current_source = None
        continue
    elif not action:
        action = "Please transcribe this audio."
        print(f"使用默认问题: {action}")

    print(f"\n问题: {action}")
    print("-" * 40)
    print("分析音频中...")

    response = ask_about_audio(current_audio, current_sr, action)

    print(f"\n回答:\n{response}")
    print("-" * 40)

"""
测试 Gemma 3n 的音频理解能力
"""
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import requests
import librosa
import numpy as np

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("测试 Gemma 3n 多模态能力 - 音频理解")
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

print("[2] 下载测试音频...")
# 使用 LibriSpeech 的测试音频
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
audio_path = "/tmp/test_audio.flac"

# 下载音频文件
response = requests.get(audio_url)
with open(audio_path, "wb") as f:
    f.write(response.content)

# 加载音频
audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
print(f"    音频采样率: {sampling_rate} Hz")
print(f"    音频长度: {len(audio_array) / sampling_rate:.2f} 秒")

print("[3] 构建输入...")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_array, "sample_rate": sampling_rate},
            {"type": "text", "text": "Please transcribe this audio and describe what you hear."}
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

# 处理音频特征
input_features = inputs.get("input_features")
input_features_mask = inputs.get("input_features_mask")
if input_features is not None:
    input_features = input_features.to(model.device, dtype=model.dtype)
if input_features_mask is not None:
    input_features_mask = input_features_mask.to(model.device)

print(f"    input_ids shape: {input_ids.shape}")
print(f"    input_features: {input_features.shape if input_features is not None else 'None'}")
print(f"    input_features_mask: {input_features_mask.shape if input_features_mask is not None else 'None'}")

print("[4] 生成回答...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        input_features_mask=input_features_mask,
        max_new_tokens=256,
        do_sample=False,
    )

print("[5] 解码结果...")
# 只解码新生成的部分
generated_ids = outputs[0][input_ids.shape[1]:]
response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n" + "=" * 60)
print("音频转录结果:")
print("=" * 60)
print(response)
print("=" * 60)

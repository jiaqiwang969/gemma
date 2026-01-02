#!/usr/bin/env python3
"""
Debug audio processing in Gemma 3n
"""
import os
import torch
import warnings
import librosa

warnings.filterwarnings("ignore")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from transformers import AutoProcessor, Gemma3nForConditionalGeneration

print("Loading model...")
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio duration: {len(audio)/sr:.2f}s")

# Get inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio, "sample_rate": sr},
            {"type": "text", "text": "Transcribe this audio."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

input_features = inputs["input_features"]
input_features_mask = inputs["input_features_mask"]

print(f"input_features shape: {input_features.shape}")
print(f"input_features stats: min={input_features.min():.4f}, max={input_features.max():.4f}")

# Check the actual forward flow by looking at get_audio_features
print("\n=== Checking get_audio_features ===")
with torch.inference_mode():
    # Use the model's actual method
    audio_features, audio_attention_mask = model.model.get_audio_features(
        input_features,
        ~input_features_mask  # Note: get_audio_features expects inverted mask
    )
    print(f"audio_features shape: {audio_features.shape}")
    print(f"audio_features stats: min={audio_features.min().item():.6f}, max={audio_features.max().item():.6f}, mean={audio_features.float().mean().item():.6f}")
    print(f"First 10 values [0,0,:10]: {audio_features[0,0,:10].float().cpu().numpy()}")
    print(f"Last 10 values [0,-1,-10:]: {audio_features[0,-1,-10:].float().cpu().numpy()}")

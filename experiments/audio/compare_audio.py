#!/usr/bin/env python3
"""
Compare PyTorch and llama.cpp audio features for Gemma 3n
"""
import os
import sys
import torch
import numpy as np
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
print(f"Audio duration: {len(audio)/sr:.2f}s, samples: {len(audio)}")

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

print(f"\ninput_features shape: {input_features.shape}")
print(f"input_features_mask shape: {input_features_mask.shape}")
print(f"input_features stats: min={input_features.min():.4f}, max={input_features.max():.4f}, mean={input_features.float().mean():.4f}")

# Get audio features step by step
with torch.inference_mode():
    # Step 1: Audio tower
    audio_outputs, audio_mask = model.model.audio_tower(input_features, input_features_mask)
    print(f"\n=== audio_tower output ===")
    print(f"shape: {audio_outputs.shape}")
    print(f"stats: min={audio_outputs.min().item():.6f}, max={audio_outputs.max().item():.6f}, mean={audio_outputs.float().mean().item():.6f}")
    print(f"first 10 values [0,0,:10]: {audio_outputs[0,0,:10].float().cpu().numpy()}")
    print(f"last 10 values [0,-1,-10:]: {audio_outputs[0,-1,-10:].float().cpu().numpy()}")

    # Step 2: embed_audio
    final_embeds = model.model.embed_audio(inputs_embeds=audio_outputs)
    print(f"\n=== embed_audio output ===")
    print(f"shape: {final_embeds.shape}")
    print(f"stats: min={final_embeds.min().item():.6f}, max={final_embeds.max().item():.6f}, mean={final_embeds.float().mean().item():.6f}")
    print(f"first 10 values [0,0,:10]: {final_embeds[0,0,:10].float().cpu().numpy()}")
    print(f"last 10 values [0,-1,-10:]: {final_embeds[0,-1,-10:].float().cpu().numpy()}")

# Now try generation
print("\n=== Generating with PyTorch ===")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
)

response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {response[-500:]}")

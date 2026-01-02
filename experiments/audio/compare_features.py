#!/usr/bin/env python3
"""
Compare PyTorch and llama.cpp audio embeddings
"""
import os
import torch
import warnings
import librosa
import numpy as np

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

# Get the audio features
print("\n=== PyTorch audio_features ===")
with torch.inference_mode():
    audio_features, audio_attention_mask = model.model.get_audio_features(
        input_features,
        ~input_features_mask
    )
    audio_np = audio_features[0].float().cpu().numpy()  # [82, 2048]
    print(f"shape: {audio_np.shape}")
    print(f"stats: min={audio_np.min():.6f}, max={audio_np.max():.6f}, mean={audio_np.mean():.6f}")

    # Print first few values in a format easy to compare with llama.cpp
    print("\nFirst token (token 0), first 20 values:")
    print(np.array2string(audio_np[0, :20], precision=4, separator=', '))

    print("\nLast token (token 81), first 20 values:")
    print(np.array2string(audio_np[-1, :20], precision=4, separator=', '))

    # Save to file for comparison
    np.save('/tmp/pytorch_audio_features.npy', audio_np)
    print(f"\nSaved to /tmp/pytorch_audio_features.npy")

# Also check input_features (mel spectrogram)
print("\n=== Input features (mel spectrogram) ===")
mel_np = input_features[0].float().cpu().numpy()  # [1297, 128]
print(f"shape: {mel_np.shape}")
print(f"stats: min={mel_np.min():.6f}, max={mel_np.max():.6f}, mean={mel_np.mean():.6f}")
print("\nFirst frame (frame 0), first 20 mel bins:")
print(np.array2string(mel_np[0, :20], precision=4, separator=', '))
print("\nFrame 100, first 20 mel bins:")
print(np.array2string(mel_np[100, :20], precision=4, separator=', '))

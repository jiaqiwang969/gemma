#!/usr/bin/env python3
"""
Compare mel spectrogram between PyTorch and llama.cpp
"""
import os
import torch
import numpy as np
import warnings
import librosa
from scipy.io import wavfile

warnings.filterwarnings("ignore")

from transformers import AutoProcessor

print("Loading processor...")
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio duration: {len(audio)/sr:.2f}s, samples: {len(audio)}")

# Get mel from feature extractor
fe = processor.feature_extractor
print(f"\nFeature extractor settings:")
print(f"  fft_length: {fe.fft_length}")
print(f"  frame_length: {fe.frame_length}")
print(f"  hop_length: {fe.hop_length}")
print(f"  preemphasis: {fe.preemphasis}")
print(f"  preemphasis_htk_flavor: {fe.preemphasis_htk_flavor}")
print(f"  min_frequency: {fe.min_frequency}")
print(f"  max_frequency: {fe.max_frequency}")
print(f"  do_normalize: {fe.do_normalize}")
print(f"  mel_floor: {fe.mel_floor}")

# Call the feature extractor directly
# Look at the _extract_fbank_features method
print("\n=== Extracting features ===")

# Check what method is used
import inspect
if hasattr(fe, '_extract_fbank_features'):
    print("Using _extract_fbank_features method")
    print(inspect.signature(fe._extract_fbank_features))

# Call through __call__
# Need to wrap audio in batch
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
print(f"Input audio shape: {audio_tensor.shape}")

# Get the mel spectrogram directly
try:
    # Call with return_attention_mask=True
    result = fe(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    for key, val in result.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}")
            if val.numel() > 0:
                print(f"    min={val.min():.4f}, max={val.max():.4f}, mean={val.float().mean():.4f}")
except Exception as e:
    print(f"Error: {e}")

# Try to manually compute mel to understand the pipeline
print("\n=== Manual mel computation ===")

# Save mel for comparison
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
)

input_features = inputs["input_features"]
print(f"input_features shape: {input_features.shape}")
print(f"input_features stats: min={input_features.min():.4f}, max={input_features.max():.4f}, mean={input_features.float().mean():.4f}")

# Save first frame
print(f"\nFirst 10 frames, first 10 mel bins:")
mel_np = input_features[0].float().cpu().numpy()
for i in range(min(10, mel_np.shape[0])):
    print(f"  frame {i}: {mel_np[i, :10]}")

# Save to file
np.save('/tmp/pytorch_mel.npy', mel_np)
print(f"\nSaved mel to /tmp/pytorch_mel.npy, shape: {mel_np.shape}")

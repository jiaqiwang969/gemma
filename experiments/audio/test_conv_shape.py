#!/usr/bin/env python3
"""
Compare PyTorch conv2d output with what we expect from llama.cpp
"""
import torch
import numpy as np
from transformers import AutoModelForImageTextToText
import soundfile as sf

# Load audio mel
audio, sr = sf.read('/tmp/mlk_speech_16k.wav')
audio = audio.astype(np.float32)

model_name = "google/gemma-3n-E2B-it"
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio, "sample_rate": sr},
        {"type": "text", "text": "Transcribe."}
    ]}
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

mel = inputs['input_features']  # [1, 1297, 128]
print(f"Mel shape: {mel.shape}")
print(f"Mel first 5 values: {mel[0, 0, :5].numpy()}")

# Get conv weights
subsample = model.model.audio_tower.subsample_conv_projection
conv0_weight = subsample.conv_0.conv.weight.detach()  # [128, 1, 3, 3]
print(f"\nConv0 weight shape: {conv0_weight.shape}")
print(f"Conv0 weight first values: {conv0_weight[0, 0, 0, :].numpy()}")

# Save weights for llama.cpp comparison
np.save('/tmp/conv0_weight.npy', conv0_weight.numpy())
print(f"Saved conv0 weight to /tmp/conv0_weight.npy")

# Manual conv2d
import torch.nn.functional as F

# Input: [B, T, F] -> [B, 1, T, F]
x = mel.unsqueeze(1)  # [1, 1, 1297, 128]
print(f"\nInput for conv: {x.shape}")

# Padding: F.pad(x, (1, 1, 0, 2)) adds 1 left/right on width (F), 0 top/2 bottom on height (T)
x_padded = F.pad(x, (1, 1, 0, 2))
print(f"After padding: {x_padded.shape}")  # [1, 1, 1299, 130]

# Conv2d with stride 2
conv0_out = F.conv2d(x_padded, conv0_weight, stride=2)
print(f"Conv0 output: {conv0_out.shape}")  # Expected: [1, 128, 649, 64]
print(f"Conv0 output first 5 values: {conv0_out[0, 0, 0, :5].numpy()}")

# Save the mel and conv output for llama.cpp comparison
np.save('/tmp/pytorch_conv0_input.npy', x_padded.numpy())
np.save('/tmp/pytorch_conv0_output.npy', conv0_out.numpy())

print("\n=== Key shapes for llama.cpp ===")
print("PyTorch convention: [B, C, H, W] where H=time, W=freq")
print(f"Input padded: {x_padded.shape} = [B={x_padded.shape[0]}, C={x_padded.shape[1]}, H={x_padded.shape[2]}, W={x_padded.shape[3]}]")
print(f"Output: {conv0_out.shape} = [B={conv0_out.shape[0]}, C={conv0_out.shape[1]}, H={conv0_out.shape[2]}, W={conv0_out.shape[3]}]")
print(f"Weight: {conv0_weight.shape} = [OC={conv0_weight.shape[0]}, IC={conv0_weight.shape[1]}, KH={conv0_weight.shape[2]}, KW={conv0_weight.shape[3]}]")

print("\n=== ggml mapping ===")
print("ggml tensor is column-major: ne[0] is innermost (stride 1)")
print("For ggml_conv_2d, data should be [W, H, C, B] (from ggml's perspective)")
print(f"So mel should be stored as [{x_padded.shape[3]}, {x_padded.shape[2]}, {x_padded.shape[1]}, {x_padded.shape[0]}] = [W=freq, H=time, C=1, B=1]")

#!/usr/bin/env python3
"""
Test PyTorch model with the same audio
"""
import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load audio
audio, sr = sf.read('/tmp/mlk_speech_16k.wav')
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)} samples at {sr} Hz")

# Load model
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Create message with audio
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio, "sample_rate": sr},
            {"type": "text", "text": "Transcribe this audio."}
        ]
    }
]

# Process
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print(f"\nInput keys: {inputs.keys()}")
if 'input_features' in inputs:
    mel = inputs['input_features']
    print(f"Mel shape: {mel.shape}")
    print(f"Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

# Generate
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# Decode
output_text = processor.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{output_text}")

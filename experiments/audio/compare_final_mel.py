#!/usr/bin/env python3
"""
Compare mel spectrograms from different audio inputs
"""
import numpy as np
import librosa
from transformers import AutoProcessor

# Load audio
audio_path = "assets/data/audio/mlk_speech.flac"
audio_librosa, sr = librosa.load(audio_path, sr=16000)

# Get PyTorch mel spectrogram
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

messages = [{"role": "user", "content": [
    {"type": "audio", "audio": audio_librosa, "sample_rate": sr},
    {"type": "text", "text": "Transcribe."}
]}]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=True, return_dict=True, return_tensors="pt")

pytorch_mel = inputs["input_features"][0].float().cpu().numpy()

print(f"PyTorch mel shape: {pytorch_mel.shape}")
print(f"PyTorch mel stats: min={pytorch_mel.min():.4f}, max={pytorch_mel.max():.4f}, mean={pytorch_mel.mean():.4f}")
print(f"PyTorch mel first frame: {pytorch_mel[0, :10]}")

# llama.cpp output (from debug):
# n_mel=128, n_len=1297
# stats: min=-9.9682, max=4.1866, mean=-1.2410
# frame 0: [-5.0397, -3.9858, -3.6777, -3.8044, -4.1889, -3.9665, -3.8591, -3.8898, -3.2134, -2.7919]

print(f"\nllama.cpp mel stats:")
print(f"  n_mel=128, n_len=1297")
print(f"  min=-9.9682, max=4.1866, mean=-1.2410")
print(f"  frame 0: [-5.0397, -3.9858, -3.6777, -3.8044, -4.1889, -3.9665, -3.8591, -3.8898, -3.2134, -2.7919]")

# Compare
print(f"\nComparison:")
print(f"  Mean difference: {abs(-1.2410 - pytorch_mel.mean()):.4f}")
print(f"  Frame 0 difference (first 10 bins):")
llama_frame0 = np.array([-5.0397, -3.9858, -3.6777, -3.8044, -4.1889, -3.9665, -3.8591, -3.8898, -3.2134, -2.7919])
for i in range(10):
    print(f"    bin {i}: PyTorch={pytorch_mel[0, i]:.4f}, llama.cpp={llama_frame0[i]:.4f}, diff={abs(pytorch_mel[0, i] - llama_frame0[i]):.4f}")

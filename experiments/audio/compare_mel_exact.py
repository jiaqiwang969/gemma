#!/usr/bin/env python3
"""
Exact reproduction of Gemma 3n mel spectrogram computation
"""
import numpy as np
import librosa
from transformers import AutoProcessor

# Load audio
audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

# Get PyTorch result
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
fe = processor.feature_extractor

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

pytorch_mel = inputs["input_features"][0].float().cpu().numpy()  # [n_frames, n_mel]
print(f"\nPyTorch mel shape: {pytorch_mel.shape}")
print(f"PyTorch mel stats: min={pytorch_mel.min():.4f}, max={pytorch_mel.max():.4f}, mean={pytorch_mel.mean():.4f}")

# Now compute EXACTLY like PyTorch
print("\n=== Exact manual computation ===")

fft_length = 1024
frame_length = 512
hop_length = 160
preemphasis = 0.97
mel_floor = 1e-5
n_mel = 128

# Create window (periodic Hann)
hann_arange = np.arange(frame_length, dtype=np.float32)
window = 0.5 * (1 - np.cos(2 * np.pi * hann_arange / frame_length))
print(f"Window length: {len(window)}, sum: {window.sum():.4f}")

# Unfold with frame_size_for_unfold = frame_length + 1 = 513
frame_size_for_unfold = frame_length + 1

# Simulate unfold: extract frames of size 513 with hop_length
n_frames = (len(audio) - frame_size_for_unfold) // hop_length + 1
frames_to_process = np.zeros((n_frames, frame_size_for_unfold))
for i in range(n_frames):
    start = i * hop_length
    frames_to_process[i] = audio[start:start + frame_size_for_unfold]

print(f"Number of frames: {n_frames}")

# Apply preemphasis (HTK flavor)
# first_in_frame = frames_to_process[..., :1] * (1.0 - preemphasis)
# rest_in_frame = frames_to_process[..., 1:-1] - preemphasis * frames_to_process[..., :-2]
# frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
first_in_frame = frames_to_process[:, :1] * (1.0 - preemphasis)  # [n_frames, 1]
rest_in_frame = frames_to_process[:, 1:-1] - preemphasis * frames_to_process[:, :-2]  # [n_frames, 511]
frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)  # [n_frames, 512]

print(f"Frames shape after preemphasis: {frames.shape}")

# Apply window
frames = frames * window  # [n_frames, 512]

# FFT with n=fft_length (1024, zero-padded)
stft = np.fft.rfft(frames, n=fft_length, axis=-1)  # [n_frames, 513]
print(f"STFT shape: {stft.shape}")

# Magnitude (NOT power!)
magnitude_spec = np.abs(stft)  # [n_frames, 513]
print(f"Magnitude spec stats: min={magnitude_spec.min():.6f}, max={magnitude_spec.max():.6f}")

# Apply mel filters
# PyTorch mel_filters shape is (513, 128)
# So: magnitude_spec [n_frames, 513] @ mel_filters [513, 128] = [n_frames, 128]
mel_spec = np.matmul(magnitude_spec, fe.mel_filters)
print(f"Mel spec shape: {mel_spec.shape}")

# Floor and log
log_mel = np.log(np.maximum(mel_spec, mel_floor))

print(f"\nManual log mel shape: {log_mel.shape}")
print(f"Manual log mel stats: min={log_mel.min():.4f}, max={log_mel.max():.4f}, mean={log_mel.mean():.4f}")

# Compare
print("\n=== Comparison ===")
min_frames = min(len(pytorch_mel), len(log_mel))
print("Frame 0, first 10 mel bins:")
print(f"  PyTorch: {pytorch_mel[0, :10]}")
print(f"  Manual:  {log_mel[0, :10]}")

print("\nFrame 100, first 10 mel bins:")
print(f"  PyTorch: {pytorch_mel[100, :10]}")
print(f"  Manual:  {log_mel[100, :10]}")

diff = np.abs(pytorch_mel[:min_frames] - log_mel[:min_frames])
print(f"\nMean absolute difference: {diff.mean():.6f}")
print(f"Max absolute difference: {diff.max():.6f}")

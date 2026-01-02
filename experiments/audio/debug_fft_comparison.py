#!/usr/bin/env python3
"""
Debug FFT comparison - print exact same values as llama.cpp debug output
"""
import numpy as np
import librosa

# Load audio
audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

# Parameters (same as llama.cpp gemma3n_log_mel_spectrogram)
frame_length = 512
fft_length = 1024
hop_length = 160
preemph = 0.97
n_fft_bins = fft_length // 2 + 1  # 513

# Frame extraction: extract frames of size 513
frame_size_for_unfold = frame_length + 1  # 513

# Periodic Hann window (same as llama.cpp)
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_length) / frame_length))
print(f"\nHann window (first 10): {hann[:10]}")
print(f"Hann window (last 10): {hann[-10:]}")

# Extract first frame
offset = 0
frame_raw = audio[offset:offset + frame_size_for_unfold]
print(f"\nRaw frame (first 10): {frame_raw[:10]}")

# Apply HTK-style preemphasis
# frame[0] = samples[offset] * (1 - preemph)
# frame[j] = samples[offset + j] - preemph * samples[offset + j - 1]  for j in [1, frame_length-1]
frame = np.zeros(frame_length, dtype=np.float32)
frame[0] = audio[offset] * (1.0 - preemph)
for j in range(1, frame_length):
    frame[j] = audio[offset + j] - preemph * audio[offset + j - 1]

print(f"\nFrame after preemphasis (first 10): {frame[:10]}")

# Apply window
frame_windowed = frame * hann
print(f"\nFrame after window (first 10): ", end="")
for j in range(10):
    print(f"{frame_windowed[j]:.6f} ", end="")
print()

# Zero-pad to fft_length
padded = np.zeros(fft_length, dtype=np.float32)
padded[:frame_length] = frame_windowed

# FFT
fft_result = np.fft.fft(padded)
print(f"\nFrame 0 FFT (first 5 bins):")
for j in range(5):
    re = fft_result[j].real
    im = fft_result[j].imag
    mag = np.abs(fft_result[j])
    print(f"  bin {j}: re={re:.6f}, im={im:.6f}, mag={mag:.6f}")

# Magnitude
magnitude = np.abs(fft_result[:n_fft_bins])
print(f"\nFrame 0 magnitude (first 10): ", end="")
for j in range(10):
    print(f"{magnitude[j]:.6f} ", end="")
print()

# Also compare with numpy.fft.rfft
rfft_result = np.fft.rfft(padded)
print(f"\nUsing np.fft.rfft:")
print(f"Frame 0 FFT (first 5 bins):")
for j in range(5):
    re = rfft_result[j].real
    im = rfft_result[j].imag
    mag = np.abs(rfft_result[j])
    print(f"  bin {j}: re={re:.6f}, im={im:.6f}, mag={mag:.6f}")

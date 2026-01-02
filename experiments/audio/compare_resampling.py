#!/usr/bin/env python3
"""
Compare different resampling methods
"""
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

# Load audio in different ways
audio_path = "assets/data/audio/mlk_speech.flac"

# 1. librosa load (default resampling)
audio_librosa, sr = librosa.load(audio_path, sr=16000)
print(f"1. librosa.load (sr=16000):")
print(f"   Samples: {len(audio_librosa)}")
print(f"   First 10: {audio_librosa[:10]}")

# 2. librosa with different res_type
# audio_librosa_kaiser = librosa.load(audio_path, sr=16000, res_type='kaiser_best')[0]
# print(f"\n2. librosa.load (res_type='kaiser_best'):")
# print(f"   First 10: {audio_librosa_kaiser[:10]}")

# 3. scipy resample
audio_raw, sr_raw = sf.read(audio_path)
audio_scipy = scipy.signal.resample_poly(audio_raw, 16000, sr_raw)
print(f"\n3. scipy.signal.resample_poly:")
print(f"   Samples: {len(audio_scipy)}")
print(f"   First 10: {audio_scipy[:10]}")

# 4. scipy resample with different method
audio_scipy_fft = scipy.signal.resample(audio_raw, int(len(audio_raw) * 16000 / sr_raw))
print(f"\n4. scipy.signal.resample (FFT-based):")
print(f"   Samples: {len(audio_scipy_fft)}")
print(f"   First 10: {audio_scipy_fft[:10]}")

# 5. Raw audio at 22050
print(f"\n5. Raw audio (22050 Hz):")
print(f"   Samples: {len(audio_raw)}")
print(f"   First 10: {audio_raw[:10]}")

# Compare what llama.cpp got:
llama_samples = [0.00000000, 0.00435165, 0.00480405, -0.00221930, 0.00861714,
                 0.02797187, 0.02184660, 0.00949992, 0.00445748, 0.00625751]
print(f"\n6. llama.cpp (miniaudio):")
print(f"   First 10: {llama_samples}")

# Check ratios
print(f"\nRatios (llama/librosa):")
for i in range(1, 10):
    if abs(audio_librosa[i]) > 0.001:
        print(f"  [{i}]: {llama_samples[i] / audio_librosa[i]:.4f}")

# Check if miniaudio is using a simpler resampling like linear interpolation
print(f"\n7. Linear interpolation (simple):")
ratio = sr_raw / 16000.0
linear_samples = []
for i in range(10):
    src_idx = i * ratio
    src_low = int(src_idx)
    src_high = min(src_low + 1, len(audio_raw) - 1)
    frac = src_idx - src_low
    val = audio_raw[src_low] * (1 - frac) + audio_raw[src_high] * frac
    linear_samples.append(val)
print(f"   First 10: {linear_samples}")

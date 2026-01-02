#!/usr/bin/env python3
"""
Check if llama.cpp audio is offset from librosa audio
"""
import numpy as np
import librosa
import soundfile as sf

# Load audio
audio_path = "assets/data/audio/mlk_speech.flac"
audio_librosa, sr = librosa.load(audio_path, sr=16000)
audio_raw, sr_raw = sf.read(audio_path)

# llama.cpp samples
llama_samples = np.array([0.00000000, 0.00435165, 0.00480405, -0.00221930, 0.00861714,
                          0.02797187, 0.02184660, 0.00949992, 0.00445748, 0.00625751,
                          0.01774129, 0.02118726, 0.00825382, -0.00117356, -0.00462411,
                          -0.00600049, -0.00278857, -0.00639056, -0.00746705, 0.00096494])

print(f"Checking for offset match...")
print(f"llama.cpp: {llama_samples[:10]}")
print(f"librosa:   {audio_librosa[:10]}")

# Try different offsets
best_offset = 0
best_corr = 0
for offset in range(-5, 6):
    if offset >= 0:
        llama_sub = llama_samples[:10]
        librosa_sub = audio_librosa[offset:offset+10]
    else:
        llama_sub = llama_samples[-offset:-offset+10]
        librosa_sub = audio_librosa[:10]

    if len(llama_sub) == len(librosa_sub):
        corr = np.corrcoef(llama_sub, librosa_sub)[0, 1]
        print(f"Offset {offset:+d}: correlation = {corr:.4f}")
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

print(f"\nBest offset: {best_offset} with correlation {best_corr:.4f}")

# Check if llama.cpp uses raw audio (22050) interpolation
print(f"\n=== Checking raw audio interpolation ===")
# ratio = 22050 / 16000 = 1.378125
ratio = sr_raw / 16000.0
print(f"Ratio: {ratio}")

# Try linear interpolation from raw audio
linear_from_raw = []
for i in range(len(llama_samples)):
    src_idx = i * ratio
    src_low = int(src_idx)
    src_high = min(src_low + 1, len(audio_raw) - 1)
    frac = src_idx - src_low
    val = audio_raw[src_low] * (1 - frac) + audio_raw[src_high] * frac
    linear_from_raw.append(val)

print(f"Linear from raw: {linear_from_raw[:10]}")
print(f"llama.cpp:       {list(llama_samples[:10])}")

# Compute mean absolute error
mae = np.mean(np.abs(np.array(linear_from_raw) - llama_samples))
print(f"MAE: {mae:.6f}")

# The first sample is 0 in llama, but should be audio_raw[0]
# This might indicate the audio starts with silence or padding
print(f"\nRaw audio first few: {audio_raw[:5]}")
print(f"llama.cpp first few: {llama_samples[:5]}")

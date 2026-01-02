#!/usr/bin/env python3
"""
Check if llama.cpp is loading the audio correctly
"""
import numpy as np
import librosa

# Load with librosa (this is what I've been using)
audio_librosa, sr = librosa.load("assets/data/audio/mlk_speech.flac", sr=16000)
print(f"Librosa audio: {len(audio_librosa)} samples")
print(f"  First 10: {audio_librosa[:10]}")

# Load with soundfile directly
import soundfile as sf
audio_sf, sr_sf = sf.read("assets/data/audio/mlk_speech.flac")
print(f"\nSoundfile audio: {len(audio_sf)} samples, sr={sr_sf}")
print(f"  First 10: {audio_sf[:10]}")

# Check if there's resampling
if sr_sf != 16000:
    print(f"  Resampling from {sr_sf} to 16000...")
    audio_sf_resampled = librosa.resample(audio_sf, orig_sr=sr_sf, target_sr=16000)
    print(f"  Resampled: {len(audio_sf_resampled)} samples")
    print(f"  First 10: {audio_sf_resampled[:10]}")

# What about miniaudio (used by llama.cpp)?
# llama.cpp uses miniaudio to load audio files
# Let's also check the raw samples from the file
print(f"\nExpected llama.cpp windowed frame 0 (first 10):")
print(f"  From llama.cpp debug: 0.000000 0.000000 0.000000 -0.000002 0.000006 0.000018 -0.000007 -0.000022 -0.000011 0.000006")

# The values are very different - the llama.cpp signal looks much smaller
# Let's check if the audio is being loaded differently

# From llama.cpp, the raw signal seems to have much smaller values
# First value in windowed signal ≈ 0, but should be around 0.00017655 after preemph

print(f"\n=== Analysis ===")
print(f"PyTorch windowed[0:5]: 0.000000 -0.000000 -0.000001 0.000006 0.000009")
print(f"llama.cpp windowed[0:5]: 0.000000 0.000000 0.000000 -0.000002 0.000006")
print(f"These are DIFFERENT!")
print(f"")
print(f"The pattern suggests llama.cpp audio samples are DIFFERENT from librosa")
print(f"")
print(f"Let me compute windowed values from llama.cpp raw samples:")

# From llama.cpp we know:
# windowed[4] ≈ 0.000006
# This should be frame[4] * hann[4] where hann[4] ≈ 0.000602

hann_4 = 0.5 * (1 - np.cos(2 * np.pi * 4 / 512))
print(f"hann[4] = {hann_4}")

# If windowed[4] = 0.000006, then frame[4] = 0.000006 / hann[4]
frame_4_llama = 0.000006 / hann_4
print(f"If windowed[4] = 0.000006, then frame[4] = {frame_4_llama}")

# From Python, frame[4] should be:
# frame[4] = audio[4] - 0.97 * audio[3]
# = 0.02899666 - 0.97 * 0.01366709 = 0.01573958
print(f"From librosa: frame[4] = 0.02899666 - 0.97 * 0.01366709 = {0.02899666 - 0.97 * 0.01366709}")

print(f"\nSo llama.cpp frame[4] ≈ {frame_4_llama}")
print(f"But Python frame[4] ≈ 0.01573958")
print(f"")
print(f"Ratio: {0.01573958 / frame_4_llama}")
print(f"")
print(f"The audio in llama.cpp is about 1580x smaller!")
print(f"This suggests the audio is in int16 format and not normalized!")

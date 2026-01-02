#!/usr/bin/env python3
"""
Debug preemphasis calculation - trace exact PyTorch behavior
"""
import numpy as np
import librosa

# Load audio
audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio: {len(audio)} samples")
print(f"Audio first 20: {audio[:20]}")

# Parameters
frame_length = 512
fft_length = 1024
hop_length = 160
preemph = 0.97
frame_size_for_unfold = frame_length + 1  # 513

# Frame 0
offset = 0
raw_frame = audio[offset:offset + frame_size_for_unfold]
print(f"\nRaw frame 0 (513 samples):")
print(f"  [0] = {raw_frame[0]}")
print(f"  [1] = {raw_frame[1]}")
print(f"  [2] = {raw_frame[2]}")
print(f"  [-3] = {raw_frame[-3]}")
print(f"  [-2] = {raw_frame[-2]}")
print(f"  [-1] = {raw_frame[-1]}")

# PyTorch preemphasis (HTK flavor):
# first_in_frame = frames_to_process[..., :1] * (1.0 - preemphasis)
# rest_in_frame = frames_to_process[..., 1:-1] - preemphasis * frames_to_process[..., :-2]
# frames = torch.cat((first_in_frame, rest_in_frame), dim=-1)

first_in_frame = raw_frame[:1] * (1.0 - preemph)  # [0] * 0.03
rest_in_frame = raw_frame[1:-1] - preemph * raw_frame[:-2]  # [1:512] - 0.97 * [0:511]

print(f"\nPreemphasis calculation:")
print(f"  first_in_frame[0] = raw[0] * 0.03 = {raw_frame[0]} * 0.03 = {first_in_frame[0]}")
print(f"  rest_in_frame[0] = raw[1] - 0.97 * raw[0] = {raw_frame[1]} - 0.97 * {raw_frame[0]} = {rest_in_frame[0]}")
print(f"  rest_in_frame[1] = raw[2] - 0.97 * raw[1] = {raw_frame[2]} - 0.97 * {raw_frame[1]} = {rest_in_frame[1]}")
print(f"  rest_in_frame[-1] = raw[511] - 0.97 * raw[510] = {raw_frame[511]} - 0.97 * {raw_frame[510]} = {rest_in_frame[-1]}")

frame = np.concatenate([first_in_frame, rest_in_frame])
print(f"\nFrame shape: {frame.shape}")
print(f"Frame first 10: {frame[:10]}")

# Now show what llama.cpp is doing (CURRENT WRONG implementation):
print(f"\n=== llama.cpp current implementation ===")
llama_frame = np.zeros(frame_length)
llama_frame[0] = audio[offset] * (1.0 - preemph)  # This is correct
for j in range(1, frame_length):
    llama_frame[j] = audio[offset + j] - preemph * audio[offset + j - 1]

print(f"llama_frame first 10: {llama_frame[:10]}")
print(f"Difference: {np.abs(frame[:10] - llama_frame[:10])}")

# The issue: PyTorch uses indices [0] and [1:-1] from 513-sample frame
# Which means: rest_in_frame[i] = raw[i+1] - 0.97 * raw[i-1+1] = raw[i+1] - 0.97 * raw[i]
# Wait, let me re-check...

print(f"\n=== Re-analyzing PyTorch indexing ===")
print(f"raw_frame.shape = {raw_frame.shape}")
print(f"raw_frame[1:-1].shape = {raw_frame[1:-1].shape}")  # Should be 511
print(f"raw_frame[:-2].shape = {raw_frame[:-2].shape}")    # Should be 511

# raw_frame[1:-1] = [1, 2, ..., 511]  (indices into 513-sample array)
# raw_frame[:-2]  = [0, 1, ..., 510]  (indices into 513-sample array)
# So rest_in_frame[i] = raw[i+1] - 0.97 * raw[i]

print(f"\nSo PyTorch is doing:")
print(f"  frame[0] = raw[0] * 0.03")
print(f"  frame[1] = raw[1] - 0.97 * raw[0]  (via rest_in_frame[0])")
print(f"  frame[2] = raw[2] - 0.97 * raw[1]  (via rest_in_frame[1])")
print(f"  ...")
print(f"  frame[511] = raw[511] - 0.97 * raw[510]  (via rest_in_frame[510])")

print(f"\nAnd llama.cpp is doing:")
print(f"  frame[0] = audio[offset] * 0.03 = audio[0] * 0.03")
print(f"  frame[1] = audio[offset+1] - 0.97 * audio[offset+0] = audio[1] - 0.97 * audio[0]")
print(f"  frame[2] = audio[offset+2] - 0.97 * audio[offset+1] = audio[2] - 0.97 * audio[1]")

print(f"\nThey should be THE SAME for frame 0!")
print(f"Let me check the actual values...")
print(f"PyTorch frame[0:5]: {frame[:5]}")
print(f"llama   frame[0:5]: {llama_frame[:5]}")

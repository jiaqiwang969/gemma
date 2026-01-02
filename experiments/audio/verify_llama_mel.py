#!/usr/bin/env python3
"""
Use exact same audio as llama.cpp to compute mel
"""
import numpy as np

# These are the EXACT samples from llama.cpp (miniaudio resampling)
llama_audio = np.array([
    0.00000000, 0.00435165, 0.00480405, -0.00221930, 0.00861714, 0.02797187,
    0.02184660, 0.00949992, 0.00445748, 0.00625751, 0.01774129, 0.02118726,
    0.00825382, -0.00117356, -0.00462411, -0.00600049, -0.00278857, -0.00639056,
    -0.00746705, 0.00096494
], dtype=np.float32)

# Parameters
frame_length = 512
hop_length = 160
preemph = 0.97

# Periodic Hann window
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_length) / frame_length)).astype(np.float32)

print("Hann window (first 10):")
for j in range(10):
    print(f"  [{j}]: {hann[j]:.10f}")

# Apply preemphasis and window (just like llama.cpp)
frame = np.zeros(frame_length, dtype=np.float32)
frame[0] = llama_audio[0] * (1.0 - preemph)
for j in range(1, min(frame_length, len(llama_audio))):
    frame[j] = llama_audio[j] - preemph * llama_audio[j-1]

print(f"\nFrame after preemphasis (first 10):")
for j in range(10):
    print(f"  [{j}]: {frame[j]:.10f}")

# Apply window
frame_windowed = frame * hann

print(f"\nFrame after window (first 10):")
for j in range(10):
    print(f"  [{j}]: {frame_windowed[j]:.10f}")

# Compare with llama.cpp output:
# Frame 0 after window (first 10): 0.000000 0.000000 0.000000 -0.000002 0.000006 0.000018 -0.000007 -0.000022 -0.000011 0.000006

print(f"\nllama.cpp Frame 0 after window (first 10):")
print("  0.000000 0.000000 0.000000 -0.000002 0.000006 0.000018 -0.000007 -0.000022 -0.000011 0.000006")

# Compute FFT
fft_length = 1024
padded = np.zeros(fft_length, dtype=np.float32)
padded[:frame_length] = frame_windowed
fft_result = np.fft.fft(padded)

print(f"\nFFT (first 5 bins):")
for j in range(5):
    re = fft_result[j].real
    im = fft_result[j].imag
    mag = np.abs(fft_result[j])
    print(f"  bin {j}: re={re:.6f}, im={im:.6f}, mag={mag:.6f}")

# Compare with llama.cpp output:
# Frame 0 FFT (first 5 bins):
#   bin 0: re=-0.002128, im=0.000000, mag=0.002128
#   bin 1: re=-0.003642, im=0.002844, mag=0.004621
#   bin 2: re=0.003827, im=0.006741, mag=0.007752
#   bin 3: re=0.008277, im=-0.003287, mag=0.008906
#   bin 4: re=-0.001692, im=-0.007271, mag=0.007465
print(f"\nllama.cpp FFT (first 5 bins):")
print("  bin 0: re=-0.002128, im=0.000000, mag=0.002128")
print("  bin 1: re=-0.003642, im=0.002844, mag=0.004621")
print("  bin 2: re=0.003827, im=0.006741, mag=0.007752")
print("  bin 3: re=0.008277, im=-0.003287, mag=0.008906")
print("  bin 4: re=-0.001692, im=-0.007271, mag=0.007465")

#!/usr/bin/env python3
"""
Compute mel spectrogram exactly like llama.cpp does
"""
import numpy as np
import soundfile as sf

# Load pre-resampled audio
audio, sr = sf.read('/tmp/mlk_speech_16k.wav')
print(f"Audio: {len(audio)} samples at {sr} Hz")
print(f"First 10 samples: {audio[:10]}")

# Parameters (matching llama.cpp)
frame_length = 512
fft_length = 1024
hop_length = 160
preemph = 0.97
mel_floor = 1e-5
n_mel = 128
n_fft_bins = fft_length // 2 + 1

# Frame extraction
frame_size_for_unfold = frame_length + 1
n_frames = (len(audio) - frame_size_for_unfold) // hop_length + 1
print(f"Number of frames: {n_frames}")

# Periodic Hann window
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_length) / frame_length))

# HTK mel filterbank
def hz_to_mel_htk(f_hz):
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)

def mel_to_hz_htk(m):
    return 700.0 * (np.power(10.0, m / 2595.0) - 1.0)

fmin, fmax = 125.0, 7600.0
m_lo = hz_to_mel_htk(fmin)
m_hi = hz_to_mel_htk(fmax)
mel_pts = np.linspace(m_lo, m_hi, n_mel + 2)
hz_pts = mel_to_hz_htk(mel_pts)
bin_hz_step = sr / fft_length

filterbank = np.zeros((n_mel, n_fft_bins))
for m in range(n_mel):
    f_left = hz_pts[m]
    f_center = hz_pts[m + 1]
    f_right = hz_pts[m + 2]
    for k in range(n_fft_bins):
        f = k * bin_hz_step
        if f >= f_left and f <= f_center:
            filterbank[m, k] = (f - f_left) / max(1e-30, f_center - f_left)
        elif f > f_center and f <= f_right:
            filterbank[m, k] = (f_right - f) / max(1e-30, f_right - f_center)

# Compute mel spectrogram
log_mel = np.zeros((n_frames, n_mel))
for i in range(n_frames):
    offset = i * hop_length

    # Apply preemphasis
    frame = np.zeros(frame_length)
    frame[0] = audio[offset] * (1.0 - preemph)
    for j in range(1, frame_length):
        frame[j] = audio[offset + j] - preemph * audio[offset + j - 1]

    # Apply window
    frame *= hann

    # Debug: first frame
    if i == 0:
        print(f"\nFrame 0 after window (first 10): {frame[:10]}")

    # Zero-pad to fft_length and FFT
    padded = np.zeros(fft_length)
    padded[:frame_length] = frame
    fft_result = np.fft.fft(padded)

    if i == 0:
        print(f"Frame 0 FFT (first 5 bins):")
        for j in range(5):
            print(f"  bin {j}: re={fft_result[j].real:.6f}, im={fft_result[j].imag:.6f}, mag={np.abs(fft_result[j]):.6f}")

    # Magnitude (NOT power)
    magnitude = np.abs(fft_result[:n_fft_bins])

    if i == 0:
        print(f"Frame 0 magnitude (first 10): {magnitude[:10]}")

    # Apply mel filters and log
    mel_spec = np.dot(magnitude, filterbank.T)
    log_mel[i] = np.log(np.maximum(mel_spec, mel_floor))

print(f"\nMel spectrogram stats:")
print(f"  shape: ({n_frames}, {n_mel})")
print(f"  min: {log_mel.min():.4f}")
print(f"  max: {log_mel.max():.4f}")
print(f"  mean: {log_mel.mean():.4f}")

print(f"\nFirst 10 frames, first 10 mel bins:")
for i in range(10):
    print(f"  frame {i}: [{', '.join([f'{log_mel[i, j]:.4f}' for j in range(10)])}]")

# Compare with llama.cpp output:
# mean=-1.1816
print(f"\n=== Comparison with llama.cpp ===")
print(f"Python mean: {log_mel.mean():.4f}")
print(f"llama.cpp mean: -1.1816")
print(f"Difference: {abs(log_mel.mean() - (-1.1816)):.6f}")

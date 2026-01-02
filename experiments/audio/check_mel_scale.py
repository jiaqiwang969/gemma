#!/usr/bin/env python3
"""
Check Gemma 3n mel scale - HTK vs Slaney
"""
import numpy as np
from transformers import AutoProcessor

model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
fe = processor.feature_extractor

print("=== Mel filter analysis ===")
print(f"mel_filters shape: {fe.mel_filters.shape}")  # (n_fft_bins, n_mel)

# Check which bins have non-zero values for each mel
mel_filters = fe.mel_filters  # (513, 128)
for mel_idx in [0, 1, 2, 10, 50, 100, 127]:
    # Find non-zero bins for this mel channel
    non_zero = np.where(mel_filters[:, mel_idx] > 0)[0]
    if len(non_zero) > 0:
        center_bin = non_zero[len(non_zero)//2]
        # Convert bin to Hz: bin * sample_rate / n_fft
        center_hz = center_bin * 16000 / 1024
        print(f"Mel {mel_idx}: bins {non_zero[0]}-{non_zero[-1]}, center bin={center_bin}, center_hz={center_hz:.1f} Hz")
        print(f"  Filter values (first 5): {mel_filters[non_zero[:5], mel_idx]}")
    else:
        print(f"Mel {mel_idx}: no non-zero values")

# Check if it looks like HTK or Slaney scale
# HTK: mel = 2595 * log10(1 + f/700)
# Slaney: linear below 1000 Hz, log above

# Calculate expected center frequencies for HTK scale
print("\n=== HTK mel scale center frequencies ===")
n_mel = 128
fmin, fmax = 125.0, 7600.0

# HTK scale
def hz_to_mel_htk(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz_htk(m):
    return 700 * (10 ** (m / 2595) - 1)

mel_min = hz_to_mel_htk(fmin)
mel_max = hz_to_mel_htk(fmax)
mel_centers = np.linspace(mel_min, mel_max, n_mel + 2)[1:-1]  # n_mel center frequencies
hz_centers_htk = mel_to_hz_htk(mel_centers)
print(f"HTK: First 5 center Hz: {hz_centers_htk[:5]}")
print(f"HTK: Last 5 center Hz: {hz_centers_htk[-5:]}")

# Slaney scale
def hz_to_mel_slaney(f):
    min_log_hz = 1000.0
    lin_slope = 3 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = np.log(6.4) / 27.0
    if f < min_log_hz:
        return f * lin_slope
    else:
        return min_log_mel + np.log(f / min_log_hz) / log_step

def mel_to_hz_slaney(m):
    min_log_hz = 1000.0
    lin_slope = 3 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = np.log(6.4) / 27.0
    if m < min_log_mel:
        return m / lin_slope
    else:
        return min_log_hz * np.exp((m - min_log_mel) * log_step)

mel_min_s = hz_to_mel_slaney(fmin)
mel_max_s = hz_to_mel_slaney(fmax)
mel_centers_s = np.linspace(mel_min_s, mel_max_s, n_mel + 2)[1:-1]
hz_centers_slaney = np.array([mel_to_hz_slaney(m) for m in mel_centers_s])
print(f"\nSlaney: First 5 center Hz: {hz_centers_slaney[:5]}")
print(f"Slaney: Last 5 center Hz: {hz_centers_slaney[-5:]}")

# Check which matches PyTorch
print("\n=== Comparing to PyTorch mel centers ===")
for mel_idx in [0, 5, 10, 50, 100]:
    non_zero = np.where(mel_filters[:, mel_idx] > 0)[0]
    if len(non_zero) > 0:
        # Find center (peak of triangle)
        peak_idx = np.argmax(mel_filters[non_zero, mel_idx])
        center_bin = non_zero[peak_idx]
        center_hz_actual = center_bin * 16000 / 1024
        print(f"Mel {mel_idx}: actual center={center_hz_actual:.1f} Hz, HTK={hz_centers_htk[mel_idx]:.1f} Hz, Slaney={hz_centers_slaney[mel_idx]:.1f} Hz")

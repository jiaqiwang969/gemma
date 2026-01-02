#!/usr/bin/env python3
"""
Compare mel filter values between PyTorch and llama.cpp calculation
"""
import numpy as np
from transformers import AutoProcessor

model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
fe = processor.feature_extractor

print("=== PyTorch mel filters ===")
print(f"Shape: {fe.mel_filters.shape}")  # (n_fft_bins, n_mel)
print(f"Sum of each mel filter:")

# PyTorch stores as (n_fft_bins, n_mel) = (513, 128)
# Let's compute the sum of each mel filter
for mel_idx in [0, 1, 2, 10, 50, 100, 127]:
    filter_sum = fe.mel_filters[:, mel_idx].sum()
    non_zero = np.where(fe.mel_filters[:, mel_idx] > 0)[0]
    if len(non_zero) > 0:
        max_val = fe.mel_filters[:, mel_idx].max()
        print(f"  Mel {mel_idx}: sum={filter_sum:.4f}, max={max_val:.4f}, non_zero bins: {len(non_zero)}")

# Now let's manually compute what llama.cpp should produce
print("\n=== Manual HTK mel filter calculation ===")
n_mel = 128
n_fft = 1024
sample_rate = 16000
fmin = 125.0
fmax = 7600.0
n_fft_bins = n_fft // 2 + 1

def hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz_htk(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

mel_min = hz_to_mel_htk(fmin)
mel_max = hz_to_mel_htk(fmax)
mel_pts = np.linspace(mel_min, mel_max, n_mel + 2)
hz_pts = mel_to_hz_htk(mel_pts)

bin_hz_step = sample_rate / n_fft

# Compute filters without slaney normalization
filters_manual = np.zeros((n_mel, n_fft_bins))
for m in range(n_mel):
    f_left = hz_pts[m]
    f_center = hz_pts[m + 1]
    f_right = hz_pts[m + 2]

    for k in range(n_fft_bins):
        f = k * bin_hz_step
        if f >= f_left and f <= f_center:
            filters_manual[m, k] = (f - f_left) / max(1e-30, f_center - f_left)
        elif f > f_center and f <= f_right:
            filters_manual[m, k] = (f_right - f) / max(1e-30, f_right - f_center)

# Compare
print("Comparing filter sums:")
for mel_idx in [0, 1, 2, 10, 50, 100, 127]:
    manual_sum = filters_manual[mel_idx].sum()
    pytorch_sum = fe.mel_filters[:, mel_idx].sum()
    manual_max = filters_manual[mel_idx].max() if filters_manual[mel_idx].max() > 0 else 0
    pytorch_max = fe.mel_filters[:, mel_idx].max()
    print(f"  Mel {mel_idx}: manual sum={manual_sum:.4f}, pytorch sum={pytorch_sum:.4f}")
    print(f"           manual max={manual_max:.4f}, pytorch max={pytorch_max:.4f}")

    # Find non-zero bins
    manual_nz = np.where(filters_manual[mel_idx] > 0)[0]
    pytorch_nz = np.where(fe.mel_filters[:, mel_idx] > 0)[0]
    print(f"           manual bins: {manual_nz.tolist()[:5]}, pytorch bins: {pytorch_nz.tolist()[:5]}")

# Check the actual filter values for mel 0
print("\n=== Detailed filter values for mel 0 ===")
mel_idx = 0
pytorch_filter = fe.mel_filters[:, mel_idx]
manual_filter = filters_manual[mel_idx]
pytorch_nz = np.where(pytorch_filter > 0)[0]
manual_nz = np.where(manual_filter > 0)[0]
print(f"PyTorch non-zero bins: {pytorch_nz}")
print(f"PyTorch values: {pytorch_filter[pytorch_nz]}")
print(f"Manual non-zero bins: {manual_nz}")
print(f"Manual values: {manual_filter[manual_nz]}")

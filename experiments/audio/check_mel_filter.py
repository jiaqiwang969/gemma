#!/usr/bin/env python3
"""
Check HTK mel filter bank implementation
"""
import numpy as np

def hz_to_mel_htk(f_hz):
    """HTK mel scale: mel = 2595 * log10(1 + f/700)"""
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)

def mel_to_hz_htk(m):
    """Inverse of HTK mel scale"""
    return 700.0 * (np.power(10.0, m / 2595.0) - 1.0)

def create_mel_filterbank_htk(n_mel, n_fft, sample_rate, fmin=0.0, fmax=None, normalize=False):
    """Create HTK-style mel filterbank (no Slaney normalization)"""
    if fmax is None:
        fmax = sample_rate / 2.0

    n_fft_bins = n_fft // 2 + 1
    bin_hz_step = sample_rate / n_fft

    # Mel grid: n_mel + 2 edges
    m_lo = hz_to_mel_htk(fmin)
    m_hi = hz_to_mel_htk(fmax)
    mel_pts = np.linspace(m_lo, m_hi, n_mel + 2)
    hz_pts = mel_to_hz_htk(mel_pts)

    # Create filterbank
    filterbank = np.zeros((n_mel, n_fft_bins))
    for m in range(n_mel):
        f_left = hz_pts[m]
        f_center = hz_pts[m + 1]
        f_right = hz_pts[m + 2]

        for k in range(n_fft_bins):
            f = k * bin_hz_step
            if f >= f_left and f <= f_center:
                filterbank[m, k] = (f - f_left) / max(1e-10, f_center - f_left)
            elif f > f_center and f <= f_right:
                filterbank[m, k] = (f_right - f) / max(1e-10, f_right - f_center)

        # Slaney area normalization: multiply by 2 / (f_right - f_left)
        if normalize:
            enorm = 2.0 / max(1e-10, f_right - f_left)
            filterbank[m, :] *= enorm

    return filterbank

# Gemma 3n parameters
n_mel = 128
n_fft = 1024
sample_rate = 16000
fmin = 125.0
fmax = 7600.0

# Create filterbank WITHOUT slaney normalization (like Gemma 3n)
filterbank = create_mel_filterbank_htk(n_mel, n_fft, sample_rate, fmin, fmax, normalize=False)
print(f"Filterbank shape: {filterbank.shape}")
print(f"Filterbank nonzero entries: {np.count_nonzero(filterbank)}")

# Print first mel filter
print(f"\nMel filter 0 (first few bins with non-zero values):")
nonzero_idx = np.where(filterbank[0] > 0)[0]
for idx in nonzero_idx[:10]:
    print(f"  bin {idx} ({idx * sample_rate / n_fft:.1f} Hz): {filterbank[0, idx]:.6f}")

# Print last mel filter
print(f"\nMel filter 127 (last few bins with non-zero values):")
nonzero_idx = np.where(filterbank[127] > 0)[0]
for idx in nonzero_idx[-10:]:
    print(f"  bin {idx} ({idx * sample_rate / n_fft:.1f} Hz): {filterbank[127, idx]:.6f}")

# Compare with llama.cpp output
# From debug output: filters[60939] = 296.252563
# This is index into flattened array of size 128 * 513 = 65664
# mel_idx = 60939 // 513 = 118
# bin_idx = 60939 % 513 = 405
print(f"\n=== Comparison with llama.cpp ===")
print(f"llama.cpp filterbank size: 65664 (128 * 513)")

# llama.cpp prints values * 1000
# filters[60939] = 296.252563 means actual value = 0.296252563
# Index 60939: mel=118, bin=405
mel_idx = 60939 // 513
bin_idx = 60939 % 513
print(f"llama.cpp filters[60939] = 296.252563 (mel={mel_idx}, bin={bin_idx})")
print(f"Python filterbank[{mel_idx}, {bin_idx}] = {filterbank[mel_idx, bin_idx]:.6f}")
print(f"Expected (llama * 1000): {filterbank[mel_idx, bin_idx] * 1000:.6f}")

# Check a few more
test_indices = [60940, 60941, 61447, 61448]
for idx in test_indices:
    mel_idx = idx // 513
    bin_idx = idx % 513
    print(f"  filters[{idx}]: llama.cpp prints value*1000, mel={mel_idx}, bin={bin_idx}")
    print(f"    Python: {filterbank[mel_idx, bin_idx] * 1000:.6f}")

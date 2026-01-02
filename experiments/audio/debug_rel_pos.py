#!/usr/bin/env python3
"""
Debug relative position encoding in Gemma 3n Conformer
"""
import numpy as np

# Parameters from Gemma 3n
d_model = 1536
max_backward = 12
max_forward = 0
max_span = max_backward + 1  # 13
half_dim = d_model // 2  # 768

print(f"d_model = {d_model}")
print(f"max_span = {max_span}")
print(f"half_dim = {half_dim}")

# Generate inv_timescales (same as PyTorch)
# PyTorch: 1.0 / (10000 ** (torch.arange(0, half_dim) / (half_dim - 1)))
inv_timescales = 1.0 / (10000.0 ** (np.arange(half_dim) / (half_dim - 1)))
print(f"\ninv_timescales first 5: {inv_timescales[:5]}")
print(f"inv_timescales last 5: {inv_timescales[-5:]}")

# llama.cpp style:
# inv_timescales[i] = exp(-i * log(10000) / (half_dim - 1))
inv_timescales_llama = np.exp(-np.arange(half_dim) * np.log(10000.0) / (half_dim - 1))
print(f"\nllama.cpp inv_timescales first 5: {inv_timescales_llama[:5]}")
print(f"llama.cpp inv_timescales last 5: {inv_timescales_llama[-5:]}")

# Verify they're the same
print(f"\nDifference: {np.abs(inv_timescales - inv_timescales_llama).max()}")

# Generate position embeddings
# PyTorch stores positions as [max_backward, max_backward-1, ..., 0, -1, ..., -max_forward]
# For Gemma 3n with max_forward=0: [12, 11, ..., 0]
# Shape: [max_span, d_model] = [13, 1536]

# PyTorch version
pos_emb_pytorch = np.zeros((max_span, d_model))
positions = np.arange(max_backward, -(max_forward + 1), -1)  # [12, 11, ..., 0]
print(f"\nPositions: {positions}")

for pos_idx, pos in enumerate(positions):
    for i in range(half_dim):
        ang = pos * inv_timescales[i]
        pos_emb_pytorch[pos_idx, 2*i + 0] = np.sin(ang)  # sin at even indices
        pos_emb_pytorch[pos_idx, 2*i + 1] = np.cos(ang)  # cos at odd indices

print(f"\nPyTorch pos_emb shape: {pos_emb_pytorch.shape}")
print(f"PyTorch pos_emb[0, :10]: {pos_emb_pytorch[0, :10]}")  # position 12
print(f"PyTorch pos_emb[12, :10]: {pos_emb_pytorch[12, :10]}")  # position 0

# llama.cpp version (position idx goes from 0 to 12, rel_pos = 12-idx = 12, 11, ..., 0)
# This matches the above!

# Check what pos_emb tensor should look like
# In llama.cpp graph:
# pos_emb tensor: [G3NA_N_CHANNELS, G3NA_MAX_SPAN] = [1536, 13]
# This is transposed from PyTorch's [13, 1536]

pos_emb_llama = pos_emb_pytorch.T  # [1536, 13]
print(f"\nllama.cpp pos_emb shape (transposed): {pos_emb_llama.shape}")

# Now let's check term_bd calculation
# term_bd = Q @ sin_emb^T
# Q: [d_head, n_tokens, n_head] = [192, 325, 8]
# sin_emb (after projection): [d_head, max_span, n_head] = [192, 13, 8]
# term_bd_raw: [max_span, n_tokens, n_head] = [13, 325, 8]

# After relative shift, term_bd should become [n_tokens, n_tokens, n_head]
# The shift converts position indices to attention matrix indices

print("\n=== Relative shift explanation ===")
print("term_bd_raw[pos_idx, q, h] = Q[q, h] @ sin_emb[pos_idx, h]")
print("pos_idx 0 corresponds to relative position 12 (k is 12 positions before q)")
print("pos_idx 12 corresponds to relative position 0 (k == q)")
print("")
print("For attention[q, k], we want:")
print("  attention_bias[q, k] = term_bd[pos_idx] where pos_idx = 12 - (q - k)")
print("  This is only valid when 0 <= q - k <= 12 (i.e., k <= q and within max_backward)")
print("")
print("The relative shift operation creates a Toeplitz structure:")
print("  result[k, q] = term_bd_raw[12 - (q - k), q] for valid positions")

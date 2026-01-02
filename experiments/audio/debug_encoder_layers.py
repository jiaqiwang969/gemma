#!/usr/bin/env python3
"""
Debug encoder layers - dump intermediate values from PyTorch for comparison with llama.cpp
"""
import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load audio
audio, sr = sf.read('/tmp/mlk_speech_16k.wav')
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)} samples at {sr} Hz")

# Load model
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

# Prepare inputs
messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio, "sample_rate": sr},
        {"type": "text", "text": "Transcribe this audio."}
    ]}
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# Get mel spectrogram
mel = inputs['input_features']
mel_mask = inputs['input_features_mask']
print(f"\nMel shape: {mel.shape}")

# Get audio tower
audio_tower = model.model.audio_tower
subsample = audio_tower.subsample_conv_projection

# Trace through subsample_conv_projection manually
with torch.no_grad():
    # Input is [batch, time, mel] = [1, 1297, 128]
    x = mel
    print(f"\n=== Subsample Conv Projection ===")
    print(f"Input: {x.shape}")

    # Reshape for conv: [batch, 1, mel, time]
    x = x.unsqueeze(1).permute(0, 1, 3, 2)
    print(f"After reshape for conv: {x.shape}")

    # Conv0 with manual padding
    import torch.nn.functional as F
    x_padded = F.pad(x, (1, 1, 0, 2))  # (left, right, top, bottom)
    print(f"After padding (1,1,0,2): {x_padded.shape}")

    x = subsample.conv_0.conv(x_padded)
    print(f"After conv0: {x.shape}")

    # Save conv0 output
    np.save('/tmp/pytorch_conv0_out.npy', x.numpy())
    print(f"First 5 values of conv0 output: {x.flatten()[:5].numpy()}")

    # Apply norm and relu
    # The norm is special - Gemma3nAudioCumulativeGroupNorm
    x = subsample.conv_0.norm(x)
    x = subsample.conv_0.activation(x)
    print(f"After norm+relu: {x.shape}")

    # Conv1 with manual padding
    x_padded = F.pad(x, (1, 1, 0, 2))
    print(f"After padding (1,1,0,2): {x_padded.shape}")

    x = subsample.conv_1.conv(x_padded)
    print(f"After conv1: {x.shape}")

    # Save conv1 output
    np.save('/tmp/pytorch_conv1_out.npy', x.numpy())
    print(f"First 5 values of conv1 output: {x.flatten()[:5].numpy()}")

    # Apply norm and relu
    x = subsample.conv_1.norm(x)
    x = subsample.conv_1.activation(x)
    print(f"After norm+relu: {x.shape}")

    # Flatten: [batch, channels, mel, time] -> [batch, time, channels*mel]
    batch, channels, mel_dim, time_dim = x.shape
    x = x.permute(0, 3, 1, 2).reshape(batch, time_dim, channels * mel_dim)
    print(f"After flatten: {x.shape}")

    # Save flattened output
    np.save('/tmp/pytorch_flattened.npy', x.numpy())
    print(f"First 5 values of flattened: {x.flatten()[:5].numpy()}")

    # Input projection
    x = subsample.input_proj_linear(x)
    print(f"After input projection: {x.shape}")

    # Save input projection output
    np.save('/tmp/pytorch_input_proj.npy', x.numpy())
    print(f"First 5 values of input_proj: {x.flatten()[:5].numpy()}")

    print(f"\n=== Conformer Blocks ===")
    # Run through conformer blocks
    for i, block in enumerate(audio_tower.conformer):
        x = block(x, mel_mask[:, :x.shape[1]])
        if i == 0:
            np.save('/tmp/pytorch_conformer_block0.npy', x.numpy())
            print(f"Block {i} output first 5: {x.flatten()[:5].numpy()}")
        if i == 11:  # Last block
            np.save('/tmp/pytorch_conformer_block11.npy', x.numpy())
            print(f"Block {i} (last) output first 5: {x.flatten()[:5].numpy()}")

    print(f"\nFinal conformer output: {x.shape}")

    # Temporal reduction
    reduction_factor = audio_tower.config.conf_reduction_factor
    print(f"Reduction factor: {reduction_factor}")
    x_reduced = x[:, ::reduction_factor]
    print(f"After temporal reduction: {x_reduced.shape}")

    np.save('/tmp/pytorch_temporal_reduced.npy', x_reduced.numpy())
    print(f"First 5 values after reduction: {x_reduced.flatten()[:5].numpy()}")

    print(f"\n=== Embed Audio ===")
    # Now apply embed_audio
    embed_audio = model.model.embed_audio

    # soft_embedding_norm
    x_norm = embed_audio.soft_embedding_norm(x_reduced)
    print(f"After soft_embedding_norm: {x_norm.shape}")
    np.save('/tmp/pytorch_soft_norm.npy', x_norm.numpy())
    print(f"First 5 values: {x_norm.flatten()[:5].numpy()}")

    # embedding_projection
    x_proj = embed_audio.embedding_projection(x_norm)
    print(f"After embedding_projection: {x_proj.shape}")
    np.save('/tmp/pytorch_emb_proj.npy', x_proj.numpy())
    print(f"First 5 values: {x_proj.flatten()[:5].numpy()}")

    # embedding_post_projection_norm
    x_final = embed_audio.embedding_post_projection_norm(x_proj)
    print(f"After embedding_post_projection_norm: {x_final.shape}")
    np.save('/tmp/pytorch_final.npy', x_final.numpy())
    print(f"First 5 values: {x_final.flatten()[:5].numpy()}")

    print(f"\n=== Stats ===")
    print(f"Final shape: {x_final.shape}")
    print(f"Final stats: min={x_final.min():.4f}, max={x_final.max():.4f}, mean={x_final.mean():.4f}")

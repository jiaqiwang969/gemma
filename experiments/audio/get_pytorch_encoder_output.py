#!/usr/bin/env python3
"""
Get PyTorch audio encoder output for comparison
"""
import numpy as np
import torch
import soundfile as sf

# Try to import the processor and model
try:
    from transformers import AutoProcessor

    # Load audio
    audio, sr = sf.read('/tmp/mlk_speech_16k.wav')
    audio = audio.astype(np.float32)
    print(f"Audio: {len(audio)} samples at {sr} Hz")

    # Load processor
    model_name = "google/gemma-3n-E2B-it"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Get audio feature extractor using the audio processor
    # For Gemma 3n, audio features are computed during apply_chat_template
    from transformers import Gemma3nAudioFeatureExtractor
    fe = Gemma3nAudioFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    print(f"\nFeature extractor config:")
    print(f"  fft_length: {fe.fft_length}")
    print(f"  frame_length: {fe.frame_length}")
    print(f"  hop_length: {fe.hop_length}")
    print(f"  mel_filters shape: {fe.mel_filters.shape}")

    # Extract features
    result = fe(audio, sampling_rate=sr, return_tensors='pt')
    print(f"\nResult keys: {result.keys()}")

    if 'input_features' in result:
        mel = result['input_features'][0]
        print(f"\nMel spectrogram shape: {mel.shape}")
        print(f"Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

        # Print first frame
        print(f"\nFirst frame (first 10 bins): {mel[0, :10].numpy()}")

    # Save mel for comparison
    if 'input_features' in result:
        np.save('/tmp/pytorch_mel.npy', mel.numpy())
        print(f"\nSaved mel to /tmp/pytorch_mel.npy")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

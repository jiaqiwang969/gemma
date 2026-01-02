#!/usr/bin/env python3
"""
Check Gemma 3n audio preprocessing details
"""
import os
import torch
import warnings
import librosa
import numpy as np
import inspect

warnings.filterwarnings("ignore")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from transformers import AutoProcessor

print("Loading processor...")
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Check processor attributes
print(f"Processor type: {type(processor)}")
print(f"\nProcessor attributes:")
for key in dir(processor):
    if not key.startswith('_'):
        try:
            val = getattr(processor, key)
            if not callable(val):
                print(f"  {key}: {type(val).__name__}")
        except:
            pass

# Check feature_extractor
if hasattr(processor, 'feature_extractor'):
    fe = processor.feature_extractor
    print(f"\nFeature extractor type: {type(fe)}")
    for key in dir(fe):
        if not key.startswith('_'):
            try:
                val = getattr(fe, key)
                if not callable(val):
                    print(f"  {key}: {val}")
            except:
                pass

# Load audio and extract features
audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"\nAudio: {len(audio)/sr:.2f}s, samples: {len(audio)}")

# Get raw mel from feature extractor
if hasattr(processor, 'feature_extractor'):
    fe = processor.feature_extractor
    # Try to extract features directly
    try:
        features = fe(audio, sampling_rate=sr, return_tensors="pt")
        print(f"\nExtracted features:")
        for k, v in features.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                if v.numel() < 100:
                    print(f"    values: {v}")
                else:
                    print(f"    min={v.min():.4f}, max={v.max():.4f}, mean={v.float().mean():.4f}")
    except Exception as e:
        print(f"Error extracting features: {e}")

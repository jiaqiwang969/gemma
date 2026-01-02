#!/usr/bin/env python3
"""
Deep dive into Gemma 3n audio preprocessing
"""
import os
import torch
import numpy as np
import warnings
import librosa

warnings.filterwarnings("ignore")

from transformers import AutoProcessor
from transformers.models.gemma3n.feature_extraction_gemma3n import Gemma3nAudioFeatureExtractor

print("Loading processor...")
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
fe = processor.feature_extractor

print("\n=== Feature extractor internal state ===")
print(f"mel_filters shape: {fe.mel_filters.shape}")
print(f"mel_filters min: {fe.mel_filters.min()}")
print(f"mel_filters max: {fe.mel_filters.max()}")
print(f"mel_filters sum per mel bin (first 10): {fe.mel_filters[:10].sum(axis=1)}")

# Look at the source code
import inspect
print("\n=== Looking at _extract_fbank_features ===")
if hasattr(fe, '_extract_fbank_features'):
    source = inspect.getsource(fe._extract_fbank_features)
    # Print first 200 lines
    lines = source.split('\n')[:100]
    for line in lines:
        print(line)

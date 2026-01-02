#!/usr/bin/env python3
"""
Step by step mel spectrogram computation for Gemma 3n
"""
import numpy as np
import librosa
from scipy.signal import get_window
from transformers import AutoProcessor

# Load audio
audio_path = "assets/data/audio/mlk_speech.flac"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

# Get PyTorch result
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
fe = processor.feature_extractor

# Get features
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio, "sample_rate": sr},
            {"type": "text", "text": "Transcribe this audio."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

pytorch_mel = inputs["input_features"][0].float().cpu().numpy()  # [n_frames, n_mel]
print(f"\nPyTorch mel shape: {pytorch_mel.shape}")
print(f"PyTorch mel stats: min={pytorch_mel.min():.4f}, max={pytorch_mel.max():.4f}, mean={pytorch_mel.mean():.4f}")

# Now compute manually
print("\n=== Manual computation ===")

fft_length = 1024
frame_length = 512  # window size
hop_length = 160
preemphasis = 0.97
mel_floor = 1e-5
fmin = 125.0
fmax = 7600.0
n_mel = 128

# Step 1: Preemphasis (HTK flavor)
# According to transformers code, preemphasis_htk_flavor=True applies:
# x[i] = x[i] - coeff * x[i-1]
audio_preemph = np.copy(audio)
for i in range(1, len(audio_preemph)):
    audio_preemph[i] = audio_preemph[i] - preemphasis * audio_preemph[i-1]

# Step 2: Center padding
pad_amount = fft_length // 2
audio_padded = np.pad(audio_preemph, (pad_amount, pad_amount), mode='reflect')
print(f"Padded audio length: {len(audio_padded)}")

# Step 3: Create Hann window (periodic)
window = get_window('hann', frame_length, fftbins=True)
# Pad window to fft_length (center pad)
if frame_length < fft_length:
    pad_left = (fft_length - frame_length) // 2
    pad_right = fft_length - frame_length - pad_left
    window = np.pad(window, (pad_left, pad_right), mode='constant')
print(f"Window length: {len(window)}, sum: {window.sum():.4f}")

# Step 4: STFT
n_frames = (len(audio_padded) - fft_length) // hop_length + 1
print(f"Number of frames: {n_frames}")

# Compute spectrogram
spectrogram = np.zeros((fft_length // 2 + 1, n_frames))
for i in range(n_frames):
    start = i * hop_length
    frame = audio_padded[start:start + fft_length] * window
    fft_result = np.fft.rfft(frame)
    # Power spectrum (magnitude squared)
    spectrogram[:, i] = np.abs(fft_result) ** 2

print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Spectrogram stats: min={spectrogram.min():.6f}, max={spectrogram.max():.6f}")

# Step 5: Mel filterbank
def hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz_htk(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

n_fft_bins = fft_length // 2 + 1
mel_min = hz_to_mel_htk(fmin)
mel_max = hz_to_mel_htk(fmax)
mel_pts = np.linspace(mel_min, mel_max, n_mel + 2)
hz_pts = mel_to_hz_htk(mel_pts)
bin_hz_step = sr / fft_length

mel_filters = np.zeros((n_mel, n_fft_bins))
for m in range(n_mel):
    f_left = hz_pts[m]
    f_center = hz_pts[m + 1]
    f_right = hz_pts[m + 2]

    for k in range(n_fft_bins):
        f = k * bin_hz_step
        if f >= f_left and f <= f_center:
            mel_filters[m, k] = (f - f_left) / max(1e-30, f_center - f_left)
        elif f > f_center and f <= f_right:
            mel_filters[m, k] = (f_right - f) / max(1e-30, f_right - f_center)

# Apply mel filters: (n_mel, n_fft_bins) @ (n_fft_bins, n_frames) -> (n_mel, n_frames)
mel_spec = np.dot(mel_filters, spectrogram)
print(f"Mel spec shape: {mel_spec.shape}")

# Apply floor and log
mel_spec = np.maximum(mel_spec, mel_floor)
log_mel = np.log(mel_spec)

# Transpose to match PyTorch (n_frames, n_mel)
log_mel = log_mel.T

print(f"Manual log mel shape: {log_mel.shape}")
print(f"Manual log mel stats: min={log_mel.min():.4f}, max={log_mel.max():.4f}, mean={log_mel.mean():.4f}")

# Compare first few frames
print("\n=== Comparison ===")
print("Frame 0, first 10 mel bins:")
print(f"  PyTorch: {pytorch_mel[0, :10]}")
print(f"  Manual:  {log_mel[0, :10]}")

print("\nFrame 100, first 10 mel bins:")
print(f"  PyTorch: {pytorch_mel[100, :10]}")
print(f"  Manual:  {log_mel[100, :10]}")

# Check difference
diff = np.abs(pytorch_mel[:min(len(log_mel), len(pytorch_mel))] - log_mel[:min(len(log_mel), len(pytorch_mel))])
print(f"\nMean absolute difference: {diff.mean():.4f}")
print(f"Max absolute difference: {diff.max():.4f}")

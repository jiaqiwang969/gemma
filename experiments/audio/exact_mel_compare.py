#!/usr/bin/env python3
"""
精确对比 mel 谱图差异来源
"""
import numpy as np
import librosa
from transformers import AutoProcessor

# 加载音频
audio, sr = librosa.load("assets/data/audio/mlk_speech.flac", sr=16000)
print(f"音频: {len(audio)} 样本, {len(audio)/sr:.2f}s")

# PyTorch 结果
model_name = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
fe = processor.feature_extractor

messages = [{"role": "user", "content": [
    {"type": "audio", "audio": audio, "sample_rate": sr},
    {"type": "text", "text": "Transcribe."}
]}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
)

pytorch_mel = inputs["input_features"][0].float().cpu().numpy()
print(f"\nPyTorch mel: shape={pytorch_mel.shape}, mean={pytorch_mel.mean():.6f}")

# 精确手动计算
frame_length = 512
fft_length = 1024
hop_length = 160
preemph = 0.97
mel_floor = 1e-5

# Hann 窗口 (非周期)
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_length) / frame_length))

# 帧数
frame_size_for_unfold = frame_length + 1
n_frames = (len(audio) - frame_size_for_unfold) // hop_length + 1
print(f"帧数: {n_frames}")

# 计算每帧
log_mel = np.zeros((n_frames, 128))
for i in range(n_frames):
    offset = i * hop_length

    # 提取 513 样本并应用预加重
    first = audio[offset] * (1 - preemph)
    rest = audio[offset+1:offset+frame_length] - preemph * audio[offset:offset+frame_length-1]
    frame = np.concatenate([[first], rest])

    # 应用窗口
    frame = frame * hann

    # FFT (零填充到 1024)
    stft = np.fft.rfft(frame, n=fft_length)

    # 幅度谱
    magnitude = np.abs(stft)

    # 应用 mel 滤波器
    mel_spec = np.dot(magnitude, fe.mel_filters)

    # 取对数
    log_mel[i] = np.log(np.maximum(mel_spec, mel_floor))

print(f"手动 mel: shape={log_mel.shape}, mean={log_mel.mean():.6f}")

# 比较
diff = np.abs(pytorch_mel[:n_frames] - log_mel)
print(f"\n差异统计:")
print(f"  平均差异: {diff.mean():.8f}")
print(f"  最大差异: {diff.max():.8f}")
print(f"  差异 > 0.01 的比例: {(diff > 0.01).mean()*100:.2f}%")
print(f"  差异 > 0.001 的比例: {(diff > 0.001).mean()*100:.2f}%")

# 保存手动计算的 mel 给 llama.cpp 使用
np.save('/tmp/manual_mel.npy', log_mel.astype(np.float32))
print(f"\n已保存到 /tmp/manual_mel.npy")

# 打印前几帧进行对比
print("\n前3帧对比:")
for i in range(3):
    print(f"帧 {i}:")
    print(f"  PyTorch: {pytorch_mel[i, :5]}")
    print(f"  手动:    {log_mel[i, :5]}")
    print(f"  差异:    {np.abs(pytorch_mel[i, :5] - log_mel[i, :5])}")

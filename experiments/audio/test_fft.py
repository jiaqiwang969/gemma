#!/usr/bin/env python3
"""
验证 FFT 实现 - 生成测试数据供 llama.cpp 比较
"""
import numpy as np

# 创建一个简单的测试信号
np.random.seed(42)
test_signal = np.random.randn(512).astype(np.float32)

# 零填充到 1024
padded = np.zeros(1024, dtype=np.float32)
padded[:512] = test_signal

# FFT
fft_result = np.fft.rfft(padded)

print("测试信号 (前10个):", test_signal[:10])
print("\nFFT 结果 (前10个频率 bins):")
for i in range(10):
    print(f"  bin {i}: real={fft_result[i].real:.6f}, imag={fft_result[i].imag:.6f}, mag={np.abs(fft_result[i]):.6f}")

# 保存测试信号供 C++ 使用
np.save('/tmp/test_signal.npy', test_signal)
print("\n已保存测试信号到 /tmp/test_signal.npy")

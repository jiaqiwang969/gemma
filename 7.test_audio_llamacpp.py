#!/usr/bin/env python3
"""
测试 Gemma 3n 音频编码器 with llama.cpp
此脚本用于:
1. 转换音频编码器到 GGUF 格式
2. 使用 llama-mtmd-cli 测试音频理解

使用方法:
    python 7.test_audio_llamacpp.py [--convert-only] [--test-only]
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# 配置路径
BASE_DIR = Path(__file__).parent
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
BUILD_DIR = LLAMA_CPP_DIR / "build"
MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# 模型配置
HF_MODEL_NAME = "google/gemma-3n-E2B-it"
OUTPUT_DIR = BASE_DIR / "outputs"

# 输出文件
AUDIO_MMPROJ_GGUF = OUTPUT_DIR / "gemma-3n-audio-mmproj-f16.gguf"

def find_model_path():
    """查找 HuggingFace 缓存的模型路径"""
    # 尝试常见的缓存位置
    cache_patterns = [
        MODEL_DIR / "models--google--gemma-3n-E2B-it" / "snapshots",
        Path("/root/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots"),
    ]

    for cache_dir in cache_patterns:
        if cache_dir.exists():
            # 获取最新的快照
            snapshots = list(cache_dir.iterdir())
            if snapshots:
                return snapshots[0]

    # 如果找不到缓存，返回 HuggingFace 模型名称
    return HF_MODEL_NAME


def convert_audio_encoder():
    """转换音频编码器到 GGUF 格式"""
    print("=" * 60)
    print("步骤 1: 转换音频编码器到 GGUF")
    print("=" * 60)

    model_path = find_model_path()
    print(f"模型路径: {model_path}")

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"错误: 找不到转换脚本 {convert_script}")
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(AUDIO_MMPROJ_GGUF),
        "--mmproj",
        "--mmproj-name", "Gemma3nAudioModel",
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 60)
        print(f"转换成功! 输出文件: {AUDIO_MMPROJ_GGUF}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")
        return False


def download_test_audio():
    """下载测试音频"""
    import requests

    audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    audio_path = Path("/tmp/test_audio.flac")

    if audio_path.exists():
        print(f"使用已有测试音频: {audio_path}")
        return audio_path

    print(f"下载测试音频: {audio_url}")
    response = requests.get(audio_url)
    with open(audio_path, "wb") as f:
        f.write(response.content)
    print(f"音频已保存到: {audio_path}")
    return audio_path


def convert_audio_to_wav(input_path, output_path="/tmp/test_audio.wav"):
    """将音频转换为 WAV 格式 (llama.cpp 需要 WAV)"""
    try:
        import librosa
        import soundfile as sf

        print(f"转换音频格式: {input_path} -> {output_path}")
        audio, sr = librosa.load(input_path, sr=16000)
        sf.write(output_path, audio, sr)
        print(f"转换完成: 采样率={sr}Hz, 长度={len(audio)/sr:.2f}秒")
        return output_path
    except ImportError:
        print("警告: 需要安装 librosa 和 soundfile")
        print("运行: pip install librosa soundfile")
        return None


def test_audio_with_llamacpp(text_model_gguf=None):
    """使用 llama-mtmd-cli 测试音频"""
    print("\n" + "=" * 60)
    print("步骤 2: 使用 llama-mtmd-cli 测试音频")
    print("=" * 60)

    mtmd_cli = BUILD_DIR / "bin" / "llama-mtmd-cli"
    if not mtmd_cli.exists():
        mtmd_cli = BUILD_DIR / "llama-mtmd-cli"

    if not mtmd_cli.exists():
        print(f"错误: 找不到 llama-mtmd-cli")
        print(f"请先编译: cd {BUILD_DIR} && cmake --build . --target llama-mtmd-cli")
        return False

    if not AUDIO_MMPROJ_GGUF.exists():
        print(f"错误: 找不到音频 mmproj 文件: {AUDIO_MMPROJ_GGUF}")
        print("请先运行转换: python 7.test_audio_llamacpp.py --convert-only")
        return False

    # 查找文本模型
    if text_model_gguf is None:
        # 尝试查找已有的量化模型
        possible_models = [
            OUTPUT_DIR / "gemma-3n-finetuned-Q4_K_M.gguf",
            OUTPUT_DIR / "gemma-3n-Q4_K_M.gguf",
            OUTPUT_DIR / "gemma-3n.gguf",
        ]
        for model in possible_models:
            if model.exists():
                text_model_gguf = model
                break

    if text_model_gguf is None or not Path(text_model_gguf).exists():
        print("错误: 找不到文本模型 GGUF 文件")
        print("请指定文本模型路径，或确保 outputs/ 目录下有 gemma-3n-*.gguf 文件")
        return False

    # 下载并转换测试音频
    audio_path = download_test_audio()
    wav_path = convert_audio_to_wav(audio_path)

    if wav_path is None:
        print("无法转换音频文件")
        return False

    cmd = [
        str(mtmd_cli),
        "-m", str(text_model_gguf),
        "--mmproj", str(AUDIO_MMPROJ_GGUF),
        "-f", str(wav_path),
        "-p", "Please transcribe this audio.",
        "-n", "256",
    ]

    print(f"\n执行命令:")
    print(f"  {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        return False


def check_dependencies():
    """检查依赖"""
    missing = []

    try:
        import requests
    except ImportError:
        missing.append("requests")

    try:
        import librosa
    except ImportError:
        missing.append("librosa")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    if missing:
        print("警告: 缺少以下依赖:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="测试 Gemma 3n 音频编码器 with llama.cpp")
    parser.add_argument("--convert-only", action="store_true", help="仅转换,不测试")
    parser.add_argument("--test-only", action="store_true", help="仅测试,不转换")
    parser.add_argument("--model", type=str, help="文本模型 GGUF 路径")
    args = parser.parse_args()

    print("=" * 60)
    print("Gemma 3n 音频编码器 llama.cpp 测试")
    print("=" * 60)
    print(f"llama.cpp 路径: {LLAMA_CPP_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    if not check_dependencies():
        print("\n请安装缺少的依赖后重试")
        sys.exit(1)

    if args.test_only:
        success = test_audio_with_llamacpp(args.model)
    elif args.convert_only:
        success = convert_audio_encoder()
    else:
        # 默认: 先转换再测试
        success = convert_audio_encoder()
        if success:
            success = test_audio_with_llamacpp(args.model)

    if success:
        print("\n" + "=" * 60)
        print("完成!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("出现错误,请检查上面的输出")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

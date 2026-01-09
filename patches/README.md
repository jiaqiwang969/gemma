# Gemma-3n Multimodal Patches for llama.cpp

This directory contains patches to add **Gemma-3n multimodal (vision + audio)** support to llama.cpp.

## Origin

This patch is based on **PR #18256** by [@simrnsingh](https://github.com/simrnsingh):
- **PR Link**: https://github.com/ggerganov/llama.cpp/pull/18256
- **PR Title**: "Add Gemma3n multimodal support with MobileNetV5 vision encoder"

We extended the original vision-only PR to add **audio encoder (Conformer)** support.

## Target Version

| Item | Value |
|------|-------|
| **Base Repository** | https://github.com/ggerganov/llama.cpp |
| **Base PR** | [#18256](https://github.com/ggerganov/llama.cpp/pull/18256) by @simrnsingh |
| **Merge Base Commit** | `fd05c51cec7e233bddf2d2bae85ddf8aa6b0226c` (master at 2025-12-21) |
| **Our Branch** | `pr-18256` + audio encoder additions + bug fixes |

## Patch Contents

`gemma3n-multimodal.patch` adds support for:

### Vision Encoder (MobileNetV5)
- Image size: 768x768
- Output: 256 vision tokens (16x16 grid)
- Architecture: MobileNetV5 with MSFA (Multi-Scale Feature Aggregation)

### Audio Encoder (Conformer)
- Sample rate: 16kHz
- Mel bins: 128
- Output: 188 audio soft tokens -> 2048 dimensions
- Architecture: 12-layer Conformer with LConv1D

### Bug Fixes (2025-01-09)

1. **Input dimension transpose** (`gemma3na.cpp`): Fixed input tensor dimension order from `[T, F, 1]` to `[F, T, 1]`
2. **Fixed output token count** (`clip.cpp`): Gemma 3n audio always outputs 188 soft tokens
3. **CUDA kernel size 5 support** (`ggml-cuda/ssm-conv.cu`): Added support for kernel size 5 in SSM conv

### Modified Files

```
convert_hf_to_gguf.py             | +332  (Gemma3nVisionModel, Gemma3nAudioModel classes)
gguf-py/gguf/constants.py         | +110  (new tensor types for vision/audio)
gguf-py/gguf/tensor_mapping.py    | +138  (HF -> GGUF tensor name mapping)
src/models/gemma3n-iswa.cpp       | +18   (per-layer vision embedding fix)
tools/mtmd/CMakeLists.txt         | +2    (build config)
tools/mtmd/clip-impl.h            | +82   (tensor name macros)
tools/mtmd/clip-model.h           | +56   (model structures)
tools/mtmd/clip.cpp               | +225  (Gemma3na projector handling + fixes)
tools/mtmd/models/gemma3na.cpp    | +722  (Conformer audio encoder + fixes)
tools/mtmd/models/mobilenetv5.cpp | +453  (MobileNetV5 vision encoder)
tools/mtmd/models/models.h        | +38   (model declarations)
tools/mtmd/mtmd.cpp               | +11   (multi-mmproj loading)
ggml/src/ggml-cuda/ssm-conv.cu    | +1    (kernel size 5 support)
```

**Total: 13 files, ~2200 lines**

## Source Files (src/)

For convenience, we also provide the **complete source files** (not just diffs) for the core components:

```
src/
├── gemma3na.cpp      # Conformer audio encoder (722 lines) - CRITICAL
├── mobilenetv5.cpp   # MobileNetV5 vision encoder (453 lines)
├── models.h          # Model declarations
├── clip.cpp          # CLIP implementation with Gemma3na support
├── clip-impl.h       # Tensor name macros
├── clip-model.h      # Model structures
└── ggml-cuda/
    └── ssm-conv.cu   # CUDA SSM conv with kernel size 5 support
```

**Why provide source files?**
- The patch file requires exact base commit match
- Source files can be directly copied to `tools/mtmd/models/` if patch fails
- Easier to review and understand the implementation

### Direct Copy Method (Alternative to Patch)

If the patch doesn't apply cleanly:

```bash
# Copy source files directly
cp src/gemma3na.cpp /path/to/llama.cpp/tools/mtmd/models/
cp src/mobilenetv5.cpp /path/to/llama.cpp/tools/mtmd/models/
cp src/models.h /path/to/llama.cpp/tools/mtmd/models/
cp src/clip-impl.h /path/to/llama.cpp/tools/mtmd/
cp src/clip-model.h /path/to/llama.cpp/tools/mtmd/
cp src/clip.cpp /path/to/llama.cpp/tools/mtmd/

# For CUDA support (kernel size 5)
cp src/ggml-cuda/ssm-conv.cu /path/to/llama.cpp/ggml/src/ggml-cuda/
```

## How to Apply

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Checkout to the merge base commit
git checkout fd05c51cec7e233bddf2d2bae85ddf8aa6b0226c

# Apply the patch
git apply /path/to/gemma3n-multimodal.patch

# Build with multimodal support
cmake -B build -DGGML_METAL=ON  # macOS with Metal
# or
cmake -B build -DGGML_CUDA=ON   # Linux/Windows with NVIDIA GPU

cmake --build build --config Release -j
```

## Platform Support

| Platform | GPU | Status | Notes |
|----------|-----|--------|-------|
| macOS (Apple Silicon) | Metal | Tested | Full support |
| Linux x86_64 | NVIDIA CUDA | Tested | Full support (with ssm-conv fix) |
| Linux x86_64 | CPU only | Tested | Works but slower |
| Windows | NVIDIA CUDA | Untested | Should work |

## Convert Models

```bash
# Convert vision mmproj
python convert_hf_to_gguf.py \
  /path/to/gemma-3n-model \
  --outfile gemma-3n-vision-mmproj-f16.gguf \
  --outtype f16 \
  --mmproj \
  --mmproj-type vision

# Convert audio mmproj
python convert_hf_to_gguf.py \
  /path/to/gemma-3n-model \
  --outfile gemma-3n-audio-mmproj-f16.gguf \
  --outtype f16 \
  --mmproj \
  --mmproj-type audio
```

## Run Inference

```bash
# Vision + Audio (both mmproj files)
./build/bin/llama-mtmd-cli \
  -m gemma-3n-E2B-it-Q4_K_M.gguf \
  --mmproj "gemma-3n-vision-mmproj-f16.gguf,gemma-3n-audio-mmproj-f16.gguf" \
  --image test.jpg \
  -p "Describe this image"

# Audio only
./build/bin/llama-mtmd-cli \
  -m gemma-3n-E2B-it-Q4_K_M.gguf \
  --mmproj gemma-3n-audio-mmproj-f16.gguf \
  --audio test.wav \
  -p "Transcribe this audio"

# With CUDA GPU acceleration
./build/bin/llama-mtmd-cli \
  -ngl 99 \
  -m gemma-3n-E2B-it-Q4_K_M.gguf \
  --mmproj gemma-3n-audio-mmproj-f16.gguf \
  --audio test.wav \
  -p "Transcribe this audio"
```

## License

These patches are provided under the same license as llama.cpp (MIT License).

## Credits

- **Vision encoder (MobileNetV5)**: [@simrnsingh](https://github.com/simrnsingh) - [PR #18256](https://github.com/ggerganov/llama.cpp/pull/18256)
- **Audio encoder (Conformer)**: LingKong AI Team
- **Bug fixes (input transpose, token count, CUDA kernel)**: LingKong AI Team
- **Original llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Gemma-3n model**: Google DeepMind

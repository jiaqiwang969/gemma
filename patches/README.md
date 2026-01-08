# Gemma-3n Multimodal Patches for llama.cpp

This directory contains patches to add **Gemma-3n multimodal (vision + audio)** support to llama.cpp.

## Target Version

| Item | Value |
|------|-------|
| **Base Repository** | https://github.com/ggerganov/llama.cpp |
| **Base Tag** | `b7499` |
| **Base Commit** | `fd05c51cec7e233bddf2d2bae85ddf8aa6b0226c` |
| **Base Date** | 2025-12-21 |

## Patch Contents

`gemma3n-multimodal.patch` adds support for:

### Vision Encoder (MobileNetV5)
- Image size: 768x768
- Output: 256 vision tokens (16x16 grid)
- Architecture: MobileNetV5 with MSFA (Multi-Scale Feature Aggregation)

### Audio Encoder (Conformer)
- Sample rate: 16kHz
- Mel bins: 128
- Output: 188 audio soft tokens → 2048 dimensions
- Architecture: 12-layer Conformer with LConv1D

### Modified Files

```
convert_hf_to_gguf.py             | +332  (Gemma3nVisionModel, Gemma3nAudioModel classes)
gguf-py/gguf/constants.py         | +110  (new tensor types for vision/audio)
gguf-py/gguf/tensor_mapping.py    | +138  (HF → GGUF tensor name mapping)
src/models/gemma3n-iswa.cpp       | +18   (per-layer vision embedding fix)
tools/mtmd/CMakeLists.txt         | +2    (build config)
tools/mtmd/clip-impl.h            | +82   (tensor name macros)
tools/mtmd/clip-model.h           | +56   (model structures)
tools/mtmd/clip.cpp               | +219  (Gemma3na projector handling)
tools/mtmd/models/gemma3na.cpp    | +716  (Conformer audio encoder)
tools/mtmd/models/mobilenetv5.cpp | +453  (MobileNetV5 vision encoder)
tools/mtmd/models/models.h        | +38   (model declarations)
tools/mtmd/mtmd.cpp               | +11   (multi-mmproj loading)
```

**Total: 12 files, +2163 lines**

## How to Apply

```bash
# Clone llama.cpp at the correct version
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b7499

# Apply the patch
git apply /path/to/gemma3n-multimodal.patch

# Build with multimodal support
cmake -B build -DGGML_METAL=ON  # or -DGGML_CUDA=ON for NVIDIA
cmake --build build --config Release -j
```

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
```

## License

These patches are provided under the same license as llama.cpp (MIT License).

## Credits

- Original llama.cpp: https://github.com/ggerganov/llama.cpp
- Gemma-3n model: Google DeepMind
- Patches: LingKong AI Team

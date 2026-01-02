# Naming Conventions

This project uses consistent, descriptive names to keep entrypoints and artifacts discoverable.

## Directories

- Use lower-case words; multi-word directories use `snake_case`.
- Primary entrypoints live under `apps/` and `scripts/`.

## Python files

- Use `snake_case.py`.
- Prefer explicit, task-focused names: `finetune_lora.py`, `multimodal_inference.py`.

## Shell scripts

- Entrypoints use `run_<task>.sh` (e.g., `run_text.sh`).
- llama.cpp-specific runners use `*_cpp.sh` (e.g., `run_audio_cpp.sh`).

## Artifacts

- GGUF models: `gemma-3n-<variant>-<quant>.gguf` (e.g., `gemma-3n-finetuned-Q4_K_M.gguf`).
- mmproj encoders: `gemma-3n-<modality>-mmproj-<dtype>.gguf` (e.g., `gemma-3n-image-mmproj-f16.gguf`).

## Docs

- Use lower-kebab-case for markdown filenames.

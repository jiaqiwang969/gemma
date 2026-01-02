# Directory Structure

Goal: group code by bounded context and make runtime entrypoints explicit.

## Target Layout

apps/
  webui/               # UI + API server (Inference entrypoint)
  examples/            # runnable demos (Inference + Training + ModelOps)
contexts/
  inference/           # inference-facing scripts and adapters
  modelops/            # conversion and artifact management
  training/            # finetuning and evaluation
infra/
  llama.cpp/           # external runtime and gguf tooling (modified)
artifacts/
  gguf/                # gguf + mmproj outputs
assets/
  data/                # sample data (audio/image)
experiments/
  audio/               # debug/compare audio scripts
  vision/              # debug/compare vision scripts
  notebooks/           # notebooks and scratch work
scripts/
  bootstrap/           # setup helpers and wrappers
  maintenance/         # cleanup, checks

docs/
  architecture/        # DDD docs and conventions

## Migration Rules

- Entry points live under apps/ or scripts/.
- Shared domain logic should move under contexts/.
- Large third-party code stays under infra/.
- Model artifacts live under artifacts/gguf.
- Sample data lives under assets/data.
- Compatibility symlinks are not used; paths should reference the new layout directly.

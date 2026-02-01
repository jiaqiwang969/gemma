# Context Map (DDD)

This repo is reorganized around three bounded contexts with explicit dependencies.

## Bounded Contexts

1) Inference
- Responsibility: multimodal inference (text / image / audio), session handling, runtime prompts
- Core entities: Session, MediaInput, InferenceRequest, InferenceResult
- Interfaces: CLI/Web UI, model runtime adapters

2) Assistant (OpenClaw)
- Responsibility: WhatsApp gateway, multi-agent orchestration, skills/tools, long-running automations
- Core entities: Agent, Session, Skill, ToolCall, ChannelMessage
- Interfaces: WhatsApp (networked), LingKong local API (localhost), local shell/tools (offline-only)

3) ModelOps
- Responsibility: model artifacts, gguf/mmproj conversion, quantization, registry
- Core entities: ModelArtifact, Projector, ConversionJob
- Interfaces: conversion scripts, artifact storage

4) Training
- Responsibility: finetuning, evaluation, export
- Core entities: TrainingRun, Dataset, Checkpoint
- Interfaces: training scripts, evaluation runners

## Dependencies

Training -> ModelOps -> Inference
Assistant -> Inference

- Training produces checkpoints and merged weights.
- ModelOps transforms artifacts for inference (gguf, mmproj).
- Inference consumes artifacts and serves requests.
- Assistant consumes inference APIs and exposes them through messaging channels.

## Context Interaction

[Training] --(artifacts)-> [ModelOps] --(runtime assets)-> [Inference]
[Assistant] --(localhost API)-> [Inference]

## External Systems

- llama.cpp (mtmd + gguf runtime) is treated as Infrastructure for Inference
- transformers (MPS runtime) is treated as Infrastructure for Inference
- OpenClaw upstream is treated as Infrastructure for Assistant (vendored via submodule + patches)

# Offline Assistant (macOS arm64)

Goal: ship an all-in-one, local-first assistant that runs fully offline **except** for the WhatsApp transport.

## Components

- LingKong (Inference)
  - Provides local text/image/audio inference via a localhost API.
  - Must not require cloud calls at runtime.

- OpenClaw / Clawdbot (Assistant)
  - Provides WhatsApp gateway + multi-agent orchestration + skills/tools.
  - Must be configured and guarded so that it does not access the public internet at runtime.
  - The only permitted network traffic is WhatsApp itself + localhost calls to LingKong.

## Voice UX (WhatsApp)

- Inbound WhatsApp voice note → OpenClaw downloads media → offline STT via `whisper-cli` → local LLM via LingKong (Gemini-compatible) → offline TTS via macOS `say` → sends back a WhatsApp voice note/audio.
- For inbound voice notes, the default UX is **voice-only replies** (no caption text). If the audio send fails, OpenClaw falls back to text-only (without technical error noise).

## Offline policy

- Allowed:
  - WhatsApp network traffic (Baileys).
  - `localhost` / `127.0.0.1` calls to LingKong.

- Disallowed:
  - Any other outbound network: web search/fetch, remote model APIs, update checks, browser automation, package downloads, etc.

Enforcement is implemented in two layers:

1) Configuration defaults: ship an offline profile that disables all networked tools/providers.
2) Runtime guards: block any non-local HTTP(S) requests even if misconfigured.

## Performance knobs

- Whisper STT:
  - `WHISPER_CPP_MODEL` (default: `~/.lingkong/models/whisper/ggml-small.bin`)
  - `WHISPER_CPP_THREADS` (default: auto; capped)
  - `WHISPER_CPP_LANG` (default: `auto`)
- Multimodal projectors (reduce llama-server overhead if you only need text/WhatsApp voice):
  - `LINGKONG_ENABLE_VISION_MMPROJ=0|1` (default: 1)
  - `LINGKONG_ENABLE_AUDIO_MMPROJ=0|1` (default: 0)

## Repo integration approach

- Track OpenClaw upstream as a git submodule under `infra/openclaw` pinned to a specific commit.
- Keep all LingKong-specific changes as patches under `patches/openclaw/`.
- Build a macOS arm64 bundle in CI and publish it alongside `install.sh`, so the installer always pulls a matching artifact.

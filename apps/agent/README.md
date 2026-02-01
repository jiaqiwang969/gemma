# Agent (OpenClaw integration)

This app will host the glue layer that turns LingKong (local inference) + OpenClaw/Clawdbot (WhatsApp gateway) into a single offline-first product.

Planned responsibilities:

- generate default OpenClaw config for “offline mode” (no network except WhatsApp + localhost)
- wiring for STT (WhatsApp voice -> LingKong audio -> text)
- wiring for offline TTS (text -> macOS say -> audio -> WhatsApp voice note)
- service supervision (launchd) and health checks

Current artifacts:

- Offline config template (to be copied to `~/.lingkong/openclaw/openclaw.json`):
  - `apps/agent/config/openclaw.offline.macos.arm64.json5`

Implementation will land in phases; see the plan `lingkong-openclaw-offline-macos`.

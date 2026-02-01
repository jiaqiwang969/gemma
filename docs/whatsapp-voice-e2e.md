# WhatsApp Voice E2E (macOS arm64, Offline)

This is a manual smoke test for the offline WhatsApp voice assistant flow:

voice note -> local STT -> local LLM -> local TTS -> WhatsApp voice reply.

## Prereqs

- macOS arm64
- Installed via `curl -fsSL https://lingkong.xyz/install.sh | bash`
- Services running locally (`~/.lingkong/bin/lingkong start`)

## Steps

1) Check local services:

```bash
~/.lingkong/bin/lingkong status
curl -fsS http://127.0.0.1:5001/health >/dev/null && echo OK
```

2) Check OpenClaw gateway:

```bash
~/.lingkong/bin/openclaw gateway health --json
```

If WhatsApp is not linked, do:

```bash
~/.lingkong/bin/lingkong agent login
```

This will generate `~/.lingkong/tmp/whatsapp-qr.png` (and open it on macOS), then wait for the scan.

Fallback (ASCII QR in terminal):

```bash
~/.lingkong/bin/openclaw channels login --verbose
```

3) Send a WhatsApp voice note to the bot.

Expected behavior:

- Bot replies with a **voice message** (no caption text by default).
- If audio send fails, bot falls back to **text-only** (no technical warnings).

## Performance knobs

- Whisper STT:
  - `WHISPER_CPP_THREADS=4` (try 2/4/6/8)
  - `WHISPER_CPP_MODEL=~/.lingkong/models/whisper/ggml-small.bin` (can swap to smaller models for speed)
- Disable optional multimodal projectors (to reduce llama-server overhead):
  - `LINGKONG_ENABLE_AUDIO_MMPROJ=0` (default)
  - `LINGKONG_ENABLE_VISION_MMPROJ=0` (optional)

Example:

```bash
LINGKONG_ENABLE_VISION_MMPROJ=0 \
LINGKONG_ENABLE_AUDIO_MMPROJ=0 \
WHISPER_CPP_THREADS=4 \
~/.lingkong/bin/lingkong restart
```

## Logs

```bash
tail -n 200 ~/.lingkong/logs/gemini.log
tail -n 200 ~/.lingkong/logs/openclaw.log
tail -n 200 /tmp/openclaw/openclaw-*.log
```

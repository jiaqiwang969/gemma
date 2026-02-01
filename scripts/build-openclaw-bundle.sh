#!/usr/bin/env bash
set -euo pipefail

# Build a macOS arm64 runtime bundle for OpenClaw (Clawdbot) that can run offline
# (except WhatsApp) after installation.
#
# This is intentionally a repo-local build step; `scripts/install.sh` should only
# download the published bundle and unpack it.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OPENCLAW_DIR="$ROOT_DIR/infra/openclaw"
PATCH_DIR="$ROOT_DIR/patches/openclaw"

OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bundles"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

if [[ ! -d "$OPENCLAW_DIR/.git" ]]; then
  echo "ERROR: OpenClaw submodule not initialized at: $OPENCLAW_DIR" >&2
  echo "Run: git submodule update --init --recursive infra/openclaw" >&2
  exit 1
fi

platform="$(uname -s)-$(uname -m)"
if [[ "$platform" != "Darwin-arm64" ]]; then
  echo "ERROR: This bundle script currently targets macOS arm64 only (got $platform)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

openclaw_version="$(node -p \"require('$OPENCLAW_DIR/package.json').version\" 2>/dev/null || echo unknown)"
openclaw_commit="$(git -C \"$OPENCLAW_DIR\" rev-parse --short=12 HEAD)"
bundle_name="openclaw-macos-arm64-${openclaw_version}-${openclaw_commit}.tar.gz"

echo "Building OpenClaw bundle:"
echo "- version: $openclaw_version"
echo "- commit:  $openclaw_commit"
echo "- out:     $OUT_DIR/$bundle_name"
echo

# TODO(phase-2): implement build + patch application + prod dependency packing + node runtime inclusion.
cat <<EOF
TODO:
- Apply patches from $PATCH_DIR/*.patch onto a clean checkout of infra/openclaw
- Install and pack production node_modules for macOS arm64
- Optionally include a pinned Node runtime (downloaded in CI) to avoid system Node dependency
- Emit a bundle layout under a temp dir then tar.gz into $OUT_DIR
EOF

exit 1


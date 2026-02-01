#!/usr/bin/env bash
set -euo pipefail

# Install/update launchd services for the offline assistant on macOS.
#
# Phase-1 placeholder: we will add concrete plist templates once the bundle
# layout and commands are finalized.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

echo "TODO: launchd installation is not implemented yet."
echo "Planned services:"
echo "- lingkong (local inference API)"
echo "- openclaw (whatsapp gateway + offline agent)"
echo
echo "Repo: $ROOT_DIR"
exit 1


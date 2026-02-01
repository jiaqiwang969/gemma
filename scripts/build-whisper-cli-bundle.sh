#!/usr/bin/env bash
set -euo pipefail

# Build a macOS arm64 whisper.cpp CLI bundle ("whisper-cli") for offline STT.
#
# Output:
# - artifacts/bundles/whisper-cli-macos-arm64-<tag>-<commit>.tar.gz
#
# The installer expects a published stable name at:
# - https://lingkong.xyz/bin/whisper-cli-macos-arm64.tar.gz

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WHISPER_DIR="$ROOT_DIR/infra/whisper.cpp"

OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bundles"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

if [[ ! -e "$WHISPER_DIR/.git" ]]; then
  echo "ERROR: whisper.cpp submodule not initialized at: $WHISPER_DIR" >&2
  echo "Run: git submodule update --init --recursive infra/whisper.cpp" >&2
  exit 1
fi

platform="$(uname -s)-$(uname -m)"
if [[ "$platform" != "Darwin-arm64" ]]; then
  echo "ERROR: This bundle script currently targets macOS arm64 only (got $platform)" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake is required to build whisper.cpp (missing in PATH)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

whisper_commit="$(git -C "$WHISPER_DIR" rev-parse --short=12 HEAD)"
whisper_tag="$(git -C "$WHISPER_DIR" describe --tags --abbrev=0 2>/dev/null || echo unknown)"

bundle_name="whisper-cli-macos-arm64-${whisper_tag}-${whisper_commit}.tar.gz"

echo "Building whisper-cli bundle:"
echo "- tag:     $whisper_tag"
echo "- commit:  $whisper_commit"
echo "- out:     $OUT_DIR/$bundle_name"
echo

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/whisper-cli-bundle.XXXXXX")"
worktree_dir="$tmp_root/whisper-src"
stage_dir="$tmp_root/stage"

cleanup() {
  set +e
  if [[ -d "$worktree_dir" ]]; then
    git -C "$WHISPER_DIR" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Preparing clean whisper.cpp worktree..."
git -C "$WHISPER_DIR" worktree add --detach "$worktree_dir" HEAD >/dev/null

echo "Configuring (Release + Metal)..."
cmake -S "$worktree_dir" -B "$worktree_dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_BUILD_EXAMPLES=ON \
  -DGGML_METAL=ON
echo

echo "Building whisper-cli..."
cmake --build "$worktree_dir/build" --config Release -j "$(sysctl -n hw.ncpu)"
echo

bin_path="$worktree_dir/build/bin/whisper-cli"
if [[ ! -x "$bin_path" ]]; then
  echo "ERROR: build succeeded but whisper-cli missing at: $bin_path" >&2
  exit 1
fi

echo "Staging bundle layout..."
mkdir -p "$stage_dir/bin"
cp "$bin_path" "$stage_dir/bin/whisper-cli"
chmod +x "$stage_dir/bin/whisper-cli"

cat >"$stage_dir/BUILD_INFO.txt" <<EOF
whisper_tag=$whisper_tag
whisper_commit=$whisper_commit
built_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Creating tarball..."
tar -czf "$OUT_DIR/$bundle_name" -C "$stage_dir" .

echo "OK: wrote $OUT_DIR/$bundle_name"


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
install_dir="$tmp_root/install"
keep_tmp="${WHISPER_BUNDLE_KEEP_TMP:-0}"

cleanup() {
  set +e
  if [[ -d "$worktree_dir" ]]; then
    git -C "$WHISPER_DIR" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
  fi
  if [[ "$keep_tmp" == "1" ]]; then
    echo "NOTE: keeping temp dir for inspection: $tmp_root" >&2
    return 0
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
  -DGGML_METAL=ON \
  -DCMAKE_INSTALL_PREFIX="$install_dir"
echo

echo "Building whisper-cli..."
cmake --build "$worktree_dir/build" --config Release -j "$(sysctl -n hw.ncpu)"
echo

echo "Installing into staging prefix..."
cmake --install "$worktree_dir/build" --config Release --prefix "$install_dir"
echo

bin_path="$install_dir/bin/whisper-cli"
if [[ ! -x "$bin_path" ]]; then
  echo "ERROR: install succeeded but whisper-cli missing at: $bin_path" >&2
  exit 1
fi

fix_rpaths() {
  local target="$1"
  local desired='@loader_path/../lib'

  # Remove any existing RPATHs (CMake leaves build-tree rpaths by default).
  local rpaths
  rpaths="$(otool -l "$target" | awk '$1=="cmd" && $2=="LC_RPATH" {want=1; next} want && $1=="path" {print $2; want=0}')"
  if [[ -n "${rpaths:-}" ]]; then
    while IFS= read -r rpath; do
      [[ -z "$rpath" ]] && continue
      # Best effort: ignore failures (e.g., duplicates already removed).
      install_name_tool -delete_rpath "$rpath" "$target" 2>/dev/null || true
    done <<<"$rpaths"
  fi

  # Ensure a stable relative RPATH (same as Homebrew packaging).
  install_name_tool -add_rpath "$desired" "$target" 2>/dev/null || true
}

rewrite_to_rpath() {
  # Rewrite any absolute dylib references into @rpath/<basename> to make the
  # bundle portable across machines.
  local target="$1"
  local abs_prefixes=("$install_dir" "$tmp_root")

  # List load dependencies (skip the first line which is the binary itself).
  local deps
  deps="$(otool -L "$target" | awk 'NR>1 {print $1}')"
  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    for prefix in "${abs_prefixes[@]}"; do
      if [[ "$dep" == "$prefix"* ]]; then
        local base
        base="$(basename "$dep")"
        install_name_tool -change "$dep" "@rpath/$base" "$target" 2>/dev/null || true
      fi
    done
  done <<<"$deps"
}

echo "Staging bundle layout..."
mkdir -p "$stage_dir/bin" "$stage_dir/lib"
cp "$bin_path" "$stage_dir/bin/whisper-cli"
chmod +x "$stage_dir/bin/whisper-cli"

# Copy the dylib dependencies next to the binary (../lib), matching the default
# @loader_path/../lib rpath used by Homebrew's whisper-cpp packaging.
if [[ -d "$install_dir/lib" ]]; then
  cp "$install_dir"/lib/libwhisper*.dylib "$stage_dir/lib/" 2>/dev/null || true
  cp "$install_dir"/lib/libggml*.dylib "$stage_dir/lib/" 2>/dev/null || true
fi

echo "Rewriting install names + RPATHs for portability..."
for lib in "$stage_dir"/lib/*.dylib; do
  if [[ -f "$lib" ]]; then
    # Ensure the dylib "id" is relative so dependents don't hardcode build paths.
    install_name_tool -id "@rpath/$(basename "$lib")" "$lib" 2>/dev/null || true
  fi
done

rewrite_to_rpath "$stage_dir/bin/whisper-cli"
fix_rpaths "$stage_dir/bin/whisper-cli"
for lib in "$stage_dir"/lib/*.dylib; do
  if [[ -f "$lib" ]]; then
    rewrite_to_rpath "$lib"
    fix_rpaths "$lib"
  fi
done
echo

# Sanity check: the installed binary must NOT contain absolute build rpaths.
# We expect @loader_path/../lib and no references to the build temp dir.
if otool -l "$stage_dir/bin/whisper-cli" | sed '1d' | grep -q "$tmp_root"; then
  echo "ERROR: whisper-cli contains non-portable load commands (references $tmp_root)" >&2
  echo "Tip: re-run with WHISPER_BUNDLE_KEEP_TMP=1 to inspect the staged binary." >&2
  exit 1
fi

if ! otool -l "$stage_dir/bin/whisper-cli" | grep -q "@loader_path/../lib"; then
  echo "ERROR: whisper-cli missing expected RPATH @loader_path/../lib" >&2
  exit 1
fi

# Ensure the bundle includes every @rpath dependency referenced by the binary.
missing=0
while IFS= read -r dep; do
  [[ -z "$dep" ]] && continue
  if [[ ! -f "$stage_dir/lib/$dep" ]]; then
    echo "ERROR: missing dylib dependency in bundle: lib/$dep" >&2
    missing=1
  fi
done < <(otool -L "$stage_dir/bin/whisper-cli" | awk 'NR>1 {print $1}' | grep '^@rpath/' | sed 's#^@rpath/##')
if [[ "$missing" == "1" ]]; then
  exit 1
fi

cat >"$stage_dir/BUILD_INFO.txt" <<EOF
whisper_tag=$whisper_tag
whisper_commit=$whisper_commit
built_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Creating tarball..."
tar -czf "$OUT_DIR/$bundle_name" -C "$stage_dir" .

echo "OK: wrote $OUT_DIR/$bundle_name"

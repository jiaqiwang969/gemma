#!/usr/bin/env bash
set -euo pipefail

# Build a macOS arm64 runtime bundle for OpenClaw that can run offline
# (except WhatsApp transport) after installation.
#
# Notes:
# - This is a repo-local build step; `scripts/install.sh` should only download
#   a published bundle and unpack it.
# - We apply LingKong's patch series onto a clean OpenClaw checkout.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OPENCLAW_DIR="$ROOT_DIR/infra/openclaw"
PATCH_DIR="$ROOT_DIR/patches/openclaw"

OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bundles"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

RUN_TESTS="${OPENCLAW_BUNDLE_RUN_TESTS:-0}"

if [[ ! -e "$OPENCLAW_DIR/.git" ]]; then
  echo "ERROR: OpenClaw submodule not initialized at: $OPENCLAW_DIR" >&2
  echo "Run: git submodule update --init --recursive infra/openclaw" >&2
  exit 1
fi

platform="$(uname -s)-$(uname -m)"
if [[ "$platform" != "Darwin-arm64" ]]; then
  echo "ERROR: This bundle script currently targets macOS arm64 only (got $platform)" >&2
  exit 1
fi

if ! command -v pnpm >/dev/null 2>&1; then
  echo "ERROR: pnpm is required to build OpenClaw (missing in PATH)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

openclaw_version="$(node -p "require('$OPENCLAW_DIR/package.json').version" 2>/dev/null || echo unknown)"
openclaw_commit="$(git -C "$OPENCLAW_DIR" rev-parse --short=12 HEAD)"

patch_fingerprint="none"
shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [[ ${#patches[@]} -gt 0 ]]; then
  # Order matters; hash the ordered contents to make bundles traceable.
  patch_fingerprint="$(
    for p in "${patches[@]}"; do
      echo "==> $(basename "$p")"
      cat "$p"
      echo
    done | shasum -a 256 | awk '{print $1}' | cut -c1-12
  )"
fi

bundle_name="openclaw-macos-arm64-${openclaw_version}-${openclaw_commit}-p${patch_fingerprint}.tar.gz"

echo "Building OpenClaw bundle:"
echo "- version:  $openclaw_version"
echo "- commit:   $openclaw_commit"
echo "- patches:  ${#patches[@]} (fingerprint p${patch_fingerprint})"
echo "- out:      $OUT_DIR/$bundle_name"
echo

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/openclaw-bundle.XXXXXX")"
worktree_dir="$tmp_root/openclaw-src"
stage_dir="$tmp_root/stage"

cleanup() {
  # Best-effort cleanup; don't mask original failure.
  set +e
  if [[ -d "$worktree_dir" ]]; then
    git -C "$OPENCLAW_DIR" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Preparing clean OpenClaw worktree..."
git -C "$OPENCLAW_DIR" worktree add --detach "$worktree_dir" HEAD >/dev/null

if [[ ${#patches[@]} -gt 0 ]]; then
  echo "Applying patches:"
  for p in "${patches[@]}"; do
    echo "- $(basename "$p")"
  done
  # Apply sequentially so later patches can build on earlier ones.
  for p in "${patches[@]}"; do
    git -C "$worktree_dir" apply --whitespace=nowarn --check "$p"
    git -C "$worktree_dir" apply --whitespace=nowarn "$p"
  done
else
  echo "No patches found in $PATCH_DIR (unexpected for LingKong builds)."
fi
echo

echo "Installing dependencies..."
(cd "$worktree_dir" && pnpm install --frozen-lockfile)
echo

echo "Building..."
(cd "$worktree_dir" && pnpm build)
echo

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "Running tests (OPENCLAW_BUNDLE_RUN_TESTS=1)..."
  (cd "$worktree_dir" && pnpm test)
  echo
fi

echo "Pruning to production dependencies..."
(cd "$worktree_dir" && CI=1 pnpm prune --prod)
echo

echo "Staging bundle layout..."
mkdir -p "$stage_dir/openclaw"

copy_dir() {
  local src="$1"
  local dst="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src" "$dst"
  else
    cp -R "$src" "$dst"
  fi
}

copy_dir "$worktree_dir/openclaw.mjs" "$stage_dir/openclaw/"
copy_dir "$worktree_dir/package.json" "$stage_dir/openclaw/"
copy_dir "$worktree_dir/dist" "$stage_dir/openclaw/"
copy_dir "$worktree_dir/node_modules" "$stage_dir/openclaw/"
if [[ -d "$worktree_dir/extensions" ]]; then
  # OpenClaw loads built-in plugins (e.g. memory-core, WhatsApp) from the monorepo
  # extensions folder. Without this, default configs may fail validation.
  copy_dir "$worktree_dir/extensions" "$stage_dir/openclaw/"
fi
if [[ -d "$worktree_dir/docs" ]]; then
  # The gateway uses templates (AGENTS.md, HEARTBEAT.md, etc) when bootstrapping
  # the agent workspace. Package docs/reference/templates to avoid runtime errors.
  copy_dir "$worktree_dir/docs" "$stage_dir/openclaw/"
fi
if [[ -d "$worktree_dir/skills" ]]; then
  copy_dir "$worktree_dir/skills" "$stage_dir/openclaw/"
fi
if [[ -d "$worktree_dir/assets" ]]; then
  copy_dir "$worktree_dir/assets" "$stage_dir/openclaw/"
fi

cat >"$stage_dir/BUILD_INFO.txt" <<EOF
openclaw_version=$openclaw_version
openclaw_commit=$openclaw_commit
patch_fingerprint=p$patch_fingerprint
built_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Creating tarball..."
tar -czf "$OUT_DIR/$bundle_name" -C "$stage_dir" .

echo "OK: wrote $OUT_DIR/$bundle_name"

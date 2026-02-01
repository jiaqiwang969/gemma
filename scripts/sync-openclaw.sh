#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OPENCLAW_DIR="$ROOT_DIR/infra/openclaw"
PATCH_DIR="$ROOT_DIR/patches/openclaw"

if [[ ! -e "$OPENCLAW_DIR/.git" ]]; then
  echo "ERROR: OpenClaw submodule not initialized at: $OPENCLAW_DIR" >&2
  echo "Run: git submodule update --init --recursive infra/openclaw" >&2
  exit 1
fi

echo "OpenClaw upstream:"
git -C "$OPENCLAW_DIR" remote -v || true
echo

echo "Pinned commit:"
git -C "$OPENCLAW_DIR" rev-parse HEAD
echo

if [[ -d "$PATCH_DIR" ]]; then
  echo "Checking patches in: $PATCH_DIR"
  shopt -s nullglob
  patches=("$PATCH_DIR"/*.patch)
  if [[ ${#patches[@]} -eq 0 ]]; then
    echo "No patches found."
  else
    for p in "${patches[@]}"; do
      echo "- $(basename "$p")"
    done
    echo
    echo "Running: patch apply check (sequential, in a clean worktree)"

    tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/openclaw-patch-check.XXXXXX")"
    worktree_dir="$tmp_root/openclaw-src"

    cleanup() {
      set +e
      git -C "$OPENCLAW_DIR" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
      rm -rf "$tmp_root" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    git -C "$OPENCLAW_DIR" worktree add --detach "$worktree_dir" HEAD >/dev/null
    for p in "${patches[@]}"; do
      git -C "$worktree_dir" apply --whitespace=nowarn --check "$p"
      git -C "$worktree_dir" apply --whitespace=nowarn "$p"
    done
    echo "OK: all patches apply cleanly (as a series)."
  fi
  echo
fi

cat <<'EOF'
Next steps (manual for now):
- To move to a newer upstream commit:
    git -C infra/openclaw fetch --tags origin
    git -C infra/openclaw checkout <tag-or-commit>
    git add infra/openclaw
- Then re-run this script to ensure patches still apply.
EOF

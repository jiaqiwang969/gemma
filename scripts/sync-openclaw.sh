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
    echo "Running: git apply --check (in patch order)"
    git -C "$OPENCLAW_DIR" apply --check "${patches[@]}"
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

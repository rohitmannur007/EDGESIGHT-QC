#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="edgesight-qc-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/dist/${NAME}.tar.gz"
mkdir -p "${ROOT}/dist"

tar -C "$ROOT" -czf "$OUT" \
  --exclude='./.git' --exclude='./.venv' --exclude='./.env' \
  --exclude='./data/*' --exclude='./models/*' --exclude='./results/*' \
  --exclude='./dist' --exclude='__pycache__' --exclude='*.pyc' \
  .

LIST="$(tar -tzf "$OUT")"
for required in README.md pyproject.toml Dockerfile configs/default.yaml; do
  printf '%s\n' "$LIST" | grep -qx "./${required}" || {
    echo "Archive validation failed: missing ${required}" >&2
    exit 1
  }
done
if printf '%s\n' "$LIST" | grep -Eq '(^|/)(\.env|\.git)(/|$)|\.(pt|pth|ckpt|onnx)$'; then
  echo "Archive validation failed: private or heavyweight content included" >&2
  exit 1
fi
tar -tzf "$OUT" >/dev/null
shasum -a 256 "$OUT" >"${OUT}.sha256"
echo "$OUT"

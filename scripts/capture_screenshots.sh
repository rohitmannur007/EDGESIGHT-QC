#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://127.0.0.1:8080/docs}"
OUTPUT="${2:-results/screenshots/api-docs.png}"
mkdir -p "$(dirname "$OUTPUT")"

if ! curl --fail --silent --show-error "$URL" >/dev/null; then
  echo "Refusing to create a placeholder: no live page is reachable at $URL" >&2
  exit 1
fi
if ! command -v npx >/dev/null; then
  echo "npx is required. Install Node.js, then run: npx playwright install chromium" >&2
  exit 1
fi

# The screenshot is evidence of the live URL at capture time, not a design mockup.
npx --yes playwright@1.49.1 screenshot --wait-for-timeout=1500 --full-page "$URL" "$OUTPUT"
test -s "$OUTPUT"
echo "Captured live page: $OUTPUT"

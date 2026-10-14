#!/usr/bin/env bash
set -euo pipefail

URL="${1:?Usage: scripts/smoke_cloud_run.sh SERVICE_URL}"
TOKEN="$(gcloud auth print-identity-token)"

retry() {
  local path="$1"
  for attempt in 1 2 3 4 5; do
    if curl --fail --silent --show-error --max-time 15 \
      -H "Authorization: Bearer ${TOKEN}" "${URL}${path}" >/dev/null; then
      echo "PASS ${path}"
      return 0
    fi
    sleep "$attempt"
  done
  echo "FAIL ${path}" >&2
  return 1
}

retry /health/live
retry /health/ready
echo "Smoke checks passed. Inference was not exercised because no licensed test image was supplied."

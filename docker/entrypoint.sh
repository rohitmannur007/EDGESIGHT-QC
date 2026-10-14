#!/usr/bin/env sh
set -eu

if [ "${1:-}" = "uvicorn" ] && [ ! -r "${EDGESIGHT_CONFIG:-/app/configs/production.yaml}" ]; then
  echo "Configuration is not readable: ${EDGESIGHT_CONFIG:-/app/configs/production.yaml}" >&2
  exit 64
fi

exec "$@"

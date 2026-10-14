#!/usr/bin/env bash
set -euo pipefail

BROKERS="${EDGESIGHT_KAFKA_BROKERS:-localhost:19092}"
RPK=(docker compose exec -T redpanda rpk)
if command -v rpk >/dev/null; then
  RPK=(rpk)
fi

for topic in "${EDGESIGHT_INPUT_TOPIC:-frames}" "${EDGESIGHT_OUTPUT_TOPIC:-results}" "${EDGESIGHT_DLQ_TOPIC:-frames.dlq}" "${EDGESIGHT_MONITORING_TOPIC:-monitoring.events}"; do
  "${RPK[@]}" topic create "$topic" --brokers "$BROKERS" --partitions 3 --replicas 1 \
    || "${RPK[@]}" topic describe "$topic" --brokers "$BROKERS" >/dev/null
done
"${RPK[@]}" topic list --brokers "$BROKERS"

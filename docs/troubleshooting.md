# Troubleshooting

## Environment

Run `make check-env` first. Python must be 3.11; recreate `.venv` if another interpreter was used. If `uv sync --frozen` reports a stale/missing lock, run `uv lock`, inspect the diff, and commit `uv.lock`.

## API never becomes ready

Check `EDGESIGHT_CONFIG`, model and metadata paths, read permissions, model checksum, ONNX opset/runtime compatibility, and warmup logs. `/health/live` can pass while `/health/ready` fails; this distinction is expected.

## Out of memory or slow inference

Keep one process per container, reduce concurrency/batch size, inspect input dimensions, and confirm ONNX Runtime CPU is used. Measure on target hardware before changing thresholds or quantizing; quantization requires a new evaluation.

## Invalid image failures

Confirm MIME type, encoded size, decode success, channel ordering, and pixel bounds. Do not disable validation to accept malformed input. Compare preprocessing metadata with the promoted bundle.

## Stream worker lag

Inspect Redpanda health and console, consumer-group lag, partition count, fetch latency, inference time, rebalances, and DLQ. Adding workers beyond input partitions adds no active parallelism. Avoid embedding image bytes in events.

## Compose health failures

Use `docker compose ps` and `docker compose logs redpanda console`. Ports 19092/18082/19644/8081 must be free. Remove the volume only when intentionally discarding local broker data.

## Cloud Run 401/403

The service is private. Use an identity with `roles/run.invoker` and send an identity token. Separately verify runtime service-account access to model storage and logging.

## Missing screenshots or metrics

The screenshot script refuses placeholders when the page is unreachable. The report script states when no JSON artifacts exist. Start the real service/run the real evaluation rather than creating representative output.

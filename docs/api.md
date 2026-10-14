# HTTP API

The implementation exposes `edgesight.api.main:app`; `edgesight.api.app:create_app` supports dependency injection. OpenAPI at `/docs` is authoritative once running.

## Endpoints

- `GET /health/live`: process liveness; no dependency work.
- `GET /health/ready`: pipeline and persistence readiness.
- `GET /metrics`: Prometheus text format when enabled.
- `POST /v1/infer`: multipart field `file`; request IDs use `X-Request-ID`.

Illustrative response schema (field presence is the contract; values are not measured claims):

```json
{
  "request_id": "caller-or-generated-id",
  "model_version": "bundle-version",
  "decision": "review",
  "score": 0.0,
  "predictions": [],
  "processing_ms": 0.0
}
```

`decision` is `pass`, `review`, or `fail`. Scores must document direction and calibration in model metadata. Error bodies should include `code`, safe `detail`, and `request_id`.

## Controls

Accept only configured MIME types, enforce encoded and decoded size limits, reject malformed images, and time-bound processing. Production must require platform IAM or an API key; never put keys in URLs. CORS is disabled by default. Do not log image bytes.

## Compatibility

Use a `/v1` path for breaking-schema isolation. Additive fields are allowed; clients must ignore unknown fields. A model change is not an API version change if semantics remain compatible, but responses must identify the model version.

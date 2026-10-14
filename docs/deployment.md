# Deployment

## Container

The Dockerfile builds dependencies separately and runs on Debian slim as UID/GID 10001. It includes CPU runtime libraries only, has a liveness health check, and does not copy datasets, results, `.env`, or development tools. Commit `uv.lock` before release for reproducibility.

```bash
docker build -t edgesight-qc:local .
docker run --rm -p 8080:8080 \
  -v "$PWD/models:/app/models:ro" \
  --env-file .env edgesight-qc:local
```

Treat model bundles as immutable, checksum-verified release inputs. A production orchestrator should mount/download one approved bundle before readiness, use a read-only root filesystem where possible, drop capabilities, restrict egress, and set CPU/memory limits.

## Cloud Run

Prerequisites: authenticated `gcloud`, an existing Artifact Registry repository, enabled Cloud Build/Run APIs, and IAM permissions. The script deploys a private service:

```bash
export GOOGLE_CLOUD_PROJECT=example
export GOOGLE_CLOUD_REGION=us-central1
export ARTIFACT_REPOSITORY=edgesight
bash scripts/deploy_cloud_run.sh
bash scripts/smoke_cloud_run.sh https://SERVICE_URL
```

Do not bake proprietary model weights into a generally distributed image. Prefer a controlled artifact pipeline or private storage download with checksum verification. Grant the runtime identity only object-read and telemetry permissions it needs. Put secrets in Secret Manager, not environment files or command history.

## Release gate

Run `uv sync --frozen --all-extras`, `make check`, image vulnerability scanning, software/license review, model-card review, CPU load testing, and rollback rehearsal. Deploy a revision without traffic, smoke `/health/live` and `/health/ready`, then shift traffic gradually. Monitor errors, latency, memory, decision distribution, and drift signals.

## Rollback

Retain the previous image digest and model bundle. Route traffic back first; do not retrain during an incident. Record the revision, model checksum, reason, impact window, and follow-up action.

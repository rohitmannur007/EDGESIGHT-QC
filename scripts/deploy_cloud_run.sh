#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
SERVICE="${EDGESIGHT_SERVICE:-edgesight-qc}"
REPOSITORY="${ARTIFACT_REPOSITORY:-edgesight}"
IMAGE="${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${REPOSITORY}/${SERVICE}:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

command -v gcloud >/dev/null || { echo "gcloud CLI is required" >&2; exit 127; }
gcloud artifacts repositories describe "$REPOSITORY" --location "$REGION" >/dev/null
gcloud builds submit --project "$GOOGLE_CLOUD_PROJECT" --tag "$IMAGE" .
gcloud run deploy "$SERVICE" \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --port 8080 \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 8 \
  --min-instances 0 \
  --max-instances 10 \
  --no-allow-unauthenticated \
  --set-env-vars "EDGESIGHT_ENV=production,EDGESIGHT_CONFIG=/app/configs/production.yaml"

URL="$(gcloud run services describe "$SERVICE" --project "$GOOGLE_CLOUD_PROJECT" --region "$REGION" --format='value(status.url)')"
echo "Deployed ${IMAGE} to ${URL}"
echo "Run: scripts/smoke_cloud_run.sh ${URL}"

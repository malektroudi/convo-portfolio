#!/usr/bin/env bash
# Deploy the hub to Cloud Run. Defaults chosen to sit inside the always-free tier:
# scale to zero between visits, one small instance, a low ceiling.
#
#   ./scripts/deploy.sh
#   SERVICE=convo-portfolio REGION=europe-west1 ./scripts/deploy.sh

set -euo pipefail

SERVICE="${SERVICE:-convo-portfolio}"
REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not on PATH. Install the Google Cloud CLI first:" >&2
  echo "  https://cloud.google.com/sdk/docs/install" >&2
  echo "Then: gcloud init  (and gcloud auth login)" >&2
  exit 1
fi

if [ -z "${PROJECT}" ] || [ "${PROJECT}" = "(unset)" ]; then
  echo "No project set. Run: gcloud config set project <id>" >&2
  exit 1
fi

echo "Deploying '${SERVICE}' to ${REGION} in project ${PROJECT}"

if [ "${SKIP_APIS:-0}" != "1" ]; then
  echo "Enabling the three APIs this needs (idempotent)..."
  gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project "${PROJECT}"
fi

# --source builds the Dockerfile with Cloud Build, so no local docker push is needed.
gcloud run deploy "${SERVICE}" \
  --source . \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 20 \
  --timeout 300

URL="$(gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" --region "${REGION}" --format 'value(status.url)')"

echo
echo "Live at: ${URL}"
echo "min-instances is 0, so the first request after a quiet spell takes a few seconds."

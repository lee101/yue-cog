#!/usr/bin/env bash
# Build both image tags, push them, save a new RunPod template and point the
# endpoint at it. 10s scaledown + max 1 worker keeps idle cost at zero.
#   RUNPOD_API_KEY=... HF_TOKEN=... scripts/build_push_deploy.sh [endpoint-id]
set -euo pipefail
ENDPOINT_ID="${1:-}"
IMAGE="${IMAGE:-ghcr.io/lee101/yue-cog}"
cd "$(dirname "$0")/.."
DOCKER_BUILDKIT=1 docker build --network host -f Dockerfile.cog -t "$IMAGE:latest" .
docker build --network host -f Dockerfile.sls -t "$IMAGE:sls" .
docker push "$IMAGE:latest" | tail -1
docker push "$IMAGE:sls" | tail -1
OUT=$(python3 scripts/runpod_deploy.py ${ENDPOINT_ID:+--endpoint-id "$ENDPOINT_ID"} --image "$IMAGE:sls" --name yue2 \
  --gpu-ids "${GPU_IDS:-ADA_24,AMPERE_24}" --workers-max "${WORKERS_MAX:-1}" --idle "${IDLE_SECONDS:-10}" --disk-gb "${DISK_GB:-40}" \
  ${RUNPOD_REGISTRY_AUTH_ID:+--registry-auth-id "$RUNPOD_REGISTRY_AUTH_ID"})
echo "$OUT"

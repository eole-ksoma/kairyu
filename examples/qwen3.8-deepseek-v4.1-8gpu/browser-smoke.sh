#!/usr/bin/env bash
# Verify the live tiered product through the pinned Open WebUI browser surface.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BROWSER_IMAGE="kairyu-webui-browser:playwright-1.60.0"
WEBUI_BASE_URL="${WEBUI_BASE_URL:-http://127.0.0.1:3009}"

docker build \
  --file "$REPO_ROOT/deploy/compose/Dockerfile.webui-browser" \
  --tag "$BROWSER_IMAGE" \
  "$REPO_ROOT"

docker run --rm --init --network host \
  --env WEBUI_SMOKE_PHASE=tiered \
  --env WEBUI_SMOKE_BASE_URL="$WEBUI_BASE_URL" \
  --env EXPECTED_PRODUCT_MODEL=kairyu-auto-max \
  --env WEBUI_SMOKE_RESPONSE_TIMEOUT_MS="${WEBUI_SMOKE_RESPONSE_TIMEOUT_MS:-1800000}" \
  --env WEBUI_SMOKE_PHASE_TIMEOUT_MS="${WEBUI_SMOKE_PHASE_TIMEOUT_MS:-2100000}" \
  --volume "$REPO_ROOT/scripts/webui_browser_smoke.mjs:/work/webui_browser_smoke.mjs:ro" \
  "$BROWSER_IMAGE" node /work/webui_browser_smoke.mjs

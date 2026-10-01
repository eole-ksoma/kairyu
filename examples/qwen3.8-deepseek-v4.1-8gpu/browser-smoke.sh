#!/usr/bin/env bash
# Verify the live product through the pinned Open WebUI browser surface with
# this example's own browser gate (webui-browser-smoke.mjs).
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
  --env WEBUI_SMOKE_BASE_URL="$WEBUI_BASE_URL" \
  --env WEBUI_SMOKE_RESPONSE_TIMEOUT_MS="${WEBUI_SMOKE_RESPONSE_TIMEOUT_MS:-1800000}" \
  --env WEBUI_SMOKE_PHASE_TIMEOUT_MS="${WEBUI_SMOKE_PHASE_TIMEOUT_MS:-2100000}" \
  --volume "$SCRIPT_DIR/webui-browser-smoke.mjs:/work/webui_browser_smoke.mjs:ro" \
  "$BROWSER_IMAGE" node /work/webui_browser_smoke.mjs

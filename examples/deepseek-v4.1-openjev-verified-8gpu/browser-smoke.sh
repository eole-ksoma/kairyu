#!/usr/bin/env bash
# Verify the live answer page in a real browser (playground-smoke.mjs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BROWSER_IMAGE="kairyu-webui-browser:playwright-1.60.0"

docker build \
  --file "$REPO_ROOT/deploy/compose/Dockerfile.webui-browser" \
  --tag "$BROWSER_IMAGE" \
  "$REPO_ROOT"

docker run --rm --init --network host \
  --env PLAYGROUND_SMOKE_BASE_URL="${PLAYGROUND_SMOKE_BASE_URL:-http://127.0.0.1:3013}" \
  --volume "$SCRIPT_DIR/playground-smoke.mjs:/work/playground_smoke.mjs:ro" \
  "$BROWSER_IMAGE" node /work/playground_smoke.mjs

#!/usr/bin/env bash
# Locally reproducible PostgreSQL integration suite.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POSTGRES_IMAGE=${KAIRYU_TEST_POSTGRES_IMAGE:-postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3}
CONTAINER="kairyu-postgres-test-$$"
PASSWORD="kairyu-test"
DATABASE="kairyu_test"
READY_ATTEMPTS=${KAIRYU_TEST_POSTGRES_READY_ATTEMPTS:-60}

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUN=(uv run --frozen)
elif command -v poetry >/dev/null 2>&1; then
  PYTHON_RUN=(poetry run)
else
  echo "PostgreSQL integration requires uv or poetry" >&2
  exit 2
fi

if [[ ! "$READY_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "KAIRYU_TEST_POSTGRES_READY_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

cleanup() {
  status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    docker logs "$CONTAINER" >&2 || true
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

cd "$REPO_ROOT"
"${PYTHON_RUN[@]}" python scripts/test_prerequisites.py \
  --require-executable docker \
  --require-module psycopg

docker run --detach \
  --name "$CONTAINER" \
  --publish 127.0.0.1::5432 \
  --env "POSTGRES_PASSWORD=$PASSWORD" \
  --env "POSTGRES_DB=$DATABASE" \
  "$POSTGRES_IMAGE" >/dev/null

binding=$(docker port "$CONTAINER" 5432/tcp | head -n 1)
port=${binding##*:}
[[ "$port" =~ ^[1-9][0-9]*$ ]]
export KAIRYU_TEST_POSTGRES_DSN="postgresql://postgres:${PASSWORD}@127.0.0.1:${port}/${DATABASE}"

postgres_ready() {
  "${PYTHON_RUN[@]}" python -c \
    'import os, psycopg; connection = psycopg.connect(os.environ["KAIRYU_TEST_POSTGRES_DSN"], connect_timeout=2); assert connection.execute("SELECT 1").fetchone() == (1,); connection.close()' \
    >/dev/null 2>&1
}

# The official image briefly runs an initialization-only Unix-socket server
# before stopping it and starting the final TCP server. In-container
# pg_isready can observe that temporary process and race the shutdown. Probe
# the exact published DSN that the tests will use instead.
ready=false
for _ in $(seq 1 "$READY_ATTEMPTS"); do
  if postgres_ready; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "PostgreSQL did not accept external TCP queries after $READY_ATTEMPTS attempts" >&2
  exit 1
fi

"${PYTHON_RUN[@]}" pytest --fail-on-skip \
  -m postgres \
  tests/unit/test_postgres_batch_store.py \
  tests/unit/test_postgres_runner_leadership.py \
  tests/unit/test_postgres_runner_scaling_log.py \
  tests/unit/test_postgres_request_store.py \
  -v --no-cov

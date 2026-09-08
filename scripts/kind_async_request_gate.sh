#!/usr/bin/env bash
# AsyncRequest staged CPU gate: proven F1c foundation plus durable request smoke.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/kind_async_request_gate.sh [--keep-cluster]

Environment overrides:
  KIND, KUBECTL, UV, GIT, TIMEOUT
  F1C_CLUSTER_NAME, F1C_RESULTS_DIR, ASYNC_REQUEST_RESULTS_DIR

Runs the existing binding F1c three-gateway gate first, then verifies shared
AsyncRequest state, cancellation, deadlines, owner takeover, and DB reconnect.
This CPU gate intentionally excludes GPU performance and soak testing.
EOF
}

KEEP_CLUSTER=0
while (($#)); do
  case "$1" in
    --keep-cluster)
      KEEP_CLUSTER=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "$REPO_ROOT"

KIND=${KIND:-kind}
KUBECTL=${KUBECTL:-kubectl}
UV=${UV:-uv}
GIT=${GIT:-git}
TIMEOUT=${TIMEOUT:-timeout}
CLUSTER_NAME=${F1C_CLUSTER_NAME:-kairyu-f1c}
F1C_RESULTS_DIR=${F1C_RESULTS_DIR:-bench/results/f1c-three-gateway-live}
ASYNC_REQUEST_RESULTS_DIR=${ASYNC_REQUEST_RESULTS_DIR:-bench/results/async-request-kind-live}
REPORT=${ASYNC_REQUEST_RESULTS_DIR}/report.json
GATE_LOCK_DIR=/tmp/kairyu-f1c-kind-gate.lock

for tool in "$KIND" "$KUBECTL" "$UV" "$GIT" "$TIMEOUT"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "required tool is unavailable: $tool" >&2
    exit 1
  fi
done
if [[ ! "$CLUSTER_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "F1C_CLUSTER_NAME must be a valid DNS label" >&2
  exit 1
fi

SOURCE_COMMIT=$("$GIT" rev-parse HEAD)
assert_clean_source() {
  local current_commit tracked_status
  current_commit=$("$GIT" rev-parse HEAD)
  tracked_status=$("$GIT" status --porcelain --untracked-files=no)
  if [[ "$current_commit" != "$SOURCE_COMMIT" || -n "$tracked_status" ]]; then
    echo "AsyncRequest kind gate requires one unchanged clean source commit" >&2
    printf '%s\n' "$tracked_status" >&2
    exit 1
  fi
}
assert_clean_source

run_bounded() {
  local duration=$1
  shift
  "$TIMEOUT" --signal=TERM --kill-after=5s "$duration" "$@"
}

CLUSTER_OWNED=0
GATE_LOCK_OWNED=0
collect_evidence() {
  if ((CLUSTER_OWNED == 0)); then
    return
  fi
  run_bounded 20s "$KUBECTL" -n kairyu-f1c \
    get pods,deployments,statefulsets,endpointslices,services -o wide \
    >"${ASYNC_REQUEST_RESULTS_DIR}/kubernetes-final.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1c \
    get events --sort-by=.lastTimestamp \
    >"${ASYNC_REQUEST_RESULTS_DIR}/events.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1c \
    get pods,deployments,statefulsets,endpointslices,services -o yaml \
    >"${ASYNC_REQUEST_RESULTS_DIR}/live-state.yaml" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1c logs deployment/f1c-postgres \
    >"${ASYNC_REQUEST_RESULTS_DIR}/postgres.log" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1c logs deployment/f1c-lb \
    >"${ASYNC_REQUEST_RESULTS_DIR}/lb.log" 2>&1 || true
  local gateway_id
  for gateway_id in a b c; do
    run_bounded 20s "$KUBECTL" -n kairyu-f1c \
      logs "deployment/f1c-gateway-${gateway_id}" \
      >"${ASYNC_REQUEST_RESULTS_DIR}/gateway-${gateway_id}.log" 2>&1 || true
  done
  run_bounded 20s "$KUBECTL" -n kairyu-f1c \
    exec deployment/f1c-postgres -- psql \
    postgresql://kairyu:f1c-kind-only@127.0.0.1:5432/kairyu \
    -XAt -v ON_ERROR_STOP=1 -c \
    "SELECT row_to_json(a)::text FROM (SELECT sequence, request_id, worker_id, fencing_token, at, event, lease_until, details FROM async_request_claim_audit WHERE store_id = 'kairyu-f1c-async' ORDER BY sequence) AS a" \
    >"${ASYNC_REQUEST_RESULTS_DIR}/claim-audit.jsonl" 2>&1 || true
}

cleanup() {
  local status=$?
  local cleanup_failed=0
  trap - EXIT
  collect_evidence
  if ((CLUSTER_OWNED == 1 && KEEP_CLUSTER == 0)); then
    if ! run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"; then
      echo "failed to delete kind cluster ${CLUSTER_NAME}" >&2
      cleanup_failed=1
    fi
  fi
  if ((GATE_LOCK_OWNED == 1)); then
    if ! rmdir -- "$GATE_LOCK_DIR"; then
      echo "failed to release F1c gate lock ${GATE_LOCK_DIR}" >&2
      cleanup_failed=1
    fi
  fi
  if ((status == 0 && cleanup_failed != 0)); then
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

# Serialize the cluster name and both result directories. The nested F1c gate
# recognizes this inherited lock, so direct and composed executions cannot race.
if ! mkdir -- "$GATE_LOCK_DIR"; then
  echo "another F1c/AsyncRequest kind gate owns ${GATE_LOCK_DIR}" >&2
  exit 1
fi
GATE_LOCK_OWNED=1

# Refuse an existing cluster so cleanup can never remove a cluster this wrapper
# did not create. The nested F1c gate otherwise treats its name as disposable.
while IFS= read -r existing; do
  if [[ "$existing" == "$CLUSTER_NAME" ]]; then
    echo "kind cluster ${CLUSTER_NAME} already exists; choose another name" >&2
    exit 1
  fi
done < <("$KIND" get clusters)

mkdir -p "$ASYNC_REQUEST_RESULTS_DIR"
known_results=(
  report.json
  kubernetes-final.txt
  events.txt
  live-state.yaml
  postgres.log
  lb.log
  gateway-a.log
  gateway-b.log
  gateway-c.log
  claim-audit.jsonl
)
for result_name in "${known_results[@]}"; do
  rm -f -- "${ASYNC_REQUEST_RESULTS_DIR}/${result_name}"
done

# The global lock and the refuse-existing mode make this cluster name ours for
# the complete nested invocation, including its failure paths.
CLUSTER_OWNED=1
F1C_CLUSTER_NAME="$CLUSTER_NAME" \
F1C_RESULTS_DIR="$F1C_RESULTS_DIR" \
F1C_REFUSE_EXISTING_CLUSTER=1 \
F1C_GATE_LOCK_HELD=1 \
  "$SCRIPT_DIR/kind_gateway_gate.sh" --keep-cluster
CLUSTER_OWNED=1

assert_clean_source
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/kairyu-async-request-uv-cache}
"$UV" run --frozen python \
  verification/fleet/resilience/async_request_gateway_smoke.py \
  --gateway-url http://127.0.0.1:18082 \
  --kubectl "$KUBECTL" \
  --namespace kairyu-f1c \
  --timeout-seconds 90 \
  --source-commit "$SOURCE_COMMIT" \
  --output "$REPORT"
assert_clean_source

echo "AsyncRequest staged CPU gate passed: ${REPORT}"

#!/usr/bin/env bash
# G5 F1c: clean-head build -> one-node kind -> three-gateway shared-store gate.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/kind_gateway_gate.sh [--keep-cluster]

Environment overrides:
  KIND, KUBECTL, DOCKER, UV, GIT, TAR, TIMEOUT, CURL
  F1C_CLUSTER_NAME, F1C_RESULTS_DIR, F1C_REFUSE_EXISTING_CLUSTER

This is the one binding CPU-only F1c integration test. It intentionally does not rerun
F1a churn, F1b rollout, a GPU workload, or a latency threshold.
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
DOCKER=${DOCKER:-docker}
UV=${UV:-uv}
GIT=${GIT:-git}
TAR=${TAR:-tar}
TIMEOUT=${TIMEOUT:-timeout}
CURL=${CURL:-curl}

CLUSTER_NAME=${F1C_CLUSTER_NAME:-kairyu-f1c}
RESULTS_DIR=${F1C_RESULTS_DIR:-bench/results/f1c-three-gateway-live}
KIND_CONFIG=deploy/kind/f1c/kind-config.yaml
MANIFEST_DIR=deploy/kind/f1c
GATEWAY_URL=http://127.0.0.1:18082
NAMESPACE=kairyu-f1c
REPLICA_COUNT=12
REFUSE_EXISTING_CLUSTER=${F1C_REFUSE_EXISTING_CLUSTER:-0}
GATE_LOCK_HELD=${F1C_GATE_LOCK_HELD:-0}
GATE_LOCK_DIR=/tmp/kairyu-f1c-kind-gate.lock

GATEWAY_IMAGE=kairyu:dev
MOCK_IMAGE=kairyu-f1a-mock:dev
POSTGRES_IMAGE=postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3
# Classic Docker cannot hand a tag@digest reference directly to
# ``kind load docker-image`` even though the pinned pull succeeded. Load and
# inspect the same manifest through its local tag; the Pod spec and verifier
# still bind the immutable digest above.
POSTGRES_LOAD_IMAGE=postgres:17.6-bookworm
GATEWAY_CRI_IMAGE_METADATA="${RESULTS_DIR}/gateway-cri-image.json"
MOCK_CRI_IMAGE_METADATA="${RESULTS_DIR}/mock-cri-image.json"
POSTGRES_CRI_IMAGE_METADATA="${RESULTS_DIR}/postgres-cri-image.json"
POSTGRES_DOCKER_IMAGE_METADATA="${RESULTS_DIR}/postgres-docker-image.json"

run_bounded() {
  local duration=$1
  shift
  "$TIMEOUT" --signal=TERM --kill-after=5s "$duration" "$@"
}

for tool in \
  "$KIND" "$KUBECTL" "$DOCKER" "$UV" "$GIT" "$TAR" "$TIMEOUT" "$CURL"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "required tool is unavailable: $tool" >&2
    exit 1
  fi
done
if [[ ! "$CLUSTER_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "F1C_CLUSTER_NAME must be a valid DNS label" >&2
  exit 1
fi
if [[ "$REFUSE_EXISTING_CLUSTER" != 0 && "$REFUSE_EXISTING_CLUSTER" != 1 ]]; then
  echo "F1C_REFUSE_EXISTING_CLUSTER must be 0 or 1" >&2
  exit 1
fi
if [[ "$GATE_LOCK_HELD" != 0 && "$GATE_LOCK_HELD" != 1 ]]; then
  echo "F1C_GATE_LOCK_HELD must be 0 or 1" >&2
  exit 1
fi
if [[ ! -f "$KIND_CONFIG" || ! -f "${MANIFEST_DIR}/kustomization.yaml" ]]; then
  echo "committed F1c kind inputs are incomplete" >&2
  exit 1
fi

CLUSTER_CREATED=0
CLUSTER_MAY_EXIST=0
SOURCE_ARCHIVE_DIR=
GATE_LOCK_OWNED=0

collect_evidence() {
  if ((CLUSTER_CREATED == 0)); then
    return
  fi
  run_bounded 20s "$KUBECTL" -n "$NAMESPACE" \
    get pods,deployments,statefulsets,endpointslices,services -o wide \
    >"${RESULTS_DIR}/kubernetes-final.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n "$NAMESPACE" \
    get events --sort-by=.lastTimestamp \
    >"${RESULTS_DIR}/events.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n "$NAMESPACE" get \
    serviceaccounts,roles,rolebindings,configmaps,secrets,services,statefulsets,deployments \
    -o yaml >"${RESULTS_DIR}/live-applied.yaml" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n "$NAMESPACE" get pods \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.uid}{"\t"}{range .spec.containers[*]}{.name}{"="}{.image}{"\t"}{end}{range .status.containerStatuses[*]}{.name}{"="}{.imageID}{"\t"}{end}{"\n"}{end}' \
    >"${RESULTS_DIR}/runtime-images.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n "$NAMESPACE" logs deployment/f1c-postgres \
    >"${RESULTS_DIR}/postgres.log" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n "$NAMESPACE" logs deployment/f1c-lb \
    >"${RESULTS_DIR}/lb.log" 2>&1 || true
  local gateway_id
  for gateway_id in a b c; do
    run_bounded 20s "$KUBECTL" -n "$NAMESPACE" \
      logs "deployment/f1c-gateway-${gateway_id}" \
      >"${RESULTS_DIR}/gateway-${gateway_id}.log" 2>&1 || true
  done
}

cleanup() {
  local status=$?
  local cleanup_failed=0
  set +e
  collect_evidence
  if ((CLUSTER_MAY_EXIST == 1 && (CLUSTER_CREATED == 0 || KEEP_CLUSTER == 0))); then
    if ! run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"; then
      echo "failed to delete kind cluster ${CLUSTER_NAME}" >&2
      cleanup_failed=1
    fi
  fi
  if [[ -n "$SOURCE_ARCHIVE_DIR" && -d "$SOURCE_ARCHIVE_DIR" ]]; then
    case "$SOURCE_ARCHIVE_DIR" in
      /tmp/kairyu-f1c-source.*)
        if ! rm -rf -- "$SOURCE_ARCHIVE_DIR"; then
          echo "failed to remove source archive ${SOURCE_ARCHIVE_DIR}" >&2
          cleanup_failed=1
        fi
        ;;
      *)
        echo "refusing to remove unexpected source path ${SOURCE_ARCHIVE_DIR}" >&2
        cleanup_failed=1
        ;;
      esac
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

if ((GATE_LOCK_HELD == 0)); then
  if ! mkdir -- "$GATE_LOCK_DIR"; then
    echo "another F1c/AsyncRequest kind gate owns ${GATE_LOCK_DIR}" >&2
    exit 1
  fi
  GATE_LOCK_OWNED=1
elif [[ ! -d "$GATE_LOCK_DIR" ]]; then
  echo "external F1c gate lock is not present: ${GATE_LOCK_DIR}" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
known_results=(
  manifest.json
  verifier.json
  traffic.jsonl
  gateway-pods.jsonl
  replica-pods.jsonl
  postgres-pods.jsonl
  gateway-cri-image.json
  mock-cri-image.json
  postgres-cri-image.json
  postgres-docker-image.json
  store-identities.jsonl
  batch-http.jsonl
  batch-lifecycle.jsonl
  batch-claims.jsonl
  lb-decisions.jsonl
  placements-a.jsonl
  placements-b.jsonl
  placements-c.jsonl
  batch-output-a.jsonl
  batch-output-b.jsonl
  batch-output-c.jsonl
  rendered-manifest.yaml
  kubernetes-final.txt
  live-applied.yaml
  events.txt
  runtime-images.txt
  postgres.log
  lb.log
  gateway-a.log
  gateway-b.log
  gateway-c.log
  failure.txt
)
for result_name in "${known_results[@]}"; do
  rm -f -- "${RESULTS_DIR}/${result_name}"
done

SOURCE_COMMIT=$("$GIT" rev-parse HEAD)
source_inputs=(
  Dockerfile
  README.md
  pyproject.toml
  uv.lock
  kairyu
  verification/fleet/resilience/fleet_gateway_bench.py
  scripts/kind_gateway_gate.sh
  deploy/kind/f1a/mock
  deploy/kind/f1c
  .github/workflows/f1c-gateway.yml
)

assert_clean_source() {
  local current_commit tracked_status scoped_status
  current_commit=$("$GIT" rev-parse HEAD)
  tracked_status=$("$GIT" status --porcelain --untracked-files=no)
  scoped_status=$(
    "$GIT" status --porcelain --untracked-files=all -- "${source_inputs[@]}"
  )
  if [[ "$current_commit" != "$SOURCE_COMMIT" ||
        -n "$tracked_status" ||
        -n "$scoped_status" ]]; then
    echo "F1c gate requires one unchanged clean source commit" >&2
    printf '%s\n' "$tracked_status" "$scoped_status" >&2
    exit 1
  fi
}
assert_clean_source

SOURCE_ARCHIVE_DIR=$(mktemp -d /tmp/kairyu-f1c-source.XXXXXX)
"$GIT" archive --format=tar "$SOURCE_COMMIT" |
  "$TAR" -xf - -C "$SOURCE_ARCHIVE_DIR"
ARCHIVE_KIND_CONFIG="${SOURCE_ARCHIVE_DIR}/${KIND_CONFIG}"
ARCHIVE_MANIFEST_DIR="${SOURCE_ARCHIVE_DIR}/${MANIFEST_DIR}"
ARCHIVE_MOCK_DIR="${SOURCE_ARCHIVE_DIR}/deploy/kind/f1a/mock"
ARCHIVE_DRIVER="${SOURCE_ARCHIVE_DIR}/verification/fleet/resilience/fleet_gateway_bench.py"
if [[ ! -f "$ARCHIVE_KIND_CONFIG" ||
      ! -f "${ARCHIVE_MANIFEST_DIR}/kustomization.yaml" ||
      ! -f "${ARCHIVE_MOCK_DIR}/Dockerfile" ||
      ! -f "$ARCHIVE_DRIVER" ]]; then
  echo "clean source archive is missing F1c inputs" >&2
  exit 1
fi

"$KUBECTL" kustomize "$ARCHIVE_MANIFEST_DIR" \
  >"${RESULTS_DIR}/rendered-manifest.yaml"

"$DOCKER" build --ulimit nofile=65536:65536 \
  --provenance=false \
  --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" \
  -t "$GATEWAY_IMAGE" "$SOURCE_ARCHIVE_DIR"
"$DOCKER" build --ulimit nofile=65536:65536 \
  --provenance=false \
  --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" \
  -t "$MOCK_IMAGE" "$ARCHIVE_MOCK_DIR"
"$DOCKER" pull "$POSTGRES_IMAGE"
"$DOCKER" image tag "$POSTGRES_IMAGE" "$POSTGRES_LOAD_IMAGE"
"$DOCKER" image inspect "$POSTGRES_IMAGE" >"$POSTGRES_DOCKER_IMAGE_METADATA"
assert_clean_source

GATEWAY_IMAGE_DIGEST=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$GATEWAY_IMAGE"
)
MOCK_IMAGE_DIGEST=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$MOCK_IMAGE"
)
POSTGRES_IMAGE_CONFIG_ID=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$POSTGRES_IMAGE"
)
for image in "$GATEWAY_IMAGE" "$MOCK_IMAGE"; do
  image_revision=$(
    "$DOCKER" image inspect \
      --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
      "$image"
  )
  if [[ "$image_revision" != "$SOURCE_COMMIT" ]]; then
    echo "image ${image} revision ${image_revision} != ${SOURCE_COMMIT}" >&2
    exit 1
  fi
done
for digest in \
  "$GATEWAY_IMAGE_DIGEST" "$MOCK_IMAGE_DIGEST" "$POSTGRES_IMAGE_CONFIG_ID"; do
  if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "invalid runtime image digest: ${digest}" >&2
    exit 1
  fi
done

while IFS= read -r existing; do
  if [[ "$existing" == "$CLUSTER_NAME" ]]; then
    if ((REFUSE_EXISTING_CLUSTER == 1)); then
      echo "kind cluster ${CLUSTER_NAME} already exists" >&2
      exit 1
    fi
    CLUSTER_MAY_EXIST=1
    run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"
    CLUSTER_MAY_EXIST=0
    break
  fi
done < <("$KIND" get clusters)

CLUSTER_MAY_EXIST=1
"$KIND" create cluster \
  --name "$CLUSTER_NAME" \
  --config "$ARCHIVE_KIND_CONFIG" \
  --wait 180s
CLUSTER_CREATED=1
"$KIND" load docker-image \
  "$GATEWAY_IMAGE" "$MOCK_IMAGE" "$POSTGRES_LOAD_IMAGE" \
  --name "$CLUSTER_NAME"
kind_nodes=()
while IFS= read -r kind_node; do
  kind_nodes+=("$kind_node")
done < <("$KIND" get nodes --name "$CLUSTER_NAME")
if ((${#kind_nodes[@]} != 1)); then
  echo "expected one F1c kind node, found ${#kind_nodes[@]}" >&2
  exit 1
fi
CONTROL_PLANE=${kind_nodes[0]}
run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  crictl inspecti --output json "$GATEWAY_IMAGE" \
  >"$GATEWAY_CRI_IMAGE_METADATA"
run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  crictl inspecti --output json "$MOCK_IMAGE" \
  >"$MOCK_CRI_IMAGE_METADATA"
run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  crictl inspecti --output json "$POSTGRES_LOAD_IMAGE" \
  >"$POSTGRES_CRI_IMAGE_METADATA"
cri_status_digest() {
  local image_reference=$1
  run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
    crictl inspecti --output go-template --template '{{.status.id}}' \
    "$image_reference"
}
GATEWAY_CRI_STATUS_DIGEST=$(cri_status_digest "$GATEWAY_IMAGE")
MOCK_CRI_STATUS_DIGEST=$(cri_status_digest "$MOCK_IMAGE")
POSTGRES_CRI_STATUS_DIGEST=$(cri_status_digest "$POSTGRES_LOAD_IMAGE")
if [[ "$GATEWAY_CRI_STATUS_DIGEST" != "$GATEWAY_IMAGE_DIGEST" ||
      "$MOCK_CRI_STATUS_DIGEST" != "$MOCK_IMAGE_DIGEST" ||
      "$POSTGRES_CRI_STATUS_DIGEST" != "$POSTGRES_IMAGE_CONFIG_ID" ]]; then
  echo "kind CRI config ID does not match the clean-source Docker image" >&2
  exit 1
fi

# Apply the exact rendered bytes whose hash the offline verifier later checks.
"$KUBECTL" apply -f "${RESULTS_DIR}/rendered-manifest.yaml"
run_bounded 180s "$KUBECTL" -n "$NAMESPACE" rollout status \
  deployment/f1c-postgres --timeout=170s
run_bounded 180s "$KUBECTL" -n "$NAMESPACE" wait \
  --for=jsonpath='{.status.readyReplicas}'="$REPLICA_COUNT" \
  statefulset/f1c-replica --timeout=170s
for deployment in f1c-gateway-a f1c-gateway-b f1c-gateway-c f1c-lb; do
  run_bounded 180s "$KUBECTL" -n "$NAMESPACE" rollout status \
    "deployment/${deployment}" --timeout=170s
done

lb_ready=0
for _ in $(seq 1 120); do
  if "$CURL" -sf "${GATEWAY_URL}/readyz" >/dev/null; then
    lb_ready=1
    break
  fi
  sleep 1
done
if ((lb_ready == 0)); then
  echo "F1c load balancer did not become ready at ${GATEWAY_URL}" >&2
  exit 1
fi

membership_ready=0
for _ in $(seq 1 180); do
  if run_bounded 20s "$KUBECTL" -n "$NAMESPACE" exec deployment/f1c-lb -- \
    python -c '
import sys
from urllib.request import urlopen

expected = int(sys.argv[1])
for gateway in ("a", "b", "c"):
    text = urlopen(
        f"http://f1c-gateway-{gateway}:8000/metrics", timeout=3
    ).read().decode()
    wanted = (
        "kairyu_pool_replicas"
        "{pool=\"f1a\",state=\"healthy\"} "
        f"{expected}.0"
    )
    if wanted not in text:
        raise SystemExit(f"{gateway}: missing {wanted}")
' "$REPLICA_COUNT" >/dev/null; then
    membership_ready=1
    break
  fi
  sleep 1
done
if ((membership_ready == 0)); then
  echo "three gateway replica rings did not converge to ${REPLICA_COUNT}" >&2
  exit 1
fi

export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/kairyu-f1c-uv-cache}
assert_clean_source
if ! run_bounded 600s "$UV" run --frozen python "$ARCHIVE_DRIVER" \
  --gateway-url "$GATEWAY_URL" \
  --kubectl "$KUBECTL" \
  --namespace "$NAMESPACE" \
  --replica-count "$REPLICA_COUNT" \
  --batch-lines 200 \
  --timeout-seconds 360 \
  --results-dir "$RESULTS_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --source-clean \
  --gateway-image-digest "$GATEWAY_IMAGE_DIGEST" \
  --mock-image-digest "$MOCK_IMAGE_DIGEST" \
  --postgres-image "$POSTGRES_IMAGE" \
  --rendered-manifest "${RESULTS_DIR}/rendered-manifest.yaml" \
  --gateway-cri-image-metadata "$GATEWAY_CRI_IMAGE_METADATA" \
  --mock-cri-image-metadata "$MOCK_CRI_IMAGE_METADATA" \
  --postgres-cri-image-metadata "$POSTGRES_CRI_IMAGE_METADATA" \
  --postgres-docker-image-metadata "$POSTGRES_DOCKER_IMAGE_METADATA"; then
  printf '%s\n' "live F1c driver failed" >"${RESULTS_DIR}/failure.txt"
  exit 1
fi

# A second measurement is waste. Replay only the already-written raw sidecars.
"$UV" run --frozen python "$ARCHIVE_DRIVER" \
  --results-dir "$RESULTS_DIR" \
  --verify-only
assert_clean_source

echo "F1c three-gateway shared BatchStore gate passed: ${RESULTS_DIR}/manifest.json"

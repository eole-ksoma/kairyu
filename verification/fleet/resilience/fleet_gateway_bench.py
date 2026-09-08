#!/usr/bin/env python3
"""F1c three-gateway affinity and shared BatchStore gate (issue #177).

The binding run is intentionally small and CPU-only.  It proves the new
responsibility boundary rather than repeating F1a's 200-replica churn or F1b's
100-step rollout:

* an auditable L7 rendezvous-hash load balancer spreads sessions over three
  independently identifiable gateways;
* every gateway independently chooses the same immutable replica Pod UID for a
  session from the same EndpointSlice membership;
* PostgreSQL-backed files and batches are visible through any gateway;
* after the active batch claimant Pod UID is killed, a different gateway
  reclaims the expired lease with a higher fencing token and is the only writer
  allowed to publish the terminal result.

Raw JSONL and output bytes are the authority.  ``verify_evidence`` reconstructs
all binding checks from those sidecars and validates their SHA-256 digests; it
does not trust booleans written by the live driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx

SCHEMA_VERSION = 1
GATEWAY_IDS = ("a", "b", "c")
DEFAULT_NAMESPACE = "kairyu-f1c"
DEFAULT_REPLICA_COUNT = 12
DEFAULT_BATCH_LINES = 200
DEFAULT_TIMEOUT_SECONDS = 360.0
BASELINE_ROUNDS = 2
SESSIONS_PER_GATEWAY = 2

_MOCK_IDENTITY = re.compile(
    r"\bkairyu-f1a-mock pod_name=(?P<pod_name>\S+) "
    r"pod_uid=(?P<pod_uid>\S+)"
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
_RUNTIME_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CROSS_PROCESS_CLOCK_SKEW_NS = 5_000_000

RAW_ARTIFACTS = (
    "traffic.jsonl",
    "gateway-pods.jsonl",
    "replica-pods.jsonl",
    "postgres-pods.jsonl",
    "store-identities.jsonl",
    "batch-http.jsonl",
    "batch-lifecycle.jsonl",
    "batch-claims.jsonl",
    "lb-decisions.jsonl",
    "placements-a.jsonl",
    "placements-b.jsonl",
    "placements-c.jsonl",
    "batch-output-a.jsonl",
    "batch-output-b.jsonl",
    "batch-output-c.jsonl",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wall_clock_envelope_contains(
    outer_started_ns: object,
    inner_started_ns: object,
    inner_finished_ns: object,
    outer_finished_ns: object,
    *,
    tolerance_ns: int = _CROSS_PROCESS_CLOCK_SKEW_NS,
) -> bool:
    """Compare host/container wall clocks with one explicit narrow tolerance."""
    values = (
        outer_started_ns,
        inner_started_ns,
        inner_finished_ns,
        outer_finished_ns,
    )
    return (
        type(tolerance_ns) is int
        and tolerance_ns >= 0
        and all(type(value) is int for value in values)
        and outer_started_ns <= outer_finished_ns
        and inner_started_ns <= inner_finished_ns
        and outer_started_ns - tolerance_ns <= inner_started_ns
        and inner_finished_ns <= outer_finished_ns + tolerance_ns
    )


def _safe_artifact_path(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"unsafe artifact path: {name!r}")
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError(f"artifact escapes result directory: {name!r}")
    return path


def _runtime_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    matches = _RUNTIME_DIGEST.findall(value)
    return matches[-1] if matches else None


def _normalized_image_reference(value: str) -> str:
    for prefix in ("docker.io/library/", "docker.io/", "library/"):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def _normalized_image_repository(value: str) -> str:
    name, separator, _digest = value.rpartition("@")
    if not separator:
        name = value
    normalized = _normalized_image_reference(name)
    prefix, slash, leaf = normalized.rpartition("/")
    repository = leaf.rpartition(":")[0] if ":" in leaf else leaf
    return f"{prefix}/{repository}" if slash else repository


def _container_image_matches(
    pod: Mapping[str, Any],
    *,
    container_name: str,
    spec_image: str,
    runtime_digests: set[str],
) -> bool:
    images = pod.get("images")
    if not isinstance(images, list):
        return False
    matches = [
        image
        for image in images
        if isinstance(image, dict) and image.get("name") == container_name
    ]
    observed_image = matches[0].get("image") if len(matches) == 1 else None
    if not isinstance(observed_image, str):
        return False
    observed = _runtime_digest(matches[0].get("image_id"))
    pinned_name, separator, pinned_digest = spec_image.rpartition("@")
    if separator and _RUNTIME_DIGEST.fullmatch(pinned_digest) is not None:
        # Kubernetes may report only the display tag in status.image even when
        # the Pod spec was digest-pinned. status.imageID remains the runtime
        # identity, so require it to equal the source pin exactly.
        allowed_images = {
            _normalized_image_reference(spec_image),
            _normalized_image_reference(pinned_name),
        }
        return (
            _normalized_image_reference(observed_image) in allowed_images
            and observed is not None
            and observed in runtime_digests
        )
    if _normalized_image_reference(observed_image) != _normalized_image_reference(
        spec_image
    ):
        return False
    return observed is not None and observed in runtime_digests


def _cri_image_metadata(path: Path, artifact_dir: Path) -> dict[str, Any]:
    path = path.resolve()
    artifact_dir = artifact_dir.resolve()
    if path.parent != artifact_dir:
        raise ValueError(f"CRI metadata must be beside the manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status") if isinstance(payload, dict) else None
    info = payload.get("info") if isinstance(payload, dict) else None
    image_spec = info.get("imageSpec") if isinstance(info, dict) else None
    config = image_spec.get("config") if isinstance(image_spec, dict) else None
    labels = config.get("Labels") if isinstance(config, dict) else None
    if labels is None:
        labels = {}
    if not isinstance(status, dict) or not isinstance(labels, dict):
        raise ValueError(f"incomplete CRI image metadata: {path}")

    status_id = status.get("id")
    repo_digests = status.get("repoDigests")
    repo_tags = status.get("repoTags")
    if (
        not isinstance(status_id, str)
        or _RUNTIME_DIGEST.fullmatch(status_id) is None
        or not isinstance(repo_digests, list)
        or not all(isinstance(value, str) for value in repo_digests)
        or not isinstance(repo_tags, list)
        or not repo_tags
        or not all(isinstance(value, str) and value for value in repo_tags)
    ):
        raise ValueError(f"invalid CRI image identity: {path}")
    runtime_digests = {status_id}
    for value in repo_digests:
        _name, separator, digest = value.rpartition("@")
        if not separator or _RUNTIME_DIGEST.fullmatch(digest) is None:
            raise ValueError(f"invalid CRI repo digest in {path}: {value!r}")
        runtime_digests.add(digest)
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "status_id": status_id,
        "repo_digests": sorted(repo_digests),
        "repo_tags": sorted(repo_tags),
        "runtime_digests": sorted(runtime_digests),
        "revision": labels.get("org.opencontainers.image.revision"),
    }


def _docker_image_metadata(path: Path, artifact_dir: Path) -> dict[str, Any]:
    path = path.resolve()
    artifact_dir = artifact_dir.resolve()
    if path.parent != artifact_dir:
        raise ValueError(f"Docker metadata must be beside the manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    image = payload[0] if isinstance(payload, list) and len(payload) == 1 else None
    if not isinstance(image, dict):
        raise ValueError(f"expected one Docker image record: {path}")

    config_id = image.get("Id")
    repo_digests = image.get("RepoDigests")
    repo_tags = image.get("RepoTags")
    if (
        not isinstance(config_id, str)
        or _RUNTIME_DIGEST.fullmatch(config_id) is None
        or not isinstance(repo_digests, list)
        or not repo_digests
        or not all(isinstance(value, str) for value in repo_digests)
        or not isinstance(repo_tags, list)
        or not repo_tags
        or not all(isinstance(value, str) and value for value in repo_tags)
    ):
        raise ValueError(f"invalid Docker image identity: {path}")
    for value in repo_digests:
        name, separator, digest = value.rpartition("@")
        if (
            not separator
            or not name
            or _RUNTIME_DIGEST.fullmatch(digest) is None
        ):
            raise ValueError(f"invalid Docker repo digest in {path}: {value!r}")
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "config_id": config_id,
        "repo_digests": sorted(repo_digests),
        "repo_tags": sorted(repo_tags),
    }


def _adjacent_file_metadata(path: Path, artifact_dir: Path) -> dict[str, Any]:
    path = path.resolve()
    artifact_dir = artifact_dir.resolve()
    if path.parent != artifact_dir or not path.is_file():
        raise ValueError(f"source artifact must be beside the manifest: {path}")
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n"):
                raise ValueError(f"{path.name}:{line_number}: missing newline")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: row is not an object")
            rows.append(row)
    return rows


class JsonlRecorder:
    """Write raw evidence immediately so a failed drill still leaves a trace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self._handle.write(_canonical_json(dict(row)) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def rendezvous_order(key: str, candidates: Sequence[str]) -> tuple[str, ...]:
    """Return the stable highest-random-weight order used by the F1c LB."""

    if not key:
        raise ValueError("rendezvous key must not be empty")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("rendezvous candidates must be non-empty and unique")
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                hashlib.sha256(f"{key}:{candidate}".encode()).digest(),
                candidate,
            ),
            reverse=True,
        )
    )


def session_for_gateway(
    gateway_id: str,
    *,
    ordinal: int = 0,
    candidates: Sequence[str] = GATEWAY_IDS,
) -> str:
    """Find a deterministic fixture session whose primary is ``gateway_id``."""

    if gateway_id not in candidates or ordinal < 0:
        raise ValueError("invalid gateway target or ordinal")
    matched = 0
    for sequence in range(1_000_000):
        session = f"f1c-session-{sequence:06d}"
        if rendezvous_order(session, candidates)[0] != gateway_id:
            continue
        if matched == ordinal:
            return session
        matched += 1
    raise RuntimeError(f"failed to find a session for gateway {gateway_id}")


def replica_winner(session: str, replica_uids: Sequence[str]) -> str:
    return rendezvous_order(session, replica_uids)[0]


def _ready(pod: Mapping[str, Any]) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions", ())
        if isinstance(condition, dict)
    )


def _pod_identity(pod: Mapping[str, Any]) -> dict[str, Any]:
    metadata = pod.get("metadata") or {}
    status = pod.get("status") or {}
    labels = metadata.get("labels") or {}
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "gateway_id": labels.get("kairyu.ai/gateway-id"),
        "deployment": labels.get("kairyu.ai/deployment"),
        "ready": _ready(pod),
        "phase": status.get("phase"),
        "pod_ip": status.get("podIP"),
        "node": pod.get("spec", {}).get("nodeName"),
        "restart_count": sum(
            item.get("restartCount", 0)
            for item in status.get("containerStatuses", ())
            if isinstance(item, dict)
        ),
        "images": [
            {
                "name": item.get("name"),
                "image": item.get("image"),
                "image_id": item.get("imageID"),
            }
            for item in status.get("containerStatuses", ())
            if isinstance(item, dict)
        ],
    }


def _run(
    *command: str,
    timeout: float = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _kubectl(
    kubectl: str,
    namespace: str,
    *arguments: str,
    timeout: float = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        kubectl,
        "-n",
        namespace,
        *arguments,
        timeout=timeout,
        check=check,
    )


def _kubectl_json(
    kubectl: str,
    namespace: str,
    *arguments: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    completed = _kubectl(
        kubectl,
        namespace,
        *arguments,
        "-o",
        "json",
        timeout=timeout,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("kubectl JSON response is not an object")
    return value


def _pods(
    kubectl: str,
    namespace: str,
    selector: str,
) -> list[dict[str, Any]]:
    payload = _kubectl_json(kubectl, namespace, "get", "pods", "-l", selector)
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Pod list is missing items")
    return [_pod_identity(item) for item in items if isinstance(item, dict)]


def _gateway_map(
    kubectl: str,
    namespace: str,
    *,
    require_ready: bool,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for pod in _pods(
        kubectl,
        namespace,
        "app.kubernetes.io/component=gateway",
    ):
        gateway_id = pod.get("gateway_id")
        if (
            not isinstance(gateway_id, str)
            or gateway_id not in GATEWAY_IDS
            or gateway_id in result
        ):
            raise RuntimeError(f"invalid or duplicate gateway identity: {pod!r}")
        result[gateway_id] = pod
    if set(result) != set(GATEWAY_IDS):
        raise RuntimeError(f"expected gateways {GATEWAY_IDS}, got {sorted(result)}")
    if require_ready and not all(
        pod.get("ready") is True and pod.get("phase") == "Running"
        for pod in result.values()
    ):
        raise RuntimeError("not all three gateways are Ready")
    return result


def _wait_gateway_map(
    kubectl: str,
    namespace: str,
    *,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _gateway_map(kubectl, namespace, require_ready=True)
        except Exception as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError("three gateways did not become Ready") from last_error


def _replica_map(
    kubectl: str,
    namespace: str,
    *,
    expected_count: int,
) -> dict[str, dict[str, Any]]:
    values = _pods(
        kubectl,
        namespace,
        "app.kubernetes.io/name=f1c-replica",
    )
    result = {
        str(pod["uid"]): pod
        for pod in values
        if isinstance(pod.get("uid"), str) and pod.get("uid")
    }
    if len(values) != expected_count or len(result) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} unique replica Pods, got "
            f"{len(values)}/{len(result)}"
        )
    if not all(
        pod.get("ready") is True and pod.get("phase") == "Running"
        for pod in result.values()
    ):
        raise RuntimeError("not all replica Pods are Ready")
    return result


def _postgres_pod(
    kubectl: str,
    namespace: str,
) -> dict[str, Any]:
    pods = _pods(
        kubectl,
        namespace,
        "app.kubernetes.io/name=f1c-postgres",
    )
    if len(pods) != 1:
        raise RuntimeError(f"expected one PostgreSQL Pod, got {len(pods)}")
    pod = pods[0]
    if pod.get("ready") is not True or pod.get("phase") != "Running":
        raise RuntimeError("F1c PostgreSQL Pod is not Ready")
    return pod


def _chat_identity(payload: object) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    choices = payload.get("choices")
    content = (
        (choices[0].get("message") or {}).get("content")
        if isinstance(choices, list)
        and choices
        and isinstance(choices[0], dict)
        else None
    )
    match = _MOCK_IDENTITY.fullmatch(content) if isinstance(content, str) else None
    if match is None:
        return None, None
    return match.group("pod_name"), match.group("pod_uid")


def _http_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    session: str,
    operation: str,
    request_id: str | None = None,
    json_body: object | None = None,
    files: Mapping[str, Any] | None = None,
    data: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], httpx.Response]:
    request_id = request_id or f"f1c-{uuid.uuid4().hex}"
    started_ns = time.time_ns()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "http",
        "operation": operation,
        "request_id": request_id,
        "session": session,
        "session_sha256": _sha256_bytes(session.encode()),
        "method": method,
        "path": path,
        "request_json": json_body,
        "started_unix_ns": started_ns,
        "status_code": None,
        "transport_error": None,
        "gateway_id": None,
        "gateway_request_id": None,
        "lb_request_id": None,
        "response_json": None,
        "request_body_bytes": None,
        "request_body_sha256": None,
    }
    try:
        headers = {
            "X-Session-ID": session,
            "X-Request-ID": request_id,
        }
        request_kwargs: dict[str, Any]
        if json_body is not None:
            # Freeze one reconstructable JSON wire representation.  The
            # offline verifier can now join semantic request fields to the
            # LB's independently recorded body digest.
            headers["Content-Type"] = "application/json"
            request_kwargs = {
                "content": _canonical_json(json_body).encode(),
            }
        else:
            request_kwargs = {
                "files": files,
                "data": data,
            }
        request = client.build_request(
            method,
            path,
            headers=headers,
            **request_kwargs,
        )
        request_body = request.read()
        record["request_body_bytes"] = len(request_body)
        record["request_body_sha256"] = _sha256_bytes(request_body)
        response = client.send(request)
    except Exception as error:
        record["transport_error"] = type(error).__name__
        record["ended_unix_ns"] = time.time_ns()
        raise RuntimeError(f"{operation} transport failure: {error}") from error
    record["status_code"] = response.status_code
    record["gateway_id"] = response.headers.get("x-kairyu-gateway-id")
    record["gateway_request_id"] = response.headers.get("x-request-id")
    record["lb_request_id"] = response.headers.get("x-kairyu-lb-request-id")
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    record["response_json"] = payload
    record["response_sha256"] = _sha256_bytes(response.content)
    record["ended_unix_ns"] = time.time_ns()
    return record, response


def _affinity_request(
    client: httpx.Client,
    *,
    session: str,
    phase: str,
    sequence: int,
) -> dict[str, Any]:
    body = {
        "model": "f1a",
        "messages": [
            {
                "role": "user",
                "content": f"F1c affinity {phase} {sequence}",
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
    }
    record, response = _http_json(
        client,
        "POST",
        "/v1/chat/completions",
        session=session,
        operation=f"affinity_{phase}",
        request_id=f"f1c-affinity-{phase}-{sequence:04d}",
        json_body=body,
    )
    pod_name, pod_uid = _chat_identity(record["response_json"])
    record.update(
        {
            "kind": "affinity",
            "phase": phase,
            "sequence": sequence,
            "pod_name": pod_name,
            "pod_uid": pod_uid,
            "response_valid": (
                response.status_code == 200
                and pod_name is not None
                and pod_uid is not None
            ),
        }
    )
    if not record["response_valid"]:
        raise RuntimeError(f"invalid affinity response: {record!r}")
    return record


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _postgres_query(
    kubectl: str,
    namespace: str,
    query: str,
) -> list[str]:
    completed = _kubectl(
        kubectl,
        namespace,
        "exec",
        "deployment/f1c-postgres",
        "--",
        "psql",
        "postgresql://kairyu:f1c-kind-only@127.0.0.1:5432/kairyu",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-At",
        "-c",
        query,
        timeout=30.0,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _claim_rows(
    kubectl: str,
    namespace: str,
    batch_id: str,
) -> list[dict[str, Any]]:
    query = f"""
SELECT row_to_json(e)
FROM (
  SELECT sequence, batch_id, event, worker_id, fencing_token,
         (extract(epoch FROM at) * 1000000000)::bigint AS at_unix_ns,
         CASE
           WHEN lease_until IS NULL THEN NULL
           ELSE (extract(epoch FROM lease_until) * 1000000000)::bigint
         END AS lease_until_unix_ns,
         CASE
           WHEN previous_lease_until IS NULL THEN NULL
           ELSE (
             extract(epoch FROM previous_lease_until) * 1000000000
           )::bigint
         END AS previous_lease_until_unix_ns
  FROM kairyu_batch_claim_audit
  WHERE store_id = 'kairyu-f1c'
    AND batch_id = {_sql_literal(batch_id)}
  ORDER BY sequence
) AS e
""".strip()
    rows = []
    for raw in _postgres_query(kubectl, namespace, query):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("claim audit row is not an object")
        rows.append(value)
    return rows


def _postgres_now_ns(kubectl: str, namespace: str) -> int:
    rows = _postgres_query(
        kubectl,
        namespace,
        "SELECT (extract(epoch FROM clock_timestamp()) * 1000000000)::bigint",
    )
    if len(rows) != 1:
        raise RuntimeError("could not read one PostgreSQL clock value")
    try:
        return int(rows[0])
    except ValueError as error:
        raise RuntimeError(f"invalid PostgreSQL clock value: {rows[0]!r}") from error


def _claimed(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("event") in {"claimed", "reclaimed"}
        and isinstance(row.get("worker_id"), str)
        and type(row.get("fencing_token")) is int
    ]


def _wait_claim(
    kubectl: str,
    namespace: str,
    batch_id: str,
    *,
    timeout_seconds: float,
    after_token: int | None = None,
    excluded_worker: str | None = None,
    expected_event: str | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        latest = _claimed(_claim_rows(kubectl, namespace, batch_id))
        for row in latest:
            token = row["fencing_token"]
            worker = row["worker_id"]
            if (
                (after_token is None or token > after_token)
                and (excluded_worker is None or worker != excluded_worker)
                and (expected_event is None or row.get("event") == expected_event)
            ):
                return row
        time.sleep(0.2)
    raise TimeoutError(
        f"batch {batch_id} claim did not appear after token {after_token}; "
        f"last={latest!r}"
    )


def _copy_pod_file(
    kubectl: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
    destination: Path,
) -> None:
    completed = _kubectl(
        kubectl,
        namespace,
        "exec",
        pod_name,
        "--",
        "python",
        "-c",
        (
            "from pathlib import Path; import sys; "
            "path=Path(sys.argv[1]); "
            "sys.stdout.write(path.read_text() if path.exists() else '')"
        ),
        remote_path,
        timeout=30.0,
    )
    destination.write_text(completed.stdout, encoding="utf-8")


def _record_topology(
    recorder: JsonlRecorder,
    *,
    phase: str,
    gateways: Mapping[str, Mapping[str, Any]],
) -> None:
    for gateway_id in GATEWAY_IDS:
        pod = dict(gateways[gateway_id])
        pod_phase = pod.pop("phase", None)
        recorder.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "gateway_pod",
                "evidence_phase": phase,
                "pod_phase": pod_phase,
                **pod,
            }
        )


def _batch_line(index: int) -> dict[str, Any]:
    return {
        "custom_id": f"f1c-{index:06d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "f1a",
            "messages": [
                {
                    "role": "user",
                    "content": f"F1c shared batch line {index}",
                }
            ],
            "max_tokens": 1,
            "temperature": 0,
        },
    }


def _batch_payload(line_count: int) -> bytes:
    return b"".join(
        (_canonical_json(_batch_line(index)) + "\n").encode()
        for index in range(line_count)
    )


def _write_claims(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with JsonlRecorder(path) as recorder:
        for row in rows:
            recorder.write({"schema_version": SCHEMA_VERSION, **dict(row)})


def _wait_batch_completed(
    client: httpx.Client,
    recorder: JsonlRecorder,
    *,
    batch_id: str,
    session: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    sequence = 0
    while time.monotonic() < deadline:
        record, response = _http_json(
            client,
            "GET",
            f"/v1/batches/{batch_id}",
            session=session,
            operation="poll_batch_c",
            request_id=f"f1c-batch-poll-{sequence:05d}",
        )
        recorder.write(record)
        if response.status_code != 200:
            raise RuntimeError(f"batch poll failed: {record!r}")
        payload = record.get("response_json")
        status = payload.get("status") if isinstance(payload, dict) else None
        if status == "completed":
            return payload
        if status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"batch reached terminal failure state {status}")
        sequence += 1
        time.sleep(0.25)
    raise TimeoutError(f"batch {batch_id} did not complete")


def _gateway_for_worker(
    worker_id: str,
    gateways: Mapping[str, Mapping[str, Any]],
) -> str:
    matches = [
        gateway_id
        for gateway_id, pod in gateways.items()
        if pod.get("uid") == worker_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"claim worker {worker_id!r} does not identify one gateway Pod"
        )
    return matches[0]


def _store_identity_rows(
    kubectl: str,
    namespace: str,
    gateways: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    system_ids = _postgres_query(
        kubectl,
        namespace,
        "SELECT system_identifier::text FROM pg_control_system()",
    )
    if len(system_ids) != 1:
        raise RuntimeError("could not read one PostgreSQL system identifier")
    rows: list[dict[str, Any]] = []
    for gateway_id, pod in gateways.items():
        completed = _kubectl(
            kubectl,
            namespace,
            "exec",
            str(pod["name"]),
            "--",
            "python",
            "-c",
            (
                "import hashlib, os; "
                "print(hashlib.sha256("
                "os.environ['KAIRYU_BATCH_POSTGRES_DSN'].encode()).hexdigest())"
            ),
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "store_identity",
                "gateway_id": gateway_id,
                "gateway_uid": pod["uid"],
                "dsn_sha256": completed.stdout.strip(),
                "postgres_system_identifier": system_ids[0],
            }
        )
    return rows


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    for name in (*RAW_ARTIFACTS, "manifest.json", "verifier.json"):
        path = _safe_artifact_path(results, name)
        if path.exists():
            path.unlink()

    initial_gateways = _wait_gateway_map(
        args.kubectl,
        args.namespace,
        timeout_seconds=args.timeout_seconds,
    )
    initial_replicas = _replica_map(
        args.kubectl,
        args.namespace,
        expected_count=args.replica_count,
    )
    initial_postgres = _postgres_pod(args.kubectl, args.namespace)
    sessions = {
        gateway_id: [
            session_for_gateway(gateway_id, ordinal=ordinal)
            for ordinal in range(SESSIONS_PER_GATEWAY)
        ]
        for gateway_id in GATEWAY_IDS
    }

    with (
        JsonlRecorder(results / "traffic.jsonl") as traffic,
        JsonlRecorder(results / "gateway-pods.jsonl") as gateway_pods,
        JsonlRecorder(results / "replica-pods.jsonl") as replica_pods,
        JsonlRecorder(results / "postgres-pods.jsonl") as postgres_pods,
        JsonlRecorder(results / "store-identities.jsonl") as store_identities,
        JsonlRecorder(results / "batch-http.jsonl") as batch_http,
        JsonlRecorder(results / "batch-lifecycle.jsonl") as lifecycle,
        httpx.Client(
            base_url=args.gateway_url,
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
            trust_env=False,
        ) as client,
    ):
        _record_topology(
            gateway_pods,
            phase="initial",
            gateways=initial_gateways,
        )
        for pod in initial_replicas.values():
            identity = dict(pod)
            pod_phase = identity.pop("phase", None)
            replica_pods.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "replica_pod",
                    "evidence_phase": "initial",
                    "pod_phase": pod_phase,
                    **identity,
                }
            )
        postgres_identity = dict(initial_postgres)
        postgres_phase = postgres_identity.pop("phase", None)
        postgres_pods.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "postgres_pod",
                "evidence_phase": "initial",
                "pod_phase": postgres_phase,
                **postgres_identity,
            }
        )
        for row in _store_identity_rows(
            args.kubectl,
            args.namespace,
            initial_gateways,
        ):
            store_identities.write(row)

        sequence = 0
        baseline_by_session: dict[str, dict[str, Any]] = {}
        for _round in range(BASELINE_ROUNDS):
            for gateway_id in GATEWAY_IDS:
                for session in sessions[gateway_id]:
                    record = _affinity_request(
                        client,
                        session=session,
                        phase="baseline",
                        sequence=sequence,
                    )
                    traffic.write(record)
                    baseline_by_session.setdefault(session, record)
                    sequence += 1

        upload_session = sessions["a"][0]
        create_session = sessions["b"][0]
        poll_session = sessions["c"][0]
        input_bytes = _batch_payload(args.batch_lines)
        upload, response = _http_json(
            client,
            "POST",
            "/v1/files",
            session=upload_session,
            operation="upload_input_a",
            files={
                "file": (
                    "f1c-input.jsonl",
                    input_bytes,
                    "application/jsonl",
                )
            },
            data={"purpose": "batch"},
        )
        batch_http.write(upload)
        if response.status_code != 200 or not isinstance(upload["response_json"], dict):
            raise RuntimeError(f"batch upload failed: {upload!r}")
        input_file_id = upload["response_json"].get("id")
        if not isinstance(input_file_id, str):
            raise RuntimeError("batch upload response is missing file id")

        cross_read, response = _http_json(
            client,
            "GET",
            f"/v1/files/{input_file_id}",
            session=sessions["b"][1],
            operation="read_input_b",
        )
        batch_http.write(cross_read)
        if response.status_code != 200:
            raise RuntimeError(f"cross-gateway input read failed: {cross_read!r}")

        create, response = _http_json(
            client,
            "POST",
            "/v1/batches",
            session=create_session,
            operation="create_batch_b",
            json_body={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"gate": "f1c", "line_count": args.batch_lines},
            },
        )
        batch_http.write(create)
        if response.status_code != 200 or not isinstance(create["response_json"], dict):
            raise RuntimeError(f"batch create failed: {create!r}")
        batch_id = create["response_json"].get("id")
        if not isinstance(batch_id, str):
            raise RuntimeError("batch create response is missing id")
        lifecycle.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "created",
                "batch_id": batch_id,
                "input_file_id": input_file_id,
                "input_sha256": _sha256_bytes(input_bytes),
                "line_count": args.batch_lines,
            }
        )

        old_claim = _wait_claim(
            args.kubectl,
            args.namespace,
            batch_id,
            timeout_seconds=30.0,
            expected_event="claimed",
        )
        killed_gateway = _gateway_for_worker(
            old_claim["worker_id"],
            initial_gateways,
        )
        killed_deployment = f"f1c-gateway-{killed_gateway}"
        lifecycle.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "owner_claim",
                "batch_id": batch_id,
                "gateway_id": killed_gateway,
                **old_claim,
            }
        )

        # Use the same PostgreSQL clock domain as leases and claim-audit rows;
        # this makes the causal inequalities independent of host clock skew.
        kill_requested_at_ns = _postgres_now_ns(args.kubectl, args.namespace)
        lifecycle.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "owner_kill_requested",
                "batch_id": batch_id,
                "gateway_id": killed_gateway,
                "deployment": killed_deployment,
                "worker_id": old_claim["worker_id"],
                "fencing_token": old_claim["fencing_token"],
                "at_unix_ns": kill_requested_at_ns,
                "clock": "postgres_clock_timestamp",
            }
        )
        _kubectl(
            args.kubectl,
            args.namespace,
            "scale",
            f"deployment/{killed_deployment}",
            "--replicas=0",
        )
        old_worker = old_claim["worker_id"]
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            current = _pods(
                args.kubectl,
                args.namespace,
                f"kairyu.ai/gateway-id={killed_gateway}",
            )
            if all(pod.get("uid") != old_worker for pod in current):
                owner_absent_at_ns = _postgres_now_ns(
                    args.kubectl,
                    args.namespace,
                )
                break
            time.sleep(0.25)
        else:
            raise TimeoutError(f"claim owner Pod UID {old_worker} was not killed")
        lifecycle.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "owner_killed",
                "batch_id": batch_id,
                "gateway_id": killed_gateway,
                "deployment": killed_deployment,
                "worker_id": old_worker,
                "fencing_token": old_claim["fencing_token"],
                "at_unix_ns": owner_absent_at_ns,
                "clock": "postgres_clock_timestamp",
            }
        )

        failover_session = sessions[killed_gateway][1]
        failover = _affinity_request(
            client,
            session=failover_session,
            phase="failover",
            sequence=sequence,
        )
        traffic.write(failover)
        sequence += 1

        new_claim = _wait_claim(
            args.kubectl,
            args.namespace,
            batch_id,
            timeout_seconds=30.0,
            after_token=old_claim["fencing_token"],
            excluded_worker=old_worker,
            expected_event="reclaimed",
        )
        surviving_gateways = {
            gateway_id: pod
            for gateway_id, pod in initial_gateways.items()
            if gateway_id != killed_gateway
        }
        reclaimed_gateway = _gateway_for_worker(
            new_claim["worker_id"],
            surviving_gateways,
        )
        lifecycle.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "reclaimed",
                "batch_id": batch_id,
                "gateway_id": reclaimed_gateway,
                **new_claim,
            }
        )

        _kubectl(
            args.kubectl,
            args.namespace,
            "scale",
            f"deployment/{killed_deployment}",
            "--replicas=1",
        )
        final_gateways = _wait_gateway_map(
            args.kubectl,
            args.namespace,
            timeout_seconds=120.0,
        )
        _record_topology(
            gateway_pods,
            phase="final",
            gateways=final_gateways,
        )
        for row in _store_identity_rows(
            args.kubectl,
            args.namespace,
            final_gateways,
        ):
            store_identities.write({**row, "phase": "final"})

        recovered = _affinity_request(
            client,
            session=failover_session,
            phase="recovered",
            sequence=sequence,
        )
        traffic.write(recovered)

        completed = _wait_batch_completed(
            client,
            batch_http,
            batch_id=batch_id,
            session=poll_session,
            timeout_seconds=args.timeout_seconds,
        )
        lifecycle.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "completed",
                "batch_id": batch_id,
                "status": completed.get("status"),
                "output_file_id": completed.get("output_file_id"),
                "error_file_id": completed.get("error_file_id"),
                "request_counts": completed.get("request_counts"),
                "errors": completed.get("errors"),
                "at_unix_ns": _postgres_now_ns(
                    args.kubectl,
                    args.namespace,
                ),
                "clock": "postgres_clock_timestamp",
            }
        )
        output_file_id = completed.get("output_file_id")
        if not isinstance(output_file_id, str):
            raise RuntimeError("completed batch does not reference an output file")

        for gateway_id in GATEWAY_IDS:
            output_record, output_response = _http_json(
                client,
                "GET",
                f"/v1/files/{output_file_id}/content",
                session=sessions[gateway_id][0],
                operation=f"read_output_{gateway_id}",
            )
            output_record["content_sha256"] = _sha256_bytes(output_response.content)
            output_record["content_bytes"] = len(output_response.content)
            # Binary content is committed separately; do not make the JSON
            # sidecar depend on attempting to parse it as an HTTP JSON body.
            output_record["response_json"] = None
            batch_http.write(output_record)
            if output_response.status_code != 200:
                raise RuntimeError(
                    f"output read through gateway {gateway_id} failed"
                )
            (results / f"batch-output-{gateway_id}.jsonl").write_bytes(
                output_response.content
            )

        final_replicas = _replica_map(
            args.kubectl,
            args.namespace,
            expected_count=args.replica_count,
        )
        for pod in final_replicas.values():
            identity = dict(pod)
            pod_phase = identity.pop("phase", None)
            replica_pods.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "replica_pod",
                    "evidence_phase": "final",
                    "pod_phase": pod_phase,
                    **identity,
                }
            )
        final_postgres = _postgres_pod(args.kubectl, args.namespace)
        postgres_identity = dict(final_postgres)
        postgres_phase = postgres_identity.pop("phase", None)
        postgres_pods.write(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "postgres_pod",
                "evidence_phase": "final",
                "pod_phase": postgres_phase,
                **postgres_identity,
            }
        )

    # Copy the durable hostPath-backed logs after the restarted gateway has
    # flushed its recovered request.
    time.sleep(1.0)
    final_gateways = _gateway_map(
        args.kubectl,
        args.namespace,
        require_ready=True,
    )
    for gateway_id, pod in final_gateways.items():
        _copy_pod_file(
            args.kubectl,
            args.namespace,
            str(pod["name"]),
            "/evidence/placements.jsonl",
            results / f"placements-{gateway_id}.jsonl",
        )
    lb_pods = _pods(
        args.kubectl,
        args.namespace,
        "app.kubernetes.io/name=f1c-lb",
    )
    if len(lb_pods) != 1:
        raise RuntimeError(f"expected one F1c LB Pod, got {len(lb_pods)}")
    _copy_pod_file(
        args.kubectl,
        args.namespace,
        str(lb_pods[0]["name"]),
        "/evidence/decisions.jsonl",
        results / "lb-decisions.jsonl",
    )
    claim_rows = _claim_rows(args.kubectl, args.namespace, batch_id)
    _write_claims(results / "batch-claims.jsonl", claim_rows)

    artifact_metadata = {
        name: {
            "sha256": _sha256_file(results / name),
            "bytes": (results / name).stat().st_size,
            "rows": (
                len(_read_jsonl(results / name))
                if name.endswith(".jsonl")
                else None
            ),
        }
        for name in RAW_ARTIFACTS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gate": "g5-f1c-three-gateway-shared-batchstore",
        "passed": False,
        "created_unix_ns": time.time_ns(),
        "config": {
            "namespace": args.namespace,
            "gateway_ids": list(GATEWAY_IDS),
            "replica_count": args.replica_count,
            "batch_lines": args.batch_lines,
            "baseline_rounds": BASELINE_ROUNDS,
            "sessions_per_gateway": SESSIONS_PER_GATEWAY,
            "timeout_seconds": args.timeout_seconds,
            "load_balancer_hash": "sha256-hrw-x-session-id",
            "batch_delivery": (
                "at-least-once execution with one fenced terminal publication"
            ),
        },
        "source": {
            "commit": args.source_commit,
            "tracked_clean": args.source_clean,
            "gateway_image_digest": args.gateway_image_digest,
            "mock_image_digest": args.mock_image_digest,
            "postgres_image": args.postgres_image,
            "gateway_cri_image_metadata": _cri_image_metadata(
                args.gateway_cri_image_metadata,
                results,
            ),
            "mock_cri_image_metadata": _cri_image_metadata(
                args.mock_cri_image_metadata,
                results,
            ),
            "postgres_cri_image_metadata": _cri_image_metadata(
                args.postgres_cri_image_metadata,
                results,
            ),
            "postgres_docker_image_metadata": _docker_image_metadata(
                args.postgres_docker_image_metadata,
                results,
            ),
            "rendered_manifest": _adjacent_file_metadata(
                args.rendered_manifest,
                results,
            ),
        },
        "batch_id": batch_id,
        "artifacts": artifact_metadata,
        "checks": {},
    }
    (results / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verifier = verify_evidence(results)
    manifest["checks"] = verifier["checks"]
    manifest["passed"] = verifier["passed"]
    (results / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verifier = verify_evidence(results)
    (results / "verifier.json").write_text(
        json.dumps(verifier, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verifier["passed"]:
        failed = [name for name, passed in verifier["checks"].items() if not passed]
        raise RuntimeError(f"F1c evidence verification failed: {failed}")
    return verifier


def _placement_records(
    results: Path,
) -> dict[str, list[dict[str, Any]]]:
    return {
        gateway_id: _read_jsonl(
            results / f"placements-{gateway_id}.jsonl"
        )
        for gateway_id in GATEWAY_IDS
    }


def _output_custom_ids(path: Path) -> tuple[list[str], bool]:
    rows = _read_jsonl(path)
    custom_ids: list[str] = []
    valid = True
    for row in rows:
        custom_id = row.get("custom_id")
        response = row.get("response")
        if (
            not isinstance(custom_id, str)
            or not isinstance(response, dict)
            or response.get("status_code") != 200
            or row.get("error") is not None
        ):
            valid = False
        else:
            custom_ids.append(custom_id)
    return custom_ids, valid


def verify_evidence(results_dir: str | Path) -> dict[str, Any]:
    """Reconstruct the complete F1c decision from immutable raw sidecars."""

    results = Path(results_dir).resolve()
    manifest_path = results / "manifest.json"
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise ValueError("manifest must be an object")
    manifest = manifest_value
    artifacts = manifest.get("artifacts")
    artifact_shape = (
        isinstance(artifacts, dict)
        and set(artifacts) == set(RAW_ARTIFACTS)
    )
    artifact_integrity = artifact_shape
    if artifact_shape:
        for name, metadata in artifacts.items():
            try:
                path = _safe_artifact_path(results, name)
            except ValueError:
                artifact_integrity = False
                continue
            if (
                not path.is_file()
                or not isinstance(metadata, dict)
                or not _HEX_SHA256.fullmatch(str(metadata.get("sha256", "")))
                or metadata.get("sha256") != _sha256_file(path)
                or metadata.get("bytes") != path.stat().st_size
            ):
                artifact_integrity = False
                continue
            if name.endswith(".jsonl"):
                try:
                    rows = _read_jsonl(path)
                except (OSError, ValueError):
                    artifact_integrity = False
                    continue
                if metadata.get("rows") != len(rows):
                    artifact_integrity = False

    try:
        traffic = _read_jsonl(results / "traffic.jsonl")
        gateway_rows = _read_jsonl(results / "gateway-pods.jsonl")
        replica_rows = _read_jsonl(results / "replica-pods.jsonl")
        postgres_rows = _read_jsonl(results / "postgres-pods.jsonl")
        store_rows = _read_jsonl(results / "store-identities.jsonl")
        batch_http = _read_jsonl(results / "batch-http.jsonl")
        lifecycle = _read_jsonl(results / "batch-lifecycle.jsonl")
        claims = _read_jsonl(results / "batch-claims.jsonl")
        lb_decisions = _read_jsonl(results / "lb-decisions.jsonl")
        placements = _placement_records(results)
    except (OSError, ValueError, json.JSONDecodeError):
        traffic = []
        gateway_rows = []
        replica_rows = []
        postgres_rows = []
        store_rows = []
        batch_http = []
        lifecycle = []
        claims = []
        lb_decisions = []
        placements = {gateway_id: [] for gateway_id in GATEWAY_IDS}

    config = manifest.get("config")
    config_valid = (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("gate") == "g5-f1c-three-gateway-shared-batchstore"
        and isinstance(config, dict)
        and config.get("namespace") == DEFAULT_NAMESPACE
        and config.get("gateway_ids") == list(GATEWAY_IDS)
        and config.get("replica_count") == DEFAULT_REPLICA_COUNT
        and config.get("batch_lines") == DEFAULT_BATCH_LINES
        and config.get("baseline_rounds") == BASELINE_ROUNDS
        and config.get("sessions_per_gateway") == SESSIONS_PER_GATEWAY
        and config.get("timeout_seconds") == DEFAULT_TIMEOUT_SECONDS
        and config.get("load_balancer_hash") == "sha256-hrw-x-session-id"
        and config.get("batch_delivery")
        == "at-least-once execution with one fenced terminal publication"
    )
    raw_replica_count = (
        config.get("replica_count") if isinstance(config, dict) else None
    )
    raw_batch_lines = (
        config.get("batch_lines") if isinstance(config, dict) else None
    )
    replica_count = (
        raw_replica_count if type(raw_replica_count) is int else -1
    )
    batch_lines = raw_batch_lines if type(raw_batch_lines) is int else -1

    initial_gateway_rows = [
        row
        for row in gateway_rows
        if row.get("evidence_phase") == "initial"
        and row.get("kind") == "gateway_pod"
    ]
    final_gateway_rows = [
        row
        for row in gateway_rows
        if row.get("evidence_phase") == "final"
        and row.get("kind") == "gateway_pod"
    ]
    initial_gateways = {
        row.get("gateway_id"): row
        for row in initial_gateway_rows
        if isinstance(row.get("gateway_id"), str)
    }
    final_gateways = {
        row.get("gateway_id"): row
        for row in final_gateway_rows
        if isinstance(row.get("gateway_id"), str)
    }
    initial_gateway_valid = (
        len(gateway_rows) == len(GATEWAY_IDS) * 2
        and len(initial_gateway_rows) == len(GATEWAY_IDS)
        and len(final_gateway_rows) == len(GATEWAY_IDS)
        and set(initial_gateways) == set(GATEWAY_IDS)
        and len(
            {
                row.get("uid")
                for row in initial_gateways.values()
                if isinstance(row.get("uid"), str)
            }
        )
        == len(GATEWAY_IDS)
        and all(
            row.get("ready") is True and row.get("pod_phase") == "Running"
            for row in initial_gateways.values()
        )
    )
    final_gateway_valid = (
        len(gateway_rows) == len(GATEWAY_IDS) * 2
        and len(initial_gateway_rows) == len(GATEWAY_IDS)
        and len(final_gateway_rows) == len(GATEWAY_IDS)
        and set(final_gateways) == set(GATEWAY_IDS)
        and len(
            {
                row.get("uid")
                for row in final_gateways.values()
                if isinstance(row.get("uid"), str)
            }
        )
        == len(GATEWAY_IDS)
        and all(
            row.get("ready") is True and row.get("pod_phase") == "Running"
            for row in final_gateways.values()
        )
    )

    initial_replica_rows = [
        row
        for row in replica_rows
        if row.get("evidence_phase") == "initial"
        and row.get("kind") == "replica_pod"
    ]
    final_replica_rows = [
        row
        for row in replica_rows
        if row.get("evidence_phase") == "final"
        and row.get("kind") == "replica_pod"
    ]
    initial_replicas = {
        row.get("uid"): row
        for row in initial_replica_rows
        if isinstance(row.get("uid"), str)
    }
    final_replicas = {
        row.get("uid"): row
        for row in final_replica_rows
        if isinstance(row.get("uid"), str)
    }
    replica_membership_valid = (
        len(replica_rows) == replica_count * 2
        and len(initial_replica_rows) == replica_count
        and len(final_replica_rows) == replica_count
        and len(initial_replicas) == replica_count
        and set(initial_replicas) == set(final_replicas)
        and all(
            row.get("ready") is True and row.get("pod_phase") == "Running"
            for row in (*initial_replicas.values(), *final_replicas.values())
        )
    )
    replica_uids = tuple(sorted(initial_replicas))
    initial_postgres_rows = [
        row
        for row in postgres_rows
        if row.get("evidence_phase") == "initial"
        and row.get("kind") == "postgres_pod"
    ]
    final_postgres_rows = [
        row
        for row in postgres_rows
        if row.get("evidence_phase") == "final"
        and row.get("kind") == "postgres_pod"
    ]
    postgres_topology_valid = (
        len(postgres_rows) == 2
        and len(initial_postgres_rows) == len(final_postgres_rows) == 1
        and initial_postgres_rows[0].get("uid") == final_postgres_rows[0].get("uid")
        and all(
            row.get("ready") is True and row.get("pod_phase") == "Running"
            for row in (*initial_postgres_rows, *final_postgres_rows)
        )
    )

    initial_store = [
        row
        for row in store_rows
        if row.get("kind") == "store_identity" and row.get("phase") is None
    ]
    final_store = [
        row
        for row in store_rows
        if row.get("kind") == "store_identity" and row.get("phase") == "final"
    ]
    all_store = initial_store + final_store
    shared_store_identity = (
        len(store_rows) == len(GATEWAY_IDS) * 2
        and len(initial_store) == len(final_store) == len(GATEWAY_IDS)
        and {row.get("gateway_id") for row in initial_store}
        == {row.get("gateway_id") for row in final_store}
        == set(GATEWAY_IDS)
        and len({row.get("dsn_sha256") for row in all_store}) == 1
        and len({row.get("postgres_system_identifier") for row in all_store}) == 1
        and all(
            _HEX_SHA256.fullmatch(str(row.get("dsn_sha256", "")))
            and isinstance(row.get("postgres_system_identifier"), str)
            and row["postgres_system_identifier"]
            for row in all_store
        )
        and all(
            row.get("gateway_uid")
            == initial_gateways.get(row.get("gateway_id"), {}).get("uid")
            for row in initial_store
        )
        and all(
            row.get("gateway_uid")
            == final_gateways.get(row.get("gateway_id"), {}).get("uid")
            for row in final_store
        )
    )

    affinity = [row for row in traffic if row.get("kind") == "affinity"]
    baseline = [row for row in affinity if row.get("phase") == "baseline"]
    failover = [row for row in affinity if row.get("phase") == "failover"]
    recovered = [row for row in affinity if row.get("phase") == "recovered"]
    expected_baseline_sessions = len(GATEWAY_IDS) * SESSIONS_PER_GATEWAY
    expected_baseline_requests = expected_baseline_sessions * BASELINE_ROUNDS
    expected_affinity_requests = expected_baseline_requests + 2
    requests_successful = (
        len(traffic) == len(affinity) == expected_affinity_requests
        and all(
            row.get("status_code") == 200
            and row.get("transport_error") is None
            and row.get("method") == "POST"
            and row.get("path") == "/v1/chat/completions"
            and row.get("response_valid") is True
            and isinstance(row.get("lb_request_id"), str)
            and row["lb_request_id"]
            for row in affinity
        )
    )
    affinity_response_identity_exact = all(
        (identity := _chat_identity(row.get("response_json")))
        != (None, None)
        and identity == (row.get("pod_name"), row.get("pod_uid"))
        for row in affinity
    )
    baseline_sessions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline:
        session = row.get("session")
        if isinstance(session, str):
            baseline_sessions[session].append(row)
    lb_hash_exact = bool(baseline_sessions) and all(
        row.get("gateway_id") == rendezvous_order(str(row.get("session")), GATEWAY_IDS)[0]
        for row in baseline
    )
    every_gateway_used = {row.get("gateway_id") for row in baseline} == set(GATEWAY_IDS)
    baseline_sticky = (
        len(baseline) == expected_baseline_requests
        and len(baseline_sessions) == expected_baseline_sessions
        and all(
            len(
                {
                    row.get("session")
                    for row in baseline
                    if row.get("gateway_id") == gateway_id
                }
            )
            == SESSIONS_PER_GATEWAY
            for gateway_id in GATEWAY_IDS
        )
        and all(
            len(rows) == BASELINE_ROUNDS
            and len({row.get("gateway_id") for row in rows}) == 1
            and len({row.get("pod_uid") for row in rows}) == 1
            for rows in baseline_sessions.values()
        )
    )
    replica_affinity_exact = (
        bool(replica_uids)
        and all(
            isinstance(row.get("session"), str)
            and row.get("pod_uid")
            == replica_winner(str(row["session"]), replica_uids)
            for row in affinity
        )
    )

    placement_by_request: dict[tuple[str, str], Mapping[str, Any]] = {}
    placement_shape = True
    for gateway_id, rows in placements.items():
        for row in rows:
            if row.get("kind") == "membership":
                continue
            if row.get("kind") != "replica":
                placement_shape = False
                continue
            session_hash = row.get("session_sha256")
            replica_id = row.get("replica_id")
            gateway_request_id = row.get("request_id")
            if (
                not isinstance(session_hash, str)
                or not _HEX_SHA256.fullmatch(session_hash)
                or not isinstance(replica_id, str)
                or row.get("reason") != "session_affinity"
            ):
                # Batch requests are session-less and legitimately use another
                # placement reason; only affinity rows participate in this join.
                if session_hash is not None:
                    placement_shape = False
                continue
            if not isinstance(gateway_request_id, str) or not gateway_request_id:
                placement_shape = False
                continue
            key = (gateway_id, gateway_request_id)
            if key in placement_by_request:
                placement_shape = False
                continue
            placement_by_request[key] = row
    placements_join = placement_shape and all(
        isinstance(row.get("gateway_id"), str)
        and isinstance(row.get("gateway_request_id"), str)
        and (
            placement := placement_by_request.get(
                (row["gateway_id"], row["gateway_request_id"])
            )
        )
        is not None
        and placement.get("session_sha256") == row.get("session_sha256")
        and placement.get("replica_id") == row.get("pod_uid")
        and placement.get("request_id") == row.get("gateway_request_id")
        for row in affinity
    )
    expected_placement_keys = {
        (row.get("gateway_id"), row.get("gateway_request_id"))
        for row in affinity
        if isinstance(row.get("gateway_id"), str)
        and isinstance(row.get("gateway_request_id"), str)
    }
    placements_join = (
        placements_join
        and len(expected_placement_keys) == len(affinity)
        and set(placement_by_request) == expected_placement_keys
    )
    expected_replica_uids = set(replica_uids)

    def exact_replica_uid_list(value: object) -> bool:
        return (
            isinstance(value, list)
            and len(value) == replica_count
            and all(isinstance(replica_uid, str) for replica_uid in value)
            and len(set(value)) == replica_count
            and set(value) == expected_replica_uids
        )

    converged_membership_exact = (
        replica_membership_valid
        and all(
            bool(
                membership_rows := [
                    row
                    for row in placements[gateway_id]
                    if row.get("kind") == "membership"
                ]
            )
            and type(membership_rows[-1].get("observed_ns")) is int
            and exact_replica_uid_list(membership_rows[-1].get("replica_ids"))
            and exact_replica_uid_list(membership_rows[-1].get("healthy_ids"))
            and exact_replica_uid_list(membership_rows[-1].get("eligible_ids"))
            for gateway_id in GATEWAY_IDS
        )
    )

    lb_decisions_by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    lb_shape = True
    for row in lb_decisions:
        request_id = row.get("request_id")
        if (
            not isinstance(request_id, str)
            or row.get("kind") != "f1c_lb_decision"
            or row.get("selection_reason") != "session_hrw"
        ):
            lb_shape = False
            continue
        lb_decisions_by_request[request_id].append(row)
    lb_http_rows = [*affinity, *batch_http]
    expected_lb_request_ids = {
        row.get("lb_request_id")
        for row in lb_http_rows
        if isinstance(row.get("lb_request_id"), str)
    }

    def attempts_are_exact(
        http_row: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> bool:
        session = http_row.get("session")
        attempts = decision.get("attempts")
        if (
            not isinstance(session, str)
            or not isinstance(attempts, list)
            or not attempts
            or not all(isinstance(attempt, dict) for attempt in attempts)
        ):
            return False
        candidates = rendezvous_order(session, GATEWAY_IDS)
        if [attempt.get("gateway_id") for attempt in attempts] != list(
            candidates[: len(attempts)]
        ):
            return False
        response_attempt = attempts[-1]
        response_exact = (
            response_attempt.get("outcome") == "response"
            and response_attempt.get("status") == http_row.get("status_code")
            and response_attempt.get("upstream_request_id")
            == http_row.get("gateway_request_id")
            and response_attempt.get("gateway_id") == http_row.get("gateway_id")
        )
        if http_row.get("kind") == "affinity" and http_row.get("phase") == "failover":
            first_attempt = attempts[0]
            return (
                len(attempts) == 2
                and first_attempt.get("outcome") == "transport_error"
                and first_attempt.get("status") is None
                and first_attempt.get("upstream_request_id") is None
                and isinstance(first_attempt.get("error_type"), str)
                and bool(first_attempt["error_type"])
                and response_exact
            )
        return len(attempts) == 1 and response_exact

    lb_raw_join = (
        lb_shape
        and len(expected_lb_request_ids) == len(lb_http_rows)
        and set(lb_decisions_by_request) == expected_lb_request_ids
        and all(
            isinstance(row.get("lb_request_id"), str)
            and isinstance(row.get("session"), str)
            and isinstance(row.get("method"), str)
            and isinstance(row.get("path"), str)
            and row.get("session_sha256")
            == _sha256_bytes(row["session"].encode())
            and len(lb_decisions_by_request.get(row["lb_request_id"], ())) == 1
            and (
                decision := lb_decisions_by_request[row["lb_request_id"]][0]
            ).get("selected_gateway_id")
            == row.get("gateway_id")
            and decision.get("final_status") == row.get("status_code")
            and decision.get("session_sha256") == row.get("session_sha256")
            and decision.get("method") == row.get("method")
            and decision.get("path") == row.get("path")
            and type(row.get("request_body_bytes")) is int
            and row["request_body_bytes"] >= 0
            and _HEX_SHA256.fullmatch(
                str(row.get("request_body_sha256", ""))
            )
            is not None
            and decision.get("body_bytes") == row.get("request_body_bytes")
            and decision.get("body_sha256") == row.get("request_body_sha256")
            and (
                row.get("request_json") is None
                or (
                    row.get("request_body_bytes")
                    == len(_canonical_json(row["request_json"]).encode())
                    and row.get("request_body_sha256")
                    == _sha256_bytes(
                        _canonical_json(row["request_json"]).encode()
                    )
                )
            )
            and wall_clock_envelope_contains(
                row.get("started_unix_ns"),
                decision.get("started_ns"),
                decision.get("finished_ns"),
                row.get("ended_unix_ns"),
            )
            and decision.get("candidate_gateway_ids")
            == list(rendezvous_order(row["session"], GATEWAY_IDS))
            and attempts_are_exact(row, decision)
            for row in lb_http_rows
        )
    )

    created_rows = [row for row in lifecycle if row.get("kind") == "created"]
    owner_claim_rows = [
        row for row in lifecycle if row.get("kind") == "owner_claim"
    ]
    kill_requested_rows = [
        row for row in lifecycle if row.get("kind") == "owner_kill_requested"
    ]
    killed_rows = [row for row in lifecycle if row.get("kind") == "owner_killed"]
    reclaim_rows = [row for row in lifecycle if row.get("kind") == "reclaimed"]
    completed_rows = [row for row in lifecycle if row.get("kind") == "completed"]
    expected_lifecycle_kinds = [
        "created",
        "owner_claim",
        "owner_kill_requested",
        "owner_killed",
        "reclaimed",
        "completed",
    ]
    lifecycle_exact = (
        [row.get("kind") for row in lifecycle] == expected_lifecycle_kinds
        and all(
            row.get("schema_version") == SCHEMA_VERSION
            and row.get("batch_id") == manifest.get("batch_id")
            for row in lifecycle
        )
        and len(created_rows)
        == len(owner_claim_rows)
        == len(kill_requested_rows)
        == len(killed_rows)
        == len(reclaim_rows)
        == len(completed_rows)
        == 1
    )
    created = created_rows[0] if created_rows else {}
    old_claim = owner_claim_rows[0] if owner_claim_rows else {}
    kill_requested = kill_requested_rows[0] if kill_requested_rows else {}
    killed = killed_rows[0] if killed_rows else {}
    new_claim = reclaim_rows[0] if reclaim_rows else {}
    completed = completed_rows[0] if completed_rows else {}
    killed_gateway = old_claim.get("gateway_id")
    killed_deployment = (
        f"f1c-gateway-{killed_gateway}" if killed_gateway in GATEWAY_IDS else None
    )
    owner_killed_by_uid = (
        lifecycle_exact
        and killed_gateway in GATEWAY_IDS
        and initial_gateway_valid
        and old_claim.get("worker_id") == initial_gateways[killed_gateway].get("uid")
        and old_claim.get("event") == "claimed"
        and kill_requested.get("gateway_id") == killed_gateway
        and kill_requested.get("deployment") == killed_deployment
        and kill_requested.get("worker_id") == old_claim.get("worker_id")
        and kill_requested.get("fencing_token")
        == old_claim.get("fencing_token")
        and killed.get("gateway_id") == killed_gateway
        and killed.get("deployment") == killed_deployment
        and killed.get("worker_id") == old_claim.get("worker_id")
        and killed.get("fencing_token") == old_claim.get("fencing_token")
        and kill_requested.get("clock") == "postgres_clock_timestamp"
        and killed.get("clock") == "postgres_clock_timestamp"
        and type(kill_requested.get("at_unix_ns")) is int
        and type(killed.get("at_unix_ns")) is int
        and kill_requested["at_unix_ns"] <= killed["at_unix_ns"]
    )
    gateway_restart_exact = (
        owner_killed_by_uid
        and final_gateway_valid
        and final_gateways[killed_gateway].get("uid")
        != initial_gateways[killed_gateway].get("uid")
        and all(
            final_gateways[gateway_id].get("uid")
            == initial_gateways[gateway_id].get("uid")
            for gateway_id in GATEWAY_IDS
            if gateway_id != killed_gateway
        )
    )
    fenced_reclaim = (
        lifecycle_exact
        and new_claim.get("gateway_id") in GATEWAY_IDS
        and new_claim.get("gateway_id") != killed_gateway
        and new_claim.get("worker_id") != old_claim.get("worker_id")
        and new_claim.get("worker_id")
        == initial_gateways.get(new_claim.get("gateway_id"), {}).get("uid")
        and new_claim.get("worker_id")
        == final_gateways.get(new_claim.get("gateway_id"), {}).get("uid")
        and new_claim.get("event") == "reclaimed"
        and type(old_claim.get("fencing_token")) is int
        and type(new_claim.get("fencing_token")) is int
        and new_claim["fencing_token"] == old_claim["fencing_token"] + 1
    )

    claims_match_batch = bool(claims) and all(
        row.get("schema_version") == SCHEMA_VERSION
        and row.get("batch_id") == manifest.get("batch_id")
        for row in claims
    )
    claim_events = claims if claims_match_batch else []
    claim_sequences = [row.get("sequence") for row in claim_events]
    sequence_ordered = (
        bool(claim_sequences)
        and all(type(sequence) is int for sequence in claim_sequences)
        and claim_sequences == sorted(set(claim_sequences))
    )
    claim_times = [row.get("at_unix_ns") for row in claim_events]
    claim_times_ordered = (
        bool(claim_times)
        and all(type(at_unix_ns) is int for at_unix_ns in claim_times)
        and claim_times == sorted(claim_times)
    )
    claim_event_names = [row.get("event") for row in claim_events]
    reclaim_indexes = [
        index
        for index, event in enumerate(claim_event_names)
        if event == "reclaimed"
    ]
    claim_event_shape = (
        len(claim_events) >= 3
        and claim_event_names[0] == "claimed"
        and claim_event_names[-1] == "terminal_committed"
        and bool(reclaim_indexes)
        and all(
            event in {"renewed", "reclaimed"}
            for event in claim_event_names[1:-1]
        )
    )
    segment_starts = [0, *reclaim_indexes] if claim_event_shape else []
    segment_ends = (
        [*reclaim_indexes, len(claim_events) - 1]
        if claim_event_shape
        else []
    )
    lease_segments = [
        claim_events[start:end]
        for start, end in zip(segment_starts, segment_ends, strict=True)
    ]
    old_audit = lease_segments[0][0] if lease_segments else {}
    new_audit = lease_segments[1][0] if len(lease_segments) >= 2 else {}
    terminal_audit = claim_events[-1] if claim_event_shape else {}
    owner_lease_events = lease_segments[0] if lease_segments else []
    final_lease_events = lease_segments[-1] if lease_segments else []
    last_owner_lease = owner_lease_events[-1] if owner_lease_events else {}

    def audit_matches_lifecycle(
        audit: Mapping[str, Any],
        lifecycle_row: Mapping[str, Any],
    ) -> bool:
        return all(
            audit.get(field) == lifecycle_row.get(field)
            for field in (
                "sequence",
                "batch_id",
                "event",
                "worker_id",
                "fencing_token",
                "at_unix_ns",
                "lease_until_unix_ns",
                "previous_lease_until_unix_ns",
            )
        )

    active_leases_valid = all(
        type(row.get("at_unix_ns")) is int
        and type(row.get("lease_until_unix_ns")) is int
        and row["at_unix_ns"] < row["lease_until_unix_ns"]
        for segment in lease_segments
        for row in segment
    )
    valid_gateway_worker_ids = {
        gateway.get("uid")
        for gateway in (*initial_gateways.values(), *final_gateways.values())
        if isinstance(gateway.get("uid"), str)
    }
    fence_segments_exact = bool(lease_segments) and all(
        segment
        and isinstance(segment[0].get("worker_id"), str)
        and segment[0].get("worker_id") in valid_gateway_worker_ids
        and type(segment[0].get("fencing_token")) is int
        and all(
            row.get("worker_id") == segment[0].get("worker_id")
            and row.get("fencing_token") == segment[0].get("fencing_token")
            for row in segment
        )
        for segment in lease_segments
    )
    reclaim_chain_valid = len(lease_segments) >= 2 and all(
        next_segment[0].get("event") == "reclaimed"
        and type(previous_segment[0].get("fencing_token")) is int
        and type(next_segment[0].get("fencing_token")) is int
        and next_segment[0]["fencing_token"]
        == previous_segment[0]["fencing_token"] + 1
        and type(previous_segment[-1].get("lease_until_unix_ns")) is int
        and type(
            next_segment[0].get("previous_lease_until_unix_ns")
        )
        is int
        and next_segment[0]["previous_lease_until_unix_ns"]
        == previous_segment[-1]["lease_until_unix_ns"]
        and type(next_segment[0].get("at_unix_ns")) is int
        and previous_segment[-1]["lease_until_unix_ns"]
        <= next_segment[0]["at_unix_ns"]
        for previous_segment, next_segment in pairwise(lease_segments)
    )
    kill_requested_ns = kill_requested.get("at_unix_ns")
    kill_covered_by_old_lease = type(kill_requested_ns) is int and any(
        row["at_unix_ns"] <= kill_requested_ns < row["lease_until_unix_ns"]
        for row in owner_lease_events
        if type(row.get("at_unix_ns")) is int
        and type(row.get("lease_until_unix_ns")) is int
    )
    final_claim_audit = final_lease_events[0] if final_lease_events else {}
    last_final_lease = final_lease_events[-1] if final_lease_events else {}
    durable_claim_audit = (
        lifecycle_exact
        and claims_match_batch
        and sequence_ordered
        and claim_times_ordered
        and claim_event_shape
        and audit_matches_lifecycle(old_audit, old_claim)
        and audit_matches_lifecycle(new_audit, new_claim)
        and active_leases_valid
        and fence_segments_exact
        and reclaim_chain_valid
        and kill_covered_by_old_lease
        and type(last_owner_lease.get("at_unix_ns")) is int
        and type(killed.get("at_unix_ns")) is int
        and last_owner_lease["at_unix_ns"] < killed["at_unix_ns"]
        and type(new_audit.get("at_unix_ns")) is int
        # owner_killed is the time a polling observer noticed Pod absence, not
        # the actual stop time; scheduler/API jitter may place that observation
        # after a valid post-expiry reclaim.
        and kill_requested_ns < new_audit["at_unix_ns"]
        and terminal_audit.get("worker_id") == final_claim_audit.get("worker_id")
        and terminal_audit.get("fencing_token")
        == final_claim_audit.get("fencing_token")
        and type(terminal_audit.get("at_unix_ns")) is int
        and terminal_audit["at_unix_ns"] >= final_claim_audit["at_unix_ns"]
        and type(last_final_lease.get("at_unix_ns")) is int
        and type(last_final_lease.get("lease_until_unix_ns")) is int
        and last_final_lease["at_unix_ns"]
        <= terminal_audit["at_unix_ns"]
        < last_final_lease["lease_until_unix_ns"]
        and type(completed.get("at_unix_ns")) is int
        and completed.get("clock") == "postgres_clock_timestamp"
        and completed["at_unix_ns"] >= terminal_audit["at_unix_ns"]
    )

    failover_decision = (
        lb_decisions_by_request.get(str(failover[0].get("lb_request_id")), [None])[0]
        if len(failover) == 1
        and len(
            lb_decisions_by_request.get(
                str(failover[0].get("lb_request_id")), ()
            )
        )
        == 1
        else None
    )
    failover_preserves_replica = (
        len(failover) == len(recovered) == 1
        and isinstance(failover[0].get("session"), str)
        and failover[0].get("session") == recovered[0].get("session")
        and failover[0].get("pod_uid") == recovered[0].get("pod_uid")
        and failover[0].get("pod_uid")
        == replica_winner(str(failover[0].get("session")), replica_uids)
        and killed_gateway in GATEWAY_IDS
        and rendezvous_order(
            str(failover[0].get("session")),
            GATEWAY_IDS,
        )[0]
        == killed_gateway
        and rendezvous_order(
            str(failover[0].get("session")),
            tuple(gateway for gateway in GATEWAY_IDS if gateway != killed_gateway),
        )[0]
        == failover[0].get("gateway_id")
        and recovered[0].get("gateway_id") == killed_gateway
        and isinstance(failover_decision, dict)
        and attempts_are_exact(failover[0], failover_decision)
        and isinstance(failover_decision.get("candidate_gateway_ids"), list)
        and failover_decision["candidate_gateway_ids"][:2]
        == [killed_gateway, failover[0].get("gateway_id")]
    )

    operation_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in batch_http:
        operation = row.get("operation")
        if isinstance(operation, str):
            operation_rows[operation].append(row)
    expected_batch_operations = {
        "upload_input_a",
        "read_input_b",
        "create_batch_b",
        "poll_batch_c",
        *(f"read_output_{gateway_id}" for gateway_id in GATEWAY_IDS),
    }
    upload_row = (
        operation_rows["upload_input_a"][0]
        if len(operation_rows["upload_input_a"]) == 1
        else {}
    )
    upload_payload = upload_row.get("response_json")
    input_file_id = (
        upload_payload.get("id") if isinstance(upload_payload, dict) else None
    )
    input_read_row = (
        operation_rows["read_input_b"][0]
        if len(operation_rows["read_input_b"]) == 1
        else {}
    )
    input_read_payload = input_read_row.get("response_json")
    create_row = (
        operation_rows["create_batch_b"][0]
        if len(operation_rows["create_batch_b"]) == 1
        else {}
    )
    create_request = create_row.get("request_json")
    create_payload = create_row.get("response_json")
    batch_id = manifest.get("batch_id")
    output_file_id = completed.get("output_file_id")
    poll_rows = operation_rows["poll_batch_c"]
    poll_payloads = [row.get("response_json") for row in poll_rows]
    poll_statuses = [
        payload.get("status") if isinstance(payload, dict) else None
        for payload in poll_payloads
    ]
    nonterminal_poll_ranks = [
        {"validating": 0, "in_progress": 1}.get(status, -1)
        for status in poll_statuses[:-1]
    ]
    batch_object_graph_exact = (
        set(operation_rows) == expected_batch_operations
        and len(batch_http)
        == sum(len(rows) for rows in operation_rows.values())
        and isinstance(input_file_id, str)
        and bool(input_file_id)
        and upload_row.get("method") == "POST"
        and upload_row.get("path") == "/v1/files"
        and input_read_row.get("method") == "GET"
        and input_read_row.get("path") == f"/v1/files/{input_file_id}"
        and isinstance(input_read_payload, dict)
        and input_read_payload.get("id") == input_file_id
        and create_row.get("method") == "POST"
        and create_row.get("path") == "/v1/batches"
        and isinstance(create_request, dict)
        and create_request.get("input_file_id") == input_file_id
        and isinstance(create_payload, dict)
        and isinstance(batch_id, str)
        and bool(batch_id)
        and create_payload.get("id") == batch_id
        and create_payload.get("input_file_id") == input_file_id
        and created.get("input_file_id") == input_file_id
        and created.get("line_count") == batch_lines
        and created.get("input_sha256")
        == _sha256_bytes(_batch_payload(batch_lines))
        and all(row.get("batch_id") == batch_id for row in lifecycle)
        and bool(poll_rows)
        and all(
            row.get("method") == "GET"
            and row.get("path") == f"/v1/batches/{batch_id}"
            and isinstance(row.get("response_json"), dict)
            and row["response_json"].get("id") == batch_id
            for row in poll_rows
        )
        and all(
            status in {"validating", "in_progress"}
            for status in poll_statuses[:-1]
        )
        and nonterminal_poll_ranks == sorted(nonterminal_poll_ranks)
        and poll_statuses[-1] == "completed"
        and isinstance(output_file_id, str)
        and bool(output_file_id)
        and all(
            len(operation_rows[f"read_output_{gateway_id}"]) == 1
            and operation_rows[f"read_output_{gateway_id}"][0].get("method")
            == "GET"
            and operation_rows[f"read_output_{gateway_id}"][0].get("path")
            == f"/v1/files/{output_file_id}/content"
            for gateway_id in GATEWAY_IDS
        )
    )
    cross_gateway_batch = (
        len(operation_rows["upload_input_a"]) == 1
        and operation_rows["upload_input_a"][0].get("gateway_id") == "a"
        and len(operation_rows["read_input_b"]) == 1
        and operation_rows["read_input_b"][0].get("gateway_id") == "b"
        and len(operation_rows["create_batch_b"]) == 1
        and operation_rows["create_batch_b"][0].get("gateway_id") == "b"
        and bool(operation_rows["poll_batch_c"])
        and all(
            row.get("gateway_id") == "c"
            for row in operation_rows["poll_batch_c"]
        )
        and all(
            len(operation_rows[f"read_output_{gateway_id}"]) == 1
            and operation_rows[f"read_output_{gateway_id}"][0].get("gateway_id")
            == gateway_id
            for gateway_id in GATEWAY_IDS
        )
        and all(
            row.get("status_code") == 200 and row.get("transport_error") is None
            for row in batch_http
        )
        and all(
            isinstance(row.get("session"), str)
            and row.get("gateway_id")
            == rendezvous_order(row["session"], GATEWAY_IDS)[0]
            for row in batch_http
        )
    )
    counts = completed.get("request_counts")
    final_poll = (
        operation_rows["poll_batch_c"][-1]
        if operation_rows["poll_batch_c"]
        else {}
    )
    final_poll_payload = final_poll.get("response_json")
    batch_completed = (
        completed.get("status") == "completed"
        and isinstance(completed.get("output_file_id"), str)
        and completed.get("error_file_id") is None
        and completed.get("errors") in (None, {})
        and isinstance(counts, dict)
        and counts.get("total") == batch_lines
        and counts.get("completed") == batch_lines
        and counts.get("failed") == 0
        and isinstance(final_poll_payload, dict)
        and final_poll_payload.get("id") == manifest.get("batch_id")
        and final_poll_payload.get("status") == completed.get("status")
        and final_poll_payload.get("output_file_id")
        == completed.get("output_file_id")
        and final_poll_payload.get("request_counts") == counts
    )

    output_digests = {
        gateway_id: _sha256_file(results / f"batch-output-{gateway_id}.jsonl")
        for gateway_id in GATEWAY_IDS
        if (results / f"batch-output-{gateway_id}.jsonl").is_file()
    }
    output_ids: dict[str, list[str]] = {}
    output_shape = len(output_digests) == len(GATEWAY_IDS)
    if output_shape:
        for gateway_id in GATEWAY_IDS:
            try:
                ids, valid = _output_custom_ids(
                    results / f"batch-output-{gateway_id}.jsonl"
                )
            except (OSError, ValueError):
                ids, valid = [], False
            output_ids[gateway_id] = ids
            output_shape = output_shape and valid
    expected_ids = [f"f1c-{index:06d}" for index in range(max(0, batch_lines))]
    shared_output_exact = (
        output_shape
        and len(set(output_digests.values())) == 1
        and all(
            operation_rows[f"read_output_{gateway_id}"][0].get(
                "content_sha256"
            )
            == output_digests[gateway_id]
            for gateway_id in GATEWAY_IDS
            if len(operation_rows[f"read_output_{gateway_id}"]) == 1
        )
        and all(
            len(ids) == batch_lines
            and len(set(ids)) == batch_lines
            and sorted(ids) == expected_ids
            for ids in output_ids.values()
        )
    )

    source = manifest.get("source")
    cri_metadata: dict[str, dict[str, Any]] = {}
    cri_metadata_valid = isinstance(source, dict)
    if isinstance(source, dict):
        for role in ("gateway", "mock", "postgres"):
            embedded = source.get(f"{role}_cri_image_metadata")
            try:
                path = _safe_artifact_path(
                    results,
                    embedded.get("path") if isinstance(embedded, dict) else None,
                )
                reconstructed = _cri_image_metadata(path, results)
            except (OSError, ValueError, json.JSONDecodeError):
                cri_metadata_valid = False
                continue
            if reconstructed != embedded:
                cri_metadata_valid = False
                continue
            cri_metadata[role] = reconstructed
    postgres_docker_metadata: dict[str, Any] = {}
    postgres_docker_metadata_valid = False
    if isinstance(source, dict):
        embedded = source.get("postgres_docker_image_metadata")
        try:
            path = _safe_artifact_path(
                results,
                embedded.get("path") if isinstance(embedded, dict) else None,
            )
            reconstructed = _docker_image_metadata(path, results)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            postgres_docker_metadata_valid = reconstructed == embedded
            if postgres_docker_metadata_valid:
                postgres_docker_metadata = reconstructed
    rendered_manifest_valid = False
    if isinstance(source, dict):
        rendered_manifest = source.get("rendered_manifest")
        try:
            rendered_path = _safe_artifact_path(
                results,
                (
                    rendered_manifest.get("path")
                    if isinstance(rendered_manifest, dict)
                    else None
                ),
            )
            reconstructed_rendered_manifest = _adjacent_file_metadata(
                rendered_path,
                results,
            )
        except (OSError, ValueError):
            pass
        else:
            rendered_manifest_valid = (
                reconstructed_rendered_manifest == rendered_manifest
                and _HEX_SHA256.fullmatch(
                    str(rendered_manifest.get("sha256", ""))
                )
                is not None
            )
    postgres_spec_image = (
        source.get("postgres_image") if isinstance(source, dict) else None
    )
    postgres_pin = _runtime_digest(postgres_spec_image)
    postgres_display_image = (
        postgres_spec_image.rpartition("@")[0]
        if isinstance(postgres_spec_image, str)
        else None
    )
    postgres_source_chain_valid = (
        postgres_docker_metadata_valid
        and isinstance(postgres_spec_image, str)
        and isinstance(postgres_display_image, str)
        and postgres_pin is not None
        and any(
            _normalized_image_repository(repo_digest)
            == _normalized_image_repository(postgres_spec_image)
            and _runtime_digest(repo_digest) == postgres_pin
            for repo_digest in postgres_docker_metadata.get("repo_digests", ())
        )
        and any(
            _normalized_image_reference(repo_tag)
            == _normalized_image_reference(postgres_display_image)
            for repo_tag in postgres_docker_metadata.get("repo_tags", ())
        )
        and postgres_docker_metadata.get("config_id")
        == cri_metadata.get("postgres", {}).get("status_id")
    )
    provenance_valid = (
        isinstance(source, dict)
        and cri_metadata_valid
        and rendered_manifest_valid
        and postgres_source_chain_valid
        and set(cri_metadata) == {"gateway", "mock", "postgres"}
        and bool(_COMMIT.fullmatch(str(source.get("commit", ""))))
        and source.get("tracked_clean") is True
        and bool(
            _IMAGE_DIGEST.fullmatch(str(source.get("gateway_image_digest", "")))
        )
        and bool(_IMAGE_DIGEST.fullmatch(str(source.get("mock_image_digest", ""))))
        and bool(_IMAGE_DIGEST.fullmatch(str(source.get("postgres_image", ""))))
        and cri_metadata["gateway"].get("status_id")
        == source.get("gateway_image_digest")
        and cri_metadata["mock"].get("status_id")
        == source.get("mock_image_digest")
        and cri_metadata["gateway"].get("revision") == source.get("commit")
        and cri_metadata["mock"].get("revision") == source.get("commit")
    )
    allowed_runtime_digests = {
        role: {
            digest
            for digest in metadata.get("runtime_digests", ())
            if isinstance(digest, str)
            and _RUNTIME_DIGEST.fullmatch(digest) is not None
        }
        for role, metadata in cri_metadata.items()
    }
    if postgres_source_chain_valid and postgres_pin is not None:
        allowed_runtime_digests["postgres"].add(postgres_pin)
    runtime_images_valid = (
        provenance_valid
        and postgres_topology_valid
        and all(allowed_runtime_digests.get(role) for role in cri_metadata)
        and isinstance(postgres_spec_image, str)
        and all(
            _container_image_matches(
                row,
                container_name="gateway",
                spec_image="kairyu:dev",
                runtime_digests=allowed_runtime_digests["gateway"],
            )
            for row in (*initial_gateways.values(), *final_gateways.values())
        )
        and all(
            _container_image_matches(
                row,
                container_name="mock",
                spec_image="kairyu-f1a-mock:dev",
                runtime_digests=allowed_runtime_digests["mock"],
            )
            for row in (*initial_replicas.values(), *final_replicas.values())
        )
        and all(
            _container_image_matches(
                row,
                container_name="postgres",
                spec_image=postgres_spec_image,
                runtime_digests=allowed_runtime_digests["postgres"],
            )
            for row in (*initial_postgres_rows, *final_postgres_rows)
        )
    )

    checks = {
        "artifact_paths_and_digests_replay": artifact_integrity,
        "frozen_gate_config_valid": config_valid,
        "clean_source_and_image_references_attested": provenance_valid,
        "runtime_gateway_mock_and_postgres_images_match_sources": (
            runtime_images_valid
        ),
        "three_initial_gateway_pod_uids_ready": initial_gateway_valid,
        "three_final_gateway_pod_uids_ready": final_gateway_valid,
        "only_killed_gateway_uid_restarted": gateway_restart_exact,
        "replica_uid_membership_stable_and_ready": replica_membership_valid,
        "all_gateways_use_one_postgres_identity": shared_store_identity,
        "all_affinity_requests_retry_free_valid_2xx": requests_successful,
        "affinity_response_json_recomputes_exact_pod_identity": (
            affinity_response_identity_exact
        ),
        "lb_sha256_hrw_matches_every_baseline_request": lb_hash_exact,
        "baseline_traffic_uses_all_three_gateways": every_gateway_used,
        "baseline_sessions_remain_sticky": baseline_sticky,
        "replica_hrw_matches_across_gateways": replica_affinity_exact,
        "gateway_placement_logs_join_raw_responses": placements_join,
        "all_gateway_membership_logs_converge_to_raw_replica_uids": (
            converged_membership_exact
        ),
        "lb_decision_log_joins_raw_responses": lb_raw_join,
        "active_claim_owner_pod_uid_was_killed": owner_killed_by_uid,
        "expired_lease_reclaimed_by_different_gateway_with_higher_fence": fenced_reclaim,
        "durable_claim_audit_has_one_new-fence_terminal_commit": durable_claim_audit,
        "gateway_failover_preserves_replica_affinity": failover_preserves_replica,
        "batch_http_ids_paths_and_lifecycle_form_one_exact_graph": (
            batch_object_graph_exact
        ),
        "file_create_poll_and_output_cross_gateways": cross_gateway_batch,
        "reclaimed_batch_completed_without_failed_lines": batch_completed,
        "output_is_byte_identical_and_complete_through_all_gateways": shared_output_exact,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": manifest.get("gate"),
        "manifest_path": "manifest.json",
        "verified_unix_ns": time.time_ns(),
        "checks": checks,
        "passed": all(checks.values()),
        "summary": {
            "affinity_requests": len(affinity),
            "baseline_sessions": len(baseline_sessions),
            "replica_uids": len(replica_uids),
            "claim_events": len(claim_events),
            "batch_lines": batch_lines,
            "output_sha256": (
                next(iter(output_digests.values()))
                if len(set(output_digests.values())) == 1 and output_digests
                else None
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:18082")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--replica-count", type=int, default=DEFAULT_REPLICA_COUNT)
    parser.add_argument("--batch-lines", type=int, default=DEFAULT_BATCH_LINES)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("bench/results/f1c-three-gateway"),
    )
    parser.add_argument("--source-commit", default="")
    parser.add_argument(
        "--source-clean",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gateway-image-digest", default="")
    parser.add_argument("--mock-image-digest", default="")
    parser.add_argument("--postgres-image", default="")
    parser.add_argument("--gateway-cri-image-metadata", type=Path)
    parser.add_argument("--mock-cri-image-metadata", type=Path)
    parser.add_argument("--postgres-cri-image-metadata", type=Path)
    parser.add_argument("--postgres-docker-image-metadata", type=Path)
    parser.add_argument("--rendered-manifest", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.replica_count < 1:
        parser.error("--replica-count must be positive")
    if args.batch_lines < 1:
        parser.error("--batch-lines must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.verify_only and (
        args.namespace != DEFAULT_NAMESPACE
        or args.replica_count != DEFAULT_REPLICA_COUNT
        or args.batch_lines != DEFAULT_BATCH_LINES
        or args.timeout_seconds != DEFAULT_TIMEOUT_SECONDS
    ):
        parser.error(
            "live F1c evidence is frozen to namespace kairyu-f1c, "
            "12 replicas, 200 batch lines, and a 360 second timeout"
        )
    if not args.verify_only and args.rendered_manifest is None:
        parser.error("--rendered-manifest is required for a live gate")
    if args.verify_only:
        verifier = verify_evidence(args.results_dir)
        (args.results_dir / "verifier.json").write_text(
            json.dumps(verifier, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not verifier["passed"]:
            failed = [
                name for name, passed in verifier["checks"].items() if not passed
            ]
            raise SystemExit(f"F1c verification failed: {failed}")
        print(json.dumps(verifier["summary"], sort_keys=True))
        return
    if any(
        path is None
        for path in (
            args.gateway_cri_image_metadata,
            args.mock_cri_image_metadata,
            args.postgres_cri_image_metadata,
            args.postgres_docker_image_metadata,
        )
    ):
        parser.error(
            "live runs require all three --*-cri-image-metadata files and "
            "--postgres-docker-image-metadata"
        )
    run_gate(args)
    print(f"F1c three-gateway gate passed: {args.results_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

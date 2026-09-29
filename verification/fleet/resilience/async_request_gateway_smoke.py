"""Live three-gateway smoke test for the durable AsyncRequest control plane."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

GATEWAY_IDS = ("a", "b", "c")
TERMINAL_STATES = ("succeeded", "failed", "cancelled", "expired")
STORE_ID = "kairyu-f1c-async"
MODEL = "async-smoke"
FAILOVER_MODEL = "async-failover-smoke"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def metric_value(text: str, name: str, **labels: str) -> float:
    """Read one Prometheus sample without depending on label output order."""
    for line in text.splitlines():
        if not line.startswith(f"{name}{{"):
            continue
        label_text, raw_value = line.rsplit("} ", 1)
        if all(f'{key}="{value}"' in label_text for key, value in labels.items()):
            return float(raw_value)
    raise KeyError(f"metric sample {name!r} with labels {labels!r} was not found")


def terminal_state_counts(text: str) -> dict[str, float]:
    """Read the terminal request gauges from one shared metrics snapshot."""
    return {
        state: metric_value(
            text,
            "kairyu_async_request_state",
            store=STORE_ID,
            state=state,
        )
        for state in TERMINAL_STATES
    }


def rank_gateways(session_id: str) -> tuple[str, ...]:
    """Match the F1c load balancer's frozen rendezvous selection."""
    return tuple(
        sorted(
            GATEWAY_IDS,
            key=lambda gateway_id: (
                hashlib.sha256(f"{session_id}:{gateway_id}".encode()).digest(),
                gateway_id,
            ),
            reverse=True,
        )
    )


def sessions_by_gateway() -> dict[str, str]:
    sessions: dict[str, str] = {}
    for index in range(10_000):
        session_id = f"async-request-smoke-{index}"
        sessions.setdefault(rank_gateways(session_id)[0], session_id)
        if len(sessions) == len(GATEWAY_IDS):
            return sessions
    raise RuntimeError("could not find a deterministic session for every gateway")


def find_takeover(
    events: list[dict[str, Any]],
    *,
    original_worker: str,
    original_fence: int,
) -> dict[str, Any] | None:
    for event in events:
        if (
            event.get("event") == "reclaim"
            and event.get("worker_id") != original_worker
            and int(event.get("fencing_token", 0)) > original_fence
        ):
            return event
    return None


class Smoke:
    def __init__(
        self,
        *,
        gateway_url: str,
        kubectl: str,
        namespace: str,
        timeout_seconds: float,
    ) -> None:
        self._gateway_url = gateway_url.rstrip("/")
        self._kubectl = kubectl
        self._namespace = namespace
        self._timeout_seconds = timeout_seconds
        self._sessions = sessions_by_gateway()
        self._client = httpx.Client(timeout=15.0)
        self.checks: list[dict[str, Any]] = []
        self.request_ids: list[str] = []

    def close(self) -> None:
        self._client.close()

    def _kubectl_run(
        self,
        *args: str,
        check: bool = True,
        timeout: float = 30.0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._kubectl, "-n", self._namespace, *args],
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        gateway_id: str,
        expected_status: int | set[int] = 200,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        verify_gateway: bool = True,
    ) -> httpx.Response:
        expected = {expected_status} if isinstance(expected_status, int) else expected_status
        request_headers = {"X-Session-ID": self._sessions[gateway_id]}
        request_headers.update(headers or {})
        response = self._client.request(
            method,
            f"{self._gateway_url}{path}",
            json=json_body,
            headers=request_headers,
        )
        if response.status_code not in expected:
            raise AssertionError(
                f"{method} {path} returned {response.status_code}: {response.text[:500]}"
            )
        selected = response.headers.get("X-Kairyu-Gateway-ID")
        if verify_gateway and selected != gateway_id:
            raise AssertionError(
                f"{method} {path} selected gateway {selected!r}, expected {gateway_id!r}"
            )
        return response

    def _submit(
        self,
        *,
        gateway_id: str,
        prompt: str,
        idempotency_key: str,
        deadline_at: datetime | None = None,
        model: str = MODEL,
    ) -> str:
        headers = {"Idempotency-Key": idempotency_key}
        if deadline_at is not None:
            headers["X-Kairyu-Deadline-At"] = deadline_at.isoformat()
        response = self._request(
            "POST",
            "/v1/async/chat/completions",
            gateway_id=gateway_id,
            expected_status=202,
            headers=headers,
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        payload = response.json()
        request_id = payload["request"]["id"]
        if not _REQUEST_ID.fullmatch(request_id):
            raise AssertionError(f"unsafe request id returned by API: {request_id!r}")
        self.request_ids.append(request_id)
        return request_id

    def _status(self, request_id: str, gateway_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/requests/{request_id}",
            gateway_id=gateway_id,
        ).json()

    def _wait_state(
        self,
        request_id: str,
        states: set[str],
        *,
        gateway_id: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_seconds
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                last = self._status(request_id, gateway_id)
            except (httpx.HTTPError, AssertionError):
                time.sleep(0.2)
                continue
            if last.get("state") in states:
                return last
            time.sleep(0.1)
        raise AssertionError(f"request {request_id} did not reach {sorted(states)}; last={last}")

    def _audit(self, request_id: str) -> list[dict[str, Any]]:
        if not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("request id is not safe for the fixture SQL query")
        sql = (
            "SELECT row_to_json(a)::text FROM ("
            "SELECT sequence, request_id, worker_id, fencing_token, "
            "event, at, lease_until, details "
            "FROM async_request_claim_audit "
            f"WHERE store_id = '{STORE_ID}' AND request_id = '{request_id}' "
            "ORDER BY sequence) AS a"
        )
        completed = self._kubectl_run(
            "exec",
            "deployment/f1c-postgres",
            "--",
            "psql",
            "postgresql://kairyu:f1c-kind-only@127.0.0.1:5432/kairyu",
            "-XAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            sql,
        )
        return [json.loads(line) for line in completed.stdout.splitlines() if line]

    def _sql_scalar(self, sql: str) -> str:
        completed = self._kubectl_run(
            "exec",
            "deployment/f1c-postgres",
            "--",
            "psql",
            "postgresql://kairyu:f1c-kind-only@127.0.0.1:5432/kairyu",
            "-XAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            sql,
        )
        return completed.stdout.strip()

    def _retention_cli(self, *args: str) -> tuple[dict[str, Any], str]:
        completed = self._kubectl_run(
            "exec",
            "deployment/f1c-gateway-a",
            "--",
            "python",
            "/app/scripts/async_request_retention.py",
            "/etc/kairyu/config.yaml",
            *args,
            timeout=120.0,
        )
        output = completed.stdout.strip()
        if not output:
            raise AssertionError("retention CLI returned no structured result")
        return json.loads(output.splitlines()[-1]), output

    def _wait_audit_event(
        self,
        request_id: str,
        event_name: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_seconds
        latest: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                latest = self._audit(request_id)
            except (subprocess.SubprocessError, json.JSONDecodeError):
                time.sleep(0.2)
                continue
            matches = [event for event in latest if event.get("event") == event_name]
            if matches:
                return matches[-1]
            time.sleep(0.1)
        raise AssertionError(f"request {request_id} has no {event_name!r} audit event: {latest}")

    def _advisory_lock_count(self, *, granted: bool) -> int:
        completed = self._kubectl_run(
            "exec",
            "deployment/f1c-postgres",
            "--",
            "psql",
            "postgresql://kairyu:f1c-kind-only@127.0.0.1:5432/kairyu",
            "-XAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            (
                "SELECT count(*) FROM pg_locks "
                f"WHERE locktype = 'advisory' AND granted IS {str(granted).upper()}"
            ),
        )
        return int(completed.stdout.strip())

    def _wait_for_advisory_lock(
        self,
        *,
        granted: bool,
        holder: subprocess.Popen[bytes],
        submission: concurrent.futures.Future[str] | None = None,
    ) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if holder.poll() is not None:
                raise AssertionError("PostgreSQL advisory-lock holder exited early")
            if submission is not None and submission.done():
                raise AssertionError(
                    "large submission completed before entering the server-side lock wait"
                )
            if self._advisory_lock_count(granted=granted) > 0:
                return
            time.sleep(0.05)
        state = "granted holder" if granted else "blocked submit waiter"
        raise AssertionError(f"did not observe the expected PostgreSQL {state}")

    def _gateway_pods(self) -> dict[str, dict[str, Any]]:
        completed = self._kubectl_run(
            "get",
            "pods",
            "-l",
            "app.kubernetes.io/name=f1c-gateway",
            "-o",
            "json",
        )
        pods: dict[str, dict[str, Any]] = {}
        for pod in json.loads(completed.stdout)["items"]:
            gateway_id = pod["metadata"]["labels"]["kairyu.ai/gateway-id"]
            pods[gateway_id] = {
                "name": pod["metadata"]["name"],
                "uid": pod["metadata"]["uid"],
            }
        if set(pods) != set(GATEWAY_IDS):
            raise AssertionError(f"expected three gateway pods, found {sorted(pods)}")
        return pods

    def shared_state_and_idempotency(self) -> None:
        request_id = self._submit(
            gateway_id="a",
            prompt="shared-state",
            idempotency_key="async-smoke-shared",
        )
        replay = self._submit(
            gateway_id="b",
            prompt="shared-state",
            idempotency_key="async-smoke-shared",
        )
        if replay != request_id:
            raise AssertionError("cross-gateway idempotent replay created a second request")
        for gateway_id in GATEWAY_IDS:
            if self._status(request_id, gateway_id)["id"] != request_id:
                raise AssertionError(f"gateway {gateway_id} did not read shared state")
        status = self._wait_state(request_id, {"succeeded"}, gateway_id="c")
        result = self._request(
            "GET",
            f"/v1/requests/{request_id}/result",
            gateway_id="a",
        ).json()
        if result.get("object") != "chat.completion" or not status.get("has_result"):
            raise AssertionError("successful durable result is not a Chat Completion")
        self.checks.append({"name": "shared_state_and_idempotency", "request_id": request_id})

    def cancellation(self) -> None:
        request_id = self._submit(
            gateway_id="a",
            prompt="cancel-me",
            idempotency_key="async-smoke-cancel",
        )
        running = self._wait_audit_event(request_id, "running")
        owner_gateway = self._gateway_for_worker(running["worker_id"])
        cancelling_gateway = next(
            gateway_id for gateway_id in GATEWAY_IDS if gateway_id != owner_gateway
        )
        response = self._request(
            "POST",
            f"/v1/requests/{request_id}/cancel",
            gateway_id=cancelling_gateway,
        ).json()
        status = self._wait_state(
            request_id,
            {"cancelled"},
            gateway_id=cancelling_gateway,
        )
        if response["state"] != "cancelled" or status["state"] != "cancelled":
            raise AssertionError("remote cancellation did not become durable")
        self._wait_audit_event(request_id, "cancel")
        self.checks.append(
            {
                "name": "cross_gateway_cancellation",
                "request_id": request_id,
                "owner_gateway": owner_gateway,
                "cancelling_gateway": cancelling_gateway,
            }
        )

    def deadline(self) -> None:
        request_id = self._submit(
            gateway_id="b",
            prompt="expire-me",
            idempotency_key="async-smoke-deadline",
            deadline_at=datetime.now(UTC) + timedelta(seconds=1.5),
        )
        status = self._wait_state(request_id, {"expired"}, gateway_id="c")
        if status["state"] != "expired":
            raise AssertionError("deadline did not produce an expired state")
        self._wait_audit_event(request_id, "expire")
        self.checks.append({"name": "durable_deadline", "request_id": request_id})

    def large_body_responsiveness(self) -> None:
        owner_lock_key = json.dumps([STORE_ID, "default"])
        lock_sql = (
            "BEGIN; SELECT pg_advisory_xact_lock(hashtextextended("
            f"'{owner_lock_key}', 0)); SELECT pg_sleep(8); COMMIT"
        )
        holder = subprocess.Popen(
            [
                self._kubectl,
                "-n",
                self._namespace,
                "exec",
                "deployment/f1c-postgres",
                "--",
                "psql",
                "postgresql://kairyu:f1c-kind-only@127.0.0.1:5432/kairyu",
                "-XAt",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                lock_sql,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            self._wait_for_advisory_lock(granted=True, holder=holder)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                submit_started = time.monotonic()
                submission = executor.submit(
                    self._submit,
                    gateway_id="c",
                    prompt="x" * (768 * 1024),
                    idempotency_key="async-smoke-large-body",
                )
                self._wait_for_advisory_lock(
                    granted=False,
                    holder=holder,
                    submission=submission,
                )
                health_started = time.monotonic()
                self._request("GET", "/v1/models", gateway_id="c")
                health_seconds = time.monotonic() - health_started
                if holder.wait(timeout=12.0) != 0:
                    raise AssertionError("PostgreSQL advisory-lock holder failed")
                request_id = submission.result(timeout=15.0)
                submit_seconds = time.monotonic() - submit_started
        finally:
            if holder.poll() is None:
                holder.terminate()
                try:
                    holder.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    holder.kill()
                    holder.wait(timeout=3.0)
        self._request(
            "POST",
            f"/v1/requests/{request_id}/cancel",
            gateway_id="a",
        )
        self._wait_state(request_id, {"cancelled"}, gateway_id="b")
        if health_seconds >= 5.0:
            raise AssertionError(
                "large durable submission blocked gateway responsiveness: "
                f"submit={submit_seconds:.3f}s health={health_seconds:.3f}s"
            )
        self.checks.append(
            {
                "name": "large_body_responsiveness",
                "request_id": request_id,
                "submit_seconds": round(submit_seconds, 6),
                "models_seconds": round(health_seconds, 6),
            }
        )

    def _gateway_for_worker(self, worker_id: str) -> str:
        pods = self._gateway_pods()
        for gateway_id, pod in pods.items():
            if pod["uid"] == worker_id:
                return gateway_id
        raise AssertionError(f"worker {worker_id!r} is not a current gateway Pod UID")

    def owner_failover(self) -> None:
        request_id = self._submit(
            gateway_id="a",
            prompt="survive-owner-loss",
            idempotency_key="async-smoke-failover",
            model=FAILOVER_MODEL,
        )
        running = self._wait_audit_event(request_id, "running")
        original_worker = running["worker_id"]
        original_fence = int(running["fencing_token"])
        owner_gateway = self._gateway_for_worker(original_worker)
        deployment = f"deployment/f1c-gateway-{owner_gateway}"
        survivor = next(gateway_id for gateway_id in GATEWAY_IDS if gateway_id != owner_gateway)
        self._kubectl_run("scale", deployment, "--replicas=0")
        restored = False
        try:
            deadline = time.monotonic() + self._timeout_seconds
            takeover: dict[str, Any] | None = None
            latest: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                latest = self._audit(request_id)
                takeover = find_takeover(
                    latest,
                    original_worker=original_worker,
                    original_fence=original_fence,
                )
                if takeover is not None:
                    break
                time.sleep(0.2)
            if takeover is None:
                raise AssertionError(f"no fenced takeover was observed: {latest}")
            status = self._wait_state(request_id, {"succeeded"}, gateway_id=survivor)
            if int(status["attempt"]) < 2:
                raise AssertionError("takeover did not increment the public attempt")
            self._request(
                "GET",
                f"/v1/requests/{request_id}/result",
                gateway_id=survivor,
            )
            self._kubectl_run("scale", deployment, "--replicas=1")
            self._kubectl_run(
                "rollout",
                "status",
                deployment,
                "--timeout=120s",
                timeout=130.0,
            )
            restored = True
            self.checks.append(
                {
                    "name": "lease_fenced_owner_failover",
                    "request_id": request_id,
                    "failed_gateway": owner_gateway,
                    "original_worker": original_worker,
                    "takeover_worker": takeover["worker_id"],
                    "original_fence": original_fence,
                    "takeover_fence": int(takeover["fencing_token"]),
                }
            )
        finally:
            if not restored:
                self._kubectl_run("scale", deployment, "--replicas=1", check=False)
                self._kubectl_run(
                    "rollout",
                    "status",
                    deployment,
                    "--timeout=120s",
                    check=False,
                    timeout=130.0,
                )

    def database_reconnect(self) -> None:
        durable_request = self.checks[0]["request_id"]
        completed = self._kubectl_run(
            "get",
            "pod",
            "-l",
            "app.kubernetes.io/name=f1c-postgres",
            "-o",
            "json",
        )
        pod = json.loads(completed.stdout)["items"][0]
        pod_name = pod["metadata"]["name"]
        old_restarts = int(pod["status"]["containerStatuses"][0]["restartCount"])
        self._kubectl_run(
            "exec",
            pod_name,
            "--",
            "pg_ctl",
            "-D",
            "/var/lib/postgresql/data/pgdata",
            "stop",
            "-m",
            "fast",
            "-w",
            check=False,
        )
        deadline = time.monotonic() + self._timeout_seconds
        new_restarts = old_restarts
        ready = False
        while time.monotonic() < deadline:
            try:
                current = json.loads(self._kubectl_run("get", "pod", pod_name, "-o", "json").stdout)
                container = current["status"]["containerStatuses"][0]
                new_restarts = int(container["restartCount"])
                ready = bool(container.get("ready"))
                if new_restarts > old_restarts and ready:
                    break
            except (subprocess.SubprocessError, KeyError, IndexError, json.JSONDecodeError):
                pass
            time.sleep(0.5)
        if new_restarts <= old_restarts or not ready:
            raise AssertionError(
                f"PostgreSQL did not restart cleanly: {old_restarts} -> {new_restarts}"
            )
        for gateway_id in GATEWAY_IDS:
            status = self._wait_state(
                durable_request,
                {"succeeded"},
                gateway_id=gateway_id,
            )
            if not status["has_result"]:
                raise AssertionError(
                    f"gateway {gateway_id} lost the result after PostgreSQL restart"
                )
            result = self._request(
                "GET",
                f"/v1/requests/{durable_request}/result",
                gateway_id=gateway_id,
            ).json()
            if result.get("object") != "chat.completion":
                raise AssertionError(
                    f"gateway {gateway_id} returned an invalid result after reconnect"
                )
        reconnect_request: str | None = None
        submit_deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < submit_deadline:
            try:
                reconnect_request = self._submit(
                    gateway_id="a",
                    prompt="post-reconnect-work",
                    idempotency_key="async-smoke-postgres-reconnect",
                )
                break
            except (httpx.HTTPError, AssertionError):
                time.sleep(0.2)
        if reconnect_request is None:
            raise AssertionError("gateway did not recover async submission after DB restart")
        self._wait_audit_event(reconnect_request, "running")
        self._wait_state(reconnect_request, {"succeeded"}, gateway_id="b")
        reconnect_result = self._request(
            "GET",
            f"/v1/requests/{reconnect_request}/result",
            gateway_id="c",
        ).json()
        if reconnect_result.get("object") != "chat.completion":
            raise AssertionError("worker did not publish a result after DB reconnect")
        self.checks.append(
            {
                "name": "postgres_reconnect_and_persistence",
                "request_id": durable_request,
                "post_reconnect_request_id": reconnect_request,
                "restart_count_before": old_restarts,
                "restart_count_after": new_restarts,
            }
        )

    def shared_queue_telemetry(self) -> None:
        deadline = time.monotonic() + self._timeout_seconds
        snapshots: dict[str, str] = {}
        selected: dict[str, tuple[str, ...]] = {}
        while time.monotonic() < deadline:
            snapshots = {}
            for gateway_id in GATEWAY_IDS:
                text = self._request("GET", "/metrics", gateway_id=gateway_id).text
                if (
                    metric_value(
                        text,
                        "kairyu_async_request_metrics_snapshot_success",
                        store=STORE_ID,
                    )
                    == 1
                ):
                    snapshots[gateway_id] = text
            if len(snapshots) == len(GATEWAY_IDS):
                selected = {
                    gateway_id: tuple(
                        sorted(
                            line
                            for line in text.splitlines()
                            if line.startswith("kairyu_async_request_")
                        )
                    )
                    for gateway_id, text in snapshots.items()
                }
                if len(set(selected.values())) == 1:
                    break
            time.sleep(0.2)
        else:
            raise AssertionError("gateways exposed inconsistent shared queue metrics")

        reference = snapshots["a"]
        if (
            metric_value(
                reference,
                "kairyu_async_request_queue_depth",
                store=STORE_ID,
            )
            != 0
        ):
            raise AssertionError("completed smoke left durable work queued")
        minimums = {
            ("kairyu_async_request_transitions_total", "reclaim"): 1,
            ("kairyu_async_request_transitions_total", "expire"): 1,
            ("kairyu_async_request_transitions_total", "cancel"): 2,
            ("kairyu_async_request_transitions_total", "succeed"): 3,
        }
        for (name, event), minimum in minimums.items():
            observed = metric_value(reference, name, store=STORE_ID, event=event)
            if observed < minimum:
                raise AssertionError(
                    f"{name} event={event!r} was {observed}, expected >= {minimum}"
                )
        attempts = metric_value(
            reference,
            "kairyu_async_request_attempts_total",
            store=STORE_ID,
        )
        if attempts < 6:
            raise AssertionError(f"attempt counter was {attempts}, expected >= 6")
        for request_id in self.request_ids:
            if request_id in reference:
                raise AssertionError("request ID leaked into Prometheus labels")
        for prompt in ("shared-state", "cancel-me", "expire-me", "survive-owner-loss"):
            if prompt in reference:
                raise AssertionError("request prompt leaked into Prometheus labels")
        self.checks.append(
            {
                "name": "shared_low_cardinality_queue_telemetry",
                "gateways": list(GATEWAY_IDS),
                "attempts_total": attempts,
            }
        )

    def retention_and_audit_archive(self) -> None:
        request_id = str(self.checks[0]["request_id"])
        before_terminal_count = int(
            self._sql_scalar(
                "SELECT COALESCE(sum(total), 0) "
                "FROM async_request_state_shards "
                f"WHERE store_id = '{STORE_ID}' "
                "AND state IN ('succeeded', 'failed', 'cancelled', 'expired')"
            )
        )
        before_succeeded = int(
            self._sql_scalar(
                "SELECT COALESCE(sum(total), 0) "
                "FROM async_request_state_shards "
                f"WHERE store_id = '{STORE_ID}' AND state = 'succeeded'"
            )
        )
        before_transitions = int(
            self._sql_scalar(
                "SELECT COALESCE(sum(total), 0) "
                "FROM async_request_metric_shards "
                f"WHERE store_id = '{STORE_ID}' AND event = 'succeed'"
            )
        )
        self._retention_cli("--mode", "prepare", "--apply")
        safe_request_id = request_id.replace("'", "''")
        self._sql_scalar(
            "UPDATE async_requests "
            "SET completed_at = clock_timestamp() - interval '2 hours', "
            "updated_at = clock_timestamp() - interval '2 hours' "
            f"WHERE store_id = '{STORE_ID}' AND request_id = '{safe_request_id}'; "
            "UPDATE async_request_claim_audit "
            "SET at = clock_timestamp() - interval '2 hours' "
            f"WHERE store_id = '{STORE_ID}' AND request_id = '{safe_request_id}'"
        )
        preview, preview_output = self._retention_cli("--mode", "purge")
        if preview.get("applied") is not False:
            raise AssertionError("retention preview unexpectedly mutated data")
        preview_work = sum(
            int(preview.get(field, 0))
            for field in (
                "terminal_requests_deleted",
                "audit_events_archived",
                "audit_events_deleted",
                "owner_deferrals_deleted",
            )
        )
        if preview_work < 1 and preview.get("has_more") is not True:
            raise AssertionError(f"retention preview found no eligible work: {preview}")
        self._status(request_id, "b")
        applied, applied_output = self._retention_cli(
            "--mode",
            "purge",
            "--apply",
            "--max-batches",
            "10",
        )
        deleted_requests = int(applied.get("terminal_requests_deleted", 0))
        if deleted_requests < 1:
            raise AssertionError(f"retention did not delete the target request: {applied}")
        archived = int(
            self._sql_scalar(
                "SELECT count(*) FROM async_request_claim_audit_archive "
                f"WHERE store_id = '{STORE_ID}' "
                f"AND request_id = '{safe_request_id}' AND owner = 'default'"
            )
        )
        if archived <= 0:
            raise AssertionError("request audit did not survive payload retention")
        for gateway_id in GATEWAY_IDS:
            self._request(
                "GET",
                f"/v1/requests/{request_id}",
                gateway_id=gateway_id,
                expected_status=404,
            )
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            current = self._request("GET", "/metrics", gateway_id="c").text
            current_terminal = terminal_state_counts(current)
            snapshot_success = metric_value(
                current,
                "kairyu_async_request_metrics_snapshot_success",
                store=STORE_ID,
            )
            current_transitions = metric_value(
                current,
                "kairyu_async_request_transitions_total",
                store=STORE_ID,
                event="succeed",
            )
            if (
                snapshot_success == 1
                and sum(current_terminal.values()) == before_terminal_count - deleted_requests
                and current_terminal["succeeded"] <= before_succeeded - 1
                and current_transitions == before_transitions
            ):
                break
            time.sleep(0.2)
        else:
            raise AssertionError("retention metrics did not converge")
        self._sql_scalar(
            "UPDATE async_request_claim_audit_archive "
            "SET at = clock_timestamp() - interval '2 days' "
            f"WHERE store_id = '{STORE_ID}' AND request_id = '{safe_request_id}'"
        )
        audit_purge, audit_output = self._retention_cli(
            "--mode",
            "purge",
            "--apply",
            "--max-batches",
            "10",
        )
        if int(audit_purge.get("audit_events_deleted", 0)) != archived:
            raise AssertionError("audit TTL did not remove the archived events")
        if any(
            request_id in output or "shared-state" in output
            for output in (preview_output, applied_output, audit_output)
        ):
            raise AssertionError("retention output leaked request data")
        self.checks.append(
            {
                "name": "bounded_retention_and_independent_audit_ttl",
                "archived_audit_events": archived,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:18082")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--namespace", default="kairyu-f1c")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/async-request-kind-live/report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _COMMIT.fullmatch(args.source_commit):
        raise SystemExit("--source-commit must be a full lowercase Git SHA-1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    smoke = Smoke(
        gateway_url=args.gateway_url,
        kubectl=args.kubectl,
        namespace=args.namespace,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        smoke.shared_state_and_idempotency()
        smoke.cancellation()
        smoke.deadline()
        smoke.large_body_responsiveness()
        smoke.owner_failover()
        smoke.database_reconnect()
        smoke.shared_queue_telemetry()
        smoke.retention_and_audit_archive()
        report = {
            "schema_version": 1,
            "gate": "async-request-three-gateway-smoke",
            "verdict": "pass",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "gateway_url": args.gateway_url,
            "namespace": args.namespace,
            "models": [MODEL, FAILOVER_MODEL],
            "store_id": STORE_ID,
            "source_commit": args.source_commit,
            "sessions_by_gateway": smoke._sessions,
            "checks": smoke.checks,
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"AsyncRequest three-gateway smoke passed: {args.output}")
        return 0
    except Exception as error:
        report = {
            "schema_version": 1,
            "gate": "async-request-three-gateway-smoke",
            "verdict": "fail",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "source_commit": args.source_commit,
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "checks": smoke.checks,
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        smoke.close()


if __name__ == "__main__":
    raise SystemExit(main())

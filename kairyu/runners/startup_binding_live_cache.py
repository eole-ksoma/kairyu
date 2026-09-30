"""Authenticated node-agent aggregation for live placement-binding evidence."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kairyu.runners.cache_agent_live_evidence import (
    NodeModelCacheLiveEvidenceRequest,
    NodeModelCacheLiveEvidenceResponse,
)
from kairyu.runners.prewarm import (
    ModelCachePlacementCandidate,
    build_cache_placement_snapshot,
    plan_cache_aware_scale_up,
)
from kairyu.runners.scaling_log import ScalingDecisionRecord
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorizationDeniedError,
    RunnerCachePlacementBindingAuthorizationError,
    validate_runner_cache_placement_bearer_token,
)
from kairyu.runners.startup_binding_live_source import (
    RunnerCachePlacementBindingCacheState,
)

_MAX_SIGNED_BIGINT = 2**63 - 1


def _text(value: str, *, name: str, max_length: int = 255) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > max_length
    ):
        raise ValueError(f"{name} must be a bounded non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _validate_budget(deadline_monotonic: float, backend_timeout_s: float) -> None:
    for name, value in (
        ("deadline_monotonic", deadline_monotonic),
        ("backend_timeout_s", backend_timeout_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{name} must be a finite number")
    if backend_timeout_s <= 0:
        raise ValueError("backend_timeout_s must be positive")


async def _read_response(
    response: httpx.Response,
    *,
    max_bytes: int,
    deadline_monotonic: float,
    monotonic_clock,
) -> bytes:
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent returned an invalid response"
            ) from exc
        if content_length < 0 or content_length > max_bytes:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent response exceeds the configured limit"
            )
    chunks: list[bytes] = []
    received = 0
    async for chunk in response.aiter_bytes():
        if monotonic_clock() >= deadline_monotonic:
            raise TimeoutError("node cache agent response deadline expired")
        received += len(chunk)
        if received > max_bytes:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent response exceeds the configured limit"
            )
        chunks.append(chunk)
    if monotonic_clock() >= deadline_monotonic:
        raise TimeoutError("node cache agent response deadline expired")
    return b"".join(chunks)


def _json_model(raw: bytes, model):
    value = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    return model.model_validate(value)  # type: ignore[attr-defined,no-any-return]


async def _bounded_async_map(values: tuple, call, *, max_workers: int) -> tuple:
    """Run at most ``max_workers`` tasks without materializing one task per value."""

    iterator = iter(enumerate(values))
    results = [None] * len(values)

    async def worker() -> None:
        while True:
            try:
                index, value = next(iterator)
            except StopIteration:
                return
            results[index] = await call(value)

    tasks = tuple(asyncio.create_task(worker()) for _ in range(min(max_workers, len(values))))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return tuple(results)


class NodeModelCacheAgentEndpoint(BaseModel):
    """Trusted controller configuration for one node-agent HTTPS origin."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    node_id: str = Field(max_length=253)
    base_url: str = Field(max_length=2048)

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        return _text(value, name="node_id", max_length=253)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = _text(value, name="base_url", max_length=2048)
        if any(character.isspace() or ord(character) < 0x20 for character in value):
            raise ValueError("base_url must not contain whitespace or controls")
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("base_url contains an invalid port") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError("base_url must be one credential-free HTTPS origin")
        return value.rstrip("/")

    @property
    def evidence_url(self) -> str:
        return f"{self.base_url}/v1/cache/live-evidence"

    @property
    def readiness_url(self) -> str:
        return f"{self.base_url}/readyz"


@runtime_checkable
class NodeModelCacheLiveEvidenceReader(Protocol):
    """Deadline-aware controller transport for configured node agents."""

    def read(
        self,
        request: NodeModelCacheLiveEvidenceRequest,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> NodeModelCacheLiveEvidenceResponse: ...

    def read_many(
        self,
        requests: tuple[NodeModelCacheLiveEvidenceRequest, ...],
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> tuple[NodeModelCacheLiveEvidenceResponse, ...]: ...

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None: ...


class AuthenticatedNodeModelCacheLiveEvidenceClient:
    """Strict bounded client for the D3.13 node-agent endpoint."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        endpoints: tuple[NodeModelCacheAgentEndpoint, ...],
        bearer_token: str,
        request_limit_bytes: int = 1024 * 1024,
        response_limit_bytes: int = 1024 * 1024,
        max_parallel_requests: int = 16,
        monotonic_clock=time.monotonic,
    ) -> None:
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("client must be an httpx.AsyncClient")
        if client.follow_redirects:
            raise ValueError("node cache agent client must not follow redirects")
        if client.trust_env:
            raise ValueError("node cache agent client must disable environment trust")
        if not isinstance(endpoints, tuple) or not endpoints:
            raise TypeError("endpoints must be a non-empty tuple")
        if len(endpoints) > 100_000:
            raise ValueError("endpoints exceed maximum node count")
        validated = tuple(
            NodeModelCacheAgentEndpoint.model_validate(endpoint.model_dump())
            if isinstance(endpoint, NodeModelCacheAgentEndpoint)
            else None
            for endpoint in endpoints
        )
        if any(endpoint is None for endpoint in validated):
            raise TypeError("endpoints must contain NodeModelCacheAgentEndpoint values")
        values = tuple(endpoint for endpoint in validated if endpoint is not None)
        node_ids = tuple(endpoint.node_id for endpoint in values)
        if node_ids != tuple(sorted(node_ids)) or len(set(node_ids)) != len(node_ids):
            raise ValueError("endpoints must use unique canonical node IDs")
        for name, value in (
            ("request_limit_bytes", request_limit_bytes),
            ("response_limit_bytes", response_limit_bytes),
        ):
            if type(value) is not int or not 1 <= value <= 16 * 1024 * 1024:
                raise ValueError(f"{name} must be an integer in [1, 16777216]")
        if type(max_parallel_requests) is not int or not 1 <= max_parallel_requests <= 256:
            raise ValueError("max_parallel_requests must be an integer in [1, 256]")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._client = client
        self._endpoints = {endpoint.node_id: endpoint for endpoint in values}
        self._bearer_token = validate_runner_cache_placement_bearer_token(bearer_token)
        self._request_limit_bytes = request_limit_bytes
        self._response_limit_bytes = response_limit_bytes
        self._max_parallel_requests = max_parallel_requests
        self._monotonic_clock = monotonic_clock
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="kairyu-node-cache-live-evidence",
            daemon=True,
        )
        self._closed = False
        self._close_lock = threading.Lock()
        self._inflight = set()
        self._loop_thread.start()

    def _remaining_timeout(
        self,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> float:
        remaining = deadline_monotonic - self._monotonic_clock()
        if remaining <= 0:
            raise TimeoutError("node cache evidence deadline expired")
        return min(backend_timeout_s, remaining)

    def _run(self, coroutine):
        with self._close_lock:
            if self._closed:
                coroutine.close()
                raise RunnerCachePlacementBindingAuthorizationError(
                    "node cache agent client is closed"
                )
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            self._inflight.add(future)
        try:
            return future.result()
        finally:
            with self._close_lock:
                self._inflight.discard(future)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._bearer_token}",
            "Accept": "application/json",
            "Cache-Control": "no-store",
        }

    @staticmethod
    def _require_json(response: httpx.Response) -> None:
        if response.headers.get("content-type", "").partition(";")[0].lower() != (
            "application/json"
        ):
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent returned an invalid response"
            )

    @staticmethod
    def _validate_response_binding(
        request: NodeModelCacheLiveEvidenceRequest,
        response: NodeModelCacheLiveEvidenceResponse,
    ) -> None:
        placement = request.placement
        record = response.prestage_record
        command = record.command
        hint = response.placement_hint
        pin = response.pin_evidence
        resident = hint.resident_for(
            manifest_digest=placement.manifest_digest,
            model_id=request.model_id,
            model_revision=request.model_revision,
        )
        if (
            response.node_id != placement.node_name
            or command.placement_id != placement.placement_id
            or command.node_id != placement.node_name
            or command.command_id != placement.prestage_command_id
            or command.command_generation != placement.prestage_command_generation
            or command.pin_owner != placement.pin_owner
            or command.manifest_digest != placement.manifest_digest
            or command.model_id != request.model_id
            or command.model_revision != request.model_revision
            or record.pin_record_generation != placement.resident_record_generation
            or hint.node_id != placement.node_name
            or hint.index_revision < placement.hint_index_revision
            or hint.observed_at < placement.hint_observed_at
            or len(hint.residents) != 1
            or resident is None
            or resident.record_generation != placement.resident_record_generation
            or not resident.pinned
            or pin.node_id != placement.node_name
            or pin.index_revision != hint.index_revision
            or pin.observed_at != hint.observed_at
            or pin.manifest_digest != placement.manifest_digest
            or pin.model_id != request.model_id
            or pin.model_revision != request.model_revision
            or pin.record_generation != placement.resident_record_generation
            or placement.pin_owner not in pin.pin_owners
            or record.updated_at > hint.observed_at
        ):
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "node cache evidence does not match the requested placement"
            )

    async def _read_async(
        self,
        request: NodeModelCacheLiveEvidenceRequest,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> NodeModelCacheLiveEvidenceResponse:
        endpoint = self._endpoints.get(request.placement.node_name)
        if endpoint is None:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent endpoint is not configured"
            )
        body = request.model_dump_json().encode("utf-8")
        if len(body) > self._request_limit_bytes:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache evidence request exceeds the configured limit"
            )
        remaining = self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        try:
            async with asyncio.timeout(remaining):
                async with self._client.stream(
                    "POST",
                    endpoint.evidence_url,
                    content=body,
                    headers={**self._headers, "Content-Type": "application/json"},
                    timeout=min(backend_timeout_s, remaining),
                ) as response:
                    if response.status_code == 409:
                        raise RunnerCachePlacementBindingAuthorizationDeniedError(
                            "node cache agent rejected the requested placement"
                        )
                    if response.status_code != 200:
                        raise RunnerCachePlacementBindingAuthorizationError(
                            "node cache agent rejected live evidence"
                        )
                    self._require_json(response)
                    cache_control = {
                        token.strip().lower()
                        for token in response.headers.get("cache-control", "").split(",")
                    }
                    if "no-store" not in cache_control:
                        raise RunnerCachePlacementBindingAuthorizationError(
                            "node cache agent response is not marked no-store"
                        )
                    raw = await _read_response(
                        response,
                        max_bytes=self._response_limit_bytes,
                        deadline_monotonic=deadline_monotonic,
                        monotonic_clock=self._monotonic_clock,
                    )
                result = _json_model(raw, NodeModelCacheLiveEvidenceResponse)
        except RunnerCachePlacementBindingAuthorizationError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise TimeoutError("node cache evidence request timed out") from exc
        except (
            httpx.HTTPError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent is unavailable or returned an invalid response"
            ) from exc
        self._validate_response_binding(request, result)
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return result

    def read(
        self,
        request: NodeModelCacheLiveEvidenceRequest,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> NodeModelCacheLiveEvidenceResponse:
        return self.read_many(
            (request,),
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )[0]

    async def _read_many_async(
        self,
        requests: tuple[NodeModelCacheLiveEvidenceRequest, ...],
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> tuple[NodeModelCacheLiveEvidenceResponse, ...]:
        remaining = self._remaining_timeout(deadline_monotonic, backend_timeout_s)

        async def read_one(request: NodeModelCacheLiveEvidenceRequest):
            return await self._read_async(
                request,
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )

        try:
            async with asyncio.timeout(remaining):
                return await _bounded_async_map(
                    requests,
                    read_one,
                    max_workers=self._max_parallel_requests,
                )
        except RunnerCachePlacementBindingAuthorizationError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise TimeoutError("node cache evidence aggregation deadline expired") from exc

    def read_many(
        self,
        requests: tuple[NodeModelCacheLiveEvidenceRequest, ...],
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> tuple[NodeModelCacheLiveEvidenceResponse, ...]:
        if not isinstance(requests, tuple) or not requests:
            raise TypeError("requests must be a non-empty tuple")
        if len(requests) > 100_000:
            raise ValueError("requests exceed maximum placement count")
        validated = []
        for request in requests:
            if not isinstance(request, NodeModelCacheLiveEvidenceRequest):
                raise TypeError("requests must contain NodeModelCacheLiveEvidenceRequest values")
            validated.append(NodeModelCacheLiveEvidenceRequest.model_validate(request.model_dump()))
        _validate_budget(deadline_monotonic, backend_timeout_s)
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return self._run(
            self._read_many_async(
                tuple(validated),
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )
        )

    async def _check_ready_async(
        self,
        endpoint: NodeModelCacheAgentEndpoint,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        remaining = self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        try:
            async with asyncio.timeout(remaining):
                async with self._client.stream(
                    "GET",
                    endpoint.readiness_url,
                    headers=self._headers,
                    timeout=min(backend_timeout_s, remaining),
                ) as response:
                    if response.status_code != 200:
                        raise RunnerCachePlacementBindingAuthorizationError(
                            "node cache agent is not ready"
                        )
                    self._require_json(response)
                    raw = await _read_response(
                        response,
                        max_bytes=8192,
                        deadline_monotonic=deadline_monotonic,
                        monotonic_clock=self._monotonic_clock,
                    )
                payload = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                    parse_float=_finite_float,
                )
                if payload != {"status": "ready", "node_id": endpoint.node_id}:
                    raise RunnerCachePlacementBindingAuthorizationError(
                        "node cache agent returned invalid readiness"
                    )
        except RunnerCachePlacementBindingAuthorizationError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise TimeoutError("node cache agent readiness timed out") from exc
        except (
            httpx.HTTPError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise RunnerCachePlacementBindingAuthorizationError(
                "node cache agent readiness is unavailable"
            ) from exc
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        _validate_budget(deadline_monotonic, backend_timeout_s)
        endpoints = tuple(self._endpoints[node_id] for node_id in sorted(self._endpoints))
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        self._run(
            self._readiness_async(
                endpoints,
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )
        )

    async def _readiness_async(
        self,
        endpoints: tuple[NodeModelCacheAgentEndpoint, ...],
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        remaining = self._remaining_timeout(deadline_monotonic, backend_timeout_s)

        async def check_one(endpoint: NodeModelCacheAgentEndpoint) -> None:
            await self._check_ready_async(
                endpoint,
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )

        try:
            async with asyncio.timeout(remaining):
                await _bounded_async_map(
                    endpoints,
                    check_one,
                    max_workers=self._max_parallel_requests,
                )
        except RunnerCachePlacementBindingAuthorizationError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise TimeoutError("node cache agent readiness deadline expired") from exc

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            inflight = tuple(self._inflight)
        for request in inflight:
            request.cancel()
        for request in inflight:
            try:
                request.result()
            except BaseException:
                pass
        future = asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop)
        try:
            future.result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
            self._loop.close()


class RunnerCachePlacementBindingInventory(BaseModel):
    """Fresh controller-owned placement facts observed after node evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-cache-placement-binding-inventory-v1"] = (
        "runner-cache-placement-binding-inventory-v1"
    )
    snapshot_id: str = Field(max_length=255)
    cache_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    observed_at: datetime
    candidates: tuple[ModelCachePlacementCandidate, ...] = Field(
        min_length=1,
        max_length=100_000,
    )

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _text(value, name="snapshot_id")

    @field_validator("cache_revision", mode="before")
    @classmethod
    def validate_cache_revision(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("cache_revision must be an integer")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_candidates(self) -> RunnerCachePlacementBindingInventory:
        placement_ids = tuple(candidate.placement_id for candidate in self.candidates)
        if placement_ids != tuple(sorted(placement_ids)) or len(set(placement_ids)) != len(
            placement_ids
        ):
            raise ValueError("inventory candidates must use unique canonical placement IDs")
        node_ids = tuple(candidate.node_name for candidate in self.candidates)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("inventory candidates must use unique nodes")
        return self


@runtime_checkable
class RunnerCachePlacementBindingInventoryReader(Protocol):
    """Read current scheduler placement facts after physical cache evidence."""

    def read_inventory(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingInventory: ...

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None: ...


class AggregatingRunnerCachePlacementBindingCacheReader:
    """Join exact per-node evidence to freshly observed placement inventory."""

    def __init__(
        self,
        *,
        evidence: NodeModelCacheLiveEvidenceReader,
        inventory: RunnerCachePlacementBindingInventoryReader,
        monotonic_clock=time.monotonic,
    ) -> None:
        if not isinstance(evidence, NodeModelCacheLiveEvidenceReader):
            raise TypeError("evidence must implement NodeModelCacheLiveEvidenceReader")
        if not isinstance(inventory, RunnerCachePlacementBindingInventoryReader):
            raise TypeError("inventory must implement RunnerCachePlacementBindingInventoryReader")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._evidence = evidence
        self._inventory = inventory
        self._monotonic_clock = monotonic_clock

    def _remaining_timeout(
        self,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> float:
        remaining = deadline_monotonic - self._monotonic_clock()
        if remaining <= 0:
            raise TimeoutError("cache evidence aggregation deadline expired")
        return min(backend_timeout_s, remaining)

    def read_cache(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingCacheState:
        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        _validate_budget(deadline_monotonic, backend_timeout_s)
        original = decision.prewarm_plan
        if original is None:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "binding decision has no prewarm authority"
            )
        expected_placements = tuple(placement.placement_id for placement in binding.placements)
        if expected_placements != original.runner_start_placement_ids:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "binding placements do not match the prewarm authority"
            )
        requests = tuple(
            NodeModelCacheLiveEvidenceRequest(
                placement=placement,
                model_id=binding.model_id,
                model_revision=binding.model_revision,
            )
            for placement in binding.placements
        )
        responses = self._evidence.read_many(
            requests,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        for response in responses:
            if not isinstance(response, NodeModelCacheLiveEvidenceResponse):
                raise TypeError("evidence reader returned an invalid response")
        responses = tuple(
            NodeModelCacheLiveEvidenceResponse.model_validate(response.model_dump())
            for response in responses
        )

        inventory = self._inventory.read_inventory(
            binding,
            decision,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=self._remaining_timeout(
                deadline_monotonic,
                backend_timeout_s,
            ),
        )
        if not isinstance(inventory, RunnerCachePlacementBindingInventory):
            raise TypeError("inventory reader returned an invalid inventory")
        inventory = RunnerCachePlacementBindingInventory.model_validate(inventory.model_dump())
        candidate_ids = tuple(candidate.placement_id for candidate in inventory.candidates)
        hints = tuple(response.placement_hint for response in responses)
        if (
            candidate_ids != expected_placements
            or inventory.cache_revision < original.snapshot.cache_revision
            or any(
                hint.observed_at > inventory.observed_at
                or inventory.observed_at >= hint.valid_until
                for hint in hints
            )
        ):
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "current placement inventory does not cover live node evidence"
            )
        snapshot = build_cache_placement_snapshot(
            tuple(sorted(hints, key=lambda hint: hint.node_id)),
            inventory.candidates,
            snapshot_id=inventory.snapshot_id,
            cache_revision=inventory.cache_revision,
            observed_at=inventory.observed_at,
            model_class=binding.model_class,
            model_id=binding.model_id,
            model_revision=binding.model_revision,
            artifact_digest=binding.manifest_digest,
            placement_binding_id=binding.placement_binding_id,
        )
        plan = plan_cache_aware_scale_up(
            snapshot,
            current_replicas=original.current_replicas,
            quota_target_replicas=original.quota_target_replicas,
            resource_flavor=original.resource_flavor,
        )
        if plan.runner_start_placement_ids != expected_placements:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "live cache evidence no longer supports the bound placements"
            )
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return RunnerCachePlacementBindingCacheState(
            prewarm_plan=plan,
            prestage_records=tuple(
                sorted(
                    (response.prestage_record for response in responses),
                    key=lambda record: record.command.placement_id,
                )
            ),
            placement_hints=tuple(sorted(hints, key=lambda hint: hint.node_id)),
            pin_evidence=tuple(
                sorted(
                    (response.pin_evidence for response in responses),
                    key=lambda evidence: evidence.node_id,
                )
            ),
        )

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        _validate_budget(deadline_monotonic, backend_timeout_s)
        self._evidence.readiness(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=self._remaining_timeout(
                deadline_monotonic,
                backend_timeout_s,
            ),
        )
        self._inventory.readiness(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=self._remaining_timeout(
                deadline_monotonic,
                backend_timeout_s,
            ),
        )
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)


__all__ = [
    "AggregatingRunnerCachePlacementBindingCacheReader",
    "AuthenticatedNodeModelCacheLiveEvidenceClient",
    "NodeModelCacheAgentEndpoint",
    "NodeModelCacheLiveEvidenceReader",
    "RunnerCachePlacementBindingInventory",
    "RunnerCachePlacementBindingInventoryReader",
]

"""Multi-tenant limits + usage metering (m11 D3).

Identity: AuthMiddleware stores the matched key in scope state (A6);
``TenantLimitMiddleware`` maps key → tenant and enforces a per-tenant token
bucket on /v1/* — it runs INSIDE auth (401 wins over 429; unauthenticated
requests never drain buckets). Keyless mode maps everything to "default".

Ledger (A7): O_APPEND single-writer JSONL, one record per successful public
execution, written from handlers or the bounded batch consumer (middleware
cannot see token usage). Counts use backend usage when present and otherwise
the same derived approximation returned on the wire.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from kairyu.audit_io import BoundedJsonlWriter
from kairyu.entrypoints.server.messages_protocol import (
    anthropic_error_payload,
    wants_anthropic_envelope,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TenantLimits:
    requests_per_minute: int = 600
    tokens_per_minute: int = 200_000
    # Keep the original four positional parameters stable for Python callers.
    interactive_priority: int = 0
    batch_priority: int = 1
    request_burst: int | None = None
    token_burst: int | None = None
    max_in_flight: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("requests_per_minute", self.requests_per_minute),
            ("tokens_per_minute", self.tokens_per_minute),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("request_burst", self.request_burst),
            ("token_burst", self.token_burst),
            ("max_in_flight", self.max_in_flight),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer or None")
            if value < 1:
                raise ValueError(f"{name} must be positive when configured")
        minimum, maximum = -(2**63), 2**63 - 1
        for name, value in (
            ("interactive_priority", self.interactive_priority),
            ("batch_priority", self.batch_priority),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must fit signed int64")
        if self.interactive_priority >= self.batch_priority:
            raise ValueError(
                "interactive_priority must be smaller than batch_priority"
            )


@dataclass(frozen=True)
class TenantConfig:
    """key -> tenant mapping plus per-tenant limits."""

    key_tenants: dict[str, str] = field(default_factory=dict, repr=False)
    limits: dict[str, TenantLimits] = field(default_factory=dict)
    default_tenant: str = "default"

    @classmethod
    def from_mapping(
        cls,
        *,
        key_tenants: Mapping[str, str],
        resolved_api_keys: Collection[str],
        limits: Mapping[str, TenantLimits] | None = None,
        default_tenant: str = "default",
    ) -> TenantConfig:
        """Validate deployment mappings against already-resolved API keys."""
        copied_key_tenants = dict(key_tenants)
        copied_limits = dict(limits or {})
        if isinstance(resolved_api_keys, str):
            raise ValueError("resolved API keys must not be a string")
        resolved_keys = frozenset(resolved_api_keys)

        if not default_tenant.strip():
            raise ValueError("default tenant must not be empty")
        for api_key, tenant in copied_key_tenants.items():
            if not api_key.strip():
                raise ValueError("tenant mapping key must not be empty")
            if not tenant.strip():
                raise ValueError("tenant name must not be empty")
            if api_key not in resolved_keys:
                raise ValueError(f"tenant mapping references unknown API key {api_key!r}")

        known_tenants = {*copied_key_tenants.values(), default_tenant}
        for tenant in copied_limits:
            if tenant not in known_tenants:
                raise ValueError(f"limits reference unknown tenant {tenant!r}")

        return cls(
            key_tenants=copied_key_tenants,
            limits=copied_limits,
            default_tenant=default_tenant,
        )

    def tenant_for_key(self, api_key: str | None) -> str:
        if api_key is None:
            return self.default_tenant
        return self.key_tenants.get(api_key, self.default_tenant)

    def limits_for(self, tenant: str) -> TenantLimits:
        return self.limits.get(tenant, TenantLimits())


class _Bucket:
    def __init__(
        self,
        per_minute: int,
        now: Callable[[], float],
        *,
        capacity: int | None = None,
    ) -> None:
        self.capacity = float(per_minute if capacity is None else capacity)
        self.tokens = self.capacity
        self.rate = per_minute / 60.0
        self.updated = now()

    def take(self, amount: float, now: float) -> bool:
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = max(self.updated, now)
        # Float clocks routinely represent an exact refill boundary (for
        # example 6 RPM at t=10s) a few ulps below the requested amount.  Do
        # not turn that into an extra second of throttling.
        if self.tokens + 1e-9 < amount:
            return False
        self.tokens = max(0.0, self.tokens - amount)
        return True

    def debit(self, amount: float, now: float) -> None:
        """Unconditionally subtract (may go negative — overage carries forward)."""
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = max(self.updated, now)
        self.tokens -= amount

    def refund(self, amount: float, now: float) -> None:
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate + amount)
        self.updated = max(self.updated, now)


class TenantLimiter:
    """Per-tenant request-rate AND token-rate buckets (single-gateway; the
    distributed limiter is a G6 note)."""

    def __init__(self, config: TenantConfig, now: Callable[[], float] = time.monotonic):
        self._config = config
        self._now = now
        self._buckets: dict[str, _Bucket] = {}
        self._async_submission_buckets: dict[str, _Bucket] = {}
        self._token_buckets: dict[str, _Bucket] = {}
        self._in_flight: dict[str, int] = {}
        self._reserved_tokens: dict[str, int] = {}
        self._bound_violations: dict[str, int] = {}

    def _request_bucket(self, tenant: str) -> _Bucket:
        bucket = self._buckets.get(tenant)
        if bucket is None:
            limits = self._config.limits_for(tenant)
            bucket = _Bucket(
                limits.requests_per_minute,
                self._now,
                capacity=limits.request_burst,
            )
            self._buckets[tenant] = bucket
        return bucket

    def _token_bucket(self, tenant: str) -> _Bucket:
        bucket = self._token_buckets.get(tenant)
        if bucket is None:
            limits = self._config.limits_for(tenant)
            bucket = _Bucket(
                limits.tokens_per_minute,
                self._now,
                capacity=limits.token_burst,
            )
            self._token_buckets[tenant] = bucket
        return bucket

    def admit_async_submission(self, tenant: str) -> bool:
        """Rate-limit durable queue growth separately from inference RPM."""
        bucket = self._async_submission_buckets.get(tenant)
        if bucket is None:
            limits = self._config.limits_for(tenant)
            bucket = _Bucket(
                limits.requests_per_minute,
                self._now,
                capacity=limits.request_burst,
            )
            self._async_submission_buckets[tenant] = bucket
        return bucket.take(1.0, self._now())

    def acquire(self, tenant: str) -> TenantAdmission:
        """Acquire one downstream execution lease before shared admission."""

        # A tenant that has already burned its token budget is refused before a
        # new request runs. Tokens are settled after execution; max_in_flight
        # bounds that post-settlement exposure when configured.
        now = self._now()
        token_bucket = self._token_bucket(tenant)
        token_bucket.take(0.0, now)  # refill only
        if token_bucket.tokens < 1.0:
            return TenantAdmission.rejected(tenant, "token_quota")
        limits = self._config.limits_for(tenant)
        active = self._in_flight.get(tenant, 0)
        if limits.max_in_flight is not None and active >= limits.max_in_flight:
            return TenantAdmission.rejected(tenant, "in_flight")
        if not self._request_bucket(tenant).take(1.0, now):
            return TenantAdmission.rejected(tenant, "request_quota")
        self._in_flight[tenant] = active + 1
        return TenantAdmission.acquired(tenant, self)

    def acquire_reserved(
        self,
        tenant: str,
        tokens: int,
        *,
        refundable_on_exact_usage: bool = True,
    ) -> TenantAdmission:
        """Atomically acquire request, in-flight, and token capacity."""
        if type(tokens) is not int or tokens < 1:
            raise ValueError("token reservation must be a positive integer")
        now = self._now()
        token_bucket = self._token_bucket(tenant)
        token_bucket.take(0.0, now)
        if tokens > token_bucket.capacity:
            return TenantAdmission.rejected(tenant, "token_request_too_large")
        if token_bucket.tokens + 1e-9 < tokens:
            return TenantAdmission.rejected(
                tenant,
                "token_quota",
                retry_after_s=(tokens - token_bucket.tokens) / token_bucket.rate,
            )
        limits = self._config.limits_for(tenant)
        active = self._in_flight.get(tenant, 0)
        if limits.max_in_flight is not None and active >= limits.max_in_flight:
            return TenantAdmission.rejected(tenant, "in_flight", retry_after_s=1.0)
        request_bucket = self._request_bucket(tenant)
        request_bucket.take(0.0, now)
        if request_bucket.tokens + 1e-9 < 1.0:
            return TenantAdmission.rejected(
                tenant,
                "request_quota",
                retry_after_s=(1.0 - request_bucket.tokens) / request_bucket.rate,
            )
        request_taken = request_bucket.take(1.0, now)
        token_taken = token_bucket.take(float(tokens), now)
        if not request_taken or not token_taken:  # defensive: synchronous checks above
            raise RuntimeError("tenant quota changed during atomic reservation")
        self._in_flight[tenant] = active + 1
        self._reserved_tokens[tenant] = self._reserved_tokens.get(tenant, 0) + tokens
        admission = TenantAdmission.acquired(tenant, self)
        admission._reserved_tokens = tokens
        admission._refundable_on_exact_usage = refundable_on_exact_usage
        return admission

    def admit(self, tenant: str) -> bool:
        """Compatibility one-shot admission without holding an in-flight lease."""

        admission = self.acquire(tenant)
        admission.release()
        return admission.admitted

    def _release(self, tenant: str) -> None:
        active = self._in_flight.get(tenant, 0)
        if active < 1:
            raise RuntimeError(f"tenant {tenant!r} has no in-flight lease to release")
        if active == 1:
            self._in_flight.pop(tenant, None)
        else:
            self._in_flight[tenant] = active - 1

    def in_flight(self, tenant: str) -> int:
        return self._in_flight.get(tenant, 0)

    def charge_tokens(self, tenant: str, tokens: int) -> None:
        """Debit a completed request's tokens from the tenant's token bucket."""
        self._token_bucket(tenant).debit(float(tokens), self._now())  # may go negative

    def _reserve_tokens(self, tenant: str, tokens: int) -> bool:
        if not self._token_bucket(tenant).take(float(tokens), self._now()):
            return False
        self._reserved_tokens[tenant] = (
            self._reserved_tokens.get(tenant, 0) + tokens
        )
        return True

    def _finish_reservation(self, tenant: str, reserved: int) -> None:
        outstanding = self._reserved_tokens.get(tenant, 0)
        if reserved > outstanding:
            raise RuntimeError("released tenant tokens exceed reservations")
        remaining = outstanding - reserved
        if remaining:
            self._reserved_tokens[tenant] = remaining
        else:
            self._reserved_tokens.pop(tenant, None)

    def _settle_tokens(
        self,
        tenant: str,
        reserved: int,
        actual: int,
        *,
        refund_surplus: bool,
    ) -> None:
        bucket = self._token_bucket(tenant)
        difference = actual - reserved
        if difference > 0:
            self._bound_violations[tenant] = (
                self._bound_violations.get(tenant, 0) + 1
            )
            logger.error(
                "tenant reservation bound violated",
                extra={
                    "tenant": tenant,
                    "reserved_tokens": reserved,
                    "actual_tokens": actual,
                },
            )
            bucket.debit(float(difference), self._now())
        elif difference < 0 and refund_surplus:
            bucket.refund(float(-difference), self._now())
        self._finish_reservation(tenant, reserved)

    def _refund_tokens(self, tenant: str, tokens: int) -> None:
        self._token_bucket(tenant).refund(float(tokens), self._now())
        self._finish_reservation(tenant, tokens)

    def _consume_reserved_tokens(self, tenant: str, tokens: int) -> None:
        self._finish_reservation(tenant, tokens)

    def token_balance(self, tenant: str) -> float:
        """Current balance after refill, exposed for deterministic gate tests."""

        bucket = self._token_bucket(tenant)
        bucket.take(0.0, self._now())
        return bucket.tokens

    def reservation_snapshot(self) -> dict[str, int]:
        tenants = {
            self._config.default_tenant,
            *self._config.key_tenants.values(),
            *self._config.limits,
        }
        return {
            tenant: self._reserved_tokens.get(tenant, 0)
            for tenant in tenants
        }

    def bound_violation_snapshot(self) -> dict[str, int]:
        return {
            tenant: self._bound_violations.get(tenant, 0)
            for tenant in self.reservation_snapshot()
        }


class TenantAdmission:
    """Idempotent lease returned by ``TenantLimiter.acquire``."""

    __slots__ = (
        "tenant",
        "reason",
        "retry_after_s",
        "_limiter",
        "_released",
        "_reserved_tokens",
        "_dispatched",
        "_settled",
        "_refundable_on_exact_usage",
    )

    def __init__(
        self,
        tenant: str,
        reason: str,
        limiter: TenantLimiter | None,
        retry_after_s: float | None = None,
    ) -> None:
        self.tenant = tenant
        self.reason = reason
        self.retry_after_s = retry_after_s
        self._limiter = limiter
        self._released = False
        self._reserved_tokens = 0
        self._dispatched = False
        self._settled = False
        self._refundable_on_exact_usage = True

    @classmethod
    def acquired(
        cls,
        tenant: str,
        limiter: TenantLimiter,
    ) -> TenantAdmission:
        return cls(tenant, "quota_available", limiter)

    @classmethod
    def rejected(
        cls,
        tenant: str,
        reason: str,
        *,
        retry_after_s: float | None = None,
    ) -> TenantAdmission:
        return cls(tenant, reason, None, retry_after_s)

    @property
    def admitted(self) -> bool:
        return self._limiter is not None

    @property
    def reserved_tokens(self) -> int:
        return self._reserved_tokens

    @property
    def dispatched(self) -> bool:
        return self._dispatched

    def reserve_tokens(
        self,
        tokens: int,
        *,
        refundable_on_exact_usage: bool = True,
    ) -> bool:
        """Atomically reserve worst-case compute before shared dispatch."""

        if type(tokens) is not int or tokens < 1:
            raise ValueError("token reservation must be a positive integer")
        if self._released:
            raise RuntimeError("cannot reserve tokens on a released admission")
        if self._limiter is None:
            return False
        if self._reserved_tokens:
            raise RuntimeError("tenant admission already has a token reservation")
        if not self._limiter._reserve_tokens(self.tenant, tokens):
            self.reason = "token_quota"
            return False
        self._reserved_tokens = tokens
        self._refundable_on_exact_usage = refundable_on_exact_usage
        self.reason = "quota_available"
        return True

    def mark_dispatched(self) -> None:
        """Declare that the reservation crossed the shared compute boundary."""

        if self._released:
            raise RuntimeError("cannot dispatch a released admission")
        if self._limiter is None:
            return
        if self._reserved_tokens < 1:
            raise RuntimeError("tenant tokens must be reserved before dispatch")
        self._dispatched = True

    def settle_tokens(
        self,
        actual_tokens: int,
        *,
        exact: bool = True,
        force_refund_surplus: bool = False,
    ) -> None:
        """Reconcile a reservation once authoritative usage is available."""

        if type(actual_tokens) is not int or actual_tokens < 0:
            raise ValueError("actual token count must be a non-negative integer")
        if force_refund_surplus and not exact:
            raise ValueError("surplus refund requires exact token usage")
        if self._released:
            raise RuntimeError("cannot settle a released admission")
        if self._limiter is None:
            return
        if self._reserved_tokens < 1:
            raise RuntimeError("tenant admission has no token reservation")
        if self._settled:
            return
        # Every observed count is a lower bound on work, even when it is not
        # authoritative enough to refund.  Always debit and report a positive
        # overage; only exact, single-candidate usage may return surplus.
        self._limiter._settle_tokens(
            self.tenant,
            self._reserved_tokens,
            actual_tokens,
            refund_surplus=exact
            and (self._refundable_on_exact_usage or force_refund_surplus),
        )
        self._settled = True

    def release(self) -> None:
        if self._released or self._limiter is None:
            return
        # Pre-dispatch validation/cancellation consumed no shared compute, so
        # the reservation is refundable.  Once dispatched, missing usage
        # (backend failure or disconnect) retains the worst-case debit.
        if self._reserved_tokens and not self._dispatched and not self._settled:
            self._limiter._refund_tokens(self.tenant, self._reserved_tokens)
        elif self._reserved_tokens and self._dispatched and not self._settled:
            self._limiter._consume_reserved_tokens(
                self.tenant,
                self._reserved_tokens,
            )
        self._released = True
        self._limiter._release(self.tenant)


class TenantLimitMiddleware:
    """Pure ASGI; requires auth to have stored the key in scope state (A6)."""

    def __init__(
        self,
        app,
        *,
        config: TenantConfig,
        limiter: TenantLimiter,
        metrics=None,
    ) -> None:
        self.app = app
        self._config = config
        self._limiter = limiter
        self._metrics = metrics

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not (
            path.startswith("/v1/") or path.startswith("/admin/usage")
        ):
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        if state.get("is_admin") and not state.get("is_data_plane"):
            await self.app(scope, receive, send)
            return
        tenant = self._config.tenant_for_key(state.get("api_key"))
        state["tenant"] = tenant
        state["priority"] = self._config.limits_for(tenant).interactive_priority
        state["scheduling_class"] = "interactive"
        async_control_path = (
            path == "/v1/async/chat/completions"
            or path == "/v1/requests"
            or path.startswith("/v1/requests/")
        )
        if path == "/v1/async/chat/completions":
            admitted = self._limiter.admit_async_submission(tenant)
            if self._metrics is not None:
                self._metrics.record_tenant_admission(
                    tenant,
                    source="async_submit",
                    admitted=admitted,
                    reason="admitted" if admitted else "request_quota",
                )
                if admitted:
                    self._metrics.record_tenant_release(
                        tenant,
                        source="async_submit",
                    )
            if not admitted:
                body = json.dumps(
                    {
                        "error": {
                            "message": (
                                f"tenant {tenant!r} async submission limit exceeded"
                            ),
                            "type": "rate_limit_error",
                            "code": "tenant_rate_limited",
                        }
                    }
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"retry-after", b"1"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            await self.app(scope, receive, send)
            return
        if not path.startswith("/v1/") or async_control_path:
            # Durable async control-plane calls are authenticated and scoped,
            # but the worker owns the one inference admission charge.
            await self.app(scope, receive, send)  # identity only, no bucket
            return
        admission = self._limiter.acquire(tenant)
        state["tenant_admission"] = admission
        if self._metrics is not None and not admission.admitted:
            self._metrics.record_tenant_admission(
                tenant,
                source="http",
                admitted=False,
                reason=admission.reason,
            )
        if not admission.admitted:
            message = (
                f"tenant {tenant!r} admission limit exceeded "
                f"({admission.reason})"
            )
            if wants_anthropic_envelope(path):
                # /v1/messages speaks the Anthropic error envelope (issue #508)
                payload = anthropic_error_payload(
                    message,
                    error_type="rate_limit_error",
                    request_id=state.get("request_id"),
                )
            else:
                payload = {
                    "error": {
                        "message": message,
                        "type": "rate_limit_error",
                        "code": "tenant_rate_limited",
                    }
                }
            body = json.dumps(payload).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"retry-after", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        try:
            await self.app(scope, receive, send)
        finally:
            admission.release()
            if self._metrics is not None and state.get(
                "tenant_metric_admitted",
                False,
            ):
                self._metrics.record_tenant_release(tenant, source="http")


class UsageLedger:
    """Bounded asynchronous JSONL (A7), one line per successful execution.

    ``record`` only admits to a bounded queue. The lifecycle-owned background
    writer batches append+flush work; ``totals`` inserts an ordered flush
    barrier before reading, and ``close`` drains all accepted records."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_pending_records: int = 4_096,
        batch_size: int = 128,
        _open_file: Callable[..., IO[str]] = open,
    ) -> None:
        self._path = Path(path)
        self._writer = BoundedJsonlWriter(
            self._path,
            max_pending=max_pending_records,
            batch_size=batch_size,
            open_file=_open_file,
        )
        self._malformed_lines = 0

    @property
    def _handle(self) -> IO[str] | None:
        """Compatibility diagnostic for the app-lifecycle tests."""
        return self._writer.handle

    @property
    def malformed_lines(self) -> int:
        """Number of malformed non-whitespace records in the latest scan."""
        return self._malformed_lines

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        tenant: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> None:
        if (
            type(prompt_tokens) is not int
            or type(completion_tokens) is not int
            or type(cached_tokens) is not int
            or prompt_tokens < 0
            or completion_tokens < 0
            or cached_tokens < 0
            or cached_tokens > prompt_tokens
        ):
            raise ValueError(
                "usage token counts must be non-negative integers and "
                "cached_tokens must not exceed prompt_tokens"
            )
        payload = {
            "tenant": tenant,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "uncached_tokens": prompt_tokens - cached_tokens,
            "ts": time.time(),
        }
        self._writer.append_deferred(lambda payload=payload: json.dumps(payload))

    def flush(self) -> None:
        """Wait until every accepted record is flushed and reader-visible."""
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def snapshot_bytes(self) -> bytes:
        """Return a reader-stable byte snapshot through the latest barrier."""
        self.flush()
        if not self._path.is_file():
            return b""
        size = self._path.stat().st_size
        with open(self._path, "rb") as handle:
            return handle.read(size)

    def totals(self, tenant: str | None = None) -> dict[str, dict[str, int]]:
        """Aggregate by tenant (optionally filtered) for /admin/usage."""
        self.flush()
        totals: dict[str, dict[str, int]] = {}
        self._malformed_lines = 0
        if not self._path.is_file():
            return totals
        with open(self._path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record_tenant = record["tenant"]
                    prompt_tokens = record["prompt_tokens"]
                    completion_tokens = record["completion_tokens"]
                    cached_tokens = record.get("cached_tokens", 0)
                    uncached_tokens = record.get(
                        "uncached_tokens",
                        (
                            prompt_tokens - cached_tokens
                            if type(prompt_tokens) is int
                            and type(cached_tokens) is int
                            else None
                        ),
                    )
                    if (
                        not isinstance(record_tenant, str)
                        or type(prompt_tokens) is not int
                        or type(completion_tokens) is not int
                        or type(cached_tokens) is not int
                        or type(uncached_tokens) is not int
                        or prompt_tokens < 0
                        or completion_tokens < 0
                        or cached_tokens < 0
                        or uncached_tokens < 0
                        or cached_tokens + uncached_tokens != prompt_tokens
                    ):
                        raise TypeError("usage ledger record has invalid field types")
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    self._malformed_lines += 1
                    if not line.endswith("\n"):
                        logger.warning(
                            "skipping truncated usage ledger tail in %s at line %d (%s)",
                            self._path,
                            line_number,
                            type(error).__name__,
                        )
                    else:
                        logger.error(
                            "skipping malformed usage ledger record in %s at line %d (%s)",
                            self._path,
                            line_number,
                            type(error).__name__,
                        )
                    continue
                if tenant is not None and record_tenant != tenant:
                    continue
                bucket = totals.setdefault(
                    record_tenant,
                    {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                        "uncached_tokens": 0,
                    },
                )
                bucket["requests"] += 1
                bucket["prompt_tokens"] += prompt_tokens
                bucket["completion_tokens"] += completion_tokens
                bucket["cached_tokens"] += cached_tokens
                bucket["uncached_tokens"] += uncached_tokens
        return totals

"""Prometheus metrics for the serve layer (design m7 D8).

Each app gets its own ``CollectorRegistry`` so multiple ``create_app`` calls
(tests, embedded use) never collide on timeseries names. Pool gauges are read
at scrape time through ``ReplicaPool``'s read-only accessors — the pool stays
passive (m5 D4).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

from kairyu.async_requests.models import AsyncRequestState
from kairyu.async_requests.store import (
    ASYNC_REQUEST_TRANSITION_EVENTS,
    RequestQueueMetricsSnapshot,
)
from kairyu.orchestration.replica import ReplicaPool

logger = logging.getLogger("kairyu.server.metrics")

_PREPLACEMENT_ENDPOINTS = frozenset(
    {"chat", "completions", "responses", "messages", "route", "batch"}
)
_PREPLACEMENT_PHASES = frozenset(
    {
        "ingress_to_handler",
        "request_validation",
        "routing_preflight",
        "backend_prepare",
        "admission",
    }
)


class _AsyncRequestStoreCollector:
    """Fail-open cached view over shared durable request stores.

    Blocking backends perform their first warmup in the metrics worker thread,
    after application startup, then refresh in the background at a bounded
    frequency. The ASGI event loop never performs the store query.
    """

    _REFRESH_INTERVAL_S = 1.0

    def __init__(self) -> None:
        self._stores: dict[str, object] = {}
        self._last_good: dict[str, RequestQueueMetricsSnapshot] = {}
        self._snapshot_ok: dict[str, bool] = {}
        self._last_refresh_started: dict[str, float] = {}
        self._refreshing: set[str] = set()
        self._attempted: set[str] = set()
        self._lock = threading.RLock()

    def add(self, store: object) -> None:
        snapshot = getattr(store, "metrics_snapshot", None)
        store_id = getattr(store, "store_id", None)
        if callable(snapshot) and isinstance(store_id, str) and store_id:
            with self._lock:
                self._stores[store_id] = store
                self._snapshot_ok.setdefault(store_id, False)

    def _refresh(self, store_id: str, store: object) -> None:
        try:
            snapshot = self._validate(store.metrics_snapshot())
        except Exception:
            logger.warning(
                "async request metrics snapshot failed for store %r",
                store_id,
                exc_info=True,
            )
            with self._lock:
                self._snapshot_ok[store_id] = False
        else:
            with self._lock:
                self._last_good[store_id] = snapshot
                self._snapshot_ok[store_id] = True
        finally:
            with self._lock:
                self._refreshing.discard(store_id)

    def _schedule_refresh(
        self,
        store_id: str,
        store: object,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            last_started = self._last_refresh_started.get(store_id, float("-inf"))
            if store_id in self._refreshing or (
                not force and now - last_started < self._REFRESH_INTERVAL_S
            ):
                return
            self._refreshing.add(store_id)
            self._attempted.add(store_id)
            self._last_refresh_started[store_id] = now
        threading.Thread(
            target=self._refresh,
            args=(store_id, store),
            name=f"kairyu-async-metrics-{store_id}",
            daemon=True,
        ).start()

    @staticmethod
    def _empty_snapshot() -> RequestQueueMetricsSnapshot:
        return RequestQueueMetricsSnapshot(
            state_counts={state: 0 for state in AsyncRequestState},
            oldest_queued_age_seconds=0.0,
            transition_counts={
                event: 0 for event in ASYNC_REQUEST_TRANSITION_EVENTS
            },
        )

    @staticmethod
    def _validate(snapshot: object) -> RequestQueueMetricsSnapshot:
        if not isinstance(snapshot, RequestQueueMetricsSnapshot):
            raise TypeError("async request metrics source returned an invalid snapshot")
        counts = tuple(snapshot.state_counts.get(state, 0) for state in AsyncRequestState)
        transitions = tuple(
            snapshot.transition_counts.get(event, 0)
            for event in ASYNC_REQUEST_TRANSITION_EVENTS
        )
        if any(type(value) is not int or value < 0 for value in (*counts, *transitions)):
            raise ValueError("async request metric counts must be non-negative integers")
        age = snapshot.oldest_queued_age_seconds
        if not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0:
            raise ValueError("async request oldest queue age must be finite and non-negative")
        return snapshot

    def collect(self) -> Iterator[Metric]:
        states = GaugeMetricFamily(
            "kairyu_async_request_state",
            "Shared durable requests by bounded lifecycle state",
            labels=["store", "state"],
        )
        depth = GaugeMetricFamily(
            "kairyu_async_request_queue_depth",
            "Current non-expired queued requests in the shared durable store",
            labels=["store"],
        )
        oldest = GaugeMetricFamily(
            "kairyu_async_request_oldest_queued_age_seconds",
            "Age of the oldest non-expired queued request, or zero when empty",
            labels=["store"],
        )
        transitions = CounterMetricFamily(
            "kairyu_async_request_transitions",
            "Durable lifecycle transition events by bounded event",
            labels=["store", "event"],
        )
        attempts = CounterMetricFamily(
            "kairyu_async_request_attempts",
            "Durable execution attempts, including lease reclaims",
            labels=["store"],
        )
        success = GaugeMetricFamily(
            "kairyu_async_request_metrics_snapshot_success",
            "Whether the current shared-store metrics snapshot succeeded",
            labels=["store"],
        )
        with self._lock:
            stores = tuple(self._stores.items())
        for store_id, store in stores:
            if bool(getattr(store, "metrics_snapshot_nonblocking", False)):
                self._refresh(store_id, store)
            else:
                with self._lock:
                    first_refresh = store_id not in self._attempted
                    if first_refresh:
                        self._attempted.add(store_id)
                        self._refreshing.add(store_id)
                        self._last_refresh_started[store_id] = time.monotonic()
                if first_refresh:
                    self._refresh(store_id, store)
                else:
                    self._schedule_refresh(store_id, store)
            with self._lock:
                snapshot_ok = self._snapshot_ok.get(store_id, False)
                snapshot = self._last_good.get(store_id, self._empty_snapshot())
            for state in AsyncRequestState:
                states.add_metric(
                    [store_id, state.value],
                    snapshot.state_counts.get(state, 0),
                )
            depth.add_metric([store_id], snapshot.queue_depth)
            oldest.add_metric([store_id], snapshot.oldest_queued_age_seconds)
            for event in ASYNC_REQUEST_TRANSITION_EVENTS:
                transitions.add_metric(
                    [store_id, event],
                    snapshot.transition_counts.get(event, 0),
                )
            attempts.add_metric([store_id], snapshot.attempts_total)
            success.add_metric([store_id], 1.0 if snapshot_ok else 0.0)
        yield states
        yield depth
        yield oldest
        yield transitions
        yield attempts
        yield success


class _PoolCollector:
    """Scrape-time view over tracked ReplicaPools (no background sampling)."""

    def __init__(self) -> None:
        self._pools: dict[str, ReplicaPool] = {}

    def add(self, name: str, pool: ReplicaPool) -> None:
        self._pools[name] = pool

    def collect(self) -> Iterator[Metric]:
        outstanding = GaugeMetricFamily(
            "kairyu_replica_outstanding",
            "In-flight requests per replica",
            labels=["pool", "replica"],
        )
        healthy = GaugeMetricFamily(
            "kairyu_replica_healthy",
            (
                "Replica health (1 = readiness-validated and below the "
                "failure threshold; draining is reported separately)"
            ),
            labels=["pool", "replica"],
        )
        decisions = CounterMetricFamily(
            "kairyu_pool_decisions",
            "Placement decisions by reason (session_affinity is the cache-affinity signal)",
            labels=["pool", "reason"],
        )
        membership = GaugeMetricFamily(
            "kairyu_pool_replicas",
            "Replica membership by bounded state",
            labels=["pool", "state"],
        )
        for name, pool in self._pools.items():
            outstanding_by_id = pool.outstanding_by_id()
            healthy_by_id = pool.healthy_by_id()
            draining_by_id = pool.draining_by_id()
            for replica_id, count in outstanding_by_id.items():
                outstanding.add_metric([name, replica_id], count)
            for replica_id, is_healthy in healthy_by_id.items():
                healthy.add_metric(
                    [name, replica_id], 1.0 if is_healthy else 0.0
                )
            for reason, count in pool.decision_counts.items():
                decisions.add_metric([name, reason], count)
            membership.add_metric([name, "configured"], len(healthy_by_id))
            membership.add_metric(
                [name, "healthy"],
                sum(healthy_by_id.values()),
            )
            membership.add_metric(
                [name, "draining"],
                sum(draining_by_id.values()),
            )
        yield outstanding
        yield healthy
        yield decisions
        yield membership


class _SchedulerCollector:
    """Scrape-time native scheduler view with bounded class/event labels."""

    _CLASSES = ("interactive", "batch")
    _EVENTS = ("enqueue", "admit", "preempt", "complete")

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}

    def add(self, name: str, engine: object) -> None:
        if callable(getattr(engine, "scheduler_priority_metrics", None)):
            self._engines[name] = engine

    def collect(self) -> Iterator[Metric]:
        events = CounterMetricFamily(
            "kairyu_scheduler_priority_events",
            "Native scheduler events by bounded request class",
            labels=["model", "request_class", "event"],
        )
        depth = GaugeMetricFamily(
            "kairyu_scheduler_queue_depth",
            "Current native scheduler waiting depth by bounded request class",
            labels=["model", "request_class"],
        )
        high_watermark = GaugeMetricFamily(
            "kairyu_scheduler_queue_high_watermark",
            "Maximum native scheduler waiting depth by bounded request class",
            labels=["model", "request_class"],
        )
        for model, engine in self._engines.items():
            snapshot = engine.scheduler_priority_metrics()
            if not snapshot:
                continue
            event_counts = snapshot["events"]
            for request_class in self._CLASSES:
                for event in self._EVENTS:
                    events.add_metric(
                        [model, request_class, event],
                        event_counts.get((request_class, event), 0),
                    )
                depth.add_metric(
                    [model, request_class],
                    snapshot["queue_depth"].get(request_class, 0),
                )
                high_watermark.add_metric(
                    [model, request_class],
                    snapshot["queue_high_watermark"].get(request_class, 0),
                )
        yield events
        yield depth
        yield high_watermark


class _CudaGraphCollector:
    """Monotonic scrape-time counters from native model runners and pools."""

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}
        self._source_counters: dict[
            str,
            dict[tuple[str, str, str, str], tuple[int, int]],
        ] = {}
        self._retired_totals: dict[str, int] = {}
        self._ever_supported: set[str] = set()

    def add(self, name: str, engine: object) -> None:
        if callable(
            getattr(engine, "decode_graph_metadata_snapshot", None)
        ) or callable(getattr(engine, "decode_graph_metric_sources_snapshot", None)):
            self._engines[name] = engine

    @staticmethod
    def _snapshots(
        engine: object,
    ) -> tuple[
        tuple[tuple[str, str, str, str], dict[str, object] | None], ...
    ] | None:
        pool_getter = getattr(engine, "decode_graph_metric_sources_snapshot", None)
        if callable(pool_getter):
            try:
                rows = pool_getter()
            except Exception:
                return None
            if not isinstance(rows, tuple):
                return None
            snapshots: list[
                tuple[tuple[str, str, str, str], dict[str, object] | None]
            ] = []
            for row in rows:
                if (
                    not isinstance(row, tuple)
                    or len(row) != 4
                    or not isinstance(row[0], str)
                    or not isinstance(row[1], str)
                    or (row[2] is not None and type(row[2]) not in (int, str))
                    or (row[3] is not None and not isinstance(row[3], dict))
                ):
                    return None
                snapshots.append(
                    (
                        ("replica", row[0], row[1], repr(row[2])),
                        row[3],
                    )
                )
            return tuple(snapshots)

        getter = getattr(engine, "decode_graph_metadata_snapshot", None)
        if not callable(getter):
            return ()
        generation_getter = getattr(
            engine,
            "decode_graph_metric_generation_snapshot",
            None,
        )
        try:
            service_generation = (
                generation_getter() if callable(generation_getter) else None
            )
            if (
                service_generation is not None
                and type(service_generation) not in (int, str)
            ):
                return None
            snapshot = getter()
        except Exception:
            return None
        if not isinstance(snapshot, dict):
            snapshot = None
        return (
            (
                ("engine", "", str(id(engine)), repr(service_generation)),
                snapshot,
            ),
        )

    def collect(self) -> Iterator[Metric]:
        fallbacks = CounterMetricFamily(
            "kairyu_cuda_graph_eager_fallbacks",
            "Decode batches that exceeded a configured CUDA graph shape",
            labels=["model"],
        )
        for model, engine in self._engines.items():
            snapshots = self._snapshots(engine)
            states = self._source_counters.setdefault(model, {})
            if snapshots is None:
                if model in self._ever_supported:
                    fallbacks.add_metric(
                        [model],
                        self._retired_totals.get(model, 0)
                        + sum(offset + raw for raw, offset in states.values()),
                    )
                continue

            active_sources: set[tuple[str, str, str, str]] = set()
            for source, snapshot in snapshots:
                active_sources.add(source)
                if snapshot is None:
                    continue
                count = snapshot.get("cuda_graph_eager_fallbacks")
                if type(count) is not int or count < 0:
                    continue
                self._ever_supported.add(model)
                prior = states.get(source)
                if prior is None:
                    states[source] = (count, 0)
                    continue
                prior_raw, offset = prior
                if count < prior_raw:
                    offset += prior_raw
                states[source] = (count, offset)

            retired = self._retired_totals.get(model, 0)
            for source in tuple(states):
                if source in active_sources:
                    continue
                raw, offset = states.pop(source)
                retired += offset + raw
            self._retired_totals[model] = retired
            if model in self._ever_supported:
                fallbacks.add_metric(
                    [model],
                    retired + sum(offset + raw for raw, offset in states.values()),
                )
        yield fallbacks


class _TenantLimiterCollector:
    """Scrape-time view of reservations not yet settled or consumed."""

    def __init__(self) -> None:
        self._limiter = None

    def set(self, limiter: object) -> None:
        self._limiter = limiter

    def collect(self) -> Iterator[Metric]:
        reserved = GaugeMetricFamily(
            "kairyu_tenant_reserved_tokens",
            "Worst-case tenant compute tokens reserved before shared dispatch",
            labels=["tenant"],
        )
        violations = CounterMetricFamily(
            "kairyu_tenant_reservation_bound_violations",
            "Actual tenant work exceeding its pre-dispatch reservation",
            labels=["tenant"],
        )
        if self._limiter is not None:
            snapshot = self._limiter.reservation_snapshot()
            for tenant, tokens in sorted(snapshot.items()):
                reserved.add_metric([tenant], tokens)
            for tenant, count in sorted(
                self._limiter.bound_violation_snapshot().items()
            ):
                violations.add_metric([tenant], count)
        yield reserved
        yield violations


class _SLOAdmissionCollector:
    """Scrape-time view of the app-owned TTFT admission predictor."""

    def __init__(self) -> None:
        self._controller = None

    def set(self, controller: object) -> None:
        self._controller = controller

    def collect(self) -> Iterator[Metric]:
        families = (
            GaugeMetricFamily(
                "kairyu_slo_admission_in_flight",
                "Direct-chat requests holding SLO admission leases",
            ),
            GaugeMetricFamily(
                "kairyu_slo_admission_interactive_in_flight",
                "Interactive direct-chat requests holding SLO admission leases",
            ),
            GaugeMetricFamily(
                "kairyu_slo_admission_deferred_in_flight",
                "Deferred direct-chat requests holding SLO admission leases",
            ),
            GaugeMetricFamily(
                "kairyu_slo_admission_ttft_per_unit_ema_seconds",
                "EMA of post-admission interactive TTFT per concurrency unit",
            ),
            GaugeMetricFamily(
                "kairyu_slo_admission_predicted_ttft_seconds",
                "Predicted TTFT for the next interactive direct-chat request",
            ),
            GaugeMetricFamily(
                "kairyu_slo_admission_defer_pressure_seconds",
                "Predicted total active pressure used to bound deferred work",
            ),
        )
        if self._controller is not None:
            snapshot = self._controller.snapshot()
            values = (
                snapshot.in_flight,
                snapshot.interactive_in_flight,
                snapshot.deferred_in_flight,
                snapshot.ttft_per_unit_ema_s,
                snapshot.predicted_ttft_s,
                snapshot.defer_pressure_s,
            )
            for family, value in zip(families, values, strict=True):
                family.add_metric([], value)
        yield from families


class ServerMetrics:
    """Registry + the request-level metrics recorded by the middleware."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "kairyu_requests",
            "API requests by served model and HTTP status",
            ["model", "code"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "kairyu_request_duration_seconds",
            "Request wall time by path",
            ["path"],
            registry=self.registry,
        )
        self.preplacement_phase_seconds = Histogram(
            "kairyu_preplacement_phase_seconds",
            (
                "Request ingress-to-placement work split into bounded serving "
                "phases"
            ),
            ["endpoint", "phase"],
            registry=self.registry,
        )
        self.conductor_stage_seconds = Histogram(
            "kairyu_conductor_stage_seconds",
            (
                "Orchestration DAG stage wall time by public model, stage "
                "node, operation, and terminal status (issue #495 latency "
                "attribution)"
            ),
            ["model", "node", "operation", "status"],
            registry=self.registry,
            buckets=(
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
                120.0,
                300.0,
            ),
        )
        self.admission_active_requests = Gauge(
            "kairyu_admission_active_requests",
            "Requests holding backend-aware HTTP admission slots",
            registry=self.registry,
        )
        self.admission_waiting_requests = Gauge(
            "kairyu_admission_waiting_requests",
            "Requests waiting for backend-aware HTTP admission slots",
            registry=self.registry,
        )
        self.admission_rejections_total = Counter(
            "kairyu_admission_rejections",
            "Backend-aware HTTP admission rejections by bounded reason",
            ["reason"],
            registry=self.registry,
        )
        self.batch_jobs_total = Counter(
            "kairyu_batch_jobs",
            "Batch jobs by terminal state",
            ["state"],
            registry=self.registry,
        )
        self.priority_requests_total = Counter(
            "kairyu_priority_requests",
            "Requests dispatched by bounded scheduling class and ingress source",
            ["request_class", "source"],
            registry=self.registry,
        )
        self.tenant_admission_total = Counter(
            "kairyu_tenant_admission",
            "Tenant admission decisions before shared downstream capacity",
            ["tenant", "source", "decision", "reason"],
            registry=self.registry,
        )
        self.tenant_in_flight_requests = Gauge(
            "kairyu_tenant_in_flight_requests",
            "Admitted tenant requests currently holding downstream capacity",
            ["tenant", "source"],
            registry=self.registry,
        )
        self.usage_requests_total = Counter(
            "kairyu_usage_requests",
            "Successful metered executions by tenant",
            ["tenant"],
            registry=self.registry,
        )
        self.usage_tokens_total = Counter(
            "kairyu_usage_tokens",
            "Metered tokens by tenant and token type",
            ["tenant", "type"],
            registry=self.registry,
        )
        self._pool_collector = _PoolCollector()
        self._scheduler_collector = _SchedulerCollector()
        self._cuda_graph_collector = _CudaGraphCollector()
        self._tenant_limiter_collector = _TenantLimiterCollector()
        self._slo_admission_collector = _SLOAdmissionCollector()
        self._async_request_store_collector = _AsyncRequestStoreCollector()
        self.registry.register(self._pool_collector)
        self.registry.register(self._scheduler_collector)
        self.registry.register(self._cuda_graph_collector)
        self.registry.register(self._tenant_limiter_collector)
        self.registry.register(self._slo_admission_collector)
        self.registry.register(self._async_request_store_collector)

    def track_pool(self, name: str, pool: ReplicaPool) -> None:
        self._pool_collector.add(name, pool)

    def track_scheduler(self, name: str, engine: object) -> None:
        self._scheduler_collector.add(name, engine)

    def track_cuda_graph(self, name: str, engine: object) -> None:
        self._cuda_graph_collector.add(name, engine)

    def track_tenant_limiter(self, limiter: object) -> None:
        self._tenant_limiter_collector.set(limiter)

    def track_slo_admission(self, controller: object) -> None:
        self._slo_admission_collector.set(controller)

    def track_async_request_store(self, store: object) -> None:
        self._async_request_store_collector.add(store)

    def set_admission_depth(self, *, active: int, waiting: int) -> None:
        self.admission_active_requests.set(active)
        self.admission_waiting_requests.set(waiting)

    def record_admission_rejection(self, reason: str) -> None:
        if reason not in {"overflow", "timeout"}:
            raise ValueError(f"invalid admission rejection reason {reason!r}")
        self.admission_rejections_total.labels(reason=reason).inc()

    def record_conductor_stage(
        self,
        model: str,
        node: str,
        operation: str,
        status: str,
        seconds: float,
    ) -> None:
        """Record one orchestration DAG stage observation (issue #495)."""

        if seconds < 0:
            raise ValueError("stage duration must be non-negative")
        self.conductor_stage_seconds.labels(model, node, operation, status).observe(
            seconds
        )

    def record_preplacement_phase(
        self,
        endpoint: str,
        phase: str,
        duration_ns: int,
    ) -> None:
        """Record one bounded-cardinality pre-placement phase observation."""

        if endpoint not in _PREPLACEMENT_ENDPOINTS:
            raise ValueError(f"invalid preplacement endpoint {endpoint!r}")
        if phase not in _PREPLACEMENT_PHASES:
            raise ValueError(f"invalid preplacement phase {phase!r}")
        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("preplacement duration_ns must be a non-negative integer")
        self.preplacement_phase_seconds.labels(
            endpoint=endpoint,
            phase=phase,
        ).observe(duration_ns / 1_000_000_000)

    def record_priority(self, request_class: str, *, source: str) -> None:
        """Record an explicit bounded class without labeling priority integers."""

        if request_class not in {"interactive", "batch"}:
            raise ValueError(f"invalid scheduling class {request_class!r}")
        self.priority_requests_total.labels(
            request_class=request_class,
            source=source,
        ).inc()

    def record_tenant_admission(
        self,
        tenant: str,
        *,
        source: str,
        admitted: bool,
        reason: str,
    ) -> None:
        decision = "admitted" if admitted else "rejected"
        self.tenant_admission_total.labels(
            tenant=tenant,
            source=source,
            decision=decision,
            reason=reason,
        ).inc()
        in_flight = self.tenant_in_flight_requests.labels(
            tenant=tenant,
            source=source,
        )
        if admitted:
            in_flight.inc()

    def record_tenant_release(self, tenant: str, *, source: str) -> None:
        self.tenant_in_flight_requests.labels(
            tenant=tenant,
            source=source,
        ).dec()

    def record_usage(
        self,
        tenant: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
    ) -> None:
        """Mirror one accepted ledger row at the tenant aggregation boundary."""
        self.usage_requests_total.labels(tenant=tenant).inc()
        self.usage_tokens_total.labels(tenant=tenant, type="prompt").inc(
            prompt_tokens
        )
        self.usage_tokens_total.labels(tenant=tenant, type="completion").inc(
            completion_tokens
        )
        self.usage_tokens_total.labels(tenant=tenant, type="cached").inc(
            cached_tokens
        )
        self.usage_tokens_total.labels(tenant=tenant, type="uncached").inc(
            prompt_tokens - cached_tokens
        )

    def restore_usage_totals(
        self,
        totals: dict[str, dict[str, int]],
    ) -> None:
        """Restore process-local counters from the append-only ledger on startup."""
        for tenant, usage in totals.items():
            self.usage_requests_total.labels(tenant=tenant).inc(
                usage["requests"]
            )
            self.usage_tokens_total.labels(tenant=tenant, type="prompt").inc(
                usage["prompt_tokens"]
            )
            self.usage_tokens_total.labels(
                tenant=tenant,
                type="completion",
            ).inc(usage["completion_tokens"])
            self.usage_tokens_total.labels(tenant=tenant, type="cached").inc(
                usage["cached_tokens"]
            )
            self.usage_tokens_total.labels(tenant=tenant, type="uncached").inc(
                usage["uncached_tokens"]
            )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

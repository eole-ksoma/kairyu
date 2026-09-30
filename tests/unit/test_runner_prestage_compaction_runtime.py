"""Scheduled runtime coverage for durable pre-stage tombstone compaction."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.runners import (
    InMemoryNodeModelPrestageStore,
    NodeModelPrestageCompactionRuntime,
    NodeModelPrestageCompactionRuntimeConfig,
    NodeModelPrestageHighWaterMark,
)

_NOW = datetime(2026, 9, 30, 3, 0, tzinfo=UTC)


def _mark(
    placement_id: str,
    *,
    compacted_at: datetime = _NOW,
    updated_at: datetime | None = None,
) -> NodeModelPrestageHighWaterMark:
    return NodeModelPrestageHighWaterMark(
        placement_id=placement_id,
        command_id="a" * 64,
        command_generation=2,
        election_id="election-1",
        holder_id="controller-1",
        fencing_token=3,
        target_id="production/model",
        target_revision=4,
        release_identity_digest="b" * 64,
        attempt=1,
        updated_at=updated_at or compacted_at - timedelta(seconds=61),
        compacted_at=compacted_at,
    )


class ScriptedCompactionStore(InMemoryNodeModelPrestageStore):
    def __init__(
        self,
        results: list[tuple[NodeModelPrestageHighWaterMark, ...] | Exception],
        *,
        monitoring: tuple[NodeModelPrestageHighWaterMark, ...] = (),
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(node_id="gpu-node-00")
        self.results = results
        self.monitoring = monitoring
        self.on_call = on_call
        self.calls: list[tuple[datetime, datetime, int]] = []
        self.monitoring_calls: list[tuple[str | None, int]] = []
        self._script_lock = threading.Lock()

    def compact_absent_records(
        self,
        *,
        retired_before: datetime,
        compacted_at: datetime,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        with self._script_lock:
            call_number = len(self.calls) + 1
            self.calls.append((retired_before, compacted_at, limit))
            result = self.results.pop(0) if self.results else ()
        if self.on_call is not None:
            self.on_call(call_number)
        if isinstance(result, Exception):
            raise result
        return result

    def list_high_water_marks(self) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        return self.monitoring

    def list_high_water_marks_page(
        self,
        *,
        after_placement_id: str | None = None,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        self.monitoring_calls.append((after_placement_id, limit))
        return self.monitoring[:limit]


def _config(**updates: object) -> NodeModelPrestageCompactionRuntimeConfig:
    values: dict[str, object] = {
        "retirement_age_seconds": 60,
        "interval_seconds": 10,
        "batch_size": 2,
        "max_batches_per_cycle": 3,
        "readiness_max_staleness_seconds": 30,
        "readiness_failure_threshold": 2,
        "shutdown_timeout_seconds": 1,
    }
    values.update(updates)
    return NodeModelPrestageCompactionRuntimeConfig.model_validate(values)


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.001)


def test_config_rejects_unbounded_or_incoherent_schedule_values() -> None:
    with pytest.raises(ValidationError, match="batch_size"):
        _config(batch_size=1001)
    with pytest.raises(ValidationError, match="must be an integer"):
        _config(max_batches_per_cycle=True)
    with pytest.raises(ValidationError, match="finite number"):
        _config(interval_seconds=float("inf"))
    with pytest.raises(ValidationError, match="cannot be shorter"):
        _config(interval_seconds=31)
    with pytest.raises(ValidationError, match="cannot exceed 10000"):
        _config(batch_size=1000, max_batches_per_cycle=11)


def test_run_once_uses_one_cutoff_and_stops_after_a_short_batch() -> None:
    first = _mark("placement-00")
    second = _mark("placement-01")
    third = _mark("placement-02")
    store = ScriptedCompactionStore([(first, second), (third,)])
    clock_values = iter((_NOW, _NOW + timedelta(seconds=2)))
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(),
        clock=lambda: next(clock_values),
        monotonic=lambda: 5.0,
    )

    cycle = runtime.run_once()

    assert cycle.retired_before == _NOW - timedelta(seconds=60)
    assert cycle.completed_at == _NOW + timedelta(seconds=2)
    assert cycle.batch_calls == 2
    assert cycle.batch_budget_exhausted is False
    assert cycle.compacted == (first, second, third)
    assert store.calls == [
        (cycle.retired_before, _NOW, 2),
        (cycle.retired_before, _NOW, 2),
    ]
    status = runtime.status()
    assert status.state == "new"
    assert status.ready is False
    assert status.cycles_started == status.cycles_succeeded == 1
    assert status.cycles_failed == status.consecutive_failures == 0
    assert status.batch_calls == 2
    assert status.compacted_records == 3
    assert status.last_cycle == cycle


def test_cycle_honors_batch_budget_and_reports_possible_backlog() -> None:
    store = ScriptedCompactionStore(
        [
            (_mark("placement-00"), _mark("placement-01")),
            (_mark("placement-02"), _mark("placement-03")),
            (_mark("placement-04"),),
        ]
    )
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(max_batches_per_cycle=2),
        clock=lambda: _NOW,
        monotonic=lambda: 1.0,
    )

    cycle = runtime.run_once()

    assert cycle.batch_calls == 2
    assert cycle.batch_budget_exhausted is True
    assert len(cycle.compacted) == 4
    assert len(store.calls) == 2


def test_partial_cycle_failure_retains_completed_batch_metrics_without_detail() -> None:
    private_error = RuntimeError("private database host and SQL detail")
    store = ScriptedCompactionStore([(_mark("placement-00"), _mark("placement-01")), private_error])
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(),
        clock=lambda: _NOW,
        monotonic=lambda: 1.0,
    )

    with pytest.raises(RuntimeError, match="private database"):
        runtime.run_once()

    status = runtime.status()
    assert status.cycles_started == status.cycles_failed == 1
    assert status.cycles_succeeded == 0
    assert status.batch_calls == 1
    assert status.compacted_records == 2
    assert status.consecutive_failures == 1
    assert status.last_failure_at == _NOW
    assert "database" not in status.model_dump_json()


def test_background_runtime_is_immediate_stale_aware_and_lifecycle_owned() -> None:
    called = threading.Event()
    store = ScriptedCompactionStore([()], on_call=lambda _number: called.set())
    monotonic_value = [10.0]
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(interval_seconds=20, readiness_max_staleness_seconds=25),
        clock=lambda: _NOW,
        monotonic=lambda: monotonic_value[0],
    )

    runtime.start()
    assert called.wait(timeout=1)
    _wait_until(lambda: runtime.status().cycles_succeeded == 1)
    assert runtime.status().ready is True
    runtime.check_ready()

    monotonic_value[0] = 36.0
    assert runtime.status().ready is False
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.check_ready()

    runtime.close()
    runtime.close()
    assert runtime.status().state == "stopped"
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        runtime.start()


def test_repeated_background_failures_cross_readiness_threshold_and_retry() -> None:
    store = ScriptedCompactionStore(
        [(), RuntimeError("first private failure"), RuntimeError("second private failure")]
    )
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(interval_seconds=0.01, readiness_max_staleness_seconds=1),
        clock=lambda: _NOW,
        monotonic=time.monotonic,
    )
    runtime.start()
    try:
        _wait_until(lambda: runtime.status().cycles_failed >= 2)
        status = runtime.status()
        assert status.cycles_succeeded == 1
        assert status.consecutive_failures == 2
        assert status.ready is False
        assert "private" not in status.model_dump_json()
    finally:
        runtime.close()


def test_close_reports_a_worker_that_cannot_stop_before_deadline() -> None:
    entered = threading.Event()
    release = threading.Event()

    def block(_call_number: int) -> None:
        entered.set()
        release.wait(timeout=1)

    store = ScriptedCompactionStore([()], on_call=block)
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(shutdown_timeout_seconds=0.01),
        clock=lambda: _NOW,
        monotonic=time.monotonic,
    )
    runtime.start()
    assert entered.wait(timeout=1)

    with pytest.raises(RuntimeError, match="did not stop"):
        runtime.close()
    assert runtime.status().state == "stopping"

    release.set()
    runtime.close()
    assert runtime.status().state == "stopped"


def test_close_waits_for_an_explicit_cycle_before_reporting_stopped() -> None:
    entered = threading.Event()
    release = threading.Event()
    close_returned = threading.Event()

    def block(_call_number: int) -> None:
        entered.set()
        release.wait(timeout=1)

    runtime = NodeModelPrestageCompactionRuntime(
        store=ScriptedCompactionStore([()], on_call=block),
        config=_config(shutdown_timeout_seconds=1),
        clock=lambda: _NOW,
        monotonic=time.monotonic,
    )
    cycle_thread = threading.Thread(target=runtime.run_once)
    cycle_thread.start()
    assert entered.wait(timeout=1)

    def close_runtime() -> None:
        runtime.close()
        close_returned.set()

    close_thread = threading.Thread(target=close_runtime)
    close_thread.start()
    _wait_until(lambda: runtime.status().state == "stopping")
    assert close_returned.is_set() is False

    release.set()
    cycle_thread.join(timeout=1)
    close_thread.join(timeout=1)
    assert close_returned.is_set() is True
    assert runtime.status().state == "stopped"


def test_concurrent_closes_wait_for_external_cycle_on_running_runtime() -> None:
    entered = threading.Event()
    release = threading.Event()

    def block_second(call_number: int) -> None:
        if call_number == 2:
            entered.set()
            release.wait(timeout=1)

    runtime = NodeModelPrestageCompactionRuntime(
        store=ScriptedCompactionStore([(), ()], on_call=block_second),
        config=_config(interval_seconds=20, readiness_max_staleness_seconds=25),
        clock=lambda: _NOW,
        monotonic=time.monotonic,
    )
    runtime.start()
    _wait_until(lambda: runtime.status().cycles_succeeded == 1)
    cycle_thread = threading.Thread(target=runtime.run_once)
    cycle_thread.start()
    assert entered.wait(timeout=1)

    close_returned = (threading.Event(), threading.Event())

    def close_runtime(index: int) -> None:
        runtime.close()
        close_returned[index].set()

    close_threads = tuple(
        threading.Thread(target=close_runtime, args=(index,)) for index in range(2)
    )
    for thread in close_threads:
        thread.start()
    _wait_until(lambda: runtime.status().state == "stopping")
    assert not any(event.is_set() for event in close_returned)

    release.set()
    cycle_thread.join(timeout=1)
    for thread in close_threads:
        thread.join(timeout=1)
    assert all(event.is_set() for event in close_returned)
    assert runtime.status().state == "stopped"


def test_monitoring_view_is_bounded_deep_validated_and_order_agnostic() -> None:
    first = _mark("placement-00")
    second = _mark("placement-01")
    store = ScriptedCompactionStore([], monitoring=(first, second))
    runtime = NodeModelPrestageCompactionRuntime(
        store=store,
        config=_config(),
    )
    assert runtime.list_high_water_marks_page(
        after_placement_id="placement-before",
        limit=2,
    ) == (first, second)
    assert store.monitoring_calls == [("placement-before", 2)]

    unordered = NodeModelPrestageCompactionRuntime(
        store=ScriptedCompactionStore([], monitoring=(second, first)),
        config=_config(),
    )
    assert unordered.list_high_water_marks_page() == (second, first)


def test_start_failure_enters_a_closeable_failed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = NodeModelPrestageCompactionRuntime(
        store=ScriptedCompactionStore([]),
        config=_config(),
    )

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        runtime.start()
    assert runtime.status().state == "failed"
    runtime.close()
    assert runtime.status().state == "stopped"


@pytest.mark.parametrize(
    "marks, expected",
    [
        ((_mark("placement-new", updated_at=_NOW - timedelta(seconds=30)),), "cutoff"),
        ((_mark("placement-00"), _mark("placement-00")), "duplicate"),
    ],
)
def test_cycle_rejects_invalid_store_results(
    marks: tuple[NodeModelPrestageHighWaterMark, ...],
    expected: str,
) -> None:
    runtime = NodeModelPrestageCompactionRuntime(
        store=ScriptedCompactionStore([marks]),
        config=_config(),
        clock=lambda: _NOW,
        monotonic=lambda: 1.0,
    )
    with pytest.raises(RuntimeError, match=expected):
        runtime.run_once()
    assert runtime.status().cycles_failed == 1

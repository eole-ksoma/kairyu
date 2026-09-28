"""WP4.5 capacity watermark and generation-fenced eviction coverage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from kairyu.artifacts import (
    NodeModelCacheCapacityPolicy,
    NodeModelCacheEvictionError,
    NodeModelCacheEvictionPlan,
    NodeModelCacheEvictionPlanStaleError,
    NodeModelCacheEvictor,
    NodeModelCacheIndex,
    NodeModelCacheIndexEvictionConflictError,
    NodeModelCacheIndexPinnedError,
    plan_node_model_cache_eviction,
)


class MutableClock:
    def __init__(self, value: int = 100) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def _index(tmp_path: Path, clock: MutableClock | None = None) -> NodeModelCacheIndex:
    return NodeModelCacheIndex(
        tmp_path / "cache/cache-index.sqlite3",
        node_id="node-a",
        clock_ns=clock or MutableClock(),
    )


def _record(
    index: NodeModelCacheIndex,
    digest_character: str,
    *,
    total_bytes: int = 40,
):
    digest = digest_character * 64
    published = index.path.parent / "artifacts" / digest
    (published / "tree").mkdir(mode=0o700, parents=True)
    return index.record_verified(
        manifest_digest=digest,
        model_id="org/model",
        model_revision=f"release-{digest_character}",
        artifact_path=published / "tree",
        total_bytes=total_bytes,
        file_count=1,
        verification_source="filled",
    )


def _policy(*, high: int = 100, low: int = 50) -> NodeModelCacheCapacityPolicy:
    return NodeModelCacheCapacityPolicy(
        high_watermark_bytes=high,
        low_watermark_bytes=low,
    )


def test_policy_requires_strict_hysteresis_and_integer_bytes():
    with pytest.raises(ValidationError, match="lower than high"):
        _policy(high=100, low=100)
    with pytest.raises(ValidationError, match="must be an integer"):
        NodeModelCacheCapacityPolicy(
            high_watermark_bytes=True,
            low_watermark_bytes=50,
        )


def test_plan_starts_only_above_high_and_reclaims_to_low_in_lru_order(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock)
    oldest = _record(index, "a")
    clock.value = 200
    middle = _record(index, "b")
    clock.value = 300
    newest = _record(index, "c")

    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())

    assert plan.triggered is True
    assert plan.observed_used_bytes == 120
    assert plan.target_reclaim_bytes == 70
    assert plan.planned_reclaim_bytes == 80
    assert plan.blocked_reclaim_bytes == 0
    assert tuple(victim.manifest_digest for victim in plan.victims) == (
        oldest.manifest_digest,
        middle.manifest_digest,
    )
    assert newest.manifest_digest not in {victim.manifest_digest for victim in plan.victims}

    boundary = plan_node_model_cache_eviction(index.snapshot(), _policy(high=120, low=50))
    assert boundary.triggered is False
    assert boundary.victims == ()


def test_active_rollback_and_manual_pins_are_never_selected(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock)
    active = _record(index, "a")
    clock.value = 200
    rollback = _record(index, "b")
    clock.value = 300
    eligible = _record(index, "c")
    index.pin(active.manifest_digest, owner="deployment/active", reason="active")
    index.pin(rollback.manifest_digest, owner="deployment/rollback", reason="rollback")

    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())

    assert tuple(victim.manifest_digest for victim in plan.victims) == (eligible.manifest_digest,)
    assert plan.planned_reclaim_bytes == 40
    assert plan.blocked_reclaim_bytes == 30


def test_executor_detaches_tree_deletes_exact_generation_and_advances_revision(
    tmp_path: Path,
):
    index = _index(tmp_path)
    record = _record(index, "a", total_bytes=120)
    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())
    source_revision = plan.index_revision

    result = NodeModelCacheEvictor(index.path.parent, index).execute(plan)

    assert result.source_index_revision == source_revision
    assert result.ending_index_revision == source_revision + 1
    assert result.reclaimed_bytes == 120
    assert result.evicted == plan.victims
    assert index.get(record.manifest_digest) is None
    assert not record.artifact_path.parent.exists()
    assert tuple((index.path.parent / ".evicting").iterdir()) == ()


def test_executor_rejects_stale_plan_before_any_filesystem_change(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock)
    record = _record(index, "a", total_bytes=120)
    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())
    clock.value = 200
    index.touch(record.manifest_digest)

    with pytest.raises(NodeModelCacheEvictionPlanStaleError, match="revision changed"):
        NodeModelCacheEvictor(index.path.parent, index).execute(plan)

    assert index.get(record.manifest_digest) is not None
    assert record.artifact_path.parent.is_dir()


def test_transactional_revision_fence_rejects_victim_change_after_plan_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = MutableClock(100)
    index = _index(tmp_path, clock)
    record = _record(index, "a", total_bytes=120)
    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())
    evictor = NodeModelCacheEvictor(index.path.parent, index)
    original = evictor._evict_one

    def mutate_before_fence(victim, *, expected_index_revision):
        clock.value = 200
        index.touch(record.manifest_digest)
        original(victim, expected_index_revision=expected_index_revision)

    monkeypatch.setattr(evictor, "_evict_one", mutate_before_fence)

    with pytest.raises(NodeModelCacheIndexEvictionConflictError, match="revision changed"):
        evictor.execute(plan)

    assert index.get(record.manifest_digest) is not None
    assert record.artifact_path.parent.is_dir()


def test_pin_fence_rejects_late_pin_without_detach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index = _index(tmp_path)
    record = _record(index, "a", total_bytes=120)
    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())
    evictor = NodeModelCacheEvictor(index.path.parent, index)
    original = evictor._evict_one

    def pin_before_fence(victim, *, expected_index_revision):
        index.pin(record.manifest_digest, owner="deployment/active", reason="active")
        original(victim, expected_index_revision=expected_index_revision)

    monkeypatch.setattr(evictor, "_evict_one", pin_before_fence)

    with pytest.raises(NodeModelCacheIndexEvictionConflictError, match="revision changed"):
        evictor.execute(plan)

    assert index.get(record.manifest_digest) is not None
    assert record.artifact_path.parent.is_dir()


def test_index_fence_rejects_current_pinned_generation(tmp_path: Path):
    index = _index(tmp_path)
    record = _record(index, "a", total_bytes=120)
    pinned = index.pin(record.manifest_digest, owner="deployment/active", reason="active")

    with pytest.raises(NodeModelCacheIndexPinnedError, match="cannot be evicted"):
        with index.fenced_eviction(
            record.manifest_digest,
            expected_index_revision=index.snapshot().revision,
            expected_generation=pinned.generation,
        ):
            pass


def test_executor_rejects_forged_noncanonical_victim_set(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock)
    _record(index, "a")
    clock.value = 200
    second = _record(index, "b")
    clock.value = 300
    _record(index, "c")
    canonical = plan_node_model_cache_eviction(index.snapshot(), _policy())
    payload = canonical.model_dump()
    payload["victims"] = payload["victims"][1:]
    payload["planned_reclaim_bytes"] = second.total_bytes
    payload["blocked_reclaim_bytes"] = canonical.target_reclaim_bytes - second.total_bytes
    forged = NodeModelCacheEvictionPlan.model_validate(payload)

    with pytest.raises(NodeModelCacheEvictionPlanStaleError, match="not canonical"):
        NodeModelCacheEvictor(index.path.parent, index).execute(forged)

    assert len(index.snapshot().records) == 3


def test_executor_stops_remaining_victims_after_unrelated_mid_plan_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = MutableClock(100)
    index = _index(tmp_path, clock)
    first = _record(index, "a")
    clock.value = 200
    second = _record(index, "b")
    clock.value = 300
    unrelated = _record(index, "c")
    plan = plan_node_model_cache_eviction(index.snapshot(), _policy())
    evictor = NodeModelCacheEvictor(index.path.parent, index)
    original = evictor._evict_one
    calls = 0

    def mutate_after_first(victim, *, expected_index_revision):
        nonlocal calls
        original(victim, expected_index_revision=expected_index_revision)
        calls += 1
        if calls == 1:
            clock.value = 400
            index.touch(unrelated.manifest_digest)

    monkeypatch.setattr(evictor, "_evict_one", mutate_after_first)

    with pytest.raises(NodeModelCacheEvictionPlanStaleError, match="during eviction"):
        evictor.execute(plan)

    assert index.get(first.manifest_digest) is None
    assert index.get(second.manifest_digest) is not None
    assert second.artifact_path.parent.is_dir()


def test_recovery_restores_rolled_back_detach_and_removes_committed_tombstone(
    tmp_path: Path,
):
    index = _index(tmp_path)
    record = _record(index, "a", total_bytes=120)
    evictor = NodeModelCacheEvictor(index.path.parent, index)
    evictor.recover_interrupted()
    index_revision = index.snapshot().revision
    tombstone = (
        index.path.parent
        / ".evicting"
        / f"{record.manifest_digest}.{index_revision}.{record.generation}"
    )
    os.rename(record.artifact_path.parent, tombstone)

    evictor.recover_interrupted()

    assert record.artifact_path.parent.is_dir()
    assert not tombstone.exists()

    os.rename(record.artifact_path.parent, tombstone)
    with index.fenced_eviction(
        record.manifest_digest,
        expected_index_revision=index_revision,
        expected_generation=record.generation,
    ):
        pass

    evictor.recover_interrupted()

    assert index.get(record.manifest_digest) is None
    assert not tombstone.exists()


def test_recovery_preserves_tombstone_when_published_path_is_unsafe(tmp_path: Path):
    index = _index(tmp_path)
    record = _record(index, "a", total_bytes=120)
    evictor = NodeModelCacheEvictor(index.path.parent, index)
    evictor.recover_interrupted()
    index_revision = index.snapshot().revision
    published = record.artifact_path.parent
    tombstone = (
        index.path.parent
        / ".evicting"
        / f"{record.manifest_digest}.{index_revision}.{record.generation}"
    )
    os.rename(published, tombstone)
    published.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(
        NodeModelCacheEvictionError,
        match="published cache entry is not a directory",
    ):
        evictor.recover_interrupted()

    assert tombstone.is_dir()
    assert published.is_symlink()


def test_recovery_does_not_restore_old_tree_after_generation_reset_refill(tmp_path: Path):
    index = _index(tmp_path)
    old = _record(index, "a", total_bytes=120)
    evictor = NodeModelCacheEvictor(index.path.parent, index)
    evictor.recover_interrupted()
    index_revision = index.snapshot().revision
    tombstone = (
        index.path.parent / ".evicting" / f"{old.manifest_digest}.{index_revision}.{old.generation}"
    )
    os.rename(old.artifact_path.parent, tombstone)
    with index.fenced_eviction(
        old.manifest_digest,
        expected_index_revision=index_revision,
        expected_generation=old.generation,
    ):
        pass
    refilled = _record(index, "a", total_bytes=120)

    evictor.recover_interrupted()

    assert refilled.generation == old.generation == 1
    assert index.get(old.manifest_digest) == refilled
    assert refilled.artifact_path.parent.is_dir()
    assert not tombstone.exists()

"""WP4.4 verified node-cache placement hint coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kairyu.artifacts import (
    NodeModelCacheIndex,
    NodeModelCachePlacementHintPublisher,
    NodeModelCachePlacementHintSnapshot,
)
from kairyu.runners import (
    ModelCachePlacementCandidate,
    ModelCachePlacementState,
    build_cache_placement_snapshot,
    plan_cache_aware_scale_up,
)

NOW = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _index(tmp_path: Path, *, node_id: str = "gpu-node-a") -> NodeModelCacheIndex:
    return NodeModelCacheIndex(
        tmp_path / node_id / "cache-index.sqlite3",
        node_id=node_id,
        clock_ns=lambda: 100,
    )


def _record(
    index: NodeModelCacheIndex,
    *,
    digest: str = DIGEST,
    revision: str = "release-1",
):
    return index.record_verified(
        manifest_digest=digest,
        model_id="org/model",
        model_revision=revision,
        artifact_path=index.path.parent / "artifacts" / digest / "tree",
        total_bytes=27,
        file_count=2,
        verification_source="filled",
    )


def _hint(
    tmp_path: Path,
    *,
    node_id: str = "gpu-node-a",
    observed_at: datetime = NOW,
    record: bool = True,
) -> NodeModelCachePlacementHintSnapshot:
    index = _index(tmp_path, node_id=node_id)
    if record:
        _record(index)
    return NodeModelCachePlacementHintPublisher(
        index,
        ttl_seconds=60,
        clock=lambda: observed_at,
    ).snapshot()


def _candidate(
    suffix: str,
    *,
    node_name: str | None = None,
    **updates,
) -> ModelCachePlacementCandidate:
    values = {
        "placement_id": f"placement-{suffix}",
        "node_name": node_name or f"gpu-node-{suffix}",
        "resource_flavor": "h100-sxm",
        "profile_id": "h100-sxm-tp1",
        "compatibility_approval_id": "compat-h100-model-v1",
    }
    values.update(updates)
    return ModelCachePlacementCandidate(**values)


def _aggregate(
    hints: tuple[NodeModelCachePlacementHintSnapshot, ...],
    candidates: tuple[ModelCachePlacementCandidate, ...],
    *,
    observed_at: datetime = NOW,
    digest: str = DIGEST,
    model_id: str = "org/model",
    revision: str = "release-1",
):
    return build_cache_placement_snapshot(
        hints,
        candidates,
        snapshot_id="controller-cache-17",
        cache_revision=17,
        observed_at=observed_at,
        model_class="model-class-a",
        model_id=model_id,
        model_revision=revision,
        artifact_digest=digest,
        placement_binding_id="deployment/model-a",
    )


def test_publisher_exposes_only_verified_path_free_residency(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)
    _record(index, digest=OTHER_DIGEST, revision="release-2")
    index.pin(DIGEST, owner="deployment/a", reason="active")
    index.mark_unverified(OTHER_DIGEST, reason="digest mismatch")

    snapshot = NodeModelCachePlacementHintPublisher(
        index,
        ttl_seconds=30,
        clock=lambda: NOW,
    ).snapshot()

    assert snapshot.node_id == "gpu-node-a"
    assert snapshot.index_revision == index.snapshot().revision
    assert snapshot.observed_at == NOW
    assert snapshot.valid_until == NOW + timedelta(seconds=30)
    assert len(snapshot.residents) == 1
    resident = snapshot.residents[0]
    assert resident.manifest_digest == DIGEST
    assert resident.verified is True
    assert resident.pinned is True
    assert "artifact_path" not in resident.model_dump()
    assert "pin_owners" not in resident.model_dump()


def test_publisher_rejects_unbounded_or_naive_freshness(tmp_path: Path):
    index = _index(tmp_path)

    with pytest.raises(ValueError, match="1 to 300"):
        NodeModelCachePlacementHintPublisher(index, ttl_seconds=301)
    with pytest.raises(ValueError, match="timezone-aware"):
        NodeModelCachePlacementHintPublisher(
            index,
            clock=lambda: datetime(2026, 9, 28, 9, 0),
        ).snapshot()


def test_hint_schema_rejects_unverified_or_overlong_publication(tmp_path: Path):
    payload = _hint(tmp_path).model_dump()
    payload["residents"][0]["verified"] = False
    with pytest.raises(ValidationError):
        NodeModelCachePlacementHintSnapshot.model_validate(payload)

    payload = _hint(tmp_path).model_dump()
    payload["residents"][0]["verified"] = 1
    with pytest.raises(ValidationError, match="verified must be true"):
        NodeModelCachePlacementHintSnapshot.model_validate(payload)

    payload = _hint(tmp_path).model_dump()
    payload["valid_until"] = NOW + timedelta(seconds=301)
    with pytest.raises(ValidationError, match="lifetime"):
        NodeModelCachePlacementHintSnapshot.model_validate(payload)


def test_controller_joins_exact_fresh_residency_in_canonical_placement_order(
    tmp_path: Path,
):
    snapshot = _aggregate(
        (_hint(tmp_path, node_id="gpu-node-a"),),
        (_candidate("b"), _candidate("a")),
    )

    assert tuple(placement.placement_id for placement in snapshot.placements) == (
        "placement-a",
        "placement-b",
    )
    assert snapshot.placements[0].state is ModelCachePlacementState.READY
    assert snapshot.placements[1].state is ModelCachePlacementState.ABSENT


@pytest.mark.parametrize(
    ("observed_at", "digest", "model_id", "revision"),
    [
        (NOW + timedelta(seconds=60), DIGEST, "org/model", "release-1"),
        (NOW - timedelta(seconds=1), DIGEST, "org/model", "release-1"),
        (NOW, OTHER_DIGEST, "org/model", "release-1"),
        (NOW, DIGEST, "org/other-model", "release-1"),
        (NOW, DIGEST, "org/model", "release-2"),
    ],
)
def test_stale_future_or_non_exact_hint_never_claims_ready(
    tmp_path: Path,
    observed_at: datetime,
    digest: str,
    model_id: str,
    revision: str,
):
    snapshot = _aggregate(
        (_hint(tmp_path),),
        (_candidate("a"),),
        observed_at=observed_at,
        digest=digest,
        model_id=model_id,
        revision=revision,
    )

    assert snapshot.placements[0].state is ModelCachePlacementState.ABSENT


def test_cache_locality_does_not_override_controller_health_or_assignment(
    tmp_path: Path,
):
    snapshot = _aggregate(
        (_hint(tmp_path),),
        (
            _candidate("a", healthy=False),
            _candidate("b", node_name="gpu-node-a", assigned=True),
            _candidate("c", node_name="gpu-node-a"),
            _candidate("d", node_name="gpu-node-a", schedulable=False),
            _candidate("e", node_name="gpu-node-a", resource_flavor="l40s"),
        ),
    )
    plan = plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=5,
        resource_flavor="h100-sxm",
    )

    assert all(
        placement.state is ModelCachePlacementState.READY for placement in snapshot.placements
    )
    assert plan.runner_start_placement_ids == ("placement-c",)
    assert plan.unplanned_replicas == 4


def test_duplicate_node_publications_or_placement_ids_fail_closed(tmp_path: Path):
    hint = _hint(tmp_path)
    with pytest.raises(ValueError, match="unique node IDs"):
        _aggregate((hint, hint), (_candidate("a"),))
    with pytest.raises(ValueError, match="unique placement IDs"):
        _aggregate((hint,), (_candidate("a"), _candidate("a")))

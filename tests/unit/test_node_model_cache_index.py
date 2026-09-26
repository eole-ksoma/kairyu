"""WP4.3 durable node-local model cache index coverage."""

from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from kairyu.artifacts import (
    NodeModelCacheIndex,
    NodeModelCacheIndexEntryNotFoundError,
    NodeModelCacheIndexError,
    NodeModelCacheIndexIdentityError,
    NodeModelCacheIndexUnverifiedError,
)


class MutableClock:
    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def _index(tmp_path: Path, *, clock: MutableClock | None = None) -> NodeModelCacheIndex:
    return NodeModelCacheIndex(
        tmp_path / "cache/cache-index.sqlite3",
        node_id="node-a",
        clock_ns=clock or MutableClock(100),
    )


def _record(
    index: NodeModelCacheIndex,
    *,
    digest: str = "a" * 64,
    model_id: str = "org/model",
    model_revision: str = "release-1",
    total_bytes: int = 27,
    file_count: int = 2,
    verification_source: str = "filled",
):
    return index.record_verified(
        manifest_digest=digest,
        model_id=model_id,
        model_revision=model_revision,
        artifact_path=index.path.parent / "artifacts" / digest / "tree",
        total_bytes=total_bytes,
        file_count=file_count,
        verification_source=verification_source,
    )


def test_record_verified_persists_node_model_bytes_and_access_time(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock=clock)
    clock.value = 200

    created = _record(index)
    reopened = NodeModelCacheIndex(index.path, node_id="node-a", clock_ns=clock)

    assert reopened.get("a" * 64) == created
    assert created.node_id == "node-a"
    assert created.model_id == "org/model"
    assert created.model_revision == "release-1"
    assert created.total_bytes == 27
    assert created.file_count == 2
    assert created.verified is True
    assert created.verification_source == "filled"
    assert created.verified_at_ns == 200
    assert created.last_access_at_ns == 200
    assert created.pin_owners == ()
    assert created.pinned is False
    assert created.generation == 1


def test_identical_record_touches_without_downgrading_fill_evidence(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock=clock)
    first = _record(index)
    clock.value = 200

    touched = _record(index, verification_source="published_marker")

    assert touched.verified_at_ns == first.verified_at_ns
    assert touched.last_access_at_ns == 200
    assert touched.verification_source == "filled"
    assert touched.generation == first.generation + 1


def test_clock_regression_does_not_move_last_access_or_generation_backward(tmp_path: Path):
    clock = MutableClock(200)
    index = _index(tmp_path, clock=clock)
    created = _record(index)
    clock.value = 100

    touched = index.touch("a" * 64)

    assert touched.last_access_at_ns == created.last_access_at_ns
    assert touched.generation == created.generation


def test_digest_identity_conflict_is_rejected_without_overwrite(tmp_path: Path):
    index = _index(tmp_path)
    original = _record(index)

    with pytest.raises(NodeModelCacheIndexIdentityError, match="different cache metadata"):
        _record(index, model_revision="release-2")

    assert index.get("a" * 64) == original


def test_artifact_path_is_bound_to_index_root_and_digest(tmp_path: Path):
    index = _index(tmp_path)

    with pytest.raises(NodeModelCacheIndexIdentityError, match="cache root and digest"):
        index.record_verified(
            manifest_digest="a" * 64,
            model_id="org/model",
            model_revision="release-1",
            artifact_path=tmp_path / "outside/tree",
            total_bytes=1,
            file_count=1,
            verification_source="filled",
        )


def test_owner_scoped_pins_are_idempotent_and_do_not_release_each_other(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock=clock)
    created = _record(index)
    first = index.pin("a" * 64, owner="deployment/a", reason="active")
    repeated = index.pin("a" * 64, owner="deployment/a", reason="active")
    second = index.pin("a" * 64, owner="rollback/a", reason="rollback-target")

    assert first.pin_owners == ("deployment/a",)
    assert repeated.generation == first.generation
    assert second.pin_owners == ("deployment/a", "rollback/a")
    assert second.pinned is True
    assert second.generation == created.generation + 2

    one_left = index.unpin("a" * 64, owner="deployment/a")
    repeated_release = index.unpin("a" * 64, owner="deployment/a")
    none_left = index.unpin("a" * 64, owner="rollback/a")

    assert one_left.pin_owners == ("rollback/a",)
    assert repeated_release.generation == one_left.generation
    assert none_left.pin_owners == ()
    assert none_left.pinned is False


def test_unknown_entry_cannot_be_touched_or_pinned(tmp_path: Path):
    index = _index(tmp_path)

    with pytest.raises(NodeModelCacheIndexEntryNotFoundError, match="does not exist"):
        index.touch("b" * 64)
    with pytest.raises(NodeModelCacheIndexEntryNotFoundError, match="does not exist"):
        index.pin("b" * 64, owner="deployment/a", reason="active")


def test_unverified_state_blocks_access_until_digest_verified_refill(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock=clock)
    created = _record(index)

    unverified = index.mark_unverified("a" * 64, reason="same-size digest mismatch")

    assert unverified.verified is False
    assert unverified.verification_failure == "same-size digest mismatch"
    assert unverified.generation == created.generation + 1
    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="cannot be touched"):
        index.touch("a" * 64)
    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="cannot be pinned"):
        index.pin("a" * 64, owner="deployment/a", reason="active")
    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="verified refill"):
        _record(index, verification_source="published_marker")

    clock.value = 200
    recovered = _record(index, verification_source="filled")
    assert recovered.verified is True
    assert recovered.verification_failure is None
    assert recovered.verification_source == "filled"
    assert recovered.verified_at_ns == 200


def test_index_is_bound_to_one_node_identity(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)

    with pytest.raises(NodeModelCacheIndexIdentityError, match="another schema, node"):
        NodeModelCacheIndex(index.path, node_id="node-b")


def test_index_is_bound_to_its_cache_root_even_after_database_copy(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)
    copied_path = tmp_path / "other-cache/cache-index.sqlite3"
    copied_path.parent.mkdir(mode=0o700)
    with (
        sqlite3.connect(index.path) as source,
        sqlite3.connect(copied_path) as destination,
    ):
        source.backup(destination)
    copied_path.chmod(0o600)

    with pytest.raises(NodeModelCacheIndexIdentityError, match="cache root"):
        NodeModelCacheIndex(copied_path, node_id="node-a")


def test_list_records_has_stable_model_revision_digest_order(tmp_path: Path):
    index = _index(tmp_path)
    _record(
        index,
        digest="c" * 64,
        model_id="z/model",
        model_revision="r2",
    )
    _record(
        index,
        digest="b" * 64,
        model_id="a/model",
        model_revision="r2",
    )
    _record(
        index,
        digest="a" * 64,
        model_id="a/model",
        model_revision="r1",
    )

    assert [record.manifest_digest for record in index.list_records()] == [
        "a" * 64,
        "b" * 64,
        "c" * 64,
    ]


def test_concurrent_owner_pins_are_serialized_without_loss(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)
    owners = [f"deployment/{position:02d}" for position in range(20)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = tuple(
            pool.map(
                lambda owner: index.pin("a" * 64, owner=owner, reason="active"),
                owners,
            )
        )

    assert all(record.pinned for record in records)
    assert index.get("a" * 64).pin_owners == tuple(owners)


def test_get_reads_entry_and_pins_from_one_generation_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index = _index(tmp_path)
    created = _record(index)
    writer = NodeModelCacheIndex(index.path, node_id="node-a")
    entry_selected = Event()
    continue_read = Event()
    original = index._record_from_row

    def blocked_record(connection, row):
        entry_selected.set()
        assert continue_read.wait(timeout=5)
        return original(connection, row)

    monkeypatch.setattr(index, "_record_from_row", blocked_record)
    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(index.get, "a" * 64)
        assert entry_selected.wait(timeout=5)
        pinned = writer.pin("a" * 64, owner="deployment/a", reason="active")
        continue_read.set()
        snapshot = read.result(timeout=5)

    assert snapshot.generation == created.generation
    assert snapshot.pin_owners == ()
    assert pinned.generation == created.generation + 1
    assert pinned.pin_owners == ("deployment/a",)


def test_schema_initialization_rolls_back_as_one_transaction(tmp_path: Path):
    class FailingClockIndex(NodeModelCacheIndex):
        def _now_ns(self) -> int:
            raise RuntimeError("injected initialization failure")

    path = tmp_path / "cache/cache-index.sqlite3"
    with pytest.raises(RuntimeError, match="injected initialization failure"):
        FailingClockIndex(path, node_id="node-a")

    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        assert tables == []
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0

    assert NodeModelCacheIndex(path, node_id="node-a").list_records() == ()


def test_concurrent_constructors_share_one_complete_schema(tmp_path: Path):
    path = tmp_path / "cache/cache-index.sqlite3"

    with ThreadPoolExecutor(max_workers=8) as pool:
        indexes = tuple(
            pool.map(
                lambda _: NodeModelCacheIndex(path, node_id="node-a"),
                range(16),
            )
        )

    assert all(index.list_records() == () for index in indexes)


def test_sqlite_contract_uses_wal_full_sync_and_foreign_keys(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)

    with sqlite3.connect(index.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x4B414943
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"cache_index_meta", "cache_entries", "cache_pins"} <= tables


def test_insecure_or_hardlinked_index_file_is_rejected(tmp_path: Path):
    index = _index(tmp_path)
    index.path.chmod(0o666)
    with pytest.raises(NodeModelCacheIndexError, match="group/world writable"):
        index.get("a" * 64)

    index.path.chmod(0o600)
    alias = tmp_path / "index-alias.sqlite3"
    os.link(index.path, alias)
    with pytest.raises(NodeModelCacheIndexError, match="unsafe link count"):
        NodeModelCacheIndex(index.path, node_id="node-a")


def test_corrupt_sqlite_fails_closed(tmp_path: Path):
    index = _index(tmp_path)
    index.path.write_bytes(b"not a sqlite database")

    with pytest.raises(NodeModelCacheIndexError):
        index.get("a" * 64)

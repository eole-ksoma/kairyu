"""WP4.3 durable node-local model cache index coverage."""

from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import kairyu.artifacts.cache_index as cache_index_module
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


def _legacy_v1_index(
    tmp_path: Path,
    *,
    generation: int = 1,
    pinned: bool = False,
) -> Path:
    path = tmp_path / "cache/cache-index.sqlite3"
    path.parent.mkdir(mode=0o700, parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA application_id = 1262569795;
            PRAGMA user_version = 1;
            CREATE TABLE cache_index_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL,
                node_id TEXT NOT NULL,
                cache_root TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL CHECK (created_at_ns >= 0)
            );
            CREATE TABLE cache_entries (
                manifest_digest TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
                file_count INTEGER NOT NULL CHECK (file_count >= 1),
                verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
                verification_source TEXT NOT NULL CHECK (
                    verification_source IN ('filled', 'published_marker')
                ),
                verification_failure TEXT,
                verified_at_ns INTEGER NOT NULL CHECK (verified_at_ns >= 0),
                last_access_at_ns INTEGER NOT NULL CHECK (
                    last_access_at_ns >= verified_at_ns
                ),
                generation INTEGER NOT NULL CHECK (generation >= 1)
            );
            CREATE TABLE cache_pins (
                manifest_digest TEXT NOT NULL REFERENCES cache_entries(
                    manifest_digest
                ) ON DELETE CASCADE,
                owner TEXT NOT NULL,
                reason TEXT NOT NULL,
                pinned_at_ns INTEGER NOT NULL CHECK (pinned_at_ns >= 0),
                PRIMARY KEY (manifest_digest, owner)
            );
            CREATE INDEX cache_entries_lru
                ON cache_entries(last_access_at_ns, manifest_digest);
            CREATE INDEX cache_entries_model
                ON cache_entries(model_id, model_revision, manifest_digest);
            """
        )
        connection.execute(
            """
            INSERT INTO cache_index_meta (
                singleton, schema_version, node_id, cache_root, created_at_ns
            ) VALUES (1, ?, ?, ?, 100)
            """,
            ("kairyu-node-model-cache-index-v1", "node-a", str(path.parent)),
        )
        connection.execute(
            """
            INSERT INTO cache_entries (
                manifest_digest, model_id, model_revision, artifact_path,
                total_bytes, file_count, verified, verification_source,
                verification_failure, verified_at_ns, last_access_at_ns, generation
            ) VALUES (?, 'org/model', 'release-1', ?, 27, 2, 1, 'filled',
                      NULL, 100, 100, ?)
            """,
            (
                "a" * 64,
                str(path.parent / "artifacts" / ("a" * 64) / "tree"),
                generation,
            ),
        )
        if pinned:
            connection.execute(
                """
                INSERT INTO cache_pins (manifest_digest, owner, reason, pinned_at_ns)
                VALUES (?, 'deployment/a', 'active', 100)
                """,
                ("a" * 64,),
            )
    path.chmod(0o600)
    return path


def _legacy_v2_index(tmp_path: Path) -> Path:
    path = _legacy_v1_index(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cache_index_meta SET schema_version = ? WHERE singleton = 1",
            ("kairyu-node-model-cache-index-v2",),
        )
        connection.execute("PRAGMA user_version = 2")
        connection.execute(
            """
            CREATE TABLE cache_index_revision (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                revision INTEGER NOT NULL CHECK (revision BETWEEN 1 AND 9223372036854775807)
            )
            """
        )
        connection.execute("INSERT INTO cache_index_revision (singleton, revision) VALUES (1, 2)")
        events = {
            "cache_entries_revision_insert": "AFTER INSERT ON cache_entries",
            "cache_entries_revision_update": (
                "AFTER UPDATE OF last_access_at_ns, verification_source, verified, "
                "verification_failure, verified_at_ns ON cache_entries"
            ),
            "cache_entries_revision_delete": "AFTER DELETE ON cache_entries",
            "cache_pins_revision_insert": "AFTER INSERT ON cache_pins",
            "cache_pins_revision_update": "AFTER UPDATE ON cache_pins",
            "cache_pins_revision_delete": "AFTER DELETE ON cache_pins",
        }
        for name, event in events.items():
            connection.execute(
                f"""
                CREATE TRIGGER {name} {event}
                BEGIN
                    UPDATE cache_index_revision SET revision = revision + 1
                    WHERE singleton = 1;
                END
                """
            )
    return path


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


def test_snapshot_revision_advances_only_with_observable_index_changes(tmp_path: Path):
    clock = MutableClock(100)
    index = _index(tmp_path, clock=clock)

    empty = index.snapshot()
    created = _record(index)
    after_create = index.snapshot()
    unchanged = index.touch("a" * 64)
    after_unchanged = index.snapshot()
    pinned = index.pin("a" * 64, owner="deployment/a", reason="active")
    after_pin = index.snapshot()

    assert empty.revision == 1
    assert empty.records == ()
    assert after_create.revision == 2
    assert after_create.records == (created,)
    assert unchanged.generation == created.generation
    assert after_unchanged.revision == after_create.revision
    assert after_pin.revision == 3
    assert after_pin.records == (pinned,)
    assert NodeModelCacheIndex(index.path, node_id="node-a").snapshot() == after_pin


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


def test_recovery_required_state_needs_exact_audited_completion(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)

    recovery = index.begin_recovery("a" * 64, reason="digest_mismatch")

    assert recovery.verified is False
    assert recovery.recovery_id is not None
    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="fenced audit completion"):
        _record(index, verification_source="filled")
    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="fence no longer matches"):
        index.complete_recovery(
            manifest_digest="a" * 64,
            recovery_id=recovery.recovery_id,
            expected_generation=recovery.generation + 1,
            model_id="org/model",
            model_revision="release-1",
            artifact_path=index.path.parent / "artifacts" / ("a" * 64) / "tree",
            total_bytes=27,
            file_count=2,
        )

    completed = index.complete_recovery(
        manifest_digest="a" * 64,
        recovery_id=recovery.recovery_id,
        expected_generation=recovery.generation,
        model_id="org/model",
        model_revision="release-1",
        artifact_path=index.path.parent / "artifacts" / ("a" * 64) / "tree",
        total_bytes=27,
        file_count=2,
    )
    assert completed.verified is True
    assert completed.recovery_id is None


def test_recovery_id_does_not_repeat_after_delete_and_generation_reset(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)
    first = index.begin_recovery("a" * 64, reason="digest_mismatch")
    assert first.recovery_id is not None
    completed = index.complete_recovery(
        manifest_digest="a" * 64,
        recovery_id=first.recovery_id,
        expected_generation=first.generation,
        model_id="org/model",
        model_revision="release-1",
        artifact_path=index.path.parent / "artifacts" / ("a" * 64) / "tree",
        total_bytes=27,
        file_count=2,
    )
    with index.fenced_eviction(
        "a" * 64,
        expected_index_revision=index.snapshot().revision,
        expected_generation=completed.generation,
    ):
        pass
    refilled = _record(index)
    assert refilled.generation == 1

    second = index.begin_recovery("a" * 64, reason="digest_mismatch")

    assert second.recovery_id is not None
    assert second.recovery_id != first.recovery_id


def test_v2_migration_guards_recovery_from_prechecked_old_writer(tmp_path: Path):
    path = _legacy_v2_index(tmp_path)
    old_writer = sqlite3.connect(path)
    try:
        assert old_writer.execute("PRAGMA user_version").fetchone()[0] == 2
        index = NodeModelCacheIndex(path, node_id="node-a")
        recovery = index.begin_recovery("a" * 64, reason="digest_mismatch")
        assert recovery.recovery_id is not None

        with pytest.raises(sqlite3.IntegrityError, match="recovery-required"):
            old_writer.execute(
                """
                UPDATE cache_entries
                SET verified = 1, verification_failure = NULL,
                    generation = generation + 1
                WHERE manifest_digest = ?
                """,
                ("a" * 64,),
            )
        old_writer.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="recovery-required"):
            old_writer.execute(
                "DELETE FROM cache_entries WHERE manifest_digest = ?",
                ("a" * 64,),
            )
    finally:
        old_writer.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            connection.execute(
                "SELECT schema_version FROM cache_index_meta WHERE singleton = 1"
            ).fetchone()[0]
            == "kairyu-node-model-cache-index-v3"
        )


def test_concurrent_v2_constructors_serialize_one_v3_migration(tmp_path: Path):
    path = _legacy_v2_index(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        migrated = tuple(
            pool.map(
                lambda _: NodeModelCacheIndex(path, node_id="node-a"),
                range(16),
            )
        )

    assert all(value.snapshot().records[0].recovery_id is None for value in migrated)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_v2_migration_ddl_failure_rolls_back_without_partial_v3(tmp_path: Path):
    path = _legacy_v2_index(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER cache_pins_revision_delete")

    with pytest.raises(NodeModelCacheIndexError, match="cannot initialize"):
        NodeModelCacheIndex(path, node_id="node-a")

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT schema_version FROM cache_index_meta WHERE singleton = 1"
            ).fetchone()[0]
            == "kairyu-node-model-cache-index-v2"
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cache_entries)").fetchall()
        }
        assert "recovery_id" not in columns


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


def test_snapshot_revision_and_records_are_read_from_one_sqlite_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index = _index(tmp_path)
    created = _record(index)
    before = index.snapshot()
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
        read = pool.submit(index.snapshot)
        assert entry_selected.wait(timeout=5)
        pinned = writer.pin("a" * 64, owner="deployment/a", reason="active")
        continue_read.set()
        snapshot = read.result(timeout=5)

    assert snapshot.revision == before.revision
    assert snapshot.records == (created,)
    assert writer.snapshot().revision == before.revision + 1
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


def test_wal_enable_retries_sqlite_lock_within_busy_timeout(monkeypatch):
    class LockingConnection:
        attempts = 0

        def execute(self, statement: str):
            assert statement == "PRAGMA journal_mode = WAL"
            self.attempts += 1
            if self.attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return self

        def fetchone(self):
            return ("wal",)

    monotonic_values = iter((1_000_000_000, 1_001_000_000, 1_002_000_000))
    sleeps: list[float] = []
    monkeypatch.setattr(
        cache_index_module.time,
        "monotonic_ns",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(cache_index_module.time, "sleep", sleeps.append)
    index = object.__new__(NodeModelCacheIndex)
    index._busy_timeout_ms = 50
    connection = LockingConnection()

    assert index._enable_wal_mode(connection) == "wal"
    assert connection.attempts == 3
    assert sleeps == [0.01, 0.01]


def test_legacy_v1_index_migrates_transactionally_and_fences_old_writers(
    tmp_path: Path,
):
    path = _legacy_v1_index(tmp_path, generation=2, pinned=True)

    migrated = NodeModelCacheIndex(path, node_id="node-a")
    snapshot = migrated.snapshot()

    assert snapshot.revision == 3
    assert snapshot.records[0].generation == 2
    assert snapshot.records[0].pin_owners == ("deployment/a",)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            connection.execute(
                "SELECT schema_version FROM cache_index_meta WHERE singleton = 1"
            ).fetchone()[0]
            == "kairyu-node-model-cache-index-v3"
        )
        entries_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cache_entries'"
        ).fetchone()[0]
        assert "generation BETWEEN 1 AND 9223372036854775807" in entries_sql
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE cache_entries SET generation = 9.223372036854776e18")


def test_concurrent_legacy_v1_constructors_serialize_one_migration(tmp_path: Path):
    path = _legacy_v1_index(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        migrated = tuple(
            pool.map(
                lambda _: NodeModelCacheIndex(path, node_id="node-a"),
                range(16),
            )
        )

    assert all(value.snapshot().records[0].manifest_digest == "a" * 64 for value in migrated)
    assert {value.snapshot().revision for value in migrated} == {2}


def test_legacy_revision_overflow_rolls_back_v2_migration(tmp_path: Path):
    path = _legacy_v1_index(tmp_path, generation=2**63 - 1)

    with pytest.raises(NodeModelCacheIndexIdentityError, match="represented safely"):
        NodeModelCacheIndex(path, node_id="node-a")

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT schema_version FROM cache_index_meta WHERE singleton = 1"
            ).fetchone()[0]
            == "kairyu-node-model-cache-index-v1"
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'cache_index_revision'"
            ).fetchone()
            is None
        )


def test_prechecked_legacy_writer_cannot_bypass_v2_revision_trigger(tmp_path: Path):
    path = _legacy_v1_index(tmp_path)
    legacy_connection = sqlite3.connect(path, isolation_level=None)
    try:
        assert legacy_connection.execute("PRAGMA user_version").fetchone()[0] == 1
        migrated = NodeModelCacheIndex(path, node_id="node-a")
        before = migrated.snapshot()

        legacy_connection.execute("BEGIN IMMEDIATE")
        legacy_connection.execute(
            """
            UPDATE cache_entries
            SET last_access_at_ns = 200, generation = generation + 1
            WHERE manifest_digest = ?
            """,
            ("a" * 64,),
        )
        legacy_connection.commit()

        after = migrated.snapshot()
        assert after.revision == before.revision + 1
        assert after.records[0].generation == 2
        assert after.records[0].last_access_at_ns == 200
    finally:
        legacy_connection.close()


def test_sqlite_contract_uses_wal_full_sync_and_foreign_keys(tmp_path: Path):
    index = _index(tmp_path)
    _record(index)

    with sqlite3.connect(index.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x4B414943
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {
        "cache_index_meta",
        "cache_entries",
        "cache_pins",
        "cache_index_revision",
    } <= tables


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

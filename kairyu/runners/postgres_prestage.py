"""PostgreSQL-backed node model pre-stage placement state."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from datetime import datetime
from types import TracebackType
from typing import Literal, Self

from kairyu.artifacts.node_cache import NodeModelCacheFillResult
from kairyu.runners.prestage import (
    NodeModelPrestageCommand,
    NodeModelPrestageHighWaterMark,
    NodeModelPrestageRecord,
    _aware,
    _copy_high_water_mark,
    _copy_prestage_record,
    _NodeModelPrestageTransitions,
    _text,
)

try:  # Optional deployment dependency.
    import psycopg  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - core-only installation.
    psycopg = None  # type: ignore[assignment]

_SCHEMA_VERSION = 2
_SCHEMA_NAME = "public"
_SCHEMA_LOCK = (1_261_587_810, 9)
_EXPECTED_COLUMNS = {
    "node_model_prestage_store_registry": (
        ("store_id", "text", True),
        ("schema_version", "integer", True),
        ("node_id", "text", True),
        ("max_placements", "bigint", True),
        ("created_at", "timestamp with time zone", True),
    ),
    "node_model_prestage_records": (
        ("store_id", "text", True),
        ("placement_id", "text", True),
        ("command_id", "text", True),
        ("command_generation", "bigint", True),
        ("fencing_token", "bigint", True),
        ("target_revision", "bigint", True),
        ("state", "text", True),
        ("updated_at", "timestamp with time zone", True),
        ("record", "jsonb", True),
        ("created_at", "timestamp with time zone", True),
    ),
    "node_model_prestage_high_water_marks": (
        ("store_id", "text", True),
        ("placement_id", "text", True),
        ("command_id", "text", True),
        ("command_generation", "bigint", True),
        ("election_id", "text", True),
        ("holder_id", "text", True),
        ("fencing_token", "bigint", True),
        ("target_id", "text", True),
        ("target_revision", "bigint", True),
        ("release_identity_digest", "text", True),
        ("attempt", "bigint", True),
        ("updated_at", "timestamp with time zone", True),
        ("compacted_at", "timestamp with time zone", True),
        ("mark", "jsonb", True),
        ("created_at", "timestamp with time zone", True),
    ),
}
_EXPECTED_REGISTRY_CONSTRAINTS = {
    ("c", "CHECK ((store_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((node_id <> ''::text))", 0, True, False, False),
    (
        "c",
        "CHECK (((max_placements > 0) AND (max_placements <= 100000)))",
        0,
        True,
        False,
        False,
    ),
    ("p", "PRIMARY KEY (store_id)", 0, True, False, False),
}
_EXPECTED_RECORD_CONSTRAINTS_WITHOUT_TARGET = {
    ("c", "CHECK ((placement_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((command_id <> ''::text))", 0, True, False, False),
    (
        "c",
        "CHECK ((command_generation > 0))",
        0,
        True,
        False,
        False,
    ),
    ("c", "CHECK ((fencing_token > 0))", 0, True, False, False),
    ("c", "CHECK ((target_revision > 0))", 0, True, False, False),
    (
        "c",
        "CHECK ((state = ANY (ARRAY['absent'::text, 'filling'::text, "
        "'ready'::text, 'failed'::text])))",
        0,
        True,
        False,
        False,
    ),
    ("p", "PRIMARY KEY (store_id, placement_id)", 0, True, False, False),
}
_EXPECTED_HIGH_WATER_CONSTRAINTS_WITHOUT_TARGET = {
    ("c", "CHECK ((placement_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((command_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((command_generation > 0))", 0, True, False, False),
    ("c", "CHECK ((election_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((holder_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((fencing_token > 0))", 0, True, False, False),
    ("c", "CHECK ((target_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((target_revision > 0))", 0, True, False, False),
    ("c", "CHECK ((release_identity_digest <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((attempt >= 0))", 0, True, False, False),
    ("c", "CHECK ((compacted_at >= updated_at))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id, placement_id)", 0, True, False, False),
}
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS public.node_model_prestage_store_registry (
        store_id TEXT PRIMARY KEY CHECK (store_id <> ''),
        schema_version INTEGER NOT NULL,
        node_id TEXT NOT NULL CHECK (node_id <> ''),
        max_placements BIGINT NOT NULL
            CHECK (max_placements > 0 AND max_placements <= 100000),
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.node_model_prestage_records (
        store_id TEXT NOT NULL REFERENCES
            public.node_model_prestage_store_registry(store_id) ON DELETE CASCADE,
        placement_id TEXT NOT NULL CHECK (placement_id <> ''),
        command_id TEXT NOT NULL CHECK (command_id <> ''),
        command_generation BIGINT NOT NULL CHECK (command_generation > 0),
        fencing_token BIGINT NOT NULL CHECK (fencing_token > 0),
        target_revision BIGINT NOT NULL CHECK (target_revision > 0),
        state TEXT NOT NULL CHECK (state IN ('absent', 'filling', 'ready', 'failed')),
        updated_at TIMESTAMPTZ NOT NULL,
        record JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (store_id, placement_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.node_model_prestage_high_water_marks (
        store_id TEXT NOT NULL REFERENCES
            public.node_model_prestage_store_registry(store_id) ON DELETE CASCADE,
        placement_id TEXT NOT NULL CHECK (placement_id <> ''),
        command_id TEXT NOT NULL CHECK (command_id <> ''),
        command_generation BIGINT NOT NULL CHECK (command_generation > 0),
        election_id TEXT NOT NULL CHECK (election_id <> ''),
        holder_id TEXT NOT NULL CHECK (holder_id <> ''),
        fencing_token BIGINT NOT NULL CHECK (fencing_token > 0),
        target_id TEXT NOT NULL CHECK (target_id <> ''),
        target_revision BIGINT NOT NULL CHECK (target_revision > 0),
        release_identity_digest TEXT NOT NULL CHECK (release_identity_digest <> ''),
        attempt BIGINT NOT NULL CHECK (attempt >= 0),
        updated_at TIMESTAMPTZ NOT NULL,
        compacted_at TIMESTAMPTZ NOT NULL CHECK (compacted_at >= updated_at),
        mark JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (store_id, placement_id)
    )
    """,
)


class PostgresNodeModelPrestageStore:
    """Linearizable, durable placement store scoped to one node identity."""

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        node_id: str,
        max_placements: int = 100_000,
        connect_timeout_s: float = 10.0,
        eager_connect: bool = True,
        initialize_schema: bool = True,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "PostgresNodeModelPrestageStore requires psycopg; install the fleet dependency"
            )
        self._store_id = _text(store_id, name="store_id")
        self._node_id = _text(node_id, name="node_id", max_length=253)
        if not isinstance(dsn, str) or not dsn.strip() or "\x00" in dsn:
            raise ValueError("dsn must be a non-empty string without NUL")
        if type(max_placements) is not int or not 1 <= max_placements <= 100_000:
            raise ValueError("max_placements must be an integer in [1, 100000]")
        if isinstance(connect_timeout_s, bool) or not isinstance(connect_timeout_s, (int, float)):
            raise ValueError("connect_timeout_s must be a number")
        timeout = float(connect_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("connect_timeout_s must be finite and greater than zero")
        self._dsn = dsn
        self._max_placements = max_placements
        self._connect_timeout = max(1, math.ceil(timeout))
        self._initialize_schema_on_open = bool(initialize_schema)
        self._transitions = _NodeModelPrestageTransitions(node_id=self._node_id)
        self._connection = None
        self._namespace_oid: int | None = None
        self._closed = False
        self._lock = threading.RLock()
        if eager_connect:
            self._open()

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def node_id(self) -> str:
        return self._node_id

    def _connect(self):
        assert psycopg is not None
        timeout_ms = self._connect_timeout * 1000
        parameters = psycopg.conninfo.conninfo_to_dict(self._dsn)
        configured_options = parameters.pop("options", "")
        options = (
            f"{configured_options} -c statement_timeout={timeout_ms} -c lock_timeout={timeout_ms}"
        ).strip()
        return psycopg.connect(
            psycopg.conninfo.make_conninfo(**parameters),
            autocommit=True,
            connect_timeout=self._connect_timeout,
            options=options,
            keepalives=1,
            keepalives_idle=self._connect_timeout,
            keepalives_interval=self._connect_timeout,
            keepalives_count=1,
        )

    def _require_open(self, *, allow_unstarted: bool = False) -> None:
        if self._closed:
            raise RuntimeError("PostgresNodeModelPrestageStore is closed")
        if not allow_unstarted and self._connection is None:
            raise RuntimeError("PostgresNodeModelPrestageStore has not started")

    def _ensure_connection(self) -> None:
        assert self._connection is not None
        if self._connection.closed or self._connection.broken:
            self._connection.close()
            self._connection = self._connect()
            try:
                self._validate_schema(expected_namespace_oid=self._namespace_oid)
            except BaseException:
                self._connection.close()
                self._connection = None
                raise

    def _open(self) -> None:
        with self._lock:
            self._require_open(allow_unstarted=True)
            if self._connection is not None:
                return
            self._connection = self._connect()
            try:
                if self._initialize_schema_on_open:
                    namespace_oid = self._initialize_schema()
                else:
                    namespace_oid = self._validate_schema()
                self._namespace_oid = namespace_oid
            except BaseException:
                assert self._connection is not None
                self._connection.close()
                self._connection = None
                raise

    def startup(self) -> None:
        self._open()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._connection is not None:
                self._connection.close()
            self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def _initialize_schema(self) -> int:
        self._require_open()
        assert self._connection is not None
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute("SET LOCAL search_path = public")
                cursor.execute("SELECT pg_advisory_xact_lock(%s, %s)", _SCHEMA_LOCK)
                for statement in _SCHEMA_STATEMENTS:
                    cursor.execute(statement)
                namespace_oid = self._validate_schema_objects_cursor(cursor)
                cursor.execute(
                    """
                    UPDATE public.node_model_prestage_store_registry
                    SET schema_version = %s
                    WHERE store_id = %s AND schema_version = 1
                    """,
                    (_SCHEMA_VERSION, self._store_id),
                )
                cursor.execute(
                    """
                    INSERT INTO public.node_model_prestage_store_registry (
                        store_id, schema_version, node_id, max_placements
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (store_id) DO NOTHING
                    """,
                    (
                        self._store_id,
                        _SCHEMA_VERSION,
                        self._node_id,
                        self._max_placements,
                    ),
                )
                self._validate_registry_cursor(cursor)
                return namespace_oid

    def _validate_schema(self, *, expected_namespace_oid: int | None = None) -> int:
        self._require_open()
        assert self._connection is not None
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute("SET LOCAL search_path = public")
                namespace_oid = self._validate_schema_objects_cursor(cursor)
                if expected_namespace_oid is not None and namespace_oid != expected_namespace_oid:
                    raise RuntimeError("Node model pre-stage PostgreSQL namespace identity changed")
                self._validate_registry_cursor(cursor)
                return namespace_oid

    def _validate_registry_cursor(self, cursor, *, for_update: bool = False) -> None:
        lock = " FOR UPDATE" if for_update else ""
        cursor.execute(
            """
            SELECT schema_version, node_id, max_placements
            FROM public.node_model_prestage_store_registry
            WHERE store_id = %s
            """
            + lock,
            (self._store_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"unknown node model pre-stage store {self._store_id!r}")
        expected = (_SCHEMA_VERSION, self._node_id, self._max_placements)
        if row != expected:
            raise RuntimeError(
                "node model pre-stage store configuration changed: "
                f"observed {row!r}; expected {expected!r}"
            )

    @staticmethod
    def _validate_schema_objects_cursor(cursor) -> int:
        cursor.execute(
            """
            SELECT registry.oid, records.oid, high_water.oid,
                   registry.relnamespace, records.relnamespace, high_water.relnamespace,
                   registry.relkind, records.relkind, high_water.relkind,
                   registry.relpersistence, records.relpersistence,
                   high_water.relpersistence
            FROM pg_catalog.pg_class AS registry
            CROSS JOIN pg_catalog.pg_class AS records
            CROSS JOIN pg_catalog.pg_class AS high_water
            WHERE registry.oid = pg_catalog.to_regclass(
                      'public.node_model_prestage_store_registry'
                  )
              AND records.oid = pg_catalog.to_regclass(
                      'public.node_model_prestage_records'
                  )
              AND high_water.oid = pg_catalog.to_regclass(
                      'public.node_model_prestage_high_water_marks'
                  )
            """
        )
        objects = cursor.fetchone()
        if objects is None:
            raise RuntimeError("node model pre-stage schema is missing required tables")
        (
            registry_oid,
            records_oid,
            high_water_oid,
            registry_namespace,
            records_namespace,
            high_water_namespace,
            *kinds,
        ) = objects
        cursor.execute("SELECT pg_catalog.to_regnamespace(%s)::oid", (_SCHEMA_NAME,))
        namespace_row = cursor.fetchone()
        if (
            registry_namespace != records_namespace
            or registry_namespace != high_water_namespace
            or namespace_row is None
            or registry_namespace != namespace_row[0]
        ):
            raise RuntimeError("node model pre-stage tables must use the public namespace")
        if kinds != ["r", "r", "r", "p", "p", "p"]:
            raise RuntimeError(
                "node model pre-stage schema objects must be permanent ordinary tables"
            )
        for table_name, table_oid in (
            ("node_model_prestage_store_registry", registry_oid),
            ("node_model_prestage_records", records_oid),
            ("node_model_prestage_high_water_marks", high_water_oid),
        ):
            cursor.execute(
                """
                SELECT attribute.attname,
                       pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
                       attribute.attnotnull
                FROM pg_catalog.pg_attribute AS attribute
                WHERE attribute.attrelid = %s
                  AND attribute.attnum > 0
                  AND NOT attribute.attisdropped
                ORDER BY attribute.attnum
                """,
                (table_oid,),
            )
            if tuple(cursor.fetchall()) != _EXPECTED_COLUMNS[table_name]:
                raise RuntimeError(
                    f"node model pre-stage table {table_name!r} has incompatible columns"
                )

        def constraints(table_oid: int) -> set[tuple]:
            cursor.execute(
                """
                SELECT con.contype, pg_catalog.pg_get_constraintdef(con.oid),
                       con.confrelid, con.convalidated, con.condeferrable,
                       con.condeferred
                FROM pg_catalog.pg_constraint AS con
                WHERE con.conrelid = %s
                """,
                (table_oid,),
            )
            return set(cursor.fetchall())

        if constraints(registry_oid) != _EXPECTED_REGISTRY_CONSTRAINTS:
            raise RuntimeError("node model pre-stage registry has incompatible constraints")
        expected_record_constraints = _EXPECTED_RECORD_CONSTRAINTS_WITHOUT_TARGET | {
            (
                "f",
                "FOREIGN KEY (store_id) REFERENCES "
                "node_model_prestage_store_registry(store_id) ON DELETE CASCADE",
                registry_oid,
                True,
                False,
                False,
            )
        }
        if constraints(records_oid) != expected_record_constraints:
            raise RuntimeError("node model pre-stage records have incompatible constraints")
        expected_high_water_constraints = _EXPECTED_HIGH_WATER_CONSTRAINTS_WITHOUT_TARGET | {
            (
                "f",
                "FOREIGN KEY (store_id) REFERENCES "
                "node_model_prestage_store_registry(store_id) ON DELETE CASCADE",
                registry_oid,
                True,
                False,
                False,
            )
        }
        if constraints(high_water_oid) != expected_high_water_constraints:
            raise RuntimeError(
                "node model pre-stage high-water marks have incompatible constraints"
            )
        return registry_namespace

    @staticmethod
    def _select_columns() -> str:
        return """
            placement_id, command_id, command_generation, fencing_token,
            target_revision, state, updated_at, record
        """

    @staticmethod
    def _record(row: tuple, *, node_id: str) -> NodeModelPrestageRecord:
        (
            placement_id,
            command_id,
            command_generation,
            fencing_token,
            target_revision,
            state,
            updated_at,
            payload,
        ) = row
        record = NodeModelPrestageRecord.model_validate(payload)
        if record.command.node_id != node_id:
            raise RuntimeError("stored node model pre-stage record targets another node")
        expected = (
            record.command.placement_id,
            record.command.command_id,
            record.command.command_generation,
            record.command.authority.fencing_token,
            record.command.target_revision,
            record.state.value,
            record.updated_at,
        )
        if expected != (
            placement_id,
            command_id,
            command_generation,
            fencing_token,
            target_revision,
            state,
            updated_at,
        ):
            raise RuntimeError("stored node model pre-stage metadata is inconsistent")
        return record

    def _select_record_cursor(self, cursor, placement_id: str):
        cursor.execute(
            f"""
            SELECT {self._select_columns()}
            FROM public.node_model_prestage_records
            WHERE store_id = %s AND placement_id = %s
            """,
            (self._store_id, placement_id),
        )
        row = cursor.fetchone()
        return None if row is None else self._record(row, node_id=self._node_id)

    @staticmethod
    def _high_water_select_columns() -> str:
        return """
            placement_id, command_id, command_generation, election_id, holder_id,
            fencing_token, target_id, target_revision, release_identity_digest,
            attempt, updated_at, compacted_at, mark
        """

    @staticmethod
    def _high_water_mark(row: tuple) -> NodeModelPrestageHighWaterMark:
        *metadata, payload = row
        mark = NodeModelPrestageHighWaterMark.model_validate(payload)
        expected = (
            mark.placement_id,
            mark.command_id,
            mark.command_generation,
            mark.election_id,
            mark.holder_id,
            mark.fencing_token,
            mark.target_id,
            mark.target_revision,
            mark.release_identity_digest,
            mark.attempt,
            mark.updated_at,
            mark.compacted_at,
        )
        if expected != tuple(metadata):
            raise RuntimeError("stored node model pre-stage high-water metadata is inconsistent")
        return mark

    def _select_high_water_cursor(self, cursor, placement_id: str):
        cursor.execute(
            f"""
            SELECT {self._high_water_select_columns()}
            FROM public.node_model_prestage_high_water_marks
            WHERE store_id = %s AND placement_id = %s
            """,
            (self._store_id, placement_id),
        )
        row = cursor.fetchone()
        return None if row is None else self._high_water_mark(row)

    def _upsert_high_water_cursor(
        self,
        cursor,
        mark: NodeModelPrestageHighWaterMark,
    ) -> None:
        assert psycopg is not None
        cursor.execute(
            """
            INSERT INTO public.node_model_prestage_high_water_marks (
                store_id, placement_id, command_id, command_generation,
                election_id, holder_id, fencing_token, target_id, target_revision,
                release_identity_digest, attempt, updated_at, compacted_at, mark
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (store_id, placement_id) DO UPDATE SET
                command_id = EXCLUDED.command_id,
                command_generation = EXCLUDED.command_generation,
                election_id = EXCLUDED.election_id,
                holder_id = EXCLUDED.holder_id,
                fencing_token = EXCLUDED.fencing_token,
                target_id = EXCLUDED.target_id,
                target_revision = EXCLUDED.target_revision,
                release_identity_digest = EXCLUDED.release_identity_digest,
                attempt = EXCLUDED.attempt,
                updated_at = EXCLUDED.updated_at,
                compacted_at = EXCLUDED.compacted_at,
                mark = EXCLUDED.mark
            """,
            (
                self._store_id,
                mark.placement_id,
                mark.command_id,
                mark.command_generation,
                mark.election_id,
                mark.holder_id,
                mark.fencing_token,
                mark.target_id,
                mark.target_revision,
                mark.release_identity_digest,
                mark.attempt,
                mark.updated_at,
                mark.compacted_at,
                psycopg.types.json.Jsonb(mark.model_dump(mode="json")),
            ),
        )

    def _mutate(
        self,
        command: NodeModelPrestageCommand,
        transition: Callable[
            [NodeModelPrestageRecord | None, NodeModelPrestageHighWaterMark | None, int],
            NodeModelPrestageRecord,
        ],
    ) -> NodeModelPrestageRecord:
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor, for_update=True)
                    existing = self._select_record_cursor(cursor, command.placement_id)
                    high_water = self._select_high_water_cursor(cursor, command.placement_id)
                    cursor.execute(
                        """
                        SELECT count(*)
                        FROM public.node_model_prestage_records
                        WHERE store_id = %s
                        """,
                        (self._store_id,),
                    )
                    count_row = cursor.fetchone()
                    assert count_row is not None
                    record = transition(existing, high_water, count_row[0])
                    assert psycopg is not None
                    cursor.execute(
                        """
                        INSERT INTO public.node_model_prestage_records (
                            store_id, placement_id, command_id, command_generation,
                            fencing_token, target_revision, state, updated_at, record
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (store_id, placement_id) DO UPDATE SET
                            command_id = EXCLUDED.command_id,
                            command_generation = EXCLUDED.command_generation,
                            fencing_token = EXCLUDED.fencing_token,
                            target_revision = EXCLUDED.target_revision,
                            state = EXCLUDED.state,
                            updated_at = EXCLUDED.updated_at,
                            record = EXCLUDED.record
                        """,
                        (
                            self._store_id,
                            record.command.placement_id,
                            record.command.command_id,
                            record.command.command_generation,
                            record.command.authority.fencing_token,
                            record.command.target_revision,
                            record.state.value,
                            record.updated_at,
                            psycopg.types.json.Jsonb(record.model_dump(mode="json")),
                        ),
                    )
                    return _copy_prestage_record(record)

    def claim(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(command, now=now, require_active=False)
        return self._mutate(
            command,
            lambda existing, high_water, count: self._transitions.claim(
                existing,
                high_water=high_water,
                record_count=count,
                max_placements=self._max_placements,
                command=command,
                claim_id=claim_id,
                now=now,
            ),
        )

    def complete(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        fill_result: NodeModelCacheFillResult,
        pin_record_generation: int | None = None,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(command, now=now, require_active=False)
        return self._mutate(
            command,
            lambda existing, _high_water, _count: self._transitions.complete(
                existing,
                command=command,
                claim_id=claim_id,
                fill_result=fill_result,
                pin_record_generation=pin_record_generation,
                now=now,
            ),
        )

    def fail(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        failure: str,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(command, now=now, require_active=False)
        return self._mutate(
            command,
            lambda existing, _high_water, _count: self._transitions.fail(
                existing,
                command=command,
                claim_id=claim_id,
                failure=failure,
                now=now,
            ),
        )

    def release(
        self,
        command: NodeModelPrestageCommand,
        *,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(command, now=now, require_active=True)
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor, for_update=True)
                    existing = self._select_record_cursor(cursor, command.placement_id)
                    high_water = self._select_high_water_cursor(cursor, command.placement_id)
                    record = self._transitions.release(
                        existing,
                        high_water=high_water,
                        command=command,
                        now=now,
                    )
                    if existing is not None:
                        assert psycopg is not None
                        cursor.execute(
                            """
                            UPDATE public.node_model_prestage_records
                            SET command_id = %s, command_generation = %s,
                                fencing_token = %s, target_revision = %s,
                                state = %s, updated_at = %s, record = %s
                            WHERE store_id = %s AND placement_id = %s
                            """,
                            (
                                record.command.command_id,
                                record.command.command_generation,
                                record.command.authority.fencing_token,
                                record.command.target_revision,
                                record.state.value,
                                record.updated_at,
                                psycopg.types.json.Jsonb(record.model_dump(mode="json")),
                                self._store_id,
                                record.command.placement_id,
                            ),
                        )
                    else:
                        assert high_water is not None
                        mark = NodeModelPrestageHighWaterMark.from_absent_record(
                            record,
                            compacted_at=max(now, high_water.compacted_at),
                        )
                        self._upsert_high_water_cursor(cursor, mark)
                    return _copy_prestage_record(record)

    def list_records(self) -> tuple[NodeModelPrestageRecord, ...]:
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)
                    cursor.execute(
                        f"""
                        SELECT {self._select_columns()}
                        FROM public.node_model_prestage_records
                        WHERE store_id = %s
                        ORDER BY placement_id
                        """,
                        (self._store_id,),
                    )
                    return tuple(
                        self._record(row, node_id=self._node_id) for row in cursor.fetchall()
                    )

    def get_record(self, placement_id: str) -> NodeModelPrestageRecord | None:
        placement_id = _text(placement_id, name="placement_id")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)
                    return self._select_record_cursor(cursor, placement_id)

    def list_records_page(
        self,
        *,
        after_placement_id: str | None = None,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageRecord, ...]:
        if after_placement_id is not None:
            after_placement_id = _text(after_placement_id, name="after_placement_id")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)
                    if after_placement_id is None:
                        cursor.execute(
                            f"""
                            SELECT {self._select_columns()}
                            FROM public.node_model_prestage_records
                            WHERE store_id = %s
                            ORDER BY placement_id
                            LIMIT %s
                            """,
                            (self._store_id, limit),
                        )
                    else:
                        cursor.execute(
                            f"""
                            SELECT {self._select_columns()}
                            FROM public.node_model_prestage_records
                            WHERE store_id = %s AND placement_id > %s
                            ORDER BY placement_id
                            LIMIT %s
                            """,
                            (self._store_id, after_placement_id, limit),
                        )
                    return tuple(
                        self._record(row, node_id=self._node_id) for row in cursor.fetchall()
                    )

    def compact_absent_records(
        self,
        *,
        retired_before: datetime,
        compacted_at: datetime,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        retired_before = _aware(retired_before, name="retired_before")
        compacted_at = _aware(compacted_at, name="compacted_at")
        if compacted_at < retired_before:
            raise ValueError("compacted_at cannot predate retired_before")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor, for_update=True)
                    cursor.execute(
                        f"""
                        SELECT {self._select_columns()}
                        FROM public.node_model_prestage_records
                        WHERE store_id = %s AND state = 'absent' AND updated_at <= %s
                        ORDER BY updated_at, placement_id
                        LIMIT %s
                        """,
                        (self._store_id, retired_before, limit),
                    )
                    records = tuple(
                        self._record(row, node_id=self._node_id) for row in cursor.fetchall()
                    )
                    marks: list[NodeModelPrestageHighWaterMark] = []
                    for record in records:
                        mark = NodeModelPrestageHighWaterMark.from_absent_record(
                            record,
                            compacted_at=compacted_at,
                        )
                        previous = self._select_high_water_cursor(cursor, mark.placement_id)
                        if (
                            previous is not None
                            and previous.command_generation >= mark.command_generation
                        ):
                            raise RuntimeError("pre-stage high-water mark would not advance")
                        self._upsert_high_water_cursor(cursor, mark)
                        cursor.execute(
                            """
                            DELETE FROM public.node_model_prestage_records
                            WHERE store_id = %s AND placement_id = %s
                              AND command_id = %s AND state = 'absent'
                            """,
                            (self._store_id, mark.placement_id, mark.command_id),
                        )
                        if cursor.rowcount != 1:
                            raise RuntimeError("pre-stage compaction lost its locked record")
                        marks.append(_copy_high_water_mark(mark))
                    return tuple(marks)

    def list_high_water_marks(self) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)
                    cursor.execute(
                        f"""
                        SELECT {self._high_water_select_columns()}
                        FROM public.node_model_prestage_high_water_marks
                        WHERE store_id = %s
                        ORDER BY placement_id
                        """,
                        (self._store_id,),
                    )
                    return tuple(self._high_water_mark(row) for row in cursor.fetchall())

    def list_high_water_marks_page(
        self,
        *,
        after_placement_id: str | None = None,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        if after_placement_id is not None:
            after_placement_id = _text(after_placement_id, name="after_placement_id")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)
                    if after_placement_id is None:
                        cursor.execute(
                            f"""
                            SELECT {self._high_water_select_columns()}
                            FROM public.node_model_prestage_high_water_marks
                            WHERE store_id = %s
                            ORDER BY placement_id
                            LIMIT %s
                            """,
                            (self._store_id, limit),
                        )
                    else:
                        cursor.execute(
                            f"""
                            SELECT {self._high_water_select_columns()}
                            FROM public.node_model_prestage_high_water_marks
                            WHERE store_id = %s AND placement_id > %s
                            ORDER BY placement_id
                            LIMIT %s
                            """,
                            (self._store_id, after_placement_id, limit),
                        )
                    return tuple(self._high_water_mark(row) for row in cursor.fetchall())

    def check_ready(self) -> None:
        """Validate the live connection and this store's durable registry binding."""

        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)

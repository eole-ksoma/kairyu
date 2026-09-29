"""PostgreSQL-backed append-only autoscaler decision log."""

from __future__ import annotations

import math
import threading
from datetime import datetime
from types import TracebackType
from typing import Literal, Self

from kairyu.runners.scaling import _MAX_SIGNED_BIGINT
from kairyu.runners.scaling_log import (
    ScalingDecisionCapacityError,
    ScalingDecisionConflictError,
    ScalingDecisionRecord,
    _aware,
    _non_empty,
)

try:  # Optional deployment dependency.
    import psycopg  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - core-only installation.
    psycopg = None  # type: ignore[assignment]

_SCHEMA_VERSION = 1
_SCHEMA_NAME = "public"
_SCHEMA_LOCK = (1_261_587_810, 8)
_EXPECTED_COLUMNS = {
    "runner_scaling_log_registry": (
        ("store_id", "text", True),
        ("schema_version", "integer", True),
        ("max_records", "bigint", True),
        ("created_at", "timestamp with time zone", True),
    ),
    "runner_scaling_decisions": (
        ("store_id", "text", True),
        ("decision_id", "text", True),
        ("model_class", "text", True),
        ("decided_at", "timestamp with time zone", True),
        ("window_started_at", "timestamp with time zone", True),
        ("window_ended_at", "timestamp with time zone", True),
        ("catalog_revision", "bigint", True),
        ("policy_revision", "bigint", True),
        ("action", "text", True),
        ("record_fingerprint", "text", True),
        ("record", "jsonb", True),
        ("created_at", "timestamp with time zone", True),
    ),
}
_EXPECTED_REGISTRY_CONSTRAINTS = {
    ("c", "CHECK ((store_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((max_records > 0))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id)", 0, True, False, False),
}
_EXPECTED_DECISION_CONSTRAINTS_WITHOUT_TARGET = {
    ("c", "CHECK ((decision_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((model_class <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((catalog_revision > 0))", 0, True, False, False),
    ("c", "CHECK ((policy_revision > 0))", 0, True, False, False),
    (
        "c",
        "CHECK ((window_ended_at >= window_started_at))",
        0,
        True,
        False,
        False,
    ),
    ("p", "PRIMARY KEY (store_id, decision_id)", 0, True, False, False),
}
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS public.runner_scaling_log_registry (
        store_id TEXT PRIMARY KEY CHECK (store_id <> ''),
        schema_version INTEGER NOT NULL,
        max_records BIGINT NOT NULL CHECK (max_records > 0),
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.runner_scaling_decisions (
        store_id TEXT NOT NULL
            REFERENCES public.runner_scaling_log_registry(store_id) ON DELETE CASCADE,
        decision_id TEXT NOT NULL CHECK (decision_id <> ''),
        model_class TEXT NOT NULL CHECK (model_class <> ''),
        decided_at TIMESTAMPTZ NOT NULL,
        window_started_at TIMESTAMPTZ NOT NULL,
        window_ended_at TIMESTAMPTZ NOT NULL,
        catalog_revision BIGINT NOT NULL CHECK (catalog_revision > 0),
        policy_revision BIGINT NOT NULL CHECK (policy_revision > 0),
        action TEXT NOT NULL,
        record_fingerprint TEXT NOT NULL,
        record JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (store_id, decision_id),
        CHECK (window_ended_at >= window_started_at)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS runner_scaling_decisions_model_time_idx
    ON public.runner_scaling_decisions (
        store_id, model_class, decided_at DESC, decision_id DESC
    )
    """,
)


class PostgresScalingDecisionLog:
    """Shared durable decision log with exact replay and bounded capacity."""

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        max_records: int = 1_000_000,
        connect_timeout_s: float = 10.0,
        eager_connect: bool = True,
        initialize_schema: bool = True,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "PostgresScalingDecisionLog requires psycopg; install the fleet dependency"
            )
        self._store_id = _non_empty(store_id, name="store_id")
        if not isinstance(dsn, str) or not dsn.strip() or "\x00" in dsn:
            raise ValueError("dsn must be a non-empty string without NUL")
        if (
            type(max_records) is not int
            or max_records <= 0
            or max_records > _MAX_SIGNED_BIGINT
        ):
            raise ValueError("max_records must be a positive signed 64-bit integer")
        if isinstance(connect_timeout_s, bool) or not isinstance(
            connect_timeout_s, (int, float)
        ):
            raise ValueError("connect_timeout_s must be a number")
        timeout = float(connect_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("connect_timeout_s must be finite and greater than zero")
        self._dsn = dsn
        self._max_records = max_records
        self._connect_timeout = max(1, math.ceil(timeout))
        self._initialize_schema_on_open = bool(initialize_schema)
        self._connection = None
        self._namespace_oid: int | None = None
        self._closed = False
        self._lock = threading.RLock()
        if eager_connect:
            self._open()

    @property
    def store_id(self) -> str:
        return self._store_id

    def _connect(self):
        assert psycopg is not None
        timeout_ms = self._connect_timeout * 1000
        parameters = psycopg.conninfo.conninfo_to_dict(self._dsn)
        configured_options = parameters.pop("options", "")
        options = (
            f"{configured_options} -c statement_timeout={timeout_ms} "
            f"-c lock_timeout={timeout_ms}"
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
            raise RuntimeError("PostgresScalingDecisionLog is closed")
        if not allow_unstarted and self._connection is None:
            raise RuntimeError("PostgresScalingDecisionLog has not started")

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
                    INSERT INTO public.runner_scaling_log_registry (
                        store_id, schema_version, max_records
                    ) VALUES (%s, %s, %s)
                    ON CONFLICT (store_id) DO NOTHING
                    """,
                    (self._store_id, _SCHEMA_VERSION, self._max_records),
                )
                self._validate_schema_cursor(
                    cursor,
                    expected_namespace_oid=namespace_oid,
                )
                return namespace_oid

    def _validate_schema(self, *, expected_namespace_oid: int | None = None) -> int:
        self._require_open()
        assert self._connection is not None
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute("SET LOCAL search_path = public")
                return self._validate_schema_cursor(
                    cursor,
                    expected_namespace_oid=expected_namespace_oid,
                )

    def _validate_schema_cursor(
        self,
        cursor,
        *,
        expected_namespace_oid: int | None = None,
    ) -> int:
        namespace_oid = self._validate_schema_objects_cursor(cursor)
        if expected_namespace_oid is not None and namespace_oid != expected_namespace_oid:
            raise RuntimeError("Runner scaling log PostgreSQL namespace identity changed")
        cursor.execute(
            """
            SELECT schema_version, max_records
            FROM public.runner_scaling_log_registry
            WHERE store_id = %s
            """,
            (self._store_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"unknown Runner scaling log store {self._store_id!r}")
        if row[0] != _SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported Runner scaling log schema version {row[0]!r}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if row[1] != self._max_records:
            raise RuntimeError(
                f"Runner scaling log capacity is {row[1]!r}; "
                f"configured {self._max_records!r}"
            )
        return namespace_oid

    @staticmethod
    def _validate_schema_objects_cursor(cursor) -> int:
        cursor.execute(
            """
            SELECT registry.oid, decisions.oid,
                   registry.relnamespace, decisions.relnamespace,
                   registry.relkind, decisions.relkind,
                   registry.relpersistence, decisions.relpersistence
            FROM pg_catalog.pg_class AS registry
            CROSS JOIN pg_catalog.pg_class AS decisions
            WHERE registry.oid = pg_catalog.to_regclass(
                      'public.runner_scaling_log_registry'
                  )
              AND decisions.oid = pg_catalog.to_regclass(
                      'public.runner_scaling_decisions'
                  )
            """
        )
        objects = cursor.fetchone()
        if objects is None:
            raise RuntimeError("Runner scaling log schema is missing required tables")
        registry_oid, decisions_oid, registry_namespace, decisions_namespace, *kinds = (
            objects
        )
        cursor.execute(
            "SELECT pg_catalog.to_regnamespace(%s)::oid",
            (_SCHEMA_NAME,),
        )
        namespace_row = cursor.fetchone()
        if (
            registry_namespace != decisions_namespace
            or namespace_row is None
            or registry_namespace != namespace_row[0]
        ):
            raise RuntimeError("Runner scaling log tables must use the public namespace")
        if kinds != ["r", "r", "p", "p"]:
            raise RuntimeError(
                "Runner scaling log schema objects must be permanent ordinary tables"
            )

        for table_name, table_oid in (
            ("runner_scaling_log_registry", registry_oid),
            ("runner_scaling_decisions", decisions_oid),
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
                    f"Runner scaling log table {table_name!r} has incompatible columns"
                )

        def constraints(table_oid: int) -> set[tuple[str, str, int, bool, bool, bool]]:
            cursor.execute(
                """
                SELECT con.contype,
                       pg_catalog.pg_get_constraintdef(con.oid),
                       con.confrelid,
                       con.convalidated,
                       con.condeferrable,
                       con.condeferred
                FROM pg_catalog.pg_constraint AS con
                WHERE con.conrelid = %s
                """,
                (table_oid,),
            )
            return set(cursor.fetchall())

        if constraints(registry_oid) != _EXPECTED_REGISTRY_CONSTRAINTS:
            raise RuntimeError("Runner scaling log registry has incompatible constraints")
        expected_decision_constraints = _EXPECTED_DECISION_CONSTRAINTS_WITHOUT_TARGET | {
            (
                "f",
                "FOREIGN KEY (store_id) REFERENCES "
                "runner_scaling_log_registry(store_id) ON DELETE CASCADE",
                registry_oid,
                True,
                False,
                False,
            )
        }
        if constraints(decisions_oid) != expected_decision_constraints:
            raise RuntimeError("Runner scaling decision table has incompatible constraints")

        cursor.execute(
            """
            SELECT index.indisunique, index.indisvalid, index.indisready,
                   index.indnatts = index.indnkeyatts,
                   ARRAY(
                       SELECT pg_catalog.pg_get_indexdef(
                           index.indexrelid, key_position, true
                       )
                       FROM generate_series(
                           1, index.indnkeyatts
                       ) AS key_position
                       ORDER BY key_position
                   ),
                   index.indpred IS NULL,
                   index.indoption::smallint[]
            FROM pg_catalog.pg_index AS index
            JOIN pg_catalog.pg_class AS index_class
              ON index_class.oid = index.indexrelid
            WHERE index.indrelid = %s
              AND index_class.relname = 'runner_scaling_decisions_model_time_idx'
              AND index_class.relnamespace = %s
            """,
            (decisions_oid, registry_namespace),
        )
        index_row = cursor.fetchone()
        expected_index = (
            False,
            True,
            True,
            True,
            ["store_id", "model_class", "decided_at", "decision_id"],
            True,
            [0, 0, 3, 3],
        )
        if index_row != expected_index:
            raise RuntimeError(
                "Runner scaling decision index is incompatible: "
                f"observed {index_row!r}"
            )
        return registry_namespace

    @staticmethod
    def _validated(record: ScalingDecisionRecord) -> ScalingDecisionRecord:
        if not isinstance(record, ScalingDecisionRecord):
            raise TypeError("record must be a ScalingDecisionRecord")
        return ScalingDecisionRecord.model_validate(record.model_dump())

    @staticmethod
    def _record(row: tuple) -> ScalingDecisionRecord:
        (
            decision_id,
            model_class,
            decided_at,
            window_started_at,
            window_ended_at,
            catalog_revision,
            policy_revision,
            action,
            fingerprint,
            payload,
        ) = row
        record = ScalingDecisionRecord.model_validate(payload)
        expected = (
            record.decision_id,
            record.policy.model_class,
            record.decided_at,
            record.window.started_at,
            record.window.ended_at,
            record.catalog_revision,
            record.policy.policy_revision,
            record.action.value,
            record.fingerprint,
        )
        if expected != (
            decision_id,
            model_class,
            decided_at,
            window_started_at,
            window_ended_at,
            catalog_revision,
            policy_revision,
            action,
            fingerprint,
        ):
            raise RuntimeError("stored Runner scaling decision metadata is inconsistent")
        return record

    @staticmethod
    def _select_columns() -> str:
        return """
            decision_id, model_class, decided_at, window_started_at,
            window_ended_at, catalog_revision, policy_revision, action,
            record_fingerprint, record
        """

    def append(self, record: ScalingDecisionRecord) -> ScalingDecisionRecord:
        record = self._validated(record)
        fingerprint = record.fingerprint
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT schema_version, max_records
                        FROM public.runner_scaling_log_registry
                        WHERE store_id = %s
                        FOR UPDATE
                        """,
                        (self._store_id,),
                    )
                    if cursor.fetchone() != (_SCHEMA_VERSION, self._max_records):
                        raise RuntimeError("Runner scaling log registry changed")
                    cursor.execute(
                        f"""
                        SELECT {self._select_columns()}
                        FROM public.runner_scaling_decisions
                        WHERE store_id = %s AND decision_id = %s
                        """,
                        (self._store_id, record.decision_id),
                    )
                    existing = cursor.fetchone()
                    if existing is not None:
                        stored = self._record(existing)
                        if stored.fingerprint != fingerprint:
                            raise ScalingDecisionConflictError(
                                "decision ID was already used with different content"
                            )
                        return stored
                    cursor.execute(
                        """
                        SELECT count(*)
                        FROM public.runner_scaling_decisions
                        WHERE store_id = %s
                        """,
                        (self._store_id,),
                    )
                    count_row = cursor.fetchone()
                    assert count_row is not None
                    if count_row[0] >= self._max_records:
                        raise ScalingDecisionCapacityError(
                            "decision log capacity is exhausted"
                        )
                    assert psycopg is not None
                    cursor.execute(
                        """
                        INSERT INTO public.runner_scaling_decisions (
                            store_id, decision_id, model_class, decided_at,
                            window_started_at, window_ended_at, catalog_revision,
                            policy_revision, action, record_fingerprint, record
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            self._store_id,
                            record.decision_id,
                            record.policy.model_class,
                            record.decided_at,
                            record.window.started_at,
                            record.window.ended_at,
                            record.catalog_revision,
                            record.policy.policy_revision,
                            record.action.value,
                            fingerprint,
                            psycopg.types.json.Jsonb(record.model_dump(mode="json")),
                        ),
                    )
                    return record

    def get(self, decision_id: str) -> ScalingDecisionRecord:
        decision_id = _non_empty(decision_id, name="decision_id")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {self._select_columns()}
                    FROM public.runner_scaling_decisions
                    WHERE store_id = %s AND decision_id = %s
                    """,
                    (self._store_id, decision_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(f"unknown scaling decision {decision_id!r}")
                return self._record(row)

    def list(
        self,
        *,
        model_class: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> tuple[ScalingDecisionRecord, ...]:
        if model_class is not None:
            model_class = _non_empty(model_class, name="model_class")
        if since is not None:
            since = _aware(since, name="since")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer from 1 to 1000")
        clauses = ["store_id = %s"]
        parameters: list[object] = [self._store_id]
        if model_class is not None:
            clauses.append("model_class = %s")
            parameters.append(model_class)
        if since is not None:
            clauses.append("decided_at >= %s")
            parameters.append(since)
        parameters.append(limit)
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {self._select_columns()}
                    FROM public.runner_scaling_decisions
                    WHERE {' AND '.join(clauses)}
                    ORDER BY decided_at DESC, decision_id DESC
                    LIMIT %s
                    """,
                    tuple(parameters),
                )
                return tuple(self._record(row) for row in cursor.fetchall())

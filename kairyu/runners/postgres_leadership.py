"""PostgreSQL-backed Runner leader election with DB-clock fencing."""

from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta
from types import TracebackType
from typing import Literal, Self

from kairyu.runners.leadership import (
    RunnerLeaderCapacityError,
    RunnerLeaderLease,
    RunnerWriterAuthority,
    StaleRunnerLeaderLeaseError,
    _identity,
    _lease_duration,
    _validated_lease,
)

try:  # Optional deployment dependency; construction reports a useful error.
    import psycopg  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - core-only installation.
    psycopg = None  # type: ignore[assignment]

_SCHEMA_VERSION = 1
_SCHEMA_LOCK = (1_261_587_810, 6)
_SCHEMA_NAME = "public"
_MAX_FENCING_TOKEN = 2**63 - 1
_EXPECTED_COLUMNS = {
    "runner_leader_store_registry": (
        ("store_id", "text", True),
        ("schema_version", "integer", True),
        ("created_at", "timestamp with time zone", True),
    ),
    "runner_leader_leases": (
        ("store_id", "text", True),
        ("election_id", "text", True),
        ("holder_id", "text", True),
        ("fencing_token", "bigint", True),
        ("acquired_at", "timestamp with time zone", True),
        ("renewed_at", "timestamp with time zone", True),
        ("lease_until", "timestamp with time zone", True),
    ),
}
_EXPECTED_REGISTRY_CONSTRAINTS = {
    ("c", "CHECK ((store_id <> ''::text))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id)", 0, True, False, False),
}
_EXPECTED_LEASE_CONSTRAINTS_WITHOUT_TARGET = {
    ("c", "CHECK ((election_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((holder_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((fencing_token > 0))", 0, True, False, False),
    ("c", "CHECK ((renewed_at >= acquired_at))", 0, True, False, False),
    ("c", "CHECK ((lease_until >= renewed_at))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id, election_id)", 0, True, False, False),
}
_LEASE_COLUMNS = """
    election_id, holder_id, fencing_token, acquired_at, renewed_at, lease_until
"""
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS public.runner_leader_store_registry (
        store_id TEXT PRIMARY KEY CHECK (store_id <> ''),
        schema_version INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.runner_leader_leases (
        store_id TEXT NOT NULL
            REFERENCES public.runner_leader_store_registry(store_id) ON DELETE CASCADE,
        election_id TEXT NOT NULL CHECK (election_id <> ''),
        holder_id TEXT NOT NULL CHECK (holder_id <> ''),
        fencing_token BIGINT NOT NULL CHECK (fencing_token > 0),
        acquired_at TIMESTAMPTZ NOT NULL,
        renewed_at TIMESTAMPTZ NOT NULL,
        lease_until TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (store_id, election_id),
        CHECK (renewed_at >= acquired_at),
        CHECK (lease_until >= renewed_at)
    )
    """,
)


class PostgresRunnerLeaderLeaseStore:
    """Shared, durable lease store using PostgreSQL locks and clock."""

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        connect_timeout_s: float = 10.0,
        eager_connect: bool = True,
        initialize_schema: bool = True,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "PostgresRunnerLeaderLeaseStore requires psycopg; install the fleet dependency"
            )
        self._store_id = _identity(store_id, name="store_id")
        if not isinstance(dsn, str) or not dsn.strip() or "\x00" in dsn:
            raise ValueError("dsn must be a non-empty string without NUL")
        if isinstance(connect_timeout_s, bool) or not isinstance(connect_timeout_s, (int, float)):
            raise ValueError("connect_timeout_s must be a number")
        timeout = float(connect_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("connect_timeout_s must be finite and greater than zero")
        self._dsn = dsn
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
        connection_parameters = psycopg.conninfo.conninfo_to_dict(self._dsn)
        configured_options = connection_parameters.pop("options", "")
        runtime_options = (
            f"{configured_options} -c statement_timeout={timeout_ms} "
            f"-c lock_timeout={timeout_ms}"
        ).strip()
        return psycopg.connect(
            psycopg.conninfo.make_conninfo(**connection_parameters),
            autocommit=True,
            connect_timeout=self._connect_timeout,
            options=runtime_options,
            keepalives=1,
            keepalives_idle=self._connect_timeout,
            keepalives_interval=self._connect_timeout,
            keepalives_count=1,
        )

    def _require_open(self, *, allow_unstarted: bool = False) -> None:
        if self._closed:
            raise RuntimeError("PostgresRunnerLeaderLeaseStore is closed")
        if not allow_unstarted and self._connection is None:
            raise RuntimeError("PostgresRunnerLeaderLeaseStore has not started")

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
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(%s, %s)",
                    _SCHEMA_LOCK,
                )
                for statement in _SCHEMA_STATEMENTS:
                    cursor.execute(statement)
                namespace_oid = self._validate_schema_objects_cursor(cursor)
                cursor.execute(
                    """
                    INSERT INTO public.runner_leader_store_registry (
                        store_id, schema_version
                    ) VALUES (%s, %s)
                    ON CONFLICT (store_id) DO NOTHING
                    """,
                    (self._store_id, _SCHEMA_VERSION),
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
            raise RuntimeError("Runner leader PostgreSQL namespace identity changed")
        cursor.execute(
            """
            SELECT schema_version
            FROM public.runner_leader_store_registry
            WHERE store_id = %s
            """,
            (self._store_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"unknown Runner leader store {self._store_id!r}")
        if row[0] != _SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported Runner leader schema version {row[0]!r}; expected {_SCHEMA_VERSION}"
            )
        return namespace_oid

    @staticmethod
    def _validate_schema_objects_cursor(cursor) -> int:
        cursor.execute(
            """
            SELECT registry.oid, leases.oid,
                   registry.relnamespace, leases.relnamespace,
                   registry.relkind, leases.relkind,
                   registry.relpersistence, leases.relpersistence
            FROM pg_catalog.pg_class AS registry
            CROSS JOIN pg_catalog.pg_class AS leases
            WHERE registry.oid = pg_catalog.to_regclass(
                      'public.runner_leader_store_registry'
                  )
              AND leases.oid = pg_catalog.to_regclass(
                      'public.runner_leader_leases'
                  )
            """
        )
        objects = cursor.fetchone()
        if objects is None:
            raise RuntimeError("Runner leader schema is missing required tables")
        registry_oid, leases_oid, registry_namespace, leases_namespace, *kinds = objects
        if registry_namespace != leases_namespace:
            raise RuntimeError("Runner leader tables must use the same PostgreSQL namespace")
        cursor.execute(
            "SELECT pg_catalog.to_regnamespace(%s)::oid",
            (_SCHEMA_NAME,),
        )
        namespace_row = cursor.fetchone()
        if namespace_row is None or registry_namespace != namespace_row[0]:
            raise RuntimeError("Runner leader tables are outside the required namespace")
        if kinds != ["r", "r", "p", "p"]:
            raise RuntimeError(
                "Runner leader schema objects must be permanent ordinary tables"
            )

        for table_name, table_oid in (
            ("runner_leader_store_registry", registry_oid),
            ("runner_leader_leases", leases_oid),
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
            columns = tuple(cursor.fetchall())
            if columns != _EXPECTED_COLUMNS[table_name]:
                raise RuntimeError(
                    f"Runner leader table {table_name!r} has incompatible columns"
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
            raise RuntimeError("Runner leader registry has incompatible constraints")
        expected_lease_constraints = _EXPECTED_LEASE_CONSTRAINTS_WITHOUT_TARGET | {
            (
                "f",
                "FOREIGN KEY (store_id) REFERENCES "
                "runner_leader_store_registry(store_id) ON DELETE CASCADE",
                registry_oid,
                True,
                False,
                False,
            )
        }
        if constraints(leases_oid) != expected_lease_constraints:
            raise RuntimeError("Runner leader lease table has incompatible constraints")
        return registry_namespace

    def _lock_election(self, cursor, election_id: str) -> None:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{self._store_id}:{election_id}",),
        )

    def _read_locked(self, cursor, election_id: str) -> tuple[datetime, tuple | None]:
        self._lock_election(cursor, election_id)
        cursor.execute(
            f"""
            SELECT {_LEASE_COLUMNS}
            FROM public.runner_leader_leases
            WHERE store_id = %s AND election_id = %s
            FOR UPDATE
            """,
            (self._store_id, election_id),
        )
        row = cursor.fetchone()
        cursor.execute("SELECT clock_timestamp()")
        now_row = cursor.fetchone()
        assert now_row is not None
        return now_row[0], row

    @staticmethod
    def _lease(row: tuple) -> RunnerLeaderLease:
        return RunnerLeaderLease(
            election_id=row[0],
            holder_id=row[1],
            fencing_token=row[2],
            acquired_at=row[3],
            renewed_at=row[4],
            lease_until=row[5],
        )

    @staticmethod
    def _current(
        row: tuple | None,
        *,
        now: datetime,
        expected: RunnerLeaderLease,
    ) -> RunnerLeaderLease:
        if row is None or row[5] <= now:
            raise StaleRunnerLeaderLeaseError("leader lease is not current")
        current = PostgresRunnerLeaderLeaseStore._lease(row)
        if current.tenure != expected.tenure:
            raise StaleRunnerLeaderLeaseError("leader lease is not current")
        return current

    def acquire(
        self,
        election_id: str,
        holder_id: str,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease | None:
        election_id = _identity(election_id, name="election_id")
        holder_id = _identity(holder_id, name="holder_id")
        duration = _lease_duration(lease_seconds)
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    now, row = self._read_locked(cursor, election_id)
                    if row is not None:
                        if row[5] > now:
                            current = self._lease(row)
                            return current if current.holder_id == holder_id else None
                        previous_token = row[2]
                        if previous_token >= _MAX_FENCING_TOKEN:
                            raise RunnerLeaderCapacityError("leader fencing token is exhausted")
                        token = previous_token + 1
                        cursor.execute(
                            """
                            UPDATE public.runner_leader_leases
                            SET holder_id = %s, fencing_token = %s,
                                acquired_at = %s, renewed_at = %s,
                                lease_until = %s
                            WHERE store_id = %s AND election_id = %s
                            """,
                            (
                                holder_id,
                                token,
                                now,
                                now,
                                now + timedelta(seconds=duration),
                                self._store_id,
                                election_id,
                            ),
                        )
                    else:
                        token = 1
                        cursor.execute(
                            """
                            INSERT INTO public.runner_leader_leases (
                                store_id, election_id, holder_id, fencing_token,
                                acquired_at, renewed_at, lease_until
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self._store_id,
                                election_id,
                                holder_id,
                                token,
                                now,
                                now,
                                now + timedelta(seconds=duration),
                            ),
                        )
                    return RunnerLeaderLease(
                        election_id=election_id,
                        holder_id=holder_id,
                        fencing_token=token,
                        acquired_at=now,
                        renewed_at=now,
                        lease_until=now + timedelta(seconds=duration),
                    )

    def renew(
        self,
        lease: RunnerLeaderLease,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease:
        lease = _validated_lease(lease)
        duration = _lease_duration(lease_seconds)
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    now, row = self._read_locked(cursor, lease.election_id)
                    current = self._current(row, now=now, expected=lease)
                    renewed = RunnerLeaderLease(
                        election_id=current.election_id,
                        holder_id=current.holder_id,
                        fencing_token=current.fencing_token,
                        acquired_at=current.acquired_at,
                        renewed_at=now,
                        lease_until=now + timedelta(seconds=duration),
                    )
                    cursor.execute(
                        """
                        UPDATE public.runner_leader_leases
                        SET renewed_at = %s, lease_until = %s
                        WHERE store_id = %s AND election_id = %s
                        """,
                        (
                            renewed.renewed_at,
                            renewed.lease_until,
                            self._store_id,
                            renewed.election_id,
                        ),
                    )
                    return renewed

    def authorize(self, lease: RunnerLeaderLease) -> RunnerWriterAuthority:
        lease = _validated_lease(lease)
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    now, row = self._read_locked(cursor, lease.election_id)
                    current = self._current(row, now=now, expected=lease)
                    return RunnerWriterAuthority(
                        election_id=current.election_id,
                        holder_id=current.holder_id,
                        fencing_token=current.fencing_token,
                        validated_at=now,
                        lease_until=current.lease_until,
                    )

    def release(self, lease: RunnerLeaderLease) -> None:
        lease = _validated_lease(lease)
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    now, row = self._read_locked(cursor, lease.election_id)
                    current = self._current(row, now=now, expected=lease)
                    cursor.execute(
                        """
                        UPDATE public.runner_leader_leases
                        SET lease_until = %s
                        WHERE store_id = %s AND election_id = %s
                        """,
                        (now, self._store_id, current.election_id),
                    )

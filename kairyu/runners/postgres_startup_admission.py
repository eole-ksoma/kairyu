"""PostgreSQL-backed incremental Runner cache placement admission state."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime, timedelta
from types import TracebackType
from typing import Literal, Self

from kairyu.runners.startup_admission import (
    RunnerCachePlacementAdmissionAllocation,
    RunnerCachePlacementAdmissionClaim,
    RunnerCachePlacementAdmissionConflictError,
    RunnerCachePlacementAdmissionPlan,
    _aware,
    _text,
)

try:  # Optional deployment dependency.
    import psycopg  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - core-only installation.
    psycopg = None  # type: ignore[assignment]

_SCHEMA_VERSION = 1
_SCHEMA_NAME = "public"
_SCHEMA_LOCK = (1_261_587_810, 10)
_EXPECTED_COLUMNS = {
    "runner_cache_placement_admission_store_registry": (
        ("store_id", "text", True),
        ("schema_version", "integer", True),
        ("max_targets", "bigint", True),
        ("replay_safety_window_us", "bigint", True),
        ("created_at", "timestamp with time zone", True),
    ),
    "runner_cache_placement_admission_plans": (
        ("store_id", "text", True),
        ("target_id", "text", True),
        ("binding_id", "text", True),
        ("plan_digest", "text", True),
        ("valid_until", "timestamp with time zone", True),
        ("registered_at", "timestamp with time zone", True),
        ("plan", "jsonb", True),
        ("created_at", "timestamp with time zone", True),
    ),
    "runner_cache_placement_admission_claims": (
        ("store_id", "text", True),
        ("target_id", "text", True),
        ("pod_key", "text", True),
        ("binding_id", "text", True),
        ("admission_uid", "text", True),
        ("placement_id", "text", True),
        ("claimed_at", "timestamp with time zone", True),
        ("protected", "boolean", True),
        ("claim", "jsonb", True),
        ("created_at", "timestamp with time zone", True),
    ),
}
_EXPECTED_REGISTRY_CONSTRAINTS = {
    ("c", "CHECK ((store_id <> ''::text))", 0, True, False, False),
    (
        "c",
        "CHECK (((max_targets > 0) AND (max_targets <= 100000)))",
        0,
        True,
        False,
        False,
    ),
    ("c", "CHECK ((replay_safety_window_us > 0))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id)", 0, True, False, False),
}
_EXPECTED_PLAN_CONSTRAINTS_WITHOUT_TARGET = {
    ("c", "CHECK ((target_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((binding_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((plan_digest <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((registered_at < valid_until))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id, target_id)", 0, True, False, False),
}
_EXPECTED_CLAIM_CONSTRAINTS_WITHOUT_TARGET = {
    ("c", "CHECK ((target_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((pod_key <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((binding_id <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((admission_uid <> ''::text))", 0, True, False, False),
    ("c", "CHECK ((placement_id <> ''::text))", 0, True, False, False),
    ("p", "PRIMARY KEY (store_id, target_id, pod_key)", 0, True, False, False),
    ("u", "UNIQUE (store_id, target_id, placement_id)", 0, True, False, False),
}
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS public.runner_cache_placement_admission_store_registry (
        store_id TEXT PRIMARY KEY CHECK (store_id <> ''),
        schema_version INTEGER NOT NULL,
        max_targets BIGINT NOT NULL CHECK (max_targets > 0 AND max_targets <= 100000),
        replay_safety_window_us BIGINT NOT NULL CHECK (replay_safety_window_us > 0),
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.runner_cache_placement_admission_plans (
        store_id TEXT NOT NULL REFERENCES
            public.runner_cache_placement_admission_store_registry(store_id)
            ON DELETE CASCADE,
        target_id TEXT NOT NULL CHECK (target_id <> ''),
        binding_id TEXT NOT NULL CHECK (binding_id <> ''),
        plan_digest TEXT NOT NULL CHECK (plan_digest <> ''),
        valid_until TIMESTAMPTZ NOT NULL,
        registered_at TIMESTAMPTZ NOT NULL,
        plan JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        CHECK (registered_at < valid_until),
        PRIMARY KEY (store_id, target_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.runner_cache_placement_admission_claims (
        store_id TEXT NOT NULL,
        target_id TEXT NOT NULL CHECK (target_id <> ''),
        pod_key TEXT NOT NULL CHECK (pod_key <> ''),
        binding_id TEXT NOT NULL CHECK (binding_id <> ''),
        admission_uid TEXT NOT NULL CHECK (admission_uid <> ''),
        placement_id TEXT NOT NULL CHECK (placement_id <> ''),
        claimed_at TIMESTAMPTZ NOT NULL,
        protected BOOLEAN NOT NULL,
        claim JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (store_id, target_id, pod_key),
        UNIQUE (store_id, target_id, placement_id),
        FOREIGN KEY (store_id, target_id) REFERENCES
            public.runner_cache_placement_admission_plans(store_id, target_id)
            ON DELETE CASCADE
    )
    """,
)


def _timedelta_microseconds(value: timedelta) -> int:
    if not isinstance(value, timedelta) or value <= timedelta(0):
        raise ValueError("replay_safety_window must be positive")
    microseconds = (
        value.days * 86_400_000_000
        + value.seconds * 1_000_000
        + value.microseconds
    )
    if microseconds > 9_223_372_036_854_775_807:
        raise ValueError("replay_safety_window exceeds PostgreSQL BIGINT capacity")
    return microseconds


def _plan_digest(plan: RunnerCachePlacementAdmissionPlan) -> str:
    payload = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class PostgresRunnerCachePlacementAdmissionStore:
    """Linearizable shared plan/claim store for admission webhook replicas."""

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        max_targets: int = 10_000,
        replay_safety_window: timedelta = timedelta(minutes=5),
        connect_timeout_s: float = 10.0,
        eager_connect: bool = True,
        initialize_schema: bool = True,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "PostgresRunnerCachePlacementAdmissionStore requires psycopg; "
                "install the fleet dependency"
            )
        if not isinstance(dsn, str) or not dsn.strip() or "\x00" in dsn:
            raise ValueError("dsn must be a non-empty string without NUL")
        self._store_id = _text(store_id, name="store_id")
        if type(max_targets) is not int or not 1 <= max_targets <= 100_000:
            raise ValueError("max_targets must be an integer in [1, 100000]")
        replay_window_us = _timedelta_microseconds(replay_safety_window)
        if isinstance(connect_timeout_s, bool) or not isinstance(
            connect_timeout_s, (int, float)
        ):
            raise ValueError("connect_timeout_s must be a number")
        timeout = float(connect_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("connect_timeout_s must be finite and greater than zero")
        self._dsn = dsn
        self._max_targets = max_targets
        self._replay_safety_window = replay_safety_window
        self._replay_safety_window_us = replay_window_us
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
            raise RuntimeError("PostgresRunnerCachePlacementAdmissionStore is closed")
        if not allow_unstarted and self._connection is None:
            raise RuntimeError(
                "PostgresRunnerCachePlacementAdmissionStore has not started"
            )

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
                    INSERT INTO public.runner_cache_placement_admission_store_registry (
                        store_id, schema_version, max_targets, replay_safety_window_us
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (store_id) DO NOTHING
                    """,
                    (
                        self._store_id,
                        _SCHEMA_VERSION,
                        self._max_targets,
                        self._replay_safety_window_us,
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
                if (
                    expected_namespace_oid is not None
                    and namespace_oid != expected_namespace_oid
                ):
                    raise RuntimeError(
                        "Runner cache placement admission PostgreSQL namespace "
                        "identity changed"
                    )
                self._validate_registry_cursor(cursor)
                return namespace_oid

    def _validate_registry_cursor(self, cursor, *, for_update: bool = False) -> None:
        lock = " FOR UPDATE" if for_update else ""
        cursor.execute(
            """
            SELECT schema_version, max_targets, replay_safety_window_us
            FROM public.runner_cache_placement_admission_store_registry
            WHERE store_id = %s
            """
            + lock,
            (self._store_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"unknown Runner cache placement admission store {self._store_id!r}"
            )
        expected = (
            _SCHEMA_VERSION,
            self._max_targets,
            self._replay_safety_window_us,
        )
        if row != expected:
            raise RuntimeError(
                "Runner cache placement admission store configuration changed: "
                f"observed {row!r}; expected {expected!r}"
            )

    @staticmethod
    def _validate_schema_objects_cursor(cursor) -> int:
        cursor.execute(
            """
            SELECT registry.oid, plans.oid, claims.oid,
                   registry.relnamespace, plans.relnamespace, claims.relnamespace,
                   registry.relkind, plans.relkind, claims.relkind,
                   registry.relpersistence, plans.relpersistence,
                   claims.relpersistence
            FROM pg_catalog.pg_class AS registry
            CROSS JOIN pg_catalog.pg_class AS plans
            CROSS JOIN pg_catalog.pg_class AS claims
            WHERE registry.oid = pg_catalog.to_regclass(
                'public.runner_cache_placement_admission_store_registry'
            )
              AND plans.oid = pg_catalog.to_regclass(
                'public.runner_cache_placement_admission_plans'
              )
              AND claims.oid = pg_catalog.to_regclass(
                'public.runner_cache_placement_admission_claims'
              )
            """
        )
        objects = cursor.fetchone()
        if objects is None:
            raise RuntimeError(
                "Runner cache placement admission schema is missing required tables"
            )
        (
            registry_oid,
            plans_oid,
            claims_oid,
            registry_namespace,
            plans_namespace,
            claims_namespace,
            *kinds,
        ) = objects
        cursor.execute("SELECT pg_catalog.to_regnamespace(%s)::oid", (_SCHEMA_NAME,))
        namespace_row = cursor.fetchone()
        if (
            registry_namespace != plans_namespace
            or registry_namespace != claims_namespace
            or namespace_row is None
            or registry_namespace != namespace_row[0]
        ):
            raise RuntimeError(
                "Runner cache placement admission tables must use the public namespace"
            )
        if kinds != ["r", "r", "r", "p", "p", "p"]:
            raise RuntimeError(
                "Runner cache placement admission schema objects must be permanent "
                "ordinary tables"
            )
        for table_name, table_oid in (
            ("runner_cache_placement_admission_store_registry", registry_oid),
            ("runner_cache_placement_admission_plans", plans_oid),
            ("runner_cache_placement_admission_claims", claims_oid),
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
                    f"Runner cache placement admission table {table_name!r} "
                    "has incompatible columns"
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
            raise RuntimeError(
                "Runner cache placement admission registry has incompatible constraints"
            )
        expected_plan_constraints = _EXPECTED_PLAN_CONSTRAINTS_WITHOUT_TARGET | {
            (
                "f",
                "FOREIGN KEY (store_id) REFERENCES "
                "runner_cache_placement_admission_store_registry(store_id) "
                "ON DELETE CASCADE",
                registry_oid,
                True,
                False,
                False,
            )
        }
        if constraints(plans_oid) != expected_plan_constraints:
            raise RuntimeError(
                "Runner cache placement admission plans have incompatible constraints"
            )
        expected_claim_constraints = _EXPECTED_CLAIM_CONSTRAINTS_WITHOUT_TARGET | {
            (
                "f",
                "FOREIGN KEY (store_id, target_id) REFERENCES "
                "runner_cache_placement_admission_plans(store_id, target_id) "
                "ON DELETE CASCADE",
                plans_oid,
                True,
                False,
                False,
            )
        }
        if constraints(claims_oid) != expected_claim_constraints:
            raise RuntimeError(
                "Runner cache placement admission claims have incompatible constraints"
            )
        return registry_namespace

    @staticmethod
    def _plan(row) -> RunnerCachePlacementAdmissionPlan:
        target_id, binding_id, plan_digest, valid_until, registered_at, payload = row
        plan = RunnerCachePlacementAdmissionPlan.model_validate(payload)
        if plan_digest != _plan_digest(plan):
            raise RuntimeError(
                "stored Runner cache placement admission plan digest is inconsistent"
            )
        if (
            target_id,
            binding_id,
            valid_until,
            registered_at,
        ) != (
            plan.binding.target_id,
            plan.binding.binding_id,
            plan.binding.valid_until,
            plan.registered_at,
        ):
            raise RuntimeError(
                "stored Runner cache placement admission plan metadata is inconsistent"
            )
        return plan

    @staticmethod
    def _claim(row) -> tuple[RunnerCachePlacementAdmissionClaim, bool]:
        (
            target_id,
            pod_key,
            binding_id,
            admission_uid,
            placement_id,
            claimed_at,
            protected,
            payload,
        ) = row
        claim = RunnerCachePlacementAdmissionClaim.model_validate(payload)
        if (
            target_id,
            pod_key,
            binding_id,
            admission_uid,
            placement_id,
            claimed_at,
        ) != (
            claim.target_id,
            claim.pod_key,
            claim.binding_id,
            claim.admission_uid,
            claim.placement_id,
            claim.claimed_at,
        ):
            raise RuntimeError(
                "stored Runner cache placement admission claim metadata is inconsistent"
            )
        if not isinstance(protected, bool):
            raise RuntimeError("stored admission claim protection flag is invalid")
        return claim, protected

    @staticmethod
    def _validate_claim_binding(
        claim: RunnerCachePlacementAdmissionClaim,
        *,
        binding_id: str,
        placement_ids: frozenset[str],
    ) -> None:
        if claim.binding_id != binding_id:
            raise RuntimeError(
                "stored Runner cache placement admission claim references another binding"
            )
        if claim.placement_id not in placement_ids:
            raise RuntimeError(
                "stored Runner cache placement admission claim references an unknown "
                "placement"
            )

    @staticmethod
    def _plan_columns() -> str:
        return (
            "target_id, binding_id, plan_digest, valid_until, registered_at, plan"
        )

    @staticmethod
    def _database_now_cursor(cursor) -> datetime:
        cursor.execute("SELECT clock_timestamp()")
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return its current timestamp")
        return _aware(row[0], name="database_now")

    @staticmethod
    def _claim_columns() -> str:
        return (
            "target_id, pod_key, binding_id, admission_uid, placement_id, "
            "claimed_at, protected, claim"
        )

    def register(self, plan: RunnerCachePlacementAdmissionPlan) -> None:
        if not isinstance(plan, RunnerCachePlacementAdmissionPlan):
            raise TypeError("plan must be a RunnerCachePlacementAdmissionPlan")
        plan = RunnerCachePlacementAdmissionPlan.model_validate(plan.model_dump())
        target_id = plan.binding.target_id
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor, for_update=True)
                    cursor.execute(
                        f"""
                        SELECT {self._plan_columns()}
                        FROM public.runner_cache_placement_admission_plans
                        WHERE store_id = %s AND target_id = %s
                        """,
                        (self._store_id, target_id),
                    )
                    row = cursor.fetchone()
                    previous = None if row is None else self._plan(row)
                    if (
                        previous is not None
                        and previous.binding.binding_id == plan.binding.binding_id
                    ):
                        if previous != plan:
                            raise RunnerCachePlacementAdmissionConflictError(
                                "binding ID is already registered with different plan "
                                "evidence"
                            )
                        return
                    database_now = self._database_now_cursor(cursor)
                    if not (
                        plan.binding.bound_at
                        <= database_now
                        < plan.binding.valid_until
                    ):
                        raise RunnerCachePlacementAdmissionConflictError(
                            "cache placement admission binding is not live at the "
                            "database clock"
                        )
                    cursor.execute(
                        """
                        SELECT count(*)
                        FROM public.runner_cache_placement_admission_claims
                        WHERE store_id = %s AND target_id = %s
                        """,
                        (self._store_id, target_id),
                    )
                    count_row = cursor.fetchone()
                    assert count_row is not None
                    prior_claim_count = count_row[0]
                    if previous is not None:
                        database_elapsed = database_now - previous.binding.valid_until
                        registered_elapsed = (
                            plan.registered_at - previous.binding.valid_until
                        )
                        required_elapsed = (
                            self._replay_safety_window
                            if prior_claim_count
                            else timedelta(0)
                        )
                        if (
                            database_elapsed < required_elapsed
                            or registered_elapsed < required_elapsed
                        ):
                            raise RunnerCachePlacementAdmissionConflictError(
                                "admission plan cannot be replaced before its replay "
                                "safety window"
                            )
                    else:
                        cursor.execute(
                            """
                            SELECT count(*)
                            FROM public.runner_cache_placement_admission_plans
                            WHERE store_id = %s
                            """,
                            (self._store_id,),
                        )
                        plan_count_row = cursor.fetchone()
                        assert plan_count_row is not None
                        if plan_count_row[0] >= self._max_targets:
                            raise RunnerCachePlacementAdmissionConflictError(
                                "admission plan capacity is exhausted"
                            )
                    assert psycopg is not None
                    cursor.execute(
                        """
                        DELETE FROM public.runner_cache_placement_admission_claims
                        WHERE store_id = %s AND target_id = %s
                        """,
                        (self._store_id, target_id),
                    )
                    cursor.execute(
                        """
                        INSERT INTO public.runner_cache_placement_admission_plans (
                            store_id, target_id, binding_id, plan_digest, valid_until,
                            registered_at, plan
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (store_id, target_id) DO UPDATE SET
                            binding_id = EXCLUDED.binding_id,
                            plan_digest = EXCLUDED.plan_digest,
                            valid_until = EXCLUDED.valid_until,
                            registered_at = EXCLUDED.registered_at,
                            plan = EXCLUDED.plan
                        """,
                        (
                            self._store_id,
                            target_id,
                            plan.binding.binding_id,
                            _plan_digest(plan),
                            plan.binding.valid_until,
                            plan.registered_at,
                            psycopg.types.json.Jsonb(plan.model_dump(mode="json")),
                        ),
                    )

    def resolve(self, target_id: str) -> RunnerCachePlacementAdmissionPlan:
        target_id = _text(target_id, name="target_id")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)
                    cursor.execute(
                        f"""
                        SELECT {self._plan_columns()}
                        FROM public.runner_cache_placement_admission_plans
                        WHERE store_id = %s AND target_id = %s
                        """,
                        (self._store_id, target_id),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RunnerCachePlacementAdmissionConflictError(
                            "no active cache placement admission plan"
                        )
                    return RunnerCachePlacementAdmissionPlan.model_validate(
                        self._plan(row).model_dump()
                    )

    def claim(
        self,
        *,
        target_id: str,
        binding_id: str,
        pod_key: str,
        admission_uid: str,
        claimed_at: datetime,
    ) -> RunnerCachePlacementAdmissionAllocation:
        target_id = _text(target_id, name="target_id")
        binding_id = _text(binding_id, name="binding_id")
        pod_key = _text(pod_key, name="pod_key", max_length=512)
        admission_uid = _text(admission_uid, name="admission_uid")
        claimed_at = _aware(claimed_at, name="claimed_at")
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor, for_update=True)
                    cursor.execute(
                        f"""
                        SELECT {self._plan_columns()}
                        FROM public.runner_cache_placement_admission_plans
                        WHERE store_id = %s AND target_id = %s
                        """,
                        (self._store_id, target_id),
                    )
                    row = cursor.fetchone()
                    plan = None if row is None else self._plan(row)
                    if plan is None or plan.binding.binding_id != binding_id:
                        raise RunnerCachePlacementAdmissionConflictError(
                            "admission plan changed before placement claim"
                        )
                    database_now = self._database_now_cursor(cursor)
                    if not (
                        plan.binding.bound_at
                        <= database_now
                        < plan.binding.valid_until
                    ) or not (
                        plan.binding.bound_at
                        <= claimed_at
                        < plan.binding.valid_until
                    ):
                        raise RunnerCachePlacementAdmissionConflictError(
                            "cache placement admission binding is not live"
                        )
                    placement_ids = frozenset(
                        placement.placement_id for placement in plan.binding.placements
                    )
                    cursor.execute(
                        f"""
                        SELECT {self._claim_columns()}
                        FROM public.runner_cache_placement_admission_claims
                        WHERE store_id = %s AND target_id = %s AND pod_key = %s
                        """,
                        (self._store_id, target_id, pod_key),
                    )
                    existing_row = cursor.fetchone()
                    if existing_row is not None:
                        existing, protected = self._claim(existing_row)
                        self._validate_claim_binding(
                            existing,
                            binding_id=binding_id,
                            placement_ids=placement_ids,
                        )
                        if not protected:
                            cursor.execute(
                                """
                                UPDATE public.runner_cache_placement_admission_claims
                                SET protected = TRUE
                                WHERE store_id = %s AND target_id = %s AND pod_key = %s
                                """,
                                (self._store_id, target_id, pod_key),
                            )
                        return RunnerCachePlacementAdmissionAllocation(
                            claim=RunnerCachePlacementAdmissionClaim.model_validate(
                                existing.model_dump()
                            ),
                            created=False,
                        )
                    cursor.execute(
                        f"""
                        SELECT {self._claim_columns()}
                        FROM public.runner_cache_placement_admission_claims
                        WHERE store_id = %s AND target_id = %s
                        ORDER BY pod_key
                        """,
                        (self._store_id, target_id),
                    )
                    used: set[str] = set()
                    for item in cursor.fetchall():
                        stored_claim, _protected = self._claim(item)
                        self._validate_claim_binding(
                            stored_claim,
                            binding_id=binding_id,
                            placement_ids=placement_ids,
                        )
                        used.add(stored_claim.placement_id)
                    placement = next(
                        (
                            candidate
                            for candidate in plan.binding.placements
                            if candidate.placement_id not in used
                        ),
                        None,
                    )
                    if placement is None:
                        raise RunnerCachePlacementAdmissionConflictError(
                            "cache placement admission plan is exhausted"
                        )
                    claim = RunnerCachePlacementAdmissionClaim(
                        binding_id=binding_id,
                        target_id=target_id,
                        pod_key=pod_key,
                        admission_uid=admission_uid,
                        placement_id=placement.placement_id,
                        claimed_at=claimed_at,
                    )
                    assert psycopg is not None
                    cursor.execute(
                        """
                        INSERT INTO public.runner_cache_placement_admission_claims (
                            store_id, target_id, pod_key, binding_id, admission_uid,
                            placement_id, claimed_at, protected, claim
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, %s)
                        """,
                        (
                            self._store_id,
                            claim.target_id,
                            claim.pod_key,
                            claim.binding_id,
                            claim.admission_uid,
                            claim.placement_id,
                            claim.claimed_at,
                            psycopg.types.json.Jsonb(claim.model_dump(mode="json")),
                        ),
                    )
                    return RunnerCachePlacementAdmissionAllocation(
                        claim=claim,
                        created=True,
                    )

    def release(self, claim: RunnerCachePlacementAdmissionClaim) -> None:
        if not isinstance(claim, RunnerCachePlacementAdmissionClaim):
            raise TypeError("claim must be a RunnerCachePlacementAdmissionClaim")
        claim = RunnerCachePlacementAdmissionClaim.model_validate(claim.model_dump())
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor, for_update=True)
                    cursor.execute(
                        f"""
                        SELECT {self._claim_columns()}
                        FROM public.runner_cache_placement_admission_claims
                        WHERE store_id = %s AND target_id = %s AND pod_key = %s
                        """,
                        (self._store_id, claim.target_id, claim.pod_key),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return
                    current, protected = self._claim(row)
                    cursor.execute(
                        f"""
                        SELECT {self._plan_columns()}
                        FROM public.runner_cache_placement_admission_plans
                        WHERE store_id = %s AND target_id = %s
                        """,
                        (self._store_id, claim.target_id),
                    )
                    plan_row = cursor.fetchone()
                    if plan_row is None:
                        raise RuntimeError(
                            "stored admission claim has no parent admission plan"
                        )
                    plan = self._plan(plan_row)
                    self._validate_claim_binding(
                        current,
                        binding_id=plan.binding.binding_id,
                        placement_ids=frozenset(
                            placement.placement_id
                            for placement in plan.binding.placements
                        ),
                    )
                    if current == claim and not protected:
                        cursor.execute(
                            """
                            DELETE FROM public.runner_cache_placement_admission_claims
                            WHERE store_id = %s AND target_id = %s AND pod_key = %s
                            """,
                            (self._store_id, claim.target_id, claim.pod_key),
                        )

    def check_ready(self) -> None:
        """Validate the live connection and durable store configuration."""
        with self._lock:
            self._require_open()
            self._ensure_connection()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._validate_registry_cursor(cursor)

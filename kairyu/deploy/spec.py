"""DeploymentSpec: the YAML schema consumed by `kairyu serve` (design m7 D3).

Deliberately separate from ``ClusterSpec`` (m6 D1): ClusterSpec validates the
2-node TP/PP/P-D coherence domain; a serving deployment declares which models
the process serves and which remote replicas a pool fans out to. A replica
node's intra-node GPU layout may reference a ClusterSpec file — the two specs
compose, they do not merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.resolver import BaseResolver

from kairyu.engine.openai_capabilities import resolve_openai_capabilities
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.pricing import PriceSheet

_DEFAULT_PORT = 8000
_MERGE_TAG = "tag:yaml.org,2002:merge"
_MERGE_KEY_IDENTITY = object()


class _DuplicateKeyError(ValueError):
    """Duplicate-key error retaining lossless path segments for safe rendering."""

    def __init__(self, key: object, path: tuple[str, ...]) -> None:
        location = ".".join(path) or "<root>"
        super().__init__(f"duplicate mapping key {key!r} at {location}")
        self.path = path


class BackendSpec(BaseModel):
    """One engine backend: registry name + factory kwargs."""

    model_config = ConfigDict(frozen=True)

    backend: str
    options: dict = Field(default_factory=dict)
    health_url: str | None = Field(
        default=None,
        description=(
            "Readiness endpoint the prober GETs for this replica; defaults to "
            "<base_url minus /v1>/readyz for openai backends, None otherwise. "
            "Readiness (not liveness) so a drained/wedged node stays ejected (O3)."
        ),
    )

    @model_validator(mode="after")
    def _validate_openai_capabilities(self) -> BackendSpec:
        if self.backend == "openai":
            resolve_openai_capabilities(
                self.options.get("upstream", "generic"),
                self.options.get("capabilities"),
            )
        return self

    def resolved_health_url(self) -> str | None:
        if self.health_url is not None:
            return self.health_url
        base = self.options.get("base_url")
        if not isinstance(base, str):
            return None
        root = base.rstrip("/")
        root = root.removesuffix("/v1")
        return f"{root}/readyz"


class KubernetesEndpointSliceSpec(BaseModel):
    """In-cluster EndpointSlice discovery for one headless replica Service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["kubernetes_endpoints"] = "kubernetes_endpoints"
    service: str = Field(
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
    )
    namespace: str | None = Field(
        default=None,
        description="Defaults to the mounted service-account namespace.",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
    )
    port: str | int = "http"
    scheme: Literal["http", "https"] = "http"
    address_family_preference: Literal["ipv4", "ipv6"] = "ipv4"
    model: str | None = None
    api_key_env: str | None = None
    poll_interval_s: float = Field(default=0.25, gt=0)

    @field_validator("port", mode="before")
    @classmethod
    def _reject_boolean_port(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("discovery port must be a name or an integer")
        return value

    @model_validator(mode="after")
    def _valid_port(self) -> KubernetesEndpointSliceSpec:
        if isinstance(self.port, int):
            if not 1 <= self.port <= 65535:
                raise ValueError("discovery integer port must be in 1..65535")
        elif not self.port.strip():
            raise ValueError("discovery port name must not be empty")
        return self


class PoolSpec(BaseModel):
    """A static or EndpointSlice-discovered ReplicaPool."""

    model_config = ConfigDict(frozen=True)

    replicas: tuple[BackendSpec, ...] = ()
    discovery: KubernetesEndpointSliceSpec | None = None
    unhealthy_after: int = Field(default=3, ge=1)
    queue_depth_threshold: int = Field(default=8, ge=0)
    prefix_index: bool = Field(
        default=False,
        description=(
            "Enable the bounded process-local PrefixIndex for cross-session "
            "KV-aware placement."
        ),
    )
    probe_interval_s: float = Field(
        default=5.0,
        gt=0,
        description="Interval of the serve-layer health prober (m7 D4).",
    )
    placement_log_path: str | None = Field(
        default=None,
        description=(
            "Optional bounded JSONL placement/membership evidence path. "
            "Session IDs remain SHA-256-only."
        ),
    )

    @model_validator(mode="after")
    def _membership_source(self) -> PoolSpec:
        if bool(self.replicas) == (self.discovery is not None):
            raise ValueError(
                "pool must declare exactly one of replicas or discovery"
            )
        return self


class ServerSection(BaseModel):
    """Versioned deployment-facing server schema.

    The YAML surface deliberately owns its fields instead of inheriting the
    runtime ``ServerSettings`` model. ``to_server_settings`` is the one
    explicit translation boundary, so runtime-only additions cannot silently
    change accepted deployment artifacts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=_DEFAULT_PORT, ge=1, le=65535)
    api_keys_env: str | None = Field(
        default=None,
        description=(
            "Env var holding comma-separated API keys; None disables auth "
            "(keyless node-to-node replicas, design m6 D2 / m7 D5)."
        ),
    )
    responses_compaction_secret_env: str | None = Field(
        default=None,
        description=(
            "Env var holding a secret used to encrypt Responses compaction tokens; "
            "None uses a process-local ephemeral key."
        ),
    )
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Global active-plus-queued cap on /v1/* requests; None disables "
            "the guard."
        ),
    )
    admission_wait_timeout_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Maximum wait for a backend-aware active request slot; None "
            "preserves immediate saturation rejection."
        ),
    )
    ttft_slo_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "TTFT target for direct-chat SLO admission; None disables "
            "predictive admit/defer/shed decisions."
        ),
    )
    max_chat_body_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum raw POST body for /v1/chat/completions; None disables "
            "the process-level guard."
        ),
    )
    metrics: bool = Field(default=True, description="Expose /metrics (Prometheus).")
    protect_metrics: bool = Field(
        default=False,
        description="Require an API key for /metrics too.",
    )
    access_log: bool = Field(
        default=True,
        description="Emit one JSON access-log line per request.",
    )
    tracing: bool = Field(
        default=False,
        description="Enable OTel spans (needs the otel extra; no-op without it).",
    )
    usage_ledger_path: str | None = Field(
        default=None,
        description="JSONL usage-ledger path; None disables metering (m11 D3).",
    )
    admin_keys_env: str | None = Field(
        default=None,
        description=(
            "Env var holding comma-separated ADMIN API keys; when set, /admin/* "
            "state changes (drain/undrain) require one of these, so an ordinary "
            "data-plane key cannot take the node out of service (S5)."
        ),
    )

    @model_validator(mode="after")
    def _admission_queue_requires_total_bound(self) -> ServerSection:
        if self.admission_wait_timeout_s is not None and self.max_concurrency is None:
            raise ValueError(
                "admission_wait_timeout_s requires max_concurrency"
            )
        return self

    def to_server_settings(self) -> ServerSettings:
        """Translate the deployment vocabulary to runtime settings explicitly."""

        return ServerSettings(
            api_keys_env=self.api_keys_env,
            responses_compaction_secret_env=self.responses_compaction_secret_env,
            max_concurrency=self.max_concurrency,
            admission_wait_timeout_s=self.admission_wait_timeout_s,
            ttft_slo_s=self.ttft_slo_s,
            max_chat_body_bytes=self.max_chat_body_bytes,
            metrics=self.metrics,
            protect_metrics=self.protect_metrics,
            access_log=self.access_log,
            tracing=self.tracing,
            usage_ledger_path=self.usage_ledger_path,
            admin_keys_env=self.admin_keys_env,
        )


class OrchestratorSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: str = Field(description="Path to an OrchestratorSpec YAML (kairyu.dsl).")


class ExecutorSection(BaseModel):
    """One deployment-owned sandbox execution service (ECO-D1).

    Executors are internal orchestration resources, never served models; an
    orchestration worker borrows one by ``executor_ref``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(min_length=1)
    uds_path: str | None = Field(default=None, min_length=1, pattern=r"^/")
    timeout_s: float = Field(default=25.0, gt=0, le=300)
    queue_wait_s: float = Field(default=0.0, ge=0, le=300)


class MockEmbeddingSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["mock"]
    dimensions: int = Field(default=64, ge=1)


class FastEmbedEmbeddingSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["fastembed"]
    dimensions: int = Field(ge=1)
    model: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_size: int = Field(default=64, ge=1, le=2048)
    threads: int = Field(default=2, ge=1, le=128)
    max_concurrency: int = Field(default=2, ge=1, le=64)


EmbeddingSection = Annotated[
    MockEmbeddingSection | FastEmbedEmbeddingSection,
    Field(discriminator="backend"),
]


class BatchSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    store: Literal["filesystem", "postgres"] = "filesystem"
    data_dir: str | None = None
    dsn_env: str = Field(
        default="KAIRYU_BATCH_POSTGRES_DSN",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    store_id: str = Field(default="kairyu", min_length=1, max_length=128)
    spool_dir: str | None = None
    max_concurrency: int = Field(default=4, ge=1)
    poll_interval_s: float = Field(default=0.5, ge=0.05)
    lease_seconds: float = Field(default=30.0, ge=1.0)

    @model_validator(mode="after")
    def _store_config(self) -> BatchSection:
        if self.store == "filesystem" and not self.data_dir:
            raise ValueError("batch.data_dir is required for the filesystem store")
        if self.store == "filesystem" and self.spool_dir is not None:
            raise ValueError("batch.spool_dir is only valid for the postgres store")
        if self.store == "postgres" and self.data_dir is not None:
            raise ValueError("batch.data_dir cannot be set for the postgres store")
        if not self.store_id.strip() or "\x00" in self.store_id:
            raise ValueError("batch.store_id must be a non-empty PostgreSQL identity")
        return self


class AsyncRequestsSection(BaseModel):
    """Durable asynchronous Chat Completions worker configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn_env: str = Field(
        default="KAIRYU_ASYNC_REQUEST_POSTGRES_DSN",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    store_id: str = Field(default="kairyu", min_length=1, max_length=128)
    max_concurrency: int = Field(default=4, ge=1)
    max_body_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    max_records_per_tenant: int = Field(default=64, ge=1)
    poll_interval_s: float = Field(default=0.5, ge=0.05)
    lease_seconds: float = Field(default=30.0, ge=1.0)
    request_retention_s: float | None = Field(
        default=None,
        ge=60.0,
        le=315_360_000.0,
    )
    audit_retention_s: float | None = Field(
        default=None,
        ge=60.0,
        le=315_360_000.0,
    )
    retention_batch_size: int = Field(default=500, ge=1, le=10_000)

    @field_validator("store_id")
    @classmethod
    def _valid_store_id(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError(
                "async_requests.store_id must be a non-empty PostgreSQL identity"
            )
        return value


class TenantLimitsSection(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    requests_per_minute: int = Field(default=600, ge=1)
    tokens_per_minute: int = Field(default=200_000, ge=1)
    request_burst: int | None = Field(default=None, ge=1)
    token_burst: int | None = Field(default=None, ge=1)
    max_in_flight: int | None = Field(default=None, ge=1)
    interactive_priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)
    batch_priority: int = Field(default=1, ge=-(2**63), le=2**63 - 1)

    @model_validator(mode="after")
    def _priority_precedence(self) -> TenantLimitsSection:
        if self.interactive_priority >= self.batch_priority:
            raise ValueError(
                "interactive_priority must be smaller than batch_priority"
            )
        return self


class TenantSection(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    default_tenant: str = "default"
    key_tenants: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )
    limits: dict[str, TenantLimitsSection] = Field(default_factory=dict)


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    server: ServerSection = ServerSection()
    engines: dict[str, BackendSpec] = Field(default_factory=dict)
    pools: dict[str, PoolSpec] = Field(default_factory=dict)
    # served model name -> HF Jinja chat template (inline text or *.jinja path);
    # per-MODEL, not per-replica — one map avoids engines-vs-pools ambiguity
    # and stays out of BackendSpec.options (factory kwargs). m9 D2.
    chat_templates: dict[str, str] = Field(default_factory=dict)
    # Explicit escape hatch for deployments that intentionally retain the
    # pre-M9 role-prefix concatenator. Template-preserving local deployments
    # should instead use their tokenizer-owned template (auto-loaded by the
    # builder) or a ``chat_templates`` override. Remote/discovery/orchestration
    # paths currently require this model-scoped compatibility choice because
    # they cannot retain pre-rendered tokenizer ownership downstream.
    legacy_chat_models: frozenset[str] = frozenset()
    orchestrator: OrchestratorSection | None = None
    # served auto-model name -> orchestrator spec; serves ANY number of named
    # orchestrations (e.g. kairyu-auto + kairyu-auto-max) from one YAML. The
    # legacy single `orchestrator:` key stays and is served as "kairyu-auto".
    orchestrators: dict[str, OrchestratorSection] = Field(default_factory=dict)
    # deployment-owned sandbox execution services borrowed by orchestration
    # executor workers via executor_ref (ECO-D1); not served models.
    executors: dict[str, ExecutorSection] = Field(default_factory=dict)
    embeddings: dict[str, EmbeddingSection] = Field(default_factory=dict)
    # Omission preserves the generic all-model server. An explicit allowlist
    # separates public dispatch from deployment-owned internal resources.
    public_models: frozenset[str] | None = None
    batch: BatchSection | None = None
    async_requests: AsyncRequestsSection | None = None
    tenants: TenantSection | None = None
    pricing: PriceSheet | None = None

    @model_validator(mode="after")
    def _validate(self, info: ValidationInfo) -> DeploymentSpec:
        if not self.engines and not self.pools:
            raise ValueError("deployment spec must declare at least one engine or pool")
        overlap = self.engines.keys() & self.pools.keys()
        if overlap:
            raise ValueError(
                f"names {sorted(overlap)} appear in both engines: and pools:; "
                "served model names must be unique"
            )
        orchestration_names = set(self.orchestrators)
        if self.orchestrator is not None:
            orchestration_names.add("kairyu-auto")
        unknown = (
            self.chat_templates.keys()
            - self.engines.keys()
            - self.pools.keys()
            - orchestration_names
        )
        if unknown:
            raise ValueError(
                f"chat_templates for unknown models {sorted(unknown)}; "
                "keys must match engines:, pools:, or orchestrators: names"
            )
        unknown_legacy = (
            self.legacy_chat_models
            - self.engines.keys()
            - self.pools.keys()
            - orchestration_names
        )
        if unknown_legacy:
            raise ValueError(
                f"legacy_chat_models contains unknown models {sorted(unknown_legacy)}; "
                "names must match engines:, pools:, or orchestrators: names"
            )
        conflicting_chat_policy = self.legacy_chat_models & self.chat_templates.keys()
        if conflicting_chat_policy:
            raise ValueError(
                "models cannot declare both a real chat template and the legacy "
                f"chat fallback: {sorted(conflicting_chat_policy)}"
            )
        if any(not name for name in self.orchestrators):
            raise ValueError("orchestrators: names must be non-empty strings")
        auto_overlap = self.orchestrators.keys() & (self.engines.keys() | self.pools.keys())
        if auto_overlap:
            raise ValueError(
                f"orchestrators names {sorted(auto_overlap)} collide with "
                "engines:/pools: names; served model names must be unique"
            )
        if self.orchestrator is not None and "kairyu-auto" in self.orchestrators:
            raise ValueError(
                'both orchestrator: and orchestrators["kairyu-auto"] are set; '
                "declare kairyu-auto once (the legacy orchestrator: key is "
                'served as "kairyu-auto")'
            )
        if any(not name.strip() for name in self.executors):
            raise ValueError("executors: names must be non-empty strings")
        executor_overlap = self.executors.keys() & (
            self.engines.keys() | self.pools.keys() | orchestration_names
        )
        if executor_overlap:
            raise ValueError(
                f"executor names {sorted(executor_overlap)} collide with "
                "engines:/pools:/orchestrators: names"
            )
        if any(not name.strip() for name in self.embeddings):
            raise ValueError("embeddings: names must be non-empty strings")
        embedding_overlap = self.embeddings.keys() & (
            self.engines.keys() | self.pools.keys() | orchestration_names
        )
        if embedding_overlap:
            raise ValueError(
                f"embedding names {sorted(embedding_overlap)} collide with "
                "engines:/pools:/orchestrators: names; served model names must be unique"
            )
        if self.public_models is not None:
            if not self.public_models:
                raise ValueError("public_models must contain at least one model")
            known_models = {
                *self.engines,
                *self.pools,
                *orchestration_names,
                *self.embeddings,
            }
            unknown_public = self.public_models - known_models
            if unknown_public:
                raise ValueError(
                    f"public_models contains unknown models {sorted(unknown_public)}"
                )
        if self.tenants is not None:
            resolve_credentials = (
                info.context is None
                or info.context.get("resolve_credentials", True)
            )
            resolved_api_keys = (
                self.server.to_server_settings().resolve_api_keys()
                if resolve_credentials
                else frozenset(self.tenants.key_tenants)
            )
            TenantConfig.from_mapping(
                key_tenants=self.tenants.key_tenants,
                limits={
                    tenant: TenantLimits(
                        requests_per_minute=section.requests_per_minute,
                        tokens_per_minute=section.tokens_per_minute,
                        request_burst=section.request_burst,
                        token_burst=section.token_burst,
                        max_in_flight=section.max_in_flight,
                        interactive_priority=section.interactive_priority,
                        batch_priority=section.batch_priority,
                    )
                    for tenant, section in self.tenants.limits.items()
                },
                default_tenant=self.tenants.default_tenant,
                resolved_api_keys=resolved_api_keys,
            )
        if self.pricing is not None:
            if self.server.usage_ledger_path is None:
                raise ValueError("pricing requires server.usage_ledger_path")
            known_tenants = {"default"}
            if self.tenants is not None:
                known_tenants = {
                    self.tenants.default_tenant,
                    *self.tenants.key_tenants.values(),
                }
            unknown_discounts = self.pricing.tenant_discounts.keys() - known_tenants
            if unknown_discounts:
                raise ValueError(
                    f"pricing discounts reference unknown tenants {sorted(unknown_discounts)}"
                )
        return self


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that retains a path for duplicate-key diagnostics."""

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self._mapping_paths: dict[int, tuple[str, ...]] = {}

    def get_single_data(self):
        node = self.get_single_node()
        if node is None:
            return None
        self._index_paths(node, ())
        return self.construct_document(node)

    def _index_paths(
        self,
        node: Node,
        path: tuple[str, ...],
        visited: set[int] | None = None,
    ) -> None:
        if visited is None:
            visited = set()
        identity = id(node)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(node, MappingNode):
            self._mapping_paths[identity] = path
            scalar_keys: set[object] = set()
            for key_node, value_node in node.value:
                if isinstance(key_node, ScalarNode):
                    if key_node.tag == _MERGE_TAG:
                        key: object = _MERGE_KEY_IDENTITY
                        display_key: object = key_node.value
                    else:
                        # Use this SafeLoader's constructors so equality matches
                        # the keys that construct_mapping will actually insert.
                        key = self.construct_object(key_node)
                        display_key = key
                    try:
                        duplicate = key in scalar_keys
                        scalar_keys.add(key)
                    except TypeError:
                        duplicate = False
                    if duplicate:
                        raise _DuplicateKeyError(display_key, path)
                segment = key_node.value if isinstance(key_node, ScalarNode) else "<key>"
                self._index_paths(value_node, (*path, segment), visited)
        elif isinstance(node, SequenceNode):
            for index, value_node in enumerate(node.value):
                self._index_paths(value_node, (*path, f"[{index}]"), visited)


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
):
    seen: set[object] = set()
    for key_node, _ in node.value:
        if key_node.tag == _MERGE_TAG:
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError:
            continue
        if duplicate:
            raise _DuplicateKeyError(
                key,
                loader._mapping_paths.get(id(node), ()),
            )
    mapping: dict[object, object] = {}
    yield mapping
    mapping.update(yaml.SafeLoader.construct_mapping(loader, node, deep=deep))


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_deployment_spec(
    source: str | Path,
    *,
    resolve_credentials: bool = True,
) -> DeploymentSpec:
    """Load a DeploymentSpec from a YAML file path or a YAML string.

    ``resolve_credentials=False`` retains tenant topology validation without
    reading API-key environment variables. Serving keeps the default credential
    resolution behavior.
    """
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source.strip() and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(data, dict):
        raise ValueError("deployment spec YAML must be a mapping at the top level")
    return DeploymentSpec.model_validate(
        data,
        context={"resolve_credentials": resolve_credentials},
    )

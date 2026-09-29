"""DeploymentSpec -> running FastAPI app: engines, pools, prober, lifespan (m7 D3/D4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from ssl import SSLContext

import httpx
from fastapi import FastAPI
from jinja2.exceptions import TemplateError

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.deploy.prober import HealthProber
from kairyu.deploy.registry import (
    KubernetesEndpointSliceDiscovery,
    PoolReconciler,
    openai_replica_factory,
)
from kairyu.deploy.spec import DeploymentSpec, load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.dsl.spec import OrchestratorSpec
from kairyu.engine.backend import (
    EngineBackend,
    shutdown_all,
)
from kairyu.engine.embedding import (
    EmbeddingBackend,
    FastEmbedEmbeddingBackend,
    MockEmbeddingBackend,
)
from kairyu.engine.registry import create_backend
from kairyu.engine.tokenizer import (
    TokenizerChatMetadata,
    load_tokenizer_chat_metadata,
)
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.models.generation import validate_generation_config_mode
from kairyu.orchestration.execution import HttpExecutionBackend
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.prefix_index import PrefixIndex
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog

_EMBEDDING_BACKEND_FACTORIES: dict[str, Callable[..., EmbeddingBackend]] = {
    "fastembed": FastEmbedEmbeddingBackend,
    "mock": MockEmbeddingBackend,
}
_SERVICE_ACCOUNT_NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
logger = logging.getLogger("kairyu.deploy")


_DYNAMIC_REPLICA_MAX_KEEPALIVE = 64
_DYNAMIC_REPLICA_UNBOUNDED_KEEPALIVE = 8
_DYNAMIC_PROBE_CONCURRENCY = 16
_TEMPLATED_PROMPT_BACKENDS = frozenset(
    {"kairyu", "kairyu-proc", "mock", "vllm"}
)


@dataclass(frozen=True)
class _ResolvedChatPolicy:
    """One preflighted renderer policy shared by every request transport."""

    templates: dict[str, ChatTemplate]
    legacy_models: frozenset[str]


def _effective_tokenizer_source(entry) -> Path | None:
    """Return the local tokenizer source the concrete backend will consume.

    Native Kairyu uses an explicit tokenizer override before ``model_path``;
    direct vLLM follows its tokenizer-before-model contract.  Hub identifiers
    deliberately stay unresolved here: deployment preflight is offline and
    must not make an implicit network fetch merely to decide prompt framing.
    """

    source: object | None
    if entry.backend in {"kairyu", "kairyu-proc"}:
        source = entry.options.get("tokenizer")
        if source is None:
            source = entry.options.get("model_path")
    elif entry.backend == "vllm":
        source = entry.options.get("tokenizer")
        if source is None:
            source = entry.options.get("model")
    elif (
        entry.backend == "openai"
        and entry.options.get("allow_templated_chat_passthrough") is True
    ):
        source = entry.options.get("chat_tokenizer")
    else:
        return None
    if not isinstance(source, (str, os.PathLike)) or str(source) == "toy":
        return None
    path = Path(source)
    return path if path.exists() else None


def _entry_chat_metadata(
    entry,
    *,
    include_templates: bool,
) -> TokenizerChatMetadata | None:
    source = _effective_tokenizer_source(entry)
    if source is None:
        return None
    return load_tokenizer_chat_metadata(
        source,
        include_templates=include_templates,
    )


def _entry_preserves_templated_prompt(entry) -> bool:
    return entry.backend in _TEMPLATED_PROMPT_BACKENDS or (
        entry.backend == "openai"
        and entry.options.get("allow_templated_chat_passthrough") is True
    )


def _metadata_identity(metadata: TokenizerChatMetadata) -> tuple[object, ...]:
    return (
        tuple(sorted(metadata.templates.items())),
        tuple(sorted(metadata.special_tokens.items())),
    )


def _resolved_template_path(source: str, base_dir: Path | None) -> str:
    if not source.endswith(".jinja") or base_dir is None:
        return source
    path = Path(source)
    return str(base_dir / path) if not path.is_absolute() else source


def _resolve_chat_policy(
    spec: DeploymentSpec,
    base_dir: Path | None,
    *,
    emit_warnings: bool = True,
) -> _ResolvedChatPolicy:
    """Resolve every served model before constructing a GPU/backend resource.

    Explicit deploy templates win, then an effective local tokenizer's exact
    checkpoint template.  The old role concatenator is available only through
    ``legacy_chat_models`` (apart from the non-model ``mock`` test backend).
    Text-only remote/custom/orchestration models have no trustworthy local
    tokenizer owner and therefore fail closed instead of guessing.
    """

    model_entries: dict[str, tuple[object, ...]] = {
        name: (entry,) for name, entry in spec.engines.items()
    }
    model_entries.update(
        (name, tuple(pool.replicas)) for name, pool in spec.pools.items()
    )
    orchestration_names = set(spec.orchestrators)
    if spec.orchestrator is not None:
        orchestration_names.add("kairyu-auto")
    model_entries.update((name, ()) for name in orchestration_names)

    templates: dict[str, ChatTemplate] = {}
    legacy_models = set(spec.legacy_chat_models)
    for model_name, entries in model_entries.items():
        explicit = spec.chat_templates.get(model_name)
        if model_name in legacy_models:
            if emit_warnings:
                logger.warning(
                    "model %r explicitly enables the legacy role-concatenation chat "
                    "renderer; use a tokenizer-owned HF template for real models",
                    model_name,
                )
            # Legacy is a complete, model-scoped renderer policy. Do not parse
            # unused checkpoint chat metadata and accidentally let it override
            # the operator's explicit compatibility choice.
            continue
        if model_name in orchestration_names and explicit is None:
            # AUTO models consume the server's deterministic role-preserving
            # L2 JSON envelope, not tokenizer control tokens or legacy chat.
            continue
        try:
            metadata = tuple(
                item
                for item in (
                    _entry_chat_metadata(
                        entry,
                        include_templates=explicit is None,
                    )
                    for entry in entries
                )
                if item is not None
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"could not resolve chat metadata for model {model_name!r}: {error}"
            ) from error

        special_token_maps = {
            tuple(sorted(item.special_tokens.items())) for item in metadata
        }
        if len(special_token_maps) > 1:
            raise ValueError(
                f"model {model_name!r} resolves different tokenizer special-token "
                "maps across replicas; configure a coherent pool"
            )
        special_tokens = dict(next(iter(special_token_maps), ()))

        if explicit is not None:
            try:
                compiled = ChatTemplate.load(
                    _resolved_template_path(explicit, base_dir),
                    special_tokens=special_tokens,
                )
            except (OSError, ValueError, TemplateError) as error:
                raise ValueError(
                    f"could not load chat template for model {model_name!r}: {error}"
                ) from error
            if model_name in orchestration_names:
                raise ValueError(
                    f"orchestrated model {model_name!r} cannot safely consume a "
                    "pre-rendered chat prompt while deriving planner/worker prompts; "
                    "explicitly opt in through legacy_chat_models"
                )
            pool = spec.pools.get(model_name)
            if pool is not None and pool.discovery is not None:
                raise ValueError(
                    f"discovered pool {model_name!r} cannot prove that dynamic "
                    "replicas preserve pre-rendered chat prompts; explicitly opt "
                    "in through legacy_chat_models"
                )
            incompatible = sorted(
                {
                    entry.backend
                    for entry in entries
                    if not _entry_preserves_templated_prompt(entry)
                }
            )
            if incompatible:
                raise ValueError(
                    f"model {model_name!r} chat template produces a pre-rendered "
                    "prompt that configured backend(s) cannot preserve: "
                    f"{', '.join(incompatible)}; use a local tokenizer-owning "
                    "backend or explicitly opt in through legacy_chat_models"
                )
            if compiled.unresolved_special_token_variables:
                if len(metadata) != len(entries):
                    detail = (
                        "local tokenizer metadata is not available for every "
                        "configured backend"
                    )
                else:
                    detail = (
                        "the effective local tokenizer metadata does not define "
                        "their values"
                    )
                raise ValueError(
                    f"model {model_name!r} explicit chat template references "
                    "tokenizer-owned special-token variables "
                    f"{sorted(compiled.unresolved_special_token_variables)}, but "
                    f"{detail}; make the template self-contained or explicitly opt "
                    "in through legacy_chat_models"
                )
            if (
                compiled.referenced_special_token_variables
                and len(metadata) != len(entries)
            ):
                raise ValueError(
                    f"model {model_name!r} explicit chat template references "
                    "tokenizer-owned special-token variables "
                    f"{sorted(compiled.referenced_special_token_variables)}, but "
                    "local tokenizer metadata is not available for every configured "
                    "backend; materialize the effective tokenizer locally, make the "
                    "template self-contained, or explicitly opt in through "
                    "legacy_chat_models"
                )
            templates[model_name] = compiled
            continue

        # MockBackend is a deterministic protocol test double, not a trained
        # model with a prompt format.  Preserve builder test/smoke ergonomics
        # without weakening any real backend's fail-closed policy.
        if entries and all(entry.backend == "mock" for entry in entries):
            legacy_models.add(model_name)
            continue

        # Auto-load is safe only when every concrete replica exposes the same
        # local tokenizer metadata.  Picking one member of a heterogeneous or
        # remote pool would silently give requests a replica-dependent format.
        if entries and len(metadata) == len(entries):
            identities = {_metadata_identity(item) for item in metadata}
            if len(identities) != 1:
                raise ValueError(
                    f"model {model_name!r} resolves different chat templates "
                    "across replicas; configure a coherent pool"
                )
            resolved = metadata[0]
            if resolved.templates:
                if "default" not in resolved.templates:
                    raise ValueError(
                        f"model {model_name!r} tokenizer has multiple chat templates "
                        "but no default template"
                    )
                try:
                    templates[model_name] = ChatTemplate.from_tokenizer_metadata(
                        resolved.templates,
                        resolved.special_tokens,
                    )
                except (ValueError, TemplateError) as error:
                    raise ValueError(
                        "could not compile tokenizer chat template for model "
                        f"{model_name!r}: {error}"
                    ) from error
                if emit_warnings:
                    logger.info(
                        "auto-loaded tokenizer chat template for model %r from %s",
                        model_name,
                        ", ".join(str(path) for path in resolved.template_sources),
                    )
                continue

        if model_name in orchestration_names:
            raise ValueError(
                f"orchestrated model {model_name!r} has no supported chat policy: "
                "it cannot safely consume a pre-rendered chat prompt while "
                "deriving planner/worker prompts; "
                "explicitly opt in through legacy_chat_models"
            )
        pool = spec.pools.get(model_name)
        if pool is not None and pool.discovery is not None:
            raise ValueError(
                f"discovered pool {model_name!r} has no supported chat policy: it "
                "cannot prove that dynamic replicas preserve pre-rendered chat "
                "prompts; explicitly opt in through legacy_chat_models"
            )
        incompatible = sorted(
            {
                entry.backend
                for entry in entries
                if not _entry_preserves_templated_prompt(entry)
            }
        )
        if incompatible:
            raise ValueError(
                f"model {model_name!r} has no supported chat policy: configured "
                "backend(s) cannot preserve a tokenizer-owned pre-rendered chat "
                "prompt: "
                f"{', '.join(incompatible)}; explicitly opt in through "
                "legacy_chat_models"
            )
        raise ValueError(
            f"model {model_name!r} has no real chat template; add a checkpoint "
            "chat_template.jinja/tokenizer_config.json chat_template, configure "
            "chat_templates for this template-preserving local model, or explicitly "
            "opt in through legacy_chat_models"
        )

    return _ResolvedChatPolicy(
        templates=templates,
        legacy_models=frozenset(legacy_models),
    )


def _dynamic_replica_keepalive_limit(
    max_concurrency: int | None,
    replica_count: int,
) -> int:
    if max_concurrency is None:
        return _DYNAMIC_REPLICA_UNBOUNDED_KEEPALIVE
    live_replicas = max(1, replica_count)
    expected_per_replica = (max_concurrency + live_replicas - 1) // live_replicas
    return min(
        _DYNAMIC_REPLICA_MAX_KEEPALIVE,
        max(1, expected_per_replica),
    )


class _DynamicPoolHTTPClientFactory:
    """Cheap origin-local clients backed by one immutable TLS context.

    httpcore 1.0 keeps every origin's connections in one linear list inside an
    ``AsyncConnectionPool``. A single fleet-wide client therefore makes each
    request scan the whole fleet and makes idle cleanup quadratic. Conversely,
    constructing a default client per discovered replica reloads the CA bundle
    synchronously on the event loop. Share only the immutable ``SSLContext``:
    each replica still owns an origin-local transport and closes it when it
    leaves the pool, while client construction no longer reloads certificates.
    """

    def __init__(
        self,
        *,
        max_concurrency: int | None,
        replica_count: Callable[[], int],
    ) -> None:
        # EndpointSlice addresses are cluster-internal. Proxy environment
        # variables must not redirect either data or readiness traffic. Build
        # the context with trust_env enabled once so an explicitly supported
        # HTTPS discovery can still use SSL_CERT_FILE/SSL_CERT_DIR.
        self.ssl_context: SSLContext = httpx.create_ssl_context(
            verify=True,
            trust_env=True,
        )
        self._max_concurrency = max_concurrency
        self._replica_count = replica_count

    def create_replica_client(self) -> httpx.AsyncClient:
        # When configured, gateway admission bounds active connections. The
        # transport itself remains uncapped so concurrent long generations do
        # not queue behind a second limit.
        # Snapshot the live membership only when this origin is first used.
        # A bounded gateway retains its expected per-replica share (up to the
        # reviewed 64-request wave); an unbounded gateway uses a conservative
        # explicit fallback. The 30-second expiry is evaluated on later
        # activity for this origin, while replica removal closes the client.
        max_keepalive_connections = _dynamic_replica_keepalive_limit(
            self._max_concurrency,
            self._replica_count(),
        )
        return httpx.AsyncClient(
            verify=self.ssl_context,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=30.0,
            ),
        )

    def create_probe_client(self) -> httpx.AsyncClient:
        # Healthy replicas are not probed. Bootstrap/ejection work shares this
        # deliberately bounded control-plane pool and never shares data sockets.
        return httpx.AsyncClient(
            verify=self.ssl_context,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=_DYNAMIC_PROBE_CONCURRENCY,
                max_keepalive_connections=_DYNAMIC_PROBE_CONCURRENCY,
                keepalive_expiry=30.0,
            ),
        )


def _create_embedding_backend(section, base_dir: Path | None) -> EmbeddingBackend:
    factory = _EMBEDDING_BACKEND_FACTORIES.get(section.backend)
    if factory is None:
        known = ", ".join(sorted(_EMBEDDING_BACKEND_FACTORIES))
        raise ValueError(
            f"unknown embedding backend {section.backend!r}; "
            f"known backends: {known}"
        )
    options = section.model_dump(exclude={"backend"}, exclude_none=True)
    model_path = options.get("model_path")
    if isinstance(model_path, str):
        options["model_path"] = _resolve_path(model_path, base_dir)
    return factory(**options)


def _resolve_path(path: str, base_dir: Path | None) -> Path:
    resolved = Path(path)
    if base_dir is not None and not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved


def _resolve_kubernetes_namespace(configured: str | None) -> str:
    if configured is not None:
        return configured
    namespace = _SERVICE_ACCOUNT_NAMESPACE_PATH.read_text(encoding="utf-8").strip()
    if not namespace:
        raise ValueError("mounted Kubernetes service-account namespace is empty")
    return namespace


def _preflight_server(
    spec: DeploymentSpec,
) -> tuple[
    ServerSettings,
    TenantConfig | None,
    frozenset[str],
    frozenset[str],
    bytes,
]:
    settings = spec.server.to_server_settings()
    api_keys = settings.resolve_api_keys()
    admin_keys = settings.resolve_admin_keys()
    responses_compaction_key = settings.resolve_responses_compaction_key()
    section = spec.tenants
    if section is None:
        return settings, None, api_keys, admin_keys, responses_compaction_key
    tenant_config = TenantConfig.from_mapping(
        key_tenants=section.key_tenants,
        limits={
            tenant: TenantLimits(
                requests_per_minute=limits.requests_per_minute,
                tokens_per_minute=limits.tokens_per_minute,
                request_burst=limits.request_burst,
                token_burst=limits.token_burst,
                max_in_flight=limits.max_in_flight,
                interactive_priority=limits.interactive_priority,
                batch_priority=limits.batch_priority,
            )
            for tenant, limits in section.limits.items()
        },
        default_tenant=section.default_tenant,
        resolved_api_keys=api_keys,
    )
    return settings, tenant_config, api_keys, admin_keys, responses_compaction_key


def build_app_from_spec(
    spec: DeploymentSpec,
    base_dir: Path | None = None,
    *,
    generation_config_override: str | None = None,
) -> FastAPI:
    """Construct engines, pools, orchestrator, and the app with a prober lifespan."""
    if generation_config_override is not None:
        generation_config_override = validate_generation_config_mode(
            generation_config_override
        )
        spec = spec.model_copy(
            update={
                "engines": {
                    name: (
                        entry.model_copy(
                            update={
                                "options": {
                                    **entry.options,
                                    "generation_config": generation_config_override,
                                }
                            }
                        )
                        if entry.backend in {"kairyu", "kairyu-proc"}
                        else entry
                    )
                    for name, entry in spec.engines.items()
                },
                "pools": {
                    name: pool.model_copy(
                        update={
                            "replicas": tuple(
                                entry.model_copy(
                                    update={
                                        "options": {
                                            **entry.options,
                                            "generation_config": generation_config_override,
                                        }
                                    }
                                )
                                if entry.backend in {"kairyu", "kairyu-proc"}
                                else entry
                                for entry in pool.replicas
                            )
                        }
                    )
                    for name, pool in spec.pools.items()
                },
            }
        )
    native_generation_target = any(
        entry.backend in {"kairyu", "kairyu-proc"}
        for entry in spec.engines.values()
    ) or any(
        entry.backend in {"kairyu", "kairyu-proc"}
        for pool in spec.pools.values()
        for entry in pool.replicas
    )
    orchestrator_spec_cache: dict[Path, OrchestratorSpec] = {}

    def _read_orchestrator_spec(spec_path: str) -> OrchestratorSpec:
        path = Path(spec_path)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        loaded = orchestrator_spec_cache.get(path)
        if loaded is None:
            loaded = load_spec(path)
            orchestrator_spec_cache[path] = loaded
        return loaded

    if generation_config_override is not None and not native_generation_target:
        linked_sections = [
            section
            for section in (spec.orchestrator, *spec.orchestrators.values())
            if section is not None
        ]
        native_generation_target = any(
            worker.backend in {"kairyu", "kairyu-proc"}
            for section in linked_sections
            for worker in _read_orchestrator_spec(section.spec).workers
        )
        if not native_generation_target:
            raise ValueError(
                "--generation-config requires a local kairyu or kairyu-proc backend"
            )
    (
        server_settings,
        tenant_config,
        api_keys,
        admin_keys,
        responses_compaction_key,
    ) = _preflight_server(spec)
    batch_postgres_dsn: str | None = None
    if spec.batch is not None and spec.batch.store == "postgres":
        batch_postgres_dsn = os.environ.get(spec.batch.dsn_env)
        if not batch_postgres_dsn:
            raise ValueError(
                f"batch PostgreSQL DSN environment variable {spec.batch.dsn_env!r} is not set"
            )
    async_request_postgres_dsn: str | None = None
    if spec.async_requests is not None:
        async_request_postgres_dsn = os.environ.get(spec.async_requests.dsn_env)
        if not async_request_postgres_dsn:
            raise ValueError(
                "async request PostgreSQL DSN environment variable "
                f"{spec.async_requests.dsn_env!r} is not set"
            )
    # Chat policy is metadata-only and deliberately resolves before any
    # embedding/engine constructor can allocate model or GPU resources.
    chat_policy = _resolve_chat_policy(spec, base_dir)
    chat_templates = chat_policy.templates
    embedding_backends = {
        name: _create_embedding_backend(section, base_dir)
        for name, section in spec.embeddings.items()
    }
    engines: dict[str, EngineBackend] = {
        name: create_backend(entry.backend, **entry.options) for name, entry in spec.engines.items()
    }
    execution_backends = {
        name: HttpExecutionBackend(
            base_url=section.base_url,
            uds_path=section.uds_path,
            timeout_s=section.timeout_s,
            queue_wait_s=section.queue_wait_s,
        )
        for name, section in spec.executors.items()
    }
    # The builder owns these concrete engines and must finish any failure-prone
    # startup before FastAPI's lifespan yields (and uvicorn begins serving).
    # Pools own their replicas for shutdown, but keep the concrete replicas here
    # as startup targets because the pool itself intentionally has no startup
    # protocol. Dynamic replicas are remote resources discovered after startup.
    startup_resources: list[object] = [
        *engines.values(),
        *embedding_backends.values(),
    ]

    probers: list[HealthProber] = []
    reconcilers: list[tuple[PoolReconciler, float]] = []
    dynamic_http_client_factories: list[_DynamicPoolHTTPClientFactory] = []
    placement_log_paths: dict[str, Path] = {}
    claimed_log_paths: dict[Path, str] = {}
    for name, pool_spec in spec.pools.items():
        if pool_spec.placement_log_path is None:
            continue
        path = _resolve_path(pool_spec.placement_log_path, base_dir)
        canonical = path.resolve()
        claimed_by = claimed_log_paths.get(canonical)
        if claimed_by is not None:
            raise ValueError(
                "pool placement_log_path values must be unique; "
                f"{name!r} and {claimed_by!r} both resolve to {canonical}"
            )
        claimed_log_paths[canonical] = name
        placement_log_paths[name] = path

    for name, pool_spec in spec.pools.items():
        pool_log = (
            JsonlRouterLog(placement_log_paths[name]) if name in placement_log_paths else None
        )
        replicas = [create_backend(entry.backend, **entry.options) for entry in pool_spec.replicas]
        dynamic = pool_spec.discovery is not None
        pool = ReplicaPool(
            replicas,
            log=pool_log,
            unhealthy_after=pool_spec.unhealthy_after,
            queue_depth_threshold=pool_spec.queue_depth_threshold,
            prefix_index=PrefixIndex() if pool_spec.prefix_index else None,
            allow_empty=dynamic,
        )
        engines[name] = pool
        if not dynamic:
            startup_resources.extend(replicas)
            health_urls = [entry.resolved_health_url() for entry in pool_spec.replicas]
            if any(url is not None for url in health_urls):
                probers.append(
                    HealthProber(
                        name,
                        pool,
                        health_urls,
                        pool_spec.probe_interval_s,
                    )
                )
            continue

        discovery = pool_spec.discovery
        assert discovery is not None
        http_client_factory = _DynamicPoolHTTPClientFactory(
            max_concurrency=server_settings.max_concurrency,
            replica_count=lambda _pool=pool: _pool.replica_count,
        )
        dynamic_http_client_factories.append(http_client_factory)
        source = KubernetesEndpointSliceDiscovery(
            discovery.service,
            _resolve_kubernetes_namespace(discovery.namespace),
            discovery.port,
            model=discovery.model,
            api_key_env=discovery.api_key_env,
            scheme=discovery.scheme,
            address_family_preference=discovery.address_family_preference,
        )

        event_sink = None
        if pool_log is not None:

            def event_sink(
                actions: dict[str, object],
                *,
                _pool: ReplicaPool = pool,
                _log: JsonlRouterLog = pool_log,
            ) -> None:
                raw_replica_ids = actions.get("replica_ids")
                replica_ids = tuple(
                    str(replica_id)
                    for replica_id in (
                        raw_replica_ids if raw_replica_ids is not None else _pool.replica_ids
                    )
                )
                raw_healthy_ids = actions.get("healthy_ids")
                healthy_ids = tuple(
                    str(replica_id)
                    for replica_id in (
                        raw_healthy_ids
                        if raw_healthy_ids is not None
                        else (
                            tuple(
                                current_id
                                for current_id, healthy in (_pool.healthy_by_id().items())
                                if healthy
                            )
                        )
                    )
                )
                raw_eligible_ids = actions.get("eligible_ids")
                eligible_ids = tuple(
                    str(replica_id)
                    for replica_id in (
                        raw_eligible_ids if raw_eligible_ids is not None else _pool.eligible_ids
                    )
                )
                raw_generations = actions.get("generation_by_id", {})
                generation_by_id = (
                    {
                        str(replica_id): str(generation)
                        for replica_id, generation in raw_generations.items()
                    }
                    if isinstance(raw_generations, dict)
                    else {}
                )
                _log.record_membership(
                    actions,
                    replica_ids=replica_ids,
                    healthy_ids=healthy_ids,
                    eligible_ids=eligible_ids,
                    generation_by_id=generation_by_id,
                )

        reconciler = PoolReconciler(
            pool,
            source,
            factory=partial(
                openai_replica_factory,
                client_factory=http_client_factory.create_replica_client,
            ),
            default_model=discovery.model or name,
            event_sink=event_sink,
        )
        reconcilers.append((reconciler, discovery.poll_interval_s))
        # Dynamic candidates carry their readiness URL on the pool entry.
        # An empty mapping deliberately avoids any bootstrap replica.
        probers.append(
            HealthProber(
                name,
                pool,
                {},
                pool_spec.probe_interval_s,
                client=http_client_factory.create_probe_client(),
                max_concurrency=_DYNAMIC_PROBE_CONCURRENCY,
            )
        )

    def _load_orchestrator(spec_path: str) -> Orchestrator:
        orchestrator_spec = _read_orchestrator_spec(spec_path)
        if generation_config_override is not None:
            workers = []
            for worker in orchestrator_spec.workers:
                if worker.backend in {"kairyu", "kairyu-proc"}:
                    worker = worker.model_copy(
                        update={
                            "options": {
                                **worker.options,
                                "generation_config": generation_config_override,
                            }
                        }
                    )
                workers.append(worker)
            orchestrator_spec = orchestrator_spec.model_copy(
                update={"workers": tuple(workers)}
            )
        return build_orchestrator(
            orchestrator_spec,
            engine_refs=engines,
            executor_refs=execution_backends,
        )

    orchestrator: Orchestrator | None = None
    if spec.orchestrator is not None:
        orchestrator = _load_orchestrator(spec.orchestrator.spec)
    orchestrators: dict[str, Orchestrator] = {
        name: _load_orchestrator(section.spec) for name, section in spec.orchestrators.items()
    }
    if orchestrator is not None:
        startup_resources.append(orchestrator)
    startup_resources.extend(orchestrators.values())

    all_orchestrators = dict(orchestrators)
    if orchestrator is not None:
        all_orchestrators.setdefault("kairyu-auto", orchestrator)
    public_models = spec.public_models
    served_engines = (
        engines
        if public_models is None
        else {name: engine for name, engine in engines.items() if name in public_models}
    )
    served_orchestrators = (
        all_orchestrators
        if public_models is None
        else {
            name: selected
            for name, selected in all_orchestrators.items()
            if name in public_models
        }
    )
    served_embeddings = (
        embedding_backends
        if public_models is None
        else {
            name: backend
            for name, backend in embedding_backends.items()
            if name in public_models
        }
    )
    served_chat_templates = {
        name: template
        for name, template in chat_templates.items()
        if public_models is None or name in public_models
    }
    served_legacy_chat_models = frozenset(
        name
        for name in chat_policy.legacy_models
        if public_models is None or name in public_models
    )

    workers: list[object] = []  # filled after create_app (workers need app metrics)
    owned_stores: list[object] = []

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks: list[asyncio.Task] = []
        try:
            for resource in {id(resource): resource for resource in startup_resources}.values():
                startup = getattr(resource, "startup", None)
                if callable(startup):
                    await startup()
            tasks = [
                asyncio.create_task(reconciler.run(interval_s))
                for reconciler, interval_s in reconcilers
            ]
            tasks += [asyncio.create_task(prober.run()) for prober in probers]
            tasks += [asyncio.create_task(worker.run()) for worker in workers]
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                # a task that already crashed re-raises its stored exception on
                # await; swallow it (and the CancelledError) so one dead task
                # cannot skip the remaining awaits AND the engine shutdowns (M7)
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            shutdown_errors: list[Exception] = []
            for store in owned_stores:
                close = getattr(store, "close", None)
                if callable(close):
                    try:
                        await asyncio.to_thread(close)
                    except Exception as error:
                        shutdown_errors.append(error)
            resources = list(engines.values())
            resources.extend(execution_backends.values())
            resources.extend(embedding_backends.values())
            if orchestrator is not None:
                resources.append(orchestrator)
            resources.extend(orchestrators.values())
            # create_app owns the usage ledger in an outer lifespan wrapper,
            # so it closes only after workers stop and this shutdown completes
            # or raises an ExceptionGroup.
            try:
                await shutdown_all(resources, "application")
            except Exception as error:
                shutdown_errors.append(error)
            if len(shutdown_errors) == 1:
                raise shutdown_errors[0]
            if shutdown_errors:
                raise ExceptionGroup(
                    "application resource shutdown failed",
                    shutdown_errors,
                )

    app = create_app(
        engines=served_engines,
        orchestrators=served_orchestrators,
        settings=server_settings,
        lifespan=lifespan,
        chat_templates=served_chat_templates,
        tenant_config=tenant_config,
        embedding_backends=served_embeddings,
        resolved_api_keys=api_keys,
        resolved_admin_keys=admin_keys,
        resolved_responses_compaction_key=responses_compaction_key,
        price_sheet=spec.pricing,
        legacy_chat_models=served_legacy_chat_models,
        orchestration_chat_models=set(served_orchestrators),
        runtime_engines=engines,
        runtime_embedding_backends=embedding_backends,
        runtime_orchestrators=all_orchestrators,
        async_request_body_limit=(
            spec.async_requests.max_body_bytes
            if spec.async_requests is not None
            else None
        ),
    )
    app.state.deployment_spec = spec
    app.state.probers = tuple(probers)
    app.state.reconcilers = tuple(reconciler for reconciler, _interval_s in reconcilers)
    app.state.dynamic_pool_http_client_factories = tuple(dynamic_http_client_factories)

    if spec.batch is not None:
        from kairyu.entrypoints.server.batch_routes import add_batch_routes

        if spec.batch.store == "postgres":
            from kairyu.batch.postgres_store import PostgresBatchStore

            assert batch_postgres_dsn is not None
            spool_dir = (
                _resolve_path(spec.batch.spool_dir, base_dir)
                if spec.batch.spool_dir is not None
                else None
            )
            store = PostgresBatchStore(
                batch_postgres_dsn,
                store_id=spec.batch.store_id,
                spool_dir=spool_dir,
            )
        else:
            assert spec.batch.data_dir is not None
            store = BatchStore(spec.batch.data_dir)
            store.recover_orphans()
        owned_stores.append(store)
        worker = BatchWorker(
            store,
            engines,
            max_concurrency=spec.batch.max_concurrency,
            claim_poll_interval_s=spec.batch.poll_interval_s,
            claim_lease_seconds=spec.batch.lease_seconds,
            metrics=app.state.metrics,
            chat_templates=chat_templates,  # batch and HTTP must render identically
            legacy_chat_models=chat_policy.legacy_models,
            usage_ledger=getattr(app.state, "usage_ledger", None),
            tenant_limiter=getattr(app.state, "tenant_limiter", None),
            tenant_config=tenant_config,
            admission_controller=app.state.slo_admission,
        )
        workers.append(worker)
        add_batch_routes(app, store, worker)
        app.state.batch_store = store
        app.state.batch_worker = worker
    if spec.async_requests is not None:
        from kairyu.async_requests.postgres_store import PostgresRequestStore
        from kairyu.async_requests.worker import AsyncRequestWorker
        from kairyu.entrypoints.server.async_request_routes import (
            add_async_request_routes,
        )

        assert async_request_postgres_dsn is not None
        request_store = PostgresRequestStore(
            async_request_postgres_dsn,
            store_id=spec.async_requests.store_id,
            eager_connect=False,
            max_records_per_owner=spec.async_requests.max_records_per_tenant,
        )
        owned_stores.append(request_store)
        startup_resources.append(request_store)
        request_worker = AsyncRequestWorker(
            request_store,
            served_engines,
            max_concurrency=spec.async_requests.max_concurrency,
            poll_interval_s=spec.async_requests.poll_interval_s,
            lease_seconds=spec.async_requests.lease_seconds,
            metrics=app.state.metrics,
            chat_templates=served_chat_templates,
            legacy_chat_models=served_legacy_chat_models,
            usage_ledger=getattr(app.state, "usage_ledger", None),
            tenant_limiter=getattr(app.state, "tenant_limiter", None),
            tenant_config=tenant_config,
            admission_controller=app.state.slo_admission,
        )
        workers.append(request_worker)
        add_async_request_routes(app, request_store, request_worker)
        app.state.async_request_store = request_store
        app.state.async_request_worker = request_worker
    return app


def build_app_from_config(path: str | Path) -> FastAPI:
    """Load a DeploymentSpec YAML file and build the app (used by `kairyu serve`)."""
    config_path = Path(path)
    spec = load_deployment_spec(config_path)
    return build_app_from_spec(spec, base_dir=config_path.parent)

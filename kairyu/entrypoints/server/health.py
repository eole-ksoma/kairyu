"""Liveness, readiness, and metrics endpoints (design m7 D4)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from kairyu.engine.backend import EngineBackend, EngineReadiness
from kairyu.engine.core.attention_selector import (
    AttentionBackendDecision,
    attention_backend_identity,
    select_backend_decision,
)
from kairyu.engine.core.hw_profile import probe
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.models.generation import GenerationDefaults
from kairyu.orchestration.replica import ReplicaPool

# type(engine).__name__ -> engine-registry backend name. Kept local (not a class
# attr) so this endpoint needs no engine-class change and stays robust to tests
# that construct backends directly.
_ENGINE_LABELS = {
    "MockBackend": "mock",
    "KairyuBackend": "kairyu",
    "OpenAICompatBackend": "openai",
    "VLLMBackend": "vllm",
    "ZmqEngineBackend": "kairyu-proc",
    "ReplicaPool": "replica-pool",
}
# Engine backends that run attention locally in-process (so the resolved
# attention backend applies to them); remote/echo engines report null.
_LOCAL_ATTENTION_BACKENDS = frozenset({"kairyu", "kairyu-proc"})
_EP_COLLECTIVE_TRANSPORT_FIELDS = frozenset(
    {
        "selected_backend",
        "fallback_backend",
        "direct_nccl_active",
        "direct_nccl_library",
        "direct_nccl_version",
        "selection_reason",
    }
)


def _expert_parallel_metadata(value: object) -> dict[str, object] | None:
    """Return a complete, JSON-safe EP topology or fail closed."""

    if not isinstance(value, Mapping) or value.get("parallelism") != "expert_parallel":
        return None
    expert_parallel_size = value.get("expert_parallel_size")
    output_parallel_size = value.get("attention_output_parallel_size")
    pipeline_depth = value.get("pipeline_depth")
    decode_mode = value.get("decode_mode")
    if (
        type(expert_parallel_size) is not int
        or expert_parallel_size not in {2, 4, 8}
        or type(output_parallel_size) is not int
        or output_parallel_size < 1
        or type(pipeline_depth) is not int
        or pipeline_depth < 1
        or decode_mode not in {"eager", "cuda_graph"}
        or value.get("kv_cache_dtype")
        not in {"bfloat16", "checkpoint-native-hca-csa"}
    ):
        return None
    execution_mode = value.get("execution_mode")
    common = {
        "parallelism": "expert_parallel",
        "expert_parallel_size": expert_parallel_size,
        "attention_placement": value.get("attention_placement"),
        "attention_output_placement": value.get("attention_output_placement"),
        "attention_output_parallel_size": output_parallel_size,
        "attention_output_partial_dtype": value.get(
            "attention_output_partial_dtype"
        ),
        "execution_mode": execution_mode,
        "pipeline_depth": pipeline_depth,
        "decode_mode": decode_mode,
        "kv_cache_dtype": value.get("kv_cache_dtype"),
    }
    if execution_mode == "replicated-attention-correctness":
        if (
            expert_parallel_size not in {2, 4}
            or output_parallel_size != expert_parallel_size
            or common["attention_placement"] != "replicated"
            or common["attention_output_placement"] != "row_parallel"
            or common["attention_output_partial_dtype"] != "bfloat16"
            or pipeline_depth != 1
            or decode_mode != "eager"
        ):
            return None
        return common
    if execution_mode not in {
        "request-owned-attention-dp",
        "deepseek-v4-native-attention-dp",
    }:
        return None
    transport = value.get("moe_collective_transport")
    if not isinstance(transport, Mapping) or set(transport) != (
        _EP_COLLECTIVE_TRANSPORT_FIELDS
    ):
        return None
    direct_active = transport.get("direct_nccl_active")
    selected_backend = transport.get("selected_backend")
    fallback_backend = transport.get("fallback_backend")
    selection_reason = transport.get("selection_reason")
    direct_library = transport.get("direct_nccl_library")
    direct_version = transport.get("direct_nccl_version")
    if (
        type(direct_active) is not bool
        or fallback_backend != "torch.distributed:nccl"
        or not isinstance(selection_reason, str)
        or not selection_reason
        or (
            direct_active
            and (
                selected_backend != "direct_nccl_ctypes"
                or not isinstance(direct_library, str)
                or not direct_library
                or not isinstance(direct_version, str)
                or not direct_version
            )
        )
        or (
            not direct_active
            and (
                selected_backend not in {fallback_backend, "unavailable"}
                or direct_library is not None
                or direct_version is not None
            )
        )
    ):
        return None
    cuda_graph_decode = value.get("cuda_graph_decode")
    graph_buckets = value.get("cuda_graph_buckets")
    graph_captures = value.get("cuda_graph_captures")
    graph_replays = value.get("cuda_graph_replays")
    graph_fallbacks = value.get("cuda_graph_eager_fallbacks")
    prefill_scratch = value.get("attention_dp_prefill_scratch_pages")
    decode_scratch = value.get("attention_dp_decode_scratch_pages")
    graph_scratch = value.get("attention_dp_graph_scratch_pages")
    total_scratch = value.get("attention_dp_scratch_pages")
    deepseek_v4 = execution_mode == "deepseek-v4-native-attention-dp"
    expected_dispatcher = (
        "fixed-nccl-all-to-all-packed-fp4"
        if deepseek_v4
        else "nvfp4_allgather_reduce_scatter"
    )
    if (
        (deepseek_v4 and expert_parallel_size not in {2, 4, 8})
        or (not deepseek_v4 and expert_parallel_size != 4)
        or output_parallel_size != 1
        or common["attention_placement"] != "request_owned_data_parallel"
        or common["attention_output_placement"] != "replicated"
        or common["attention_output_partial_dtype"] is not None
        or value.get("attention_data_parallel_size") != expert_parallel_size
        or value.get("attention_tensor_parallel_size") != 1
        or value.get("moe_dispatcher") != expected_dispatcher
        or value.get("sampling_ownership") != "request_owner"
        or value.get("kv_cache_ownership") != "request_owner"
        or (
            deepseek_v4
            and (
                common["kv_cache_dtype"] != "checkpoint-native-hca-csa"
                or decode_mode != "eager"
            )
        )
        or (not deepseek_v4 and common["kv_cache_dtype"] != "bfloat16")
        or type(cuda_graph_decode) is not bool
        or cuda_graph_decode != (decode_mode == "cuda_graph")
        or not isinstance(graph_buckets, (tuple, list))
        or any(type(bucket) is not int or bucket < 1 for bucket in graph_buckets)
        or tuple(graph_buckets) != tuple(sorted(set(graph_buckets)))
        or (cuda_graph_decode and not graph_buckets)
        or (not cuda_graph_decode and bool(graph_buckets))
        or any(
            type(count) is not int or count < 0
            for count in (graph_captures, graph_replays, graph_fallbacks)
        )
        or prefill_scratch != 2
        or type(decode_scratch) is not int
        or decode_scratch < 1
        or graph_scratch != int(cuda_graph_decode)
        or total_scratch
        != prefill_scratch + decode_scratch + graph_scratch
    ):
        return None
    return {
        **common,
        "attention_data_parallel_size": expert_parallel_size,
        "attention_tensor_parallel_size": 1,
        "moe_dispatcher": expected_dispatcher,
        "sampling_ownership": "request_owner",
        "kv_cache_ownership": "request_owner",
        "cuda_graph_decode": cuda_graph_decode,
        "cuda_graph_buckets": list(graph_buckets),
        "cuda_graph_captures": graph_captures,
        "cuda_graph_replays": graph_replays,
        "cuda_graph_eager_fallbacks": graph_fallbacks,
        "attention_dp_prefill_scratch_pages": prefill_scratch,
        "attention_dp_decode_scratch_pages": decode_scratch,
        "attention_dp_graph_scratch_pages": graph_scratch,
        "attention_dp_scratch_pages": total_scratch,
        "moe_collective_transport": dict(transport),
    }


def _generation_metadata(value: object) -> dict[str, object] | None:
    """Return one complete, validated generation-default audit record."""

    if not isinstance(value, Mapping):
        return None
    mode = value.get("generation_config")
    source = value.get("generation_config_source")
    sampling = value.get("generation_defaults")
    if not isinstance(sampling, Mapping) or set(sampling) != {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
    }:
        return None
    try:
        defaults = GenerationDefaults(
            mode=mode,
            source=source,
            **sampling,
        )
    except (OverflowError, TypeError, ValueError):
        return None
    return {
        "generation_config": defaults.mode,
        "generation_config_source": defaults.source,
        "generation_defaults": defaults.sampling_defaults(),
    }


def add_health_routes(
    app: FastAPI,
    engines: Mapping[str, EngineBackend],
    metrics: ServerMetrics | None,
    admin_keys: Iterable[str] = (),
    *,
    embedding_backends: Mapping[str, object] | None = None,
    orchestrators: Mapping[str, object] | None = None,
) -> None:
    admin_key_set = frozenset(admin_keys)
    embedding_resources = dict(embedding_backends or {})
    orchestrator_resources = dict(orchestrators or {})

    def _forbidden_if_not_admin(request: Request) -> JSONResponse | None:
        # when admin keys are configured, /admin/* state changes require one, so
        # an ordinary data-plane key cannot drain the node (S5). With no admin
        # keys set, behavior is unchanged (auth-gated when api keys exist).
        if not admin_key_set:
            return None
        if request.scope.get("state", {}).get("is_admin"):
            return None
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": "admin privilege required",
                    "type": "invalid_request_error",
                    "code": "admin_required",
                }
            },
        )

    def _engine_readiness() -> dict[str, EngineReadiness]:
        """Per-engine self-report; engines without the optional hook are skipped."""
        reported = {}
        for name, engine in engines.items():
            probe_readiness = getattr(engine, "readiness", None)
            if probe_readiness is None:
                continue
            try:
                reported[name] = probe_readiness()
            except Exception as error:  # introspection must never 500
                # class name only: this reaches an unauthenticated endpoint and
                # an exception message can carry a path or an upstream URL
                reported[name] = EngineReadiness(
                    False, f"readiness check raised {type(error).__name__}"
                )
        return reported

    def _embedding_readiness() -> dict[str, EngineReadiness]:
        reported = {}
        for name, backend in embedding_resources.items():
            probe_readiness = getattr(backend, "readiness", None)
            if probe_readiness is None:
                continue
            try:
                reported[name] = probe_readiness()
            except Exception as error:
                reported[name] = EngineReadiness(
                    False,
                    f"readiness check raised {type(error).__name__}",
                )
        return reported

    @app.get("/health")
    async def health():
        """Liveness. A FATAL engine fault answers 503 so an orchestrator replaces
        the process — nothing in-process can undo dead tensor-parallel ranks, and
        without this the container stays up forever serving nothing."""
        fatal = {
            name: status.detail for name, status in _engine_readiness().items() if status.fatal
        }
        fatal_embeddings = {
            name: status.detail
            for name, status in _embedding_readiness().items()
            if status.fatal
        }
        if fatal or fatal_embeddings:
            content: dict[str, object] = {"status": "fatal"}
            if fatal:
                content["engines"] = fatal
            if fatal_embeddings:
                content["embeddings"] = fatal_embeddings
            return JSONResponse(status_code=503, content=content)
        return {"status": "ok"}

    @app.post("/admin/drain")
    async def drain_node(request: Request):
        """Node-level drain (m10a A5): flips /readyz to 503; gateway-side pool
        drains go through PoolReconciler membership instead.

        NOT in the auth exempt list; additionally requires an ADMIN key when
        admin keys are configured (S5). Keyless deployments are the node-to-node
        trusted-mesh mode (m7 D5) by explicit choice."""
        denied = _forbidden_if_not_admin(request)
        if denied is not None:
            return denied
        app.state.draining = True
        return {"status": "draining"}

    @app.post("/admin/undrain")
    async def undrain_node(request: Request):
        """Clear the drain flag so the node reports ready again (S5) — without
        this a drained node could only recover via a process restart."""
        denied = _forbidden_if_not_admin(request)
        if denied is not None:
            return denied
        app.state.draining = False
        return {"status": "ready"}

    @app.get("/readyz")
    async def readyz():
        # m10a A5: a drained node reports unready so the prober/load-balancer
        # stops sending NEW work; in-flight requests keep completing.
        if getattr(app.state, "draining", False):
            return JSONResponse(status_code=503, content={"status": "draining"})
        # Engines are constructed by the time the app exists; pools additionally
        # need >=1 validated, non-ejected replica or every request would fail.
        # Declared remote readiness URLs therefore remain false here until the
        # startup prober succeeds; backend traffic is never implicit validation.
        degraded = {
            name: {
                "healthy": list(engine.healthy),
                "draining": list(engine.draining_by_id().values()),
                "eligible_ids": list(engine.eligible_ids),
            }
            for name, engine in engines.items()
            if isinstance(engine, ReplicaPool) and not engine.eligible_ids
        }
        if degraded:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unready",
                    "pools": degraded,
                },
            )
        # Local engines get a say too. "Constructed" is not "able to serve": a
        # KairyuBackend whose step loop or spawned TP ranks have died stays
        # constructed, so without this the endpoint reports ready while the node
        # cannot emit a token — which is exactly how a benchmark ran against a
        # dead 8-GPU deployment for 14 minutes before anyone noticed.
        failed = {
            name: status.detail for name, status in _engine_readiness().items() if not status.ready
        }
        failed_embeddings = {
            name: status.detail
            for name, status in _embedding_readiness().items()
            if not status.ready
        }
        if failed or failed_embeddings:
            content: dict[str, object] = {"status": "unready"}
            if failed:
                content["engines"] = failed
            if failed_embeddings:
                content["embeddings"] = failed_embeddings
            return JSONResponse(status_code=503, content=content)
        return {"status": "ready"}

    @app.get("/backends")
    async def backends() -> dict:
        """Report the resolved attention backend, library versions, and the
        per-engine backend map (m13). Open endpoint (see middleware _OPEN_PATHS);
        disclosure level matches the existing public /readyz and /metrics.

        attention backend is a process-level decision (env override or probed hw
        profile — deterministic and shared), resolved once here with
        ``select_backend_decision(probe())`` rather than deep-walking each
        engine.

        Topology note: a pure gateway runs NO local attention — it forwards to
        replicas — so its own probe reports torch and its engines are all pools.
        To still surface the real kernel, each ReplicaPool engine asks ONE replica
        for its /backends and adopts that replica's attention backend (cached,
        best-effort, null on unreachable). `role` distinguishes the two cases."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            profile = probe()
            kernel_tier = profile.kernel_tier
        except Exception:  # introspection must never 500
            profile, kernel_tier = None, "torch"
        try:
            decision = select_backend_decision(profile)
        except Exception as error:  # invalid env must not break introspection
            override = os.environ.get("KAIRYU_ATTENTION_BACKEND")
            decision = AttentionBackendDecision(
                requested=override or "auto",
                resolved="unavailable",
                source="env" if override else "hw_profile",
                components={},
                rationale=(f"attention selection is unavailable after {type(error).__name__}"),
                architecture={
                    "arch": profile.arch if profile is not None else "unknown",
                    "device_name": (profile.device_name if profile is not None else None),
                    "sm": profile.sm if profile is not None else None,
                    "kernel_tier": kernel_tier,
                },
            )
        configured_decision = decision

        # A real local engine retains the backend decision that was actually
        # constructed, including an ``auto`` dependency fallback. Prefer that
        # over re-evaluating env/profile state at inspection time. Engines with
        # no native model (mocks, remote gateways) have no such decision and
        # keep the configured process-level report above.
        actual_decisions = {
            name: actual
            for name, engine in engines.items()
            if isinstance(
                actual := getattr(engine, "attention_backend_decision", None),
                AttentionBackendDecision,
            )
        }
        if actual_decisions:
            identities = {
                attention_backend_identity(actual) for actual in actual_decisions.values()
            }
            if len(identities) == 1:
                decision = next(iter(actual_decisions.values()))
            else:
                decision = AttentionBackendDecision(
                    requested="mixed",
                    resolved="mixed",
                    source="engine",
                    components={},
                    rationale=(
                        "local engines constructed different attention "
                        "backend decisions; inspect the per-engine entries"
                    ),
                    architecture={
                        name: dict(actual.architecture) for name, actual in actual_decisions.items()
                    },
                )
        attention = decision.resolved

        def _kernel_tier_for(
            architecture: Mapping[str, object],
            fallback: str,
        ) -> str:
            direct = architecture.get("kernel_tier")
            if isinstance(direct, str) and direct:
                return direct
            nested = {
                tier
                for value in architecture.values()
                if isinstance(value, Mapping)
                and isinstance((tier := value.get("kernel_tier")), str)
                and tier
            }
            if len(nested) == 1:
                return nested.pop()
            return "mixed" if nested else fallback

        def _pkg_version(*names: str) -> str | None:
            # flashinfer ships under a few distribution names (flashinfer-python;
            # the AOT flashinfer-jit-cache) — try each so the version isn't null.
            for name in names:
                try:
                    return version(name)
                except PackageNotFoundError:
                    continue
            return None

        def _versions_for(
            components: Mapping[str, str],
        ) -> dict[str, str | None]:
            out: dict[str, str | None] = {"torch": _pkg_version("torch")}
            used = frozenset(components.values())
            if "flashinfer" in used:
                out["flashinfer"] = _pkg_version("flashinfer", "flashinfer-python")
            if "flashattention4" in used:
                out["flash-attn-4"] = _pkg_version("flash-attn-4")
            if "flashattention3" in used:
                out["flash_attn_3"] = _pkg_version("flash_attn_3")
            return out

        engine_list = []
        for name, engine in engines.items():
            label = _ENGINE_LABELS.get(type(engine).__name__, type(engine).__name__)
            declared_metadata = (
                engine.declared_metadata_snapshot()
                if isinstance(engine, ReplicaPool)
                else {}
            )
            actual = getattr(engine, "attention_backend_decision", None)
            engine_decision = (
                actual if isinstance(actual, AttentionBackendDecision) else configured_decision
            )
            entry: dict = {
                "model": name,
                "engine_backend": label,
                "attention_backend": (
                    engine_decision.resolved if label in _LOCAL_ATTENTION_BACKENDS else None
                ),
                "attention_components": (
                    dict(engine_decision.components) if label in _LOCAL_ATTENTION_BACKENDS else None
                ),
            }
            if label in _LOCAL_ATTENTION_BACKENDS:
                entry.update(
                    {
                        "requested_attention_backend": engine_decision.requested,
                        "selection_rationale": engine_decision.rationale,
                        "source": engine_decision.source,
                        "architecture": dict(engine_decision.architecture),
                        "decision_status": (
                            "actual"
                            if isinstance(actual, AttentionBackendDecision)
                            else "configured"
                        ),
                        "versions": _versions_for(engine_decision.components),
                    }
                )
            metadata_snapshot = getattr(
                engine,
                "parallelism_metadata_snapshot",
                None,
            )
            raw_parallelism_metadata = (
                metadata_snapshot()
                if callable(metadata_snapshot)
                else getattr(engine, "parallelism_metadata", None)
            )
            ep_metadata = _expert_parallel_metadata(raw_parallelism_metadata)
            if ep_metadata is not None:
                entry.update(ep_metadata)
            else:
                declared_ep_size = getattr(
                    engine,
                    "expert_parallel_size",
                    declared_metadata.get("expert_parallel_size"),
                )
                if type(declared_ep_size) is int and declared_ep_size > 1:
                    # Never relabel a known EP engine as TP1 when its detailed
                    # runtime topology is missing or malformed.
                    entry.update(
                        {
                            "parallelism": "expert_parallel",
                            "expert_parallel_size": declared_ep_size,
                            "parallelism_metadata_status": "invalid",
                        }
                    )
                else:
                    tensor_parallel_size = getattr(
                        engine,
                        "tensor_parallel_size",
                        declared_metadata.get("tensor_parallel_size"),
                    )
                    if (
                        type(tensor_parallel_size) is int
                        and tensor_parallel_size >= 1
                    ):
                        entry["tensor_parallel_size"] = tensor_parallel_size
            requested_kv_dtype = getattr(
                engine,
                "kv_cache_dtype_requested",
                None,
            )
            resolved_kv_dtype = getattr(
                engine,
                "kv_cache_dtype_resolved",
                None,
            )
            if isinstance(requested_kv_dtype, str) and requested_kv_dtype:
                entry["requested_kv_cache_dtype"] = requested_kv_dtype
            if isinstance(resolved_kv_dtype, str) and resolved_kv_dtype:
                entry["kv_cache_dtype"] = resolved_kv_dtype
            for field in (
                "model_revision",
                "max_model_len",
                "quantization_format",
                "cache_descriptor",
                "execution_mode",
                "attention_data_parallel_size",
                "mtp_enabled",
                "dspark_enabled",
                "container_image_digest",
            ):
                value = getattr(engine, field, declared_metadata.get(field))
                if value is not None:
                    entry[field] = value
            if "container_image_digest" not in entry:
                image_digest = os.environ.get("KAIRYU_CONTAINER_IMAGE_DIGEST")
                if image_digest:
                    entry["container_image_digest"] = image_digest
            generation_defaults = getattr(
                engine,
                "generation_defaults",
                None,
            )
            if isinstance(generation_defaults, GenerationDefaults):
                entry.update(
                    {
                        "generation_config": generation_defaults.mode,
                        "generation_config_source": generation_defaults.source,
                        "generation_defaults": (
                            generation_defaults.sampling_defaults()
                        ),
                    }
                )
            dram_enabled = getattr(engine, "dram_kv_tier_enabled", None)
            if type(dram_enabled) is bool:
                entry["dram_kv_tier_enabled"] = dram_enabled
                if dram_enabled:
                    dram_capacity = getattr(
                        engine, "dram_kv_tier_capacity_pages", None
                    )
                    dram_profile_sha = getattr(
                        engine, "dram_kv_tier_profile_sha256", None
                    )
                    dram_threshold = getattr(
                        engine, "dram_kv_tier_min_restore_tokens", None
                    )
                    if type(dram_capacity) is int and dram_capacity > 0:
                        entry["dram_kv_tier_capacity_pages"] = dram_capacity
                    if isinstance(dram_profile_sha, str) and dram_profile_sha:
                        entry["dram_kv_tier_profile_sha256"] = dram_profile_sha
                    if type(dram_threshold) is int and dram_threshold > 0:
                        entry["dram_kv_tier_min_restore_tokens"] = dram_threshold
            if isinstance(engine, ReplicaPool):
                replica = await engine.probe_backends()
                if replica:  # adopt the replica's (real) attention backend
                    entry["attention_backend"] = replica.get("attention_backend")
                    replica_components = replica.get("attention_components")
                    entry["via_replica"] = {
                        "attention_backend": replica.get("attention_backend"),
                        "requested_attention_backend": replica.get("requested_attention_backend"),
                        "kernel_tier": replica.get("kernel_tier"),
                        "versions": replica.get("versions"),
                        "selection_rationale": replica.get("selection_rationale"),
                        "source": replica.get("source"),
                        "architecture": replica.get("architecture"),
                    }
                    if isinstance(replica_components, dict):
                        entry["attention_components"] = dict(replica_components)
                        entry["via_replica"]["attention_components"] = dict(replica_components)
                    replica_sizes = {
                        item.get("tensor_parallel_size")
                        for item in replica.get("engines", ())
                        if (
                            isinstance(item, dict)
                            and item.get("parallelism") != "expert_parallel"
                            and type(item.get("tensor_parallel_size")) is int
                        )
                    }
                    if len(replica_sizes) == 1:
                        replica_size = replica_sizes.pop()
                        entry["tensor_parallel_size"] = replica_size
                        entry["via_replica"]["tensor_parallel_size"] = replica_size
                    replica_engines = [
                        item
                        for item in replica.get("engines", ())
                        if isinstance(item, dict)
                    ]
                    replica_ep_metadata = [
                        _expert_parallel_metadata(item)
                        for item in replica_engines
                    ]
                    if (
                        replica_ep_metadata
                        and replica_ep_metadata[0] is not None
                        and all(
                            metadata == replica_ep_metadata[0]
                            for metadata in replica_ep_metadata
                        )
                    ):
                        entry.pop("tensor_parallel_size", None)
                        entry["via_replica"].pop("tensor_parallel_size", None)
                        entry.update(replica_ep_metadata[0])
                        entry["via_replica"].update(replica_ep_metadata[0])
                    for field in (
                        "requested_kv_cache_dtype",
                        "kv_cache_dtype",
                    ):
                        values = [
                            item.get(field)
                            for item in replica_engines
                        ]
                        if (
                            values
                            and all(isinstance(value, str) and value for value in values)
                            and len(set(values)) == 1
                        ):
                            value = values[0]
                            entry[field] = value
                            entry["via_replica"][field] = value
                    generation_metadata = [
                        _generation_metadata(item) for item in replica_engines
                    ]
                    if (
                        generation_metadata
                        and generation_metadata[0] is not None
                        and all(
                            metadata == generation_metadata[0]
                            for metadata in generation_metadata[1:]
                        )
                    ):
                        entry["via_replica"].update(generation_metadata[0])
            engine_list.append(entry)

        for name, orchestrator in orchestrator_resources.items():
            snapshot = getattr(orchestrator, "generation_defaults_snapshot", None)
            raw_defaults = snapshot() if callable(snapshot) else {}
            if not isinstance(raw_defaults, Mapping):
                raw_defaults = {}
            worker_metadata = {
                str(worker): (
                    {
                        "generation_config": defaults.mode,
                        "generation_config_source": defaults.source,
                        "generation_defaults": defaults.sampling_defaults(),
                    }
                    if isinstance(defaults, GenerationDefaults)
                    else None
                )
                for worker, defaults in raw_defaults.items()
            }
            entry = {
                "model": name,
                "engine_backend": "orchestrator",
                "attention_backend": None,
                "attention_components": None,
                "generation_defaults_by_worker": worker_metadata,
            }
            complete = [
                metadata
                for metadata in worker_metadata.values()
                if metadata is not None
            ]
            if (
                complete
                and len(complete) == len(worker_metadata)
                and all(metadata == complete[0] for metadata in complete[1:])
            ):
                entry.update(complete[0])
            engine_list.append(entry)

        role_entries = [
            entry
            for entry in engine_list
            if entry["engine_backend"] != "orchestrator"
        ]
        role = (
            "gateway"
            if role_entries
            and all(
                entry["engine_backend"] == "replica-pool"
                for entry in role_entries
            )
            else "engine-host"
        )

        return {
            "attention_backend": attention,
            "requested_attention_backend": decision.requested,
            "attention_components": dict(decision.components),
            "selection_rationale": decision.rationale,
            "source": decision.source,
            "architecture": dict(decision.architecture),
            "kernel_tier": _kernel_tier_for(
                decision.architecture,
                kernel_tier,
            ),
            "role": role,
            "versions": _versions_for(decision.components),
            "engines": engine_list,
        }

    if metrics is not None:

        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            body, content_type = await asyncio.to_thread(metrics.render)
            return Response(content=body, media_type=content_type)

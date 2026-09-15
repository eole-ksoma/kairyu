"""YAML front-end for OrchestratorSpec plus the spec -> Orchestrator builder."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from kairyu.dsl.spec import OrchestratorSpec, RoleNodeSpec, WorkerSpec
from kairyu.engine.backend import EngineBackend
from kairyu.engine.registry import create_backend
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import (
    ExecutorRoleConfig,
    RoleSamplingOverrides,
    RoleSpec,
    chars_cost_model,
    zero_cost,
)
from kairyu.orchestration.execution import (
    ExecutionBackend,
    ExecutionLimits,
    ExecutorDescriptor,
)
from kairyu.orchestration.orchestrator import (
    EngineDescriptor,
    Orchestrator,
    ProfileChoice,
    ProfileJudge,
)
from kairyu.orchestration.router import RouteThresholds, RuleRouter, load_calibrated_router
from kairyu.sampling_params import SamplingParams


def load_spec(source: str | Path) -> OrchestratorSpec:
    """Load an OrchestratorSpec from a YAML file path or a YAML string."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source.strip()
                                    and Path(source).exists()):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("orchestrator spec YAML must be a mapping at the top level")
    return OrchestratorSpec.model_validate(data)


def _role_sampling(role: RoleNodeSpec) -> RoleSamplingOverrides | None:
    if role.sampling is None:
        return None
    return RoleSamplingOverrides(
        temperature=role.sampling.temperature,
        top_p=role.sampling.top_p,
        top_k=role.sampling.top_k,
        min_p=role.sampling.min_p,
        presence_penalty=role.sampling.presence_penalty,
        repetition_penalty=role.sampling.repetition_penalty,
        max_tokens=role.sampling.max_tokens,
        max_tokens_by_effort=(
            None
            if role.sampling.max_tokens_by_effort is None
            else role.sampling.max_tokens_by_effort.model_dump()
        ),
        seed_offset=role.sampling.seed_offset,
        stop=role.sampling.stop,
    )


def _role_executor(role: RoleNodeSpec) -> ExecutorRoleConfig | None:
    if role.executor is None:
        return None
    limits = role.executor.limits
    return ExecutorRoleConfig(
        code_from=role.executor.code_from,
        tests_from=role.executor.tests_from,
        mode=role.executor.mode,
        limits=ExecutionLimits(
            wall_time_s=limits.wall_time_s,
            cpu_time_s=limits.cpu_time_s,
            memory_mb=limits.memory_mb,
            processes=limits.processes,
            output_bytes=limits.output_bytes,
        ),
    )


def _build_worker(worker: WorkerSpec) -> EngineBackend:
    if worker.engine_ref is not None:
        raise ValueError("engine_ref workers require deployment engine resolution")
    if worker.executor_ref is not None:
        raise ValueError("executor_ref workers require deployment executor resolution")
    options = dict(worker.options)
    if worker.model is not None:
        options.setdefault("model", worker.model)
    if worker.base_url is not None:
        options.setdefault("base_url", worker.base_url)
    # OpenAICompatBackend defaults to OPENAI_API_KEY when this argument is
    # omitted. Preserve WorkerSpec's None so local node-to-node workers remain
    # explicitly keyless.
    if worker.backend == "openai" or worker.api_key_env is not None:
        options.setdefault("api_key_env", worker.api_key_env)
    return create_backend(worker.backend, **options)


def build_orchestrator(
    spec: OrchestratorSpec,
    *,
    engine_refs: Mapping[str, EngineBackend] | None = None,
    executor_refs: Mapping[str, ExecutionBackend] | None = None,
) -> Orchestrator:
    available_refs = dict(engine_refs or {})
    available_executor_refs = dict(executor_refs or {})
    engines: dict[str, EngineBackend] = {}
    execution_workers: dict[str, ExecutionBackend] = {}
    executor_descriptors: dict[str, ExecutorDescriptor] = {}
    owned_engines: list[EngineBackend] = []
    for worker in spec.workers:
        if worker.engine_ref is not None:
            try:
                engines[worker.name] = available_refs[worker.engine_ref]
            except KeyError as error:
                raise ValueError(
                    f"worker {worker.name!r} references unknown deployment engine "
                    f"{worker.engine_ref!r}"
                ) from error
        elif worker.executor_ref is not None:
            try:
                backend = available_executor_refs[worker.executor_ref]
            except KeyError as error:
                raise ValueError(
                    f"worker {worker.name!r} references unknown deployment executor "
                    f"{worker.executor_ref!r}"
                ) from error
            execution_workers[worker.name] = backend
            executor_descriptors[worker.name] = ExecutorDescriptor(
                backend_type=type(backend).__name__,
                base_url=getattr(backend, "base_url", None),
            )
        else:
            engine = _build_worker(worker)
            engines[worker.name] = engine
            owned_engines.append(engine)
    engine_descriptors = {
        worker.name: EngineDescriptor(
            backend_type=(
                type(engines[worker.name]).__name__
                if worker.engine_ref is not None
                else worker.backend
            ),
            model=(
                worker.engine_ref
                if worker.engine_ref is not None
                else (
                    worker.model
                    if worker.model is not None
                    else (
                        worker.options.get("model")
                        if isinstance(worker.options.get("model"), str)
                        else None
                    )
                )
            ),
        )
        for worker in spec.workers
        if worker.executor_ref is None
    }
    def _role_spec(role: RoleNodeSpec) -> RoleSpec:
        return RoleSpec(
            name=role.name,
            worker=role.worker,
            prompt=role.prompt,
            role_type=role.role_type,
            depends_on=role.depends_on,
            verifies=role.verifies,
            sampling=_role_sampling(role),
            executor=_role_executor(role),
            prompt_suffix=role.prompt_suffix,
            prompt_headless=role.prompt_headless,
            reasoning_closed=role.reasoning_closed,
            reasoning_effort=role.reasoning_effort,
            reasoning_close_tag=role.reasoning_close_tag,
            reasoning_continuation=role.reasoning_continuation,
            reasoning_open_tag=role.reasoning_open_tag,
            requires=role.requires,
        )

    roles = tuple(_role_spec(role) for role in spec.roles) or None
    profiles = (
        {
            profile.name: tuple(_role_spec(role) for role in profile.roles)
            for profile in spec.profiles
        }
        or None
    )
    profile_judge = (
        ProfileJudge(
            worker=spec.profile_judge.worker,
            timeout_seconds=spec.profile_judge.timeout_seconds,
            max_tokens=spec.profile_judge.max_tokens,
            prompt_prefix=spec.profile_judge.prompt_prefix,
            prompt_suffix=spec.profile_judge.prompt_suffix,
            choices=tuple(
                ProfileChoice(
                    profile=choice.profile,
                    label=choice.label,
                    criteria=choice.criteria,
                )
                for choice in spec.profile_judge.choices
            ),
            fallback=spec.profile_judge.fallback,
        )
        if spec.profile_judge is not None
        else None
    )
    budget = Budget(
        max_steps=spec.budget.max_steps,
        max_refine_depth=spec.budget.max_refine_depth,
        max_cost_usd=spec.budget.max_cost_usd,
    )
    rate = spec.budget.cost_per_1k_chars_usd
    cost_model = chars_cost_model(rate) if rate is not None else zero_cost
    router = (
        RuleRouter(
            RouteThresholds(**spec.router.thresholds.model_dump())
            if spec.router.thresholds is not None
            else RouteThresholds()
        )
        if spec.router.kind == "rules"
        else load_calibrated_router(
            spec.router.artifact or "",
            expected_sha256=spec.router.sha256 or "",
            target_mode=spec.router.target_mode,
        )
    )
    return Orchestrator(
        engines=engines,
        router=router,
        roles=roles,
        profiles=profiles,
        budget=budget,
        shared_prefix=spec.shared_prefix,
        sampling_params=SamplingParams(max_tokens=spec.internal_max_tokens),
        cost_model=cost_model,
        moa_samples=spec.moa_samples,
        engine_descriptors=engine_descriptors,
        owned_engines=owned_engines,
        expose_intermediate_outputs=spec.expose_intermediate_outputs,
        execution_workers=execution_workers,
        executor_descriptors=executor_descriptors,
        profile_judge=profile_judge,
        default_reasoning_effort=spec.default_reasoning_effort,
        public_output_floor=spec.public_output_floor,
    )

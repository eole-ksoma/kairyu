"""Orchestrator facade: route a query, then dispatch to an engine or the Conductor."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace

from kairyu.async_thread import run_prompt_work, run_serialized_prompt_work
from kairyu.engine.backend import (
    UNLIMITED_OUTPUT_ADMISSION_TOKENS,
    AdmissionUpperBound,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    backend_admission_upper_bound,
    backend_max_model_len,
    backend_supports_chat_template_kwargs,
    backend_supports_prompt_kind,
    prepare_backend_request,
    shutdown_all,
    validate_backend_request_before_prepare,
)
from kairyu.engine.backend import (
    admission_upper_bound as generation_admission_upper_bound,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    TemplatedPrompt,
    derive_multimodal_prompt,
)
from kairyu.models.generation import GenerationDefaults
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.conductor import (
    Conductor,
    ConductorStreamError,
    CostModel,
    RoleSpec,
    zero_cost,
)
from kairyu.orchestration.execution import ExecutionBackend, ExecutorDescriptor
from kairyu.orchestration.features import latest_user_view
from kairyu.orchestration.moa import _build_moa_setup, _MoASetup
from kairyu.orchestration.request import (
    OrchestrationRequest,
    default_orchestration_request,
)
from kairyu.orchestration.router import RouteDecision, Router, RuleRouter
from kairyu.orchestration.trace import (
    StructuredTrace,
    TraceBudget,
    TraceError,
    TraceEvent,
    TraceTiming,
    TraceUsage,
    WorkerTraceIdentity,
    utc_now_iso,
)
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams

_KEEPALIVE_INTERVAL_S = 15.0  # SSE keep-alive cadence for long multi-stage runs (M8)


def _public_multistage_completions(
    completions: tuple[CompletionOutput, ...],
    reasoning_content: str | None = None,
) -> tuple[CompletionOutput, ...]:
    """Keep final text while replacing final-stage reasoning with policy output."""

    return tuple(
        replace(
            completion,
            reasoning_content=reasoning_content,
            reasoning_delta=None,
            reasoning_offset=None,
        )
        for completion in completions
    )


_DEFAULT_ROLES = (
    RoleSpec(
        name="planner",
        worker="tier2",
        role_type="planner",
        prompt="[planner] Break the task into a short actionable plan.\nTask: {query}",
    ),
    RoleSpec(
        name="worker",
        worker="tier1",
        prompt="[worker] Execute this plan and produce the answer.\nPlan: {planner}\nTask: {query}",
        depends_on=("planner",),
    ),
    RoleSpec(
        name="verifier",
        worker="tier2",
        role_type="verifier",
        verifies="worker",
        prompt=(
            "[verifier] Check the answer for the task. Reply PASS or FAIL with reasons.\n"
            "Task: {query}\nAnswer: {worker}"
        ),
        depends_on=("worker",),
    ),
    RoleSpec(
        name="synthesizer",
        worker="tier2",
        role_type="synthesizer",
        prompt=(
            "[synthesizer] Produce the final polished answer.\n"
            "Task: {query}\nDraft: {worker}\nVerifier notes: {verifier}"
        ),
        depends_on=("worker", "verifier"),
    ),
)


@dataclass(frozen=True)
class OrchestratorResult:
    text: str
    route: RouteDecision
    trace: tuple[str, ...]
    completions: tuple[CompletionOutput, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    structured_trace: StructuredTrace | None = None
    reasoning_content: str | None = None
    # Backend-reported tokens of the client-visible request/completion
    # (issue #496). ``prompt_tokens``/``completion_tokens`` above stay the
    # cumulative internal-call totals (m11 #196); the wire layer maps THESE
    # onto standard OpenAI usage and falls back to the m9 approximation when
    # None. Direct routes report both (one public call); multi-stage routes
    # leave the prompt None because no backend call tokenizes exactly the
    # caller's request.
    public_prompt_tokens: int | None = None
    public_completion_tokens: int | None = None


@dataclass(frozen=True)
class OrchestratorEvent:
    """Streaming event (m11 D1): status keep-alive, token delta, or final."""

    kind: str  # "status" | "reasoning" | "delta" | "error" | "result"
    text: str = ""
    completions: tuple[CompletionOutput, ...] = ()
    result: OrchestratorResult | None = None
    error_type: str | None = None


class OrchestratorExecutionError(RuntimeError):
    """Backend failure plus the partial accounting/trace safe to return."""

    def __init__(
        self,
        cause: BaseException,
        result: OrchestratorResult,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.result = result


class _DirectExecutionError(RuntimeError):
    def __init__(self, cause: BaseException, event: TraceEvent) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause
        self.event = event


class PreviewNotSupportedError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineDescriptor:
    backend_type: str
    model: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"backend_type": self.backend_type, "model": self.model}


@dataclass(frozen=True)
class ProfileChoice:
    """One route the profile judge may answer with (DTO-D13)."""

    profile: str
    label: str
    criteria: str


@dataclass(frozen=True)
class ProfileJudge:
    """LLM route selection among the configured role profiles (issue #509,
    generalized by DTO-D13).

    The judge answers exactly one choice label over the latest user turn and
    the selected profile's DAG serves the call. Timeout, backend error, and an
    unparseable verdict fall back to ``fallback``; profile selection itself
    stays a pure function of the attached verdict.
    """

    worker: str
    timeout_seconds: float = 5.0
    max_tokens: int = 8
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    choices: tuple[ProfileChoice, ...] = ()
    fallback: str = "primary"

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError("profile_judge requires at least two choices")
        labels = [choice.label for choice in self.choices]
        if len(set(labels)) != len(labels):
            raise ValueError("profile_judge choice labels must be unique")
        profiles = [choice.profile for choice in self.choices]
        if len(set(profiles)) != len(profiles):
            raise ValueError("profile_judge choices must reference distinct profiles")


# Verdict-only classification for a non-thinking engine: the judge picks one
# route label; the deployment spec owns the labels and their criteria.
_PROFILE_JUDGE_PROMPT_HEADER = (
    "You route requests inside an inference product. Choose the route that "
    "answers the user's request correctly and with high quality at the lowest "
    "latency and cost: pick the fastest route you are confident will still "
    "give a correct and complete answer, and escalate only when the request "
    "genuinely needs it. Routes, from fastest and cheapest to slowest and "
    "most thorough:\n"
)
# Bound the judged view so a long agent conversation cannot inflate the
# judge's prefill beyond a fixed latency envelope.
_PROFILE_JUDGE_VIEW_CHARS = 4000
_PROFILE_JUDGE_TRUNCATION_MARKER = "\n\n[... middle of request omitted ...]\n\n"


def _bounded_profile_judge_view(prompt: str) -> str:
    view = latest_user_view(prompt)
    if len(view) <= _PROFILE_JUDGE_VIEW_CHARS:
        return view

    available_chars = _PROFILE_JUDGE_VIEW_CHARS - len(_PROFILE_JUDGE_TRUNCATION_MARKER)
    head_chars = (available_chars + 1) // 2
    tail_chars = available_chars - head_chars
    return (
        view[:head_chars]
        + _PROFILE_JUDGE_TRUNCATION_MARKER
        + view[-tail_chars:]
    )


def _profile_judge_prompt(
    choices: tuple[ProfileChoice, ...],
    request_view: str,
    *,
    tools: bool,
    image: bool,
) -> str:
    labels = ", ".join(choice.label for choice in choices)
    return (
        _PROFILE_JUDGE_PROMPT_HEADER
        + "".join(f"- {choice.label}: {choice.criteria}\n" for choice in choices)
        + "\nRequest context: tool calling "
        + ("yes" if tools else "no")
        + "; image attached "
        + ("yes" if image else "no")
        + ".\n"
        + f"Reply with exactly one word: {labels}.\n\n"
        + f"USER REQUEST:\n{request_view}\n\nAnswer:"
    )


def _parse_profile_verdict(
    text: str,
    choices: tuple[ProfileChoice, ...],
) -> str | None:
    """The chosen profile, or None when the verdict is not exactly one offered label."""

    verdict = text.strip()
    if verdict.endswith("."):
        verdict = verdict[:-1]
    verdict = verdict.casefold()
    for choice in choices:
        if verdict == choice.label.casefold():
            return choice.profile
    return None


@dataclass(frozen=True)
class _IntentRequest:
    engine_key: str
    request: GenerationRequest
    role_name: str | None = None
    validation_only: bool = False


@dataclass
class _PreparedDispatchState:
    """One alias-shared single-use capability for prepared dispatch."""

    final_requests: tuple[_IntentRequest, ...]
    initial_requests: tuple[_IntentRequest, ...] = ()
    conductor_session: str | None = None
    moa_setup: _MoASetup | None = None
    consumed: bool = False

    def __deepcopy__(self, _memo):
        return self


@dataclass(frozen=True)
class _PreparedOrchestration:
    """Request-local preflight plan that may be consumed by direct dispatch."""

    source_request: str | OrchestrationRequest
    owner_token: object
    call: OrchestrationRequest
    decision: RouteDecision | None
    dispatch: _PreparedDispatchState

    @property
    def consumed(self) -> bool:
        return self.dispatch.consumed

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self


@dataclass(frozen=True)
class _OrchestrationPreparationPlan:
    call: OrchestrationRequest
    decision: RouteDecision
    final_requests: tuple[_IntentRequest, ...]
    internal_requests: tuple[_IntentRequest, ...]
    initial_requests: tuple[_IntentRequest, ...]
    conductor_session: str | None
    moa_setup: _MoASetup | None


def _trace_started_at_from(judge_event: TraceEvent | None) -> str:
    """Trace envelope start for a judged request.

    The judge event is the earliest event in the envelope and stamps
    ``queued_at`` one clock read before ``started_at``; deriving the envelope
    from ``started_at`` let a millisecond boundary between the two reads put
    the judge's ``queued_at`` outside the envelope (serving_bench rejects the
    whole trace). Anchor on the earliest stamp the event carries.
    """

    if judge_event is not None and judge_event.timing is not None:
        timing = judge_event.timing
        return timing.queued_at or timing.started_at or utc_now_iso()
    return utc_now_iso()


class Orchestrator:
    def __init__(
        self,
        engines: Mapping[str, EngineBackend],
        router: Router | None = None,
        roles: tuple[RoleSpec, ...] | None = None,
        profiles: Mapping[str, tuple[RoleSpec, ...]] | None = None,
        budget: Budget | None = None,
        shared_prefix: str = "",
        sampling_params: SamplingParams | None = None,
        cost_model: CostModel = zero_cost,
        moa_samples: int = 0,
        engine_descriptors: Mapping[str, EngineDescriptor] | None = None,
        owned_engines: Iterable[EngineBackend] | None = None,
        expose_intermediate_outputs: bool = False,
        execution_workers: Mapping[str, ExecutionBackend] | None = None,
        executor_descriptors: Mapping[str, ExecutorDescriptor] | None = None,
        profile_judge: ProfileJudge | None = None,
        default_reasoning_effort: str | None = None,
        public_output_floor: int | None = None,
    ) -> None:
        if not engines:
            raise ValueError("Orchestrator requires at least one engine")
        if isinstance(shared_prefix, TemplatedPrompt):
            raise ValueError(
                "orchestration shared_prefix cannot be a tokenizer-owned pre-rendered chat prompt"
            )
        if public_output_floor is not None and (
            isinstance(public_output_floor, bool)
            or not isinstance(public_output_floor, int)
            or not 1 <= public_output_floor <= 131_072
        ):
            raise ValueError(
                "public_output_floor must be an integer between 1 and 131072"
            )
        if public_output_floor is not None and moa_samples > 0:
            raise ValueError(
                "public_output_floor cannot be combined with moa_samples > 0; "
                "the floor applies only to the Conductor final unit"
            )
        if profiles and moa_samples > 0:
            raise ValueError(
                "profiles cannot be combined with moa_samples > 0; "
                "choose role DAG profiles or MoA mode"
            )
        self._engines = dict(engines)
        self._owned_engines = tuple(
            {
                id(engine): engine
                for engine in (self._engines.values() if owned_engines is None else owned_engines)
            }.values()
        )
        self._owner_token = object()
        supplied_descriptors = dict(engine_descriptors or {})
        self._engine_descriptors = {
            name: supplied_descriptors.get(
                name,
                EngineDescriptor(backend_type=type(engine).__name__),
            )
            for name, engine in self._engines.items()
        }
        self._router = router or RuleRouter()
        self._router_gate = asyncio.Lock()
        self._roles = roles or _DEFAULT_ROLES
        # Named alternative role DAGs (issue #509, generalized by DTO-D13):
        # one profile is selected per request; every profile runs the
        # Conductor, whether it is a full ensemble or a single-role route.
        # "primary" is always the roles DAG above.
        extra_profiles = dict(profiles or {})
        if "primary" in extra_profiles:
            raise ValueError("profile name 'primary' is reserved for the roles DAG")
        self._extra_profiles: dict[str, tuple[RoleSpec, ...]] | None = (
            extra_profiles or None
        )
        self._profiles: dict[str, tuple[RoleSpec, ...]] = {
            "primary": self._roles,
            **extra_profiles,
        }
        # Optional LLM verdict selecting the profile (issue #509 amendment).
        # Judgment is attached to the call at the serving boundary; profile
        # selection itself stays a pure function of the call.
        self._profile_judge = profile_judge
        if profile_judge is not None:
            if self._extra_profiles is None:
                raise ValueError("profile_judge requires at least one alternative profile")
            if profile_judge.worker not in self._engines:
                raise ValueError(
                    f"profile_judge references unknown worker {profile_judge.worker!r}"
                )
            unknown = [
                choice.profile
                for choice in profile_judge.choices
                if choice.profile not in self._profiles
            ]
            if unknown:
                raise ValueError(
                    f"profile_judge choices reference unknown profiles {unknown}"
                )
            if profile_judge.fallback not in self._profiles:
                raise ValueError(
                    f"profile_judge fallback references unknown profile "
                    f"{profile_judge.fallback!r}"
                )
        self._budget = budget or Budget()
        self._shared_prefix = shared_prefix
        self._sampling_params = sampling_params or SamplingParams(max_tokens=1024)
        self._cost_model = cost_model
        # m11 A4: >0 routes multi_agent through MoA (the deep kairyu-auto-max tier)
        self._moa_samples = moa_samples
        if default_reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError(
                "default_reasoning_effort must be low, high, max, or null"
            )
        # Effort assumed when a call carries none; an explicit request-level
        # effort always overrides it.
        self._default_reasoning_effort = default_reasoning_effort
        self._public_output_floor = public_output_floor
        self._expose_intermediate_outputs = expose_intermediate_outputs
        # Serving-layer hook for aggregate per-stage latency attribution
        # (issue #495); set after construction because the deploy builder,
        # not the HTTP app that owns metrics, constructs orchestrators.
        self._stage_observer: Callable[[TraceEvent], None] | None = None
        self._execution_workers = dict(execution_workers or {})
        supplied_executor_descriptors = dict(executor_descriptors or {})
        self._executor_descriptors = {
            name: supplied_executor_descriptors.get(
                name,
                ExecutorDescriptor(backend_type=type(backend).__name__),
            )
            for name, backend in self._execution_workers.items()
        }
        if self._moa_samples == 0:
            # Fail fast on an invalid role DAG (head shape, executor bindings,
            # final-unit overrides) at construction instead of per request —
            # for every configured profile.
            for profile_roles in self._profiles.values():
                workers = self._conductor_workers(profile_roles, [])
                Conductor(
                    roles=profile_roles,
                    workers=workers,
                    shared_prefix=self._shared_prefix,
                    sampling_params=self._sampling_params,
                    execution_workers=self._execution_workers,
                    public_output_floor=self._profile_output_floor(profile_roles),
                )
            if self._public_output_floor is not None and not any(
                self._profile_output_floor(profile_roles) is not None
                for profile_roles in self._profiles.values()
            ):
                # A floor no profile can consume would silently never reserve
                # anything — fail the deployment loudly instead (issue #542).
                raise ValueError(
                    "public_output_floor requires the final unit of at least one "
                    "profile to declare reasoning_close_tag"
                )
                # Whether an image-conditional role's worker accepts images is
                # a live capability (a replica pool publishes the intersection
                # of its currently placeable replicas, which is empty before
                # the first health probe), so it is checked per image request
                # in _validate_call, not here.

    def generation_defaults_snapshot(
        self,
    ) -> dict[str, GenerationDefaults | None]:
        """Return immutable worker policies without exposing backend ownership."""

        return {
            name: (
                defaults
                if isinstance(
                    defaults := getattr(engine, "generation_defaults", None),
                    GenerationDefaults,
                )
                else None
            )
            for name, engine in self._engines.items()
        }

    async def startup(self) -> None:
        """Eagerly start owned process resources before serving auto models."""

        for engine in self._owned_engines:
            startup = getattr(engine, "startup", None)
            if callable(startup):
                await startup()

    def preview_route(self, prompt: str) -> RouteDecision:
        if isinstance(prompt, TemplatedPrompt):
            raise ValueError(
                "orchestration cannot safely derive prompts from an already rendered chat template"
            )
        preview = getattr(self._router, "preview", None)
        if preview is None:
            raise PreviewNotSupportedError(
                f"router {type(self._router).__name__} does not support preview"
            )
        try:
            return preview(prompt)
        except NotImplementedError as error:
            raise PreviewNotSupportedError(str(error)) from error

    async def preview_route_async(self, prompt: str) -> RouteDecision:
        """Run request-sized preview work without blocking the serving loop."""

        return await run_serialized_prompt_work(
            self._router_gate,
            None,
            self.preview_route,
            prompt,
        )

    def _resolved_engine_descriptor(self, key: str) -> dict[str, object]:
        configured = key in self._engines
        effective = key if configured else next(iter(self._engines))
        return {
            "configured": configured,
            "engine": effective,
            "fallback": not configured,
        }

    def describe_routing(self) -> dict[str, object]:
        describe = getattr(self._router, "describe", None)
        router = (
            describe() if describe is not None else {"router_type": type(self._router).__name__}
        )
        role_workers = tuple(
            dict.fromkeys(
                role.worker
                for roles in self._profiles.values()
                for role in self._generation_roles(roles)
            )
        )
        if self._moa_samples > 0:
            multi_engines = [self._resolved_engine_descriptor(key) for key in ("tier1", "tier2")]
            multi_mode = "moa"
        else:
            multi_engines = [self._resolved_engine_descriptor(key) for key in role_workers]
            multi_mode = "roles"

        def role_payload(role: RoleSpec) -> dict[str, object]:
            payload: dict[str, object] = {
                "name": role.name,
                "worker": role.worker,
                "role_type": role.role_type,
                "depends_on": list(role.depends_on),
                "verifies": role.verifies,
            }
            if role.sampling is not None:
                sampling = {
                    key: value
                    for key, value in asdict(role.sampling).items()
                    if value not in (None, ())
                }
                max_tokens_by_effort = sampling.get("max_tokens_by_effort")
                if max_tokens_by_effort is not None:
                    sampling["max_tokens_by_effort"] = dict(
                        max_tokens_by_effort
                    )
                payload["sampling"] = sampling
            if role.executor is not None:
                payload["executor"] = {
                    "code_from": list(role.executor.code_from),
                    "tests_from": list(role.executor.tests_from),
                    "mode": role.executor.mode,
                }
            return payload

        head = self._conductor_head_role(self._roles)
        profile_payload = (
            {
                "profiles": {
                    name: [role_payload(role) for role in profile_roles]
                    for name, profile_roles in self._extra_profiles.items()
                },
                "profile_selector": (
                    "per the attached LLM profile-judge verdict over the latest "
                    "user turn (fallback "
                    f"{self._profile_judge.fallback!r} on timeout, backend "
                    "error, or an unparseable verdict); every profile is a "
                    "Conductor DAG (issue #509, DTO-D13)"
                    if self._profile_judge is not None
                    else "primary; no profile judge is configured"
                ),
                **(
                    {
                        "profile_judge": {
                            "worker": self._profile_judge.worker,
                            "timeout_seconds": self._profile_judge.timeout_seconds,
                            "max_tokens": self._profile_judge.max_tokens,
                            "prompt_prefix": self._profile_judge.prompt_prefix,
                            "prompt_suffix": self._profile_judge.prompt_suffix,
                            "fallback": self._profile_judge.fallback,
                            "choices": [
                                {
                                    "profile": choice.profile,
                                    "label": choice.label,
                                    "criteria": choice.criteria,
                                }
                                for choice in self._profile_judge.choices
                            ],
                        }
                    }
                    if self._profile_judge is not None
                    else {}
                ),
            }
            if self._extra_profiles is not None
            else {}
        )
        return {
            "router": router,
            "targets": ["tier1", "tier2", "multi_agent"],
            **profile_payload,
            "configured_engines": {
                name: descriptor.as_dict() for name, descriptor in self._engine_descriptors.items()
            },
            "configured_executors": {
                name: descriptor.as_dict()
                for name, descriptor in self._executor_descriptors.items()
            },
            "target_resolution": {
                "tier1": self._resolved_engine_descriptor("tier1"),
                "tier2": self._resolved_engine_descriptor("tier2"),
                "multi_agent": {
                    "mode": multi_mode,
                    "engines": multi_engines,
                },
            },
            "roles": [role_payload(role) for role in self._roles],
            "stream_head": head.name if head is not None else None,
            "budget": asdict(self._budget),
            "moa_samples": self._moa_samples,
            "internal_max_tokens": self._sampling_params.max_tokens,
            "expose_intermediate_outputs": self._expose_intermediate_outputs,
        }

    async def shutdown(self) -> None:
        await shutdown_all(self._owned_engines, "Orchestrator")

    def _resolve_engine_name(self, tier: str, notes: list[str]) -> str:
        if tier in self._engines:
            return tier
        fallback_name = next(iter(self._engines))
        notes.append(f"fallback: engine {tier!r} not configured, using {fallback_name!r}")
        return fallback_name

    def _resolve_engine(self, tier: str, notes: list[str]) -> EngineBackend:
        return self._engines[self._resolve_engine_name(tier, notes)]

    def _request(self, request: str | OrchestrationRequest) -> OrchestrationRequest:
        if isinstance(request, OrchestrationRequest):
            call = request
        else:
            call = default_orchestration_request(request, self._sampling_params)
        if isinstance(call.prompt, TemplatedPrompt):
            raise ValueError(
                "orchestration cannot safely derive prompts from an already rendered chat template"
            )
        return call

    def _route(self, prompt: str) -> RouteDecision:
        """Invoke a potentially stateful synchronous router serially off-loop."""

        return self._router.route(prompt)

    async def _route_async(self, prompt: str) -> RouteDecision:
        """Queue before the executor so a stateful router occupies one lane."""

        return await run_serialized_prompt_work(
            self._router_gate,
            None,
            self._route,
            prompt,
        )

    def _internal_sampling_params(
        self,
        call: OrchestrationRequest,
    ) -> SamplingParams:
        return call.internal_sampling_params(
            max_tokens_cap=self._sampling_params.max_tokens,
        )

    def _profile_output_floor(self, roles: tuple[RoleSpec, ...]) -> int | None:
        """The public-output floor this profile can consume (issue #542).

        The floor reserves answer tokens behind a final unit whose scaffold
        opens a reasoning span (``reasoning_close_tag``); a profile whose
        final unit has no such span cannot consume it and runs without one.
        """

        if self._public_output_floor is None or self._moa_samples > 0:
            return None
        final = self._conductor_final_role(roles)
        if not final.reasoning_close_tag or final.reasoning_closed:
            return None
        return self._public_output_floor

    def _fallback_profile(self) -> str:
        return self._profile_judge.fallback if self._profile_judge is not None else "primary"

    def _role_profile(self, call: OrchestrationRequest) -> str:
        """Deterministic per-request DAG profile (issue #509, DTO-D13).

        Pure function of the call so preflight, admission, and execution
        always agree: the attached judge verdict names the profile; without
        one (no judge, judge failure, unparseable verdict) the fallback
        profile serves. Every profile is a Conductor DAG.
        """

        if self._extra_profiles is None:
            return "primary"
        judgment = call.role_profile_judgment
        if judgment is not None and judgment in self._profiles:
            return judgment
        return self._fallback_profile()

    def _roles_for(self, call: OrchestrationRequest) -> tuple[RoleSpec, ...]:
        return self._profiles[self._role_profile(call)]

    def _profile_accepts_images(self, name: str) -> bool:
        fallback = next(iter(self._engines))
        return any(
            backend_supports_prompt_kind(
                self._engines[role.worker if role.worker in self._engines else fallback],
                "multimodal",
            )
            for role in self._generation_roles(self._profiles[name])
        )

    def _offered_choices(self, call: OrchestrationRequest) -> tuple[ProfileChoice, ...]:
        """The judge's choices for this call: image requests are offered only
        profiles with an image-capable worker (the same live capability
        ``_validate_call`` enforces), so a verdict can never select a DAG that
        would drop the image."""

        assert self._profile_judge is not None
        choices = self._profile_judge.choices
        if call.multimodal_prompt is None:
            return choices
        return tuple(
            choice for choice in choices if self._profile_accepts_images(choice.profile)
        )

    def will_judge_role_profile(
        self,
        request: str | OrchestrationRequest,
    ) -> bool:
        """Whether this call would dispatch the optional profile judge."""

        call = self._request(request)
        return not (
            self._profile_judge is None
            or self._extra_profiles is None
            or self._moa_samples > 0
            or call.role_profile_judgment is not None
            or call.role_profile_judge_event is not None
            or not isinstance(call.prompt, str)
            or len(self._offered_choices(call)) < 2
        )

    def _profile_judge_request(self, call: OrchestrationRequest) -> GenerationRequest:
        assert self._profile_judge is not None
        view = _bounded_profile_judge_view(call.prompt)
        prompt = (
            self._profile_judge.prompt_prefix
            + _profile_judge_prompt(
                self._offered_choices(call),
                view,
                tools=bool(call.tools or call.tools_in_prompt),
                image=call.multimodal_prompt is not None,
            )
            + self._profile_judge.prompt_suffix
        )
        if self._profile_judge.prompt_prefix or self._profile_judge.prompt_suffix:
            prompt = TemplatedPrompt(prompt)
        # The judge never thinks; an upstream reasoning parser must be told so
        # explicitly or it drops the non-streamed verdict (vLLM Qwen3 parser
        # contract, DTO-D13 amendment). Pre-rendered scaffolds carry no
        # template variables.
        chat_template_kwargs = (
            {"enable_thinking": False}
            if not isinstance(prompt, TemplatedPrompt)
            and backend_supports_chat_template_kwargs(
                self._engines[self._profile_judge.worker],
                frozenset({"enable_thinking"}),
            )
            else None
        )
        return GenerationRequest(
            request_id=f"profile-judge-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=self._profile_judge.max_tokens,
                temperature=0.0,
            ),
            chat_template_kwargs=chat_template_kwargs,
        )

    @staticmethod
    def _profile_judge_usage(call: OrchestrationRequest) -> GenerationUsage:
        event = call.role_profile_judge_event
        usage = event.usage if event is not None else None
        return GenerationUsage(
            prompt_tokens=usage.prompt_tokens if usage is not None else 0,
            completion_tokens=usage.completion_tokens if usage is not None else 0,
            cached_tokens=usage.cached_tokens if usage is not None else 0,
        )

    @classmethod
    def _usage_with_profile_judge(
        cls,
        call: OrchestrationRequest,
        usage: GenerationUsage,
    ) -> GenerationUsage:
        judge = cls._profile_judge_usage(call)
        return GenerationUsage(
            prompt_tokens=judge.prompt_tokens + usage.prompt_tokens,
            completion_tokens=judge.completion_tokens + usage.completion_tokens,
            cached_tokens=judge.cached_tokens + usage.cached_tokens,
        )

    async def judge_role_profile(
        self,
        request: str | OrchestrationRequest,
    ) -> OrchestrationRequest:
        """Attach one accounted profile verdict before preflight."""

        call = self._request(request)
        if not self.will_judge_role_profile(call):
            return call
        judge_request = self._profile_judge_request(call)
        queued_at = utc_now_iso()
        assert self._profile_judge is not None
        engine = self._engines[self._profile_judge.worker]
        descriptor = self._engine_descriptors[self._profile_judge.worker]
        started_at = utc_now_iso()
        try:
            result = await asyncio.wait_for(
                engine.generate(judge_request),
                timeout=self._profile_judge.timeout_seconds,
            )
        except Exception as error:
            event = TraceEvent(
                node="profile_judge",
                kind="fallback",
                detail="backend failure; fallback profile applies",
                operation="classification",
                status="failed",
                role="profile_judge",
                worker=self._profile_judge.worker,
                engine=self._profile_judge.worker,
                model=descriptor.model,
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                ),
                metadata={"fallback": "backend_error"},
                error=TraceError(type=type(error).__name__),
            )
            if self._stage_observer is not None:
                self._stage_observer(event)
            return replace(call, role_profile_judge_event=event)
        offered = self._offered_choices(call)
        verdict = _parse_profile_verdict(result.text, offered)
        trace_usage = (
            TraceUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cached_tokens=result.usage.cached_tokens,
            )
            if result.usage is not None
            else None
        )
        event = TraceEvent(
            node="profile_judge",
            kind="judged" if verdict is not None else "fallback",
            detail=(
                f"profile: {verdict}"
                if verdict is not None
                else "unparseable verdict; fallback profile applies"
            ),
            operation="classification",
            status="success" if verdict is not None else "failed",
            role="profile_judge",
            worker=self._profile_judge.worker,
            engine=self._profile_judge.worker,
            model=descriptor.model,
            timing=TraceTiming(
                queued_at=queued_at,
                started_at=started_at,
                completed_at=utc_now_iso(),
            ),
            usage=trace_usage,
            metadata={
                "verdict": verdict,
                "offered": [choice.label for choice in offered],
                "fallback": None if verdict is not None else "unparseable_verdict",
            },
        )
        if self._stage_observer is not None:
            self._stage_observer(event)
        return replace(
            call,
            role_profile_judgment=verdict,
            role_profile_judge_event=event,
        )

    def _conductor_final_role(self, roles: tuple[RoleSpec, ...]) -> RoleSpec:
        units = [role for role in roles if role.role_type != "verifier"]
        dependents = {dependency for role in units for dependency in role.depends_on}
        terminal = [
            role
            for role in units
            if role.name not in dependents and role.role_type not in {"head", "executor"}
        ]
        synthesizers = [role for role in terminal if role.role_type == "synthesizer"]
        if not terminal:
            raise ValueError("orchestration requires at least one terminal role")
        return (synthesizers + terminal)[0]

    def _conductor_head_role(self, roles: tuple[RoleSpec, ...]) -> RoleSpec | None:
        return next((role for role in roles if role.role_type == "head"), None)

    def _final_engine_keys(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> tuple[str, ...]:
        roles = self._roles_for(call)
        if decision is None:
            keys = {"tier1", "tier2", self._conductor_final_role(roles).worker}
        elif decision.target == "multi_agent":
            keys = {"tier2" if self._moa_samples > 0 else self._conductor_final_role(roles).worker}
        else:
            keys = {decision.target}
        fallback = next(iter(self._engines))
        return tuple(
            dict.fromkeys(key if key in self._engines else fallback for key in sorted(keys))
        )

    def _internal_engine_keys(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> tuple[str, ...]:
        if decision is not None and decision.target != "multi_agent":
            return ()
        if self._moa_samples > 0:
            keys = {"tier1"}
        else:
            roles = self._roles_for(call)
            final_role = self._conductor_final_role(roles)
            keys = {
                role.worker
                for role in self._generation_roles(roles)
                if role.name != final_role.name
            }
        fallback = next(iter(self._engines))
        return tuple(
            dict.fromkeys(key if key in self._engines else fallback for key in sorted(keys))
        )

    def _validate_call(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        if call.multimodal_prompt is not None:
            if self._moa_samples > 0 and (decision is None or decision.target == "multi_agent"):
                raise ValueError("multimodal orchestration does not support MoA sampling")
            if decision is not None and decision.target != "multi_agent":
                keys = self._final_engine_keys(call, decision)
                if not all(
                    backend_supports_prompt_kind(self._engines[key], "multimodal") for key in keys
                ):
                    raise ValueError(
                        f"orchestration route {decision.target!r} does not support image inputs"
                    )
                media_keys = keys
            else:
                fallback = next(iter(self._engines))
                role_keys = {
                    role.worker if role.worker in self._engines else fallback
                    for role in self._generation_roles(self._roles_for(call))
                }
                if not any(
                    backend_supports_prompt_kind(self._engines[key], "multimodal")
                    for key in role_keys
                ):
                    raise ValueError(
                        "multimodal orchestration requires at least one image-capable worker"
                    )
                for role in self._generation_roles(self._roles_for(call)):
                    # An image-conditional role only ever runs with an image
                    # attached, so its worker must accept one (DTO-D11) — a
                    # text-only rendering would silently drop the image.
                    if role.requires == "image" and not backend_supports_prompt_kind(
                        self._engines[
                            role.worker if role.worker in self._engines else fallback
                        ],
                        "multimodal",
                    ):
                        raise ValueError(
                            f"role {role.name!r} requires image input but worker "
                            f"{role.worker!r} does not accept multimodal prompts"
                        )
                media_keys = tuple(
                    key
                    for key in role_keys
                    if backend_supports_prompt_kind(
                        self._engines[key],
                        "multimodal",
                    )
                )
            if call.chat_template_kwargs is not None and not all(
                backend_supports_chat_template_kwargs(
                    self._engines[key],
                    frozenset(call.chat_template_kwargs),
                )
                for key in media_keys
            ):
                raise ValueError(
                    "image-capable orchestration workers do not all support chat_template_kwargs"
                )
        if call.sampling_params.n <= 1:
            return
        # A verifier on the final role is skipped for n > 1 inside the
        # Conductor (one verdict cannot judge independent choices), so only
        # engine support for n gates the call here.
        unsupported = [
            key
            for key in self._final_engine_keys(call, decision)
            if getattr(self._engines[key], "supports_n", True) is False
        ]
        if unsupported:
            raise ValueError(
                "n > 1 is not supported by final orchestration engine(s): " + ", ".join(unsupported)
            )

    def _final_intent_requests(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> tuple[_IntentRequest, ...]:
        sampling_params = call.sampling_params
        reasoning_effort = self._effective_reasoning_effort(call)
        conductor = None
        retry_assistant_prefill = None
        if (
            decision is not None
            and decision.target == "multi_agent"
            and self._moa_samples == 0
        ):
            conductor = self._new_conductor(call, [])
            sampling_params = conductor.final_intent_sampling_params()
            reasoning_effort = conductor.final_intent_reasoning_effort()
            retry_assistant_prefill = (
                conductor.final_retry_intent_assistant_prefill()
            )
        requests: list[_IntentRequest] = []
        for key in self._final_engine_keys(call, decision):
            prompt = self._engine_prompt(
                key,
                call,
                f"{self._shared_prefix}{call.prompt}",
            )
            chat_template_kwargs = (
                conductor.final_intent_chat_template_kwargs(prompt)
                if conductor is not None
                else (
                    call.chat_template_kwargs
                    if isinstance(prompt, MultimodalPrompt)
                    else None
                )
            )
            requests.append(
                _IntentRequest(
                    key,
                    GenerationRequest(
                        request_id=f"preflight-{key}",
                        prompt=prompt,
                        sampling_params=sampling_params,
                        tools=call.tools,
                        tool_choice=call.tool_choice,
                        tools_in_prompt=call.tools_in_prompt,
                        parallel_tool_calls=call.parallel_tool_calls,
                        tool_call_protocol=call.tool_call_protocol,
                        reasoning_effort=reasoning_effort,
                        chat_template_kwargs=chat_template_kwargs,
                    ),
                )
            )
            if retry_assistant_prefill is not None:
                requests.append(
                    _IntentRequest(
                        key,
                        replace(
                            requests[-1].request,
                            request_id=f"preflight-retry-{key}",
                            assistant_prefill=retry_assistant_prefill,
                        ),
                        validation_only=True,
                    )
                )
        return tuple(requests)

    def _direct_generation_request(
        self,
        call: OrchestrationRequest,
        request_id: str,
        engine_key: str,
    ) -> GenerationRequest:
        """Build a direct request without copying its prompt on the loop."""

        prompt = self._engine_prompt(
            engine_key,
            call,
            f"{self._shared_prefix}{call.prompt}",
        )
        return GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=call.sampling_params,
            # No random session: route by shared-prefix overlap and load.
            cache_hint=None,
            tools=call.tools,
            tool_choice=call.tool_choice,
            tools_in_prompt=call.tools_in_prompt,
            parallel_tool_calls=call.parallel_tool_calls,
            tool_call_protocol=call.tool_call_protocol,
            reasoning_effort=self._effective_reasoning_effort(call),
            chat_template_kwargs=(
                call.chat_template_kwargs if isinstance(prompt, MultimodalPrompt) else None
            ),
        )

    def _engine_prompt(
        self,
        engine_key: str,
        call: OrchestrationRequest,
        text: str,
    ):
        if call.multimodal_prompt is None or not backend_supports_prompt_kind(
            self._engines[engine_key],
            "multimodal",
        ):
            return text
        return derive_multimodal_prompt(call.multimodal_prompt, text)

    def _internal_intent_requests(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> tuple[_IntentRequest, ...]:
        sampling_params = self._internal_sampling_params(call)
        if self._moa_samples > 0 and (decision is None or decision.target == "multi_agent"):
            setup = _build_moa_setup(
                call.prompt,
                self._moa_samples,
                sampling_params,
                call.sampling_params,
                self._shared_prefix,
                session="validation-moa",
                reasoning_effort=self._effective_reasoning_effort(call),
            )
            key = self._internal_engine_keys(call, decision)[0]
            return tuple(_IntentRequest(key, request) for request in setup.proposal_requests)
        requests: list[_IntentRequest] = []
        for key in self._internal_engine_keys(call, decision):
            prompt = self._engine_prompt(
                key,
                call,
                f"{self._shared_prefix}{call.prompt}",
            )
            requests.append(
                _IntentRequest(
                    key,
                    GenerationRequest(
                        request_id=f"preflight-internal-{key}",
                        prompt=prompt,
                        sampling_params=sampling_params,
                        reasoning_effort=self._effective_reasoning_effort(call),
                        chat_template_kwargs=(
                            call.chat_template_kwargs
                            if isinstance(prompt, MultimodalPrompt)
                            else None
                        ),
                    ),
                )
            )
        return tuple(requests)

    def _initial_role_intent_requests(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
        *,
        request_id_suffix: str,
    ) -> tuple[_IntentRequest, ...]:
        """Return exact first-wave Conductor prompts for role orchestration."""

        if self._moa_samples > 0 or (decision is not None and decision.target != "multi_agent"):
            return ()
        conductor = self._new_conductor(call, [])
        fallback = next(iter(self._engines))
        return tuple(
            _IntentRequest(
                spec.worker if spec.worker in self._engines else fallback,
                request,
                spec.name,
            )
            for spec, request in conductor.initial_requests(
                call.prompt,
                request_id_suffix=request_id_suffix,
            )
        )

    @staticmethod
    def _intent_failure_message(kind: str, failures: list[str]) -> str:
        return f"{kind} orchestration intent is unsupported: " + "; ".join(failures)

    def _validate_intent_requests(
        self,
        kind: str,
        requests: tuple[_IntentRequest, ...],
        *,
        before_prepare: bool = False,
    ) -> None:
        failures: list[str] = []
        for intent in requests:
            try:
                engine = self._engines[intent.engine_key]
                if before_prepare:
                    validate_backend_request_before_prepare(
                        engine,
                        intent.request,
                    )
                else:
                    validate = getattr(engine, "validate_request", None)
                    if validate is None:
                        continue
                    validate(intent.request)
            except ValueError as error:
                failures.append(f"{intent.engine_key}: {error}")
        if failures:
            raise ValueError(self._intent_failure_message(kind, failures))

    def _validate_final_intent(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        self._validate_intent_requests(
            "final",
            self._final_intent_requests(call, decision),
        )

    def _validate_internal_intent(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        self._validate_intent_requests(
            "internal",
            self._internal_intent_requests(call, decision),
        )

    def _preflight_call(
        self,
        request: str | OrchestrationRequest,
    ) -> tuple[OrchestrationRequest, RouteDecision | None]:
        call = self._request(request)
        preview = getattr(self._router, "preview", None)
        if preview is None:
            decision = None
        else:
            try:
                decision = preview(call.prompt)
            except NotImplementedError:
                decision = None
        self._validate_call(call, decision)
        return call, decision

    def validate_request(self, request: str | OrchestrationRequest) -> None:
        """Preflight capability checks without dispatching generation."""

        call, decision = self._preflight_call(request)
        self._validate_final_intent(call, decision)
        self._validate_internal_intent(call, decision)
        self._validate_intent_requests(
            "initial",
            self._initial_role_intent_requests(
                call,
                decision,
                request_id_suffix="validation",
            ),
        )

    def validate_request_before_prepare(
        self,
        request: str | OrchestrationRequest,
    ) -> None:
        """Keep only bounded orchestration shape checks on the event loop."""

        self._request(request)

    def _build_preparation_plan(
        self,
        call: OrchestrationRequest,
        suffix: str,
        decision: RouteDecision,
    ) -> _OrchestrationPreparationPlan:
        """Route, render, and validate request-sized intents off-loop."""

        self._validate_call(call, decision)
        final_requests = self._final_intent_requests(call, decision)
        if decision.target != "multi_agent":
            final_requests = tuple(
                replace(
                    intent,
                    request=replace(
                        intent.request,
                        request_id=f"direct-{suffix}-{intent.engine_key}",
                    ),
                )
                for intent in final_requests
            )
        else:
            final_requests = tuple(
                replace(
                    intent,
                    request=replace(
                        intent.request,
                        request_id=f"{intent.request.request_id}-{suffix}",
                    ),
                )
                for intent in final_requests
            )
        moa_setup = None
        if decision.target == "multi_agent" and self._moa_samples > 0:
            moa_setup = _build_moa_setup(
                call.prompt,
                self._moa_samples,
                self._internal_sampling_params(call),
                call.sampling_params,
                self._shared_prefix,
                session=f"preflight-moa-{suffix}",
                reasoning_effort=self._effective_reasoning_effort(call),
            )
            proposal_key = self._internal_engine_keys(call, decision)[0]
            internal_requests = tuple(
                _IntentRequest(proposal_key, request) for request in moa_setup.proposal_requests
            )
        else:
            internal_requests = tuple(
                _IntentRequest(
                    intent.engine_key,
                    replace(
                        intent.request,
                        request_id=f"{intent.request.request_id}-{suffix}",
                    ),
                )
                for intent in self._internal_intent_requests(call, decision)
            )
        initial_requests = self._initial_role_intent_requests(
            call,
            decision,
            request_id_suffix=suffix,
        )
        return _OrchestrationPreparationPlan(
            call=call,
            decision=decision,
            final_requests=final_requests,
            internal_requests=internal_requests,
            initial_requests=initial_requests,
            conductor_session=(Conductor.preflight_session(suffix) if initial_requests else None),
            moa_setup=moa_setup,
        )

    async def prepare_request(
        self,
        request: str | OrchestrationRequest,
    ) -> _PreparedOrchestration:
        """Run semantic and backend-owned prompt preflight before dispatch."""

        call = self._request(request)
        suffix = uuid.uuid4().hex[:12]
        decision = await self._route_async(call.prompt)
        plan = await run_prompt_work(
            self._build_preparation_plan,
            call,
            suffix,
            decision,
        )
        # Backend state and ReplicaPool membership are event-loop owned.  The
        # request-sized strings are already built, so these bounded fast hooks
        # stay atomic with respect to membership mutation.
        self._validate_intent_requests(
            "final",
            plan.final_requests,
            before_prepare=True,
        )
        self._validate_intent_requests(
            "internal",
            plan.internal_requests,
            before_prepare=True,
        )
        self._validate_intent_requests(
            "initial",
            plan.initial_requests,
            before_prepare=True,
        )

        preparable_final_requests = tuple(
            intent for intent in plan.final_requests if not intent.validation_only
        )
        failure_groups: list[str] = []
        for kind, intents in (
            ("final", preparable_final_requests),
            ("internal", plan.internal_requests),
            ("initial", plan.initial_requests),
        ):
            failures: list[str] = []
            for intent in intents:
                try:
                    await prepare_backend_request(
                        self._engines[intent.engine_key],
                        intent.request,
                    )
                except ValueError as error:
                    failures.append(f"{intent.engine_key}: {error}")
            if failures:
                failure_groups.append(self._intent_failure_message(kind, failures))
        if failure_groups:
            raise ValueError("; ".join(failure_groups))
        return _PreparedOrchestration(
            source_request=request,
            owner_token=self._owner_token,
            call=plan.call,
            decision=plan.decision,
            dispatch=_PreparedDispatchState(
                final_requests=preparable_final_requests,
                initial_requests=plan.initial_requests,
                conductor_session=plan.conductor_session,
                moa_setup=plan.moa_setup,
            ),
        )

    def _call_with_prepared_handle(
        self,
        request: str | OrchestrationRequest,
        prepared: _PreparedOrchestration | None,
    ) -> OrchestrationRequest:
        if (
            prepared is not None
            and prepared.owner_token is self._owner_token
            and (
                request is prepared.call
                or request is prepared.source_request
                or (
                    type(request) is str
                    and type(prepared.source_request) is str
                    and request == prepared.source_request
                )
            )
        ):
            return prepared.call
        return self._request(request)

    def _take_prepared_decision(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
    ) -> RouteDecision | None:
        """Consume the route decision bound by exact-call async preflight."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or prepared.dispatch.consumed
        ):
            return None
        prepared.dispatch.consumed = True
        assert prepared.decision is not None
        if prepared.decision.target == "multi_agent":
            # Multi-agent prompts evolve after each stage, so synthetic final
            # preparation is validation-only and cannot be dispatched exactly.
            prepared.dispatch.final_requests = ()
        return prepared.decision

    def _prepared_direct_request(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
        decision: RouteDecision,
        engine_name: str,
    ) -> GenerationRequest | None:
        """Reuse exact direct preflight work when preview and route agree."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or prepared.decision is None
            or prepared.decision.target != decision.target
            or decision.target == "multi_agent"
        ):
            return None
        request = next(
            (
                intent.request
                for intent in prepared.dispatch.final_requests
                if intent.engine_key == engine_name
            ),
            None,
        )
        # The request ID is unique but backend active-ID contracts still make
        # the handle single-use. A repeated caller safely falls back to fresh
        # preparation instead of racing the same GenerationRequest.
        prepared.dispatch.final_requests = ()
        return request

    def _take_prepared_initial_roles(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
        decision: RouteDecision,
    ) -> tuple[str | None, dict[str, GenerationRequest] | None]:
        """Consume exact dependency-free role requests bound by preflight."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or not prepared.dispatch.consumed
            or prepared.decision is None
            or prepared.decision.target != decision.target
            or decision.target != "multi_agent"
            or not prepared.dispatch.initial_requests
            or prepared.dispatch.conductor_session is None
        ):
            return None, None
        requests = {
            intent.role_name: intent.request
            for intent in prepared.dispatch.initial_requests
            if intent.role_name is not None
        }
        session = prepared.dispatch.conductor_session
        prepared.dispatch.initial_requests = ()
        prepared.dispatch.conductor_session = None
        return session, requests

    def _take_prepared_moa_setup(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
        decision: RouteDecision,
    ) -> _MoASetup | None:
        """Consume the exact proposal request set prepared for one MoA run."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or not prepared.dispatch.consumed
            or prepared.decision is None
            or prepared.decision.target != decision.target
            or decision.target != "multi_agent"
        ):
            return None
        setup = prepared.dispatch.moa_setup
        prepared.dispatch.moa_setup = None
        return setup

    def set_stage_observer(
        self,
        observer: Callable[[TraceEvent], None] | None,
    ) -> None:
        """Install the per-stage trace-event hook used for latency metrics."""

        self._stage_observer = observer

    @property
    def max_model_len(self) -> int | None:
        """Context length that every configured orchestration engine can serve.

        A public AUTO request may traverse multiple workers. Advertising the
        largest worker's context lets clients submit prompts that a mandatory
        smaller worker rejects. Unknown worker metadata likewise means there
        is no truthful global limit to publish.
        """

        lengths = [backend_max_model_len(engine) for engine in self._engines.values()]
        if not lengths or any(length is None for length in lengths):
            return None
        return min(length for length in lengths if length is not None)

    def admission_upper_bound(
        self,
        request: str | OrchestrationRequest,
    ) -> AdmissionUpperBound:
        """Worst-route compute ceiling for one public AUTO request.

        The bound belongs here rather than in the HTTP adapter because this
        object owns route fan-out, refine depth, MoA samples, and internal
        generation limits.  It includes every possible step plus triangular
        re-ingestion of preceding generated output into later prompts.
        """

        call = self._request(request)
        if self.will_judge_role_profile(call):
            assert self._profile_judge is not None
            judge_bound = backend_admission_upper_bound(
                self._engines[self._profile_judge.worker],
                self._profile_judge_request(call),
            )
            profile_bounds = (
                self.admission_upper_bound(replace(call, role_profile_judgment=profile))
                for profile in self._profiles
            )
            return AdmissionUpperBound(
                tokens=judge_bound.tokens + max(bound.tokens for bound in profile_bounds),
                refundable_on_exact_usage=False,
            )
        internal = self._internal_sampling_params(call)
        if internal.max_tokens is None:
            raise ValueError("AUTO tenant admission requires a finite internal limit")
        # An omitted public limit follows the OpenAI chat contract (bounded by
        # the serving model's context); reserve the largest engine context —
        # or the documented constant — instead of rejecting the request.
        public_output_fallback = (
            max(
                (getattr(engine, "max_model_len", None) or 0 for engine in self._engines.values()),
                default=0,
            )
            or UNLIMITED_OUTPUT_ADMISSION_TOKENS
        )
        steps = max(1, self._budget.max_steps, self._moa_samples + 1)
        candidates = max(
            call.sampling_params.n,
            call.sampling_params.best_of or call.sampling_params.n,
        )
        call_roles = self._roles_for(call)
        largest_role_prompt = max(
            call_roles,
            key=lambda role: len(role.prompt.encode("utf-8")),
            default=None,
        )
        role_prompt = largest_role_prompt.prompt if largest_role_prompt else ""
        role_bytes = len(role_prompt.encode("utf-8"))
        supplied_bytes = len(f"{self._shared_prefix}{call.prompt}".encode())
        stage_prompt = max(1, supplied_bytes + role_bytes + 256)
        internal_output = internal.max_tokens
        head = self._conductor_head_role(call_roles)
        if head is not None and head.sampling is not None and head.sampling.max_tokens:
            # The head streams public output whose role cap may exceed the
            # internal ceiling; keep the bound conservative.
            internal_output = max(internal_output, head.sampling.max_tokens)
        private_steps = max(0, steps - 1)
        private_work = (
            private_steps * (stage_prompt + internal_output)
            + internal_output * private_steps * (private_steps - 1) // 2
        )
        final_request_bound = generation_admission_upper_bound(
            GenerationRequest(
                request_id="auto-admission",
                prompt=f"{self._shared_prefix}{call.prompt}{role_prompt}",
                sampling_params=call.sampling_params,
                tools=call.tools,
                tool_choice=call.tool_choice,
                tools_in_prompt=call.tools_in_prompt,
                parallel_tool_calls=call.parallel_tool_calls,
                tool_call_protocol=call.tool_call_protocol,
                reasoning_effort=self._effective_reasoning_effort(call),
            ),
            fallback_output_tokens=public_output_fallback,
        )
        media_stage_bounds = [
            backend_admission_upper_bound(
                self._engines[intent.engine_key],
                intent.request,
            ).tokens
            for intent in self._internal_intent_requests(call, None)
            if call.multimodal_prompt is not None
            and backend_supports_prompt_kind(
                self._engines[intent.engine_key],
                "multimodal",
            )
        ]
        if media_stage_bounds:
            private_work = max(
                private_work,
                private_steps * max(media_stage_bounds),
            )
        # The last stage can ingest every preceding private output. Candidate
        # fan-out duplicates that prefill just as it duplicates the supplied
        # prompt, tools, response schema, and decode ceiling.
        final_reingestion = candidates * internal_output * private_steps
        return AdmissionUpperBound(
            tokens=private_work + final_request_bound.tokens + final_reingestion,
            refundable_on_exact_usage=False,
        )

    async def admission_upper_bound_async(
        self,
        request: str | OrchestrationRequest,
    ) -> AdmissionUpperBound:
        """Serialize tool/response intent on the bounded prompt CPU lane."""

        return await run_prompt_work(self.admission_upper_bound, request)

    @staticmethod
    def _generation_roles(roles: tuple[RoleSpec, ...]) -> tuple[RoleSpec, ...]:
        return tuple(role for role in roles if role.role_type != "executor")

    def _conductor_workers(
        self,
        roles: tuple[RoleSpec, ...],
        notes: list[str],
    ) -> dict[str, EngineBackend]:
        needed = {role.worker for role in self._generation_roles(roles)}
        return {name: self._resolve_engine(name, notes) for name in needed}

    @staticmethod
    def _conversation_affinity_key(
        call: OrchestrationRequest,
    ) -> str | None:
        """Stable KV-affinity key for consecutive turns of one conversation.

        Role-aware HTTP callers provide a hash of the immutable conversation
        opening. Transport-neutral legacy callers retain the fixed-prefix
        fallback because they provide only an opaque prompt string.
        """

        if call.conversation_affinity_key is not None:
            return call.conversation_affinity_key
        prompt = call.prompt
        if not isinstance(prompt, str) or not prompt:
            return None
        return "conv-" + hashlib.sha256(prompt[:2048].encode()).hexdigest()[:16]

    def _new_conductor(
        self,
        call: OrchestrationRequest,
        notes: list[str],
        *,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
    ) -> Conductor:
        """Construct the one exact role execution contract used by preflight/run."""

        roles = self._roles_for(call)
        return Conductor(
            roles=roles,
            workers=self._conductor_workers(roles, notes),
            shared_prefix=self._shared_prefix,
            sampling_params=self._internal_sampling_params(call),
            final_sampling_params=call.sampling_params,
            final_tools=call.tools,
            final_tool_choice=call.tool_choice,
            final_tools_in_prompt=call.tools_in_prompt,
            final_structured_format_in_prompt=call.structured_format_in_prompt,
            final_parallel_tool_calls=call.parallel_tool_calls,
            final_tool_call_protocol=call.tool_call_protocol,
            cost_model=self._cost_model,
            worker_trace=self._conductor_worker_trace(roles),
            usage_observer=usage_observer,
            stage_observer=self._stage_observer,
            affinity_key=self._conversation_affinity_key(call),
            expose_intermediate_outputs=self._expose_intermediate_outputs,
            multimodal_prompt=call.multimodal_prompt,
            chat_template_kwargs=call.chat_template_kwargs,
            execution_workers=self._execution_workers,
            reasoning_effort=self._effective_reasoning_effort(call),
            public_output_floor=self._profile_output_floor(roles),
        )

    def _effective_reasoning_effort(self, call: OrchestrationRequest) -> str | None:
        if call.reasoning_effort is not None:
            return call.reasoning_effort
        return self._default_reasoning_effort

    def _conductor_worker_trace(
        self,
        roles: tuple[RoleSpec, ...],
    ) -> dict[str, WorkerTraceIdentity]:
        fallback_name = next(iter(self._engines))
        needed = {role.worker for role in self._generation_roles(roles)}
        identities = {}
        for worker in needed:
            engine = worker if worker in self._engines else fallback_name
            descriptor = self._engine_descriptors[engine]
            identities[worker] = WorkerTraceIdentity(
                engine=engine,
                model=descriptor.model,
            )
        for role in roles:
            if role.role_type == "executor" and role.worker not in identities:
                identities[role.worker] = WorkerTraceIdentity(engine=role.worker)
        return identities

    def _route_trace_event(
        self,
        decision: RouteDecision,
        *,
        started_at: str,
        completed_at: str,
    ) -> TraceEvent:
        return TraceEvent(
            node="router",
            kind="route",
            detail=f"{decision.target} ({decision.reason})",
            operation="routing",
            status="success",
            role="router",
            timing=TraceTiming(
                started_at=started_at,
                completed_at=completed_at,
            ),
            metadata={
                "target": decision.target,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )

    async def _run_direct(
        self,
        call: OrchestrationRequest,
        tier: str,
        notes: list[str],
        *,
        prepared_request: GenerationRequest | None = None,
    ) -> tuple[GenerationResult, tuple[int, int, int], TraceEvent]:
        queued_at = utc_now_iso()
        engine_name = self._resolve_engine_name(tier, notes)
        engine = self._engines[engine_name]
        descriptor = self._engine_descriptors[engine_name]
        request = prepared_request
        if request is None:
            request = await run_prompt_work(
                self._direct_generation_request,
                call,
                f"direct-{uuid.uuid4().hex[:12]}",
                engine_name,
            )
        started_at = utc_now_iso()
        try:
            result = await engine.generate(request)
        except Exception as error:
            raise _DirectExecutionError(
                error,
                TraceEvent(
                    node=tier,
                    kind="failed",
                    operation="generation",
                    status="failed",
                    role="direct",
                    worker=tier,
                    engine=engine_name,
                    model=descriptor.model,
                    timing=TraceTiming(
                        queued_at=queued_at,
                        started_at=started_at,
                        completed_at=utc_now_iso(),
                    ),
                    error=TraceError(type=type(error).__name__),
                ),
            ) from error
        usage = (
            (
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.cached_tokens,
            )
            if result.usage is not None
            else (0, 0, 0)
        )
        trace_usage = (
            TraceUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cached_tokens=result.usage.cached_tokens,
            )
            if result.usage is not None
            else None
        )
        return (
            result,
            usage,
            TraceEvent(
                node=tier,
                kind="generated",
                operation="generation",
                status="success",
                role="direct",
                worker=tier,
                engine=engine_name,
                model=descriptor.model,
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                ),
                usage=trace_usage,
            ),
        )

    async def run(
        self,
        request: str | OrchestrationRequest,
        *,
        prepared: _PreparedOrchestration | None = None,
    ) -> OrchestratorResult:
        call = self._call_with_prepared_handle(request, prepared)
        query = call.prompt
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        judge_event = call.role_profile_judge_event
        judge_usage = self._profile_judge_usage(call)
        trace_started_at = _trace_started_at_from(judge_event)
        route_started_at = utc_now_iso()
        from kairyu.telemetry import traced_span

        with traced_span(
            "kairyu.route",
            {"kairyu.request_id": request_id},
        ) as route_span:
            decision = self._take_prepared_decision(prepared, call)
            if decision is None:
                decision = await self._route_async(query)
            self._validate_call(call, decision)
            if route_span is not None:
                route_span.set_attribute("kairyu.route.target", decision.target)
                route_span.set_attribute("kairyu.route.confidence", decision.confidence)
        route_completed_at = utc_now_iso()
        notes: list[str] = [f"route: {decision.target} ({decision.reason})"]
        if self._extra_profiles is not None and decision.target == "multi_agent":
            notes.append(f"role profile: {self._role_profile(call)}")
            if call.role_profile_judgment is not None:
                notes.append(f"profile judge: {call.role_profile_judgment}")
        trace_events = ([judge_event] if judge_event is not None else []) + [
            self._route_trace_event(
                decision,
                started_at=route_started_at,
                completed_at=route_completed_at,
            )
        ]

        def result_with_trace(
            *,
            text: str,
            completions: tuple[CompletionOutput, ...] = (),
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            cached_tokens: int = 0,
            reasoning_content: str | None = None,
            public_prompt_tokens: int | None = None,
            public_completion_tokens: int | None = None,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                completions=completions,
                prompt_tokens=judge_usage.prompt_tokens + prompt_tokens,
                completion_tokens=judge_usage.completion_tokens + completion_tokens,
                cached_tokens=judge_usage.cached_tokens + cached_tokens,
                reasoning_content=reasoning_content,
                public_prompt_tokens=public_prompt_tokens,
                public_completion_tokens=public_completion_tokens,
                structured_trace=StructuredTrace(
                    request_id=request_id,
                    started_at=trace_started_at,
                    completed_at=utc_now_iso(),
                    events=tuple(trace_events),
                ),
            )

        if decision.target == "multi_agent":
            if self._moa_samples > 0:  # m11 A4: the deep tier's MoA route
                from kairyu.orchestration.moa import (
                    MoAExecutionError,
                    _reuse_prepared_moa_setup,
                    run_moa,
                )

                moa_queued_at = utc_now_iso()
                budget_before = BudgetState(budget=self._budget)
                moa_steps = self._moa_samples + 1
                reservation = budget_before.try_reserve(
                    steps=moa_steps,
                    unknown_cost=True,
                )
                if reservation is None:
                    notes.append("moa: skipped:budget")
                    trace_events.append(
                        TraceEvent(
                            node="moa",
                            kind="skipped:budget",
                            operation="synthesis",
                            status="skipped",
                            role="moa",
                            budget=TraceBudget.between(
                                budget_before,
                                budget_before,
                            ),
                            metadata={"reason": "budget"},
                        )
                    )
                    return result_with_trace(text="")
                proposal_engine_name = self._resolve_engine_name("tier1", notes)
                synthesizer_engine_name = self._resolve_engine_name("tier2", notes)
                proposal_descriptor = self._engine_descriptors[proposal_engine_name]
                synthesizer_descriptor = self._engine_descriptors[synthesizer_engine_name]
                moa_started_at = utc_now_iso()
                moa = None
                prepared_moa = self._take_prepared_moa_setup(
                    prepared,
                    call,
                    decision,
                )
                try:
                    with _reuse_prepared_moa_setup(prepared_moa):
                        moa = await run_moa(
                            self._engines[proposal_engine_name],
                            query,
                            n_samples=self._moa_samples,
                            synthesizer=self._engines[synthesizer_engine_name],
                            sampling_params=self._internal_sampling_params(call),
                            final_sampling_params=call.sampling_params,
                            final_tools=call.tools,
                            final_tool_choice=call.tool_choice,
                            final_tools_in_prompt=call.tools_in_prompt,
                            final_parallel_tool_calls=call.parallel_tool_calls,
                            final_tool_call_protocol=call.tool_call_protocol,
                            shared_prefix=self._shared_prefix,
                            reasoning_effort=self._effective_reasoning_effort(call),
                        )
                    # M3: the deep MoA tier was invisible to the cost model / budget.
                    # Reconcile proposals + synthesis with the actual result cost;
                    # one admitted operation may visibly cross a result-priced cap.
                    moa_cost = self._cost_model(
                        GenerationRequest(
                            request_id="moa",
                            prompt=query,
                            sampling_params=call.sampling_params,
                            tools=call.tools,
                            tool_choice=call.tool_choice,
                            tools_in_prompt=call.tools_in_prompt,
                            parallel_tool_calls=call.parallel_tool_calls,
                            tool_call_protocol=call.tool_call_protocol,
                            reasoning_effort=self._effective_reasoning_effort(call),
                        ),
                        GenerationResult(
                            request_id="moa",
                            prompt=query,
                            completions=moa.completions
                            or (
                                CompletionOutput(
                                    index=0,
                                    text=moa.final_text,
                                    token_ids=(),
                                ),
                            ),
                            usage=GenerationUsage(
                                prompt_tokens=moa.usage[0],
                                completion_tokens=moa.usage[1],
                                cached_tokens=moa.cached_tokens,
                            ),
                        ),
                    )
                    budget_state = reservation.reconcile_success(
                        steps=moa_steps,
                        cost=moa_cost,
                        unknown_cost=True,
                    )
                except Exception as error:
                    released = reservation.release(
                        steps=moa_steps,
                        unknown_cost=True,
                    )
                    if not isinstance(error, MoAExecutionError):
                        raise
                    cause = error.cause
                    usage = error.usage
                    cached_tokens = error.cached_tokens
                    final_text = error.final_text
                    stage = error.stage
                    trace_events.append(
                        TraceEvent(
                            node="moa",
                            kind="failed",
                            operation="synthesis",
                            status="failed",
                            role="moa",
                            worker="tier1,tier2",
                            engine=",".join(
                                dict.fromkeys(
                                    (
                                        proposal_engine_name,
                                        synthesizer_engine_name,
                                    )
                                )
                            ),
                            model=",".join(
                                dict.fromkeys(
                                    model
                                    for model in (
                                        proposal_descriptor.model,
                                        synthesizer_descriptor.model,
                                    )
                                    if model is not None
                                )
                            )
                            or None,
                            timing=TraceTiming(
                                queued_at=moa_queued_at,
                                started_at=moa_started_at,
                                completed_at=utc_now_iso(),
                            ),
                            usage=TraceUsage(
                                prompt_tokens=usage[0],
                                completion_tokens=usage[1],
                                cached_tokens=cached_tokens,
                            ),
                            budget=TraceBudget.between(
                                budget_before,
                                released,
                            ),
                            metadata={"stage": stage},
                            error=TraceError(type=type(cause).__name__),
                        )
                    )
                    raise OrchestratorExecutionError(
                        cause,
                        result_with_trace(
                            text=final_text,
                            prompt_tokens=usage[0],
                            completion_tokens=usage[1],
                            cached_tokens=cached_tokens,
                        ),
                    ) from cause
                except BaseException:
                    reservation.release(steps=moa_steps, unknown_cost=True)
                    raise
                notes.append(
                    f"moa: {len(moa.proposals)} proposals synthesized (cost={moa_cost:.4f})"
                )
                if budget_state.is_exhausted:
                    notes.append("moa: budget exceeded")
                resolved_engines = tuple(
                    dict.fromkeys((proposal_engine_name, synthesizer_engine_name))
                )
                resolved_models = tuple(
                    dict.fromkeys(
                        model
                        for model in (
                            proposal_descriptor.model,
                            synthesizer_descriptor.model,
                        )
                        if model is not None
                    )
                )
                trace_events.append(
                    TraceEvent(
                        node="moa",
                        kind="synthesized",
                        operation="synthesis",
                        status="success",
                        role="moa",
                        worker="tier1,tier2",
                        engine=",".join(resolved_engines),
                        model=",".join(resolved_models) or None,
                        timing=TraceTiming(
                            queued_at=moa_queued_at,
                            started_at=moa_started_at,
                            completed_at=utc_now_iso(),
                        ),
                        usage=TraceUsage(
                            prompt_tokens=moa.usage[0],
                            completion_tokens=moa.usage[1],
                            cached_tokens=moa.cached_tokens,
                        ),
                        budget=TraceBudget.between(
                            budget_before,
                            budget_state,
                            steps_consumed=moa_steps,
                            cost_consumed_usd=moa_cost,
                        ),
                        metadata={
                            "proposals": len(moa.proposals),
                            "cost_usd": moa_cost,
                            "budget_exhausted": budget_state.is_exhausted,
                            "proposal_engine": proposal_engine_name,
                            "proposal_model": proposal_descriptor.model,
                            "synthesizer_engine": synthesizer_engine_name,
                            "synthesizer_model": synthesizer_descriptor.model,
                        },
                    )
                )
                return result_with_trace(
                    text=moa.final_text,
                    completions=_public_multistage_completions(
                        moa.completions
                        or (
                            CompletionOutput(
                                index=0,
                                text=moa.final_text,
                                token_ids=(),
                                finish_reason="stop",
                            ),
                        )
                    ),
                    prompt_tokens=moa.usage[0],
                    completion_tokens=moa.usage[1],
                    cached_tokens=moa.cached_tokens,
                )
            conductor = self._new_conductor(call, notes)
            conductor_session, initial_requests = self._take_prepared_initial_roles(
                prepared, call, decision
            )
            result = await conductor.run(
                query,
                budget=self._budget,
                session=conductor_session,
                prepared_initial_requests=initial_requests,
            )
            notes.extend(f"{event.node}: {event.kind} {event.detail}" for event in result.trace)
            trace_events.extend(result.trace)
            if not result.final_unit_ok and not result.final_text:
                # The selected final unit failed or produced no public text
                # after its retry; publishing an internal stage or an empty
                # "stop" would be a silent lie to the caller (issue #496).
                raise OrchestratorExecutionError(
                    RuntimeError("orchestration final unit produced no public output"),
                    result_with_trace(
                        text="",
                        prompt_tokens=result.usage[0],
                        completion_tokens=result.usage[1],
                        cached_tokens=result.cached_tokens,
                        reasoning_content=result.reasoning_content,
                    ),
                )
            return result_with_trace(
                text=result.final_text,
                completions=_public_multistage_completions(
                    result.completions,
                    result.reasoning_content,
                ),
                prompt_tokens=result.usage[0],
                completion_tokens=result.usage[1],
                cached_tokens=result.cached_tokens,
                reasoning_content=result.reasoning_content,
                public_completion_tokens=result.public_completion_tokens,
            )
        try:
            engine_name = (
                decision.target if decision.target in self._engines else next(iter(self._engines))
            )
            direct_result, usage, direct_event = await self._run_direct(
                call,
                decision.target,
                notes,
                prepared_request=self._prepared_direct_request(
                    prepared,
                    call,
                    decision,
                    engine_name,
                ),
            )
        except _DirectExecutionError as error:
            trace_events.append(error.event)
            if (
                decision.target == "tier1"
                and "tier2" in self._engines
                and (
                    call.multimodal_prompt is None
                    or backend_supports_prompt_kind(
                        self._engines["tier2"],
                        "multimodal",
                    )
                )
            ):
                notes.append("failover: tier1 failed; retrying once on tier2")
                try:
                    direct_result, usage, direct_event = await self._run_direct(
                        call,
                        "tier2",
                        notes,
                    )
                except _DirectExecutionError as failover_error:
                    trace_events.append(failover_error.event)
                    raise OrchestratorExecutionError(
                        failover_error.cause,
                        result_with_trace(text=""),
                    ) from failover_error.cause
            else:
                raise OrchestratorExecutionError(
                    error.cause,
                    result_with_trace(text=""),
                ) from error.cause
        trace_events.append(direct_event)
        return result_with_trace(
            text=direct_result.text,
            completions=direct_result.completions,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            cached_tokens=usage[2],
            # A direct route is one public call: its reported usage IS the
            # client-visible view (0 = unreported → m9 fallback).
            public_prompt_tokens=usage[0] or None,
            public_completion_tokens=usage[1] or None,
        )

    async def run_chat(
        self,
        request: str | OrchestrationRequest,
        stream: bool = False,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
        *,
        prepared: _PreparedOrchestration | None = None,
    ):
        """The m11 D1 surface: pre-rendered prompt in (A4), events out.

        Non-stream: returns OrchestratorResult (same as run()). Stream:
        async-yields OrchestratorEvent — status keep-alives while pre-final
        stages run, token deltas pulled directly from every route's FINAL
        worker/synthesizer, then one result event.
        """
        if not stream:
            return await self.run(request, prepared=prepared)
        return self._run_chat_stream(
            self._call_with_prepared_handle(request, prepared),
            usage_observer=usage_observer,
            prepared=prepared,
        )

    async def _with_initial_keepalives(self, stream) -> AsyncIterator[object | None]:
        """Emit timer sentinels until the first final-stage event is available.

        Only the first ``anext`` is task-backed so long pre-final work can be
        multiplexed with keep-alives.  Once the final backend produces its
        first event, all remaining deltas are pulled through in the caller's
        task without a per-token task or queue.
        """

        iterator = stream.__aiter__()
        first = asyncio.ensure_future(anext(iterator))
        try:
            while not first.done():
                done, _ = await asyncio.wait(
                    {first},
                    timeout=_KEEPALIVE_INTERVAL_S,
                )
                if not done:
                    yield None
            try:
                yield first.result()
            except StopAsyncIteration:
                return
            async for event in iterator:
                yield event
        finally:
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _with_keepalives(self, stream) -> AsyncIterator[object | None]:
        """Emit timer sentinels whenever the next event is slow to arrive.

        The head/incremental-reasoning Conductor stream produces events
        throughout the DAG's lifetime with potentially long gaps between
        stages, so every ``anext`` is task-backed (EO-D8).  The per-event task
        cost is microseconds against a multi-millisecond token cadence; the
        latency-critical direct route keeps its pull-through loop.
        """

        iterator = stream.__aiter__()
        try:
            while True:
                upcoming = asyncio.ensure_future(anext(iterator))
                try:
                    while not upcoming.done():
                        done, _ = await asyncio.wait(
                            {upcoming},
                            timeout=_KEEPALIVE_INTERVAL_S,
                        )
                        if not done:
                            yield None
                    try:
                        yield upcoming.result()
                    except StopAsyncIteration:
                        return
                finally:
                    if not upcoming.done():
                        upcoming.cancel()
                        await asyncio.gather(upcoming, return_exceptions=True)
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _run_chat_stream(
        self,
        call: OrchestrationRequest,
        *,
        usage_observer: Callable[[GenerationUsage], None] | None,
        prepared: _PreparedOrchestration | None,
    ):
        prompt = call.prompt
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        judge_event = call.role_profile_judge_event
        judge_usage = self._profile_judge_usage(call)
        trace_started_at = _trace_started_at_from(judge_event)
        external_usage_observer = usage_observer
        if external_usage_observer is not None:
            if judge_usage.prompt_tokens or judge_usage.completion_tokens:
                external_usage_observer(judge_usage)

            def observe_with_profile_judge(usage: GenerationUsage) -> None:
                external_usage_observer(self._usage_with_profile_judge(call, usage))

            usage_observer = observe_with_profile_judge
        route_started_at = utc_now_iso()
        from kairyu.telemetry import traced_span

        with traced_span(
            "kairyu.route",
            {"kairyu.request_id": request_id},
        ) as route_span:
            decision = self._take_prepared_decision(prepared, call)
            if decision is None:
                decision = await self._route_async(prompt)
            self._validate_call(call, decision)
            if route_span is not None:
                route_span.set_attribute("kairyu.route.target", decision.target)
                route_span.set_attribute("kairyu.route.confidence", decision.confidence)
        route_completed_at = utc_now_iso()
        notes = [f"route: {decision.target} ({decision.reason})"]
        if self._extra_profiles is not None and decision.target == "multi_agent":
            notes.append(f"role profile: {self._role_profile(call)}")
            if call.role_profile_judgment is not None:
                notes.append(f"profile judge: {call.role_profile_judgment}")
        trace_events = ([judge_event] if judge_event is not None else []) + [
            self._route_trace_event(
                decision,
                started_at=route_started_at,
                completed_at=route_completed_at,
            )
        ]

        def result_with_trace(
            *,
            text: str,
            completions: tuple[CompletionOutput, ...] = (),
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            cached_tokens: int = 0,
            reasoning_content: str | None = None,
            public_prompt_tokens: int | None = None,
            public_completion_tokens: int | None = None,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                completions=completions,
                prompt_tokens=judge_usage.prompt_tokens + prompt_tokens,
                completion_tokens=judge_usage.completion_tokens + completion_tokens,
                cached_tokens=judge_usage.cached_tokens + cached_tokens,
                reasoning_content=reasoning_content,
                public_prompt_tokens=public_prompt_tokens,
                public_completion_tokens=public_completion_tokens,
                structured_trace=StructuredTrace(
                    request_id=request_id,
                    started_at=trace_started_at,
                    completed_at=utc_now_iso(),
                    events=tuple(trace_events),
                ),
            )

        if decision.target != "multi_agent":
            queued_at = utc_now_iso()
            engine_name = self._resolve_engine_name(decision.target, notes)
            engine = self._engines[engine_name]
            descriptor = self._engine_descriptors[engine_name]
            request = self._prepared_direct_request(
                prepared,
                call,
                decision,
                engine_name,
            )
            if request is None:
                request = await run_prompt_work(
                    self._direct_generation_request,
                    call,
                    f"direct-{uuid.uuid4().hex[:12]}",
                    engine_name,
                )
            last = None
            latest_usage = None
            started_at = utc_now_iso()
            first_token_at = None
            emitted = 0
            text_parts: list[str] = []
            try:
                async for partial in engine.stream(request):
                    last = partial
                    if partial.usage is not None:
                        latest_usage = partial.usage
                        if usage_observer is not None:
                            usage_observer(latest_usage)
                    completion = partial.completions[0] if partial.completions else None
                    delta, emitted = (
                        completion.delta_after(emitted) if completion is not None else ("", emitted)
                    )
                    text_parts.append(delta)
                    if first_token_at is None and partial.completions:
                        first_token_at = utc_now_iso()
                    yield OrchestratorEvent(
                        kind="delta",
                        text=delta,
                        completions=partial.completions,
                    )
                if last is None:
                    raise RuntimeError("direct stream produced no result")
            except Exception as error:
                trace_usage = (
                    TraceUsage(
                        prompt_tokens=latest_usage.prompt_tokens,
                        completion_tokens=latest_usage.completion_tokens,
                        cached_tokens=latest_usage.cached_tokens,
                    )
                    if latest_usage is not None
                    else None
                )
                trace_events.append(
                    TraceEvent(
                        node=decision.target,
                        kind="failed",
                        operation="generation",
                        status="failed",
                        role="direct",
                        worker=decision.target,
                        engine=engine_name,
                        model=descriptor.model,
                        timing=TraceTiming(
                            queued_at=queued_at,
                            started_at=started_at,
                            first_token_at=first_token_at,
                            completed_at=utc_now_iso(),
                        ),
                        usage=trace_usage,
                        metadata={"streamed": True},
                        error=TraceError(type=type(error).__name__),
                    )
                )
                if (
                    decision.target == "tier1"
                    and "tier2" in self._engines
                    and not "".join(text_parts)
                    and (
                        call.multimodal_prompt is None
                        or backend_supports_prompt_kind(
                            self._engines["tier2"],
                            "multimodal",
                        )
                    )
                ):
                    notes.append("failover: tier1 failed before output; retrying once on tier2")
                    failover_queued_at = utc_now_iso()
                    failover_engine = self._engines["tier2"]
                    failover_descriptor = self._engine_descriptors["tier2"]
                    failover_request = await run_prompt_work(
                        self._direct_generation_request,
                        call,
                        f"direct-{uuid.uuid4().hex[:12]}",
                        "tier2",
                    )
                    failover_last = None
                    failover_usage = None
                    failover_started_at = utc_now_iso()
                    failover_first_token_at = None
                    failover_emitted = 0
                    failover_text_parts: list[str] = []
                    try:
                        async for partial in failover_engine.stream(failover_request):
                            failover_last = partial
                            if partial.usage is not None:
                                failover_usage = partial.usage
                                if usage_observer is not None:
                                    usage_observer(failover_usage)
                            completion = partial.completions[0] if partial.completions else None
                            delta, failover_emitted = (
                                completion.delta_after(failover_emitted)
                                if completion is not None
                                else ("", failover_emitted)
                            )
                            failover_text_parts.append(delta)
                            if failover_first_token_at is None and partial.completions:
                                failover_first_token_at = utc_now_iso()
                            yield OrchestratorEvent(
                                kind="delta",
                                text=delta,
                                completions=partial.completions,
                            )
                        if failover_last is None:
                            raise RuntimeError("tier2 failover stream produced no result")
                    except Exception as failover_error:
                        failover_trace_usage = (
                            TraceUsage(
                                prompt_tokens=failover_usage.prompt_tokens,
                                completion_tokens=failover_usage.completion_tokens,
                                cached_tokens=failover_usage.cached_tokens,
                            )
                            if failover_usage is not None
                            else None
                        )
                        trace_events.append(
                            TraceEvent(
                                node="tier2",
                                kind="failed",
                                operation="generation",
                                status="failed",
                                role="direct",
                                worker="tier2",
                                engine="tier2",
                                model=failover_descriptor.model,
                                timing=TraceTiming(
                                    queued_at=failover_queued_at,
                                    started_at=failover_started_at,
                                    first_token_at=failover_first_token_at,
                                    completed_at=utc_now_iso(),
                                ),
                                usage=failover_trace_usage,
                                metadata={"streamed": True, "failover": True},
                                error=TraceError(type=type(failover_error).__name__),
                            )
                        )
                        usage = failover_usage or latest_usage or GenerationUsage()
                        partial_result = result_with_trace(
                            text="".join(failover_text_parts),
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            cached_tokens=usage.cached_tokens,
                        )
                        yield OrchestratorEvent(
                            kind="error",
                            result=partial_result,
                            error_type=type(failover_error).__name__,
                        )
                        return
                    failover_trace_usage = (
                        TraceUsage(
                            prompt_tokens=failover_usage.prompt_tokens,
                            completion_tokens=failover_usage.completion_tokens,
                            cached_tokens=failover_usage.cached_tokens,
                        )
                        if failover_usage is not None
                        else None
                    )
                    trace_events.append(
                        TraceEvent(
                            node="tier2",
                            kind="generated",
                            operation="generation",
                            status="success",
                            role="direct",
                            worker="tier2",
                            engine="tier2",
                            model=failover_descriptor.model,
                            timing=TraceTiming(
                                queued_at=failover_queued_at,
                                started_at=failover_started_at,
                                first_token_at=failover_first_token_at,
                                completed_at=utc_now_iso(),
                            ),
                            usage=failover_trace_usage,
                            metadata={"streamed": True, "failover": True},
                        )
                    )
                    usage = failover_usage or GenerationUsage()
                    yield OrchestratorEvent(
                        kind="result",
                        result=result_with_trace(
                            text=(failover_last.text or "".join(failover_text_parts)),
                            completions=failover_last.completions,
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            cached_tokens=usage.cached_tokens,
                            public_prompt_tokens=usage.prompt_tokens or None,
                            public_completion_tokens=usage.completion_tokens or None,
                        ),
                    )
                    return
                usage = latest_usage or GenerationUsage()
                partial_result = result_with_trace(
                    text="".join(text_parts),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                )
                yield OrchestratorEvent(
                    kind="error",
                    result=partial_result,
                    error_type=type(error).__name__,
                )
                return
            trace_usage = (
                TraceUsage(
                    prompt_tokens=latest_usage.prompt_tokens,
                    completion_tokens=latest_usage.completion_tokens,
                    cached_tokens=latest_usage.cached_tokens,
                )
                if latest_usage is not None
                else None
            )
            trace_events.append(
                TraceEvent(
                    node=decision.target,
                    kind="generated",
                    operation="generation",
                    status="success",
                    role="direct",
                    worker=decision.target,
                    engine=engine_name,
                    model=descriptor.model,
                    timing=TraceTiming(
                        queued_at=queued_at,
                        started_at=started_at,
                        first_token_at=first_token_at,
                        completed_at=utc_now_iso(),
                    ),
                    usage=trace_usage,
                    metadata={"streamed": True},
                )
            )
            usage = (
                (
                    latest_usage.prompt_tokens,
                    latest_usage.completion_tokens,
                    latest_usage.cached_tokens,
                )
                if latest_usage is not None
                else (0, 0, 0)
            )
            yield OrchestratorEvent(
                kind="result",
                result=result_with_trace(
                    text=((last.text or "".join(text_parts)) if last is not None else ""),
                    completions=last.completions if last is not None else (),
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    cached_tokens=usage[2],
                    public_prompt_tokens=usage[0] or None,
                    public_completion_tokens=usage[1] or None,
                ),
            )
            return

        # Multi-stage routes emit comments while the pre-final DAG/proposals and
        # first final token are pending.  Final deltas then pull through from
        # Conductor/MoA without a queue bridge.
        yield OrchestratorEvent(kind="status", text=f"routing: {decision.target}")
        if self._moa_samples > 0:
            from kairyu.orchestration.moa import (
                MoAExecutionError,
                _reuse_prepared_moa_setup,
                stream_moa,
            )

            moa_queued_at = utc_now_iso()
            budget_before = BudgetState(budget=self._budget)
            moa_steps = self._moa_samples + 1
            reservation = budget_before.try_reserve(
                steps=moa_steps,
                unknown_cost=True,
            )
            if reservation is None:
                notes.append("moa: skipped:budget")
                trace_events.append(
                    TraceEvent(
                        node="moa",
                        kind="skipped:budget",
                        operation="synthesis",
                        status="skipped",
                        role="moa",
                        budget=TraceBudget.between(
                            budget_before,
                            budget_before,
                        ),
                        metadata={"reason": "budget"},
                    )
                )
                yield OrchestratorEvent(
                    kind="result",
                    result=result_with_trace(text=""),
                )
                return

            proposal_engine_name = self._resolve_engine_name("tier1", notes)
            synthesizer_engine_name = self._resolve_engine_name("tier2", notes)
            proposal_descriptor = self._engine_descriptors[proposal_engine_name]
            synthesizer_descriptor = self._engine_descriptors[synthesizer_engine_name]
            moa_started_at = utc_now_iso()
            moa_result = None
            latest_usage = GenerationUsage()
            first_token_at = None
            prepared_moa = self._take_prepared_moa_setup(
                prepared,
                call,
                decision,
            )

            def observe_moa_usage(usage: GenerationUsage) -> None:
                nonlocal latest_usage
                latest_usage = usage
                if usage_observer is not None:
                    usage_observer(usage)

            try:
                with _reuse_prepared_moa_setup(prepared_moa):
                    moa_stream = stream_moa(
                        self._engines[proposal_engine_name],
                        prompt,
                        n_samples=self._moa_samples,
                        synthesizer=self._engines[synthesizer_engine_name],
                        sampling_params=self._internal_sampling_params(call),
                        final_sampling_params=call.sampling_params,
                        final_tools=call.tools,
                        final_tool_choice=call.tool_choice,
                        final_tools_in_prompt=call.tools_in_prompt,
                        final_parallel_tool_calls=call.parallel_tool_calls,
                        final_tool_call_protocol=call.tool_call_protocol,
                        shared_prefix=self._shared_prefix,
                        usage_observer=observe_moa_usage,
                        reasoning_effort=self._effective_reasoning_effort(call),
                    )
                async for event in self._with_initial_keepalives(moa_stream):
                    if event is None:
                        yield OrchestratorEvent(kind="status", text="working")
                    elif event.kind == "delta":
                        if not event.text:
                            continue
                        if first_token_at is None:
                            first_token_at = utc_now_iso()
                        yield OrchestratorEvent(
                            kind="delta",
                            text=event.text,
                            completions=_public_multistage_completions(event.completions),
                        )
                    else:
                        moa_result = event.result
                if moa_result is None:
                    raise RuntimeError("MoA stream did not produce a final result")
                moa_cost = self._cost_model(
                    GenerationRequest(
                        request_id="moa",
                        prompt=prompt,
                        sampling_params=call.sampling_params,
                        tools=call.tools,
                        tool_choice=call.tool_choice,
                        tools_in_prompt=call.tools_in_prompt,
                        parallel_tool_calls=call.parallel_tool_calls,
                        tool_call_protocol=call.tool_call_protocol,
                        reasoning_effort=self._effective_reasoning_effort(call),
                    ),
                    GenerationResult(
                        request_id="moa",
                        prompt=prompt,
                        completions=moa_result.completions
                        or (
                            CompletionOutput(
                                index=0,
                                text=moa_result.final_text,
                                token_ids=(),
                            ),
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=moa_result.usage[0],
                            completion_tokens=moa_result.usage[1],
                            cached_tokens=moa_result.cached_tokens,
                        ),
                    ),
                )
                budget_state = reservation.reconcile_success(
                    steps=moa_steps,
                    cost=moa_cost,
                    unknown_cost=True,
                )
            except Exception as error:
                released = reservation.release(
                    steps=moa_steps,
                    unknown_cost=True,
                )
                if isinstance(error, MoAExecutionError):
                    cause = error.cause
                    usage = error.usage
                    cached_tokens = error.cached_tokens
                    partial_text = error.final_text
                    stage = error.stage
                else:
                    cause = error
                    usage = (
                        moa_result.usage
                        if moa_result is not None
                        else (
                            latest_usage.prompt_tokens,
                            latest_usage.completion_tokens,
                        )
                    )
                    cached_tokens = (
                        moa_result.cached_tokens
                        if moa_result is not None
                        else latest_usage.cached_tokens
                    )
                    partial_text = moa_result.final_text if moa_result is not None else ""
                    stage = "accounting"
                trace_events.append(
                    TraceEvent(
                        node="moa",
                        kind="failed",
                        operation="synthesis",
                        status="failed",
                        role="moa",
                        worker="tier1,tier2",
                        engine=",".join(
                            dict.fromkeys(
                                (
                                    proposal_engine_name,
                                    synthesizer_engine_name,
                                )
                            )
                        ),
                        model=",".join(
                            dict.fromkeys(
                                model
                                for model in (
                                    proposal_descriptor.model,
                                    synthesizer_descriptor.model,
                                )
                                if model is not None
                            )
                        )
                        or None,
                        timing=TraceTiming(
                            queued_at=moa_queued_at,
                            started_at=moa_started_at,
                            first_token_at=first_token_at,
                            completed_at=utc_now_iso(),
                        ),
                        usage=TraceUsage(
                            prompt_tokens=usage[0],
                            completion_tokens=usage[1],
                            cached_tokens=cached_tokens,
                        ),
                        budget=TraceBudget.between(
                            budget_before,
                            released,
                        ),
                        metadata={"stage": stage, "streamed": True},
                        error=TraceError(type=type(cause).__name__),
                    )
                )
                yield OrchestratorEvent(
                    kind="error",
                    result=result_with_trace(
                        text=partial_text,
                        prompt_tokens=usage[0],
                        completion_tokens=usage[1],
                        cached_tokens=cached_tokens,
                    ),
                    error_type=type(cause).__name__,
                )
                return
            except BaseException:
                reservation.release(steps=moa_steps, unknown_cost=True)
                raise

            notes.append(
                f"moa: {len(moa_result.proposals)} proposals synthesized (cost={moa_cost:.4f})"
            )
            if budget_state.is_exhausted:
                notes.append("moa: budget exceeded")
            resolved_engines = tuple(dict.fromkeys((proposal_engine_name, synthesizer_engine_name)))
            resolved_models = tuple(
                dict.fromkeys(
                    model
                    for model in (
                        proposal_descriptor.model,
                        synthesizer_descriptor.model,
                    )
                    if model is not None
                )
            )
            trace_events.append(
                TraceEvent(
                    node="moa",
                    kind="synthesized",
                    operation="synthesis",
                    status="success",
                    role="moa",
                    worker="tier1,tier2",
                    engine=",".join(resolved_engines),
                    model=",".join(resolved_models) or None,
                    timing=TraceTiming(
                        queued_at=moa_queued_at,
                        started_at=moa_started_at,
                        first_token_at=first_token_at,
                        completed_at=utc_now_iso(),
                    ),
                    usage=TraceUsage(
                        prompt_tokens=moa_result.usage[0],
                        completion_tokens=moa_result.usage[1],
                        cached_tokens=moa_result.cached_tokens,
                    ),
                    budget=TraceBudget.between(
                        budget_before,
                        budget_state,
                        steps_consumed=moa_steps,
                        cost_consumed_usd=moa_cost,
                    ),
                    metadata={
                        "proposals": len(moa_result.proposals),
                        "cost_usd": moa_cost,
                        "budget_exhausted": budget_state.is_exhausted,
                        "proposal_engine": proposal_engine_name,
                        "proposal_model": proposal_descriptor.model,
                        "synthesizer_engine": synthesizer_engine_name,
                        "synthesizer_model": synthesizer_descriptor.model,
                        "streamed": True,
                    },
                )
            )
            yield OrchestratorEvent(
                kind="result",
                result=result_with_trace(
                    text=moa_result.final_text,
                    completions=_public_multistage_completions(
                        moa_result.completions
                        or (
                            CompletionOutput(
                                index=0,
                                text=moa_result.final_text,
                                token_ids=(),
                                finish_reason="stop",
                            ),
                        )
                    ),
                    prompt_tokens=moa_result.usage[0],
                    completion_tokens=moa_result.usage[1],
                    cached_tokens=moa_result.cached_tokens,
                ),
            )
            return

        conductor = self._new_conductor(
            call,
            notes,
            usage_observer=usage_observer,
        )
        conductor_session, initial_requests = self._take_prepared_initial_roles(
            prepared,
            call,
            decision,
        )
        conductor_result = None
        try:
            async for event in self._with_keepalives(
                conductor.stream(
                    prompt,
                    budget=self._budget,
                    session=conductor_session,
                    prepared_initial_requests=initial_requests,
                )
            ):
                if event is None:
                    yield OrchestratorEvent(kind="status", text="working")
                elif event.kind == "reasoning":
                    if event.text:
                        yield OrchestratorEvent(kind="reasoning", text=event.text)
                elif event.kind == "delta":
                    if not event.text:
                        continue
                    yield OrchestratorEvent(
                        kind="delta",
                        text=event.text,
                        completions=_public_multistage_completions(event.completions),
                    )
                else:
                    conductor_result = event.result
        except ConductorStreamError as error:
            conductor_result = error.result
            notes.extend(
                f"{event.node}: {event.kind} {event.detail}" for event in conductor_result.trace
            )
            trace_events.extend(conductor_result.trace)
            yield OrchestratorEvent(
                kind="error",
                result=result_with_trace(
                    text=conductor_result.final_text,
                    completions=_public_multistage_completions(
                        conductor_result.completions,
                        conductor_result.reasoning_content,
                    ),
                    prompt_tokens=conductor_result.usage[0],
                    completion_tokens=conductor_result.usage[1],
                    cached_tokens=conductor_result.cached_tokens,
                    reasoning_content=conductor_result.reasoning_content,
                ),
                error_type=type(error.cause).__name__,
            )
            return
        if conductor_result is None:
            raise RuntimeError("Conductor stream did not produce a final result")
        notes.extend(
            f"{event.node}: {event.kind} {event.detail}" for event in conductor_result.trace
        )
        trace_events.extend(conductor_result.trace)
        if not conductor_result.final_unit_ok and not conductor_result.final_text:
            # Same contract as the unary route (issue #496): an empty final
            # unit becomes an explicit error event, never a silent empty stop.
            yield OrchestratorEvent(
                kind="error",
                result=result_with_trace(
                    text="",
                    prompt_tokens=conductor_result.usage[0],
                    completion_tokens=conductor_result.usage[1],
                    cached_tokens=conductor_result.cached_tokens,
                    reasoning_content=conductor_result.reasoning_content,
                ),
                error_type="EmptyFinalOutput",
            )
            return
        yield OrchestratorEvent(
            kind="result",
            result=result_with_trace(
                text=conductor_result.final_text,
                completions=_public_multistage_completions(
                    conductor_result.completions,
                    conductor_result.reasoning_content,
                ),
                prompt_tokens=conductor_result.usage[0],
                completion_tokens=conductor_result.usage[1],
                cached_tokens=conductor_result.cached_tokens,
                reasoning_content=conductor_result.reasoning_content,
                public_completion_tokens=conductor_result.public_completion_tokens,
            ),
        )

    def run_sync(self, query: str) -> OrchestratorResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(query))
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; await run() instead"
        )

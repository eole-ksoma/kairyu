import pytest
from pydantic import ValidationError

from kairyu.dsl.decorators import AgentPool
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams

YAML_SPEC = """
shared_prefix: "SYS\\n"
moa_samples: 3
internal_max_tokens: 512
workers:
  - name: tier1
    backend: mock
  - name: tier2
    backend: mock
    options:
      responses:
        "[verifier]": "PASS"
roles:
  - name: planner
    worker: tier2
    prompt: "plan: {query}"
  - name: worker
    worker: tier1
    prompt: "do: {planner}"
    depends_on: [planner]
budget:
  max_steps: 8
  max_refine_depth: 1
"""


def test_yaml_spec_round_trip():
    spec = load_spec(YAML_SPEC)
    assert [w.name for w in spec.workers] == ["tier1", "tier2"]
    assert spec.workers[1].options["responses"]["[verifier]"] == "PASS"
    assert spec.roles[1].depends_on == ("planner",)
    assert spec.budget.max_steps == 8
    assert spec.shared_prefix == "SYS\n"
    assert spec.moa_samples == 3
    assert spec.internal_max_tokens == 512
    assert spec.expose_intermediate_outputs is False


def test_intermediate_output_visibility_is_explicit():
    spec = load_spec(
        """
workers: [{name: tier1, backend: mock}]
expose_intermediate_outputs: true
"""
    )
    assert spec.expose_intermediate_outputs is True
    assert build_orchestrator(spec).describe_routing()[
        "expose_intermediate_outputs"
    ] is True


def test_yaml_spec_from_file(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(YAML_SPEC)
    assert load_spec(path).budget.max_refine_depth == 1


def test_role_referencing_unknown_worker_rejected():
    bad = YAML_SPEC.replace("worker: tier1", "worker: ghost")
    with pytest.raises(ValidationError, match="ghost"):
        load_spec(bad)


def test_orchestrator_requires_at_least_one_worker():
    with pytest.raises(ValidationError, match="workers"):
        load_spec("workers: []")


def test_duplicate_worker_names_are_rejected():
    with pytest.raises(ValidationError, match="worker names must be unique"):
        load_spec(
            """
workers:
  - {name: repeated, backend: mock}
  - {name: repeated, backend: mock}
"""
        )


@pytest.mark.parametrize(
    "factory_field",
    (
        "backend: mock",
        "model: internal",
        "base_url: http://internal/v1",
        "api_key_env: KEY",
        "options: {}",
    ),
)
def test_engine_ref_rejects_factory_fields(factory_field):
    with pytest.raises(ValidationError, match="engine_ref workers"):
        load_spec(
            f"""
workers:
  - name: tier1
    engine_ref: internal
    {factory_field}
"""
        )


async def test_engine_ref_dispatches_to_borrowed_object_without_factory(monkeypatch):
    borrowed = MockBackend(responses={"hello": "borrowed answer"})

    def unexpected_factory(*_args, **_kwargs):
        raise AssertionError("engine_ref must not construct a backend")

    monkeypatch.setattr("kairyu.dsl.loader.create_backend", unexpected_factory)
    orchestrator = build_orchestrator(
        load_spec("workers: [{name: tier1, engine_ref: internal}]"),
        engine_refs={"internal": borrowed},
    )

    result = await orchestrator.run("hello")

    assert result.text == "borrowed answer"
    assert borrowed.prompts_seen
    assert orchestrator.describe_routing()["configured_engines"] == {
        "tier1": {"backend_type": "MockBackend", "model": "internal"}
    }


def test_engine_ref_rejects_unknown_deployment_engine():
    with pytest.raises(ValueError, match="unknown deployment engine 'missing'"):
        build_orchestrator(
            load_spec("workers: [{name: tier1, engine_ref: missing}]"),
            engine_refs={},
        )


def test_decorator_pool_builds_equivalent_spec():
    pool = AgentPool(shared_prefix="SYS\n", moa_samples=3, internal_max_tokens=512)
    pool.worker("tier1", backend="mock")
    pool.worker("tier2", backend="mock", options={"responses": {"[verifier]": "PASS"}})
    pool.budget(max_steps=8, max_refine_depth=1)

    @pool.role(worker="tier2")
    def planner():
        return "plan: {query}"

    @pool.role(worker="tier1", depends_on=("planner",))
    def worker():
        return "do: {planner}"

    assert pool.to_spec() == load_spec(YAML_SPEC)


async def test_build_orchestrator_runs_end_to_end():
    orchestrator = build_orchestrator(load_spec(YAML_SPEC))
    descriptor = orchestrator.describe_routing()
    assert descriptor["configured_engines"] == {
        "tier1": {"backend_type": "mock", "model": None},
        "tier2": {"backend_type": "mock", "model": None},
    }
    assert descriptor["moa_samples"] == 3
    assert descriptor["internal_max_tokens"] == 512
    assert descriptor["target_resolution"]["multi_agent"]["mode"] == "moa"
    result = await orchestrator.run(
        "First, plan the work. Then execute it. Finally, summarize the outcome."
    )
    assert result.route.target == "multi_agent"
    assert result.text


def test_rule_router_thresholds_are_configurable_and_validated():
    spec = load_spec(
        """
workers:
  - {name: tier1, backend: mock}
  - {name: tier2, backend: mock}
router:
  kind: rules
  thresholds:
    multi_step_markers: 8
    multi_agent_min_chars: 32768
    reasoning_keywords: 6
    math_symbols: 64
    tier2_min_chars: 12288
"""
    )
    descriptor = build_orchestrator(spec).describe_routing()
    assert descriptor["router"]["thresholds"] == {
        "multi_step_markers": 8,
        "multi_agent_min_chars": 32768,
        "reasoning_keywords": 6,
        "math_symbols": 64,
        "tier2_min_chars": 12288,
    }

    with pytest.raises(ValidationError, match="multi_agent_min_chars"):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
router:
  kind: rules
  thresholds: {multi_agent_min_chars: 0}
"""
        )


def test_calibrated_router_rejects_inline_thresholds():
    with pytest.raises(ValidationError, match="pinned artifact"):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
router:
  kind: calibrated
  artifact: router.json
  sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  thresholds: {tier2_min_chars: 4096}
"""
        )


@pytest.mark.parametrize("value", [-1, 17])
def test_moa_samples_rejects_unsafe_values(value):
    with pytest.raises(ValidationError, match="moa_samples"):
        load_spec(YAML_SPEC.replace("moa_samples: 3", f"moa_samples: {value}"))


@pytest.mark.parametrize("value", [0, 131073])
def test_internal_max_tokens_rejects_unsafe_values(value):
    with pytest.raises(ValidationError, match="internal_max_tokens"):
        load_spec(
            YAML_SPEC.replace("internal_max_tokens: 512", f"internal_max_tokens: {value}")
        )


@pytest.mark.parametrize("value", [0, 131073])
def test_public_output_floor_rejects_unsafe_values(value):
    with pytest.raises(ValidationError, match="public_output_floor"):
        load_spec(
            f"""
workers: [{{name: tier1, backend: mock}}]
public_output_floor: {value}
"""
        )


def test_public_output_floor_rejects_moa_mode():
    with pytest.raises(
        ValidationError,
        match="public_output_floor cannot be combined with moa_samples > 0",
    ):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
moa_samples: 2
public_output_floor: 64
"""
        )


async def test_public_output_floor_clamps_final_unit_dispatch():
    """DTO-D9: the DSL floor must reach the final unit's dispatched budget."""

    class RecordingBackend:
        def __init__(self) -> None:
            self.requests_seen = []

        async def generate(self, request):
            self.requests_seen.append(request)
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0, text="done", token_ids=(), finish_reason="stop"
                    ),
                ),
                usage=GenerationUsage(prompt_tokens=1, completion_tokens=1),
                finished=True,
            )

        async def stream(self, request):
            yield await self.generate(request)

        async def shutdown(self) -> None:
            return None

    backend = RecordingBackend()
    spec = load_spec(
        """
workers:
  - {name: tier1, backend: mock}
  - {name: tier2, engine_ref: recorder}
public_output_floor: 64
roles:
  - name: planner
    worker: tier1
    prompt: "plan: {query}"
  - name: publisher
    worker: tier2
    role_type: publisher
    prompt: "answer: {planner}"
    prompt_suffix: "<THINK>"
    reasoning_close_tag: "</THINK>"
    depends_on: [planner]
"""
    )
    orchestrator = build_orchestrator(spec, engine_refs={"recorder": backend})
    result = await orchestrator.run(
        OrchestrationRequest(
            prompt=(
                "First, plan the work. Then execute it. "
                "Finally, summarize the outcome."
            ),
            sampling_params=SamplingParams(max_tokens=512),
        )
    )
    assert result.route.target == "multi_agent"
    assert backend.requests_seen[-1].sampling_params.max_tokens == 448  # 512 - 64


def test_build_openai_worker_preserves_keyless_auth(monkeypatch):
    captured = {}

    def fake_create_backend(name, **options):
        captured["name"] = name
        captured["options"] = options
        return object()

    monkeypatch.setattr("kairyu.dsl.loader.create_backend", fake_create_backend)
    build_orchestrator(
        load_spec(
            """
workers:
  - name: local
    backend: openai
    model: default
    base_url: http://replica:8000/v1
    api_key_env: null
"""
        )
    )

    assert captured == {
        "name": "openai",
        "options": {
            "model": "default",
            "base_url": "http://replica:8000/v1",
            "api_key_env": None,
        },
    }


def test_openai_worker_profile_is_forwarded_and_validated(monkeypatch):
    captured = {}

    def fake_create_backend(name, **options):
        captured["name"] = name
        captured["options"] = options
        return object()

    monkeypatch.setattr("kairyu.dsl.loader.create_backend", fake_create_backend)
    build_orchestrator(
        load_spec(
            """
workers:
  - name: local
    backend: openai
    model: default
    base_url: http://replica:8000/v1
    api_key_env: null
    options: {upstream: kairyu}
"""
        )
    )
    assert captured["options"]["upstream"] == "kairyu"

    with pytest.raises(ValidationError, match="unknown OpenAI-compatible upstream"):
        load_spec(
            """
workers:
  - name: invalid
    backend: openai
    options: {upstream: inferred}
"""
        )


PROFILES_SPEC = """
workers:
  - name: tier1
    backend: mock
  - name: tier2
    backend: mock
roles:
  - name: draft
    worker: tier1
    role_type: synthesizer
    prompt: "code draft: {query}"
profiles:
  - name: general
    roles:
      - name: g_prop
        worker: tier1
        prompt: "general proposal: {query}"
      - name: g_final
        worker: tier2
        role_type: synthesizer
        depends_on: [g_prop]
        prompt: "general final: {query} {g_prop}"
  - name: fast
    roles:
      - name: f_final
        worker: tier1
        role_type: publisher
        sampling: {temperature: 0.7, top_p: 0.8, max_tokens: 131072}
        prompt: "fast: {query}"
"""

JUDGE_SPEC = """
profile_judge:
  worker: tier2
  prompt_prefix: "<u>"
  prompt_suffix: "<a>"
  fallback: general
  choices:
    - {profile: fast, label: FAST, criteria: "trivial"}
    - {profile: primary, label: CODE, criteria: "code authoring"}
    - {profile: general, label: GENERAL, criteria: "everything else"}
"""


def test_profiles_load_and_build():
    spec = load_spec(PROFILES_SPEC)
    assert [profile.name for profile in spec.profiles] == ["general", "fast"]
    assert [role.name for role in spec.profiles[0].roles] == ["g_prop", "g_final"]
    # A single-role direct profile may declare sampling on its final unit
    # (DTO-D13): the style fields are policy and max_tokens a cap.
    assert spec.profiles[1].roles[0].sampling.max_tokens == 131072
    descriptor = build_orchestrator(spec).describe_routing()
    described = {
        name: [role["name"] for role in roles]
        for name, roles in descriptor["profiles"].items()
    }
    assert described == {"general": ["g_prop", "g_final"], "fast": ["f_final"]}
    assert descriptor["profiles"]["fast"][0]["sampling"]["max_tokens"] == 131072
    assert "profile_selector" in descriptor


def test_profiles_reject_moa_mode():
    with pytest.raises(
        ValidationError,
        match="profiles cannot be combined with moa_samples > 0",
    ):
        load_spec(PROFILES_SPEC + "\nmoa_samples: 2\n")


def test_profile_judge_loads_and_is_described():
    spec = load_spec(PROFILES_SPEC + JUDGE_SPEC)
    assert spec.profile_judge.worker == "tier2"
    assert spec.profile_judge.fallback == "general"
    assert [choice.label for choice in spec.profile_judge.choices] == ["FAST", "CODE", "GENERAL"]
    descriptor = build_orchestrator(spec).describe_routing()
    assert descriptor["profile_judge"] == {
        "worker": "tier2",
        "timeout_seconds": 5.0,
        "max_tokens": 8,
        "prompt_prefix": "<u>",
        "prompt_suffix": "<a>",
        "fallback": "general",
        "choices": [
            {"profile": "fast", "label": "FAST", "criteria": "trivial"},
            {"profile": "primary", "label": "CODE", "criteria": "code authoring"},
            {"profile": "general", "label": "GENERAL", "criteria": "everything else"},
        ],
    }
    assert "profile-judge verdict" in descriptor["profile_selector"]


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (
            JUDGE_SPEC.replace("worker: tier2", "worker: missing"),
            "profile_judge references unknown worker",
        ),
        (
            JUDGE_SPEC.replace("profile: fast,", "profile: nope,"),
            "references unknown profile 'nope'",
        ),
        (
            JUDGE_SPEC.replace("fallback: general", "fallback: nope"),
            "fallback references unknown profile",
        ),
        (
            JUDGE_SPEC.replace("label: FAST", "label: CODE"),
            "choice labels must be unique",
        ),
        (
            JUDGE_SPEC.replace("label: FAST", "label: fast"),
            "String should match pattern",
        ),
    ],
)
def test_profile_judge_is_validated(tail, message):
    with pytest.raises(ValidationError, match=message):
        load_spec(PROFILES_SPEC + tail)


def test_profile_judge_requires_profiles():
    with pytest.raises(ValidationError, match="profile_judge requires at least one profile"):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
roles:
  - {name: draft, worker: tier1, prompt: "d: {query}"}
profile_judge:
  worker: tier1
  choices:
    - {profile: primary, label: A, criteria: a}
    - {profile: other, label: B, criteria: b}
"""
        )


def test_profiles_are_validated_like_primary_roles():
    with pytest.raises(ValidationError, match="profiles.general.*unknown worker"):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
roles:
  - {name: draft, worker: tier1, prompt: "d: {query}"}
profiles:
  - name: general
    roles:
      - {name: g_final, worker: missing, prompt: "g: {query}"}
"""
        )
    with pytest.raises(ValidationError, match="profiles require a primary"):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
profiles:
  - name: general
    roles:
      - {name: g_final, worker: tier1, prompt: "g: {query}"}
"""
        )
    with pytest.raises(ValidationError, match="other than 'primary'"):
        load_spec(
            """
workers: [{name: tier1, backend: mock}]
roles:
  - {name: draft, worker: tier1, prompt: "d: {query}"}
profiles:
  - name: primary
    roles:
      - {name: g_final, worker: tier1, prompt: "g: {query}"}
"""
        )


class _RecordingBackend:
    """Returns one scripted text per call; optionally advertises image support."""

    def __init__(self, text: str = "done", *, vision: bool = False) -> None:
        self._text = text
        self._vision = vision
        self.requests_seen = []
        self.prepared_requests = []

    def supports_prompt_kind(self, kind: str) -> bool:
        return kind in ({"text", "multimodal"} if self._vision else {"text"})

    def supports_chat_template_kwargs(self, keys: frozenset[str]) -> bool:
        return True

    async def prepare_request(self, request):
        self.prepared_requests.append(request)

    async def generate(self, request):
        self.requests_seen.append(request)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(index=0, text=self._text, token_ids=(), finish_reason="stop"),
            ),
            usage=GenerationUsage(prompt_tokens=1, completion_tokens=1),
            finished=True,
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


_IMAGE_CONDITIONAL_SPEC = """
workers:
  - {name: vision, engine_ref: vision}
  - {name: text, engine_ref: text}
roles:
  - name: image_description
    worker: vision
    requires: image
    prompt: "describe: {query}"
  - name: policies
    worker: text
    prompt: "policies: {query} | {image_description}"
    depends_on: [image_description]
  - name: final
    worker: text
    role_type: synthesizer
    prompt: "final: {policies}"
    depends_on: [policies]
"""
_DAG_QUERY = "First, plan the work. Then execute it. Finally, summarize the outcome."


def _image_conditional_orchestrator():
    vision = _RecordingBackend("seen: a chart", vision=True)
    text = _RecordingBackend("done")
    orchestrator = build_orchestrator(
        load_spec(_IMAGE_CONDITIONAL_SPEC), engine_refs={"vision": vision, "text": text}
    )
    return orchestrator, vision, text


@pytest.mark.parametrize("stream", [False, True])
async def test_image_conditional_role_is_skipped_without_image(stream):
    """DTO-D11: a `requires: image` role never runs on a text request."""

    orchestrator, vision, text = _image_conditional_orchestrator()
    request = OrchestrationRequest(prompt=_DAG_QUERY, sampling_params=SamplingParams())

    if stream:
        events = [event async for event in await orchestrator.run_chat(request, stream=True)]
        result = events[-1].result
    else:
        result = await orchestrator.run(request)

    assert result.route.target == "multi_agent"
    assert vision.requests_seen == []
    assert text.requests_seen[0].prompt == f"policies: {_DAG_QUERY} | "
    assert result.text == "done"
    skipped = [
        event for event in result.structured_trace.events if event.kind == "skipped:condition"
    ]
    assert [(event.node, event.metadata["reason"]) for event in skipped] == [
        ("image_description", "no_image")
    ]


async def test_image_conditional_dependent_is_exactly_prepared_for_text_request():
    orchestrator, _vision, text = _image_conditional_orchestrator()
    request = OrchestrationRequest(prompt=_DAG_QUERY, sampling_params=SamplingParams())

    prepared = await orchestrator.prepare_request(request)
    expected = f"policies: {_DAG_QUERY} | "
    prepared_policy = next(
        item for item in text.prepared_requests if item.prompt == expected
    )
    result = await orchestrator.run(request, prepared=prepared)

    assert result.route.target == "multi_agent"
    assert text.requests_seen[0] is prepared_policy


async def test_image_conditional_role_runs_first_with_image():
    orchestrator, vision, text = _image_conditional_orchestrator()
    request = OrchestrationRequest(
        prompt=_DAG_QUERY,
        sampling_params=SamplingParams(),
        multimodal_prompt=MultimodalPrompt(
            _DAG_QUERY, (MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),)
        ),
    )

    result = await orchestrator.run(request)

    assert result.route.target == "multi_agent"
    assert len(vision.requests_seen) == 1
    assert isinstance(vision.requests_seen[0].prompt, MultimodalPrompt)
    assert text.requests_seen[0].prompt == f"policies: {_DAG_QUERY} | seen: a chart"
    assert not any(
        event.kind == "skipped:condition" for event in result.structured_trace.events
    )


def test_role_prompt_with_literal_brace_pair_is_rejected_at_load():
    """A literal `{...}` is a positional format field: it raised only at request
    time, after the stage's dependencies had run (tiered-example audit outage)."""

    with pytest.raises(ValidationError, match="escape literal braces"):
        load_spec(
            """
workers: [{name: w, engine_ref: w}]
roles:
  - {name: final, worker: w, prompt: "tool call like <tool_call>{...}</tool_call> for {query}"}
"""
        )


def test_image_conditional_head_is_rejected_at_load():
    with pytest.raises(ValidationError, match="head role 'head' cannot declare requires"):
        load_spec(
            """
workers: [{name: vision, engine_ref: vision}]
roles:
  - {name: head, worker: vision, role_type: head, requires: image, prompt: "h: {query}"}
  - {name: final, worker: vision, prompt: "f: {head}", depends_on: [head]}
"""
        )


def test_image_conditional_final_is_rejected_at_build():
    with pytest.raises(ValueError, match="cannot declare requires"):
        build_orchestrator(
            load_spec(
                """
workers: [{name: vision, engine_ref: vision}]
roles:
  - {name: final, worker: vision, requires: image, prompt: "f: {query}"}
"""
            ),
            engine_refs={"vision": _RecordingBackend(vision=True)},
        )


async def test_image_conditional_worker_capability_is_checked_per_image_request():
    """The image-capability of a conditional role's worker is a live property.

    A replica pool publishes the intersection of its currently placeable
    replicas, which is empty before the first health probe: build must not
    probe it (the tiered example could not boot), each image request must.
    """

    look_worker = _RecordingBackend("seen: a chart", vision=False)
    final_worker = _RecordingBackend("done", vision=True)
    orchestrator = build_orchestrator(
        load_spec(
            """
workers:
  - {name: look_worker, engine_ref: look_worker}
  - {name: final_worker, engine_ref: final_worker}
roles:
  - {name: look, worker: look_worker, requires: image, prompt: "l: {query}"}
  - {name: final, worker: final_worker, prompt: "f: {look}", depends_on: [look]}
"""
        ),
        engine_refs={"look_worker": look_worker, "final_worker": final_worker},
    )
    text_request = OrchestrationRequest(prompt=_DAG_QUERY, sampling_params=SamplingParams())
    image_request = OrchestrationRequest(
        prompt=_DAG_QUERY,
        sampling_params=SamplingParams(),
        multimodal_prompt=MultimodalPrompt(
            _DAG_QUERY, (MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),)
        ),
    )

    # Text requests never touch the conditional role's worker.
    assert (await orchestrator.run(text_request)).text == "done"
    assert look_worker.requests_seen == []

    with pytest.raises(ValueError, match="'look' requires image input but worker 'look_worker'"):
        await orchestrator.run(image_request)
    assert look_worker.requests_seen == []

    look_worker._vision = True
    result = await orchestrator.run(image_request)
    assert result.text == "done"
    assert isinstance(look_worker.requests_seen[0].prompt, MultimodalPrompt)


_CHECKLIST_SPEC = """
workers:
  - {{name: gen, engine_ref: llm}}
  - {{name: judge, {judge_ref}}}
roles:
  - {{name: answer, worker: gen, prompt: "{{query}}"}}
  - name: check
    worker: judge
    role_type: verifier
    verifies: answer
    depends_on: [answer]
    checklist:
      checks: [{{id: R1, proposition: short, primitive: {primitive}, params: {{max_chars: 10}}}}]
"""


@pytest.mark.parametrize(
    ("judge_ref", "primitive", "error"),
    [
        ("engine_ref: llm", "length", "systemone_ref worker"),
        ("systemone_ref: jev", "guess", "unknown check primitive"),
    ],
)
def test_checklist_verifier_is_validated_at_load(judge_ref, primitive, error):
    with pytest.raises((ValidationError, ValueError), match=error):
        spec = load_spec(_CHECKLIST_SPEC.format(judge_ref=judge_ref, primitive=primitive))
        build_orchestrator(spec, engine_refs={"llm": object()}, systemone_refs={"jev": object()})

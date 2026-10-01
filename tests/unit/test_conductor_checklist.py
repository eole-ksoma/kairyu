"""Checklist verifiers: deterministic checks, System One reads, and the
guarantee report published with the final answer."""

import json

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.systemone import SystemOneReply, SystemOneUnavailableError
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.checklist import (
    ChecklistCheck,
    ChecklistConfig,
    ChecklistQuestion,
    CurationConfig,
    ItemSource,
    StateSection,
)
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.outputs import CompletionOutput


class RoutedBackend:
    """Answers by the role tag at the start of the rendered prompt."""

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self._replies = {tag: list(texts) for tag, texts in replies.items()}
        self.prompts: list[str] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = str(request.prompt)
        self.prompts.append(prompt)
        tag = prompt.split("]", 1)[0].lstrip("[")
        text = self._replies[tag].pop(0)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=text, token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class FakeSystemOne:
    """Yes-probability per question chosen by a substring of its instructions."""

    def __init__(self, probabilities: dict[str, float], *, down: bool = False) -> None:
        self._probabilities = probabilities
        self._down = down
        self.bodies: list[dict] = []

    async def decide(self, body: dict) -> SystemOneReply:
        self.bodies.append(body)
        if self._down:
            raise SystemOneUnavailableError("ConnectError")
        answers = {}
        for key, question in body["questions"].items():
            text = json.dumps(question["instructions"])
            p = next(
                (value for needle, value in self._probabilities.items() if needle in text),
                0.99,
            )
            answers[key] = {"noul": p}
        return SystemOneReply(
            status=200,
            body=json.dumps({"answers": answers}).encode(),
            headers={},
            input_tokens=10,
            output_tokens=1,
        )


def _answer_roles(**config_overrides) -> tuple[RoleSpec, ...]:
    config = ChecklistConfig(
        checks=(
            ChecklistCheck(
                id="R1",
                proposition="states the answer 42",
                primitive="contains",
                params={"text": "42"},
            ),
        ),
        questions=(
            ChecklistQuestion(
                id="G1",
                proposition="every claim is grounded",
                ask="Is this claim supported?",
                context={"claim": "{item[text]}"},
                foreach=ItemSource(role="claims", path="claims"),
            ),
        ),
        state=(StateSection("request", "query"), StateSection("answer", "answer")),
        subject="the answer",
        threshold=0.8,
        max_refinements=2,
        on_exhausted="latest_checks_passed",
        on_unavailable="publish_unverified",
        unverified_from="generator",
        **config_overrides,
    )
    return (
        RoleSpec(name="generator", worker="gen", prompt="[generator] {query}"),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="",
            depends_on=("generator",),
            seed_from="generator",
            refine_prompt="[repair] {previous}\n{feedback}",
        ),
        RoleSpec(
            name="claims",
            worker="gen",
            prompt="[claims] {answer}",
            depends_on=("answer",),
        ),
        RoleSpec(
            name="checklist",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer", "claims"),
            checklist=config,
        ),
    )


def _claims(*texts: str) -> str:
    return json.dumps({"claims": [{"id": f"c{i}", "text": t} for i, t in enumerate(texts)]})


async def test_seeded_draft_that_passes_publishes_with_a_guarantee():
    backend = RoutedBackend(
        {"generator": ["The answer is 42."], "claims": [_claims("The answer is 42.")]}
    )
    judge = FakeSystemOne({})
    conductor = Conductor(_answer_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    assert result.verification is not None
    report = result.verification.as_dict()
    assert report["guaranteed"] is True and report["reason"] is None
    assert {item["id"]: item["passed"] for item in report["requirements"]} == {
        "R1": True,
        "G1": True,
    }
    # The draft is published as-is: no generation for the answer itself.
    assert [prompt.split("]")[0] for prompt in backend.prompts] == ["[generator", "[claims"]
    # Jev request shape: a JSON state object and self-contained noul questions.
    body = judge.bodies[0]
    assert body["state"] == {
        "request": "What is six times seven?",
        "answer": "The answer is 42.",
    }
    (question,) = body["questions"].values()
    assert question == {
        "type": "noul",
        "instructions": {"question": "Is this claim supported?", "claim": "The answer is 42."},
    }


async def test_deterministic_failure_repairs_before_any_judge_read():
    backend = RoutedBackend(
        {
            "generator": ["The answer is 41."],
            "repair": ["The answer is 42."],
            "claims": [_claims("The answer is 42.")],
        }
    )
    judge = FakeSystemOne({})
    conductor = Conductor(_answer_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    assert result.verification.guaranteed is True
    assert result.verification.attempts == 2
    repair_prompt = next(p for p in backend.prompts if p.startswith("[repair]"))
    assert "The answer is 41." in repair_prompt and "[R1] states the answer 42" in repair_prompt
    # The failed attempt never reached the claim extractor or System One.
    assert len(judge.bodies) == 1
    assert sum(p.startswith("[claims]") for p in backend.prompts) == 1


async def test_exhausted_refinements_publish_the_latest_attempt_that_passed_checks():
    backend = RoutedBackend(
        {
            "generator": ["The answer is 42, says the moon."],
            "repair": ["The answer is 41.", "The answer is 42, says the sun."],
            "claims": [_claims("says the moon"), _claims("says the sun")],
        }
    )
    judge = FakeSystemOne({"says the": 0.3})
    conductor = Conductor(_answer_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=16))

    assert result.final_text == "The answer is 42, says the sun."
    report = result.verification.as_dict()
    assert report["guaranteed"] is False and report["reason"] == "refinement_limit"
    assert report["attempts"] == 3
    assert [p.split("]")[0] for p in backend.prompts].count("[repair") == 2


async def test_unavailable_judge_publishes_the_draft_unverified():
    backend = RoutedBackend(
        {"generator": ["The answer is 42."], "claims": [_claims("The answer is 42.")]}
    )
    conductor = Conductor(
        _answer_roles(),
        {"gen": backend},
        decision_workers={"judge": FakeSystemOne({}, down=True)},
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_unit_ok
    assert result.final_text == "The answer is 42."
    assert result.verification.as_dict()["guaranteed"] is False
    assert result.verification.reason == "judge_unavailable"


async def test_requirement_list_is_curated_before_downstream_roles_read_it():
    extraction = json.dumps(
        {
            "units": [{"id": "U1", "text": "a"}, {"id": "U2", "text": "b"}],
            "requirements": [
                {"id": "R1", "proposition": "does a", "sources": ["U1"]},
                {"id": "R2", "proposition": "also does a", "sources": ["U1"]},
                {"id": "R3", "proposition": "does c", "sources": ["U2"]},
            ],
        }
    )
    backend = RoutedBackend({"extract": [extraction], "use": ["done"]})
    judge = FakeSystemOne({"derived:does c": 0.1, "same:does a|also does a": 0.9})
    roles = (
        RoleSpec(name="extract", worker="gen", prompt="[extract] {query}"),
        RoleSpec(
            name="requirements_check",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="extract",
            depends_on=("extract",),
            checklist=ChecklistConfig(
                questions=(
                    ChecklistQuestion(
                        id="{item[id]}",
                        proposition="derived:{item[proposition]}",
                        foreach=ItemSource(role="extract", path="requirements"),
                        group="necessity",
                    ),
                    ChecklistQuestion(
                        id="{a[id]}+{b[id]}",
                        proposition="distinct requirements",
                        ask="same:{a[proposition]}|{b[proposition]}",
                        foreach=ItemSource(
                            role="extract", path="requirements", pairs_sharing="sources"
                        ),
                        expect="no",
                        group="exclusivity",
                    ),
                ),
                state=(StateSection("request", "query"),),
                threshold=0.5,
                max_refinements=0,
                curate=CurationConfig(
                    items_path="requirements",
                    drop_group="necessity",
                    merge_group="exclusivity",
                    units_path="units",
                    pad={
                        "id": "P_{unit[id]}",
                        "proposition": "answers: {unit[text]}",
                        "kind": "semantic",
                    },
                ),
            ),
        ),
        RoleSpec(name="use", worker="gen", prompt="[use] {extract}", depends_on=("extract",)),
    )
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    await conductor.run("do a and b", budget=Budget(max_steps=8))

    curated = json.loads(backend.prompts[-1].removeprefix("[use] "))
    assert curated["requirements"] == [
        {"id": "R1", "proposition": "does a; also does a", "sources": ["U1"]},
        {"id": "P_U2", "proposition": "answers: b", "kind": "semantic", "sources": ["U2"]},
    ]


async def test_a_check_failure_still_reports_the_unread_requirements():
    backend = RoutedBackend({"generator": ["The answer is 41."], "repair": ["Still 41."]})
    judge = FakeSystemOne({})
    roles = (
        RoleSpec(name="generator", worker="gen", prompt="[generator] {query}"),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="",
            depends_on=("generator",),
            seed_from="generator",
            refine_prompt="[repair] {previous}\n{feedback}",
        ),
        RoleSpec(
            name="checklist",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=ChecklistConfig(
                checks=(
                    ChecklistCheck(
                        id="R1",
                        proposition="states 42",
                        primitive="contains",
                        params={"text": "42"},
                    ),
                ),
                questions=(ChecklistQuestion(id="R2", proposition="explains the product"),),
                state=(StateSection("answer", "answer"),),
                max_refinements=1,
            ),
        ),
    )
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=8))

    report = result.verification.as_dict()
    assert report["guaranteed"] is False and report["reason"] == "refinement_limit"
    unread = next(item for item in report["requirements"] if item["id"] == "R2")
    assert unread["judged"] is False and unread["p"] is None
    # The repair is asked to fix only what was judged, and no read happened.
    repair = next(p for p in backend.prompts if p.startswith("[repair]"))
    assert "[R1]" in repair and "[R2]" not in repair
    assert judge.bodies == []


async def test_a_failed_read_cancels_its_siblings_and_keeps_their_usage():
    import asyncio

    from kairyu.orchestration.checklist import ChecklistRun, ChecklistUnavailable

    hanging = asyncio.Event()
    cancelled = []

    class SplitJev:
        async def decide(self, body):
            (key,) = body["questions"]
            if key == "q0":
                return SystemOneReply(status=529, body=b"{}", headers={})
            if key == "q1":
                return SystemOneReply(
                    status=200,
                    body=json.dumps({"answers": {"q1": {"noul": 0.9}}}).encode(),
                    headers={},
                    input_tokens=7,
                )
            try:
                await hanging.wait()
            except asyncio.CancelledError:
                cancelled.append(key)
                raise

    config = ChecklistConfig(
        questions=tuple(ChecklistQuestion(id=f"R{n}", proposition=f"p{n}") for n in range(3)),
        state=(StateSection("answer", "answer"),),
        max_questions_per_call=1,
    )
    run = ChecklistRun(config, target_text="x", sources="q")

    try:
        await run.decide(SplitJev(), {"answer": "x"}, "q")
    except ChecklistUnavailable as error:
        usage = error.usage
    else:
        raise AssertionError("a 529 read must make the checklist unavailable")

    # No read outlives the decision, and the completed read is still billed.
    assert cancelled == ["q2"]
    assert usage == (7, 0)


async def test_a_seeded_answer_keeps_its_seeds_finish_reason():
    class LengthBackend(RoutedBackend):
        async def generate(self, request):
            result = await super().generate(request)
            completion = result.completions[0]
            object.__setattr__(completion, "finish_reason", "length")
            return result

    backend = LengthBackend(
        {"generator": ["The answer is 42"], "claims": [_claims("The answer is 42.")]}
    )
    conductor = Conductor(
        _answer_roles(), {"gen": backend}, decision_workers={"judge": FakeSystemOne({})}
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.completions[0].finish_reason == "length"


async def test_a_downstream_seed_publishes_the_attempt_the_verifier_kept():
    # Review P1 (round 2): the verifier falls back to its first attempt;
    # a later seed must not republish the rejected retry's text.
    backend = RoutedBackend({"source": ["good answer"], "fix": ["bad answer"]})
    judge = FakeSystemOne({"good answer": 0.1})
    roles = (
        RoleSpec(
            name="source",
            worker="gen",
            prompt="[source] {query}",
            refine_prompt="[fix] {previous}\n{feedback}",
        ),
        RoleSpec(
            name="check",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="source",
            depends_on=("source",),
            checklist=ChecklistConfig(
                checks=(
                    ChecklistCheck(
                        id="R1",
                        proposition="is good",
                        primitive="contains",
                        params={"text": "good"},
                    ),
                ),
                questions=(
                    ChecklistQuestion(
                        id="R2", proposition="judged", ask="Is it right?", context={"t": "{source}"}
                    ),
                ),
                state=(StateSection("source", "source"),),
                threshold=0.9,
                max_refinements=1,
                on_exhausted="latest_checks_passed",
            ),
        ),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="",
            depends_on=("source",),
            seed_from="source",
            refine_prompt="[unused] {previous}",
        ),
    )
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("q", budget=Budget(max_steps=8))

    assert result.final_text == "good answer"
    assert result.completions[0].text == "good answer"


def _inline_claims_roles(*, writer_seeded: bool) -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="draft", worker="gen", prompt="[draft] {query}"),
        RoleSpec(name="claims", worker="gen", prompt="[claims] {draft}", depends_on=("draft",)),
        RoleSpec(
            name="check",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="draft",
            depends_on=("draft", "claims"),
            checklist=ChecklistConfig(
                questions=(ChecklistQuestion(id="R1", proposition="ok"),),
                state=(StateSection("claims", "claims"),),
            ),
        ),
        RoleSpec(
            name="writer",
            worker="gen",
            prompt="" if writer_seeded else "[writer] {claims}",
            depends_on=("draft", "claims"),
            seed_from="draft" if writer_seeded else None,
            refine_prompt="[fix] {previous}" if writer_seeded else "",
        ),
    )


async def test_the_orchestrator_resolves_the_final_unit_like_the_conductor():
    # Review P2 (round 3): a dependency on an inline role is one on its target,
    # so the writer, not the draft, publishes; a seeded writer refuses n > 1.
    from kairyu.orchestration.orchestrator import Orchestrator
    from kairyu.orchestration.request import OrchestrationRequest
    from kairyu.orchestration.router import RouteThresholds, RuleRouter
    from kairyu.sampling_params import SamplingParams

    for seeded in (False, True):
        roles = _inline_claims_roles(writer_seeded=seeded)
        orchestrator = Orchestrator(
            {"gen": RoutedBackend({})},
            router=RuleRouter(RouteThresholds(multi_step_markers=0)),
            roles=roles,
            decision_workers={"judge": FakeSystemOne({})},
        )
        assert orchestrator._conductor_final_role(roles).name == "writer"
    with pytest.raises(ValueError, match="n > 1"):
        await orchestrator.run(
            OrchestrationRequest(prompt="q", sampling_params=SamplingParams(max_tokens=8, n=2))
        )


def test_a_final_checklist_cannot_curate_what_it_publishes():
    # Review P1 (round 3): the guarantee must describe the published text.
    roles = (
        RoleSpec(name="answer", worker="gen", prompt="[answer] {query}"),
        RoleSpec(
            name="check",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=ChecklistConfig(
                questions=(ChecklistQuestion(id="R1", proposition="ok"),),
                state=(StateSection("answer", "answer"),),
                curate=CurationConfig(items_path="items", drop_group="checklist"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="cannot curate"):
        Conductor(roles, {"gen": RoutedBackend({})}, decision_workers={"judge": FakeSystemOne({})})

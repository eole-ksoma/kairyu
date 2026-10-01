"""Checklist-verified answers: DeepSeek-V4.1 six-GPU + OpenJev x 2 example (VCO-D1).

The example's own kairyu.yaml / verified.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against two fake OpenJev replicas), so the tests observe what the deployed
L1 services would receive and what the caller gets back.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest
import yaml

from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.systemone import HTTPSystemOneBackend
from kairyu.entrypoints.server.chat_service import validate_orchestration_chat_input
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.sampling_params import SamplingParams

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-openjev-verified-8gpu"

CHECKLIST = {
    "units": [
        {"id": "U1", "text": "Name the capital of France"},
        {"id": "U2", "text": "in one word"},
    ],
    "requirements": [
        {
            "id": "R1",
            "proposition": "names Paris",
            "kind": "semantic",
            "origin": "explicit",
            "sources": ["U1"],
        },
        {
            "id": "R2",
            "proposition": "the answer is one word",
            "kind": "deterministic",
            "origin": "explicit",
            "sources": ["U2"],
            "check": {"primitive": "regex", "params": {"pattern": "^\\s*\\S+\\s*$"}},
        },
    ],
}


def _text(body: dict) -> str:
    return "\n".join(
        message["content"] if isinstance(message["content"], str) else ""
        for message in body["messages"]
    )


def _deepseek(
    seen: list[dict], *, draft: str, checklist: dict | None = None, draft_finish: str = "stop"
):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = _text(body)
        if text.startswith("[extract]"):
            answer = json.dumps(checklist or CHECKLIST)
        elif text.startswith("[state_builder]"):
            answer = json.dumps(
                {"claims": [{"id": "c1", "text": "Paris", "basis": "general", "evidence": ""}]}
            )
        elif text.startswith("[repair]"):
            answer = "Paris"
        else:
            answer = draft
        finish = draft_finish if answer == draft else "stop"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": finish,
                    }
                ],
                "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
            },
        )

    return handler


def _openjev(
    reads: list[dict],
    *,
    route: str = "VERIFIED",
    implicit: float = 0.9999,
    down: bool = False,
    sufficiency: float = 0.9999,
    needs: str | None = None,
    unneeded: str | None = None,
):
    def answer(question: dict, state: dict) -> dict:
        text = json.dumps(question)
        if question["type"] == "choice":
            other = next(label for label in question["criteria"] if label != route)
            return {"type": "choice", "probabilities": {route: 0.9, other: 0.1}}
        if "same thing" in text:
            return {"noul": 0.0001}
        if "did not say it" in text:
            return {"noul": implicit}
        if "fully cover this instruction unit" in text:
            if needs is not None:
                return {"noul": 0.9999 if needs in json.dumps(state) else 0.0}
            return {"noul": sufficiency}
        if unneeded is not None and unneeded in text and "ask for this condition" in text:
            return {"noul": 0.4}
        return {"noul": 0.9999}

    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("both OpenJev replicas are down", request=request)
        body = json.loads(request.content)
        reads.append({"replica": request.url.host, **body})
        answers = {
            key: answer(question, body["state"]) for key, question in body["questions"].items()
        }
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(
    seen: list[dict],
    reads: list[dict],
    *,
    draft: str,
    spec: str = "verified-always.yaml",
    route: str = "VERIFIED",
    implicit: float = 0.9999,
    checklist: dict | None = None,
    jev_down: bool = False,
    sufficiency: float = 0.9999,
    needs: str | None = None,
    unneeded: str | None = None,
    draft_finish: str = "stop",
):
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options,
            transport=httpx.MockTransport(
                _deepseek(seen, draft=draft, checklist=checklist, draft_finish=draft_finish)
            ),
        )
        for name, pool in deployment.pools.items()
    }
    judges = {
        name: HTTPSystemOneBackend(
            base_urls=section.base_urls,
            upstream_model=section.upstream_model,
            transport=httpx.MockTransport(
                _openjev(
                    reads,
                    route=route,
                    implicit=implicit,
                    down=jev_down,
                    sufficiency=sufficiency,
                    needs=needs,
                    unneeded=unneeded,
                )
            ),
        )
        for name, section in deployment.systemone.items()
    }
    return build_orchestrator(load_spec(EXAMPLE / spec), engine_refs=engines, systemone_refs=judges)


def _call(content: str, **sampling) -> OrchestrationRequest:
    """The orchestration call the chat route makes for one user message."""

    params = SamplingParams(max_tokens=4096, **sampling)
    chat = ChatCompletionRequest(
        model="kairyu-verified", messages=[{"role": "user", "content": content}]
    )
    return OrchestrationRequest(
        prompt=validate_orchestration_chat_input(chat).prompt,
        sampling_params=params,
        response_format=params.extra_args.get("response_format"),
    )


def test_gateway_builds_from_the_example_configs(tmp_path: Path) -> None:
    raw = (
        (EXAMPLE / "kairyu.yaml")
        .read_text()
        .replace("/etc/kairyu/verified.yaml", str(EXAMPLE / "verified.yaml"))
        .replace("/etc/kairyu/verified-always.yaml", str(EXAMPLE / "verified-always.yaml"))
        .replace("/var/lib/kairyu/placement", str(tmp_path))
    )
    build_app_from_spec(load_deployment_spec(raw, resolve_credentials=False), EXAMPLE)


async def test_a_one_word_draft_is_published_with_a_guarantee() -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris")

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    assert result.text == "Paris"
    report = result.verification.as_dict()
    assert report["guaranteed"] is True
    assert {item["id"] for item in report["requirements"]} == {
        "R1",
        "R2",
        "G1",
        "G1-excerpts",
        "G2",
        "G3",
    }
    by_role = {_text(body).split("]", 1)[0].lstrip("["): body for body in seen}
    # The generator sees only the conversation, never the checklist, and the
    # draft is published as-is (no repair call).
    generator = next(body for body in seen if _text(body).startswith("Kairyu L2"))
    assert "names Paris" not in _text(generator) and "[repair]" not in by_role
    # Extraction and claim lists are grammar-constrained; reads used both OpenJev replicas'
    # pool (System One) and never a chat endpoint.
    # The extractor analyses the user's request, not Kairyu's answer contract.
    extract_text = _text(by_role["extract"])
    assert "Name the capital of France in one word." in extract_text
    assert "Return only the assistant response body" not in extract_text
    assert by_role["extract"]["response_format"]["type"] == "json_schema"
    assert by_role["state_builder"]["response_format"]["type"] == "json_schema"
    # Every DeepSeek step thinks at the request's effort (default high).
    for role in ("state_builder", "extract"):
        assert by_role[role]["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in by_role[role]
    assert {read["replica"] for read in reads} <= {"openjev-0", "openjev-1"}
    # The judge gets the conversation as role-tagged messages and the
    # answer's checklist as JSON values, never as one escaped text blob.
    answer_read = next(read for read in reads if "answer" in read["state"])
    assert answer_read["state"]["conversation"][-1]["role"] == "user"
    assert answer_read["state"]["checklist"]["requirements"][0]["id"] == "R1"


async def test_a_deterministic_violation_is_repaired_then_guaranteed() -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="The capital of France is Paris.")

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    assert result.text == "Paris"
    assert result.verification.guaranteed and result.verification.attempts == 2
    repair = next(_text(body) for body in seen if _text(body).startswith("[repair]"))
    assert "[R2] the answer is one word" in repair
    assert "The capital of France is Paris." in repair


async def test_the_callers_response_format_constrains_draft_and_repair() -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris")
    schema = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "string"}}}

    await orchestrator.run(
        _call("Name the capital of France in one word.", extra_args={"response_format": schema})
    )

    generator = next(body for body in seen if _text(body).startswith("Kairyu L2"))
    assert generator["response_format"] == schema


def test_compose_gpus_match_the_allocation() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    services = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]

    def gpus(service: str) -> list[int]:
        devices = services[service]["deploy"]["resources"]["reservations"]["devices"]
        return [int(index) for index in devices[0]["device_ids"]]

    assert gpus("deepseek") == spec["allocation"]["deepseek"]["gpu_ids"]
    assert [*gpus("openjev-0"), *gpus("openjev-1")] == spec["allocation"]["openjev"]["gpu_ids"]


def test_both_models_share_one_verified_dag() -> None:
    routed = yaml.safe_load((EXAMPLE / "verified.yaml").read_text())
    always = yaml.safe_load((EXAMPLE / "verified-always.yaml").read_text())
    assert routed["roles"] == always["roles"]
    assert "profile_judge" not in always and not always.get("profiles")


@pytest.mark.parametrize(("route", "served"), [("VERIFIED", "verified"), ("THINK", "think")])
@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_jev_routes_and_every_deepseek_call_thinks_at_the_callers_effort(
    route, served, effort
) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris", spec="verified.yaml", route=route)
    call = _call("Name the capital of France in one word.")
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    route_read = next(read for read in reads if "route" in read["questions"])
    assert route_read["state"]["conversation"][-1]["role"] == "user"
    if served == "verified":
        assert result.verification is not None and result.verification.guaranteed
    else:
        assert result.verification is None
        assert [_text(body).split("]")[0] for body in seen] == ["[deepseek_think_answer"]
    # One effort for every DeepSeek step, default high (75).
    assert {body.get("reasoning_effort") for body in seen} == {effort or "high"}


async def test_an_unavailable_jev_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], draft="Paris", spec="verified.yaml", jev_down=True)

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    assert [_text(body).split("]")[0] for body in seen] == ["[deepseek_think_answer"]


@pytest.mark.parametrize(("p_expected", "kept"), [(0.9999, True), (0.05, False)])
async def test_implicit_requirements_stay_only_when_jev_finds_them_expected(
    p_expected, kept
) -> None:
    checklist = {
        **CHECKLIST,
        "requirements": [
            *CHECKLIST["requirements"],
            {
                "id": "R3",
                "proposition": "the answer is in English",
                "kind": "semantic",
                "origin": "implicit",
                "sources": ["U1"],
            },
        ],
    }
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(
        seen, reads, draft="Paris", implicit=p_expected, checklist=checklist
    )

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    implicit_questions = [
        question
        for read in reads
        for question in read["questions"].values()
        if "did not say it" in json.dumps(question)
    ]
    assert len(implicit_questions) == 1
    assert "R3: the answer is in English" in json.dumps(implicit_questions[0])
    judged = {item.id for item in result.verification.items}
    assert ("R3" in judged) is kept


async def test_an_unconfirmed_requirement_set_never_yields_a_guarantee() -> None:
    # Review P1: the extractor never covers U1 sufficiently, even after the
    # re-extraction, so the answer passing that checklist proves nothing.
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], draft="Paris", sufficiency=0.0)

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    assert result.text == "Paris"
    assert result.verification.guaranteed is False
    assert result.verification.reason == "requirements_unconfirmed"


async def test_duplicate_deterministic_conditions_keep_their_own_checks() -> None:
    # Review P1: merging used to keep only the first check ("at most two
    # words") and report "exactly two words" as passed for one word.
    checklist = {
        "units": [{"id": "U1", "text": "Answer in exactly two words"}],
        "requirements": [
            {
                "id": "R1",
                "proposition": "at most two words",
                "kind": "deterministic",
                "origin": "explicit",
                "sources": ["U1"],
                "check": {"primitive": "length", "params": {"max_words": 2}},
            },
            {
                "id": "R2",
                "proposition": "exactly two words",
                "kind": "deterministic",
                "origin": "explicit",
                "sources": ["U1"],
                "check": {"primitive": "length", "params": {"min_words": 2, "max_words": 2}},
            },
        ],
    }
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], draft="Paris", checklist=checklist)

    result = await orchestrator.run(_call("Name the capital of France in exactly two words."))

    by_id = {item.id: item for item in result.verification.items}
    assert by_id["R2"].passed is False
    assert result.verification.guaranteed is False


async def test_a_passing_checklist_is_reconfirmed_after_curation_changes_it() -> None:
    # Review P1 (round 2): curation drops R2 (necessity 0.4) from a set that
    # passed; without R2 the unit is no longer covered.
    checklist = {
        "units": [{"id": "U1", "text": "Name the capital and the country"}],
        "requirements": [
            {
                "id": "R1",
                "proposition": "names Paris",
                "kind": "semantic",
                "origin": "explicit",
                "sources": ["U1"],
            },
            {
                "id": "R2",
                "proposition": "names France",
                "kind": "semantic",
                "origin": "explicit",
                "sources": ["U1"],
            },
        ],
    }
    orchestrator = _orchestrator(
        [], [], draft="Paris", checklist=checklist, needs="names France", unneeded="names France"
    )

    result = await orchestrator.run(_call("Name the capital of France and the country."))

    assert result.verification.guaranteed is False
    assert result.verification.reason == "requirements_unconfirmed"


async def test_n_greater_than_one_is_refused_on_the_real_dag() -> None:
    # Review P2 (round 2): the final unit is the seeded answer, not the
    # inline state builder.
    orchestrator = _orchestrator([], [], draft="Paris")

    with pytest.raises(ValueError, match="n > 1"):
        await orchestrator.run(_call("Name the capital of France.", n=2))


async def test_an_unverified_draft_keeps_its_finish_reason() -> None:
    # Review P2 (round 2): the judge is down; the published draft was cut.
    orchestrator = _orchestrator([], [], draft="Paris", jev_down=True, draft_finish="length")

    result = await orchestrator.run(_call("Name the capital of France."))

    assert result.verification.reason == "judge_unavailable"
    assert result.completions[0].finish_reason == "length"

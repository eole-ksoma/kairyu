"""Checklist-verified answers: DeepSeek-V4.1 six-GPU + OpenJev x 2 example (VCO-D1).

The example's own kairyu.yaml / verified.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against two fake OpenJev replicas), so the tests observe what the deployed
L1 services would receive and what the caller gets back.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
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
        {"id": "R1", "proposition": "names Paris", "kind": "semantic", "sources": ["U1"]},
        {
            "id": "R2",
            "proposition": "the answer is one word",
            "kind": "deterministic",
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


def _deepseek(seen: list[dict], *, draft: str):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = _text(body)
        if text.startswith("[extract]"):
            answer = json.dumps(CHECKLIST)
        elif text.startswith("[state_builder]"):
            answer = json.dumps(
                {"claims": [{"id": "c1", "text": "Paris", "basis": "general", "evidence": ""}]}
            )
        elif text.startswith("[repair]"):
            answer = "Paris"
        else:
            answer = draft
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
            },
        )

    return handler


def _openjev(reads: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        reads.append({"replica": request.url.host, **body})
        answers = {
            key: {"noul": 0.0001 if "same thing" in json.dumps(question) else 0.9999}
            for key, question in body["questions"].items()
        }
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(seen: list[dict], reads: list[dict], *, draft: str):
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options,
            transport=httpx.MockTransport(_deepseek(seen, draft=draft)),
        )
        for name, pool in deployment.pools.items()
    }
    judges = {
        name: HTTPSystemOneBackend(
            base_urls=section.base_urls,
            upstream_model=section.upstream_model,
            transport=httpx.MockTransport(_openjev(reads)),
        )
        for name, section in deployment.systemone.items()
    }
    return build_orchestrator(
        load_spec(EXAMPLE / "verified.yaml"), engine_refs=engines, systemone_refs=judges
    )


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
    # V4.1 thinks by default: the claim lister runs in chat mode, the
    # extractor and the generator think.
    assert by_role["state_builder"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert by_role["extract"]["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in by_role["extract"]
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

"""Qwen3.8 x 2 + DeepSeek-V4.1 six-GPU ensemble example (DTO-D16).

The example's own configs drive the production DSL/deployment loaders and the
real OpenAI backends against a fake vLLM, so the tests observe the requests
the deployed L1 services would receive.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.sampling_params import SamplingParams

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"
# A 64x64 solid-red PNG.
RED_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsItz/fMY"
    "xgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGB"
    "ywLzk8EPlvGqjQAAAABJRU5ErkJggg=="
)
_JUDGE_MARKER = "Reply with exactly one word"


def _text(body: dict) -> str:
    parts = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(part.get("text", "") for part in content)
    return "\n".join(parts)


def _has_image(body: dict) -> bool:
    return any(
        part.get("type") == "image_url"
        for message in body["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
    )


def _orchestrator(verdict: str, seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = _text(body)
        if _JUDGE_MARKER in text:
            answer = verdict
        elif "[audit]" in text:
            answer = "PASS"
        elif "[policies]" in text:
            answer = "\n".join(f"POLICY {n}: approach {n}" for n in range(1, 5))
        else:
            answer = "The image is red."
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

    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options, transport=httpx.MockTransport(handler)
        )
        for name, pool in deployment.pools.items()
    }
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    spec = spec.model_copy(
        update={
            "router": spec.router.model_copy(update={"artifact": str(EXAMPLE / "router.json")})
        }
    )
    return build_orchestrator(spec, engine_refs=engines)


def test_gateway_builds_from_the_example_configs(tmp_path: Path, monkeypatch) -> None:
    # The gateway once crash-looped at startup on this config (a pool model
    # without a supported chat policy), which the loader alone accepted.
    monkeypatch.setenv("KAIRYU_RESPONSES_COMPACTION_SECRET", "0" * 64)
    spec_path = tmp_path / "auto-max.yaml"
    spec_path.write_text(
        (EXAMPLE / "auto-max.yaml")
        .read_text()
        .replace("/etc/kairyu/router.json", str(EXAMPLE / "router.json"))
    )
    raw = (EXAMPLE / "kairyu.yaml").read_text().replace(
        "/etc/kairyu/auto-max.yaml", str(spec_path)
    )
    deployment = load_deployment_spec(raw, resolve_credentials=False)
    # The embedding bundle exists only inside the Kairyu image.
    deployment = deployment.model_copy(
        update={"embeddings": {}, "public_models": ["kairyu-auto-max"]}
    )

    build_app_from_spec(deployment, EXAMPLE)


def _image_call() -> OrchestrationRequest:
    query = "What color is this image?"
    return OrchestrationRequest(
        prompt=query,
        sampling_params=SamplingParams(max_tokens=4096),
        multimodal_prompt=MultimodalPrompt(query, (MultimodalItem("image", "uri", RED_PNG),)),
    )


def _deepseek(seen: list[dict]) -> list[dict]:
    return [body for body in seen if body["model"] == "deepseek-v4.1-flash"]


async def test_image_ensemble_sends_the_image_to_every_deepseek_role() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator("ENSEMBLE", seen)
    call = await orchestrator.judge_role_profile(_image_call())
    assert call.role_profile_judgment == "primary"
    judge_prompt = next(_text(body) for body in seen if _JUDGE_MARKER in _text(body))
    # Every route's worker accepts images, so none is withheld (DTO-D16).
    for label in ("QWEN", "QWEN_THINK", "DEEPSEEK", "DEEPSEEK_THINK", "ENSEMBLE"):
        assert f"- {label}:" in judge_prompt

    result = await orchestrator.run(call)

    assert result.text
    deepseek = _deepseek(seen)
    roles = {tag for body in deepseek for tag in ("[policies]", "[critique]", "[synthesis]")
             if tag in _text(body)}
    assert roles == {"[policies]", "[critique]", "[synthesis]"}
    for body in deepseek:
        assert _has_image(body)
        # Thinking roles carry the L3 effort (default high) and no thinking
        # toggle, so the V4.1 encoder renders the official thinking budget.
        assert body["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in body


@pytest.mark.parametrize(
    ("verdict", "role", "effort", "kwargs"),
    [
        ("DEEPSEEK", "[deepseek_answer]", None, {"enable_thinking": False}),
        ("DEEPSEEK_THINK", "[deepseek_think_answer]", "high", None),
    ],
)
async def test_direct_deepseek_routes_select_thinking_per_request(
    verdict: str, role: str, effort: str | None, kwargs: dict | None
) -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(verdict, seen)
    call = await orchestrator.judge_role_profile(_image_call())

    await orchestrator.run(call)

    (body,) = _deepseek(seen)
    assert role in _text(body)
    assert _has_image(body)
    assert body.get("reasoning_effort") == effort
    assert body.get("chat_template_kwargs") == kwargs


def _control():
    spec = importlib.util.spec_from_file_location("v41_ensemble_control", EXAMPLE / "control.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_control_pins_deepseek_to_gpus_0_5_and_each_qwen_replica_to_its_gpu(
    tmp_path: Path,
) -> None:
    control = _control()
    sysfs = tmp_path / "sys"
    nodes = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
    rows = {}
    for index, node in nodes.items():
        bus = f"0000:{index:02x}:00.0"
        device = sysfs / "bus/pci/devices" / bus
        device.mkdir(parents=True)
        (device / "numa_node").write_text(f"{node}\n")
        rows[index] = {"pci_bus_id": f"0000{bus}"}
    cpulists = {node: f"{node * 16}-{node * 16 + 15}" for node in range(4)}

    cpusets = control.cpusets(rows, sysfs=sysfs, cpulist=cpulists.__getitem__)

    assert cpusets == {
        "DEEPSEEK_CPUSET": "0-15,16-31,32-47",
        "QWEN_0_CPUSET": "48-63",
        "QWEN_1_CPUSET": "48-63",
    }

"""Route selection by System One probabilities (m1 D9)."""

import json

import pytest

from kairyu.engine.mock import MockBackend
from kairyu.engine.systemone import SystemOneReply, SystemOneUnavailableError
from kairyu.entrypoints.server.chat_service import validate_orchestration_chat_input
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.orchestration.conductor import RoleSpec
from kairyu.orchestration.orchestrator import Orchestrator, ProfileChoice, ProfileJudge
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.orchestration.router import RouteThresholds, RuleRouter
from kairyu.sampling_params import SamplingParams


class FakeJev:
    def __init__(self, probabilities=None, *, down=False):
        self.probabilities = probabilities or {}
        self.down = down
        self.bodies = []

    async def decide(self, body):
        self.bodies.append(body)
        if self.down:
            raise SystemOneUnavailableError("ConnectError")
        answer = {"type": "choice", "probabilities": self.probabilities}
        return SystemOneReply(
            status=200,
            body=json.dumps({"answers": {"route": answer}}).encode(),
            headers={},
            input_tokens=120,
        )


def _orchestrator(jev, llm, *, prefer=None):
    verified = (RoleSpec(name="v_final", worker="llm", prompt="[verified] {query}"),)
    think = (RoleSpec(name="t_final", worker="llm", prompt="[think] {query}"),)
    return Orchestrator(
        engines={"llm": llm},
        router=RuleRouter(RouteThresholds(multi_step_markers=0)),
        roles=verified,
        profiles={"deepseek_think": think},
        decision_workers={"jev": jev},
        profile_judge=ProfileJudge(
            worker="jev",
            question="Which route should answer?",
            choices=(
                ProfileChoice("deepseek_think", "THINK", "everyday requests"),
                ProfileChoice("primary", "VERIFIED", "accuracy is required"),
            ),
            fallback="deepseek_think",
            prefer_label=prefer[0] if prefer else None,
            prefer_min_probability=prefer[1] if prefer else 0.5,
        ),
    )


def _chat(*messages):
    request = ChatCompletionRequest(model="auto", messages=list(messages))
    prompt = validate_orchestration_chat_input(request).prompt
    return OrchestrationRequest(prompt=prompt, sampling_params=SamplingParams(max_tokens=64))


@pytest.mark.parametrize(
    ("probabilities", "prefer", "profile"),
    [
        ({"THINK": 0.8, "VERIFIED": 0.2}, None, "deepseek_think"),
        ({"THINK": 0.3, "VERIFIED": 0.7}, None, "primary"),
        # Accuracy first: a modest VERIFIED probability already routes there.
        ({"THINK": 0.8, "VERIFIED": 0.2}, ("VERIFIED", 0.15), "primary"),
    ],
)
async def test_route_follows_the_probabilities(probabilities, prefer, profile):
    jev = FakeJev(probabilities)
    llm = MockBackend()
    orchestrator = _orchestrator(jev, llm, prefer=prefer)
    call = await orchestrator.judge_role_profile(
        _chat(
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"},
            {"role": "user", "content": "Wrong again. Get it right this time."},
        )
    )

    assert call.role_profile_judgment == profile
    await orchestrator.run(call)
    served = "[verified]" if profile == "primary" else "[think]"
    assert [prompt[: len(served)] for prompt in llm.prompts_seen] == [served]
    # The judge reads the whole conversation (the earlier correction
    # included) as role-tagged messages, and the choices with their criteria.
    body = jev.bodies[0]
    assert [m["role"] for m in body["state"]["conversation"]] == ["user", "assistant", "user"]
    question = body["questions"]["route"]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "THINK": "everyday requests",
        "VERIFIED": "accuracy is required",
    }
    event = call.role_profile_judge_event
    assert event.metadata["p_VERIFIED"] == pytest.approx(probabilities["VERIFIED"])


async def test_an_unavailable_judge_falls_back():
    llm = MockBackend()
    orchestrator = _orchestrator(FakeJev(down=True), llm)

    call = await orchestrator.judge_role_profile(_chat({"role": "user", "content": "Hi"}))

    assert call.role_profile_judgment is None
    assert call.role_profile_judge_event.metadata["fallback"] == "backend_error"
    await orchestrator.run(call)
    assert [prompt[:7] for prompt in llm.prompts_seen] == ["[think]"]

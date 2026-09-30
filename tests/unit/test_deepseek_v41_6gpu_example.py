"""Contracts of the six-GPU DeepSeek-V4.1-Flash example (TP2 x DP3 / EP6)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-flash-6gpu"


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for name in ("control", "verification", "patch_sm120", "tune"):
        sys.modules.pop(name, None)
    return importlib.import_module


def test_served_configuration_loads_and_agrees(example):
    """The gateway must start on the committed YAML and describe the same L1."""
    spec = json.loads((EXAMPLE / "example.json").read_text())
    command = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]["deepseek"][
        "command"
    ]
    deployment = load_deployment_spec((EXAMPLE / "kairyu.yaml").read_text())
    (replica,) = deployment.pools[spec["allocation"]["model"]].replicas
    validate_backend_options(replica.backend, replica.options)

    def flag(name):
        return command[command.index(name) + 1]

    allocation = spec["allocation"]
    assert (
        int(flag("--tensor-parallel-size"))
        == replica.options["tensor_parallel_size"]
        == allocation["tensor_parallel_size"]
    )
    assert (
        int(flag("--data-parallel-size"))
        == replica.options["attention_data_parallel_size"]
        == allocation["data_parallel_size"]
    )
    assert replica.options["expert_parallel_size"] == allocation["expert_parallel_size"]
    assert "--enable-expert-parallel" in command
    assert int(flag("--max-num-seqs")) == deployment.server.max_concurrency
    assert json.loads(flag("--override-generation-config")) == spec["model"]["sampling"]
    assert replica.options["container_image_digest"] == spec["vllm"]["image_id"]
    example("control")  # import-time allocation check: TP x DP tiles the GPUs


def test_each_tp_pair_must_share_a_numa_node(example):
    control = example("control")
    pairs = [[0, 1], [2, 3], [4, 5]]
    assert control.dp_rank_numa_layout({0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}, pairs) == [0, 1, 2]
    with pytest.raises(SystemExit, match="DP rank 1"):
        control.dp_rank_numa_layout({0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}, pairs)


@pytest.mark.parametrize(
    "logprob,content,finish,error",
    [
        (-0.01, "323", "stop", None),
        (float("nan"), "323", "stop", "non-finite"),
        (-0.01, "324", "stop", "expected '323'"),
        (-0.01, "323", "length", "finish_reason"),
    ],
)
def test_readiness_probe_rejects_wrong_or_non_finite_answers(
    example, logprob, content, finish, error
):
    control = example("control")
    body = {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": content},
                "logprobs": {"content": [{"token": "3", "logprob": logprob}]},
            }
        ]
    }
    result = control.arithmetic_answer_error(body)
    assert (result is None) if error is None else (error in result)


def test_probe_gate_requires_every_dp_engine(example):
    verification = example("verification")
    before = verification.engine_success_counts(
        'vllm:request_success_total{engine="0",finished_reason="stop"} 4\n'
        'vllm:request_success_total{engine="1",finished_reason="stop"} 4\n'
    )
    after = verification.engine_success_counts(
        'vllm:request_success_total{engine="0",finished_reason="stop"} 9\n'
        'vllm:request_success_total{engine="0",finished_reason="length"} 1\n'
        'vllm:request_success_total{engine="1",finished_reason="stop"} 4\n'
        'vllm:request_success_total{engine="2",finished_reason="stop"} 2\n'
    )
    assert verification.served_engines(before, after) == {"0", "2"}


@pytest.mark.parametrize(
    "variant,prompt,ok",
    [
        (
            "default",
            "<｜System｜>Reasoning Effort: 75 (range 1-100)<｜User｜>q<｜Assistant｜><think>",
            True,
        ),
        (
            "low",
            "<｜System｜>Reasoning Effort: 50 (range 1-100)<｜User｜>q<｜Assistant｜><think>",
            True,
        ),
        (
            "low",
            "<｜System｜>Reasoning Effort: 25 (range 1-100)<｜User｜>q<｜Assistant｜><think>",
            False,
        ),
        ("chat", "<｜User｜>q<｜Assistant｜></think>", True),
        (
            "chat",
            "<｜System｜>Reasoning Effort: 75 (range 1-100)<｜User｜>q<｜Assistant｜></think>",
            False,
        ),
    ],
)
def test_rendered_effort_follows_the_model_author(example, variant, prompt, ok):
    assert (example("verification").rendering_error(variant, prompt) is None) is ok


def test_runtime_patches_fail_closed(example):
    patch = example("patch_sm120")
    for source in ("no anchor here", "            block_size=32,\n" * 2):
        with pytest.raises(ValueError, match="exactly one"):
            patch.configurable_swa_pages(source)
    with pytest.raises(ValueError, match="legacy"):
        patch.align_efforts("REASONING_EFFORT_MAPPINGS: Dict[str, int] = {'low': 30}\n")
    guarded = patch.top_p_guard(patch.TOP_P_ANCHOR)
    assert "if not (pivot_logit < M):" in guarded
    with pytest.raises(ValueError):
        patch.top_p_guard(guarded)


def test_tuning_candidates_edit_only_their_flags(example):
    tune = example("tune")
    base = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]["deepseek"]["command"]
    dep6 = tune.candidate_command(base, "dep6-official")
    assert dep6[dep6.index("--tensor-parallel-size") + 1] == "1"
    assert dep6[dep6.index("--data-parallel-size") + 1] == "6"
    assert "--moe-backend" not in dep6 and "marlin" not in dep6
    assert "--enable-expert-parallel" in dep6
    assert tune.candidate_command(base, "baseline") == base

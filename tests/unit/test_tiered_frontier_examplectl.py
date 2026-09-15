from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl.loader import load_spec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.entrypoints.chat_template import ChatTemplate

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/qwen3.8-deepseek-v4-8gpu"
EMBEDDING_MODEL_REPOSITORY = "Qdrant/all-MiniLM-L6-v2-onnx"
EMBEDDING_MODEL_REVISION = "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
EMBEDDING_MODEL_SHA256 = (
    "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5"
)
EMBEDDING_PROVENANCE_SHA256 = (
    "57246a4990eb0f08755df06ba57c1fec161032bd588332435e89c7ece244639c"
)


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_tiered_browser_gate_requires_one_model_and_separate_reasoning_ui() -> None:
    browser = (ROOT / "scripts/webui_browser_smoke.mjs").read_text()
    wrapper = (EXAMPLE / "browser-smoke.sh").read_text()

    assert "WEBUI_SMOKE_PHASE=tiered" in wrapper
    assert "EXPECTED_PRODUCT_MODEL=kairyu-auto-max" in wrapper
    assert "inventory.body.data.map" in browser
    assert "JSON.stringify([productModel])" in browser
    assert "button[aria-expanded]" in browser
    assert "intermediate processing was not initially folded" in browser
    assert "child !== reasoningRoot" in browser
    assert "child.classList.contains('markdown-prose')" in browser
    for attribution in (
        "L2 role:",
        "L1 worker:",
        "Engine:",
        "Model:",
        "tier1",
        "tier2",
        "qwen3.8-27b",
        "deepseek-v4-flash-0731-thinking",
    ):
        assert attribution in browser


def test_tiered_example_allocates_four_qwen_replicas_and_one_deepseek_tp4() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    kairyu = yaml.safe_load((EXAMPLE / "kairyu.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 8,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert spec["deepseek_l1_loopback_port"] == 8005
    assert spec["verification"]["serving"]["auto_max_combined_max_tokens"] == 65536
    assert spec["verification"]["coding"]["auto_max_combined_max_tokens"] == 65536
    assert spec["allocation"] == {
        "tier1": {
            "model": "qwen3.8-27b",
            "gpu_ids": [0, 1, 2, 3],
            "replicas": 4,
            "tensor_parallel_size": 1,
        },
        "tier2": {
            "model": "deepseek-v4-flash-0731",
            "gpu_ids": [4, 5, 6, 7],
            "replicas": 1,
            "tensor_parallel_size": 4,
            "expert_parallel_size": 4,
        },
    }
    assert set(compose["services"]) == {
        "executor",
        "qwen-0",
        "qwen-1",
        "qwen-2",
        "qwen-3",
        "deepseek",
        "kairyu",
        "chat-ui",
    }
    executor = compose["services"]["executor"]
    assert executor["network_mode"] == "none"
    assert "networks" not in executor
    assert "networks" not in compose
    assert executor["environment"]["RUNNER_SOCKET_PATH"] == (
        "/run/kairyu-executor/executor.sock"
    )
    assert executor["volumes"] == ["executor-socket:/run/kairyu-executor"]
    assert executor["read_only"] is True
    assert executor["user"] == "65534:65534"
    assert executor["cap_drop"] == ["ALL"]
    assert executor["security_opt"] == ["no-new-privileges:true"]
    assert executor["pids_limit"] == 256
    assert all("noexec" in mount for mount in executor["tmpfs"])
    assert "socket.AF_UNIX" in executor["healthcheck"]["test"][-1]
    assert "deploy" not in executor  # CPU-only: no GPU reservation
    assert "networks" not in compose["services"]["kairyu"]
    assert "executor-socket:/run/kairyu-executor:ro" in (
        compose["services"]["kairyu"]["volumes"]
    )
    assert spec["vllm"]["qwen"]["release"] == "v0.23.0"
    assert spec["vllm"]["qwen_mtp_speculative_tokens"] == 0
    qwen_options = kairyu["pools"]["qwen3.8-27b"]["replicas"][0]["options"]
    assert qwen_options["mtp_enabled"] is False
    assert {
        compose["services"][service]["image"]
        for service in ("qwen-0", "qwen-1", "qwen-2", "qwen-3")
    } == {f"${{QWEN_VLLM_IMAGE:-{spec['vllm']['qwen']['image']}}}"}
    assert compose["services"]["deepseek"]["image"] == (
        f"${{DEEPSEEK_VLLM_IMAGE:-{spec['vllm']['deepseek']['image']}}}"
    )
    for index in range(4):
        service = compose["services"][f"qwen-{index}"]
        devices = service["deploy"]["resources"]["reservations"]["devices"][0]
        assert devices["device_ids"] == [str(index)]
        assert "--tensor-parallel-size" not in service["command"]
        assert "--language-model-only" not in service["command"]
        assert json.loads(_option(service["command"], "--mm-processor-kwargs")) == {
            "min_pixels": 65_536,
            "max_pixels": 2_097_152,
        }
        assert _option(service["command"], "--limit-mm-per-prompt.image") == "1"
        assert _option(service["command"], "--limit-mm-per-prompt.video") == "0"
        # Medium-effort thinking is declared per role in auto-max.yaml, not
        # via a service-wide default.
        assert "--default-chat-template-kwargs" not in service["command"]
        assert _option(service["command"], "--max-num-seqs") == "32"
        assert _option(service["command"], "--max-num-batched-tokens") == "32768"
        assert _option(service["command"], "--kv-cache-dtype") == "fp8"
        assert json.loads(
            _option(service["command"], "--compilation-config")
        ) == {"cudagraph_mode": "PIECEWISE"}
        # Issue #509: c1/c4/c8 gains do not cover the deployed c16/c32
        # envelope, so Qwen MTP stays candidate-only.
        assert "--speculative-config" not in service["command"]
        assert service["volumes"][-2]["target"] == "/root/.cache"
        assert service["volumes"][-1] == (
            "./qwen3.8-chat.jinja:/etc/kairyu/qwen3.8-chat.jinja:ro"
        )
        assert service["environment"] | {
            "XDG_CACHE_HOME": "/root/.cache",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
            "TRITON_CACHE_DIR": "/root/.cache/triton",
            "TILELANG_CACHE_DIR": "/root/.cache/tilelang",
            "TILELANG_TMP_DIR": "/root/.cache/tilelang/tmp",
        } == service["environment"]
    deepseek = compose["services"]["deepseek"]
    devices = deepseek["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == ["4", "5", "6", "7"]
    assert _option(deepseek["command"], "--tensor-parallel-size") == "4"
    assert deepseek["ports"] == [
        "127.0.0.1:${DEEPSEEK_L1_PORT:-8005}:8000"
    ]
    assert "--enable-expert-parallel" in deepseek["command"]
    assert "--enable-auto-tool-choice" in deepseek["command"]
    assert _option(deepseek["command"], "--tool-call-parser") == "deepseek_v4"
    assert _option(deepseek["command"], "--max-num-batched-tokens") == "16384"
    # Scaffold passthrough plus inherited-effort preamble injection (DTO-D6).
    assert _option(deepseek["command"], "--chat-template") == (
        "/etc/kairyu/deepseek-role-effort.jinja"
    )
    assert (
        "./deepseek-role-effort.jinja:/etc/kairyu/deepseek-role-effort.jinja:ro"
        in deepseek["volumes"]
    )
    assert deepseek["volumes"][1]["target"] == "/root/.cache"
    assert deepseek["environment"] | {
        "XDG_CACHE_HOME": "/root/.cache",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
        "TRITON_CACHE_DIR": "/root/.cache/triton",
        "TILELANG_CACHE_DIR": "/root/.cache/tilelang",
        "TILELANG_TMP_DIR": "/root/.cache/tilelang/tmp",
    } == deepseek["environment"]
    assert json.loads(_option(deepseek["command"], "--speculative-config")) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "greedy",
    }
    assert json.loads(_option(deepseek["command"], "--compilation-config")) == {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "custom_ops": ["all"],
    }
    assert compose["services"]["kairyu"]["build"]["args"] == {
        "KAIRYU_VISION": "1",
        "KAIRYU_EMBEDDINGS": "1",
        "KAIRYU_EMBEDDING_MODEL_REPOSITORY": EMBEDDING_MODEL_REPOSITORY,
        "KAIRYU_EMBEDDING_MODEL_REVISION": EMBEDDING_MODEL_REVISION,
        "KAIRYU_EMBEDDING_MODEL_SHA256": EMBEDDING_MODEL_SHA256,
        "KAIRYU_EMBEDDING_PROVENANCE_SHA256": EMBEDDING_PROVENANCE_SHA256,
    }
    assert spec["embedding"] == {
        "served_name": "embed-small",
        "backend": "fastembed",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_path": "/opt/kairyu/models/all-MiniLM-L6-v2",
        "repository": EMBEDDING_MODEL_REPOSITORY,
        "revision": EMBEDDING_MODEL_REVISION,
        "model_sha256": EMBEDDING_MODEL_SHA256,
        "provenance_sha256": EMBEDDING_PROVENANCE_SHA256,
        "dimensions": 384,
        "batch_size": 64,
        "threads": 2,
        "max_concurrency": 2,
    }
    assert compose["services"]["kairyu"]["ports"] == [
        "${API_BIND_ADDRESS:-0.0.0.0}:${API_PORT:-8003}:8000"
    ]


def test_tiered_gateway_owns_l2_pools_templates_and_orchestrators() -> None:
    raw = (EXAMPLE / "kairyu.yaml").read_text()
    deployment = load_deployment_spec(raw, resolve_credentials=False)

    assert set(deployment.pools) == {
        "qwen3.8-27b",
        "deepseek-v4-flash-0731",
        "deepseek-v4-flash-0731-thinking",
    }
    qwen = deployment.pools["qwen3.8-27b"]
    assert len(qwen.replicas) == 4
    assert qwen.prefix_index is True
    assert qwen.queue_depth_threshold == 0
    for replica in qwen.replicas:
        validate_backend_options(replica.backend, replica.options)
        assert replica.options["tensor_parallel_size"] == 1
        assert replica.options["upstream"] == "vllm"
        assert replica.options["capabilities"] == {
            "allow_prompt_kinds": ["multimodal"],
            "allow_chat_template_kwargs": ["enable_thinking"],
        }
        assert replica.options["image_input_policy"]["max_images"] == 1
        assert replica.options["container_image_digest"] == (
            "sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
        )
    deepseek = deployment.pools["deepseek-v4-flash-0731"]
    assert len(deepseek.replicas) == 1
    validate_backend_options(deepseek.replicas[0].backend, deepseek.replicas[0].options)
    assert deepseek.replicas[0].options["tensor_parallel_size"] == 4
    assert deepseek.replicas[0].options["expert_parallel_size"] == 4
    assert deepseek.replicas[0].options["dspark_enabled"] is True
    assert deepseek.replicas[0].options["container_image_digest"] == (
        "sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1"
    )
    assert "completion_reasoning_end_tag" not in deepseek.replicas[0].options
    thinking = deployment.pools["deepseek-v4-flash-0731-thinking"]
    assert thinking.replicas[0].options["completion_reasoning_end_tag"] == "</think>"
    assert set(deployment.orchestrators) == {
        "kairyu-auto-max",
    }
    assert set(deployment.executors) == {"sandbox-python"}
    executor = deployment.executors["sandbox-python"]
    assert executor.base_url == "http://executor"
    assert executor.uds_path == "/run/kairyu-executor/executor.sock"
    assert executor.queue_wait_s == 8
    assert list(deployment.embeddings) == ["embed-small"]
    embedding = deployment.embeddings["embed-small"]
    assert embedding.backend == "fastembed"
    assert embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding.model_path == "/opt/kairyu/models/all-MiniLM-L6-v2"
    assert embedding.revision == EMBEDDING_MODEL_REVISION
    assert embedding.model_sha256 == EMBEDDING_MODEL_SHA256
    assert embedding.provenance_sha256 == EMBEDDING_PROVENANCE_SHA256
    assert embedding.dimensions == 384
    assert embedding.batch_size == 64
    assert embedding.threads == 2
    assert embedding.max_concurrency == 2
    assert deployment.public_models == frozenset(
        {"kairyu-auto-max", "embed-small"}
    )
    assert deployment.chat_templates == {
        "deepseek-v4-flash-0731": "/etc/kairyu/deepseek-v4-0731.jinja",
        "deepseek-v4-flash-0731-thinking": "/etc/kairyu/deepseek-thinking.jinja",
    }
    assert deployment.legacy_chat_models == frozenset({"qwen3.8-27b"})
    assert deployment.server.max_chat_body_bytes == 16_777_216


def test_deepseek_role_effort_template_grades_only_thinking_scaffolds() -> None:
    template = ChatTemplate.load(str(EXAMPLE / "deepseek-role-effort.jinja"))
    thinking = (
        "<｜begin▁of▁sentence｜><｜User｜>[critique] judge<｜Assistant｜><think>"
    )
    direct = (
        "<｜begin▁of▁sentence｜><｜User｜>[policies] write<｜Assistant｜></think>"
    )

    def render(content: str, effort: str | None) -> str:
        return template.render(
            [{"role": "user", "content": content}],
            template_kwargs=(
                {"reasoning_effort": effort} if effort is not None else None
            ),
        )

    assert render(thinking, None) == thinking
    assert render(thinking, "low") == thinking
    assert render(direct, "max") == direct
    high = render(thinking, "high")
    assert high.startswith(
        "<｜begin▁of▁sentence｜>Reason privately and carefully"
    )
    assert high.endswith("<｜Assistant｜><think>")
    maxed = render(thinking, "max")
    assert maxed.startswith(
        "<｜begin▁of▁sentence｜>Use rigorous private reasoning"
    )
    assert maxed.endswith("<｜User｜>[critique] judge<｜Assistant｜><think>")


def test_tiered_private_reasoning_prompt_converges_without_requesting_a_transcript() -> None:
    template = (EXAMPLE / "deepseek-thinking.jinja").read_text()

    assert "Reason privately and carefully" in template
    assert "converge promptly" in template
    assert "non-empty final answer immediately follows </think>" in template
    assert "entire deliberation process" not in template
    assert "Do not stop reasoning" not in template


def test_tiered_l2_pins_only_the_dual_track_dag() -> None:
    config = json.loads((EXAMPLE / "example.json").read_text())
    maximum = load_spec(EXAMPLE / "auto-max.yaml")

    assert [worker.name for worker in maximum.workers] == ["tier1", "tier2", "tier2-direct"]
    assert maximum.workers[0].engine_ref == "qwen3.8-27b"
    # Every primary-profile DeepSeek role thinks (DTO-D7) on tier2; the
    # non-thinking pool serves only the deepseek_direct route (DTO-D13).
    assert maximum.workers[1].engine_ref == "deepseek-v4-flash-0731-thinking"
    assert maximum.workers[2].engine_ref == "deepseek-v4-flash-0731"
    assert maximum.router.kind == "calibrated"
    assert maximum.router.target_mode == "auto-max"
    assert maximum.moa_samples == 0
    # DTO-D12: DeepSeek budgets halved for the Terminal-Bench turn envelope;
    # the ceiling admits the max tier.
    assert maximum.internal_max_tokens == 65536
    assert maximum.public_output_floor == 256
    assert maximum.expose_intermediate_outputs is True
    assert config["orchestration"]["internal_max_output_tokens"] == 65536
    assert config["orchestration"]["product_policy"] == "judged-five-route-dual-track"
    # DTO-D10 budget: 10 generation units + 1 empty-output re-dispatch
    # + 3 audit verdicts + 3 inconclusive re-verifies + 2 refinements.
    assert config["orchestration"]["product_normal_calls"] == 10
    assert config["orchestration"]["product_max_calls"] == 19
    assert config["orchestration"]["product_max_refinements"] == 2
    assert maximum.budget.max_steps == 19
    assert maximum.budget.max_refine_depth == 2
    expected_roles = list(config["orchestration"]["roles"])
    assert [role.name for role in maximum.roles] == expected_roles == [
        "head",
        "draft",
        "image_description",
        "policies",
        "answer_1",
        "answer_2",
        "answer_3",
        "answer_4",
        "critique",
        "synthesis",
        "audit",
    ]
    by_name = {role.name: role for role in maximum.roles}
    # Effort policy (DTO-D6/D14): Qwen thinking roles are fixed spec policy
    # — draft and the answerers think at medium (spec level `high` under the
    # L3 medium→high alias), the head declares no effort and stays
    # non-thinking. DeepSeek roles inherit the caller's L3 effort, and an
    # effortless request runs at the spec default of high.
    assert maximum.default_reasoning_effort == "high"
    efforts = {role.name: role.reasoning_effort for role in maximum.roles}
    assert efforts == {
        "head": None,
        "draft": "high",
        "image_description": "high",
        "answer_1": "high",
        "answer_2": "high",
        "answer_3": "high",
        "answer_4": "high",
        "policies": "inherit",
        "critique": "inherit",
        "synthesis": "inherit",
        "audit": "high",
    }
    head = by_name["head"]
    assert head.role_type == "head" and head.worker == "tier1"
    assert head.depends_on == ()
    assert config["orchestration"]["stream_head"] == "head"
    # Track B: quick Qwen draft, critically refined by thinking DeepSeek.
    draft = by_name["draft"]
    assert draft.worker == "tier1" and draft.depends_on == ()
    critique = by_name["critique"]
    assert critique.worker == "tier2"
    assert critique.depends_on == ("draft", "image_description")
    # DTO-D11: the Qwen image_description stage runs only on image requests
    # and feeds every text-only DeepSeek role; the Qwen answerers see the
    # image natively.
    image_description = by_name["image_description"]
    assert image_description.worker == "tier1"
    assert image_description.requires == "image"
    assert image_description.depends_on == ()
    for role_name in ("policies", "critique", "synthesis", "audit"):
        assert "{image_description}" in by_name[role_name].prompt
    # Track A: one thinking DeepSeek policy call fans out to four Qwen
    # answerers, one per replica, each bound to its policy by prompt.
    policies = by_name["policies"]
    assert policies.worker == "tier2" and policies.depends_on == ("image_description",)
    answerers = [by_name[f"answer_{index}"] for index in range(1, 5)]
    assert all(role.worker == "tier1" for role in answerers)
    assert all(role.depends_on == ("policies",) for role in answerers)
    assert {role.sampling.seed_offset for role in answerers} == {1, 2, 3, 4}
    for index, role in enumerate(answerers, start=1):
        assert f"POLICY {index}" in role.prompt
    # The merge is the selected final unit: it carries the caller's public
    # intent (no sampling override), continues the committed head opening,
    # weighs the five candidates as peers (DTO-D10), and renders
    # prompt_headless on head-disabled (tool/format) turns.
    synthesis = by_name["synthesis"]
    assert synthesis.worker == "tier2" and synthesis.role_type == "publisher"
    assert synthesis.depends_on == (
        "head",
        "critique",
        "answer_1",
        "answer_2",
        "answer_3",
        "answer_4",
        "image_description",
    )
    assert "UNTRUSTED CANDIDATE 5" in synthesis.prompt
    assert "REFINED ANSWER" not in synthesis.prompt
    # A thinking synthesis must not pre-close its reasoning span: with
    # reasoning_closed the private deliberation would be reclaimed as the
    # public answer.
    assert synthesis.reasoning_closed is False
    # DTO-D9: the public-output floor force-closes synthesis's deliberation
    # in the empty-output re-dispatch via this tag.
    assert synthesis.reasoning_close_tag == "</think>"
    assert synthesis.prompt_headless
    # DTO-D10/D14: the audit verifier gates publication of the final unit
    # with the first-line PASS/FAIL protocol; never greedy (issue #509). It
    # runs on Qwen at fixed medium effort with one think+verdict cap.
    audit = by_name["audit"]
    assert audit.role_type == "verifier" and audit.worker == "tier1"
    assert audit.verifies == "synthesis"
    assert audit.depends_on == ("synthesis", "head", "image_description")
    assert audit.sampling.temperature > 0.0
    assert "PASS" in audit.prompt and "FAIL" in audit.prompt
    assert audit.reasoning_effort == "high"
    assert audit.sampling.top_k == 20 and audit.sampling.max_tokens == 16384
    assert audit.sampling.max_tokens_by_effort is None
    assert audit.prompt_suffix == ""
    # DTO-D7: every DeepSeek role deliberates — all scaffolds end open.
    for role_name in ("policies", "critique", "synthesis"):
        assert by_name[role_name].prompt_suffix == "<｜Assistant｜><think>"
    # Untrusted-data delimiters (MoA pattern) guard every cross-role payload.
    for role_name in ("critique", "synthesis"):
        assert "UNTRUSTED" in by_name[role_name].prompt
    # DTO-D13: a Qwen non-thinking route judge selects one of five profiles
    # per request; the four direct routes are single publisher roles with the
    # official per-mode sampling fixed on the final unit and the
    # vendor-official output caps. No sandbox stages anywhere.
    assert not any(role.role_type == "executor" for role in maximum.roles)
    judge = maximum.profile_judge
    assert judge is not None
    assert judge.worker == "tier1" and judge.fallback == "primary"
    assert judge.timeout_seconds == 5.0 and judge.max_tokens == 8
    assert judge.prompt_prefix == "" and judge.prompt_suffix == ""
    expected_judge = config["orchestration"]["profile_judge"]
    assert expected_judge["worker"] == "tier1" and expected_judge["fallback"] == "primary"
    assert [
        {"label": choice.label, "profile": choice.profile} for choice in judge.choices
    ] == expected_judge["choices"] == [
        {"label": "QWEN", "profile": "qwen_direct"},
        {"label": "QWEN_THINK", "profile": "qwen_think_medium"},
        {"label": "DEEPSEEK", "profile": "deepseek_direct"},
        {"label": "DEEPSEEK_THINK", "profile": "deepseek_think"},
        {"label": "ENSEMBLE", "profile": "primary"},
    ]
    assert all(choice.criteria.strip() for choice in judge.choices)
    profiles = {profile.name: profile.roles for profile in maximum.profiles}
    assert {name: [role.name for role in roles] for name, roles in profiles.items()} == {
        name: list(roles) for name, roles in config["orchestration"]["profiles"].items()
    }
    assert all(len(roles) == 1 and roles[0].role_type == "publisher" for roles in profiles.values())
    finals = {name: roles[0] for name, roles in profiles.items()}
    assert config["orchestration"]["profile_final_roles"] == {
        "primary": "synthesis",
        **{name: role.name for name, role in finals.items()},
    }
    caps = config["orchestration"]["direct_route_max_tokens"]
    assert {name: role.sampling.max_tokens for name, role in finals.items()} == caps == {
        "qwen_direct": 131072,
        "qwen_think_medium": 131072,
        "deepseek_direct": 393216,
        "deepseek_think": 393216,
    }
    sampling = {
        name: {
            key: value
            for key, value in role.sampling.model_dump().items()
            if value not in (None, ())
        }
        for name, role in finals.items()
    }
    assert sampling == config["orchestration"]["direct_route_sampling"]
    qwen_answer = finals["qwen_direct"]
    assert qwen_answer.worker == "tier1" and qwen_answer.reasoning_effort is None
    assert (qwen_answer.sampling.temperature, qwen_answer.sampling.top_p) == (0.7, 0.8)
    assert qwen_answer.sampling.top_k == 20 and qwen_answer.sampling.presence_penalty == 1.5
    qwen_think = finals["qwen_think_medium"]
    assert qwen_think.worker == "tier1" and qwen_think.reasoning_effort == "high"
    assert (qwen_think.sampling.temperature, qwen_think.sampling.top_p) == (1.0, 0.95)
    assert qwen_think.sampling.top_k == 20 and qwen_think.sampling.presence_penalty is None
    deepseek_answer = finals["deepseek_direct"]
    assert deepseek_answer.worker == "tier2-direct"
    assert deepseek_answer.reasoning_effort is None
    assert deepseek_answer.reasoning_closed is True
    assert deepseek_answer.prompt_suffix == "<｜Assistant｜></think>"
    deepseek_think = finals["deepseek_think"]
    assert deepseek_think.worker == "tier2" and deepseek_think.reasoning_effort == "inherit"
    assert deepseek_think.prompt_suffix == "<｜Assistant｜><think>"
    assert deepseek_think.reasoning_close_tag == "</think>"
    assert deepseek_think.reasoning_closed is False
    for role in (deepseek_answer, deepseek_think):
        assert (role.sampling.temperature, role.sampling.top_p) == (1.0, 0.95)
        assert role.prompt.startswith("<｜begin▁of▁sentence｜><｜User｜>")
    for role in finals.values():
        assert "{query}" in role.prompt and role.depends_on == ()
        assert "reply format the conversation demands" in " ".join(role.prompt.split())
    assert sorted(path.name for path in EXAMPLE.glob("auto*.yaml")) == ["auto-max.yaml"]
    assert "base_url: http://kairyu:8000/v1" not in (EXAMPLE / "auto-max.yaml").read_text()


def test_tiered_chat_ui_calls_kairyu_l3() -> None:
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    ui = compose["services"]["chat-ui"]
    assert ui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert json.loads(ui["environment"]["DEFAULT_MODEL_PARAMS"]) == {
        "max_tokens": 65536,
        "stream_response": False,
    }
    assert ui["environment"] | {
        "DEFAULT_MODELS": "kairyu-auto-max",
        "OPENAI_API_CONFIGS": '{"0":{"model_ids":["kairyu-auto-max"]}}',
        "ENABLE_PERSISTENT_CONFIG": "false",
        "RESET_CONFIG_ON_START": "true",
        "ENABLE_SIGNUP": "false",
        "ENABLE_LOGIN_FORM": "false",
        "WEBUI_AUTH": "false",
        "ENABLE_EVALUATION_ARENA_MODELS": "false",
    } == ui["environment"]
    assert ui["ports"] == ["${CHAT_UI_BIND_ADDRESS:-0.0.0.0}:${CHAT_UI_PORT:-3000}:8080"]
    assert ui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {
        "qwen-0": {"condition": "service_healthy"},
        "qwen-1": {"condition": "service_healthy"},
        "qwen-2": {"condition": "service_healthy"},
        "qwen-3": {"condition": "service_healthy"},
        "deepseek": {"condition": "service_healthy"},
        "executor": {"condition": "service_healthy"},
    }
    policy_mounts = {
        volume for volume in compose["services"]["kairyu"]["volumes"]
        if isinstance(volume, str) and "/auto" in volume
    }
    assert policy_mounts == {"./auto-max.yaml:/etc/kairyu/auto-max.yaml:ro"}


def _embedding_response() -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.0] * 384},
            {"object": "embedding", "index": 1, "embedding": [1.0] * 384},
        ],
        "model": "embed-small",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
    }


def test_tiered_embedding_smoke_accepts_two_finite_vectors() -> None:
    control = _load(EXAMPLE / "control.py", "tiered_embedding_smoke_valid")

    control._validate_embedding_smoke(_embedding_response())


def test_tiered_readiness_posts_two_input_embedding_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_embedding_readiness")
    posts: list[tuple[str, dict]] = []

    def fake_json(url: str) -> dict:
        if url.endswith("/readyz"):
            return {"status": "ready"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "kairyu-auto-max"}, {"id": "embed-small"}]}
        assert url.endswith("/routing")
        return {
            "models": {
                "kairyu-auto-max": {
                    "roles": [
                        {"name": name}
                        for name in (
                            "head",
                            "draft",
                            "image_description",
                            "policies",
                            "answer_1",
                            "answer_2",
                            "answer_3",
                            "answer_4",
                            "critique",
                            "synthesis",
                            "audit",
                        )
                    ],
                    "profiles": {
                        profile: [
                            {
                                "name": roles[0],
                                "sampling": control.SPEC["orchestration"][
                                    "direct_route_sampling"
                                ][profile],
                            }
                        ]
                        for profile, roles in control.SPEC["orchestration"][
                            "profiles"
                        ].items()
                    },
                    "profile_judge": {
                        "worker": "tier1",
                        "fallback": "primary",
                        "choices": [
                            {"label": label, "profile": profile, "criteria": "x"}
                            for label, profile in (
                                ("QWEN", "qwen_direct"),
                                ("QWEN_THINK", "qwen_think_medium"),
                                ("DEEPSEEK", "deepseek_direct"),
                                ("DEEPSEEK_THINK", "deepseek_think"),
                                ("ENSEMBLE", "primary"),
                            )
                        ],
                    },
                    "stream_head": "head",
                    "moa_samples": 0,
                    "budget": {"max_steps": 19, "max_refine_depth": 2},
                    "expose_intermediate_outputs": True,
                    "configured_engines": {
                        "tier1": {"model": "qwen3.8-27b"},
                        "tier2": {"model": "deepseek-v4-flash-0731-thinking"},
                        "tier2-direct": {"model": "deepseek-v4-flash-0731"},
                    },
                }
            }
        }

    def fake_post(url: str, payload: dict) -> dict:
        posts.append((url, payload))
        return _embedding_response() if url.endswith("/v1/embeddings") else {"count": 1}

    monkeypatch.setattr(control, "_json_url", fake_json)
    monkeypatch.setattr(control, "_post_json_url", fake_post)

    control._validate_ready("http://api.test", "http://tokenizer.test/tokenize")

    assert posts == [
        (
            "http://api.test/v1/embeddings",
            {
                "model": "embed-small",
                "input": ["kairyu readiness probe", "two-input contract"],
                "encoding_format": "float",
            },
        ),
        (
            "http://tokenizer.test/tokenize",
            {"model": "deepseek-v4-flash-0731", "prompt": "kairyu"},
        ),
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "model identity"),
        ("indices", "indices"),
        ("dimensions", "384 dimensions"),
        ("finite", "finite numbers"),
        ("usage", "positive exact usage"),
    ],
)
def test_tiered_embedding_smoke_rejects_malformed_contract(
    mutation: str,
    message: str,
) -> None:
    control = _load(EXAMPLE / "control.py", f"tiered_embedding_smoke_{mutation}")
    response = _embedding_response()
    if mutation == "model":
        response["model"] = "wrong-model"
    elif mutation == "indices":
        response["data"][1]["index"] = 2
    elif mutation == "dimensions":
        response["data"][1]["embedding"].pop()
    elif mutation == "finite":
        response["data"][0]["embedding"][0] = float("inf")
    else:
        response["usage"]["prompt_tokens"] = 0

    with pytest.raises(SystemExit, match=message):
        control._validate_embedding_smoke(response)


class _FakeWebUI:
    """Model the pinned Open WebUI function API, including flip-only toggles."""

    def __init__(self) -> None:
        self.functions: dict[str, dict] = {}
        self.calls: list[str] = []
        self.role = "admin"

    def __call__(self, ui_url, path, *, token=None, payload=None, method=None):
        self.calls.append(path)
        if path == "/api/v1/auths/signin":
            return {"role": self.role, "token": "session-token"}
        assert token == "session-token"
        if path == "/api/v1/functions/":
            return [dict(row) for row in self.functions.values()]
        if path == "/api/v1/functions/create":
            row = {**payload, "is_active": False, "is_global": False}
            self.functions[payload["id"]] = row
            return dict(row)
        prefix = "/api/v1/functions/id/"
        assert path.startswith(prefix)
        function_id, _, action = path[len(prefix) :].partition("/")
        row = self.functions[function_id]
        if action == "update":
            row.update(payload)
            return dict(row)
        if action in {"toggle", "toggle/global"}:
            key = "is_active" if action == "toggle" else "is_global"
            row[key] = not row[key]
            return dict(row)
        assert action == "valves/user/spec"
        return {
            "properties": {
                "reasoning_effort": {"enum": ["default", "low", "high", "max"]}
            }
        }


def test_tiered_chat_ui_effort_filter_controls_request_body() -> None:
    module = _load(
        EXAMPLE / "webui-reasoning-effort-filter.py",
        "tiered_chat_ui_effort_filter",
    )
    selector = module.Filter()

    for effort in ("low", "high", "max"):
        body = {"reasoning_effort": "stale"}
        user = {"valves": selector.UserValves(reasoning_effort=effort)}
        assert selector.inlet(body, user) == {"reasoning_effort": effort}

    body = {"reasoning_effort": "max"}
    user = {"valves": selector.UserValves(reasoning_effort="default")}
    assert selector.inlet(body, user) == {}


def test_tiered_chat_ui_effort_selector_provision_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_chat_ui_effort_selector")
    webui = _FakeWebUI()
    monkeypatch.setattr(control, "_webui_api", webui)

    control._provision_chat_ui_effort_selector("http://127.0.0.1:3000")
    row = webui.functions["reasoning_effort"]
    assert row["is_active"] and row["is_global"]
    assert "class Filter" in row["content"]
    assert webui.calls.count("/api/v1/functions/create") == 1
    toggles = [call for call in webui.calls if call.endswith(("toggle", "toggle/global"))]
    assert len(toggles) == 2

    # Re-provisioning must refresh content without flipping activation off.
    webui.calls.clear()
    control._provision_chat_ui_effort_selector("http://127.0.0.1:3000")
    assert webui.functions["reasoning_effort"]["is_active"]
    assert webui.functions["reasoning_effort"]["is_global"]
    assert "/api/v1/functions/create" not in webui.calls
    assert "/api/v1/functions/id/reasoning_effort/update" in webui.calls
    assert not [call for call in webui.calls if call.endswith(("toggle", "toggle/global"))]

    webui.role = "user"
    with pytest.raises(SystemExit, match="admin"):
        control._provision_chat_ui_effort_selector("http://127.0.0.1:3000")


def test_tiered_control_requires_exact_eight_gpu_inventory() -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_control")
    text = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, "
        f"00000000:{16 + index:02x}:00.0"
        for index in range(8)
    )
    rows = control._gpu_inventory(text)
    assert sorted(rows) == list(range(8))


def test_tiered_control_uses_explicit_public_ui_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_public_host")
    monkeypatch.setenv("PUBLIC_HOST", "gpu.example.test")
    storage = {
        "qwen_models": tmp_path / "qwen-models",
        "deepseek_models": tmp_path / "deepseek-models",
        "webui": tmp_path / "webui",
        "deepseek_cache": tmp_path / "deepseek-cache",
        **{
            f"qwen_cache_{index}": tmp_path / f"qwen-cache-{index}"
            for index in range(4)
        },
    }
    monkeypatch.setattr(control, "_storage_paths", lambda: storage)

    assert control._public_ui_host() == "gpu.example.test"
    assert control._compose_env()["API_BIND_ADDRESS"] == "0.0.0.0"
    monkeypatch.setenv("API_BIND_ADDRESS", "127.0.0.1")
    assert control._compose_env()["API_BIND_ADDRESS"] == "127.0.0.1"
    assert control._compose_env()["CHAT_UI_BIND_ADDRESS"] == "0.0.0.0"
    assert control._compose_env()["DEEPSEEK_L1_PORT"] == "8005"


def test_tiered_control_rejects_persistent_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_nvme")
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        control._nvme_root()


def test_tiered_verification_rejects_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        _load(EXAMPLE / "verification.py", "tiered_example_verification_nvme")



def test_tiered_product_serving_requires_head_and_synthesis_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "verification.py", "tiered_example_product_verification")
    observed: list[tuple[str, dict]] = []

    def fake_serving(model, _run_dir, **kwargs):
        observed.append((model, kwargs))
        return 0

    monkeypatch.setattr(benchmark, "_serving", fake_serving)
    assert benchmark.serving_auto_max(tmp_path) == 0

    assert observed == [
        (
            "kairyu-auto-max",
            {
                "tensor_parallel": 4,
                "replicas": 1,
                "expected_route": "synthesis",
                "expected_role": "publisher",
                "expected_kind": "generation",
                "warmup_requests": 4,
                "natural_completion": True,
                "require_head": True,
                "expected_generation_nodes": (
                    "draft",
                    "policies",
                    "answer_1",
                    "answer_2",
                    "answer_3",
                    "answer_4",
                    "critique",
                ),
                "expected_verification_nodes": ("audit",),
                "judged_routes": True,
            },
        )
    ]


def _write_product_serving_result(row_dir: Path, stages: list[dict]) -> None:
    result = {
        "summary": {
            "requests": 1,
            "completion_tokens_total": 32,
            "output_tokens_per_s": 1.0,
        },
        "samples": [
            {
                "completion_tokens": 32,
                "trace": {"status": "valid", "stages": stages},
            }
        ],
    }
    (row_dir / "result-serving.json").write_text(json.dumps(result))


@pytest.mark.parametrize(
    "node",
    [
        "draft",
        "policies",
        "answer_1",
        "answer_2",
        "answer_3",
        "answer_4",
        "critique",
        "audit",
    ],
)
@pytest.mark.parametrize("failure_mode", ["missing", "failed"])
def test_tiered_product_serving_requires_every_internal_stage(
    node: str,
    failure_mode: str,
    tmp_path: Path,
) -> None:
    benchmark = _load(
        EXAMPLE / "verification.py",
        f"tiered_product_stage_{node}_{failure_mode}",
    )
    stages = [
        {"node": "head", "kind": "generation", "status": "success"},
        {
            "node": "synthesis",
            "role": "publisher",
            "kind": "generation",
            "status": "success",
        },
        *[
            {"node": name, "kind": "generation", "status": "success"}
            for name in benchmark._DUAL_TRACK_INTERNAL_NODES
        ],
        *[
            {"node": name, "role": "verifier", "kind": "verification", "status": "success"}
            for name in benchmark._DUAL_TRACK_VERIFICATION_NODES
        ],
    ]

    def validate() -> int:
        return benchmark._validate_serving_row(
            tmp_path,
            1,
            32,
            expected_route="synthesis",
            expected_role="publisher",
            require_head=True,
            expected_generation_nodes=benchmark._DUAL_TRACK_INTERNAL_NODES,
            expected_verification_nodes=benchmark._DUAL_TRACK_VERIFICATION_NODES,
        )

    _write_product_serving_result(tmp_path, stages)
    assert validate() == 0

    if failure_mode == "missing":
        stages = [stage for stage in stages if stage["node"] != node]
    else:
        next(stage for stage in stages if stage["node"] == node)["status"] = "failed"
    _write_product_serving_result(tmp_path, stages)

    assert validate() == 1


def test_tiered_product_serving_judged_routes_accept_one_direct_final(
    tmp_path: Path,
) -> None:
    """DTO-D13: with judged routes every sample must trace the judge and
    exactly one profile's final unit; direct routes carry no head/internal
    stage contract, while primary keeps the full dual-track contract."""

    benchmark = _load(EXAMPLE / "verification.py", "tiered_product_judged_routes")
    judge = {"node": "profile_judge", "kind": "classification", "status": "success"}
    direct = {
        "node": "qwen_think_answer",
        "role": "publisher",
        "kind": "generation",
        "status": "success",
    }
    primary = [
        {"node": "head", "kind": "generation", "status": "success"},
        {"node": "synthesis", "role": "publisher", "kind": "generation", "status": "success"},
        *[
            {"node": name, "kind": "generation", "status": "success"}
            for name in benchmark._DUAL_TRACK_INTERNAL_NODES
        ],
        *[
            {"node": name, "role": "verifier", "kind": "verification", "status": "success"}
            for name in benchmark._DUAL_TRACK_VERIFICATION_NODES
        ],
    ]

    def validate() -> int:
        return benchmark._validate_serving_row(
            tmp_path,
            1,
            32,
            expected_route="synthesis",
            expected_role="publisher",
            require_head=True,
            expected_generation_nodes=benchmark._DUAL_TRACK_INTERNAL_NODES,
            expected_verification_nodes=benchmark._DUAL_TRACK_VERIFICATION_NODES,
            judged_routes=True,
        )

    _write_product_serving_result(tmp_path, [judge, direct])
    assert validate() == 0
    routes = json.loads((tmp_path / "routes.json").read_text())
    assert routes["routes"] == {"qwen_think_medium": {"requests": 1, "ttft_p50_ms": None}}
    _write_product_serving_result(tmp_path, [judge, *primary])
    assert validate() == 0
    # No judge stage, an ambiguous pair of finals, or a primary sample missing
    # its internal contract all fail the row.
    _write_product_serving_result(tmp_path, [direct])
    assert validate() == 1
    _write_product_serving_result(tmp_path, [judge, direct, primary[1]])
    assert validate() == 1
    _write_product_serving_result(tmp_path, [judge, *primary[:-1]])
    assert validate() == 1


def test_tiered_coding_gate_fails_when_ttft_exceeds_double_direct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "verification.py", "tiered_example_coding_gate")
    summaries: dict[str, dict] = {}
    validations: list[dict] = []

    def validate(*_args, **kwargs):
        validations.append(kwargs)
        return 0

    monkeypatch.setattr(benchmark, "_bench_row", lambda **kwargs: 0)
    monkeypatch.setattr(benchmark, "_validate_serving_row", validate)
    monkeypatch.setattr(
        benchmark,
        "_row_summary",
        lambda row_dir: summaries.get(row_dir.name),
    )
    monkeypatch.setitem(
        benchmark.SPEC["verification"]["coding"], "concurrency", [1]
    )

    # Within 2x the paired DeepSeek-direct row: PASS.
    summaries["coding-c1"] = {"ttft_p50_ms": 1500.0}
    summaries["deepseek-direct-c1"] = {"ttft_p50_ms": 800.0}
    assert benchmark.serving_auto_max_coding(tmp_path / "pass") == 0
    gate = json.loads((tmp_path / "pass" / "ttft-gate.json").read_text())
    assert gate["gates"]["1"]["passed"] is True
    assert gate["gates"]["1"]["denominator_source"] == "paired_direct"

    # Beyond 2x: the row fails the run.
    summaries["coding-c1"] = {"ttft_p50_ms": 1700.0}
    assert benchmark.serving_auto_max_coding(tmp_path / "fail") == 1

    # A missing paired row falls back to the pinned example.json denominator.
    summaries["coding-c1"] = {"ttft_p50_ms": 1500.0}
    del summaries["deepseek-direct-c1"]
    assert benchmark.serving_auto_max_coding(tmp_path / "fallback") == 0
    gate = json.loads((tmp_path / "fallback" / "ttft-gate.json").read_text())
    assert gate["gates"]["1"]["denominator_source"] == "pinned_fallback"
    assert validations
    assert all(row["judged_routes"] is True for row in validations)
    assert all(row["expected_route"] == "synthesis" for row in validations)
    assert all(
        row["expected_generation_nodes"] == benchmark._DUAL_TRACK_INTERNAL_NODES
        for row in validations
    )
    assert all(
        row["expected_verification_nodes"] == benchmark._DUAL_TRACK_VERIFICATION_NODES
        for row in validations
    )

    # DTO-D13: with a per-row route report the gate measures only the
    # TTFT-gated routes; a thinking direct route is reported, not gated.
    summaries["deepseek-direct-c1"] = {"ttft_p50_ms": 800.0}
    reports: dict[str, dict] = {}
    monkeypatch.setattr(benchmark, "_row_routes", lambda row_dir: reports.get(row_dir.name))
    reports["coding-c1"] = {
        "routes": {
            "primary": {"requests": 2, "ttft_p50_ms": 1500.0},
            "deepseek_think": {"requests": 30, "ttft_p50_ms": 9000.0},
        },
        "judge_total_ms_p50": 120.0,
    }
    assert benchmark.serving_auto_max_coding(tmp_path / "routed") == 0
    gate = json.loads((tmp_path / "routed" / "ttft-gate.json").read_text())["gates"]["1"]
    assert gate["product_semantic_ttft_p50_ms"] == 1500.0
    assert gate["status"] == "gated" and gate["passed"] is True
    assert gate["routes"]["deepseek_think"]["requests"] == 30
    reports["coding-c1"] = {
        "routes": {"deepseek_think": {"requests": 32, "ttft_p50_ms": 9000.0}},
        "judge_total_ms_p50": 120.0,
    }
    assert benchmark.serving_auto_max_coding(tmp_path / "na") == 0
    gate = json.loads((tmp_path / "na" / "ttft-gate.json").read_text())["gates"]["1"]
    assert gate["status"] == "not_applicable" and gate["passed"] is None

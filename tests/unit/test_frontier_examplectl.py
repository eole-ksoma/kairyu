from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.entrypoints.chat_template import ChatTemplate

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/deepseek-v4-flash-0731-8gpu"
QWEN_EXAMPLE = ROOT / "examples/qwen3.8-27b-1gpu"


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_examples_surface_contains_the_seven_hardware_examples() -> None:
    environments = sorted(
        path.name
        for path in (ROOT / "examples").iterdir()
        if path.is_dir() and (path / "example.json").is_file()
    )
    assert environments == [
        "deepseek-v4-flash-0731-8gpu",
        "deepseek-v4-flash-0731-dp2-8gpu",
        "deepseek-v4-flash-vision-exp-dp2-8gpu",
        "deepseek-v4.1-flash-8gpu",
        "qwen3.8-27b-1gpu",
        "qwen3.8-27b-dp8-8gpu",
        "qwen3.8-deepseek-v4-8gpu",
        "qwen3.8-flash-next-dp2-8gpu",
    ]
    qwen = json.loads((QWEN_EXAMPLE / "example.json").read_text())
    deepseek = json.loads((EXAMPLE / "example.json").read_text())
    assert qwen["vllm"]["release"] == "v0.23.0"
    assert deepseek["vllm"]["source_revision"] == (
        "aa0d51302747ea80f282e26949708b3253409fe2"
    )


def test_qwen_example_is_one_gpu_kairyu_to_vllm_to_webui_on_nvme() -> None:
    spec = json.loads((QWEN_EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((QWEN_EXAMPLE / "compose.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 1,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert spec["storage"]["root"].startswith("/mnt/nvme/")
    assert {
        key: spec["model"][key] for key in ("repo", "revision", "served_name")
    } == {
        "repo": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "served_name": "qwen3.8-27b",
    }
    assert spec["model"]["tree_sha256"] == (
        "9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e"
    )
    assert set(compose["services"]) == {"vllm", "kairyu", "chat-ui"}
    assert spec["vllm"]["release"] == "v0.23.0"
    assert compose["services"]["vllm"]["image"] == (
        f"${{VLLM_IMAGE:-{spec['vllm']['image']}}}"
    )
    webui = compose["services"]["chat-ui"]
    assert webui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert json.loads(webui["environment"]["DEFAULT_MODEL_PARAMS"]) == {
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert webui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {"vllm": {"condition": "service_healthy"}}
    devices = compose["services"]["vllm"]["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == ["${GPU_ID:-0}"]
    for service in ("vllm", "kairyu"):
        model_mount = compose["services"][service]["volumes"][0]
        assert model_mount["source"] == "${MODEL_STORAGE_PATH}"
    assert compose["services"]["vllm"]["volumes"][1]["source"] == "${VLLM_CACHE_PATH}"
    assert webui["volumes"][0]["source"] == "${WEBUI_STORAGE_PATH}"
    assert compose["services"]["kairyu"]["build"]["args"] == {"KAIRYU_VISION": "1"}


def test_qwen_vllm_command_pins_the_sm120_latency_throughput_contract() -> None:
    compose = yaml.safe_load((QWEN_EXAMPLE / "compose.yaml").read_text())
    service = compose["services"]["vllm"]
    command = service["command"]

    expected_pairs = {
        "--max-model-len": "262144",
        "--dtype": "bfloat16",
        "--kv-cache-dtype": "fp8",
        "--mamba-cache-dtype": "float16",
        "--mamba-ssm-cache-dtype": "float16",
        "--mamba-cache-mode": "align",
        "--max-num-batched-tokens": "32768",
        "--max-num-seqs": "32",
        "--attention-backend": "FLASHINFER",
    }
    for option, value in expected_pairs.items():
        assert command[command.index(option) + 1] == value
    assert "--language-model-only" not in command
    assert json.loads(command[command.index("--mm-processor-kwargs") + 1]) == {
        "min_pixels": 65_536,
        "max_pixels": 2_097_152,
    }
    assert command[command.index("--limit-mm-per-prompt.image") + 1] == "1"
    assert command[command.index("--limit-mm-per-prompt.video") + 1] == "0"
    assert "--enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command
    assert "--enable-flashinfer-autotune" in command
    assert "--enforce-eager" not in command
    assert "--speculative-config" not in command
    assert command[command.index("--chat-template") + 1] == (
        "/etc/kairyu/qwen3.8-chat.jinja"
    )
    assert json.loads((QWEN_EXAMPLE / "example.json").read_text())["vllm"][
        "mtp_speculative_tokens"
    ] == 0
    assert (
        json.loads(command[command.index("--compilation-config") + 1])["cudagraph_mode"]
        == "PIECEWISE"
    )
    assert "VLLM_USE_FASTOKENS" not in service["environment"]


def test_qwen_kairyu_l3_declares_strict_multimodal_vllm_l1() -> None:
    raw = (QWEN_EXAMPLE / "kairyu.yaml").read_text()
    spec = load_deployment_spec(raw)
    assert set(spec.engines) == {"qwen3.8-27b"}
    entry = spec.engines["qwen3.8-27b"]
    assert entry.backend == "openai"
    validate_backend_options(entry.backend, entry.options)
    assert entry.options["upstream"] == "vllm"
    assert entry.options["capabilities"] == {
        "allow_prompt_kinds": ["multimodal"],
        "allow_chat_template_kwargs": ["enable_thinking"],
    }
    assert entry.options["image_input_policy"] == {
        "max_images": 1,
        "max_image_bytes": 8_388_608,
        "max_total_image_bytes": 8_388_608,
        "max_image_pixels": 2_097_152,
        "max_total_image_pixels": 2_097_152,
        "max_image_dimension": 4096,
        "max_image_aspect_ratio": 200,
        "max_processed_prompt_tokens": 262_144,
    }
    assert entry.options["tensor_parallel_size"] == 1
    assert entry.options["mtp_enabled"] is False
    assert spec.server.max_chat_body_bytes == 16_777_216
    assert spec.chat_templates == {}
    assert spec.legacy_chat_models == frozenset({"qwen3.8-27b"})


def test_qwen_template_defaults_chat_to_direct_and_allows_explicit_thinking() -> None:
    template = ChatTemplate.load(str(QWEN_EXAMPLE / "chat_template.jinja"))
    messages = [{"role": "user", "content": "Return OK."}]
    direct = template.render(messages)
    thinking = template.render(
        messages,
        template_kwargs={"reasoning_effort": "max", "thinking_mode": "thinking"},
    )
    assert direct.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert thinking.endswith("<|im_start|>assistant\n<think>\n")


def test_qwen_control_selects_one_gpu_and_forces_nvme_storage() -> None:
    control = _load(QWEN_EXAMPLE / "control.py", "qwen_example_control")
    text = "0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, 0000:16:00.0"
    rows = control._gpu_inventory(text)
    assert rows[0]["memory_mib"] == 97887
    assert control.SPEC["storage"]["root"] == "/mnt/nvme/kairyu"


def test_example_is_exact_eight_gpu_kairyu_to_vllm_to_webui() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 8,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert set(compose["services"]) == {"vllm", "kairyu", "chat-ui"}
    assert spec["vllm"]["source_revision"] == (
        "aa0d51302747ea80f282e26949708b3253409fe2"
    )
    assert compose["services"]["vllm"]["image"] == (
        f"${{VLLM_IMAGE:-{spec['vllm']['image']}}}"
    )
    webui = compose["services"]["chat-ui"]
    assert webui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert json.loads(webui["environment"]["DEFAULT_MODEL_PARAMS"]) == {"max_tokens": 32768}
    assert webui["ports"] == ["${CHAT_UI_BIND_ADDRESS:-127.0.0.1}:${CHAT_UI_PORT:-3000}:8080"]
    assert compose["services"]["kairyu"]["ports"] == ["127.0.0.1:${API_PORT:-8002}:8000"]
    assert webui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {"vllm": {"condition": "service_healthy"}}
    devices = compose["services"]["vllm"]["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == [str(index) for index in range(8)]


def test_vllm_command_pins_the_optimized_sm120_contract() -> None:
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    command = compose["services"]["vllm"]["command"]

    expected_pairs = {
        "--max-model-len": "1048576",
        "--kv-cache-dtype": "fp8",
        "--block-size": "256",
        "--tensor-parallel-size": "8",
        "--max-num-seqs": "64",
        "--max-num-batched-tokens": "16384",
    }
    for option, value in expected_pairs.items():
        assert command[command.index(option) + 1] == value
    assert "--enable-expert-parallel" in command
    assert "--moe-backend" not in command
    assert compose["services"]["vllm"]["environment"]["VLLM_USE_DEEP_GEMM"] == "0"
    assert "--enable-prefix-caching" in command
    assert "--enforce-eager" not in command
    assert json.loads(command[command.index("--attention-config") + 1]) == {
        "use_fp4_indexer_cache": False
    }
    assert (
        json.loads(command[command.index("--speculative-config") + 1])["num_speculative_tokens"]
        == 5
    )
    assert (
        json.loads(command[command.index("--speculative-config") + 1])["method"]
        == "dspark"
    )
    assert (
        json.loads(command[command.index("--compilation-config") + 1])["cudagraph_mode"]
        == "FULL_AND_PIECEWISE"
    )


def test_kairyu_l3_declares_the_attested_vllm_l1() -> None:
    raw = (EXAMPLE / "kairyu.yaml").read_text()
    spec = load_deployment_spec(raw)
    assert set(spec.engines) == {"deepseek-v4-flash-0731"}
    entry = spec.engines["deepseek-v4-flash-0731"]
    assert entry.backend == "openai"
    validate_backend_options(entry.backend, entry.options)
    assert entry.options["upstream"] == "vllm"
    assert entry.options["allow_templated_chat_passthrough"] is True
    assert entry.options["tensor_parallel_size"] == 8
    assert entry.options["expert_parallel_size"] == 8
    assert entry.options["dspark_enabled"] is True
    assert spec.chat_templates == {"deepseek-v4-flash-0731": "/etc/kairyu/deepseek-v4-0731.jinja"}


def test_deepseek_template_matches_checkpoint_text_encoding() -> None:
    template = ChatTemplate.load(str(EXAMPLE / "deepseek-v4-0731.jinja"))
    maximum = (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to "
        "chance: exhaustively decompose the problem into its most fundamental components, "
        "trace every causal chain to its root, and resolve the underlying cause rather "
        "than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the solution from "
        "multiple angles and are certain that no assumption remains unchecked and no "
        "error remains undiscovered.\n\n"
    )
    rendered = template.render(
        [{"role": "user", "content": "Write Python."}],
        template_kwargs={"reasoning_effort": "max", "thinking_mode": "thinking"},
    )
    assert rendered == (
        "<｜begin▁of▁sentence｜>" + maximum + "<｜User｜>Write Python.<｜Assistant｜><think>"
    )

    multi = template.render(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
    )
    assert multi == (
        "<｜begin▁of▁sentence｜><｜User｜>A<｜Assistant｜></think>"
        "B<｜end▁of▁sentence｜><｜User｜>C<｜Assistant｜></think>"
    )


def test_deepseek_template_ignores_client_tool_metadata() -> None:
    template = ChatTemplate.load(str(EXAMPLE / "deepseek-v4-0731.jinja"))
    expected = "<｜begin▁of▁sentence｜><｜User｜>PAC1の適応は？<｜Assistant｜></think>"
    for tools in (
        [],
        [{"type": "function", "function": {"name": "search"}}],
    ):
        assert (
            template.render([{"role": "user", "content": "PAC1の適応は？"}], tools=tools)
            == expected
        )


def test_control_preflight_requires_exact_gpu_inventory() -> None:
    control = _load(EXAMPLE / "control.py", "deepseek_example_control")
    text = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0" for index in range(8)
    )
    rows = control._gpu_inventory(text)
    assert len(rows) == 8
    assert rows[-1]["index"] == 7



def test_qwen_serving_prompts_do_not_share_the_first_prefix(tmp_path: Path) -> None:
    benchmark = _load(QWEN_EXAMPLE / "verification.py", "qwen_example_serving")
    dataset = tmp_path / "serving.json"
    benchmark._serving_dataset(dataset, requests=4, approximate_tokens=32)
    rows = json.loads(dataset.read_text())
    prompts = [row["conversations"][0]["value"] for row in rows]
    assert len(set(prompts)) == 4
    assert [prompt.split(":", 1)[0] for prompt in prompts] == [
        "Case 0",
        "Case 1",
        "Case 2",
        "Case 3",
    ]


def test_qwen_serving_validation_rejects_partial_streams(tmp_path: Path) -> None:
    benchmark = _load(QWEN_EXAMPLE / "verification.py", "qwen_example_serving_validation")
    row_dir = tmp_path / "serving-c2"
    row_dir.mkdir()
    result = {
        "summary": {
            "requests": 2,
            "completion_tokens_total": 8,
            "output_tokens_per_s": 1.0,
        },
        "samples": [{"completion_tokens": 4}, {"completion_tokens": 4}],
    }
    output = row_dir / "result-serving.json"
    output.write_text(json.dumps(result))
    assert benchmark._validate_serving_row(row_dir, requests=2, output_tokens=4) == 0

    result["summary"]["completion_tokens_total"] = None
    result["samples"][1]["completion_tokens"] = 0
    output.write_text(json.dumps(result))
    assert benchmark._validate_serving_row(row_dir, requests=2, output_tokens=4) == 1

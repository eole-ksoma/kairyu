#!/usr/bin/env python3
"""One-command lifecycle for DeepSeek-V4.1-Flash as one TP2 x DP3 / EP6 replica on six GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
ALLOCATION = SPEC["allocation"]
GPU_IDS: list[int] = [int(index) for index in ALLOCATION["gpu_ids"]]
# DP rank r owns the TP pair at positions [r * TP, (r + 1) * TP) of the
# container's device list (vLLM's DP-major local rank order).
DP_RANK_GPU_IDS: list[list[int]] = [list(group) for group in ALLOCATION["dp_rank_gpu_ids"]]
PROJECT = SPEC["environment"].replace(".", "-")
L1_CONTAINER = f"{PROJECT}-deepseek-1"


def _check_allocation() -> None:
    tp = int(ALLOCATION["tensor_parallel_size"])
    dp = int(ALLOCATION["data_parallel_size"])
    flat = [index for group in DP_RANK_GPU_IDS for index in group]
    if (
        len(DP_RANK_GPU_IDS) != dp
        or any(len(group) != tp for group in DP_RANK_GPU_IDS)
        or flat != GPU_IDS
        or int(ALLOCATION["expert_parallel_size"]) != tp * dp
        or len(GPU_IDS) != int(SPEC["hardware"]["gpu_count"])
    ):
        raise SystemExit("example.json allocation is inconsistent (TP x DP must tile gpu_ids)")


_check_allocation()


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command[:4])
    print(f"+ {printable}{' ...' if len(command) > 4 else ''}", flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _nvme_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


def environment_storage() -> Path:
    return _nvme_root() / "model-volumes" / SPEC["environment"]


def _storage_paths() -> dict[str, Path]:
    environment = environment_storage()
    paths = {
        "models": environment / "models",
        "webui": environment / "webui-data",
        "placement_log": environment / "placement-log",
        "cache": environment / "compile-cache",
    }
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    return paths


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = _storage_paths()
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "DEEPSEEK_MODEL_STORAGE_PATH": str(paths["models"]),
            "DEEPSEEK_CACHE_PATH": str(paths["cache"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "DEEPSEEK_VLLM_IMAGE": os.environ.get("DEEPSEEK_VLLM_IMAGE", SPEC["vllm"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "127.0.0.1"),
            # Render-safe default for down/status/logs; `up` replaces it with
            # the union of the DP ranks' NUMA-local CPU sets.
            "DEEPSEEK_CPUSET": os.environ.get("DEEPSEEK_CPUSET", "0"),
        }
    )
    return env


def _compose(arguments: list[str], *, check: bool = True) -> None:
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            *arguments,
        ],
        env=_compose_env(),
        check=check,
    )


def gpu_inventory(text: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 6:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        rows[int(fields[0])] = {
            "name": fields[1],
            "memory_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "compute_capability": float(fields[4]),
            "pci_bus_id": fields[5].lower(),
        }
    return rows


def numa_node(pci_bus_id: str, sysfs: Path = Path("/sys")) -> int:
    canonical = pci_bus_id[4:] if pci_bus_id.startswith("00000000:") else pci_bus_id
    try:
        node = int((sysfs / "bus/pci/devices" / canonical / "numa_node").read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot determine NUMA node of GPU {pci_bus_id}: {error}") from error
    if node < 0:
        raise SystemExit(f"GPU {pci_bus_id} reports no NUMA node")
    return node


def dp_rank_numa_layout(nodes: dict[int, int], dp_rank_gpu_ids: list[list[int]]) -> list[int]:
    """NUMA node per DP rank; each TP pair must sit on one node.

    A TP pair split across sockets pays the SYS (inter-socket) PCIe path on
    every all-reduce; the example refuses that layout instead of silently
    running slower.
    """

    layout = []
    for rank, group in enumerate(dp_rank_gpu_ids):
        rank_nodes = {nodes[index] for index in group}
        if len(rank_nodes) != 1:
            raise SystemExit(
                f"DP rank {rank} GPUs {group} span NUMA nodes {sorted(rank_nodes)}; "
                "each TP pair must share one NUMA node"
            )
        layout.append(rank_nodes.pop())
    return layout


def _node_cpulist(node: int) -> str:
    try:
        cpus = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError as error:
        raise SystemExit(f"cannot read CPUs of NUMA node {node}: {error}") from error
    if not cpus:
        raise SystemExit(f"NUMA node {node} has no CPUs")
    return cpus


def _host_memory_available_gib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // (1024 * 1024)
    raise SystemExit("cannot read MemAvailable from /proc/meminfo")


def _l1_running() -> bool:
    state = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", L1_CONTAINER],
        capture=True,
        check=False,
    )
    return state.returncode == 0 and state.stdout.strip() == "true"


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = gpu_inventory(query)
    expected = SPEC["hardware"]
    missing = [index for index in GPU_IDS if index not in rows]
    if missing:
        raise SystemExit(f"GPUs {missing} are not present; this example uses GPUs {GPU_IDS}")
    ours = _l1_running()
    for index in GPU_IDS:
        row = rows[index]
        if row["name"] != expected["product"]:
            raise SystemExit(f"GPU {index} is {row['name']!r}; expected {expected['product']!r}")
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
        # Never evict another workload: GPUs 0-5 must be idle unless this
        # example's own L1 already holds them.
        if not ours and int(row["memory_used_mib"]) > int(expected["maximum_idle_used_mib"]):
            raise SystemExit(
                f"GPU {index} already has {row['memory_used_mib']} MiB in use; "
                "stop the workload holding it before starting this example"
            )
    if not ours:
        available = _host_memory_available_gib()
        required = int(expected["minimum_host_available_gib"])
        if available < required:
            raise SystemExit(
                f"host has {available} GiB available; the pinned Engram tables need {required} GiB"
            )
    nodes = {index: numa_node(str(rows[index]["pci_bus_id"])) for index in GPU_IDS}
    layout = dp_rank_numa_layout(nodes, DP_RANK_GPU_IDS)
    env["DEEPSEEK_CPUSET"] = ",".join(_node_cpulist(node) for node in dict.fromkeys(layout))
    ranks = "; ".join(
        f"DP{rank}=GPUs {group[0]},{group[1]} (NUMA {layout[rank]})"
        for rank, group in enumerate(DP_RANK_GPU_IDS)
    )
    print(f"hardware: {len(GPU_IDS)} x {expected['product']}; {ranks}", flush=True)


def _image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture=True,
        check=False,
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def build_arguments(source: dict) -> list[str]:
    arguments = [f"VLLM_BASE_IMAGE={source['base_image']}"]
    arguments.append(f"FLASHINFER_REVISION={source['flashinfer_revision']}")
    return [item for argument in arguments for item in ("--build-arg", argument)]


def _ensure_vllm_image(env: dict[str, str]) -> None:
    """Build this example's SM120 overlay when absent, then attest it by ID."""

    image = env["DEEPSEEK_VLLM_IMAGE"]
    source = SPEC["vllm"]
    actual = _image_id(image)
    if actual is None:
        if image != source["image"]:
            raise SystemExit(f"DEEPSEEK_VLLM_IMAGE does not exist locally: {image}")
        print("vLLM image is absent; building this example's SM120 overlay", flush=True)
        _run(
            [
                "docker",
                "build",
                "--pull",
                "--file",
                str(HERE / source["dockerfile"]),
                *build_arguments(source),
                "--tag",
                image,
                "--label",
                f"org.opencontainers.image.revision={source['source_revision']}",
                str(HERE),
            ]
        )
        actual = _image_id(image)
    if os.environ.get("DEEPSEEK_ALLOW_UNPINNED_IMAGE") == "1":
        print(f"WARNING: serving unpinned image {image} ({actual}); not evidence", flush=True)
        return
    if actual != source["image_id"]:
        raise SystemExit(
            f"vLLM image {image} has ID {actual}; example.json and kairyu.yaml pin "
            f"{source['image_id']} (update both to this build before serving)"
        )


_MODEL_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, slug, expected_tree = sys.argv[1:]
target = Path('/models') / slug
attestation = target / '.kairyu-model-attestation.json'

def inventory():
    rows = []
    for path in sorted(target.rglob('*')):
        if not path.is_file() or path == attestation or '.cache' in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(chunk)
        rows.append({'path': str(path.relative_to(target)), 'size': path.stat().st_size,
                     'sha256': digest.hexdigest()})
    tree = hashlib.sha256(json.dumps(rows, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()
    return rows, tree

if attestation.exists():
    current = json.loads(attestation.read_text())
    fields = (current.get('repo'), current.get('revision'), current.get('tree_sha256'))
    if fields != (repo, revision, expected_tree):
        raise SystemExit('model attestation does not match the pinned checkpoint')
    if os.environ.get('VERIFY_MODEL') != '1':
        raise SystemExit(0)
    rows, tree = inventory()
    if tree != expected_tree or current.get('files') != rows:
        raise SystemExit('model files differ from the pinned attestation')
    raise SystemExit(0)

snapshot_download(repo, revision=revision, local_dir=target,
                  token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
if tree != expected_tree:
    raise SystemExit(f'downloaded checkpoint tree mismatch: {tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo,
                                    'revision': revision, 'tree_sha256': tree,
                                    'files': rows}, sort_keys=True))
"""


def _seed_model(target: Path) -> bool:
    """Hard-link an already downloaded checkpoint instead of fetching 476 GiB.

    ``DEEPSEEK_MODEL_SEED`` names a local copy of the same checkpoint on the
    same filesystem. Nothing is trusted from it: the copy is re-hashed below
    against this example's pinned tree before it is served.
    """

    seed = os.environ.get("DEEPSEEK_MODEL_SEED")
    if not seed or target.exists():
        return False
    source = Path(seed).resolve()
    if not (source / "config.json").is_file():
        raise SystemExit(f"DEEPSEEK_MODEL_SEED has no checkpoint: {source}")
    _run(["cp", "-al", str(source), str(target)])
    (target / ".kairyu-model-attestation.json").unlink(missing_ok=True)
    return True


def _ensure_model(env: dict[str, str]) -> None:
    model = SPEC["model"]
    target = Path(env["DEEPSEEK_MODEL_STORAGE_PATH"]) / model["slug"]
    seeded = _seed_model(target)
    if not (target / ".kairyu-model-attestation.json").exists() and not seeded:
        free_gib = shutil.disk_usage(target.parent).free // (1024**3)
        minimum = int(SPEC["storage"]["minimum_free_gib"])
        if free_gib < minimum:
            raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--volume",
        f"{env['DEEPSEEK_MODEL_STORAGE_PATH']}:/models",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [
            env["DEEPSEEK_VLLM_IMAGE"],
            "-c",
            _MODEL_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
            model["tree_sha256"],
        ]
    )
    _run(command)


def _json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def _text_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict, *, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def _healthy_replicas(metrics: str, pool: str) -> int | None:
    pattern = re.compile(
        r'^kairyu_pool_replicas\{pool="' + re.escape(pool) + r'",state="healthy"\} (\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics)
    return int(float(match.group(1))) if match else None


def validate_ready(api_url: str) -> None:
    served = SPEC["model"]["served_name"]
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        healthy = _healthy_replicas(_text_url(f"{api_url}/metrics"), served)
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    if ready.get("status") != "ready" or models != {served}:
        raise SystemExit(
            f"Kairyu public model inventory must be exactly [{served!r}], got {sorted(models)!r}"
        )
    if healthy != 1:
        raise SystemExit(f"Kairyu pool {served!r} must report 1 healthy replica, got {healthy!r}")


def arithmetic_answer_error(body: dict) -> str | None:
    """Reject a probe that is not exactly ``323`` with finite log-probabilities.

    EP6 on an unfixed SM120 image answered its first request correctly and then
    returned NaN log-probabilities or unrelated text (tiered branch candidates
    3-4), so every probe is checked on content, finish, and finiteness.
    """

    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    if not isinstance(content, str) or content.strip() != "323":
        return f"answer is {content!r}, expected '323'"
    tokens = ((choice.get("logprobs") or {}).get("content")) or []
    if not tokens:
        return "no log-probabilities were returned"
    for token in tokens:
        value = token.get("logprob")
        if (
            not isinstance(value, (int, float))
            or value != value
            or value
            in (
                float("inf"),
                float("-inf"),
            )
        ):
            return f"non-finite log-probability {value!r}"
    return None


def arithmetic_probe(**overrides) -> dict:
    payload = {
        "model": SPEC["model"]["served_name"],
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
        "max_tokens": 8192,
        "logprobs": True,
        "top_logprobs": 3,
    }
    payload.update(overrides)
    return payload


def _validate_arithmetic(api_url: str) -> None:
    import concurrent.futures

    # One request per DP rank and mode at once, so vLLM's DP balancer hands a
    # probe to every rank; the finiteness check catches a rank that corrupts.
    variants = [{}, {"reasoning_effort": "low"}]
    requests = [variant for variant in variants for _ in ALLOCATION["dp_rank_gpu_ids"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        bodies = list(
            pool.map(
                lambda overrides: post_json(
                    f"{api_url}/v1/chat/completions",
                    arithmetic_probe(**overrides),
                    timeout_s=900,
                ),
                requests,
            )
        )
    for overrides, body in zip(requests, bodies, strict=True):
        error = arithmetic_answer_error(body)
        if error is not None:
            raise SystemExit(f"arithmetic probe {overrides or 'default'} failed: {error}")


_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def validate_tool_calling(api_url: str) -> None:
    payload = {
        "model": SPEC["model"]["served_name"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an agent operating a computer shell. Every response "
                    "MUST include at least one bash tool call; never answer in "
                    "plain text."
                ),
            },
            {"role": "user", "content": "List the files in the current directory."},
        ],
        "tools": [_BASH_TOOL],
        "max_tokens": 8192,
    }
    try:
        body = post_json(f"{api_url}/v1/chat/completions", payload, timeout_s=600)
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (KeyError, IndexError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"tool-calling probe failed: {error}") from error
    if (
        choice.get("finish_reason") != "tool_calls"
        or not calls
        or calls[0]["function"].get("name") != "bash"
        or not isinstance(arguments, dict)
        or not isinstance(arguments.get("command"), str)
        or not arguments["command"]
    ):
        raise SystemExit(
            "tool-calling probe: the API did not return an executable bash tool "
            f"call (finish_reason={choice.get('finish_reason')!r}, tool_calls="
            f"{json.dumps(calls)[:300]})"
        )


# A 64x64 solid-red PNG. The SM120 sparse-MLA prefill path for image spans is
# exercised only by an image request, so readiness includes one.
PROBE_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsItz/fMY"
    "xgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGB"
    "ywLzk8EPlvGqjQAAAABJRU5ErkJggg=="
)


def image_request(text: str, *, max_tokens: int) -> dict:
    return {
        "model": SPEC["model"]["served_name"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PROBE_IMAGE_PNG_BASE64}"},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def validate_vision(api_url: str) -> None:
    payload = image_request(
        "What single color fills this image? Answer with one word.", max_tokens=8192
    )
    try:
        body = post_json(f"{api_url}/v1/chat/completions", payload, timeout_s=900)
        message = body["choices"][0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"image probe failed: {error}") from error
    if (
        not isinstance(content, str)
        or not re.search(r"\bred\b", content, re.IGNORECASE)
        or body["choices"][0].get("finish_reason") != "stop"
    ):
        raise SystemExit(
            f"image probe: the API returned no completed red-image answer ({message!r})"
        )
    print(f"image probe: {content.strip()[:80]!r}", flush=True)


def validate_serving(api_url: str) -> None:
    """Readiness: pool state, exact finite answers on every DP rank, tools, image."""

    validate_ready(api_url)
    _validate_arithmetic(api_url)
    validate_tool_calling(api_url)
    validate_vision(api_url)


_CHAT_UI_EFFORT_FILTER_ID = "reasoning_effort"
_CHAT_UI_EFFORT_LEVELS: list[str] = list(SPEC["webui"]["reasoning_effort_levels"])


def _webui_api(
    ui_url: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{ui_url}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _provision_chat_ui_effort_selector(ui_url: str) -> None:
    """Install the Reasoning Effort dropdown filter into the pinned Open WebUI."""

    filter_source = (HERE / "webui-reasoning-effort-filter.py").read_text(encoding="utf-8")
    base = f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}"
    try:
        signin = _webui_api(ui_url, "/api/v1/auths/signin", payload={"email": "", "password": ""})
        if not isinstance(signin, dict) or not signin.get("token"):
            raise SystemExit("Chat UI signin did not return an auth-disabled session token")
        if signin.get("role") != "admin":
            raise SystemExit("Chat UI auth-disabled session must be the admin user")
        token = signin["token"]
        listed = _webui_api(ui_url, "/api/v1/functions/", token=token)
        existing = next((row for row in listed if row.get("id") == _CHAT_UI_EFFORT_FILTER_ID), None)
        body = {
            "id": _CHAT_UI_EFFORT_FILTER_ID,
            "name": "Reasoning Effort",
            "content": filter_source,
            "meta": {"description": "Select the DeepSeek reasoning effort from a dropdown."},
        }
        if existing is None:
            state = _webui_api(ui_url, "/api/v1/functions/create", token=token, payload=body)
        else:
            updated = _webui_api(ui_url, f"{base}/update", token=token, payload=body)
            state = {**existing, **(updated or {})}
        # The toggle endpoints flip state, so call them only while a flag is off.
        if not state.get("is_active"):
            state = _webui_api(ui_url, f"{base}/toggle", token=token, payload={})
        if not state.get("is_global"):
            state = _webui_api(ui_url, f"{base}/toggle/global", token=token, payload={})
        if not (state.get("is_active") and state.get("is_global")):
            raise SystemExit("Chat UI effort selector could not be activated globally")
        spec = _webui_api(ui_url, f"{base}/valves/user/spec", token=token)
        enum = spec.get("properties", {}).get("reasoning_effort", {}).get("enum")
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Chat UI effort selector provisioning failed: {error}") from error
    if enum != _CHAT_UI_EFFORT_LEVELS:
        raise SystemExit(
            f"Chat UI effort selector must expose exactly {_CHAT_UI_EFFORT_LEVELS!r}, got {enum!r}"
        )


def up() -> None:
    env = _compose_env()
    _preflight(env)
    _ensure_vllm_image(env)
    _ensure_model(env)
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            "up",
            "--build",
            "--detach",
            "--wait",
            "--wait-timeout",
            "7200",
        ],
        env=env,
    )
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    validate_serving(api_url)
    ui_bind = env["CHAT_UI_BIND_ADDRESS"]
    _provision_chat_ui_effort_selector(
        f"http://{'127.0.0.1' if ui_bind == '0.0.0.0' else ui_bind}:{env['CHAT_UI_PORT']}"
    )
    print("\nEnvironment is ready.")
    print(f"OpenAI API: {api_url}/v1")
    ui_host = os.environ.get("PUBLIC_HOST", ui_bind)
    if ui_host == "0.0.0.0":
        ui_host = "127.0.0.1"
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']} (no authentication)")
    print(
        f"Chat model: {SPEC['model']['served_name']} "
        "(one TP2 x DP3 / EP6 replica on GPUs 0-5; text + image input; thinking high)"
    )
    print(
        "Reasoning effort: Chat Controls -> Valves -> Reasoning Effort "
        f"({'/'.join(_CHAT_UI_EFFORT_LEVELS)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("up", "down", "status", "logs"), default="up")
    args = parser.parse_args()
    if args.action == "up":
        up()
    elif args.action == "down":
        _compose(["down"])
    elif args.action == "status":
        _compose(["ps"])
    else:
        _compose(["logs", "--follow", "--tail", "200"])


if __name__ == "__main__":
    main()

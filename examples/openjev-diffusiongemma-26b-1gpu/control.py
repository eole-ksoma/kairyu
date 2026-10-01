#!/usr/bin/env python3
"""One-command lifecycle for one OpenJev DiffusionGemma replica on one GPU (think = 512)."""

from __future__ import annotations

import argparse
import concurrent.futures
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
PROJECT = SPEC["environment"].replace(".", "-")
L1_CONTAINER = f"{PROJECT}-openjev-1"
SERVED = SPEC["model"]["served_name"]
INFLIGHT, QUEUE = "OPENJEV_GEN_MAX_INFLIGHT", "OPENJEV_GEN_MAX_QUEUE"


def _check_spec() -> None:
    """Refuse an example.json whose parts disagree before anything runs."""

    settings = SPEC["openjev"]["settings"]
    if (
        int(SPEC["hardware"]["gpu_count"]) != 1
        or int(SPEC["allocation"]["gpu_count"]) != 1
        or int(SPEC["allocation"]["replicas"]) != 1
        or SPEC["allocation"]["model"] != SERVED
        or not set(SPEC["openjev"]["tunable"]) <= set(settings)
        or int(settings[INFLIGHT]) + int(settings[QUEUE]) != int(SPEC["pool"]["max_concurrency"])
        # OpenJev answers System One 529 once OPENJEV_MAX_QUEUE requests wait;
        # Kairyu never forwards more than its own max_concurrency.
        or int(SPEC["systemone"]["max_concurrency"]) > int(settings["OPENJEV_MAX_QUEUE"])
    ):
        raise SystemExit(
            "example.json is inconsistent (one GPU, one replica, OpenJev's "
            "in-flight + queue must equal the pool's max_concurrency, and Kairyu "
            "must forward at most OPENJEV_MAX_QUEUE System One requests)"
        )


_check_spec()


def openjev_settings(environ: dict[str, str] | None = None) -> dict[str, str]:
    """OpenJev's settings from example.json; only the tunable ones may be overridden.

    OpenJev answers 529 once in-flight + queued generations exceed their sum,
    and Kairyu counts a 529 as a failure of the only replica. The sum must stay
    at Kairyu's max_concurrency so Kairyu answers 429 first.
    """

    environ = os.environ if environ is None else environ
    settings = {key: str(value) for key, value in SPEC["openjev"]["settings"].items()}
    for key in SPEC["openjev"]["tunable"]:
        settings[key] = environ.get(key, settings[key])
    try:
        inflight, queue = int(settings[INFLIGHT]), int(settings[QUEUE])
    except ValueError as error:
        raise SystemExit(f"{INFLIGHT} and {QUEUE} must be integers: {error}") from error
    capacity = int(SPEC["pool"]["max_concurrency"])
    if inflight < 1 or queue < 0 or inflight + queue != capacity:
        raise SystemExit(
            f"{INFLIGHT}={inflight} + {QUEUE}={queue} must equal Kairyu's "
            f"max_concurrency {capacity} (in-flight at least 1, queue not negative)"
        )
    return settings


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
    env.update(openjev_settings())
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "OPENJEV_MODEL_STORAGE_PATH": str(paths["models"]),
            "OPENJEV_CACHE_PATH": str(paths["cache"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "OPENJEV_IMAGE": os.environ.get("OPENJEV_IMAGE", SPEC["openjev"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "127.0.0.1"),
            "PLAYGROUND_IMAGE": os.environ.get("PLAYGROUND_IMAGE", SPEC["playground"]["image"]),
            "PLAYGROUND_PORT": os.environ.get("PLAYGROUND_PORT", str(SPEC["playground"]["port"])),
            "GPU_ID": os.environ.get("GPU_ID", "0"),
            # Render-safe default for down/status/logs; `up` replaces it with
            # the selected GPU's NUMA-local CPUs.
            "OPENJEV_CPUSET": os.environ.get("OPENJEV_CPUSET", "0"),
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


def _node_cpulist(node: int) -> str:
    try:
        cpus = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError as error:
        raise SystemExit(f"cannot read CPUs of NUMA node {node}: {error}") from error
    if not cpus:
        raise SystemExit(f"NUMA node {node} has no CPUs")
    return cpus


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
    try:
        gpu_id = int(env["GPU_ID"])
    except ValueError as error:
        raise SystemExit(f"GPU_ID must be one GPU index: {env['GPU_ID']!r}") from error
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = gpu_inventory(query)
    if gpu_id not in rows:
        raise SystemExit(f"GPU {gpu_id} is not present (GPUs: {sorted(rows)})")
    row, expected = rows[gpu_id], SPEC["hardware"]
    if row["name"] != expected["product"]:
        raise SystemExit(f"GPU {gpu_id} is {row['name']!r}; expected {expected['product']!r}")
    if row["memory_mib"] < expected["minimum_vram_mib"]:
        raise SystemExit(f"GPU {gpu_id} has insufficient VRAM")
    if row["compute_capability"] < expected["minimum_compute_capability"]:
        raise SystemExit(f"GPU {gpu_id} has insufficient compute capability")
    # Never evict another workload: the GPU must be idle unless this
    # example's own L1 already holds it.
    if not _l1_running() and int(row["memory_used_mib"]) > int(expected["maximum_idle_used_mib"]):
        raise SystemExit(
            f"GPU {gpu_id} already has {row['memory_used_mib']} MiB in use; "
            "stop the workload holding it or choose another GPU_ID"
        )
    node = numa_node(str(row["pci_bus_id"]))
    env["OPENJEV_CPUSET"] = _node_cpulist(node)
    print(f"hardware: GPU {gpu_id} = {expected['product']} (NUMA {node})", flush=True)


def _image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture=True,
        check=False,
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def _ensure_openjev_image(env: dict[str, str]) -> None:
    """Build this example's think-512 overlay when absent, then attest it by ID."""

    image = env["OPENJEV_IMAGE"]
    source = SPEC["openjev"]
    actual = _image_id(image)
    if actual is None:
        if image != source["image"]:
            raise SystemExit(f"OPENJEV_IMAGE does not exist locally: {image}")
        print("OpenJev image is absent; building this example's think-512 overlay", flush=True)
        _run(
            [
                "docker",
                "build",
                "--pull",
                "--file",
                str(HERE / source["dockerfile"]),
                "--build-arg",
                f"OPENJEV_BASE_IMAGE={source['base_image']}",
                "--build-arg",
                f"OPENJEV_REVISION={source['source_revision']}",
                "--build-arg",
                f"OPENJEV_VERSION={source['source_version']}",
                "--build-arg",
                f"VLLM_REVISION={source['vllm_source_revision']}",
                "--tag",
                image,
                "--label",
                f"org.opencontainers.image.revision={source['source_revision']}",
                str(HERE),
            ]
        )
        actual = _image_id(image)
    if os.environ.get("OPENJEV_ALLOW_UNPINNED_IMAGE") == "1":
        print(f"WARNING: serving unpinned image {image} ({actual}); not evidence", flush=True)
        return
    if source["image_id"] is None:
        raise SystemExit(
            f"OpenJev image {image} ({actual}) is not pinned yet: record this ID as "
            "example.json openjev.image_id and kairyu.yaml container_image_digest, or "
            "set OPENJEV_ALLOW_UNPINNED_IMAGE=1 for an unpinned (non-evidence) run"
        )
    if actual != source["image_id"]:
        raise SystemExit(
            f"OpenJev image {image} has ID {actual}; example.json and kairyu.yaml pin "
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


def _ensure_model(env: dict[str, str]) -> None:
    model = SPEC["model"]
    target = Path(env["OPENJEV_MODEL_STORAGE_PATH"]) / model["slug"]
    if not (target / ".kairyu-model-attestation.json").exists():
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
        f"{env['OPENJEV_MODEL_STORAGE_PATH']}:/models",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [
            env["OPENJEV_IMAGE"],
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
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        healthy = _healthy_replicas(_text_url(f"{api_url}/metrics"), SERVED)
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    if ready.get("status") != "ready" or models != {SERVED}:
        raise SystemExit(
            f"Kairyu public model inventory must be exactly [{SERVED!r}], got {sorted(models)!r}"
        )
    if healthy != 1:
        raise SystemExit(f"Kairyu pool {SERVED!r} must report 1 healthy replica, got {healthy!r}")


def _bare(content: str) -> str:
    """An answer without surrounding whitespace, Markdown emphasis or a final period."""

    return content.strip().strip("*`").strip().rstrip(".")


def think_answer_error(body: dict, *, expected: str | None = None) -> str | None:
    """Why a chat response is not a completed think-first answer, or None.

    Every answer carries its thought (Kairyu returns it as ``reasoning_content``,
    OpenJev as ``reasoning``), finishes with ``stop`` and has visible content.
    """

    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return "no reasoning: every answer must think first"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return "empty answer content"
    if expected is not None and _bare(content) != expected:
        return f"answer is {content.strip()[:80]!r}, expected {expected!r}"
    return None


def arithmetic_probe(**overrides) -> dict:
    payload = {
        "model": SERVED,
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
        "max_tokens": 256,
    }
    payload.update(overrides)
    return payload


def _validate_arithmetic(api_url: str) -> None:
    # A caller's effort does not change the thought: both variants must think.
    variants = [{}, {"reasoning_effort": "low"}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants)) as pool:
        bodies = list(
            pool.map(
                lambda overrides: post_json(
                    f"{api_url}/v1/chat/completions",
                    arithmetic_probe(**overrides),
                    timeout_s=600,
                ),
                variants,
            )
        )
    for overrides, body in zip(variants, bodies, strict=True):
        error = think_answer_error(body, expected="323")
        if error is not None:
            raise SystemExit(f"arithmetic probe {overrides or 'default'} failed: {error}")


BASH_TOOL = {
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


def tool_call_error(body: dict) -> str | None:
    """Why a response is not an executable bash tool call, or None."""

    try:
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return f"malformed tool-call response: {error}"
    if (
        choice.get("finish_reason") != "tool_calls"
        or not calls
        or calls[0]["function"].get("name") != "bash"
        or not isinstance(arguments, dict)
        or not isinstance(arguments.get("command"), str)
        or not arguments["command"]
    ):
        return (
            "the API did not return an executable bash tool call "
            f"(finish_reason={choice.get('finish_reason')!r}, "
            f"tool_calls={json.dumps(calls)[:300]})"
        )
    return None


def tool_request(**overrides) -> dict:
    payload = {
        "model": SERVED,
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
        "tools": [BASH_TOOL],
        "max_tokens": 1024,
    }
    payload.update(overrides)
    return payload


def validate_tool_calling(api_url: str) -> None:
    try:
        body = post_json(f"{api_url}/v1/chat/completions", tool_request(), timeout_s=600)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"tool-calling probe failed: {error}") from error
    error = tool_call_error(body)
    if error is not None:
        raise SystemExit(f"tool-calling probe: {error}")


# A 64x64 solid-red PNG, so readiness exercises the image path once.
PROBE_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsItz/fMY"
    "xgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGB"
    "ywLzk8EPlvGqjQAAAABJRU5ErkJggg=="
)


def image_request(text: str, *, max_tokens: int) -> dict:
    return {
        "model": SERVED,
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


def vision_answer_error(body: dict) -> str | None:
    error = think_answer_error(body)
    if error is not None:
        return error
    content = body["choices"][0]["message"]["content"]
    if not re.search(r"\bred\b", content, re.IGNORECASE):
        return f"the answer does not name the image's color: {content.strip()[:80]!r}"
    return None


def validate_vision(api_url: str) -> None:
    payload = image_request(
        "What single color fills this image? Answer with one word.",
        max_tokens=int(SPEC["verification"]["vision"]["max_tokens"]),
    )
    try:
        body = post_json(f"{api_url}/v1/chat/completions", payload, timeout_s=600)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"image probe failed: {error}") from error
    error = vision_answer_error(body)
    if error is not None:
        raise SystemExit(f"image probe: {error}")
    print(f"image probe: {body['choices'][0]['message']['content'].strip()[:80]!r}", flush=True)


SYSTEMONE_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {
            "outage": "service down",
            "billing": "charges, refunds",
            "feature": "requests, how-to",
        },
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}
SYSTEMONE_STATE = "Everything is down and we have a demo with our biggest client at noon."


def systemone_request(state: str = SYSTEMONE_STATE, model: str | None = None) -> dict:
    return {
        "model": model or SPEC["systemone"]["model"],
        "state": state,
        "questions": SYSTEMONE_QUESTIONS,
    }


def systemone_answer_error(body: dict) -> str | None:
    """Why a System One answer to SYSTEMONE_QUESTIONS is not a well-formed, sensible one."""

    try:
        answers, usage = body["answers"], body["usage"]
        team = answers["team"]
        total = sum(team["probabilities"].values())
        if not 0 <= answers["urgent"]["noul"] <= 1 or abs(total - 1) > 1e-3:
            return f"probabilities out of range: {answers}"
        if not isinstance(usage.get("input_tokens"), int) or usage["input_tokens"] < 1:
            return f"usage {usage!r}"
        return None
    except (KeyError, TypeError, AttributeError) as error:
        return f"malformed answer ({error!r}): {str(body)[:200]}"


def validate_systemone(api_url: str) -> None:
    for name in (SPEC["systemone"]["model"], *SPEC["systemone"]["aliases"]):
        body = post_json(f"{api_url}/v1/systemone", systemone_request(model=name), timeout_s=120)
        error = systemone_answer_error(body)
        if error:
            raise SystemExit(f"System One probe ({name}) failed: {error}")
    print(f"System One probe: team={body['answers']['team']['choice']!r}", flush=True)


def validate_serving(api_url: str) -> None:
    """Readiness: pool state, think-first answers on the text and image paths, tools,
    and System One answers under every model name."""

    validate_ready(api_url)
    _validate_arithmetic(api_url)
    validate_tool_calling(api_url)
    validate_vision(api_url)
    validate_systemone(api_url)


def up() -> None:
    env = _compose_env()
    _preflight(env)
    _ensure_openjev_image(env)
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
            "3600",
        ],
        env=env,
    )
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    validate_serving(api_url)
    print("\nEnvironment is ready.")
    print(f"OpenAI API: {api_url}/v1")
    ui_host = os.environ.get("PUBLIC_HOST", env["CHAT_UI_BIND_ADDRESS"])
    if ui_host == "0.0.0.0":
        ui_host = "127.0.0.1"
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']} (no authentication)")
    print(f"Playground: http://{ui_host}:{env['PLAYGROUND_PORT']} (System One, no authentication)")
    print(
        f"Chat model: {SERVED} (one OpenJev replica on GPU {env['GPU_ID']}; text + image "
        "input; every answer thinks first, at most 512 thought tokens)"
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

# Kairyu

**English** | [日本語](README.ja.md)

**vLLM-compatible LLM inference framework with native orchestration.**

Kairyu (海流, "ocean current") combines a vLLM drop-in inference API with a first-class
orchestration layer — a learned-router-ready **Router**, a Planner/Worker/Verifier/Synthesizer
**Conductor** (role DAG), and **Mixture-of-Agents** — all behind one Python API and one
OpenAI-compatible endpoint. Underneath, a custom engine core (Radix-Paged KV cache,
chunked-prefill scheduler, speculative decoding, xgrammar structured output, TP/EP/PP,
FP8/INT8/AWQ/GPTQ/NVFP4 quantization) serves real checkpoints through the same pluggable
backend seam.

- **Python**: 3.11+ &nbsp;|&nbsp; **License**: MIT &nbsp;|&nbsp; **Tests**: 5,000+ (coverage gate 80%)

---

## Table of contents

1. [Why Kairyu](#1-why-kairyu)
2. [Architecture](#2-architecture)
   - [2.1 Layered architecture](#21-layered-architecture)
   - [2.2 Request data flow](#22-request-data-flow)
   - [2.3 How orchestration works](#23-how-orchestration-works)
   - [2.4 Engine core internals (L1)](#24-engine-core-internals-l1)
   - [2.5 Fleet / gateway layer](#25-fleet--gateway-layer)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [Single-model setup & usage](#5-single-model-setup--usage)
6. [Orchestration setup & usage](#6-orchestration-setup--usage)
7. [Configuration reference](#7-configuration-reference)
8. [Evaluation and verification](#8-evaluation-and-verification)
9. [Development](#9-development)
10. [Documentation index](#10-documentation-index)
11. [License](#11-license)

---

## 1. Why Kairyu

Most serving stacks treat orchestration (routing, multi-agent pipelines, budgets) as an
application-side afterthought bolted onto a raw completion endpoint. Kairyu makes it native:

- **One import away from vLLM** — `from kairyu import LLM, SamplingParams` runs existing
  vLLM offline examples unchanged, verified by contract tests (`tests/compat/`).
- **Orchestration below the API line** — the Router sees engine-level signals, and the
  Conductor's steps hit warm KV prefixes (`cache_hint` plumbing), which pure API-level
  frameworks cannot do.
- **Pluggable backends** — every layer talks to a small async `EngineBackend` protocol, so
  mock (CI), vLLM (local GPU), OpenAI-compatible (external APIs), and the custom `kairyu`
  engine core are interchangeable per worker.
- **Routers that learn** — serving logs feed a distillation + contextual-bandit pipeline
  that upgrades the rule router into a `LearnedRouter` without an API change.

The whole stack is implemented and CPU-verified end to end; GPU performance gates and kernel
tuning are the remaining hardware-bound work, tracked in
[`docs/gpu-runbook.md`](docs/gpu-runbook.md) and [`PROGRESS.md`](PROGRESS.md).

## 2. Architecture

### 2.1 Layered architecture

Kairyu is layered as **L3 Interface / L2 Orchestration / L1 Engines**. Everything above L1
depends only on the `EngineBackend` protocol (`kairyu/engine/backend.py`), so the custom
engine is "one more backend", not a rewrite:

```
L3  Interface       kairyu.entrypoints   LLM / AsyncLLMEngine (vLLM drop-in),
                                         OpenAI-compatible FastAPI server (SSE, tools,
                                         batch, embeddings, responses), kairyu CLI
L2  Orchestration   kairyu.orchestration Router → Conductor (role DAG) / MoA,
                                         Budget, JSONL decision logs, learning pipeline,
                                         ReplicaPool (session/prefix/KV-aware routing)
                    kairyu.deploy        DeploymentSpec, registry/reconciler, prober
L1  Engines         kairyu.engine        EngineBackend protocol + registry:
                                         mock | kairyu | kairyu-proc | openai | vllm
                    kairyu.engine.core   Radix-Paged KV, chunked-prefill scheduler,
                                         paged model runner + sampler, spec decode,
                                         attention backends, CUDA-graph seam,
                                         TP/EP/PP, P-D separation, KV transport
                    kairyu.models        Llama-3.x / Qwen2 / Qwen3 / Qwen3-MoE /
                                         DeepSeek-V3 (+ EAGLE-3 / MTP draft heads)
                    kairyu.quant         FP8 / INT8 / AWQ / GPTQ / NVFP4
```

A design theme runs through every layer: **each seam is a small protocol with a
deterministic CPU implementation**. The Router, Conductor, and ReplicaPool depend only on
`EngineBackend`; inside the engine, `ModelRunner`, `AttentionBackend`, `Communicator`,
`KVHandoff`, `DraftSource`, and the CUDA-graph `StepExecutor` are all protocols with CPU
fakes pinned by tests — GPU and multi-process implementations swap in behind them unchanged.

### 2.2 Request data flow

What happens when a request hits `POST /v1/chat/completions`
(`kairyu/entrypoints/server/app.py`):

```mermaid
sequenceDiagram
    participant C as Client
    participant S as FastAPI server (L3)
    participant O as Orchestrator (L2)
    participant P as ReplicaPool (L2)
    participant E as EngineBackend (L1)

    C->>S: POST /v1/chat/completions {model, messages}
    S->>S: auth, tenant limits, render chat template
    alt model is an auto model (kairyu-auto, ...)
        S->>O: run(prompt) / run_chat(stream)
        O->>O: Router: features -> tier1 | tier2 | multi_agent
        O->>E: direct call, Conductor role DAG, or MoA
        E-->>O: text + usage (summed across steps)
        O-->>S: result + optional kairyu_trace
    else model is a served engine or pool
        S->>P: generate(GenerationRequest + CacheHint)
        P->>P: place: session affinity -> prefix score -> least-outstanding
        P->>E: replica.generate(...)
        E-->>S: StreamUpdates / GenerationResult
    end
    S-->>C: JSON or SSE stream (+ usage chunk)
```

Inside the `kairyu` backend (`kairyu/engine/kairyu_backend.py`), each request flows through
a synchronous step loop that owns all engine state on one thread
(`kairyu/engine/engine_loop.py`):

```
submit -> tokenize -> Scheduler.schedule()        # chunked-prefill plan, radix-KV admission
       -> ModelRunner.execute()                   # paged forward + Sampler (fixed op order)
       -> Scheduler.update()                      # commit sampled tokens to the radix tree
       -> IncrementalDetokenizer -> StreamUpdate  # SSE-safe stop-string holdback
```

The `kairyu-proc` backend (`kairyu/engine/zmq_backend.py`) drives the *same* `EngineLoop`
in a child process over ZMQ/msgpack for crash isolation — the API process survives an
engine crash and respawns it.

### 2.3 How orchestration works

The L2 pipeline behind the reserved model name `kairyu-auto`
(`kairyu/orchestration/orchestrator.py`):

```mermaid
flowchart LR
    Q[Query] --> R{Router}
    R -->|simple| T1[tier1 engine]
    R -->|hard| T2[tier2 engine]
    R -->|multi-step| MA{multi_agent}
    MA -->|default| CO[Conductor role DAG]
    MA -->|"moa_samples > 0"| MOA[Mixture-of-Agents]
    CO --> RES[Result + trace + usage]
    MOA --> RES
    T1 --> RES
    T2 --> RES
```

**Router** (`kairyu/orchestration/router.py`, `features.py`). Routing is model-free and
runs in well under 10 ms: `extract_features` computes `char_len`, `word_count`,
`has_code_fence`, `math_symbol_count`, `reasoning_keyword_count`,
`multi_step_marker_count`, and `question_count`. The default `RuleRouter` applies
thresholds (tunable via `RouteThresholds`): ≥3 multi-step markers or ≥2000 chars →
`multi_agent`; a code fence, ≥2 reasoning keywords, ≥3 math symbols, or ≥600 chars →
`tier2`; everything else → `tier1`. The same feature vector doubles as the training schema
for the learned router.

**Conductor — role DAG** (`kairyu/orchestration/conductor.py`). The default DAG is
**planner** (tier2) → **worker** (tier1) → **verifier** (tier2) → **synthesizer** (tier2).
Roles whose dependencies are satisfied run concurrently in asyncio "waves". A verifier runs
inline after its target: if the verdict is not `PASS` and the budget allows, the Conductor
builds a refine prompt from the previous attempt plus the verifier's feedback and
regenerates (up to `max_refine_depth`). All prompts render as `shared_prefix + role_suffix`
with a `CacheHint`, so successive steps land on the replica holding the warm KV prefix. A
failing unit is recorded in the trace and the run returns best-so-far — one backend error
never discards completed work.

**Mixture-of-Agents** (`kairyu/orchestration/moa.py`). `n` proposers sample in parallel
(temperature 0.9, distinct seeds), then one synthesis pass (temperature 0.3) merges the
numbered candidates. Proposers and the synthesizer can be different backends (e.g. cheap
tier1 proposers, frontier tier2 synthesizer). This is the mechanism behind the
`kairyu-auto-max` tier.

**Budget** (`kairyu/orchestration/budget.py`). `Budget(max_steps, max_refine_depth,
max_cost_usd)` is charged by every unit through a pluggable `CostModel`. Exhaustion is
queryable, not raised: the Conductor stops refining and returns the best result so far.

**Router learning** (`kairyu/orchestration/learning/`). `JsonlRouterLog` records routing
decisions and outcomes as JSONL — queries are stored as SHA-256 hashes, never raw text.
From there: (1) `build_dataset` joins decisions with outcomes and labels each query with
the highest mean-utility target (`utility = quality − cost_weight · cost_usd`);
(2) a distilled logistic-regression classifier warm-starts `LearnedRouter` on the same
`Router` protocol, falling back to the rule router below a confidence threshold;
(3) `BanditRouter` (epsilon-greedy contextual bandit) refines the policy online, deferring
to its base router until every arm has enough observations. See
[`docs/design/m4-router-learning.md`](docs/design/m4-router-learning.md).

### 2.4 Engine core internals (L1)

The custom engine behind backend name `kairyu` (`kairyu/engine/core/`):

| Component | Files | What it does |
|---|---|---|
| Radix-Paged KV cache | `radix_kv.py`, `pages.py`, `kv_pool.py` | Radix-tree prefix sharing over paged KV blocks (refcounted, LRU eviction, session pins); `PagePool` free list; `PagedKVPool` holds K/V tensors layer-major so KV transport slices contiguously. Emits vLLM-compatible KV events for fleet routing. |
| Scheduler | `scheduler.py` | Pure policy, no GPU: chunked-prefill token budgets, page-granularity admission through the radix cache, multi-token (speculative) commit, preemption, oversized-prompt rejection. |
| Step loop | `engine_core.py`, `overlap.py`, `pipeline.py` | `ModelRunner` protocol + `StepOutput` contract; `OverlapEngineCore` plans step N+1 while the device runs step N; `PipelinedEngineCore` adds inter-step pipeline parallelism. |
| Model runner + sampler | `model_runner.py`, `sampler.py` | Paged forward over real checkpoints; sampler with a fixed op order (logprobs → xgrammar grammar mask → penalties → temperature → min-p/top-k/top-p → seeded sample) and deterministic splitmix64 seeding so TP ranks sample identically. |
| Speculative decoding | `spec_runner.py`, `draft.py` | `SpeculativeRunner` wraps any `ModelRunner`: n-gram prompt-lookup drafts by default, `ModelDraftSource` for EAGLE-3 / MTP heads (`kairyu/models/eagle.py`, `mtp.py`); greedy verification with a tested output-identical invariant. |
| Attention backends | `attention/` | `AttentionBackend` protocol: `torch` (device-agnostic paged attention), FlashInfer, FlashAttention-3/4 prefill adapters, and MLA reference math for DeepSeek; selected from the hardware profile or `KAIRYU_ATTENTION_BACKEND`. |
| CUDA-graph seam | `step_executor.py`, `graph_buckets.py` | Capture-once-per-bucket replay with static device buffers, pinned on CPU against a fake graph backend; only `cuda_graph_gpu.py` touches CUDA. |
| Distributed | `worker.py`, `dist_comm.py`, `pp_worker.py` | TP (rank 0 drives the scheduler, snapshot broadcast, per-rank sharded safetensors loading), EP (MoE all-to-all), PP (stage slices) — parity-gated with gloo in the default test suite; NCCL is a constructor argument. |
| P-D separation + KV transport | `pd.py`, `pd_remote.py`, `kv_serde.py`, `kv_transport*.py` | Prefill/decode disaggregation in-process or across two real processes with byte-parity KV transfer over TCP; NIXL/RDMA adapter ready. |
| Structured output | `structured.py` | xgrammar-compiled JSON-schema grammars applied as per-step token bitmasks. |

**Models** (`kairyu/models/`): Llama-3.x, Qwen2, Qwen3 (dense), Qwen3-MoE, DeepSeek-V3
(MLA + sigmoid-routed MoE, yarn rope) — all pinned to `transformers.generate` greedy parity
through the full engine. **Quantization** (`kairyu/quant/`): FP8, INT8 W8A8, AWQ, GPTQ,
NVFP4 checkpoints auto-detected at load; all five load and run through the full engine on
CPU, with Triton kernel seams for GPU.

### 2.5 Fleet / gateway layer

A gateway node serves a `ReplicaPool` (`kairyu/orchestration/replica.py`) — itself an
`EngineBackend`, so it slots in anywhere an engine is expected. Placement order per request:

1. **Session affinity** — `session_id` (from `X-Session-ID` or the OpenAI `user` field)
   maps to a replica by rendezvous (HRW) hashing over eligible (healthy ∧ not draining)
   replicas, so multi-turn sessions keep hitting their warm radix-KV prefix.
2. **Load valve** — if the affine replica's outstanding depth exceeds
   `queue_depth_threshold`, fall back to least-outstanding.
3. **Prefix/KV-aware scoring** (opt-in) — score replicas by
   `α · prefix_overlap − β · outstanding`, using two indexes: `PrefixIndex`
   (approximate, gateway-side chained-hash text chunks) and `KvEventIndex` (precise
   per-replica KV block hashes fed by engine `BlockStored`/`BlockRemoved` events over ZMQ;
   a stale feed gracefully falls back to the approximate trie).
4. **Least outstanding** for session-less traffic.

Health: `unhealthy_after` consecutive failures ejects a replica (client 4xx errors are
*not* counted); a background `HealthProber` (`kairyu/deploy/prober.py`) probes ejected
replicas' `/readyz` and restores them. Membership is dynamic
(`add_replica`/`drain`/`remove_replica`) and can be driven by a TTL-heartbeat
`ReplicaRegistry` + `PoolReconciler` (`kairyu/deploy/registry.py`).

For horizontal gateway scale-out, an L7 load balancer consistently hashes the
explicit `X-Session-ID` across gateway identities; each gateway then applies
the same ReplicaPool HRW decision. Batch files and jobs can use the PostgreSQL
shared store with atomic claim leases and fencing. The filesystem store remains
the backward-compatible single-gateway option.

## 3. Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Kairyu is not on PyPI yet —
install from source:

```bash
git clone https://github.com/ytworks/kairyu.git && cd kairyu
uv sync                       # core only (lightweight)
uv sync --extra engine        # + torch/xgrammar/tokenizers/safetensors (real models)
uv sync --group dev           # + test/lint toolchain
```

Core dependencies are lightweight (pydantic, fastapi, httpx, pyyaml, uvicorn, jinja2).
Everything heavier is opt-in:

| extra | contents | enables |
|---|---|---|
| `--extra engine` | torch, xgrammar, tokenizers, safetensors | real checkpoints through the `kairyu` backend, `json_schema` structured output |
| `--extra hf` | tokenizers, safetensors | HF tokenizer/weights only (no torch) |
| `--extra fleet` | pyzmq, msgpack, psycopg | process-split engine, KV event transport, shared PostgreSQL BatchStore |
| `--extra otel` | opentelemetry-sdk | tracing spans (no-op without it) |
| `--extra gpu` | flashinfer, triton, nixl | GPU kernels/fabric (Linux-only markers; macOS `uv sync` skips them) |
| `--extra flashattention4` | `flash-attn-4[cu13]==4.0.0b24` | opt-in FlashAttention-4 prefill kernels using the upstream-recommended CUDA 13 extra (Linux only; combine with `--extra gpu` for delegated FlashInfer decode; both are included by `Dockerfile.cuda`) |
| `--extra evals` | datasets/HF Hub, evaluation formats, progress UI, pinned IFEval scorer dependencies | checkout-only Core, Quantization, Structured Output, and Long Context evaluation |
| `--group dev` | pytest, ruff, transformers, openai, … | test suite + parity goldens |

FlashAttention-3 is built from the official upstream source tree. The
supported build is pinned to the same immutable upstream snapshot as the FA4
package, tag `fa4-v4.0.0.beta24`:

```bash
uv sync --extra gpu
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 849f660f73b176e5ad5670e7f822c7fa9f3eaf8b
git submodule update --init --recursive
cd hopper
python setup.py install
```

Install the packaged FA4 path together with its FlashInfer decode dependency:

```bash
uv sync --extra gpu --extra flashattention4
```

Kairyu publicly supports this upstream FA3 adapter on SM90, with CUDA 12.3 or
newer. Environments without a representative SM90 GPU validate the deferred
import, API, shape, and architecture contract with an injected fake module.
Explicit FA3 selection fails closed when the package, API, shape, or hardware
does not match; fake-contract coverage is not a performance result and does
not justify selecting FA3 by default.

vLLM is only needed for the `vllm` backend on a Linux GPU host (install it in the same
environment).

## 4. Quick start

```bash
uv run pytest                                        # full suite, coverage gate 80%
uv run --frozen python -m verification list                  # inspect the verification registry
./examples/deepseek-v4-flash-0731-8gpu/run.sh         # Kairyu L3 + vLLM L1 + Chat UI
```

Then pick your path: [single model](#5-single-model-setup--usage) or
[orchestration](#6-orchestration-setup--usage).

## 5. Single-model setup & usage

### 5.1 Python API (vLLM drop-in)

`kairyu` replicates the vLLM offline surface — change one import:

```python
from kairyu import LLM, SamplingParams   # was: from vllm import ...

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8))
print(outputs[0].outputs[0].text)
```

`LLM.chat(...)` loads the chat template and named special tokens from a local
`tokenizer` snapshot (or the local `model` path) on first use. If neither local
metadata nor an explicit `chat_template=` is available, chat fails before
dispatch; the old role-prefix renderer is available only through the explicit
`legacy_chat=True` compatibility option. Plain `generate(...)` remains a
completion path and does not require chat metadata.

`SamplingParams`, `RequestOutput`, `CompletionOutput`, `AsyncEngineArgs`, and
`AsyncLLMEngine` replicate vLLM's public surface (the subset exercised by vLLM's own
examples), verified by the contract tests in `tests/compat/`. The async engine:

```python
from kairyu import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model="Qwen/Qwen2.5-7B-Instruct"))
async for out in engine.generate("Hello", SamplingParams(max_tokens=32), request_id="r1"):
    ...
```

### 5.2 Choosing a backend

Every model runs behind one of five `EngineBackend` implementations
(`kairyu/engine/registry.py`), chosen per worker/engine:

| backend | runs | when to use |
|---|---|---|
| `kairyu` | Kairyu's own engine core, in-process | local safetensors checkpoints; the native path (radix KV, spec decode, structured output) |
| `kairyu-proc` | same engine in a child process (ZMQ/msgpack) | crash isolation between the API server and the engine |
| `vllm` | `vllm.AsyncLLMEngine` on a local GPU | you already run vLLM and want Kairyu's orchestration on top |
| `openai` | any OpenAI-compatible HTTP endpoint | hosted APIs (Together, Fireworks, Groq, Moonshot, …) or your own `vllm serve` / SGLang / Ollama box |
| `mock` | deterministic canned responses | CI and tests — the entire default test suite runs on it |

### 5.3 Running a local checkpoint (`kairyu` backend)

The native engine loads HF-format safetensors directories directly:

```python
from kairyu import LLM, SamplingParams
from kairyu.engine.kairyu_backend import KairyuBackend

backend = KairyuBackend(model_path="/models/qwen2.5-0.5b-instruct")
llm = LLM(model="qwen", backend=backend)
print(llm.generate(["What is paged attention?"], SamplingParams(max_tokens=64)))
```

Supported architectures: **Llama-3.x, Qwen2, Qwen3, Qwen3-MoE, DeepSeek-V3**. Quantized
checkpoints (**FP8 / INT8 / AWQ / GPTQ / NVFP4**) are auto-detected from the checkpoint
config. Key constructor options (also available as `options:` in a DeploymentSpec — see
the [backend options table](#backend-options-enginesoptions)): `tokenizer`, `num_pages`,
`page_size`, `max_num_batched_tokens`, `speculative="ngram"`, `tensor_parallel_size`,
`pipeline_depth` (1 is synchronous compatibility; 2 enables schedule/device
overlap), `decode_mode="cuda_graph"` on CUDA, and
`pd_separation`/`pd_prefill_device`/`pd_decode_device`/`pd_defer_handoff` for
prefill/decode separation.

**Hosted API instead of local weights** — the `openai` backend points at any
OpenAI-compatible endpoint; the API key is read from the environment variable named by
`api_key_env`, never hardcoded:

```bash
export MOONSHOT_API_KEY=sk-...
```

```python
from kairyu import LLM, SamplingParams
from kairyu.engine.openai_backend import OpenAICompatBackend

backend = OpenAICompatBackend(
    base_url="https://api.moonshot.ai/v1",
    model="kimi-k2-0905-preview",           # model names are illustrative
    api_key_env="MOONSHOT_API_KEY",
)
llm = LLM(model="kimi-k2", backend=backend)
```

The same backend reaches a server you run yourself (`vllm serve`, SGLang, Ollama) — set
`base_url="http://gpu-box:8000/v1"` and point `api_key_env` at any set variable
(the variable must exist even if the server ignores auth).

### 5.4 Serving a single model over HTTP

`kairyu serve <config.yaml>` starts the hardened OpenAI-compatible server from a
**DeploymentSpec** YAML. Minimal single-model config:

```yaml
# single_model.yaml
server:
  host: 0.0.0.0
  port: 8000

engines:
  qwen:                          # served model name
    backend: kairyu              # or: kairyu-proc | vllm | openai | mock
    options:
      model_path: /models/qwen2.5-0.5b-instruct
      generation_config: auto    # auto | vllm | none; default auto
      pipeline_depth: 2        # 1 preserves synchronous behavior
      decode_mode: cuda_graph   # explicit opt-in; eager remains the default
      cuda_graph_max_batch: 8
      cuda_graph_max_pages: 64  # longer page tables safely fall back to eager
```

```bash
uv run kairyu serve single_model.yaml          # --host/--port override the YAML
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

Tensor parallelism is configured per engine in the YAML
(`options: {tensor_parallel_size: 2}`) — the serve process spawns a multi-process TP
worker group (gloo on CPU, NCCL on GPU). There is no CLI flag for it.

Native `kairyu` and `kairyu-proc` engines use `generation_config: auto` to
apply `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty` from
the model's `generation_config.json` only when the request omits that field.
An explicit request value, including a neutral value such as
`temperature: 1.0`, always wins. `vllm` keeps the model's stop-token metadata
but uses neutral sampling defaults, while `none` ignores the generation file
entirely. `kairyu serve CONFIG --generation-config MODE` overrides the YAML
mode for every locally constructed native engine, pool replica, and linked
orchestrator worker. `/backends` reports the resolved mode, source, and five
effective sampling defaults, including per-worker records for orchestrated
models.

Prefill/decode separation is also configured per engine. This example places
the roles on a peer-accessible CUDA pair:

```yaml
engines:
  qwen-pd:
    backend: kairyu
    options:
      model_path: /models/qwen3-32b
      pd_separation: true
      pd_prefill_device: cuda:0
      pd_decode_device: cuda:1
      pd_defer_handoff: true
```

Both role devices must be CPU or both CUDA. Each distinct role device is probed
and gets its own compatible attention backend; stateful backend instances are
not shared across devices. Deferred cross-device handoff requires CUDA P2P
access between the selected GPUs and fails at startup when it is unavailable.
It copies on dedicated source and destination transfer streams, but publishes
destination KV and releases source pages only after the physical completion
event returns ready from `event.query()`. If completion or cleanup can no
longer be established, the engine retains the affected ownership and fails
closed instead of publishing or reusing partial KV. Set
`pd_defer_handoff: false` for the serialized control: both transfer streams are
synchronized before publication, with the same correctness contract. The
current P-D path requires TP=1, eager decode, and no speculative decoding.

Endpoints served: `/v1/chat/completions` (SSE streaming, tools, logprobs, `n>1`,
`response_format: json_schema`, vision content parts), `/v1/completions`, `/v1/models`,
`/v1/route` (non-dispatching route preview), `/routing` (routing descriptor),
`/v1/embeddings`, `/v1/responses` (typed SSE, function tools, continuation state),
`/v1/files` + `/v1/batches`, `/health`,
`/readyz`, `/metrics` (Prometheus), `/admin/*`. Full list in the
[configuration reference](#http-surface).

## 6. Orchestration setup & usage

### 6.1 Programmatic

An `Orchestrator` wraps a dict of engines keyed by tier name. The Router picks a target
per query; `multi_agent` routes dispatch to the Conductor or MoA:

```python
from kairyu import Orchestrator
from kairyu.engine.mock import MockBackend

orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
result = orchestrator.run_sync("First, plan X. Then do Y. Finally, verify.")
print(result.route.target, result.text)     # -> multi_agent, <synthesized answer>
```

Mix real backends freely — a typical pool puts a small local model on `tier1` and a
frontier API on `tier2`:

```python
from kairyu import Orchestrator
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.vllm_backend import VLLMBackend

orchestrator = Orchestrator(engines={
    "tier1": VLLMBackend(model="Qwen/Qwen2.5-7B-Instruct"),
    "tier2": OpenAICompatBackend(
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k2-0905-preview",
        api_key_env="MOONSHOT_API_KEY",
    ),
})
```

Routing thresholds are tunable via `RouteThresholds`
(`kairyu/orchestration/router.py`); pass `moa_samples=4` to route `multi_agent` through
Mixture-of-Agents instead of the Conductor.

### 6.2 Declarative agent pools (YAML / decorators)

Workers, a role DAG, and a budget in one file:

```yaml
# pool.yaml
workers:
  - name: tier1                      # easy queries: local open model
    backend: vllm
    model: Qwen/Qwen2.5-7B-Instruct
    options:                         # extra kwargs forwarded to the backend constructor
      gpu_memory_utilization: 0.85
  - name: tier2                      # hard queries + planner/verifier roles: frontier API
    backend: openai
    model: kimi-k2-0905-preview
    base_url: https://api.moonshot.ai/v1
    api_key_env: MOONSHOT_API_KEY

roles:
  - name: planner
    worker: tier2
    role_type: planner
    prompt: "[planner] Break the task into a short plan.\nTask: {query}"
  - name: worker
    worker: tier1
    prompt: "[worker] Execute the plan.\nPlan: {planner}\nTask: {query}"
    depends_on: [planner]
  - name: synthesizer
    worker: tier2
    role_type: synthesizer
    prompt: "[synthesizer] Final answer.\nDraft: {worker}\nTask: {query}"
    depends_on: [worker]

budget:
  max_steps: 12
  max_refine_depth: 2
  max_cost_usd: 0.50                 # hard cap for one orchestrated request
  cost_per_1k_chars_usd: 0.002
```

```python
from kairyu.dsl.loader import build_orchestrator, load_spec

orchestrator = build_orchestrator(load_spec("pool.yaml"))
result = orchestrator.run_sync("Compare radix-tree and hash-based KV prefix sharing.")
print(result.route.target, result.text)
```

Role prompts are templates: `{query}` is the user query, `{<role_name>}` interpolates an
upstream role's output, `depends_on` defines the DAG edges, and `verifies` marks a role as
the verifier of another (triggering the refine loop on a non-`PASS` verdict). The same
spec is available as a decorator front-end via `kairyu.dsl.decorators.AgentPool`.

### 6.3 Serving orchestration (`kairyu-auto`)

A DeploymentSpec can serve any number of **named orchestrations alongside plain models** —
clients just pick a model name:

```yaml
# deploy.yaml
engines:
  m1: {backend: mock}                # local kairyu/vllm in production; see policy below
  m2: {backend: mock}

orchestrators:
  kairyu-auto:                       # standard tier: Router -> direct / Conductor
    spec: agent_pool.yaml
  kairyu-auto-max:                   # deep tier: multi_agent routes through MoA
    spec: agent_pool_max.yaml

legacy_chat_models: [kairyu-auto, kairyu-auto-max]  # explicit mock-demo framing
```

```bash
uv run kairyu serve deploy.yaml
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "kairyu-auto", "messages": [{"role": "user", "content": "hi"}]}'
```

- Usage in the response is the **real sum across all orchestration steps** — no made-up
  token counts.
- Streaming works with any OpenAI SDK: direct routes stream live token deltas; multi-stage
  routes emit SSE comment keep-alives between stages, then the final answer.
- Send `X-Kairyu-Trace: 1` to get the legacy `kairyu_trace` string list plus
  `kairyu_trace_v2` (versioned route / role timing / usage / budget events) in unary
  responses. Structured events exclude prompts and generated text.
- The legacy single `orchestrator:` key still works and is served as `kairyu-auto`.

### 6.4 Multi-replica fleets (gateway + replicas)

One binary, two roles, decided by the config file. A **replica** node serves a local
engine; a **gateway** node serves a `ReplicaPool` over remote replicas with auth, metrics,
health probing, and the batch API:

```yaml
# gateway.yaml (see deploy/compose/gateway.yaml)
pools:
  fleet:
    replicas:
      - {backend: openai, options: {base_url: "http://replica-a:8000/v1", upstream: kairyu}}
      - {backend: openai, options: {base_url: "http://replica-b:8000/v1", upstream: kairyu}}
      - {backend: openai, options: {base_url: "http://replica-c:8000/v1", upstream: kairyu}}
    unhealthy_after: 3
    queue_depth_threshold: 8
    probe_interval_s: 5.0

legacy_chat_models: [fleet]          # remote pool has no local tokenizer metadata
```

Requests carrying a session id (`X-Session-ID` header or the OpenAI `user` field) stick to
the replica holding their warm radix-KV prefix; prefix/KV-aware placement is described in
[§2.5](#25-fleet--gateway-layer). Ready-made topologies:

```bash
./scripts/compose_smoke.sh                     # 1 gateway + 3 replicas via Docker compose
docker compose -f deploy/compose/docker-compose.gpu.yaml up    # gateway + GPU replica
docker compose -f deploy/compose/docker-compose.webui.yaml up -d --build --wait
./scripts/webui_smoke.sh                       # full browser E2E, outage, and recovery
./scripts/webui_vlm_smoke.sh                   # 8-GPU Qwen3-VL image-chat E2E
helm install kairyu deploy/helm/kairyu         # k8s chart (+ values-gpu.yaml)
```

The Open WebUI demo mounts [`deploy/compose/config.yaml`](deploy/compose/config.yaml)
at `/etc/kairyu/config.yaml` and serves the keyless CPU-safe direct model
`default` plus the legacy orchestrated model `kairyu-auto`. Open
`http://localhost:3000`, create the first account, and select either model. The
topology pins Open WebUI v0.11.0 by immutable linux/amd64 digest; its named data
volume stores an automatically generated secret, while its mock backends are for
this local demo only. Replace the deployment and orchestrator specs and
authentication policy for production.

The advanced Reasoning Effort selector is wire-compatible: its medium default
maps to Kairyu high (minimal/low→low, xhigh/max→max).

`scripts/webui_smoke.sh` starts the complete topology and uses a separately
pinned Playwright image to prove fresh-user setup, model selection, streaming,
reload persistence, visible gateway failure, and recovery without restarting
Open WebUI. Each run uses an isolated Compose project and deletes only that
project’s ephemeral WebUI volume, leaving the normal demo's data untouched.

For real image chat, add the opt-in eight-GPU overlay. It starts a
revision-pinned Qwen3-VL-32B-Instruct stock-vLLM replica at TP8, routes its
ordered OpenAI content parts through Kairyu, and keeps the same Kairyu-only RAG
endpoint:

```bash
docker compose \
  -f deploy/compose/docker-compose.webui.yaml \
  -f deploy/compose/docker-compose.webui-vlm.yaml \
  up -d --build --wait
```

The VLM boundary accepts one inline PNG, JPEG, or WebP image; remote and local
image URLs are rejected. `scripts/webui_vlm_smoke.sh` runs both direct semantic
and exact-usage checks and a real Open WebUI owned-file upload in the pinned
Playwright browser.

Full production guidance (DC topology, systemd, rolling model updates, observability) is
in [`docs/deployment.md`](docs/deployment.md).

## 7. Configuration reference

Everything a deployment can set, in one place. The single source of truth is the
**DeploymentSpec YAML** passed to `kairyu serve <config.yaml>` (also mounted at
`/etc/kairyu/config.yaml` in the Docker/Helm images).

### DeploymentSpec (YAML)

```yaml
server:                        # versioned deployment schema; explicitly maps to runtime
  host: 0.0.0.0
  port: 8000
  api_keys_env: KAIRYU_KEYS    # env var with comma-separated keys; null = keyless
                               #   (keyless = trusted node-to-node mesh mode)
  max_concurrency: 256         # global in-flight cap on /v1/*; null disables
  ttft_slo_s: null             # direct-chat admit/defer/shed TTFT target; null disables
  metrics: true                # expose /metrics (Prometheus)
  protect_metrics: false       # require an API key for /metrics too
  access_log: true             # one JSON line per request (X-Request-ID echoed)
  tracing: false               # OTel spans (needs the otel extra; no-op without)
  usage_ledger_path: null      # JSONL usage ledger; enables GET /admin/usage
  admin_keys_env: null         # env var for /admin/* mutation keys

tenants:                       # optional authenticated API-key -> tenant mapping
  default_tenant: default      # resolved keys omitted below use this tenant
  key_tenants:                 # every key must occur in comma-separated $KAIRYU_KEYS
    key-a: team-a
    key-b: team-b
  limits:                      # optional independent buckets per known tenant
    team-a: {requests_per_minute: 60, tokens_per_minute: 10000,
             request_burst: 4, token_burst: 2000, max_in_flight: 4,
             interactive_priority: 0, batch_priority: 1}
    team-b: {requests_per_minute: 120, tokens_per_minute: 20000,
             request_burst: 8, token_burst: 4000, max_in_flight: 8,
             interactive_priority: 0, batch_priority: 1}

pricing:                       # optional; requires server.usage_ledger_path
  version: "2026-07-27"        # immutable identifier included in every CSV row
  currency: USD
  rates:                       # blended rates per million tokens
    uncached_input_per_million: "2.00"
    cached_input_discount: "0.50"  # fraction off the uncached-input rate
    output_per_million: "10.00"
  tenant_discounts:            # optional fractions applied after component charges
    team-a: "0.10"

engines:                       # served model name -> one backend
  qwen:
    backend: kairyu            # mock | kairyu | kairyu-proc | openai | vllm
    options:                   # factory kwargs (see backend options below)
      model_path: /models/qwen2.5-0.5b
  remote-a:
    backend: openai
    options: {base_url: "http://replica-a:8000/v1", upstream: kairyu}
    health_url: null           # default: <base_url minus /v1>/health

pools:                         # served model name -> ReplicaPool of N replicas
  fleet:
    replicas:
      - {backend: openai, options: {base_url: "http://replica-a:8000/v1", upstream: kairyu}}
      - {backend: openai, options: {base_url: "http://replica-b:8000/v1", upstream: kairyu}}
    unhealthy_after: 3         # consecutive failures before leaving the ring
    queue_depth_threshold: 8   # session-affinity load valve
    probe_interval_s: 5.0      # background health prober

chat_templates:                # optional local-model override; wins over tokenizer metadata
  qwen: templates/qwen.jinja

legacy_chat_models:            # explicit compatibility for non-render-preserving paths
  [remote-a, fleet, kairyu-auto, kairyu-auto-max]

orchestrator:                  # optional kairyu-auto (OrchestratorSpec YAML)
  spec: orchestrator.yaml

orchestrators:                 # optional NAMED auto models (any number; each an
  kairyu-auto-max:             #   arbitrary worker/role DAG)
    spec: agent_pool_max.yaml

batch:                         # optional OpenAI-compatible /v1/files + /v1/batches
  data_dir: /var/kairyu/batches
  max_concurrency: 4
```

For local `kairyu`/`kairyu-proc` engines, DeploymentSpec loads the chat
template from the tokenizer actually used by the engine: an explicit
`options.tokenizer` directory first, otherwise `options.model_path`. Direct
local vLLM engines use `options.tokenizer` before `options.model`. A tokenizer
root's `chat_template.jinja` and `additional_chat_templates/*.jinja` files take
precedence over `tokenizer_config.json`'s `chat_template`, matching
Transformers; named special tokens from the tokenizer metadata are injected
into the Jinja context. `chat_templates` remains the explicit per-served-model
override. If the effective tokenizer is not materialized locally, that
override must be self-contained: a reference such as `{{ bos_token }}` fails
preflight because Kairyu cannot verify its value instead of silently rendering
an empty string.

Every real text-chat model must resolve a policy before backend startup. An
explicit `chat_templates` override is supported only where Kairyu can preserve
the resulting pre-rendered prompt end to end: local `kairyu`, `kairyu-proc`,
vLLM, or deterministic `mock` engines and compatible static pools. Current OpenAI-compatible remote
backends, discovery-backed pools, and orchestrators cannot preserve that
ownership through an upstream chat template or derived planner/worker prompts,
so they must be listed in `legacy_chat_models`; configuring a template for
them fails startup. That list is a warned compatibility escape hatch for the
old `role: content` concatenator, not a process-wide fallback, and a model
cannot select both policies. The DeploymentSpec builder automatically selects
that compatibility path only for the deterministic `mock` backend, because it
is a protocol test double rather than a trained model. Lower-level
`create_app`, prompt-validation, Responses, and `BatchWorker` callers have no
mock exception. Omitting both policies emits an explicit construction warning,
and every chat request for that model is rejected before dispatch; this keeps
completion-only programmatic apps usable without restoring a silent chat
fallback. Every served chat model must receive a `ChatTemplate` or explicit
membership in `legacy_chat_models`.

VLMs have no deployment-preflight exception either. The checked-in image-chat
overlay lists its remote served model in `legacy_chat_models` for the text-only
path; image-bearing requests bypass that renderer and preserve structured
messages for the upstream processor. A pre-rendered Kairyu template is rejected
for that remote backend at startup, preventing double templating.

For distributed tracing, install `--extra otel` and set
`server.tracing: true`. Kairyu propagates W3C `traceparent`/`tracestate` only,
does not propagate baggage, and does not replace the process-global tracer
provider. An embedding application may use its existing provider or inject a
private one through `configure_tracing`. The Compose validation path uses
`OTEL_TRACES_EXPORTER=console` with a distinct `OTEL_SERVICE_NAME` per service;
its compact `KAIRYU_OTEL_SPAN` records are a deployed-smoke proof, not the
recommended production exporter. Kairyu does not put exception text, stack
traces, prompts, or outputs in its span attributes or exception events.

The shown batch block is the single-gateway filesystem default. For multiple
gateways, install `--extra fleet` and select `store: postgres`,
`dsn_env: KAIRYU_BATCH_POSTGRES_DSN`, and a common `store_id`; see
[`docs/deployment.md`](docs/deployment.md).

### Backend options (`engines.*.options`)

| backend | option | default | meaning |
|---|---|---|---|
| `kairyu` | `model_path` | — | safetensors checkpoint dir (Llama-3.x / Qwen2 / Qwen3 / Qwen3-MoE / DeepSeek-V3; FP8/INT8/AWQ/GPTQ/NVFP4 quantized checkpoints auto-detected) |
| | `generation_config` | `"auto"` | model sampling-default policy: `auto` applies the model file for omitted request fields, `vllm` uses neutral sampling defaults, and `none` ignores the file |
| | `nvfp4_accuracy_profile` | `null` | opt-in NVFP4 projection selectors (`fp8`, `dynamic_activation`, `saturation_counters`) for measured accuracy/memory experiments; default execution is unchanged |
| | `tokenizer` | model dir | HF tokenizer dir override (`tokenizer.json`) |
| | `num_pages` | 4096 | KV pool pages |
| | `page_size` | 16 | tokens per KV page |
| | `max_num_batched_tokens` | 2048 | chunked-prefill budget per step |
| | `speculative` | null | `"ngram"` (prompt-lookup), `"eagle"`, or `"mtp"` (learned draft heads) enables speculative decoding |
| | `speculative_tokens` | 4 | draft length k |
| | `tensor_parallel_size` | 1 | TP degree; >1 spawns a multi-process TP worker group from the serve process (gloo on CPU, NCCL on GPU) |
| | `pipeline_depth` | 1 | unified engine-loop depth; 1 preserves synchronous scheduling, 2+ schedules immutable step snapshots ahead of the oldest device commit |
| | `decode_mode` | `"eager"` | `"cuda_graph"` enables bucketed CUDA-graph decode for a real CUDA model; unsupported hardware/model/attention combinations fail at startup |
| | `cuda_graph_max_batch` | `max_num_seqs` (256) | largest captured decode batch; larger batches use eager decode |
| | `cuda_graph_max_pages` | covers the model context | fixed page-table width per captured bucket; the default covers the configured context bounded by usable KV pages; longer sequences use eager decode; cannot exceed the usable KV page capacity |
| | `cuda_graph_warmup_iters` | 3 | side-stream warmup iterations before first capture |
| | `pd_separation` | `false` | build separate prefill and decode engines; currently requires a real model, TP=1, eager decode, and no speculative decoding |
| | `pd_prefill_device` | auto | prefill role device (`cpu` or `cuda:N`); requires `pd_separation`; its hardware profile and attention backend are selected for this device |
| | `pd_decode_device` | auto | decode role device (`cpu` or `cuda:N`); must have the same CPU/CUDA type as the prefill role; deferred cross-device CUDA requires P2P access |
| | `pd_defer_handoff` | `true` | defer publication/release until physical transfer completion; `false` synchronizes the source and destination transfer streams before publication |
| `kairyu-proc` | same as `kairyu` | — | runs the engine in a separate process over ZMQ/msgpack (crash isolation) |
| `openai` | `base_url`, `api_key`, `model` | — | any OpenAI-compatible endpoint |
| `vllm` | vLLM engine kwargs | — | needs a Linux GPU host with vllm installed |
| `mock` | — | — | deterministic CI backend |

### Environment variables

| variable | effect |
|---|---|
| `KAIRYU_ATTENTION_BACKEND` | `auto` \| `torch` \| `flashinfer` \| `flashattention3` \| `flashattention4` — selects the attention backend |
| value of `server.api_keys_env` | comma-separated API keys |
| `KAIRYU_BENCH_CACHE` | benchmark dataset cache dir (default `~/.cache/kairyu/benchmarks`) |
| `KAIRYU_MODEL_DIR` | model volume for `docker-compose.gpu.yaml` |
| `GLOO_SOCKET_IFNAME` | set `lo0` on macOS if gloo rendezvous fails (dist tests) |
| `OTEL_TRACES_EXPORTER` | `console` enables Kairyu's private compact smoke exporter when tracing is enabled; omit for an application-owned provider |
| `OTEL_SERVICE_NAME` | service identity used by the compact smoke exporter |

Explicit selections are strict: a missing package, unsupported GPU, or
unsupported tensor shape fails startup with an actionable error. FA3 and FA4
own prefill while FlashInfer owns paged decode; `/backends` reports both
components. FA4 consumes paged KV directly on SM90/SM100/SM110. On SM120 it
preserves page identity while materializing the selected pages
device-to-device for prefill. FA4 beta24 caches one architecture per process,
so Kairyu verifies the selected device, environment override, and upstream
cache at startup; mixed-SM FA4 roles must run in separate processes. `auto`
uses the stable hardware-profile fallback (FlashInfer on supported GPU tiers,
otherwise torch) and promotes neither FA3 nor FA4 unless retained,
profile-specific correctness and performance evidence exists. If an optional
profile-selected backend cannot be constructed, `auto` alone falls back to
torch and `/backends` reports that actual fallback and its sanitized failure
type; an explicit selection still fails startup. Official deployments eagerly
construct `kairyu-proc` before accepting traffic, so the same checks apply
before `/readyz` can report the process as serving.

### HTTP surface

`/v1/chat/completions` (SSE, tools, logprobs, n>1, `response_format: json_schema`,
vision content-parts wire format), `/v1/completions`, `/v1/embeddings`
(float + base64), `/v1/responses` (`input`, `instructions`, canonical typed SSE,
flat and namespaced function tools, `function_call_output`, tenant-scoped
`previous_response_id`), `/v1/models`, `/v1/files` + `/v1/batches`, `/health`,
`/v1/route`, `/routing`,
`/readyz`, `/backends`, `/metrics`, `POST /admin/drain` / `POST /admin/undrain` (auth-protected;
drain flips readyz to 503), `GET /admin/usage?tenant=` (when the ledger is enabled).
With `pricing:` configured, `GET /admin/usage.csv?tenant=&start_ts=&end_ts=`
exports a caller-scoped invoice CSV.

For native engines, `/backends` includes the generation-config mode, resolved
source, and all five effective sampling defaults. A local pool promotes that
record only when every member supplies the same complete policy. A gateway's
single remote audit sample remains explicitly nested under `via_replica` and is
never presented as a fleet-wide default. Orchestrated models expose every
worker record and collapse one top-level policy only when all workers agree.

Request extras: `X-Session-ID` (or the OpenAI `user` field) pins a session to the
replica holding its warm KV prefix; `X-Kairyu-Trace: 1` adds the legacy
`kairyu_trace` and versioned `kairyu_trace_v2` blocks to unary `kairyu-auto`
responses; `stream_options: {include_usage: true}` appends the final usage chunk.
Named AUTO models accept the same sampling, `n`, logprob, tool-choice, and
structured-output fields as direct chat models. Scalar sampling reaches private
orchestration stages under the advertised private max-token policy, while the
exact public max-token limit, `n`, logprobs, tools, and response grammar apply
to the selected final worker or synthesis boundary.

The Responses stream uses OpenAI event names and gapless sequence numbers from
`response.created` through `response.completed` or `response.incomplete`; failures
terminate with typed error/failed events. Successful stored responses can be continued
with `previous_response_id`, but IDs never cross tenant boundaries. Function-call
arguments and outputs round-trip through the normal model chat template and request
capability checks. Run `scripts/codex_responses_smoke.sh` against a serving model to
exercise an unmodified Codex CLI over `wire_api="responses"`; deployment details and
unsupported Responses features are listed in `docs/deployment.md`.

`POST /v1/route` accepts `{model, messages}` and renders the same model-specific chat
template as actual chat before calling the Router's non-mutating `preview()`. It never
dispatches an engine and returns `binding:false`; concurrent traffic can still make a later
stateful-router decision differ. `GET /routing` returns whitelisted Router settings,
effective target fallback, safe engine metadata, role dependencies, and budget. It is not an
auth-exempt path. Unary auto-model responses include structured `kairyu_route` only with
`X-Kairyu-Trace: 1`; the same opt-in adds the privacy-safe structured execution
trace described in `docs/design/observability-trace-contract.md`. The normal
OpenAI-compatible response shape is unchanged.

### Multi-tenancy

For `kairyu serve`, the primary configuration path is the optional `tenants:` block in
the DeploymentSpec above. In that example, `KAIRYU_KEYS` must resolve to a
comma-separated value containing `key-a,key-b`: every `key_tenants` key must be a member
of the resolved data-plane API-key set, while a resolved key omitted from the mapping uses
`default_tenant`. Unknown keys and limits for unknown tenants fail during deployment
preflight, before owned backends are constructed. The mapping keys are actual API-key
values rather than environment-variable names, so protect the deployment file as
secret-bearing configuration.

Each tenant gets independent request-per-minute and token-per-minute buckets,
explicit request/token burst capacities, an optional in-flight cap, and trusted
interactive/batch scheduling classes. A configured in-flight lease is acquired
before the global concurrency guard and held through the final SSE/body byte;
after request validation, Chat, Responses, Completions arrays, Batch lines, and
AUTO atomically reserve worst-case compute tokens before replica dispatch.
Candidate prefill (`n`/`best_of`) and AUTO internal fan-out are included.
Exact single-candidate terminal usage refunds unused capacity; failure,
disconnect, missing usage, and multi-candidate work retain the conservative
reservation. This prevents a cold or resource-heavy burst from consuming every
shared gateway/GPU slot. Omitted
burst fields retain the historical one-minute bucket capacity, and omitted
`max_in_flight` remains unlimited, so latency-isolated deployments should set
all three explicitly. Smaller priority integers run first. HTTP clients
cannot self-promote through a configured gateway: interactive requests receive
`interactive_priority` (default 0), and Batch API lines receive `batch_priority`
(default 1); every profile must keep interactive priority strictly smaller than
batch priority. The trusted value and bounded class are preserved through Kairyu
replica transport, while local vLLM execution forces its priority scheduling
policy so the integer cannot be silently ignored. A tenant
without an explicit profile uses the defaults (600 requests and 200,000 tokens per
minute). Authentication runs before tenant limiting, so a rejected credential does not
consume a bucket. When `server.usage_ledger_path` is configured, successful request usage
is grouped under the mapped tenant name in the JSONL ledger and `/admin/usage` totals.
Every new row stores cached and uncached input explicitly. Older rows derive uncached
input as `prompt_tokens - cached_tokens`.

Programmatic callers can still pass the existing runtime configuration directly:
`create_app(..., tenant_config=TenantConfig(key_tenants={"key-a": "team-a"},
limits={"team-a": TenantLimits(requests_per_minute=600,
tokens_per_minute=200_000, request_burst=8, token_burst=20_000,
max_in_flight=8)}))`.

### Pricing and invoice export

`pricing:` is a versioned blended price sheet, not model-attribution billing.
All monetary inputs are parsed as decimal values; quote them in YAML to retain
the intended precision. `cached_input_discount` and `tenant_discounts` are
fractions in `[0, 1]`. Charges are rounded half-even to six currency decimal
places, component rows reconcile to the displayed subtotal, and the tenant
discount is then applied.

Invoice periods use `[start_ts, end_ts)` Unix timestamps. The CSV separates
uncached input, cached input, and output quantities, unit rates, component
charges, discount, and total. Each row includes the price-sheet version,
currency, a SHA-256 of the immutable ledger snapshot, and a deterministic
invoice ID. Admin keys can export all tenants or select one; data-plane keys
remain scoped to their mapped tenant. Malformed or truncated ledger input
returns `invoice_ledger_invalid` instead of a partial invoice.

### Deployment artifacts

| artifact | purpose |
|---|---|
| `Dockerfile` / `Dockerfile.cuda` | CPU / CUDA images (one image per role; the mounted spec decides) |
| `deploy/compose/docker-compose.yaml` | gateway + 3 CPU replicas smoke topology |
| `deploy/compose/docker-compose.gpu.yaml` | gateway + GPU replica (nvidia device reservation) |
| `deploy/compose/docker-compose.webui.yaml` | Open WebUI chat surface on the gateway |
| `deploy/compose/docker-compose.webui-vlm.yaml` | opt-in Qwen3-VL-32B TP8 image-chat overlay |
| `deploy/helm/kairyu/` (+ `values-gpu.yaml`) | k8s chart; readiness `/readyz`, per-GPU-profile nodeSelector |
| `scripts/kind_smoke.sh` | end-to-end kind cluster smoke (CI job) |
| `scripts/webui_smoke.sh` | pinned Open WebUI browser E2E: first user, direct/AUTO streaming, persistence, outage, recovery |
| `scripts/webui_vlm_smoke.sh` | eight-GPU direct + Open WebUI image-chat E2E |
| `scripts/gpu_gates/*.sh` | GPU-day gate scripts (runbook §0–§9); all support `--dry-run` |
| `verification/l1/performance/serving_bench.py`, `verification/product/performance/frontier_compare.py`, `verification/l1/performance/kv_transfer_bench.py` | latency/goodput/transfer benches |

## 8. Evaluation and verification

Repository model evaluations and Kairyu implementation gates are separate
checkout-only surfaces. Neither is shipped in the wheel.

| Surface | Owns |
|---|---|
| `evals/` | Core, Quantization, Structured Output, and Long Context model evaluations |
| `verification/` | L1, orchestration, fleet, and product correctness, performance, resilience, and diagnostic gates |
| `evidence/` | neutral strict JSON, hashing, catalog, and execution-record contracts |
| `bench/results/` | immutable legacy artifacts whose existing paths and bytes remain unchanged |

Install evaluation dependencies and run one explicit suite:

~~~bash
uv sync --extra evals --group dev
uv run --frozen python -m evals list --suite core
uv run --frozen python -m evals run --suite core --smoke \
  --base-url http://localhost:8000/v1 --model m1
uv run --frozen python -m evals run \
  --config evals/configs/structured.yaml
~~~

A suite or config is required. Core is the judge-free GSM8K, MMLU, and IFEval
regression set; Quantization adds GPQA Diamond for the declared seven-arm
quality sweep; Structured Output uses paired constrained and unconstrained
requests; Long Context measures fixed RULER NIAH lengths. Dataset pins, scorer
bytes, target identity, and sampling policy are bound into run evidence.

Kairyu engine and system claims use the verification registry:

~~~bash
uv run --frozen python -m verification list
uv run --frozen python -m verification check
uv run --frozen python -m verification run \
  orchestration.performance.router_latency -- --json --assert-gate
~~~

Formal performance execution requires an accepted correctness artifact and
records its exact SHA-256. HF/TP parity, logprob agreement, batch invariance,
quantization/KV equivalence, TTFT, TPOT/TPS, throughput, goodput, and vLLM
comparisons therefore stay out of model-evaluation rows.

Full guides: [`docs/benchmarks.md`](docs/benchmarks.md) and
[`verification/README.md`](verification/README.md).

## 9. Development

```bash
uv run --frozen pytest --fail-on-skip  # portable tests + coverage (gate: 80%)
uv run --frozen ruff check .           # lint (E, F, I, UP, B; line length 100)
uv run --frozen python scripts/verify_verification_registry.py
uv run --frozen python scripts/verify_bench_wheel.py
uv run --frozen python verification/orchestration/performance/router_latency.py
uv run --frozen python verification/orchestration/performance/orchestration_mock_bench.py
```

| pytest invocation | scope |
|---|---|
| `pytest --fail-on-skip` (default selection) | portable CPU tests, including `tests/dist`; environment suites are explicitly deselected |
| `scripts/gpu_gates/03_deferred.sh` | prerequisite-checked CUDA kernel/graph suite; selected skips fail |
| `pytest --fail-on-skip -m hf_hub` | opt-in real-checkpoint downloads after network/model prerequisites are available |
| `scripts/helm_integration.sh` | prerequisite-checked Helm rendering tests |
| `scripts/postgres_integration.sh` | pinned-container PostgreSQL integration tests |
| `scripts/kind_async_request_gate.sh` | staged CPU/kind test for three-gateway durable AsyncRequest lifecycle and failover |
| `scripts/async_request_retention.py` | preview, prepare, and apply bounded AsyncRequest request/audit retention |
| `pytest --fail-on-skip -m dist` | multi-process gloo tests (also included in the default run) |

Conventions: all CI-facing tests run against `MockBackend` (deterministic,
dependency-free) unless their named integration gate provisions the real dependency.
An absent prerequisite is reported as not run with a nonzero status; selected skips
cannot make a CI job pass. GPU-dependent claims are never reported without a `bench/`
reproduction script.

## 10. Documentation index

| document | contents |
|---|---|
| [`PROGRESS.md`](PROGRESS.md) | cross-session change log: design decisions, milestone status, blockers |
| [`docs/design/`](docs/design/) | one reviewed design doc per milestone (m1–m19, GPU-day seams) |
| [`docs/goals/`](docs/goals/) | evidence-first goal contracts (multi-GPU, MoE engine, fleet scale, product surface) |
| [`docs/deployment.md`](docs/deployment.md) | production deployment: DC topology, systemd, rolling updates, observability |
| [`docs/gpu-runbook.md`](docs/gpu-runbook.md) | consolidated GPU-day execution plan (performance gates, kernel tuning, fabric bring-up) |
| [`docs/benchmarks.md`](docs/benchmarks.md) | installed benchmark suites guide |
| [`docs/ide-clients.md`](docs/ide-clients.md) | Cline/Continue setup, supported API surface, and SOCKS validation |

## 11. License

MIT — see [LICENSE](LICENSE).

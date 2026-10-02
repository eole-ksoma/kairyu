# M1 Design: Orchestration Layer (L2) + Interface Layer (L3) on a vLLM Backend

Status: Draft for review (implementation proceeds in parallel per autonomous-goal mode; revisit on review feedback)
Milestone: M1
Date: 2026-07-02

## 1. Goal

Ship a usable Kairyu package before the custom engine (M2) exists:

- `from kairyu import LLM, SamplingParams, Orchestrator` — drop-in signature compatibility
  with vLLM's offline `LLM` API so vLLM examples run with only an import rewrite.
- L2 orchestration natively built in: pluggable **Router** (tier1 / tier2 / multi-agent),
  **Conductor** (Planner/Worker/Verifier/Synthesizer role DAG, async dispatch, budget-bounded
  recursion), and **MoA** (parallel sampling + synthesis).
- OpenAI-compatible HTTP server (`/v1/chat/completions`, SSE streaming, tool calling).
- YAML / decorator DSL for declaring agent pools, role DAGs, and budgets.

## 2. Key design decisions and rationale

### D1. Engine access goes through a small async `EngineBackend` protocol

All of L2/L3 talks to a `EngineBackend` protocol (`generate(request) -> GenerationResult`,
plus a streaming variant). Backends in M1:

| Backend | Purpose |
|---|---|
| `MockBackend` | Deterministic, dependency-free. All unit/CI tests run against it. |
| `VLLMBackend` | Wraps `vllm.AsyncLLMEngine` behind an import guard. Only loads when vLLM is installed (Linux+GPU). |
| `OpenAICompatBackend` | External API workers (OpenAI/Anthropic/Gemini via their OpenAI-compat endpoints) over httpx. Used by the Conductor for frontier-tier roles. |

Rationale: this repo is developed on macOS without CUDA; vLLM cannot even be imported here.
The protocol boundary is also exactly the seam where the M2 custom engine plugs in — M2 is
"add a fourth backend", not a rewrite. This mirrors how SGLang and vLLM both isolate their
schedulers behind an engine-client interface.

For `OpenAICompatBackend` streaming, choice presence is established by an observed upstream
choice index rather than by non-empty text. Role-only, finish-only, and empty-content choices
remain valid empty completions (including within `n > 1`), while a stream that observes no
choices at all remains an upstream failure.

**Deployment-reference amendment (2026-08-13, EO-D1).** A deployment-loaded
L2 worker may borrow an already-built L1 engine or pool by `engine_ref`. The
deployment retains lifecycle ownership and L2 dispatches to the exact object;
it must not construct an HTTP worker that loops through the deployment's own
L3 surface. Standalone factory-backed DSL workers retain their existing owned
lifecycle. See `example-layered-orchestration.md`.

**Typed-prompt amendment (2026-07-30, issue #227):** the public backend seam now
accepts a nominal `PromptInput`: legacy `str`, `TextPrompt`, `TokensPrompt`, or
`MultimodalPrompt`. Legacy strings retain their exact behavior. A
`TokensPrompt` carries an immutable, non-empty tuple of non-negative integer IDs
as the sole execution/cache/accounting authority; its optional text is display
metadata and is never encoded, compared with the IDs, or used as a cache key.
The caller owns templates, BOS/special tokens, and rendered tool instructions,
while the selected backend tokenizer still owns output detokenization, EOS, and
stop strings. Consequently an unrendered tool suffix on a token prompt fails
before dispatch.

`MultimodalPrompt` preserves an ordered text/token base and explicitly encoded
URI/base64/bytes/JSON items through one strict tagged codec. Unknown tags,
extra fields, and malformed payloads fail rather than being flattened or
dropped. No current Kairyu backend declares a modality processor, so every
multimodal request is represented losslessly but rejected during capability
preflight. Native Kairyu, `kairyu-proc`, and the vLLM adapter execute text and
token prompts; the OpenAI Chat adapter and L2 orchestration remain explicitly
text-only. `OpenAIRequestCapabilities.prompt_kinds` participates in the same
immutable replica-validation key introduced by issue #209.

`GenerationRequest.prompt` is the only prompt-content authority.
`SamplingParams.extra_args` defensively copies and freezes its top-level
mapping and rejects vLLM/OpenAI/runner alternate carriers plus the
prompt-owned `cache_salt`; `CacheHint` remains the cache-affinity authority.
The request constructor repeats this check as a dispatch-boundary defense.
Chat message extras/content parts, Responses content, and offline Mapping
inputs likewise fail if a text renderer or adapter would otherwise drop input.
Text chat renderers explicitly reject image parts and direct callers must use a
`MultimodalPrompt` with a capable backend instead.

Native RadixKV identity remains the processed token tuple, so equivalent text
and caller-tokenized requests can reuse the same pages. The gateway does not
own a tokenizer and therefore does not infer that cross-domain equivalence:
non-text requests bypass the existing 256-character `PrefixIndex` but retain
session affinity, and attaching its text-only `CacheHint.prefix_fingerprint`
to a non-text request is an error.

### D2. vLLM compatibility is signature-level, verified by contract tests

`kairyu.SamplingParams`, `kairyu.LLM`, `kairyu.RequestOutput`, `kairyu.CompletionOutput`
replicate vLLM's public constructor/attribute surface (the subset exercised by vLLM's
official `examples/offline_inference/basic.py` and the OpenAI server examples).
A contract test suite pins the surface (`tests/compat/`); when vLLM is installed the same
suite additionally cross-checks against the real vLLM classes (skipped otherwise).
We deliberately do NOT subclass or re-export vLLM types: Kairyu must work without vLLM
installed, and M2 replaces the backend entirely.

### D3. Router is a protocol with a rule-based first implementation

`Router.route(query, context) -> RouteDecision` where `RouteDecision.target` is
`tier1 | tier2 | multi_agent` plus a confidence and the extracted features (for logging /
M4 training data). First implementation `RuleRouter` uses pure-Python feature extraction
(length, code-fence presence, math/reasoning keywords, multi-step markers) — no model in
the hot path, so the <10ms latency budget is trivially met and enforced by a test.
The protocol seam is where the M4 learned classifier / contextual bandit slots in.
Every decision is emitted to a `RouterLog` (JSONL) — this is the M4 training corpus.

### D4. Conductor is an explicit role DAG executed with asyncio

Roles (`planner`, `worker`, `verifier`, `synthesizer`, or custom) are nodes of a declared
DAG; edges are data dependencies (a node's prompt template can reference upstream outputs).
Execution is a topological wave schedule with `asyncio.gather` per wave — no threads, no Ray
in M1 (YAGNI; Ray arrives with multi-node). Recursive self-correction is modeled as a
verifier-gated retry loop with depth bounded by `Budget.max_refine_depth`; spend is bounded
by `Budget.max_cost_usd`, charged per generation via a pluggable `CostModel` (default
zero-cost; `chars_cost_model` estimates from prompt+completion volume, and the DSL exposes
`budget.cost_per_1k_chars_usd`). Step admission is strict and happens synchronously before
dispatch: a generation reserves its step before any `await`, and an operation that cannot
reserve its complete step requirement is skipped. Result-priced work also claims one
exclusive unknown-cost admission slot when a cost cap is configured, so parallel waves
cannot all dispatch against the same stale pre-charge balance. Success reconciles the
reservation with actual cost exactly once; failure and cancellation release the complete
reservation before the exception propagates. MoA is one atomic operation and must reserve
all proposal plus synthesis steps (`moa_samples + 1`) and its cost slot before any proposal
dispatches.

The dispatch limits are strict, but an admitted generation's exact cost is unknowable until
its result exists. That one generation may therefore cross `max_cost_usd`; accounting keeps
the full actual cost (never clamps or hides it) and reports the exhausted/overrun state for
querying while refusing later work. A wall-clock deadline bound is deferred to M2, where
the engine can enforce it per-step. Exceeding a budget is a normal, reported outcome (best
result so far is returned), not an exception, matching Fugu's "recursion depth as
inference-time compute axis" framing.

### D5. KV-affinity is designed in now, exploited in M2

The differentiation core (multi-step orchestration hitting shared-prefix KV cache) needs the
M2 Radix-Paged KV manager for full effect. In M1 we (a) keep every Conductor/MoA step's
prompt as `shared_prefix + role_suffix` (structural invariant, tested), and (b) when the
vLLM backend is active, enable `enable_prefix_caching=True` and route all steps of one
orchestration to the same engine so vLLM's block-hash prefix cache already gets hits.
The `GenerationRequest.cache_hint` field (session id + prefix fingerprint) is plumbed
through now so M2 can consume it without interface changes.

**Cache-partition amendment (2026-08-08, issue #366):** `CacheHint` controls
placement and affinity; it is not part of native RadixKV identity. A native
engine process deliberately reuses pages only by the exact processed token
tuple and therefore shares identical prefixes across its callers. Kairyu does
not currently claim an in-process tenant cache-partition boundary. Deployments
that require cache isolation must use separate backend pools/processes; the
vLLM prompt-owned `cache_salt` remains rejected rather than being silently
dropped or misleadingly mapped to an affinity hint. Adding an identity salt
requires a separate end-to-end design across RadixKV, DRAM tier keys, KV events,
and fleet routing.

### D6. Server is FastAPI + SSE, one process, engine-agnostic

`kairyu.entrypoints.server` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
Tool calling passes `tools`/`tool_choice` through to the backend and parses tool-call
output into OpenAI's `tool_calls` schema. For required or named tool choice, every returned
choice must retain at least one permitted tool call after per-choice filtering; any mixed or
empty result is rejected without regeneration, including before buffered SSE emission, while
the consumed generation is metered exactly once. An `x-kairyu-orchestrate` request field (or
model name `kairyu-auto`) routes a request through the Orchestrator instead of a raw engine —
one endpoint, Fugu-style.

### D7. DSL: YAML is the source of truth; decorators build the same objects

YAML loader produces pydantic-validated `OrchestratorSpec` (agent pool, role DAG, budgets).
The `@role` decorator API constructs identical spec objects in Python. One schema, two
front-ends; the Conductor consumes only the spec.

### D8. Checklist verifiers: deterministic checks and System One probabilities (2026-10-01)

Status: accepted by the owner (2026-10-01, framework scope for the
checklist-verified example); CPU tests in `tests/unit/test_conductor_checklist.py`,
`tests/unit/test_checks.py`, `tests/server/test_orchestration_usage_trace.py`;
GPU evidence in `examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md`.

A verifier may declare a `checklist:` instead of a generation prompt. It judges
its target attempt without generating: deterministic checks
(`kairyu/orchestration/checks.py`) run first and FAIL the attempt without a
model call; generation roles that the verifier depends on and that depend on
the target (for example a claim extractor) run inline on every attempt; then
`noul` questions go to a System One backend (m11 D8) named by a
`systemone_ref` worker. Each checklist item gets a probability `p` (a check is
1 or 0; questions sharing an item id aggregate by minimum) and passes at
`p >= threshold`. The verdict text keeps the D4 verifier contract (first line
PASS/FAIL, one feedback line per failing item), so the refine loop and budget
accounting are unchanged. Supporting mechanisms:

- **Jev request shape (owned by Kairyu).** The checklist never sends
  free-form prompt text to System One. The `state` is a JSON object built
  from declared sections: `query` becomes the request's role-tagged messages
  (unwrapped from Kairyu's L2 transcript, shared delimiters in
  `orchestration/request.py`), role outputs that are JSON are embedded as JSON
  values, and long text is cut with a marker. Each question is a `noul`
  whose `instructions` is an object — by default `{question: "Does <subject>
  satisfy the requirement below?", requirement, ...context}` with
  `criteria.true/false` for full versus partial or missing satisfaction, or
  an explicit `ask` with its own criteria. This matches how Jev servers read
  a request (a system prompt listing `Question qN` with its yes/no meanings,
  the state as the user turn, one label token per question), keeps each
  question self-contained, and keeps judged material from posing as
  instructions to the judge.
- **Items from JSON.** Checks and questions can be expanded per item of an
  upstream role's JSON list (`foreach`), or per pair of items sharing a value
  (`pairs_sharing`), with str.format templates.
- **Internal grammar.** `sampling.response_format` gives an internal role a
  grammar, or `inherit` applies the caller's `response_format` to it too. The
  final unit still carries only the caller's intent.
- **Seeded target and refinement prompt.** `seed_from` publishes an upstream
  role's output as attempt 0 without a model call; `refine_prompt` renders a
  refinement from `{previous}` and `{feedback}` instead of the appended
  default. `checklist.max_refinements` bounds one verifier below
  `budget.max_refine_depth`.
- **Outcomes.** On exhaustion, `on_exhausted: latest_checks_passed` publishes
  the newest attempt whose deterministic checks passed. When the checklist
  cannot be judged (System One down or overloaded, a source list missing, the
  state too large, no budget), `on_unavailable: publish_unverified` publishes
  `unverified_from` (or the attempt) instead of failing the unit; once one
  checklist of a run is unjudgeable, later checklists report unverified too.
  The default stays the D4/#496 contract (an unjudged final unit is an error).
- **Curation.** `curate` drops, merges and pads the target's JSON list from
  the final probabilities, so downstream roles read the edited list.
- **Response.** A checklist on the selected final unit publishes
  `kairyu_verification: {guaranteed, reason, threshold, attempts,
  requirements: [{id, proposition, sources, kind, group, p, passed}]}` on the
  chat response (the terminal chunk when streaming), without a trace opt-in.

Why (framework boundary): (1) System One is a served Kairyu API (m11 D8), but
the L2 DSL could only branch on generated PASS/FAIL text — `engine_ref`
resolves engines and pools only, logprobs are stripped from internal stages,
and a failed verifier fails the answer. (2) The executor contract (ECO-D3) is
Python/pytest-shaped and still needs a generation verifier to decide. (3) Any
DSL that wants calibrated probability gates, cheap deterministic pre-checks,
or a verified answer with an honest "unverified" flag needs the same
mechanism, independent of the example. (4) The mechanism knows checks,
questions, thresholds and outcomes; which requirements exist, their wording,
thresholds, repair prompts and curation policy stay in the example's YAML.

### D9. System One profile judge (2026-10-01)

Status: accepted by the owner (2026-10-01); CPU tests in
`tests/unit/test_profile_judge_systemone.py`; GPU evidence in
`examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md` (`routing`).

A `profile_judge` whose `worker` is a `systemone_ref` worker routes by System
One probabilities instead of a generated label. Kairyu sends the
role-tagged conversation (each message cut to `max_message_chars`, plus
tool/image flags) as the state and one `choice` question whose criteria are
the choices' criteria. `prefer: {label, min_probability}` selects that
label whenever its probability reaches the floor (an accuracy-first
policy), otherwise the most probable route wins. Timeouts, transport
failures, non-200 replies and malformed answers apply `fallback`. The judge
event records `p_<LABEL>` per route, and admission bounds the read by its
body size (System One bills at most its input without `think`).

Why (framework boundary): (1) the LLM judge reads only a generated label
string; (2) `ProfileJudge` workers were generation engines only, so a
calibrated System One classifier could not route; (3) any DSL can route
on Jev-style probabilities with this; (4) the labels, criteria and floor
stay in the example.

Additions in the same change: generation trace events record their
`reasoning_effort`; checklist items carry report `tags`; a final-unit
checklist appends a "Verification" section to exposed internal work; an
upstream checklist that ends without PASS is judged again on its curated
output and failures in its `guarantee_groups` block the run's guarantee
(`requirements_unconfirmed`); curation merges only `merge_only_where`
items; `items_in_sources` can restrict evidence to `message_roles`; a
seeded final unit republishes its seed's completion metadata and refuses
`n > 1`; a failed System One read cancels its siblings and completed reads
keep their usage (PR #616 review).

## 3. Out of scope for M1 (deferred with reasons)

- Custom scheduler / KV manager / CUDA graphs / spec decode / quantized load — M2/M3.
- Learned router training pipeline — M4 (M1 emits the logs it will train on).
- Multi-node (Ray), P-D disaggregation — M3+.
- xgrammar structured output — arrives with the custom engine (vLLM backend already
  supports `guided_json` passthrough, exposed but not wrapped).

## 4. Testing strategy

- Unit tests for every module against `MockBackend` (no network, no GPU), pytest-asyncio.
- Contract tests pin the vLLM-compatible API surface; cross-check vs real vLLM when present.
- Server tests via `httpx.ASGITransport` (no socket).
- Router latency test asserts p99 < 10ms over 1000 routes.
- Coverage gate ≥ 80% in CI (GitHub Actions, `uv` + Python 3.11/3.12 matrix).

## 5. Package layout

```
kairyu/
  __init__.py               # LLM, SamplingParams, Orchestrator, RequestOutput, ...
  sampling_params.py        # vLLM-compatible SamplingParams
  outputs.py                # CompletionOutput / RequestOutput
  engine/
    backend.py              # EngineBackend protocol, GenerationRequest/Result, cache_hint
    mock.py                 # MockBackend
    registry.py             # backend factory/registry
    vllm_backend.py         # import-guarded vLLM adapter
    openai_backend.py       # external OpenAI-compatible API worker
  orchestration/
    features.py             # query feature extraction (pure functions)
    router.py               # Router protocol, RuleRouter, RouteDecision, RouterLog
    budget.py               # Budget, BudgetTracker
    conductor.py            # RoleSpec DAG + async executor
    moa.py                  # Mixture-of-Agents
    orchestrator.py         # Orchestrator facade (route -> engine | conductor | moa)
  dsl/
    spec.py                 # pydantic OrchestratorSpec schema
    loader.py               # YAML front-end
    decorators.py           # @role / @agent_pool front-end
  entrypoints/
    llm.py                  # vLLM-compatible LLM class
    server/
      protocol.py           # OpenAI request/response pydantic models
      app.py                # FastAPI app, SSE streaming, tool calls
bench/                      # reproduction scripts (M1: harness skeleton + mock run)
tests/{unit,compat,server}/
docs/design/
```

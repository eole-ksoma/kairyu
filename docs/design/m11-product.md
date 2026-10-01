# M11 Design: Fugu-Class Product Surface + Tenancy — Implemented

Status: **Implemented** (2026-07-03; D2 amended 2026-08-12; D4/D5/D7 amended 2026-07-31;
D1/D2/D4/D6 amended 2026-07-28; D3 amended 2026-07-27). Reviewed
(1-reviewer panel with file/line evidence + OpenAI SDK verification,
2026-07-03; §5 binding).
Milestone: M11 (roadmap P-B/P-C + F5 CPU halves; goal G6)
Date: 2026-07-03
Depends on: m7 Orchestrator/Conductor/MoA/Budget, m9 server surface, M10a
(BatchStoreProtocol, telemetry), m8 scheduler. Consumed by: production
launch (G6 gates).

## 1. Goal

Close the product gaps that make Kairyu a Fugu-class service rather than a
bare inference server: streaming orchestration with honest usage, tiered
auto models, multi-tenant limits+metering, Responses/Embeddings APIs,
vision wire format, and the F5 latency-protection logic (priority admission,
SLO shedding, autoscaler decisions) — all CPU-tested.

## 2. Key decisions

### D1 — Streaming orchestrator with token accounting

`Orchestrator.run_chat(messages, stream=False)` joining the m7 `run()`
path: (a) token accounting — every internal `GenerationResult.usage` is
summed into `OrchestratorResult.usage` (`orchestration_input/output_tokens`)
and surfaced through the API as REAL usage (removing the m9 "usage=None
until M11" fallback); (b) streaming — the FINAL unit (direct route,
conductor final unit, or MoA synthesis) streams token deltas; pre-final
stages emit typed `status` keep-alive events (`OrchestratorEvent =
status|delta|result`). `X-Kairyu-Trace: 1` request header opts into a
trace block (stage timings + decisions) in the final event. app.py's AUTO
path switches to `run_chat` for both stream and non-stream.

**Pull-through amendment (2026-07-27, issue #195).** The execution component
that knows the final prompt and budget owns the backend iterator:
`Conductor.stream` runs pre-final DAG waves then consumes its final worker
stream, while `stream_moa` runs proposals then consumes its synthesizer stream.
Orchestrator owns route/trace assembly and multiplexes keep-alive timers only
until the first final event; after that, deltas are pulled in the caller task
without a task/queue bridge. The HTTP layer only converts typed events to
OpenAI SSE. A post-generation verifier on the final role originally made
such a DAG a rejected shape because SSE cannot retract provisional deltas;
**amended 2026-08-20 (DTO-D10, owner decision)**: a verified final unit is now
supported by *deferring* its publication — `Conductor.stream` runs the unary
verify/refine loop to completion and emits the final text once, while the
head (when enabled) still streams from t=0; unverified final units keep the
live pull-through stream. This boundary was selected by measurement,
not implementation parity: a seven-round 100,000-event A/B measured 161.14
ns/event pull-through versus 7,045.2 ns/event through a bounded queue (43.721x).

**Accounting/trace amendment (2026-07-27, issue #196).** Every AUTO terminal
response exposes cumulative backend-reported internal call totals as
`usage.orchestration_input_tokens` and
`usage.orchestration_output_tokens`; usage-enabled streams carry the same
fields on their final `choices: []` metadata chunk. Direct, Conductor, and MoA
paths include retries, verifier calls, resolved fallback engines, proposals,
and synthesis. A synchronous cumulative accounting observer reaches the
request-owned meter as each internal usage result completes, so disconnect and
partial-failure finalization includes completed pre-final work without adding
a task/queue bridge. `X-Kairyu-Trace: 1` now returns
the versioned structured route/DAG/verifier trace on streaming as well as unary
responses while retaining the legacy field/comment. Partial failures return
known usage and typed failure events, never raw exception messages, prompts,
or generated intermediate text.

**Serving hot-path amendment (2026-08-07, issue #349).** The direct AUTO
stream trusts the backend's documented cumulative, prefix-stable contract and
performs only the suffix slice needed by the existing delta event surface; it
does not rescan the prior prefix. First-token time is captured once rather
than scaling with partial count. Unset AUTO-only usage counters are omitted by
`exclude=` at the finite HTTP/SSE serialization call sites, removing the
Python `model_serializer` callback from every serialized `Usage` value while
preserving the public wire shape.

**OpenAI request-intent amendment (2026-07-28, issue #208).** L3 validates and
normalizes the wire request once into an immutable `OrchestrationRequest`
(rendered prompt, `SamplingParams`, tools/tool choice, whether the template
already rendered tools, and response format). L2 derives stage-local execution
values without mutating the shared `Orchestrator`: scalar sampling, stop,
and seed intent reach every stage; private planner/verifier/proposal stages
force `n=1`, cap work at `min(public max_tokens, internal policy)`, and omit
logprob reporting, native tools, and response grammar. Only the direct call,
selected final worker, or MoA synthesis receives the complete public
max-tokens/`n`/logprobs/tools/response-format intent. MoA proposal seeds are
deterministically derived from the request seed and proposal index. This
final-boundary policy avoids multiplying every private stage by `n` or forcing
planner text through the user's output schema.

The private max-token cap is an explicit orchestration policy (1024 by default
and visible through `/routing`), not silent loss of the public answer limit. A
Qwen3-32B TP8 A/B rejected uncapped propagation after one Conductor stage ran
76.44 s and a MoA proposal exceeded the upstream 60 s timeout, failing the
fixed quality gate with 502. The bounded policy leaves the final 8192-token
allowance unchanged and completes the same gate.

**Private-work budget amendment (2026-08-12).** Declarative YAML and the
decorator DSL expose the same validated `internal_max_tokens` policy, defaulting
to 1024 for compatibility. Operators may shorten private stages independently
of the public final-answer allowance; the cap remains generic orchestration
policy and is not inferred from a model name or deployment topology.

**Tiered-example transparency amendment (2026-08-13, EO-D1..EO-D4).** The
tiered Chat UI product borrows deployment-owned L1 pools directly and exposes
only its orchestrator model. Its explicit verifier-gated DAG may opt in to
model-attributed completed intermediate output through OpenAI
`reasoning_content`; the final publisher remains ordinary `content`, and trace
v2 remains metadata-only. Other deployments retain hidden intermediates by
default. The binding example contract is
`docs/design/example-layered-orchestration.md`.

**Assistant-history compatibility amendment (2026-08-14).** A Chat
Completions assistant message may carry typed `reasoning_content`; Kairyu
preserves the field for key-sensitive model templates and L2 conversation
history so a response can be appended to the next request. Compatibility-only
`provider_specific_fields` metadata is discarded only when null. A non-null
value and every other unsupported message field still fail before backend
dispatch.

**OpenAI output-contract amendment (2026-08-16, issue #496).** Chat
Completions follows the documented OpenAI output semantics end to end:

- An omitted `max_tokens`/`max_completion_tokens` means "bounded by the
  model's remaining context", never a server-materialized 16 (issue #458).
  The gateway passes `None` through; OpenAI-compatible upstreams omit the
  field, the native engine clamps to `max_model_len - prompt`, and tenant
  admission reserves the backend's `max_model_len` (or a documented
  constant) instead of rejecting, settled by exact usage.
- A finite caller limit is ONE budget across the public answer: the head's
  role cap is clamped to it, the selected final unit receives the remaining
  budget after the head's reported completion tokens, and a fully consumed
  budget skips the continuation (`skipped:public_budget`) and reports
  `finish_reason: "length"` for the committed head-only answer. The limit
  governs the public completion (content + tool calls); private stages stay
  governed by `internal_max_tokens` (#208).
- Standard `usage.prompt_tokens`/`completion_tokens`/`total_tokens` describe
  the caller-visible request and completion — backend truth when the public
  stages reported it, otherwise the m9 wire approximation. The cumulative
  internal-call totals keep their #196 meaning exclusively on
  `orchestration_input_tokens`/`orchestration_output_tokens`, and metering
  continues to bill the cumulative totals.
- A selected final unit that fails or produces no public text gets one
  bounded re-dispatch; if still empty the response is an explicit upstream
  error (502 / SSE error event) with usage attached — never a silent
  `finish_reason: "stop"` with empty `content`, and never an internal
  stage's text published as the answer. Budget-exhaustion best-so-far
  (EO-D7/O4) is unchanged but may only fall back to generation stages,
  never executor output or the head.
- Roles may declare `reasoning_closed` (the rendered scaffold already closed
  the model's private-reasoning span, so upstream reasoning-classified
  output with empty public text is reclaimed as the answer) and
  `prompt_headless` (the complete-answer body rendered for the selected
  final unit when the head is absent or disabled — tools, `n>1`, logprobs,
  `response_format`).
- `GET /v1/models/{model}` returns the standard Model object for every
  served id (engines, AUTO orchestrators, embeddings) and the standard
  `model_not_found` error body otherwise.
- `reasoning_content` remains a documented Kairyu extension: an operator
  opt-in (EO-D4) carrying attributed intermediate stage output. Standard
  clients never need it to recover the answer; when enabled it exposes the
  configured role/worker/engine/model attribution by design, so operators
  who must not reveal internal identifiers leave
  `expose_intermediate_outputs` off.

Typed final events retain cumulative `CompletionOutput` choices, so unary and
SSE preserve every choice index, finish/stop reason, cumulative logprob, and
token logprob exactly once. Tool-enabled streams buffer until every final
choice passes required/named choice enforcement because an invalid provisional
tool delta cannot be retracted. Prompt-only engines get one generic tool-schema
instruction when the HF template did not render tools; OpenAI-compatible
engines receive native schemas and choice, and their native `message.tool_calls`
are normalized into Kairyu's existing final parser boundary. A configured
final-role post-verifier still rejects `n>1` before dispatch because it cannot
truthfully verify every sibling; an engine that declares `supports_n=False`
does likewise. These are capability-specific 400s, replacing the former
blanket AUTO rejection.

### D2 — Tiered auto models

`create_app(orchestrators: dict[str, Orchestrator])` (back-compat shim for
the single `orchestrator=` kwarg): `kairyu-auto` (default tier) and
`kairyu-auto-max` (deep tier: bigger budget, MoA enabled) are just two
configured Orchestrator instances listed in /v1/models.

**Production proof amendment (2026-07-27, issues #198/#208).**
`OrchestratorSpec` exposes bounded `moa_samples`, so declarative YAML can select
MoA. The Qwen3-32B TP8 deployment serves direct, standard AUTO, and max AUTO
together. Twelve alternating direct/standard pairs measured standard/direct
TTFT at 1.0123x p50 and 0.7666x p99 after request-intent propagation. This
serving envelope remains a verification gate; product evaluation is separate.

### D3 — Tenancy v1

`tenancy.py`: `TenantConfig` (key→tenant map, per-tenant rate + token
budgets), `TenantLimitMiddleware` (pure-ASGI, token-bucket per tenant on
/v1/*; 429 with retry-after), usage ledger — persistent O_APPEND JSONL,
one record per execution (tenant, model, prompt/completion/cached tokens, ts),
and row-isolated aggregation that preserves valid totals around malformed
records — plus `GET /admin/usage?tenant=`. A bounded lifecycle-owned writer
moves append/flush calls to one background thread, batches up to 128 records,
and drains every accepted record at app shutdown. Aggregation enters an ordered
flush barrier and performs the subsequent file scan through
`asyncio.to_thread`, preserving read-after-write without blocking the event
loop. A barrier flushes language/runtime buffers to the OS for reader
visibility; it deliberately does not call `fsync`, so no power-loss durability
is claimed.
Request and token refill rates now have independently configurable burst
capacities. An optional tenant in-flight lease is acquired before the shared
global concurrency guard and held until the final unary/SSE body byte; failure
and disconnect release it exactly once. The same lease is acquired per Batch
API line before replica dispatch, closing the previous batch quota bypass.
After validation, a second atomic admission reserves the request's worst-case
compute ceiling before ReplicaPool/scheduler placement. The backend seam owns
the direct-request bound, including tools/response metadata and
`n`/`best_of` prefill fan-out; Orchestrator owns the maximum AUTO DAG bound.
Prompt arrays reserve their sum before any member dispatches. Exact
single-candidate terminal usage may refund surplus. Failure, disconnect,
approximate/missing usage, and multi-candidate billing consume the full
reservation, so optional ledger failure cannot make GPU work free. Bounded
admission counters and in-flight/reserved-token gauges expose the boundary.

The deterministic F5b isolation gate drives one tenant at 60 requests/minute
against a 6 requests/minute quota while a good tenant remains within quota.
The protected trace admits 120/1,200 noisy requests, rejects 1,080 before the
shared scheduler, and completes all 300 good requests with TTFT p99 of two
service seconds versus one in the control. This is the minimum one-service
quantum interference bound; without admission, good p99 is 298 seconds and
the shared queue high-watermark grows from 2 to 301. Raw admission, request,
queue, scheduling, and latency evidence is published by
`verification/fleet/resilience/noisy_neighbor_bench.py`.

The real Qwen3-32B TP8 gate uses good-only latency as a secondary lower-bound
reference, not the primary isolation comparator. Its bracketed controls carry
the same good load plus a compliant noisy neighbor at 0.9x quota; the
treatment changes only the noisy offered rate to 10x quota. A full
request-bucket wash-in precedes every arm, and treatment accepted noisy work
must remain within 10% of the controls before p99 can pass the unchanged 50 ms
non-inferiority margin. This separates the cost of rejecting excess 9x traffic
from the legitimate shared compute consumed by the admitted quota.
On Qwen3-32B TP8 the gate calibrated 7.2697 requests/s and passed all 47
checks. Good-only TTFT p99 was 0.1741 s; compliant-neighbor controls measured
0.2783/0.2792 s and the 10x treatment measured 0.2801 s. The treatment admitted
66 noisy requests and rejected 639 before shared dispatch, completed all 256
good requests, matched control accepted noisy work at 1.03125x, and drained
in-flight/reserved state to zero without reservation-bound violations. The
complete pinned evidence is
`bench/results/f5b-noisy-neighbor-qwen3-32b-tp8-2026-07-28.json`; the matched
deterministic trace is
`bench/results/f5b-noisy-neighbor-cpu-2026-07-28.json`.

The earlier isolation gate established that tenant A at its limit 429s while
tenant B proceeds; ledger
totals reconcile with returned usage to <0.1%. Dedicated Prometheus counters
mirror only accepted usage rows (not generic HTTP requests): executions and
prompt/completion/cached tokens are labeled by the bounded tenant identity. On
single-gateway restart those counters are restored from ledger totals before
serving, so ledger-versus-Prometheus reconciliation remains exact across a
clean process restart. A truncated crash tail is preserved for diagnostics and
terminated before the next complete row is appended.

Fleet reconciliation does not move shared state into the request path. Each
gateway remains the sole writer of its local append-only ledger; an offline
reconciler independently aggregates request audit logs and immutable gateway
ledgers by tenant. The fixed F5e replay covers all public endpoint families,
post-dispatch disconnect/failure, pre-result failure, batch-output rollback,
three gateways, and cached-token export. A same-host shared SQLite candidate
with no network latency and durability disabled still measured worse than the
selected local asynchronous boundary: the explicit-uncached rerun's
median-of-five producer p99 was 28.026 → 19.696 us and throughput
72,374 → 95,029 rows/s over 30,000 rows.

Pricing also remains outside the request path. New ledger rows freeze both
cached and uncached input counts; legacy rows derive uncached input as
prompt-minus-cached. A validated, versioned blended `PriceSheet` applies one
cached-input discount rule and optional tenant discounts with Decimal
arithmetic. `/admin/usage.csv` reads a size-bounded ledger snapshot and exports
`[start_ts,end_ts)` invoice rows with separated input/output quantities, unit
rates, component charges, discount, total, source SHA-256, and deterministic
invoice ID. Any malformed/truncated record fails the export closed.

### D4 — `/v1/responses` developer surface + `/v1/embeddings`

Responses: `POST /v1/responses` accepting `model`, `input` (string or
message array), `previous_response_id`, `stream`; a `ResponseStore`
(protocol + in-memory impl) persists response items so
`previous_response_id` reconstructs context; OpenAI SDK
`client.responses.create` round-trip test. Embeddings: `EmbeddingBackend`
protocol (`embed(texts) -> list[vector]`), `MockEmbeddingBackend`
(deterministic hash-based vectors), `POST /v1/embeddings` with usage. Embedding
backends are registered under explicit, non-colliding served model IDs; the
handler resolves that bounded registry before work, returns `model_not_found`
for misses, exposes every configured ID through `/v1/models`, and uses only the
resolved ID for the response, request metrics, and usage accounting.

**Production embedding amendment (2026-07-31, issue #202).** The protocol
returns ordered vectors plus tokenizer usage and whether that usage is exact.
The `fastembed` backend validates a prefetched repository revision, externally
pinned provenance-manifest SHA, every recorded file size/hash, catalog
dimension, and ONNX SHA-256 before constructing one offline CPU session.
Startup warms and validates finite, normalized output in a
dedicated bounded executor; readiness remains false until warmup completes,
startup integrity/load failures are fatal, and shutdown drains accepted work
before releasing the session. The route caps both item count and aggregate
UTF-8 bytes before dispatch, validates result count/dimensions/finiteness,
sanitizes backend failures as 502, and refunds conservative tenant reservations
only when the backend returns exact usage. The deterministic mock remains for
wire/unit tests and never claims authoritative tokenizer usage.

**Typed stream/tool amendment (2026-07-28, issue #201).** Responses requests
normalize into the existing Chat Completions validation/execution boundary so
model-specific templates, sampling, structured output, named/required tool
choice, upstream capability preflight, and usage ownership stay single-sourced.
Text streams pull backend deltas directly and emit canonical, gapless
`response.*` SSE events; tool streams buffer until parsed calls satisfy the
requested policy because invalid provisional calls cannot be retracted.
Completed, incomplete, and failed terminals are typed and only successful
stored responses are continuable.

Flat function tools and Codex namespace tools normalize into the existing chat
tool protocol. Public namespace/name/call IDs are restored on output, and
`function_call_output` becomes a linked tool-role turn. The legacy prompt-only
template also preserves prior assistant calls rather than dropping them.
`previous_response_id` storage is bounded, deep-copy isolated, and tenant
scoped for both unary and stream paths. Unknown or unsupported Responses fields
fail before dispatch instead of being silently ignored. Official OpenAI SDK
sync/async streaming tests and an unmodified Codex CLI against Qwen3-32B TP8
cover the wire contract and multi-turn tool loop; the reproducible record is
`bench/results/responses-codex-qwen3-32b-tp8-2026-07-28.json`.

**AUTO-model + Codex compatibility amendment (2026-08-18, issue #530).**
`/v1/responses` resolves the union `/v1/models` advertises: engine models keep
the existing pipeline, and orchestrated (AUTO) models delegate the normalized
`ChatCompletionRequest` to the `/v1/chat/completions` handler so validation,
judge-first tenant reservation, admission, public usage, and metering stay
single-sourced in the orchestration chat contract (the delegated call meters
exactly once; the Responses-side engine pipeline and `_record_execution` are
skipped). AUTO live text streams re-encode the orchestrator's chat-chunk SSE
(status keep-alives stay comments; `reasoning_content` is dropped, matching
the documented no-reasoning-output stance); buffered replies re-enter the
existing envelope/event builders with public usage in standard fields
(`orchestration_*` extensions remain chat-only). A field-level audit of
codex-rs 0.147.0 plus live runs added the rest of the Codex contract:
buffered streams open immediately and emit `: keep-alive` comments while
generation runs (Codex aborts silent streams at 300 s; generation failures
after the stream opens surface as `error` + `response.failed`); input items
accept echoed `reasoning` items (dropped), content-item-array
`function_call_output.output` (text parts concatenated, non-text rejected),
Codex-internal passthrough fields on any item kind, and search configuration
on the disabled `web_search` tool; `GET /v1/responses` answers 426 so the
WebSocket-first built-in `openai` provider (the Harbor/Terminal-Bench shape)
falls back to HTTPS immediately; and remote compaction v2 is served — a
terminal `compaction_trigger` item runs a tool-free summarization turn
through the same engine/orchestration path and returns one `compaction` item
whose `encrypted_content` is an AES-256-GCM sealed, tenant-bound token that
decodes back into a user bridge message, failing closed on forged, modified,
or cross-tenant tokens. A configured secret derives one deployment-stable key;
without it, a process-local key deliberately limits tokens to one gateway
lifetime. Per the #531 review, compaction is honest end to end: a successful
compaction stores only the compaction item as continuation context (the
pre-compaction history is replaced, not duplicated), a length-terminated
summary returns `response.incomplete` with no compaction item and is not
continuable via `previous_response_id`, and an empty summary fails as a 502
`compaction_failed` (`error` + `response.failed` mid-stream) — Codex installs
a compaction only on `response.completed`, so a truncated or empty summary
must never reach that terminal. Known limits are deliberate: metric phase labels of a delegated AUTO
call record under "chat", the 1024-token default output cap applies to AUTO
responses as it does to engines, tenant 429s are not retried by Codex
(bench deployments should size admission accordingly), and
`/v1/completions` stays engine-only.
Acceptance: pytest contract tests plus unmodified codex-cli 0.147.0 runs —
custom-provider text/tool smokes and a Harbor-shaped (`openai_base_url`
override) tool smoke — against a mock deployment mirroring the issue
topology (all engines hidden behind `public_models`, one orchestrated
model).

### D5 — Vision wire format

`protocol.py` accepts OpenAI content-parts (`type: text|image_url`) in chat
messages. Text-only messages retain the existing `ChatTemplate` path.
Image-bearing messages instead become a typed `MultimodalPrompt` containing
the exact role order, part order, and item references; Kairyu never flattens
them or applies a text template first. A backend must explicitly declare
multimodal prompt support and an `ImageInputPolicy`, while non-vision engines
return a clean pre-dispatch 400.

**Production VLM amendment (2026-07-31, issue #203).** The first production
adapter is `OpenAICompatBackend` against a separate stock vLLM VLM replica.
Kairyu owns admission, tenant accounting, typed transport, and fail-closed
media validation; vLLM owns the Qwen processor and chat template exactly once.
Only inline PNG, JPEG, and WebP data URLs are accepted. Remote/local URLs,
MIME/magic mismatches, malformed/truncated or animated rasters, decompression
bombs, excess bytes/pixels/dimensions/aspect ratio, and unsupported modalities
fail before dispatch. Full base64 decode and Pillow verification run outside
the event loop before token reservation or stream headers. Exact processed
usage is mandatory for unary and streaming VLM responses; Kairyu never guesses
image token counts.

### D6 — F5 CPU: priority admission + SLO shed + autoscaler logic

(a) `EngineRequest.priority` already exists — `Scheduler._admit_waiting`
orders by (priority, arrival); smaller integers win (the vLLM wire contract)
and the starvation guard improves a waiting request after `age_s`. (b) SLO
early rejection: `slo.py` `AdmissionController` — a TTFT
predictor from gateway-visible in-flight work and a running EMA of observed
TTFT per admission-time concurrency unit; over-SLO requests are shed or
deferred to lower-priority batch scheduling by the caller.
(c) `autoscale.py`: pure decision function `(metrics window) →
scale_up/down/hold + reason` with hysteresis; logged, not executed (the
executor is a deploy-day k8s HPA/keda adapter).

**Indexed waiting queue amendment (2026-07-27, issue #219).** FIFO admission
uses an `OrderedDict` so append, head pop, and cancellation by request ID are
O(1). Priority admission uses a stable sequence-numbered heap plus an ID index;
removal leaves lazily reclaimed tombstones with amortized compaction. The aging
rank is factored as `priority + arrival/age` because the omitted `-now/age` term
is common to every waiting request. This preserves stable ties and starvation
prevention without re-sorting the full queue on every schedule. Recompute
preemption retains front-of-tie placement, and KV allocation still blocks at
the selected head without skip-ahead. Reproduce the 100k A/B measurements with
`uv run python verification/l1/performance/scheduler_queue_bench.py --requests 100000 --repeats 5`.

**Prefill cohort amendment (2026-08-07, issue #328).** This supersedes only the
unbounded selected-head behavior above. Native schedulers default
`max_num_partial_prefills` to two and divide each post-decode token budget among
the effective-order leading tier of eligible prefills that share one public
priority value. The full aging rank still orders that tier and prevents it from
leapfrogging an intervening priority. Shares are normally work-conserving: a
singleton retains the full budget and unused shares return to other eligible
chunks in the same step. Deferred P-D handoff may leave one prompt token on
peer cohort members after completing one member so its asynchronous KV copy
overlaps the next engine step. Existing running and newly admitted prefills use
the same budget, and one request appears at most once in a step.

When the selected waiting head cannot preserve the decode watermark or cannot
allocate KV while work is running, ordinary admission may inspect exactly its
immediate successor. That successor passes only if a cache-aware estimate proves
its new pages fit without eviction and its prompt completes within the current
share. One successful bypass exhausts the blocked head's waiting-epoch allowance;
priority preemption remains head-only. Empty and cache-oversized heads retain
their existing rejection path, and full-prompt KV reservation is unchanged.

**Overload-priority amendment (2026-07-28, issue #190).** The signed-int64
`priority` extension now flows through Chat Completions, legacy Completions,
Responses, `GenerationRequest`, ReplicaPool/OpenAI transport, vLLM, native,
and process-split engines. Native production schedulers enable the indexed
priority policy with a configurable aging interval. A configured gateway does
not trust a client's requested value: authenticated interactive work receives
the tenant's `interactive_priority` (default 0), while Batch API work receives
`batch_priority` (default 1). Kairyu/vLLM replica transports preserve that
trusted value exactly; unsupported provider profiles reject non-neutral
priority before dispatch. Tenant profiles require the interactive integer to be
strictly smaller than the batch integer. The heap ranks the full signed-int64
domain without converting priorities to float: aging uses the exact integer
numerator `priority * age_ns + arrival_ns`. Kairyu transport also carries an
explicit bounded class hint for metrics, so custom positive or negative
priority ranges cannot be misclassified. The local vLLM adapter forces its
priority scheduler policy instead of passing a value into vLLM's default FCFS
queue where it would be ignored.

Decode remains first. Before lower-priority running prefills consume the
remaining budget, a strictly outranking waiter is admitted. If sequence slots
or KV pages block it, the worst lower-priority, output-free incomplete prefill
is released and requeued for recomputation; output-bearing decode is never a
victim. This is deliberately stronger than vLLM current main, which selects a
priority victim only after running KV allocation fails and can leave a high
priority waiter behind full active slots.

The deterministic F5a gate offers one service token of interactive work and
seven batch tokens every four ticks: 0.25x + 1.75x = exactly 2x capacity.
Against the identical FCFS trace, interactive TTFT p99 is 400 → 1 ticks while
batch consumes 300 residual service ticks, retains a 58-request overload
backlog at the measurement boundary, and drains all work afterward. Raw
request, scheduling, queue-depth, priority, latency, and accounting evidence is
in `bench/results/f5a-priority-overload-cpu-2026-07-28.json`; reproduce it with
`uv run python verification/fleet/resilience/priority_overload_bench.py --assert-gate`.

The production-shaped Qwen3-32B TP8 gate calibrated 7.6304 requests/s and then
offered 0.5x interactive plus 1.5x batch work through a one-replica gateway.
Interactive TTFT p99 rose from 0.1641 s in the control window to 1.3025 s under
the planned 2.00x overload, while all 64 requests completed and 100% met the
fixed 2 s SLO. During the mixed window, batch recorded 54 HTTP dispatches, 53
scheduler admissions, 46 scheduler completions, and one queued request; it
retained backlog and drained 192/192 afterward with zero failures. Gateway and
replica class counters reconcile all interactive and batch dispatches; replica
scheduler enqueue/admit/complete counters and queue gauges bind the same trace
to native admission. A separate untimed warmup precedes capacity calibration,
and the artifact pins a clean source commit, benchmark/config hashes, container
image digest, model revision, `/backends` topology, and GPU inventory. The raw
environment, arrivals, TTFT samples, Batch API states, counters, and assertions
are in
`bench/results/f5a-priority-overload-qwen3-32b-tp8-2026-07-28.json`; reproduce
them with `uv run python verification/fleet/resilience/priority_overload_gpu_bench.py --assert-gate`.

**SLO-admission validation amendment (2026-07-28, issue #192).**
`AdmissionController.begin()` now makes the prediction and reserves the
accepted work under one lock, returning a lease with exactly-once first-token
feedback and completion. The EMA freezes interactive concurrency at admission
instead of dividing by an unrelated later count. Interactive TTFT prediction
excludes already-deferred batch work because the priority scheduler prevents
that work from advancing ahead of a new interactive request; total active work
still caps the deferred backlog before `shed`, and deferred TTFT does not train
the interactive EMA. Non-finite or inverted thresholds, lifecycle underflow,
and duplicate feedback/closure fail loudly.

The F5c harness uses fresh production `Scheduler` instances for matched
queue-and-hope and policy arms over predeclared underload, smooth exact-2x, and
bursty exact-2x traces. `admit` enters interactive priority, `defer` enters
lower-priority batch, and `shed` never reaches the scheduler. Goodput is the
number of admitted interactive requests with measured TTFT within the fixed
two-tick SLO divided by the fixed 400-tick arrival window; deferred work and
drain time cannot inflate it. FP/FN truth comes from a separate production
Scheduler replay that freezes prior policy decisions and force-admits only the
target request, so neither predictor output nor the baseline arm labels itself.
Per-request decisions/outcomes and per-tick queue/schedule deltas make the raw
queue reconstructible. The homogeneous one-token workload isolates admission
behavior and is not evidence for a variable-prompt cost model.

The clean-commit formal run passes all 50 checks. Underload remains exactly
200 → 200 SLO-successful requests. At smooth exact-2x, queue-and-hope achieves
3 successes versus 397 with admission; the policy admits 401, defers 3, and
sheds 396. At bursty exact-2x, the result is 2 → 198, with 204 admits, 196
defers, and 400 sheds. The forced-admit oracle reports zero false positives in
both saturation profiles and 4/800 and 6/800 false negatives (FNR 0.993% and
0.997%); deferred queue high-watermarks are 3 and 2. All admitted/deferred work,
leases, and queues drain. Raw evidence is in
`bench/results/f5c-slo-admission-cpu-2026-07-28.json`; reproduce it with
`uv run python verification/fleet/resilience/slo_admission_bench.py --assert-gate`.

**Live direct-chat admission amendment (2026-08-07, issue #340).**
`server.ttft_slo_s` now opt-in instantiates one gateway-visible controller and
wires direct interactive Chat Completions after request validation but before
backend preparation. The decision adds known ingress-to-admission elapsed time
once, while feedback continues to measure only the post-admission interval.
`shed` returns 429 with `Retry-After` without dispatch. On routes explicitly
attesting `supports_slo_defer`, `defer` rebuilds the immutable request with
`scheduling_class="batch"` and the lowest signed-64-bit scheduler priority;
the one colliding interactive value is always clamped one step ahead before
preparation. That contract must guarantee running deferred decode cannot delay
later interactive work; accepting numeric priority alone is insufficient once
an output-bearing decode becomes non-preemptible. Current native, process-split,
vLLM, and remote adapters therefore convert `defer` to the same 429 instead of
claiming isolation they do not provide. The request-aware contract is rechecked
against an exact prepared ReplicaPool placement lease. The outer ASGI request
boundary owns lease release across unary, streaming, errors, and client
cancellation, while only a first direct visible SSE delta whose ASGI send
succeeds supplies TTFT feedback. Six unlabeled scrape-time gauges expose the
complete `AdmissionSnapshot`. Existing batch-class traffic, orchestrated chat,
Completions, Responses, and the bounded queue tracked by issue #341 stay out of
this policy.

**Batch-pressure amendment (2026-08-07, issue #342).** Batch API work remains
outside controller leases, TTFT feedback, and admit/defer/shed accounting, but
the in-gateway worker now consumes one read-only pressure signal. Before a
consumer starts a new line, it waits while at least one interactive lease is
active and the predicted TTFT for another interactive request exceeds the SLO;
it resumes when either condition clears. The fixed batch pool remains the bound,
and in-flight generation is never interrupted.

### D7 — Open WebUI + frontier bench

`deploy/compose/docker-compose.webui.yaml` points Open WebUI at the internal
Kairyu endpoint `http://kairyu:8000/v1` and mounts standalone CPU-safe direct
and legacy-orchestration specs that expose `default` and `kairyu-auto`.
Open WebUI v0.11.0-slim and Playwright v1.60.0-noble are pinned to immutable
linux/amd64 manifests. Normal Compose startup brings up only healthy Kairyu and
Open WebUI services; the browser runner is profile-scoped.
`scripts/webui_smoke.sh` validates literal binds and the rendered endpoint
before startup, then proves first-account creation, UI discovery and selection
of both models, SSE streaming, reload persistence, a visible Kairyu-outage
failure, and recovery after restarting Kairyu without restarting Open WebUI.
Every smoke run uses a unique Compose project, so teardown removes only its
ephemeral test database and cannot reuse or delete the normal demo's data.
CI runs this as an independent mandatory job in parallel with the existing
Compose drill. The only planned custom chat UI remains the orchestration-trace
viewer tracked by G6's non-goals; Kairyu does not fork Open WebUI.

**Kairyu-only RAG amendment (2026-07-31, issue #202).** The Kairyu image used
by this topology opts into the production FastEmbed dependency and embeds an
immutable all-MiniLM-L6-v2 ONNX snapshot; the default image remains lean.
Open WebUI sends bounded embedding batches and chat requests only to Kairyu,
with query generation, full-context bypass, and reranking disabled. The
browser gate uploads a unique document, proves the query does not contain its
canary, retrieves that canary through `embed-small`, and obtains a
citation-bearing mock-chat response whose configured trigger can only be
present after retrieval. It repeats retrieval and answer after restarting only
Kairyu, while the direct contract gate also fixes two-input ordering, 384
dimensions, finite L2-normalized vectors, and exact tokenizer usage. Optional
reranking is explicitly deferred and is not a substitute for this retrieval
proof.

**Real image-chat amendment (2026-07-31, issue #203).** The GPU-only
`docker-compose.webui-vlm.yaml` overlay replaces the mock chat pool with a
revision-pinned Qwen3-VL-32B-Instruct stock-vLLM replica at TP8 while retaining
the production embedding/RAG endpoint. The vLLM image and model revision are
immutable, the Qwen image processor is bounded to 65,536–2,097,152 pixels, and
the deployment accepts one inline image with an 8,192-token complete prompt
reservation ceiling. `scripts/webui_vlm_smoke.sh` proves RED/BLUE semantic
separation, exact unary/stream usage, remote-URL rejection, and a normal
Open WebUI owned-file upload that becomes an inline data URL only inside the
Open WebUI backend before reaching Kairyu. This GPU gate remains separate from
CPU-only GitHub Actions. The clean `b8971cb` gate passed every binding check on
8× RTX PRO 6000; the retained result is
`bench/results/issue-203-vlm-image-chat-qwen3-vl-32b-tp8-2026-07-31.json`.

`verification/product/performance/frontier_compare.py`: multi-target harness (kairyu vs OpenAI vs
Anthropic endpoints), method block (same prompts, N trials, TTFT/TPOT/
quality-proxy), scoreboard JSON+md; offline unit test with mock targets.

### D8 — System One (Jev wire API) surface (amendment, 2026-10-01)

Status: accepted by the owner (2026-10-01); CPU tests in
`tests/server/test_systemone_api.py`; GPU-verified with OpenJev in
`examples/openjev-diffusiongemma-26b-1gpu` (`systemone-live`,
`systemone-serving`, `systemone-isolation`).

Kairyu serves `POST /v1/systemone`, the System One API that TypeSafe's Jev,
OpenJev and Codiv share: a `state` and typed `questions` in, per-question
probabilities, `usage` and a `Server-Timing` header out. It is a front door
like `/v1/messages`: Kairyu forwards to one configured upstream per model and
does not reimplement reads.

- **Config.** `systemone: {name: {base_url, upstream_model, aliases,
  max_concurrency, max_queue, queue_wait_s, timeout_s, max_questions,
  max_body_bytes}}`. Names and aliases are unique across every served model.
- **Not a pool member.** `HTTPSystemOneBackend` owns admission (a bounded
  in-flight count and queue) and answers 429 before the upstream's own
  overload answer (529). ReplicaPool is typed to generation requests and
  counts a 529 as a replica failure; a System One read must not eject a chat
  replica that shares the server. The route is exempt from the server-wide
  chat `max_concurrency` gate for the same reason.
- **Upstream owns the schema.** Kairyu checks only what routing and resource
  bounds need (a JSON object, a string `model`, `max_questions`, and the
  selected model's own `max_body_bytes`; the middleware bounds the largest). Upstream answers and errors pass through with `retry-after` and
  `Server-Timing`. Kairyu's own errors use Jev's shapes: a FastAPI-style
  `{"detail": [...]}` list for a malformed body, otherwise
  `{"detail": {"error_type", "message"}}`, including 401/413/429 from the
  middleware and the tenant limiter.
- **Metering.** Before dispatch a tenant reserves an upper bound on what the
  request can bill: every question may be its own read carrying the state,
  and with `think` each read writes a thought and then reads prompt +
  thought once per sample, and a `sequential` read repeats every question and
  the answers so far in each group's prompt (unbilled server re-reads are not
  reserved). `samples`, `think` and `steps` are normalized to ints the way
  the upstream parses them ("32", 32.0) before reserving and forwarding;
  other representations get a 422. The
  reservation is refunded unless the upstream answers 200; then
  `usage.input_tokens` and `usage.output_tokens` are recorded. A 200 whose
  two counts are not both valid becomes a 502, so no answer leaves unbilled
  (review amendment, PR #614).
- **Discovery.** With System One models served, `/v1/models` also returns
  Jev's `models: [{name, description}]` list (System One names and aliases,
  then the other served models). System One models stay out of OpenAI's
  `data` list, which chat clients offer as chat models.

Why: the System One API is a public wire format with several independent
servers and SDKs, so serving it through Kairyu's auth, tenancy, metering and
admission is a shared contract, not one example's workflow. Examples own the
model choice, aliases, limits and any UI.

**Replica amendment (2026-10-01).** `systemone.<name>.base_urls` (exclusive
with `base_url`) serves one model from several upstream replicas. A read goes
to the replica with the fewest reads in flight; a transport error or a 429,
502, 503, 504 or 529 moves it to another replica once (reads are side-effect
free). When every tried replica is unreachable the backend raises
`SystemOneUnavailableError` (the public route answers 503; a checklist
verifier reports `judge_unavailable`, m1 D8). Admission stays one bounded
queue per model. Why: a System One model deployed for availability or load
has more than one server, and a checklist verifier's guarantee must not
depend on one GPU. Tests: `tests/server/test_systemone_api.py`.

## 3. Non-goals

- A native in-process Kairyu VLM runner, tenant-controlled remote image
  fetching, online bandit for tiers, and billing/invoicing.
- Cross-node tenant state (single-gateway token buckets; the distributed
  limiter is a G6 note).
- Autoscaler EXECUTION (decision logic only).

## 4. Verification

- Streaming orchestrator: SSE event sequence (status* delta+ result) with
  usage totals == sum of stage usages; trace opt-in only with the header.
- AUTO accounting matrix: direct/Conductor/MoA × unary/stream reconciles
  `orchestration_input/output_tokens` to recorded internal calls; retry,
  fallback, partial-failure, cancellation, and privacy cases are fixed tests.
- AUTO request parity: direct, Conductor, and MoA unary/stream preserve scalar
  sampling, deterministic `n`, choice-scoped logprobs, tools/tool choice, and
  final-boundary JSON schema intent; concurrent requests do not share mutable
  intent. Qwen3-32B TP8 evidence covers schema-valid `n=2` with logprobs,
  required named tools, and a healthy post-request plain generation.
- Tiered AUTO production gate: both named tiers appear beside Qwen3-32B in
  `/v1/models`; paired direct-route TTFT stays <=1.5x and the fixed,
  sandbox-scored multi-agent quality slice is a strict auto-max win with call,
  token, latency, and allocated-GPU cost evidence.
- Real direct-route gate: 24 alternating-order Qwen3-32B TP8 pairs measured
  AUTO/direct TTFT ratios of 1.0096x p50 and 1.0122x p99 (both <= 1.5);
  raw samples and method are committed in
  `bench/results/orchestration-stream-qwen3-32b-tp8-2026-07-27.json`.
- Tenancy isolation + ledger reconciliation gates (D3).
- OpenAI SDK round-trips: responses.create (+previous_response_id chain),
  embeddings.create.
- Vision: ordered content-parts reach a pinned stock-vLLM Qwen3-VL replica;
  real images change output, exact usage is retained, remote/local references
  fail closed, and non-vision engines still reject image input cleanly.
- F5: priority ordering + aging; SLO shed under synthetic overload
  (deterministic fake clock); autoscaler hysteresis table test.
- Bench: mock-target run produces the scoreboard schema.

## 5. Review record (binding amendments)

- **A1 (D1)**: usage is currently DROPPED at three layers — thread it: usage
  fields on ConductorResult/OrchestratorResult/MoAResult, accumulate in
  Conductor._generate; the stream contract is "usage read from the LAST
  partial" (MockBackend final-only; KairyuBackend every-partial — both fit).
- **A2 (D1, CRITICAL)**: status keep-alives must NOT be data: lines (the
  OpenAI SDK parses every data: payload as a chunk) — SSE comment lines;
  trace rides an explicit optional ``kairyu_trace`` field (no extra=allow).
- **A3 (D1)**: only the AUTO call-site usage=None fallback is removed;
  _wire_usage's approximation branch stays for third-party backends.
- **A4 (D1/D2)**: MoA is currently unreachable from Orchestrator.run() —
  wire a ``moa`` route (tier option); run_chat receives the PRE-RENDERED
  prompt string (app.py renders; orchestrator engines have no template
  knowledge) plus messages only for future vision routing.
- **A5 (D1, amended by Issue #195; amended 2026-08-20 by DTO-D10)**: every
  supported direct, Conductor, and MoA route streams the final backend
  iterator live when the final role is unverified. A final role with its own
  post-generation verifier is published deferred: nothing provisional is ever
  streamed, the verify/refine loop completes first, and the accepted (or
  refinement-exhausted last) text is emitted once after the committed head.
  Pre-final verifier failures/refinement remain supported.
- **A6 (D3)**: AuthMiddleware stores the matched key hash in scope state;
  TenantLimitMiddleware runs INSIDE auth (added before it) so 401 wins over
  429 and unauthenticated requests never drain buckets; keyless mode →
  tenant "default".
- **A7 (D3, amended by Issues #90, #191, and #213)**: ledger = O_APPEND single-writer JSONL
  (atomic-rename doesn't fit appends); writes happen in handlers, stream
  generators, and successful batch consumers (middleware can't see usage).
  `record` performs only bounded `put_nowait` admission (4,096 records by
  default); one writer thread owns the append handle and flushes each batch of
  at most 128 records. Queue saturation raises `AuditQueueFull` immediately
  rather than blocking a request thread or silently dropping billing data.
  Append/flush/close failures become sticky `AuditWriteError`s and surface on
  the next admission or ordered barrier; no failed write is reported durable.
  Tenant work is reserved before dispatch and settled before ledger admission,
  so this failure path cannot bypass quota accounting. Only authoritative
  terminal usage can refund a single-candidate reservation; unknown work
  retains the conservative debit.
  `create_app` closes the writer in an outer lifespan after caller/builder
  cleanup finishes or raises, and normal close drains all accepted records
  before closing the handle. `/admin/usage` waits for the ordered flush barrier
  on a worker thread before scanning, so it observes every prior accepted row
  while other requests continue. Flush means runtime buffers reach the OS and
  readers; it is not an `fsync` or a power-loss guarantee. Aggregation validates
  each non-whitespace record independently: a malformed final line without a
  newline is a truncated tail (skip + warning), complete malformed lines are
  corruption (skip + per-line error), and all valid rows still count.
- **A13 (D3, Issue #199)**: generic HTTP request counters cannot reconcile
  usage because they intentionally include rejected/unknown requests.
  `kairyu_usage_requests_total{tenant}` and
  `kairyu_usage_tokens_total{tenant,type}` therefore increment at the same
  exactly-once metering seam as ledger admission across sync/stream chat,
  completions, AUTO, Responses, embeddings, and batch lines. Startup restores
  their tenant totals from the app-owned ledger; model is deliberately not a
  label because the aggregate ledger API has no model dimension and tenant
  reconciliation must survive restart without inventing attribution.
- **A14 (D3, Issue #194)**: gateway ledgers remain independently owned and
  fleet aggregation is an offline boundary. Cached prompt tokens flow from
  every generation path (including Conductor/MoA and batch) into ledger,
  `/admin/usage`, and `kairyu_usage_tokens_total{type="cached"}`. The committed
  fixed replay reconciles independent request logs and three gateway ledgers
  with 0.0% maximum error; the committed A/B selects local async writes by
  measured request-path p99 and throughput against a favorable same-host
  synchronous shared-store candidate.
- **A15 (D3, Issue #204)**: every new usage row stores
  `cached_tokens + uncached_tokens == prompt_tokens`; old rows derive the
  missing uncached count. DeploymentSpec validates a versioned blended price
  sheet, cached-input and tenant discount fractions, known tenants, and the
  required ledger. Invoice export uses Decimal half-even rounding at six
  currency decimal places and a reader-stable ledger snapshot; streaming,
  Responses, embeddings, and batch executions reconcile into the CSV while
  corrupt records produce no partial invoice.
- **A8 (D4)**: Responses usage names are input_tokens/output_tokens/
  total_tokens; output item = {type: message, role: assistant, status,
  content: [{type: output_text, text, annotations: []}]}; instructions
  supported; STREAM DESCOPED (typed response.* event protocol is its own
  milestone — recorded).
- **A16 (D4, Issue #201)**: supersedes A8's stream descoping. Text streaming
  uses direct pull-through with canonical typed events and gapless sequence
  numbers; tool streams buffer for final policy validation. Function and
  namespace tools, linked outputs, tenant-scoped continuation, completed/
  incomplete/failed terminals, sync/async official SDK parsing, and an
  unmodified Codex Responses client are binding acceptance coverage.
- **A9 (D4)**: /v1/embeddings must support encoding_format=base64 (the SDK
  DEFAULT); dev dep bumped to openai>=1.66 (client.responses exists from
  1.66).
- **A10 (D5)**: content-parts touch ChatMessage.content typing, render_chat,
  _normalize_message flattening, and the shared render_prompt (batch worker
  parity).
- **A11 (D6)**: injectable clock + arrival timestamps; effective priority is
  represented by an immutable aging rank (EngineRequest frozen). Issue #328
  supersedes unbounded selected-head blocking with one cache-safe,
  completion-only immediate-successor bypass per waiting epoch, while the
  strict priority-preemption path remains head-only. Issue #190 completed
  the formerly descoped HTTP→tenant class→replica→engine path and aligned the
  public numeric direction with vLLM (smaller wins);
  TTFT predictor uses GATEWAY-observable signals (in-flight count + observed
  TTFT EMA — engine internals invisible through ZMQ/vLLM backends).
- **A12 (D4)**: embeddings use a model-ID → backend registry, not one anonymous
  global backend. IDs are discoverable, resolve before validation/execution,
  cannot collide with chat or orchestration IDs, and are the only identities
  admitted to response, metric, and ledger labels; limiter charging occurs
  only after resolution. Misses use the shared 404 `model_not_found` response
  and record no usage.
- **A17 (D4/D7, Issue #202)**: one production backend must load a pinned local
  model without runtime network access, participate in lifecycle/health, return
  validated ordered vectors and exact tokenizer usage, and bound CPU
  concurrency. The mandatory Compose gate must use that real backend to ingest
  and retrieve a document through Open WebUI, prove retrieved context reaches a
  Kairyu chat model, and repeat after a Kairyu-only restart. Optional reranking
  may remain disabled but cannot replace the retrieval gate.
- **A18 (D5/D7, Issue #203)**: image-bearing chat must preserve roles and
  content-part order without double-templating. The production boundary accepts
  only decoded and locally verified inline rasters under explicit byte, pixel,
  dimension, aspect-ratio, count, and complete-context bounds. Media work must
  finish off the event loop before quota reservation or SSE headers. The remote
  VLM must return exact processor usage; failure without usage consumes the
  reserved upper bound rather than guessing or refunding. The GPU gate must use
  the normal Open WebUI owned-file upload path and prove that two different
  images produce different correct model answers. This acceptance item does not
  claim native multimodal tool-continuation support.
- **A19 (D4, Issue #482; supersedes the #480 assistant-history amendment)**:
  LiteLLM assistant history may carry object-valued
  `provider_specific_fields` and nullable legacy `function_call` metadata.
  Kairyu accepts those shapes only on assistant messages and drops them at the
  shared preparation boundary before rendering or orchestration. Nullable
  provider metadata remains accepted on every role for backward compatibility;
  non-object provider values, provider objects on other roles, non-null legacy
  calls, and unrelated extras remain fail-closed before dispatch. These
  compatibility fields stay absent from the typed `ChatMessage` schema.

# Change Log archive

Older `PROGRESS.md` Change Log entries, moved here verbatim by the archiving
procedure in `.claude/rules/progress-log.md`. Ordering is newest-first across
the whole file. Entries are frozen history: NEVER rewrite or delete them —
corrections are appended as new entries in `PROGRESS.md` referencing the
original.

When trimming `PROGRESS.md`, insert the trimmed entries directly below this
header (above the existing entries), keeping their original order.

<!-- ARCHIVE-INSERT-POINT: new trimmed entries go directly below this line -->

### 2026-09-07 — [progress] PostgreSQL AsyncRequest persistence
- What: added the production RequestStore schema and backend with tenant-scoped
  idempotency, SKIP LOCKED claims, DB-clock renewal/takeover, fenced terminal
  writes, deadline/cancel invalidation, lock-complete status lists, and
  transactional claim audit events.
- Refs: m10 A37; kairyu/async_requests/postgres_store.py;
  tests/unit/test_postgres_request_store.py; scripts/postgres_integration.sh

### 2026-09-07 — [design] AsyncRequest v1 store and fencing contract
- What: added a separate online async-request state model, tenant-scoped
  idempotency, deterministic priority claims, lease renewal/reclaim, fencing,
  cancellation/deadline invalidation, and a process-local reference store.
- Why: a stable backend-neutral contract is required before PostgreSQL and API
  wiring, and Batch job/file semantics must not become the online queue model.
- Refs: m10 A36; kairyu/async_requests; tests/unit/test_async_request_store.py

### 2026-09-04 — [amendment] FN-D9: vision examples GPU-verified; Qwen drops MTP k=3
- What: both vision replica examples pinned (tree SHA, image ID `b47e2210`) and all gates
  PASS — DeepSeek c64 689 tok/s, Qwen c32 548 tok/s, placement 32/32 at every row ≥c8,
  tool-calling 6/6, vision 2/2. `qwen3.8-flash-next-dp2-8gpu` now serves without the
  recipe's `--speculative-config mtp k=3` and with `--kv-cache-memory` pinned (a cold
  torch.compile cache made vLLM's start-up profile shrink replica 0's KV cache to 741K
  tokens vs 3.45M); `verify.sh vision` requires the answer to name the probe colour.
- Why: with prefix caching + MTP, `vllm@27a94d1c` corrupts batched answers on the hybrid
  GDN checkpoint (`ductduct…`; 13/274 at 2-12 concurrent, 0/1,508 with either off,
  63.8% with `--no-async-scheduling`; upstream vllm#53912). Prefix caching is what
  Kairyu's prefix-aware placement and multi-turn traffic use, so MTP is the one dropped.
- Refs: FN-D9 amendment in `docs/design/frontier-native-runtime.md`; `examples/*/MEASUREMENTS.md`; supersedes the "MTP k=3" wording in the 2026-09-04 FN-D9 entry below

### 2026-09-04 — [amendment] FN-D9: two vision replica-pool examples
- What: `examples/deepseek-v4-flash-vision-exp-dp2-8gpu` (TP4+EP4 × 2, official recipe
  + SM120 marlin, 1M ctx) and `examples/qwen3.8-flash-next-dp2-8gpu` (TP4 × 2, official
  rtx_pro_6000_4x FP8 layout, MTP k=3, 256K ctx): one public text+image model each,
  no-login Chat UI with a reasoning-effort dropdown in each model's official vocabulary,
  `verify.sh vision` gate, shared upstream-main `27a94d1c` + FlashInfer `60b49158`
  SM120 overlay image with a fail-closed image-ID pin.
- Why: both checkpoints need upstream `main` (official tags predate the support PRs;
  FlashInfer 0.6.18 breaks SM120 sparse-MLA on the first image); Qwen's template
  rejects L3-normalized efforts, so an example-local alias restores them.
- Refs: FN-D9 amendment 2026-09-04 (docs/design/frontier-native-runtime.md); tests/unit/test_replica_examplectl.py; GPU evidence pending

### 2026-09-02 — [amendment] FN-D9: replica examples must serve OpenAI tool calls
- What: DP2 DeepSeek drops the Kairyu-rendered /completions passthrough (forwards
  no tools; DSML parse is whole-block only) for the Qwen-style legacy path: vLLM
  renders with the checkpoint's deepseek_v4 encoder + `--tool-call-parser
  deepseek_v4`, Kairyu forwards tools to /chat/completions and normalizes. Both
  examples gain a fail-closed readiness tool probe + `verify.sh tool-calling`
  and non-thinking default kwargs (Qwen gate caught empty `content` on plain chat).
- Why: PR #584 review — SWE-bench Pro got `tool_calls: null` every turn (22/22
  RepeatedFormatError). GPU-verified: both tool gates 6/6, both matrices
  re-pinned, SWE-bench Pro smoke 3/3 (kairyu-bench `20260902T010540Z-3bf671e8`).
- Refs: PR #584; FN-D9 amendment; examples/{qwen3.8-27b-dp8-8gpu,deepseek-v4-flash-0731-dp2-8gpu}/

### 2026-09-02 — [amendment] FN-D9: replica placement gates reject material skew
- What: both replica-pool examples now limit a replica to 1.25× the even share.
  Behavior tests reject the 8-way `16,16,16,8,2,2,2,2` and 2-way `63,1`
  skews while the retained exact-even distributions pass. The verification-only
  config change does not require a GPU rerun.
- Why: the former 2× bound admitted materially skewed distributions as passing.
- Refs: PR #585; FN-D9; examples/{qwen3.8-27b-dp8-8gpu,deepseek-v4-flash-0731-dp2-8gpu}/

### 2026-09-01 — [progress] Production-ready split-role Helm controls
- What: chart 0.2.0 can pin repository@sha256, label gateway/replica roles,
  roll on DeploymentSpec changes, wait through model startup, drain before
  termination, disable service-account tokens, apply baseline Pod hardening,
  select Recreate for scarce GPUs, and emit an optional ServiceMonitor.
- Refs: deploy/helm/kairyu; tests/unit/test_fleet_elastic.py

### 2026-09-01 — [amendment] FN-D9: two replica-pool 8-GPU examples (no orchestration)
- What: `examples/qwen3.8-27b-dp8-8gpu` (Qwen3.8 TP1 × 8) and
  `examples/deepseek-v4-flash-0731-dp2-8gpu` (DeepSeek TP4+EP4 × 2) expose one
  public model each; L2 is only the `ReplicaPool` (`prefix_index: true`,
  `queue_depth_threshold: 0`) and `verify.sh serving` gates the per-replica
  split from `placement_log_path`. Same run/verify UX; no product code changed.
- Why: a plain scale-out serving path (one API over N identical L1 replicas)
  next to the orchestrated tiered example. GPU-verified 2026-09-01: gates green,
  exact 8x8 / 32x2 splits; Qwen 313.7 tok/s at c8 (8.0x c1), DeepSeek 471 tok/s
  at c32 (1.95x one replica) — MEASUREMENTS.md runs 20260901T133331Z / 20260901T140112Z.
- Refs: FN-D9 (docs/design/frontier-native-runtime.md); tests/unit/test_replica_examplectl.py

### 2026-08-26 — [design] DTO-D15: public-output floor for chat-template final units
- What: role-level `reasoning_continuation: chat` + `reasoning_open_tag`; the
  empty-output re-dispatch of a final unit whose span the upstream chat
  template opens continues the captured reasoning as a closed assistant turn
  (`GenerationRequest.assistant_prefill` → vLLM `continue_final_message`) with
  the reserved floor tokens. `qwen_think_answer` adopts it (floor 256 now
  covers the `qwen_think_medium` route). Served config changed — GPU
  re-verify and digest re-pin pending.
- Why: 502s on `max_tokens: 8192` agent turns — medium-tier Qwen spent the
  whole cap in `<think>` and the byte-identical retry failed the same way.
- Refs: DTO-D15; kairyu/{dsl,engine,orchestration}; examples/qwen3.8-deepseek-v4-8gpu/auto-max.yaml

### 2026-08-26 — [progress] Streamed DTO-D9 floor retry no longer 502s (delta-offset reclaim fix)
- What: `_unit_public_output` reclaim returns cumulative-form completions
  (`text_delta`/`text_offset` cleared); a reasoning-only delta stream left
  `text_offset=0` behind while the reclaimed text was already published, so
  `/v1/messages` (always streamed) failed `delta offset mismatch` → 502 after
  the full generation. Server regression test added; floor fixture made
  delta-native.
- Also: the OpenAI backend now reads the upstream span under `reasoning` as
  well as `reasoning_content` (vLLM v0.23 Qwen image); the Qwen pool's
  reasoning had been silently dropped.
- Refs: DTO-D9/DTO-D15; kairyu/{orchestration/conductor.py,engine/openai_backend.py}; tests/server/test_openai_api.py

### 2026-08-25 — [progress] DTO-D14 GPU-verified; three defects fixed en route
- What: both verify.sh serving gates green on the DTO-D14 config (runs
  20260825T161729Z coding / 20260825T173343Z generic, digest 69702ab5…).
  The runs surfaced and fixed: trace v2 rejecting list detail values (judge
  offered labels) crashing streamed SSE; trace envelope anchored at judge
  started_at instead of queued_at (1 ms outside-envelope races); the Qwen
  medium preamble missing the think-close guard sentence, letting long
  L2-wrapped contexts end inside the think span (EmptyFinalOutput).
- Refs: PR #579; DTO-D14 (amends the 2026-08-25 [design] entry below);
  kairyu/{entrypoints/server/protocol.py,orchestration/{trace,orchestrator}.py}

### 2026-08-25 — [design] DTO-D14: Qwen medium tier; audit moves to Qwen (tiered example)
- What: Qwen thinking roles (draft, image_description, answer_1..4, renamed
  qwen_think_medium route) fixed at spec `high` = medium tier (L3
  medium→high alias); example-local graded Qwen template replaces the shared
  clamped one; Qwen budgets doubled (2048/4096); audit moved to tier1 fixed
  medium, one 16384 cap, REQUEST-first Qwen prompt without scaffold.
- Why: owner request to raise Qwen deliberation to medium and audit on Qwen;
  sampling stays DTO-D8; core effort ladder untouched. Served config changed
  — GPU re-verify and digest re-pin pending.
- Refs: DTO-D14; examples/qwen3.8-deepseek-v4-8gpu/*; tests/unit/test_tiered_frontier_examplectl.py
### 2026-08-25 — [progress] Incremental Anthropic tool streaming + count_tokens (#573)
- What: `/v1/messages` streams tool calls incrementally — per-protocol
  scanners (GENERIC/LLAMA/QWEN/DSML, commit-on-close, hold-back) shared by
  stream and unary so both reconstruct one parse; text+tool_use now coexist
  (Anthropic contract); tool gates re-homed stream-side (SSE `error`, no
  `message_stop`); AUTO uses an in-process raw-stream sentinel (public chat
  contract unchanged). `count_tokens` implemented: billing-consistent tiers
  (native tokenizer exact / vLLM `/tokenize` / word-split fallback; AUTO =
  multi-stage L2 word-split — direct-route billing dichotomy documented).
- Refs: #573; kairyu/entrypoints/server/{tool_stream,messages_service,
  middleware,messages_protocol}.py; kairyu/engine/*; tests/server/test_messages_api.py

### 2026-08-25 — [progress] Anthropic Messages endpoint for Claude Code (L3, #508)
- What: `POST /v1/messages` (+`?beta=true`) as an L3 adapter over the same
  validated chat/orchestration path as `/v1/chat/completions` (AUTO via
  `chat_dispatch`): text/tool_use/tool_result subset, live text SSE, Anthropic
  error envelope on the route (incl. middleware 401/413/429), `x-api-key` auth
  fallback, `HEAD /api/hello`; `reasoning_content` never leaks into text blocks.
  Executable-tool streams temporarily buffer inference and emit ping keep-alives
  before replaying canonical blocks; this preserves Claude Code compatibility but
  is a known exception to the gateway's incremental-streaming requirement (#573).
  count_tokens is an Anthropic-shaped 404 (client falls back); stop_sequence
  attribution is unavailable (`end_turn`).
- Refs: #508; kairyu/entrypoints/server/{messages_protocol,messages_service,
  middleware,tenancy,metrics,sse_encode,extra_routes}.py; tests/server/test_messages_api.py

### 2026-08-22 — [amendment] Non-thinking L2 calls declare enable_thinking=false (DTO-D13 live fix)
- What: live check showed the route judge's verdict arriving empty on the
  deployed vLLM v0.23 Qwen service (Qwen3 parser drops non-streamed output
  without an explicit `enable_thinking=false`), so every request fell back
  to the ensemble. Judge requests and effort-less roles on workers that
  accept `enable_thinking` now send it; `GenerationRequest` allows template
  kwargs on text chat prompts (still not on pre-rendered/token prompts).
- Why: vendor/vLLM contract for Qwen3 non-thinking calls; amends the entry below.
- Refs: DTO-D13 amendment; kairyu/{engine/backend.py,orchestration/conductor.py,orchestration/orchestrator.py}

### 2026-08-22 — [design] Qwen-judged five-route selection for the tiered example (DTO-D13)
- What: L2 DSL `general_roles`/`CODE|GENERAL` judge generalized to N named
  `profiles` + `profile_judge {choices: [{profile,label,criteria}], fallback}`
  (verdict attached pre-admission; image requests offered only image-capable
  profiles; head-disable short-circuit and keyword fallback removed); the
  final unit may declare sampling (style = policy, max_tokens = cap); DSL
  token bounds → 393216. Example: Qwen non-thinking judge → qwen_direct /
  qwen_think_low / deepseek_direct (`tier2-direct`) / deepseek_think /
  ensemble; route-aware verify gates. Ensemble + Chat UI cap unchanged.
- Why: owner decision 2026-08-22 — fastest route that still answers well.
- Refs: DTO-D13; docs/design/example-dual-track-orchestration.md; examples/qwen3.8-deepseek-v4-8gpu/

### 2026-08-20 — [design] Peer synthesis + audit loop, image_description, halved DeepSeek budgets (DTO-D10..D12)
- What: tiered example `compose` → `synthesis` weighing critique + 4 policy
  answers as peers; new thinking-DeepSeek `audit` verifier on the final unit
  (PASS/FAIL, `max_refine_depth: 2`, remainder published after the verdict);
  Qwen `image_description` (`requires: image`) feeds the DeepSeek roles on
  image requests only; DeepSeek tiers 16384/65536/131072 → 8192/32768/65536,
  ceiling/Chat UI/harness cap 131072 → 65536; budget `{19, 2}`; 11 roles.
- Why: owner decisions (2026-08-20); Terminal-Bench turns timing out against
  the 900 s agent budget on the serial DeepSeek chain. GPU gates not re-run.
- Refs: DTO-D10..D12; examples/qwen3.8-deepseek-v4-8gpu/; tests/unit/test_tiered_frontier_examplectl.py

### 2026-08-20 — [design] Verified final unit streams deferred; image-conditional roles
- What: a verifier on the final unit is accepted in `Conductor.stream`: the
  unary verify/refine loop runs first and the text is published once after
  the committed head (n>1 skips the verifier, `skipped:intent`); the DTO-D9
  floor retry runs before the verdict and refinements reopen the scaffold.
  Roles gain `requires: image`: skipped entirely (`skipped:condition`) on
  text requests, dependents render the slot as ""; head/final/verifier/
  executor cannot be conditional; the worker must accept images.
- Why: owner decision for the tiered example (synthesis audit loop, image
  description stage — DTO-D10/D11); amends m11 A5 and DTO-D9 (#547).
- Refs: kairyu/{orchestration,dsl,deploy}; docs/design/m11-product.md

### 2026-08-20 — [design] Public-output floor for the thinking final unit (DTO-D9)
- What: spec-level `public_output_floor` + role `reasoning_close_tag`: the
  final unit's attempt 0 thinks within budget−floor, and the existing bounded
  empty-output re-dispatch continues the captured reasoning after a forced
  close with the reserve, reclaimed as public. Example: floor 256 on compose.
  Fixes #546–#549: direct-API bounds; verified-final/MoA rejected; n>1 no floor.
- Why: issue #542 — a small caller max_tokens was eaten whole by thinking
  and 502'd after full DAG cost; owner chose option 1. GPU gates not re-run
  (floor is 0.2% of the recorded 131072-cap runs).
- Refs: DTO-D9; issue #542; kairyu/{dsl,orchestration}; examples/qwen3.8-deepseek-v4-8gpu/

### 2026-08-20 — [progress] Chat UI reasoning-effort dropdown provisioned by the launcher
- What: the tiered example's `up` installs and self-verifies a globally
  active Open WebUI filter whose enum user valve renders Reasoning Effort as
  a Chat Controls dropdown (default/low/high/max) forwarded as
  `reasoning_effort`; re-up refreshes content without flipping activation off.
- Why: owner — the DTO-D6 Chat UI knob must be selectable, and the pinned
  Open WebUI v0.11.0 (and upstream main) only offers a free-text field.
- Refs: DTO-D6; examples/qwen3.8-deepseek-v4-8gpu/{control.py,
  webui-reasoning-effort-filter.py,README.md}

### 2026-08-20 — [amendment] Chat effort aliases match the pinned UI
- What: /v1/chat/completions now accepts the same OpenAI-style aliases as
  Responses and normalizes minimal/low→low, medium/high→high, xhigh/max→max.
  This corrects the native-level-only Chat contract in the DTO-D8 entry below.
- Why: pinned Open WebUI v0.11.0 initializes Reasoning Effort to medium;
  the prior strict low|high|max schema rejected its default request with 422.
- Refs: DTO-D8 (revised); PR #536 review; kairyu/entrypoints/server/protocol.py

### 2026-08-20 — [amendment] DTO-D8 budgets raised to the vendor starting values
- What: DeepSeek effort tiers 4096/16384/32768 → 16384/65536/131072
  (fallback = high tier), internal_max_tokens → 131072, DSL token bounds
  32768 → 131072, Chat UI default and harness public max_tokens → 131072.
  Corrects the entry below; admission upper bound now ~32× the pre-DTO-D8
  ceiling.
- Why: owner — the TTFT gate is head-based with a paired same-concurrency
  DeepSeek-direct denominator and E2E is unconstrained by design, so the
  ≤2×-DeepSeek budget never constrained DeepSeek role budgets; use the
  vendor-recommended starting values outright.
- Refs: DTO-D8 (revised); examples/qwen3.8-deepseek-v4-8gpu/; kairyu/dsl/spec.py

### 2026-08-20 — [design] Vendor-official sampling + effort-graded token budgets (DTO-D8)
- What: RoleSamplingSpec gains top_k/min_p/presence/repetition penalties and
  `max_tokens_by_effort` {low,high,max} (inherit roles; think+answer budget,
  min()'d vs public cap/internal ceiling). auto-max: official Qwen sampling,
  DeepSeek T→1.0 + caps 4096/16384/32768 per L3 effort, internal_max_tokens
  →32768; /v1/responses maps reasoning.effort (minimal..xhigh normalized).
  GPU gates not re-run — re-verify next window.
- Why: owner + vendor research — effort scales the token budget while
  sampling stays effort-invariant; T=1.0 stays anti-greedy (#509 rationale).
- Refs: DTO-D8; PR #536; kairyu/{dsl,orchestration,entrypoints/server};
  examples/qwen3.8-deepseek-v4-8gpu/

### 2026-08-19 — [design] L2 effort policy: inherit/default effort, all-thinking DeepSeek (DTO-D6/D7)
- What: role specs gain `reasoning_effort` (fixed low|high|max, or `inherit`
  = the caller's L3 effort) and orchestrator specs `default_reasoning_effort`.
  auto-max: Qwen draft/answers think at fixed low (T=1.0), head non-thinking
  (T=0.7); every DeepSeek role runs thinking on tier2 (tier2-direct removed,
  policies cap 1024→4096), inherits effort (default high), and the new
  passthrough template splices the high/max preamble; Qwen clamps to low.
- Why: owner — DeepSeek v4 flash was always meant to think; one L3 effort
  knob (default high, API/Chat UI settable) grades every DeepSeek stage.
- Refs: DTO-D6/D7; kairyu/{dsl,orchestration}; examples/qwen3.8-deepseek-v4-8gpu/

### 2026-08-18 — [amendment] Compaction hardening from the #531 review
- What: adopted all three review fixes — a successful compaction stores only
  the compaction item (history replaced, not duplicated); truncated summaries
  return response.incomplete without a compaction item (and are not
  continuable), empty ones fail as 502 compaction_failed; tokens are now
  AES-256-GCM sealed and tenant-bound (`responses_compaction_secret_env`,
  ephemeral per-process key when unset). Example gateway wires the secret.
- Why: #531 review — stored history defeated compaction, truncated/empty
  summaries silently replaced long sessions, and marker-only tokens were
  forgeable, contradicting the opaque/self-issued contract.
- Refs: #531 review, #532–#534; m11 D4 2026-08-18 review amendment;
  `kairyu/entrypoints/server/responses_service.py`; `examples/qwen3.8-deepseek-v4-8gpu/`

### 2026-08-18 — [amendment] AUTO models on /v1/responses + Codex contract (issue #530)
- What: /v1/responses resolves everything /v1/models advertises — AUTO models
  delegate to the chat orchestration contract (metering exactly once); a
  codex-rs 0.147.0 field-level audit added buffered-stream keep-alives,
  426 on WS upgrades, remote compaction v2, and tolerant input parsing
  (reasoning echoes, output-part arrays, Codex passthrough fields, disabled
  web_search config). Verified with real codex-cli 0.147.0 (custom-provider
  and Harbor-shaped smokes) against the issue's all-engines-hidden topology.
- Why: the Terminal-Bench Codex run failed — /v1/responses only knew public
  L1 engines, and the audit showed the 404 was one of several turn-killers.
- Refs: #530; m11 D4 2026-08-18 amendment; `docs/deployment.md`;
  `kairyu/entrypoints/server/{responses_service,extra_routes,app}.py`

### 2026-08-18 — [design] Dual-track policy-ensemble DAG replaces the example L2
- What: the example's coding/general two-profile L2 (judge, verifier/refine,
  sandbox stages) is replaced by one 9-role dual-track DAG: 4 DeepSeek-written
  policies → 4 parallel policy-bound Qwen answers, ∥ thinking-DeepSeek
  critique of a quick Qwen draft, merged by direct DeepSeek; head/TTFT gate
  inherited; sandbox deployed but unreferenced. No L2 core change; both
  verify.sh gates green (20260818T025710Z, c32 1.87×→0.67×).
- Why: owner-specified new process; latency/throughput requirements unchanged.
- Refs: DTO-D1..D5 `docs/design/example-dual-track-orchestration.md`;
  `examples/qwen3.8-deepseek-v4-8gpu/`; supersedes ECO-D2/D3/D5/D6

### 2026-08-17 — [amendment] Profile judge work obeys tenant accounting
- What: profile-judge GPU work is reserved before dispatch, included in
  cumulative orchestration usage and metering, and emitted as a structured
  trace event; the reservation covers the judge plus the larger profile DAG.
- Why: PR #516 review found that the pre-admission judge call bypassed tenant
  token quotas and discarded backend-reported usage.
- Refs: PR #516 review; `kairyu/orchestration/`;
  `kairyu/entrypoints/server/app.py`

### 2026-08-17 — [amendment] Tiered Qwen MTP remains candidate-only
- What: removed MTP-3 from the selected four-replica Qwen deployment while
  retaining its c1/c4/c8 measurements as candidate evidence; compose, metadata,
  tests, and operator docs now agree on no speculation.
- Why: the public product admits 256 requests and gates c16/c32, but the new
  role-shaped run stopped at c8; the matching Qwen TP1 c16/c32 rows regressed,
  so the deployed high-concurrency envelope was not proven safe.
- Refs: #509; ECO-D4 2026-08-17 Qwen MTP concurrency amendment;
  `examples/qwen3.8-deepseek-v4-8gpu/`

### 2026-08-17 — [design] General ensemble profile and Terminal-Bench latency fixes
- What: `general_roles` — a second full-ensemble DAG under one served model,
  deterministically selected (tools/format-demand/non-code → general; code
  authoring keeps the coding DAG; never a single-engine route); example gains a
  7-role all-model general profile; verifier T 0.0→0.6/top_p 0.95 (greedy
  thinking looped: 52% of verdicts burned the 4096 cap → reverify); Qwen
  prompts lead with a shared REQUEST block (prefix reuse); measured MTP-3
  adopted on Qwen workers (c1 +43.9%, c4/c8 +26%, lossless).
- Why: issue #509 — 12/27 Terminal-Bench trials hit the 900 s agent timeout at
  p50 63.9 s/request; coding contracts idled on agent JSON turns.
- Refs: #509; ECO-D6 + ECO-D4 2026-08-17 amendment; `kairyu/orchestration/`;
  `examples/qwen3.8-deepseek-v4-8gpu/`; its MEASUREMENTS.md

### 2026-08-17 — [amendment] LLM profile judge for the coding/general split
- What: the code-authoring half of profile selection is judged by an optional
  `profile_judge` worker (example: direct DeepSeek, greedy, ≤8 tokens, 5 s
  timeout); the verdict is attached to the call at the serving boundary before
  preflight/admission so selection stays a pure function; head-disable signals
  stay deterministic and are never judged; on judge failure the keyword
  code-task signal decides as before.
- Why: owner review of #510 — keyword heuristics misroute incidental code
  vocabulary into the sandbox coding DAG and miss unlisted languages; the
  split is a semantic judgment and belongs to an LLM.
- Refs: #509, #510 review; ECO-D6 2026-08-17 LLM-profile-judge amendment;
  `kairyu/orchestration/`; `examples/qwen3.8-deepseek-v4-8gpu/auto-max.yaml`

### 2026-08-16 — [amendment] Agent tool-turn latency and cancellation (issue #495)
- What: tool-turn FAIL/refine loop removed (verifier/synthesis prompts declare
  the publisher emits the tool call; inconclusive verdicts re-verify once,
  never auto-FAIL; verifier cap 2048→4096); a complete committed opening is
  declined via a stripped NO_CONTINUATION sentinel; conversation-head
  KV-affinity session; client-disconnect cancels the non-streaming AUTO DAG
  (499); `kairyu_conductor_stage_seconds` per-stage metric; `/v1/models`
  advertises `max_model_len`; example timeouts bounded (600 s / 1800 s);
  after a still-timing-out rerun, a plain-text structured-format demand
  (Terminus-2 "format as JSON") now disables the head like tools do, and the
  example refine depth drops 2→1.
- Why: a traced minimal tool turn spent 20.8 s of 25.8 s in three thinking
  verifier passes; the rerun showed head prose corrupting JSON replies and
  ~11-minute single responses against the 900 s Terminal-Bench budget.
- Refs: #495, #501 (+review PRs #502–#507), ECO-D4 2026-08-16 amendment;
  `kairyu/orchestration/`; `kairyu/entrypoints/server/`;
  `examples/qwen3.8-deepseek-v4-8gpu/{auto-max,kairyu}.yaml`

### 2026-08-16 — [amendment] OpenAI Chat Completions output contract (issue #496)
- What: omitted output limits are context-bound (16-token fallback removed,
  admission reserves max_model_len); a caller limit is one budget across
  head+continuation with honest finish_reason; standard usage now means the
  public request/completion (orchestration_* keep #196 cumulative totals,
  metering still bills them); empty final output retries once then 502;
  `reasoning_closed`/`prompt_headless` role contracts; GET /v1/models/{id}.
- Why: issue #496 — orchestrated responses ignored caller limits, reported
  deterministic "length", and lost post-tool answers to `reasoning_content`.
- Refs: #496, #458, m11 2026-08-16 amendment, ECO-D4 2026-08-16 amendment

### 2026-08-15 — [amendment] Sandbox reaps escaped submission descendants
- What: the coding executor now acts as a child subreaper and sweeps/reaps
  same-UID runner descendants not owned by another live submission on every
  completion path.
- Why: timeout-only process-group cleanup allowed `setsid()` children to retain
  resources and mutate later execution evidence after an accepted result.
- Refs: ECO-D1; `examples/qwen3.8-deepseek-v4-8gpu/sandbox/runner.py`

### 2026-08-15 — [amendment] Executor deadline includes queue admission
- What: deployment executor config now declares queue allowance; the client
  shares one wall-plus-queue deadline across retries and the runner rejects
  admission when the residual budget cannot cover the declared wall limit.
- Why: the previous 8s queue plus 10s execution exceeded the 15s client
  deadline, discarding valid work as `unavailable` under contention.
- Refs: ECO-D1; `kairyu/orchestration/execution.py`; PR #488 review

### 2026-08-15 — [amendment] Sandbox transport removes the reverse network path
- What: replaced the bidirectional internal Docker bridge with a shared Unix
  domain socket and `network_mode: none` on the executor; Kairyu mounts the
  socket volume read-only and the deployment executor client supports UDS.
- Why: an internal bridge blocks external routing but still lets hostile
  submitted code connect back to Kairyu, violating ECO-D1's no-egress
  boundary.
- Refs: PR #488 review; ECO-D1; `kairyu/orchestration/execution.py`;
  `examples/qwen3.8-deepseek-v4-8gpu/{compose.yaml,sandbox/runner.py}`

### 2026-08-15 — [design] Coding orchestration with head streaming and sandbox execution
- What: replaced the tiered example's seven-role DAG with a nine-role coding
  DAG on the unchanged L1 fleet; added L2 mechanisms for a t=0 public head
  stream + verified continuation, incremental reasoning streaming, per-role
  sampling/prompt suffixes, and deployment-owned sandbox executors run as
  untrusted-data DAG stages; TTFT gate: public p50 ≤2× DeepSeek-direct per
  concurrency (live c1 smoke ~0.3 s vs the 1.56 s budget).
- Why: frontier-level coding accuracy needs ensemble + execution-grounded
  verification, but the old collect-then-synthesize DAG left first public
  tokens ~10× over the agreed 2×-DeepSeek TTFT budget; only E2E is
  unconstrained, so the answer opening streams while the ensemble runs.
- Refs: ECO-D1..D5 in `docs/design/example-coding-orchestration.md`;
  `kairyu/orchestration/{conductor,execution}.py`;
  `examples/qwen3.8-deepseek-v4-8gpu/`

### 2026-08-15 — [progress] Frontier examples move to Qwen3.8
- What: replaced both Qwen3.6 example surfaces with attested Qwen3.8-27B-FP8,
  selected a measured 32K/no-MTP/piecewise-graph vLLM v0.23.0 L1 envelope,
  ported it unchanged to four TP1 replicas, and passed an eight-GPU product
  chat plus embedding smoke without changing the L2 role DAG or L3 contract.
- Why: DeepSeek remains on the measured `aa0d513027` DSpark build because the
  official v0.23.0 runtime cannot serve the 0731 checkpoint's draft path.
- Refs: `examples/qwen3.8-{27b-1gpu,deepseek-v4-8gpu}/`;
  `examples/qwen3.8-27b-1gpu/MEASUREMENTS.md`

### 2026-08-14 — [progress] Accuracy removal and verification migration completed
- What: removed the named Accuracy suite and its code, fixtures, dependencies,
  tests, examples, and active documentation; retained four explicit eval suites
  and moved Kairyu serving/correctness gates to checkout-only verification.
- Refs: PR #486; `evals/`; `verification/`; `evidence/`;
  `docs/design/verification-framework.md`

### 2026-08-14 — [progress] Accuracy implementation removed
- What: removed the externally migrated Accuracy adapters, judge and published
  comparison paths, fixtures, dependencies, and tests; retained eval tests now
  live under `tests/evals/` and require an explicit retained suite.
- Refs: PR #486; `evals/`; `tests/evals/`; `pyproject.toml`

### 2026-08-14 — [progress] Verification framework split implemented
- What: moved 68 Kairyu gates into checkout-only `verification/` by scope and
  kind, added a strict registry/runner and neutral `evidence/` contracts, and
  kept all 205 tracked legacy result files byte-identical at `bench/results/`.
- Refs: PR #486; `verification/registry.toml`;
  `docs/design/verification-framework.md`

### 2026-08-14 — [design] Separate evals from verification and retire Accuracy
- What: Accuracy will be removed after its external migration; the other evals
  remain checkout-only under `evals/`; correctness, performance, resilience,
  and diagnostic gates move to `verification/` with neutral evidence contracts
  and immutable legacy artifacts.
- Why: `kairyu.bench` and `bench/` conflate model evaluation with L1/system
  verification and allow performance evidence without an explicit correctness
  link.
- Refs: `docs/design/verification-framework.md`;
  `docs/superpowers/plans/2026-08-14-verification-framework-accuracy-removal.md`

### 2026-08-14 — [amendment] Exact LiteLLM assistant history is reusable
- What: Chat Completions now drops object-valued `provider_specific_fields` and nullable legacy `function_call` metadata from assistant history while retaining typed reasoning, content, and tool calls; invalid roles, value kinds, and unrelated extras remain fail-closed.
- Why: LiteLLM 1.96.2 emits both compatibility fields in a normal assistant `model_dump()`, so the nullable-only #480 policy still rejected an unmodified second-turn request.
- Refs: issue #482; `kairyu/entrypoints/server/chat_service.py`; `tests/server/test_{chat_template_policy,openai_api,orchestration_usage_trace}.py`

### 2026-08-14 — [amendment] Assistant reasoning history round-trips
- What: Chat Completions now accepts its typed `reasoning_content` response field in assistant history and drops only nullable LiteLLM `provider_specific_fields` metadata before rendering; non-null and unknown extras remain fail-closed.
- Why: The tiered product emitted visible intermediate work that normal LiteLLM serialization returned on the next agent turn, but the input schema rejected both its own field and nullable client metadata before dispatch.
- Refs: issue #480; `kairyu/entrypoints/server/{protocol,chat_service}.py`; `tests/server/test_{chat_template_policy,openai_api,orchestration_usage_trace,prompt_offload}.py`

### 2026-08-14 — [amendment] Tiered frontier API gains offline embeddings
- What: The eight-GPU example now builds the pinned FastEmbed MiniLM bundle, publishes truthful `embed-small` beside its chat product, keeps routing and Chat UI chat-only, and fails readiness unless a two-input 384-dimensional embedding smoke returns ordered finite vectors with exact usage.
- Why: tau2 banking requires embeddings before task 1, but the example exposed no embedding route; a truthful local ID avoids claiming that MiniLM is OpenAI's `text-embedding-3-large`.
- Refs: issue #479; kairyu-bench issue #5; `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,kairyu.yaml,control.py,example.json,README.md}`

### 2026-08-13 — [amendment] Tiered L3 endpoints stay locally and externally reachable
- What: The example now binds both its Chat UI and public L3 API to all host interfaces by default while advertising the real host address, so both endpoints remain reachable through localhost and the external address; explicit bind overrides remain available.
- Why: Binding the API only to the external interface made its normal host-local URL unavailable, while publishing `0.0.0.0` as a client URL confused the listen address with the address clients should use.
- Refs: PR #478; `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,control.py,README.md}`

### 2026-08-13 — [progress] Tiered Chat UI response repair closes GPU gates
- What: Default Qwen L1 chat now returns non-empty public content with no hidden reasoning, L3 returns separate non-empty final and attributed reasoning output, the pinned tiered browser smoke passes, and the deterministic CharXiv rerun completes 10/10 scored requests with zero errors or unmeasured requests.
- Refs: PR #478; commit `56c640a`; `/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/bench-results/20260813T055825Z/`

### 2026-08-13 — [progress] Tiered Chat UI publication defaults to visible content
- What: The tiered example's Qwen replicas now default upstream HF chat templating to nonthinking while preserving request-level overrides and vision handling, so proposal and publisher roles return public content instead of exhausting their budget in reasoning-only output; GPU validation is in progress.
- Refs: PR #478; `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,README.md}`; `tests/unit/test_tiered_frontier_examplectl.py`

### 2026-08-13 — [progress] Example vision CharXiv validation closes
- What: Both the single-Qwen and tiered Qwen/DeepSeek examples completed their deterministic 10-item CharXiv image runs with all 10 target requests measured and scored and zero target failures; the tiered DAG confines private DeepSeek reasoning to planning/synthesis/verification and uses its image-capable non-thinking Qwen pool for proposals and final publication.
- Refs: PR #478; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/`; `bench/results/examples/qwen3.6-27b-1gpu/charxiv-10-gpu-validation-6/`; `/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/bench-results/charxiv-10-gpu-validation-4/`

### 2026-08-13 — [verified] Single-Qwen CharXiv vision closes 10/10
- What: The 1-GPU Qwen example completed its deterministic CharXiv run with `n=10`, `n_scored=10`, `requests=10`, `errors=0`, and `unmeasured_requests=0`; score was 60%. Tiered orchestration validation remains in progress.
- Refs: `examples/qwen3.6-27b-1gpu/`; `bench/results/examples/qwen3.6-27b-1gpu/charxiv-10-gpu-validation-6/`

### 2026-08-13 — [progress] Example vision orchestration enters GPU validation
- What: L3 now preserves validated image inputs for L2, role DAGs pass media only to capable workers, both Qwen examples enable the pinned checkpoint's vision encoder, and each owns a fail-closed 10-item CharXiv command; real-GPU closure is in progress.
- Refs: `kairyu/orchestration/`; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/`

### 2026-08-13 — [amendment] Kind CI tool setup is shared and verified
- What: All four kind gates now use one pinned installer with bounded nested retries, HTTPS-only downloads, SHA-256 verification, and executable version checks; a policy test bans the flaky action path from every workflow.
- Why: Per-workflow fixes left the same dependency-acquisition failure in other gates, incorrectly surfacing CI infrastructure failures as product-quality failures.
- Refs: PR #476; `scripts/install_kind_tools.sh`; `.github/workflows/{ci,f1a-churn,f1b-rollout,f1c-gateway}.yml`; `tests/unit/test_ci_workflow_policy.py`

### 2026-08-13 — [progress] Tiered PR infrastructure flakes are bounded
- What: F1c now installs pinned kind/kubectl binaries with retried, checksum-verified downloads, and F1b retries one failed import of its already-frozen image archives. Product gate behavior is unchanged.
- Refs: PR #476; `.github/workflows/f1c-gateway.yml`; `scripts/kind_rollout_gate.sh`

### 2026-08-13 — [progress] Tiered example collapses to one layered product path
- What: The example now keeps one `auto-max.yaml` policy and one public Chat UI model; L2 borrows deployment-owned L1 pools, runs the bounded seven-role DAG, and streams attributed intermediate output separately from publisher content in the same response. Obsolete auto/MoA candidate YAMLs and commands were removed.
- Refs: PR #476; `examples/qwen3.6-deepseek-v4-8gpu/`; `scripts/webui_browser_smoke.mjs`

### 2026-08-13 — [design] Tiered UI owns one layered orchestration path
- What: The tiered example will expose one product model, borrow deployment L1 pools directly from L2, run a bounded verifier-gated DAG, and show policy-enabled model-attributed intermediate output separately from the final answer.
- Why: The prior loopback L3 workers and single-pass MoA policy did not exercise the intended L3→L2→L1 boundary, while the product UI now requires inspectable intermediate work without mixing it into the committed answer.
- Refs: EO-D1..EO-D5; `docs/design/example-layered-orchestration.md`; `docs/superpowers/plans/2026-08-13-example-layered-orchestration-correction.md`

### 2026-08-12 — [amendment] SWE-bench fixes selection before generation
- What: SWE-bench now persists the official ordered selection before generation, evaluates that full set even when a prediction is omitted, accepts the official v4.1 completed/error report overlap with error precedence, and retains redacted stage logs plus raw upstream evidence; failed runs resume while explicit reruns use isolated artifacts.
- Why: mini-SWE-agent can exit successfully after a worker omits its prediction, so deriving `--instance_ids` from `preds.json` could shrink the denominator and deleting work directories could discard the only audit and resume evidence.
- Refs: issue #472; `kairyu/bench/adapters/swebench.py`; `kairyu/bench/adapters/base.py`; `docs/superpowers/specs/2026-08-12-swe-bench-verified-design.md`

### 2026-08-12 — [progress] SWE-bench Verified launches through Accuracy
- What: Accuracy now runs the 500-task Verified test split through mini-SWE-agent's standard 250-step configuration and the official SWE-bench evaluator, preserves every selected task in the resolved-rate denominator, retains auditable commands and raw evidence, and extends the sourced comparison report with Fable 5's published result without claiming one-trial/five-trial parity.
- Refs: issue #472; `kairyu/bench/adapters/swebench*.py`; `docs/benchmarks.md`; `tests/bench/test_bench_agentic*.py`; `tests/bench/test_bench_compare.py`

### 2026-08-12 — [progress] Tiered PR CI regressions reproduced and closed
- What: The Open WebUI initial/outage/recovery browser phases pass after keeping the deterministic AUTO SSE marker inside the mock's prompt-boundary echo window; the tiered control test now stubs storage path preparation instead of attempting `/mnt/nvme` writes on hosted runners. Production NVMe enforcement and L2/L3 behavior are unchanged.
- Refs: `scripts/webui_browser_smoke.mjs`; `tests/unit/test_tiered_frontier_examplectl.py`; PR #471

### 2026-08-12 — [design] SWE-bench Verified joins the Accuracy suite
- What: The Accuracy suite will run SWE-bench Verified through mini-SWE-agent's standard 250-step `verified` flow and the official SWE-bench harness, preserve exact selected-instance denominators and failure classes, and extend the sourced frontier comparison without inventing missing exact-model scores.
- Why: Issue #472 requires the benchmark to launch like existing suites and follow their score-comparison reporting, while upstream methodology differences and OpenAI's retirement warning must remain explicit and auditable.
- Refs: issue #472; `docs/superpowers/specs/2026-08-12-swe-bench-verified-design.md`

### 2026-08-12 — [progress] Tiered RTX PRO example closes full task-set scoring
- What: The selected private-thinking MoA-3 policy launched all 89 Terminal-Bench 2.1 tasks and closed at 60/89 (67.42%) under official task verifiers and Harbor's zero-inclusive Mean: 86 verifier rewards (60 one, 26 zero), two rewardless infrastructure errors, and one rewardless operator-terminated CPU outlier. The score does not substitute diagnostic retries.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,terminalbench-result.json}`; run ID `terminalbench-2.1-auto-max-thinking2048-full-v1-20260812`; PR #471

### 2026-08-12 — [amendment] Agentic evidence survives interruption with bounded outliers
- What: Terminal-Bench now retains its raw Harbor job directory for authoritative per-task evidence/resume, and sets agent `max_timeout_sec=900` before the existing 8x multiplier so every task has a 7200-second effective agent cap. Official Harbor reward/Mean semantics remain unchanged.
- Why: A 3600-second task declaration previously expanded to eight hours and interrupt cleanup deleted the partial raw job, making one pathological task both operationally unbounded and non-resumable.
- Refs: `kairyu/bench/adapters/terminal_bench.py`; `tests/bench/test_bench_agentic*.py`; PR #471

### 2026-08-12 — [progress] Thinking MoA-3 clears the quality floor
- What: On the same fixed four Terminal-Bench 2.1 tasks, the 2048-token private-thinking MoA-3 policy scored 3/4 versus direct DeepSeek's 2/4. Both completed 4/4 items with zero failed/unjudged/skipped trials and null item errors on a clean source tree; `kairyu-auto-max` is selected for the full 89-task run.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/MEASUREMENTS.md`; run ID `terminalbench-selection-thinking2048-vs-deepseek-20260812`; PR #471

### 2026-08-12 — [progress] Thinking MoA-3 restores L3 reliability within its performance envelope
- What: The generic 2048-token internal allowance completed the quality-first MoA-3 c1/c8/c16/c32 matrix with 128/128 non-empty public answers, valid traces, exact fan-out, and zero errors. Versus the failed 1024 run it preserves c1 TTFT/E2E within 1.2%/0.8%; versus ordinary MoA-3 at c8 it costs 18.8% TTFT but only 5.0% E2E while increasing public TPS 8.0%. It advances to the fixed quality pilot.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/MEASUREMENTS.md`; run ID `l3-auto-max-thinking2048-public-v1-20260812`; PR #471

### 2026-08-12 — [progress] Ordinary synthesis fails to preserve the quality floor
- What: Ordinary-DeepSeek MoA-3 scored 0/3 completed pilot tasks, making the fourth task irrelevant to its maximum possible 1/4 score versus direct DeepSeek's clean 2/4. The quality candidate returns to private-thinking MoA-3 with a generic 2048-token internal allowance to eliminate the previously measured 1024-token boundary-exhaustion tail before re-running L3 performance and quality gates.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,auto-max.yaml}`; run ID `terminalbench-selection-moa3-vs-deepseek-20260812`; PR #471

### 2026-08-12 — [progress] MoA-2 fails the quality-selection gate
- What: The performance-winning ordinary-chat MoA-2 candidate scored 1/3 completed Terminal-Bench 2.1 pilot tasks and failed the fourth request with `BadGatewayError`, below direct DeepSeek's clean 2/4 baseline. MoA-2 is rejected; the already performance-qualified MoA-3 ordinary-chat candidate advances to the same fixed four-task gate.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,auto-max-chat.yaml}`; run ID `terminalbench-selection-moa2-vs-deepseek-20260812`; PR #471

### 2026-08-12 — [progress] MoA-2 chat synthesis wins the L3 performance gate
- What: The ordinary-DeepSeek MoA-2 candidate completed c1/c8/c16/c32 with 32/32 non-empty answers and valid traces at every row; versus MoA-3 it preserves c1 and cuts median TTFT by 17.1%/21.0%/20.7% at c8/c16/c32. The quality pilot now compares only this winner with direct DeepSeek, and validates raw zero-failed/unjudged/skipped/error evidence on a clean source tree.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,benchmark.py}`; run ID `l3-auto-max-chat-moa2-public-v1-20260812`; PR #471

### 2026-08-12 — [progress] Tiered quality candidate reduces fan-out, not proposal depth
- What: The 512-token private-cap A/B retained 32/32 valid L3 responses but did not improve c1/c8 median TTFT beyond run noise, so the quality candidate restores the 1024-token allowance and changes ordinary-chat synthesis from MoA-3 to MoA-2 for the next performance/quality gate.
- Refs: run IDs `l3-auto-max-chat-moa3-public-v1-20260812`, `l3-auto-max-chat-moa3-private512-public-v1-20260812`; `examples/qwen3.6-deepseek-v4-8gpu/`; PR #471

### 2026-08-12 — [amendment] L2 bounds private work independently of public output
- What: Orchestrator YAML and decorator specs now expose a validated, backend-neutral `internal_max_tokens` policy that caps private planning/proposal/verification generations without reducing the caller's final-answer budget. The tiered MoA-3 chat-synthesis candidate pins 512 after its reliable but 1024-token L3 matrix showed excessive c16/c32 latency.
- Why: Private proposal length is an operator latency/quality tradeoff and must remain portable policy rather than a model- or example-specific core-code branch.
- Refs: M11 D2; `kairyu/dsl/`; `examples/qwen3.6-deepseek-v4-8gpu/auto-max-chat.yaml`; run ID `l3-auto-max-chat-moa3-public-v1-20260812`; PR #471

### 2026-08-12 — [progress] Empty reasoning finals fail closed
- What: Completion reasoning filters now require non-empty public content after the private terminator, and the serving harness rejects sanitized SSE errors. The tiered example adds a matched MoA-3 ordinary-chat synthesis candidate after thinking MoA returned one empty public answer in the c8 L3 matrix.
- Refs: `kairyu/engine/openai_backend.py`; `bench/serving_bench.py`; `examples/qwen3.6-deepseek-v4-8gpu/`; run ID `l3-auto-max-moa3-public-v2-20260812`; PR #471

### 2026-08-12 — [progress] Private reasoning rejects unsafe token metadata
- What: A pre-rendered vLLM completion with a configured private-reasoning boundary now rejects logprobs before dispatch, preventing raw hidden-token metadata from bypassing the text filter; ordinary direct backends retain logprobs support.
- Refs: `kairyu/engine/openai_backend.py`; `tests/unit/test_openai_backend.py`; PR #471

### 2026-08-12 — [progress] L3 benchmarks separate public output from private work
- What: The serving harness can count final assistant content with a pinned tokenizer after the timed interval, reporting public TPS/TPOT beside cumulative orchestration tokens and E2E p50/p99. The tiered example uses natural-EOS reasoning requests plus a loopback-only DeepSeek tokenizer oracle; ChatUI still reaches only Kairyu L3.
- Refs: `bench/serving_bench.py`; `examples/qwen3.6-deepseek-v4-8gpu/`; PR #471

### 2026-08-12 — [progress] L2 exposes the answer, not private synthesis work
- What: OpenAI-compatible vLLM completion replicas can now opt into a bounded fail-closed private-reasoning terminator for buffered and streamed output; multi-stage orchestration strips reasoning metadata at its public boundary, and role-tagged conversation JSON explicitly remains context rather than a response envelope. The tiered DeepSeek thinking alias enables this generic contract.
- Refs: `kairyu/engine/openai_backend.py`; `kairyu/orchestration/orchestrator.py`; `kairyu/entrypoints/server/chat_service.py`; `examples/qwen3.6-deepseek-v4-8gpu/`; PR #471

### 2026-08-11 — [progress] Tiered Chat UI defaults to the quality-first L3 path
- What: The eight-GPU tiered launcher now exposes an explicitly unauthenticated Open WebUI on all host interfaces, prints its outward-facing URL, and disables persistent UI configuration so every start selects `kairyu-auto-max`; Kairyu L3 remains loopback-only and the UI has no L1/L2 bypass.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,control.py,README.md}`; `tests/unit/test_tiered_frontier_examplectl.py`; PR #471

### 2026-08-11 — [progress] MoA preserves the original response contract
- What: Proposal roles now independently draft against an explicit original-request boundary, while synthesis treats candidates as untrusted advice and must preserve machine-readable output contracts without leaking ensemble meta-analysis; focused MoA/orchestrator/backend regressions pass.
- Refs: `kairyu/orchestration/moa.py`; `tests/unit/test_moa.py`; PR #471

### 2026-08-11 — [progress] Prefix affinity obeys replica overload policy
- What: ReplicaPool now applies its configured queue-depth valve after both prefix-aware and session-affine placement, retaining idle KV locality while overflowing concurrent warm-prefix work to least-outstanding replicas; focused orchestration and backend regressions pass.
- Refs: `kairyu/orchestration/replica.py`; `tests/unit/test_kv_routing.py`; PR #471

### 2026-08-11 — [progress] Tiered eight-GPU vLLM example enters runtime validation
- What: Added a one-command Open WebUI -> Kairyu L3/L2 -> vLLM L1 contract for four Qwen3.6 FP8 TP1 replicas plus one DeepSeek-V4 TP4/EP4 tier, with rule routing, MoA-2/MoA-3, `/mnt/nvme`-only persistence, independent/all serving and orchestration runners, and a full-dataset Terminal-Bench 2.1 entrypoint. Static and contract tests pass; runtime tuning and complete evidence remain in progress.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/`; `tests/unit/test_tiered_frontier_examplectl.py`

### 2026-08-11 — [progress] One-GPU Qwen3.6 example completes measured gate
- What: The one-command Qwen3.6-27B FP8 stack on one RTX PRO 6000 completed its final unique-prefix 8K/256 matrix (41.43 output tok/s with 190.82 ms median TTFT at c1; 842.35 output tok/s with 1.247 s median TTFT at c32) and deterministic LiveCodeBench subset (11/20, 55.0%, 20/20 scored, zero request errors/retries/unmeasured rows). No-MTP won TTFT at every tested concurrency and aggregate throughput at c16/c32; MTP-5 crashed at c16 and the 32K prefill budget exhausted VRAM at c32.
- Refs: PR #469; `examples/qwen3.6-27b-1gpu/MEASUREMENTS.md`; run IDs `final-no-mtp-20260811` and `qwen36-fp8-no-mtp-lcb20-20260811`

### 2026-08-11 — [progress] One-GPU Qwen3.6 vLLM example enters runtime validation
- What: A second example adds one-command Open WebUI -> Kairyu L3 -> pinned vLLM L1 serving for the official Qwen3.6-27B FP8 checkpoint on one selected RTX PRO 6000; model/UI persistence is fail-closed below `/mnt/nvme`, and independent/all serving plus deterministic 20-row LiveCodeBench runners are contract-tested. Runtime tuning and measured evidence remain in progress.
- Refs: `examples/qwen3.6-27b-1gpu/`; `tests/unit/test_frontier_examplectl.py`

### 2026-08-11 — [progress] Accuracy comparisons cite every reference score
- What: Accuracy reports now compare local runs with the final eight-model benchmark matrix, preserve missing values, and attach source markers, dates, source class, conditions, and notes to selected and alternate scores in Markdown and JSON.
- Refs: `kairyu/bench/{reference.py,compare.py}`; `tests/bench/test_bench_compare.py`; `docs/superpowers/specs/2026-08-11-accuracy-reference-sources-design.md`

### 2026-08-11 — [design] Accuracy reports bind reference scores to sources
- What: The Accuracy comparison will use the supplied eight-model matrix and render a source marker, linked source metadata, measurement condition, and notes for every selected or alternate published score; missing values remain missing and old DeepSeek snapshots do not fill the 0731 column.
- Why: A numeric comparison without cell-level provenance obscures provider, third-party, harness, and version differences and cannot be audited from the generated report.
- Refs: `docs/superpowers/specs/2026-08-11-accuracy-reference-sources-design.md`; `kairyu/bench/{reference.py,compare.py}`

### 2026-08-11 — [progress] Full DeepSeek-V4 LiveCodeBench evidence completes
- What: The clean-cache TP8+EP8 DSpark-5 example completed and scored all 1,055 LiveCodeBench release_v6 problems: 759 passed (71.9431%), all target timings measured, and zero request errors, retries, or unmeasured rows. The warm serving matrix reached 168.01 output tok/s at c1 with 220.10 ms median TTFT and 972.93 output tok/s at c32.
- Refs: PR #468; `examples/deepseek-v4-flash-0731-8gpu/MEASUREMENTS.md`; run IDs `livecodebench-full-tp8-ep8-dspark5-clean-sse16m-20260811` and `tp8-ep8-dspark5-warm-20260811`

### 2026-08-11 — [progress] Frontier benchmark streams support 32K-token evidence
- What: The strict benchmark SSE reader retains its fail-closed bound but raises it from 1 MiB to 16 MiB so 32K-token DeepSeek streams are not rejected solely by per-token JSON framing. The example runner now also rejects a nominally successful CLI run unless all 1,055 LiveCodeBench rows have measured, error-free evidence.
- Refs: PR #468; `kairyu/bench/streaming.py`; `examples/deepseek-v4-flash-0731-8gpu/benchmark.py`; full-run diagnosis on 2026-08-11

### 2026-08-11 — [progress] Kairyu-owned vLLM prompts bypass a second chat template
- What: Pre-rendered `TemplatedPrompt` requests sent to a vLLM-compatible backend now use `/completions` for both buffered and streaming generation. This preserves the checkpoint's single prompt boundary, prevents DeepSeek chat mode from being wrapped into thinking mode again, and returns final answer text to Open WebUI.
- Refs: PR #468; `kairyu/engine/openai_backend.py`; `tests/unit/test_openai_backend.py`; real DeepSeek-V4 direct and streaming probes on 2026-08-11

### 2026-08-11 — [progress] Open WebUI tool metadata no longer blocks text requests
- What: The DeepSeek prompt template now ignores `tools` metadata emitted by OpenAI-compatible UIs and renders the ordinary text conversation; model-side function-tool execution remains outside this example. The failing Open WebUI prompt was reproduced against Kairyu.
- Refs: PR #468; `examples/deepseek-v4-flash-0731-8gpu/deepseek-v4-0731.jinja`; `tests/unit/test_frontier_examplectl.py`

### 2026-08-11 — [progress] SM120 example reaches end-to-end readiness
- What: The pinned vLLM build now starts TP8/EP8 with DSpark-5 and the full 1M context on eight RTX PRO 6000 Blackwell GPUs; SM100-only MegaMoE and FP4 indexer-cache paths are explicitly disabled. Open WebUI reached Kairyu L3 and completed a real model response; the public UI bind remains separate from the loopback-only API.
- Refs: PR #468; `examples/deepseek-v4-flash-0731-8gpu/{compose.yaml,control.py,README.md}`; runtime validation on 2026-08-11

### 2026-08-11 — [amendment] Example surface becomes one L3-to-vLLM deployment
- What: The three frontier environments and shared controllers are replaced by one exact 8 x RTX PRO 6000 Blackwell DeepSeek-V4-Flash-0731 stack; Open WebUI targets Kairyu L3, which owns checkpoint prompt rendering and delegates inference to an eight-GPU vLLM L1. One-command lifecycle and independent/all serving and full 1,055-row LiveCodeBench runners are added; measurement remains in progress.
- Why: The owner requested one reproducible, same-host optimized deployment and complete measured evidence instead of a matrix of unvalidated example variants.
- Refs: FN-D8; `examples/deepseek-v4-flash-0731-8gpu/`; `tests/unit/test_frontier_examplectl.py`

### 2026-08-10 — [progress] Qwen example gains a fixed 30-item performance diagnostic
- What: The full-dataset example default remains unchanged; a separate entrypoint fixes LiveCodeBench selection to `limit=30`, `seed=0`, and concurrency 16, while the shared controller records those CLI overrides in both backend runs.
- Refs: PR #465; `examples/_shared/benchctl.py`; `examples/qwen3.6-27b-1gpu/bench-livecodebench-30.sh`

### 2026-08-10 — [amendment] External benchmark concurrency increases to sixteen
- What: Main, judge, serving, and all frontier example clients now use sixteen simultaneous in-flight requests; combined orchestration applies the same external limit in addition to its internal replica and proposal fan-out.
- Why: The owner selected the higher request wave after observed KV-cache headroom showed that eight did not saturate the single-GPU engine.
- Refs: supersedes the concurrency-eight and concurrency-one entries below; PR #465; `kairyu/bench/`; `bench/{serving_bench.py,tiered_auto_bench.py,configs/}`; `examples/`

### 2026-08-10 — [amendment] External benchmark concurrency returns to eight
- What: Main, judge, serving, and all frontier example clients now use eight simultaneous in-flight requests; combined orchestration also applies the same external limit in addition to its internal replica and proposal fan-out.
- Why: The owner selected maximum aggregate per-GPU continuous-batching throughput over single-request isolation after a concurrency-1 LiveCodeBench run exposed unacceptable long-tail wall time.
- Refs: supersedes the two concurrency-1 entries below; PR #465; `kairyu/bench/`; `bench/{serving_bench.py,tiered_auto_bench.py,configs/}`; `examples/`

### 2026-08-10 — [progress] Every generic external benchmark client is serial by default
- What: Judge calls, the standalone serving runner, and checked-in accuracy, core, structured, and quantization configs now join the main runner and frontier examples at concurrency 1; load-balanced and intentional stress runs can still opt into higher values explicitly.
- Why: A separate judge semaphore and older sample configs could silently restore parallel load even when the primary benchmark client was serialized.
- Refs: PR #465; `kairyu/bench/types.py`; `bench/{serving_bench.py,configs/}`

### 2026-08-10 — [progress] Frontier benchmarks default to one external request
- What: The generic runner and all Qwen, DeepSeek, and combined example commands now default to and explicitly record external concurrency 1; combined orchestration retains its independent four-replica Qwen pool and internal proposal fan-out.
- Why: Without an explicitly load-balanced target, concurrent client requests distort single-engine accuracy, TTFT, and TPS comparisons and impose unintended GPU load.
- Refs: PR #465; `kairyu/bench/{types.py,adapters/base.py}`; `examples/{_shared,qwen3.6-27b-1gpu,deepseek-v4-flash-0731-8gpu,qwen3.6-deepseek-v4-8gpu}/`; `bench/tiered_auto_bench.py`

### 2026-08-10 — [progress] Frontier quality runs survive long reasoning phases
- What: Benchmark CLI now accepts a positive generation read timeout, and Qwen, DeepSeek, and combined frontier examples pin a one-day allowance so reasoning-only streams cannot be retried or failed by the generic 600-second limit before final code appears.
- Refs: PR #465; `kairyu/bench/{cli,config}.py`; `examples/{_shared,qwen3.6-27b-1gpu,deepseek-v4-flash-0731-8gpu,qwen3.6-deepseek-v4-8gpu}/`

### 2026-08-10 — [progress] Benchmark CLI preserves explicit output budgets
- What: `kairyu bench run` now accepts a positive `--max-output-tokens` override and applies it to every CLI-declared target, so frontier example budgets reach adapters instead of failing at argument parsing or reverting to 8,192 tokens.
- Refs: PR #465; `kairyu/bench/{cli,config,types}.py`; `tests/bench/test_bench_config.py`

### 2026-08-10 — [progress] CUDA runtime copies uv from its pinned image path
- What: The production CUDA Dockerfile now copies `uv` from `/usr/local/bin/uv`, the path present in the pinned Python-bearing uv image, instead of the absent distroless-image `/uv` path; a full SM120 AOT CUDA image build completed successfully.
- Refs: PR #465; `Dockerfile.cuda`; `tests/unit/test_dockerfile_cuda_aot.py`

### 2026-08-10 — [progress] Qwen quality runs retain the checkpoint thinking budget
- What: Single-GPU and combined Qwen3.6 examples now pin an 81,920-token quality-output budget from the checkpoint documentation, and the shared benchmark controller forwards and validates it instead of silently truncating every LiveCodeBench response at the generic 8,192-token default.
- Refs: PR #465; `examples/_shared/benchctl.py`; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/example.json`

### 2026-08-10 — [progress] Gateways attest unanimous replica deployment metadata
- What: Replica pools now publish revision, native context, quantization, cache, parallelism, feature, and image metadata when every configured replica agrees, even when an external engine such as vLLM has no Kairyu `/backends` endpoint; disagreement remains fail-closed and unreported.
- Refs: PR #465; `kairyu/orchestration/replica.py`; `kairyu/entrypoints/server/health.py`; `tests/server/test_backends.py`

### 2026-08-10 — [progress] Frontier native metadata passes static validation
- What: Native Qwen and DeepSeek examples now pass the same constructor-contract validation used at serve startup; Kairyu accepts and type-checks the frontier execution, revision, quantization, prefix-state, image, MTP, and DSpark metadata, and the DeepSeek EP example declares its required BF16 KV cache explicitly.
- Refs: PR #465; `kairyu/engine/config_validation.py`; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/*kairyu.yaml`

### 2026-08-10 — [progress] Frontier gateway metadata passes static validation
- What: OpenAI-compatible replicas now statically accept and type-check the full constructor metadata contract used by every Qwen, DeepSeek, and orchestration gateway; unsupported-option errors name the drifting fields.
- Refs: PR #465; `kairyu/engine/config_validation.py`; `examples/*/*gateway.yaml`

### 2026-08-10 — [progress] Qwen3.6 vLLM examples fit the hybrid cache budget
- What: Qwen3.6 vLLM services in the single-GPU and combined orchestration examples now cap active sequences at 64, matching the native backend and fitting the measured 714-block Mamba cache ceiling at native context.
- Refs: PR #465; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/compose.yaml`

### 2026-08-10 — [progress] Frontier Compose identities accept model-version names
- What: The shared lifecycle now normalizes environment and profile text into a Docker Compose-compatible project identity, with Qwen3.6 single-model and orchestration coverage.
- Refs: PR #465; `examples/_shared/examplectl.py`; `tests/unit/test_frontier_examplectl.py`

### 2026-08-10 — [amendment] Frontier examples use process-only configuration
- What: Qwen 1-GPU, DeepSeek 8-GPU, and combined orchestration lifecycle configuration now comes only from inherited environment variables; user and runtime dotenv files were removed, Compose's implicit dotenv loading is disabled, and compose paths no longer depend on the caller's working directory.
- Why: Repository-local dotenv files mixed credentials and operator state with examples, made non-interactive inheritance ambiguous, and allowed Docker Compose to silently substitute different inputs than the lifecycle preflight used.
- Refs: `examples/_shared/examplectl.py`; `examples/{qwen3.6-27b-1gpu,deepseek-v4-flash-0731-8gpu,qwen3.6-deepseek-v4-8gpu}/`

### 2026-08-10 — [amendment] Frontier examples use external model storage safely
- What: Frontier lifecycle preflight and model volumes now honor an absolute external storage root, download images use their available `python3`, and inherited HF credentials are forwarded by name rather than embedded in process arguments.
- Why: The Qwen3.6 1-GPU example checked the repository filesystem instead of its model volume, assumed an absent `python` executable in the pinned vLLM image, and could expose a token through failure diagnostics.
- Refs: `examples/_shared/examplectl.py`; `examples/qwen3.6-27b-1gpu/`

### 2026-08-10 — [amendment] DeepSeek V4 native EP runs packed checkpoint experts
- What: DeepSeek V4 gained strict official-checkpoint validation, direct SM120 E2M1/UE8M0 expert kernels, bounded FP4 Lightning Indexer state, request-owned EP2/4/8 Attention-DP, fixed NCCL all-to-all, EP8 example selection, and a committed 1M gate; single-GPU FP4 and two-rank ragged EP smokes pass.
- Why: the prior frontier example exposed a native target but still rejected distributed DeepSeek and could not execute its official FP4 expert ABI.
- Refs: FN-D2–FN-D4, FN-D7; `kairyu/models/deepseek_v4*.py`; `kairyu/kernels/deepseek_v4_moe_gpu.py`; `scripts/gpu_gates/deepseek_v4_native_1m.sh`

### 2026-08-09 — [amendment] Frontier production selection uses architecture state
- What: Qwen3.6/DeepSeek V4 gained composite cache descriptors, complete-prefix snapshots, transactional rollback, and a single-rank incremental runner; the example surface was rebuilt around three pinned offline environments and a report-owning benchmark CLI.
- Why: full-sequence recomputation cannot exercise native context or serving performance, while generic paged KV cannot represent DeltaNet or HCA/CSA state. DeepSeek EP/Attention-DP and GPU gates remain explicitly open.
- Refs: FN-D1–FN-D7; `docs/design/frontier-native-runtime.md`; `examples/`; `kairyu/engine/core/frontier_{cache,runner}.py`

### 2026-08-09 — [progress] Accuracy reports add sourced frontier gaps and target timing
- What: Accuracy comparison artifacts now carry a SHA-bound six-model public-score catalog and per-reference gaps; direct generation rows retain streamed TTFT/TPS evidence and every scoreboard renders target-only performance coverage.
- Refs: `kairyu/bench/{reference,compare,streaming,aggregate,types}.py`; `docs/benchmarks.md`

### 2026-08-09 — [progress] Portable CPU CI schedules valid jobs again
- What: tokenizer cache setup moved from job-level runner context into a runtime environment step, restoring the full Python 3.11/3.12 CPU matrix.
- Refs: PR #462; `.github/workflows/ci.yml`

### 2026-08-09 — [design] Frontier text models use a single-device reference path
- What: Qwen3.6 dense/MoE, DeepSeek V4, and Kimi K3 enter the model zoo through eager complete-sequence recomputation; public block-FP8/MXFP4 checkpoints use official loaders.
- Why: hybrid recurrent/compressed attention state cannot truthfully reuse the existing paged-KV contract without architecture-specific cache evidence.
- Refs: FZ-D1–FZ-D3; `docs/design/frontier-model-zoo.md`; `kairyu/engine/core/recompute_runner.py`

### 2026-08-08 — [design] Evidence mechanics are package-owned; gate meaning stays local
- What: canonical JSON/hash, strict indexed JSONL, artifact paths, raw-only replay, and exact retained-manifest verification moved to `kairyu.bench.evidence`; G4 M-A3 and G5 F4b use the full path, while F1a shares only its compatible file digest.
- Why: new gates should not duplicate tamper-resistant framing and replay, but package code must not absorb repository-only schemas, thresholds, verdicts, or heterogeneous sidecar contracts.
- Refs: issue #382; `docs/design/issue-382-evidence-library.md`; `kairyu/bench/evidence.py`

### 2026-08-08 — [amendment] W4A16 preserves FP16 scales; cache salt stays explicit
- What: AWQ/GPTQ uses FP16 dot operands with FP32 accumulation for either model dtype; router/output/draft projections remain accuracy-default dense; native cache identity remains the exact token tuple without an in-process tenant salt.
- Why: BF16 dequantization discarded checkpoint scale precision, while mapping `cache_salt` to affinity would falsely claim partitioning across RadixKV, tier, event, and routing identities.
- Refs: issue #366; M1 D5; M14 §8; `kairyu/kernels/quant_gemm_gpu.py`

### 2026-08-08 — [design] Long-context accuracy is a fixed-length NIAH curve
- What: a package-owned suite runs one deterministic exact-retrieval task at 4K/8K/16K/32K/64K/128K, with 20 evenly positioned needles per row and whole-row target-window gating.
- Why: one aggregate MRCR/LongBench score cannot show the context length where accuracy collapses; ordinary rows reuse the existing evidence and comparison contracts without a parallel harness.
- Refs: issue #374; `docs/design/issue-374-long-context-sweep.md`; `kairyu/bench/adapters/ruler_niah.py`

### 2026-08-08 — [amendment] Grouped-MoE probe failures always fall back
- What: capability and plan-warmup probes catch backend-specific ordinary exceptions and decline the derived grouped pack.
- Why: cuDNN graph-support errors do not consistently inherit `RuntimeError`; an optional optimization must not abort otherwise supported reference serving.
- Refs: issue #331; PR #453 Fable 5 re-review; M15 A12; `kairyu/models/grouped_moe.py`

### 2026-08-08 — [amendment] Grouped MoE is capability-probed and decode-bounded
- What: correcting the preceding #331 entry, grouped plans are power-of-two bucketed and prewarmed through 8,192 total rows before graph capture; larger EP prefills and unsupported/custom runtimes keep the reference transport/math.
- Why: unconditional fixed capacity multiplied prefill memory/FLOPs by EP degree, while late cuDNN plan/workspace growth could invalidate captured graph pointers.
- Refs: issue #331; PR #453 Fable 5 review; M15 A11; M16 D3; `kairyu/models/grouped_moe.py`

### 2026-08-08 — [amendment] Dense MoE uses grouped GEMM and fixed EP capacity
- What: CUDA BF16 experts use canonical-storage packs, device-side sort/offsets, and two FlashInfer cuDNN grouped GEMMs; generic EP uses fixed peer-capacity forward/reverse all-to-all buffers with host-constant splits.
- Why: data-dependent `unique`/`nonzero` expert loops and count `.item()`/`.tolist()` forced up to one host synchronization and small GEMM per expert, preventing CUDA graph capture.
- Refs: issue #331; M15 A10; M16 D3; `kairyu/models/{grouped_moe,moe,moe_parallel}.py`

### 2026-08-08 — [amendment] TP step headers remain on sleeping Gloo transport
- What: correcting the preceding #323 entry, a two-word Gloo tensor header frames every transaction; only encoded `StepDelta` payloads use the bounded NCCL model group.
- Why: posting a process-lifetime NCCL receive while idle burns resources and can strand orphaned GPU workers after rank-0 death; a Gloo tensor header avoids pickle while retaining sleeping TCP liveness.
- Refs: issue #323; PR #452 Fable 5 review; M16 D4; `kairyu/engine/core/worker.py`

### 2026-08-08 — [amendment] TP step control uses a dedicated NCCL tensor channel
- What: hot-path `StepDelta` state is losslessly encoded into int64 tensors on a long-idle NCCL subgroup; rare controls remain Gloo objects and model collectives retain their short fail-fast subgroup.
- Why: removing per-step pickle/TCP overhead must not make healthy idle workers time out or weaken bounded model-operation failure detection.
- Refs: issue #323; M16 D4; `kairyu/engine/core/{step_input,worker}.py`

### 2026-08-08 — [amendment] Structured generation uses the device sampling seam
- What: regex, EBNF, and structural-tag formats join JSON grammars; native strict tools compile parser-matched schemas (one call for Llama); both layouts synchronously isolate malformed requests; per-process vocabulary/compiler caches avoid rebuilds without compile-time global locking; CUDA masks avoid full logits-row D2H.
- Why: structured and strict-tool requests must be correctness invariants without forcing every token through the vocabulary-sized CPU sampling path.
- Refs: issue #363; M8 D2; `kairyu/engine/{backend.py,core/{structured,sampler}.py}`

### 2026-08-08 — [amendment] Deterministic drafts preserve sampled distributions
- What: n-gram/EAGLE/MTP argmax drafts use point-mass rejection verification for T>0; per-row history slicing also enables penalties while grammar/forced continuations keep the safe bypass.
- Why: one target draw exactly realizes acceptance plus residual correction for q(t)=1, avoiding vocabulary-sized draft/target probability transfer.
- Refs: issue #358; M8 D4; `kairyu/engine/core/{spec_decode,spec_runner,model_runner}.py`

### 2026-08-08 — [amendment] FP8 audit separates arithmetic noise from error
- What: calibrated dequant auditing permits a documented 2^-48 relative FP64 comparison slack while retaining exact stored-byte, finite, range, and physical E4M3 error checks.
- Why: byte-perfect writes could exceed the theoretical error boundary by a few FP64 ulps, making the calibrated gate unsatisfiable independently of model quality.
- Refs: issue #357; PR #449 Fable 5 review; `bench/fp8_kv_g4_ekv_bench.py`; `tests/verification/test_fp8_kv_g4_ekv_bench.py`

### 2026-08-08 — [progress] Calibrated FP8 KV re-bake retains FAIL
- What: the clean Qwen3-32B SM120 re-bake passed cache cosine/NRMSE and selected-logprob bounds, but failed 16K/32K exact tokens and the disjoint calibration envelope; public FP8 KV remains rejected.
- Refs: issue #357; `bench/results/g4-ekv-fp8-kv-qwen3-32b-sm120-calibrated-fail-2026-08-08/`; FlashInfer SM120 design

### 2026-08-08 — [amendment] Calibrated FP8 KV writes use the exact path
- What: correcting the preceding #357 entry, non-unit calibrated scales bypass the fused Triton writer and use the existing torch quantize-and-store path; unit-scale FP8 keeps the fused path.
- Why: real Qwen3-32B writes showed backend FP8-cast byte differences beyond division rounding alone, so calibrated serving must prefer the gate oracle over an unproven fused fast path.
- Refs: issue #357; G4 E-KV calibrated re-bake; `kairyu/{engine/core/kv_pool.py,kernels/paged_kv_write_gpu.py}`

### 2026-08-08 — [amendment] FP8 KV scaling is correctly rounded
- What: fused batched and ragged FP8 KV writers perform calibrated BF16-to-FP8 scaling through explicit FP32 round-to-nearest division; the gate audits dequantization error in FP64.
- Why: Triton's approximate division crossed E4M3 byte boundaries for a small fraction of real calibrated writes, while FP32 audit multiplication introduced false bound excesses near zero.
- Refs: issue #357; `kairyu/kernels/paged_kv_write_gpu.py`; `bench/fp8_kv_g4_ekv_bench.py`; `tests/gpu/test_paged_kv_write_gpu.py`

### 2026-08-08 — [amendment] FP8 KV scales bind per layer and K/V kind
- What: FP8 pools and fused writers apply validated static per-layer K/V scales; a disjoint long-context amax pass emits a checkpoint- and data-bound calibration artifact for the unchanged G4 E-KV thresholds.
- Why: unit scale underused E4M3 range on small-value layers and caused the retained cache-quality and output failures.
- Refs: issue #357; FlashInfer SM120 amendment; `kairyu/engine/core/{kv_pool,kv_scale_calibration}.py`; `bench/fp8_kv_g4_ekv_bench.py`

### 2026-08-08 — [amendment] Generic EP combine is local and degree-invariant
- What: correcting the preceding #359 entry, reverse all-to-all's complete returned rows are combined locally in fixed FP32 slot order, with no redundant combine collective.
- Why: FP32 rank-partial reduction removed BF16 boundaries but retained ownership-dependent addition grouping and doubled combine traffic.
- Refs: issue #359; PR #448 Fable 5 review; M16 D3; `kairyu/models/moe_parallel.py`

### 2026-08-08 — [amendment] Generic EP combine crosses one rounding boundary
- What: generic all-to-all MoE keeps rank-owned weighted partials FP32 through the all-reduce and casts once afterward; EP2/4/8 ownership partitions are regression-pinned.
- Why: casting every rank partial to BF16 before reduction made identical prompts depend on the configured EP degree.
- Refs: issue #359; M16 D3; `kairyu/models/moe_parallel.py`; `tests/unit/test_moe_precision.py`

### 2026-08-08 — [amendment] INT8 bias addition preserves oracle rounding
- What: the fused INT8 kernel now uses explicit round-to-nearest bias addition after the rounded scale product, with a compact FMA-sensitive seed and production-width GPU coverage.
- Why: compiler FMA fusion skipped the CPU oracle's intermediate FP32 rounding and caused one real-checkpoint token disagreement despite exact q/scale and int32 accumulation.
- Refs: issue #356; PR #447 Fable 5 review; `kairyu/kernels/quant_gemm_gpu.py`; `tests/gpu/test_quant_kernels.py`

### 2026-08-08 — [amendment] INT8 scale division is exact and zero-row aligned
- What: dynamic INT8 scale calculation now clamps amax before correctly rounded division, matching the CPU oracle for ordinary and all-zero rows.
- Why: exact activation rounding also requires an exact scale; an approximate reciprocal one operation earlier could move the same half-integer ties.
- Refs: issue #356; PR #447 Fable 5 review; `kairyu/kernels/quant_gemm_gpu.py`; `tests/gpu/test_quant_kernels.py`

### 2026-08-08 — [amendment] INT8 activation rounding matches its exact oracle
- What: dynamic INT8 activation quantization now uses round-to-nearest division before tie-to-even integer rounding, with production-width GPU regression coverage.
- Why: approximate Triton division crossed half-integer ties and the real-checkpoint gate exposed a model-output disagreement that small random tolerances missed.
- Refs: issue #356; m14 real-checkpoint parity amendment; `kairyu/kernels/quant_gemm_gpu.py`; `tests/gpu/test_quant_kernels.py`

### 2026-08-08 — [amendment] Real checkpoints gate INT8 and W4 formats
- What: one pinned BF16 Qwen2.5-1.5B reference now drives 64x16 teacher-forced top-64 parity arms for public compressed-tensors INT8, AWQ, and GPTQ checkpoints; raw replay binds exact ABIs, revisions, completeness, and the measured tie floor.
- Why: tiny random weights and GEMM error metrics could not expose subtle real-checkpoint packing or zero-point convention errors at model output.
- Refs: issue #356; m14 real-checkpoint parity amendment; `bench/{parity_hf,quant_checkpoint_parity_bench}.py`

### 2026-08-08 — [amendment] NVFP4 exposes measured projection accuracy levers
- What: opt-in projection selectors can observe group-16 activation saturation, use per-token NVFP4 scaling, or convert checkpoint NVFP4 weights once to FP8 runtime storage after TP/EP slicing; a strict M-A1 companion records accuracy, resident memory, and saturation curves.
- Why: the retained 235B NVFP4 gate failure had no clipping visibility or bounded way to trade memory and activation precision while preserving the default fused path.
- Refs: issue #355; m14 accuracy-profile amendment; `kairyu/quant`; `bench/g4_ma1_nvfp4_accuracy_bench.py`

### 2026-08-08 — [amendment] Streaming results carry text deltas end to end
- What: completion results expose offset-validated text deltas; native, process-split, and OpenAI-compatible streams feed server and orchestration consumers without rebuilding or rescanning cumulative text, while legacy cumulative text remains lazily compatible.
- Why: flattening each token into a growing string at several serving layers caused quadratic copies and GIL-held allocation work per request.
- Refs: issue #338; m8 wire v2; `kairyu/{outputs,engine,entrypoints/server,orchestration}`

### 2026-08-08 — [amendment] Learned draft heads enter native serving
- What: public native config wires EAGLE-3/MTP checkpoints into stateful target-hidden capture, O(T+k) cached rollout, exact accepted-row commit, per-source acceptance evidence, and request-local speculative pipeline dependencies; missing Radix hidden history safely degrades to target decode.
- Why: the trained heads and batched teacher path existed, but public serving exposed only low-acceptance n-gram proposals and serialized unrelated pipeline work behind a global barrier.
- Refs: issue #330; m17 A30-A32; `kairyu/engine/core/{draft,model_runner,spec_runner,scheduler}.py`; `kairyu/models/{eagle,mtp,llama}.py`

### 2026-08-08 — [amendment] Mixed engine steps share one ragged model chain
- What: native-ragged backends append one-token decode rows after prefill rows and execute one flat model forward; device-owned feedback uses existing decode slots, while unsupported backends, pure decode, and attention-DP's per-forward distributed protocol retain their prior paths.
- Why: colocated mixed steps otherwise traverse every model layer twice even though ragged prefill already supports a causal one-token continuation.
- Refs: issue #317; m13 D1; A12 #360; `kairyu/engine/core/model_runner.py`; `tests/{unit,gpu}/test_batched_prefill*.py`

### 2026-08-07 — [amendment] Exact KV scoring leaves routing locks
- What: exact hash-set identities and lifecycle revisions are captured under their respective locks, scored outside, then revalidated before a vector is accepted; approximate candidates do not prune the independent exact score space.
- Why: an 8K-token prompt across a large fleet performed O(replicas x prompt blocks) membership work while blocking event ingestion and replica lifecycle changes.
- Refs: issue #348; m10 A31; `kairyu/orchestration/{kv_index,kv_routing}.py`; `tests/unit/test_{kv_event_recovery,kv_routing_adapter}.py`

### 2026-08-07 — [amendment] Serving micro-overheads stay bounded
- What: metrics path templates use a bounded LRU; SSE separator escaping uses one ASCII fast-path scan; AUTO direct streaming drops its duplicate prefix scan; chat body limiting forwards validated chunks without replay buffering; unset AUTO usage fields are excluded at serialization call sites.
- Why: these repeated regex/string/Pydantic operations and duplicate request-body storage add small but compounding latency or peak memory on the public serving path.
- Refs: issue #349; m7 D4; m11 accounting/trace; `kairyu/entrypoints/server/{middleware,app,protocol}.py`; `kairyu/{sse,orchestration/orchestrator}.py`

### 2026-08-07 — [amendment] Native admission is bounded by the RoPE table
- What: real-model builders default an omitted `max_model_len` to `max_position_embeddings` and reject a larger override before loading or serving, across single-rank, TP, EP, P-D, and process-split paths.
- Why: a fixed resident RoPE table must fail cleanly at admission rather than let an out-of-range gather raise on CPU or poison a CUDA context.
- Refs: issue #332; m12 D2; `kairyu/engine/kairyu_backend.py`; `tests/unit/test_model_loader_backend.py`

### 2026-08-07 — [amendment] Model and stream hot paths avoid repeated setup
- What: rotary cos/sin is precomputed to the model position limit and gathered per step; scaled dense RoPE uses the fused kernel; invariant Q/K guards and kernel imports leave layer loops; tokenless nonterminal updates and wire materialization are skipped; sampler conversion makes one mutable copy.
- Why: decode repeatedly launched position/trig kernels, evaluated invariant guards/imports, and constructed cumulative output work even when no request produced a token.
- Refs: issue #332; m8 D1/D2/D6; m12 D2; `kairyu/models/{layers,attention}.py`; `kairyu/engine/{engine_loop,zmq_backend}.py`

### 2026-08-07 — [amendment] Grammar-free CUDA sampling is batched
- What: mixed decode rows now batch grammar-free temperature/min-p/top-k/top-p filtering and stateless Gumbel draws, use the maximum bounded top-k prefix for nucleus filtering, preserve exact full-vocabulary top-p when no finite top-k exists, and leave only grammar rows on the scalar CPU matcher path.
- Why: one non-greedy row previously forced per-row fp32 copies, kernel launches, and full-vocabulary sorts across the complete decode batch; a Qwen3-size 151,936-token vocabulary at B=16 measured 1.13 ms batched versus 14.21 ms scalar on the same GPU.
- Refs: issue #326; m8 D2; `kairyu/engine/core/{model_runner,sampler}.py`; `kairyu/kernels/sampling_gpu.py`; `tests/gpu/test_batched_sampler_gpu.py`

### 2026-08-07 — [amendment] Eager FlashInfer decode reuses the no-D2H planner
- What: ordinary tensor eager decode and graph-shape eager fallback now pass authoritative scheduler lengths into the existing fixed-shape metadata pack and FlashInfer fast planner after one stock initialization per shape.
- Why: eager and out-of-coverage graph steps rebuilt dynamic indices and copied schedule inputs to the host every decode step even though graph replay already had a validated no-sync path.
- Refs: issue #329; m17 A29; `kairyu/engine/core/{attention/{flashinfer_gpu,flashattention_gpu},model_runner,step_executor}.py`; `kairyu/models/attention.py`

### 2026-08-07 — [amendment] Batch I/O yields to interactive pressure
- What: batch output/error transactions now use bounded background JSONL writes; route and filesystem-worker store I/O runs off-loop, downloads stream fixed chunks, and configured SLO pressure pauses new batch lines without preempting running work.
- Why: per-line flushes, local metadata reads, and whole-file downloads blocked the gateway event loop, while the fixed batch pool kept dispatching as interactive TTFT risk rose.
- Refs: issue #342; m7 D7; m10 D3/A8; m11 D6; `kairyu/batch/{store,postgres_store,worker}.py`; `kairyu/entrypoints/server/batch_routes.py`

### 2026-08-07 — [amendment] Admission waits ahead of native sequence capacity
- What: an explicit wait timeout lets a single local built-in backend split configured `/v1/*` concurrency into its advertised active sequence budget plus a bounded FIFO; omission, multi-model, pool, and unknown backends retain immediate saturation 429, and queue depth/rejections are observable.
- Why: synchronized bursts were occupying server slots while queuing invisibly inside the engine scheduler, and saturation above a hand-set cap flapped immediately to 429 instead of absorbing a bounded burst.
- Refs: issue #341; m7 D5; `kairyu/entrypoints/server/{middleware,settings,app}.py`; `kairyu/engine/{backend,kairyu_backend,zmq_backend,vllm_backend}.py`

### 2026-08-07 — [amendment] Native pool validation shares immutable contracts
- What: in-process and process-split native backends now publish an exact-type `request_validation_key` from model path, effective string tokenizer source, and `max_model_len`; equivalent pool members validate synchronously once, while custom tokenizers, subclasses, and per-member async preparation stay independent.
- Why: typed prompt validation repeated the same tokenizer work on the serving loop for every equivalent replica even though the existing pool seam could safely deduplicate immutable contracts.
- Refs: issue #347; m10 A19; `kairyu/engine/{kairyu_backend,zmq_backend}.py`; `tests/unit/test_{kairyu,zmq}_backend.py`

### 2026-08-07 — [amendment] Streaming prefix roots publish at first token
- What: prefix-aware streams publish their root immediately before the first backend result is yielded, retain it after later cancellation, and promote a warm hit to its full prepared chain only on normal completion; pre-first-result failures remain unadvertised.
- Why: waiting through the entire decode left concurrent related requests looking cold even though prefill KV already existed, while dispatch-time speculation would require rollback state to avoid poisoning the index.
- Refs: issue #344; m10 D6; `kairyu/orchestration/replica.py`; `tests/unit/test_kv_routing.py`

### 2026-08-07 — [amendment] Deployment pools can enable prefix-aware routing
- What: `PoolSpec` now exposes default-off `prefix_index`; the production builder constructs the existing bounded approximate `PrefixIndex` per opted-in static or discovered pool and passes it to `ReplicaPool`.
- Why: validated KV-aware placement existed only for programmatic callers and benchmarks, so a production DeploymentSpec could not select the warm replica across related sessions.
- Refs: issue #343; m7 D3; m10 D6; `kairyu/deploy/{spec,builder}.py`; `docs/deployment.md`

### 2026-08-07 — [amendment] Cumulative engine state advances by deltas
- What: overlapping step snapshots now freeze append-only outputs by reference plus length; TP/EP sync uses epoch/length and output/page tails; output presentation caches cumulative token/logprob content while exposing immutable-length internal views; KV allocation pages are concatenated once.
- Why: copying and rescanning each request's full generated history on every step made host work quadratic in completion length.
- Refs: issue #324; m5 D2; m8 D1/D6; `kairyu/engine/{core/{frozen_prefix,step_input,scheduler,radix_kv,spec_runner},engine_loop,kairyu_backend}.py`

### 2026-08-07 — [amendment] Engine presentation leaves the core step thread
- What: all public prompt paths now prepare text outside the engine step owner; production detokenization and in-process delivery or process-wire event/msgpack work use one bounded serial output lane, overlapping at most one next raw step while ROUTER I/O stays on its socket thread; HF production requires native `DecodeStream`.
- Why: burst tokenization and cumulative presentation work serialized the next model step and inflated TTFT tail latency; bounded one-ahead execution preserves stop-string scheduler safe points without adding a new protocol or configurable queue.
- Refs: issue #327; m8 D1/D6; `kairyu/engine/{engine_loop,kairyu_backend,zmq_backend,core/engine_service,tokenizer}.py`

### 2026-08-07 — [amendment] Prefill host preparation scales by physical pages
- What: ragged prefill now validates cross-row KV ownership in one physical-page pass and packs its typed metadata into one fresh buffer, using a single pinned H2D copy on CUDA.
- Why: token-by-token ownership maps, row-pair intersections, and pageable per-field uploads added avoidable TTFT before every batched prefill model call.
- Refs: issue #322; m13 D1; `kairyu/engine/core/prefill.py`; `tests/unit/test_batched_prefill.py`

### 2026-08-07 — [amendment] Prefill budget forms bounded work-conserving cohorts
- What: native schedulers now share post-decode prefill budget across a bounded leading cohort of equal-priority partial prompts (two by default), expose `max_num_partial_prefills`, and permit one cache-safe, completion-only immediate-successor admission past a KV-blocked head; deferred P-D may retain one peer token to overlap its copy.
- Why: serial long-prefill chunks and unbounded head-of-line KV blocking inflated small-prompt TTFT, while unrestricted skip-ahead could starve the head or create recompute-preemption thrash.
- Refs: issue #328; m11 D6/A11; `kairyu/engine/core/scheduler.py`; `tests/unit/test_{scheduler,scheduler_waiting_queue}.py`

### 2026-08-07 — [amendment] Direct chat activates predictive TTFT admission
- What: added opt-in `server.ttft_slo_s`; validated direct interactive chat now includes known ingress elapsed time, atomically admits, batch-defers only on routes attesting running-decode isolation (otherwise sheds), observes the first successfully sent visible SSE delta, releases leases at the outer ASGI boundary, and exports the controller snapshot through six Prometheus gauges.
- Why: the validated F5c controller had no production call site, so requests predicted to miss the TTFT SLO still consumed serving capacity and reduced SLO-goodput.
- Refs: issue #340; `kairyu/entrypoints/server/{app,middleware,metrics,settings,slo}.py`; `kairyu/deploy/spec.py`; `tests/server/test_slo_admission_integration.py`; `docs/design/m11-product.md`

### 2026-08-06 — [progress] Pipeline-depth hypothesis closes as a measured negative
- What: retained twelve Qwen3-32B TP4 HTTP diagnostics and CPU plan-shape evidence, reverted the experimental deep strict-decode tail and metric, and kept the existing two-step admission/prefill horizon.
- Why: the initial depth-five candidate regressed throughput by 3–4%, its cohort-preserving refinement returned only to noisy parity, and the historical 35.98% result compared depth one with five rather than isolating benefit beyond depth two.
- Refs: issue #318; `bench/results/issue-318-pipeline-depth-qwen3-32b-rtxpro6000-2026-08-06/`

### 2026-08-06 — [amendment] CUDA graph coverage moves to readiness-time defaults
- What: supported real CUDA models now resolve omitted decode policy to CUDA graphs, size graph batch/page coverage from serving limits, pre-capture every bucket before single/TP/attention-DP EP readiness with bounded rank preflight and rollback, and export monotonic single/process/pool eager fallbacks to Prometheus; CPU, P-D, replicated EP, custom, and MLA paths stay eager by capability.
- Why: lazy first-use capture blocked live traffic and the former batch/page defaults left common requests permanently eager without an operator-visible counter.
- Refs: issue #320; m17 A26-A28; `kairyu/engine/{kairyu_backend,core/{step_executor,model_runner,worker}}.py`; `kairyu/entrypoints/server/metrics.py`; `tests/gpu/test_ep_attention_dp_cuda_graph_startup_gpu.py`

### 2026-08-06 — [progress] Offloaded-route endpoint context no longer trusts FastAPI's id cache
- What: `_OffloadedRequestBodyRoute` now computes the validation-error endpoint context once from the live endpoint at route build instead of calling FastAPI's `_extract_endpoint_context`, and the flaky ZMQ cancelled-submit test waits on wall-clock time for the prompt worker thread.
- Why: FastAPI 0.139.0 caches endpoint context by `id(func)` without holding a reference, so a garbage-collected endpoint's recycled address serves a stale context (observed as `chat_completions` reported for `probe` in CI); the ZMQ test's bare loop yields gave the worker thread no time on loaded runners.
- Refs: PR #423; `kairyu/entrypoints/server/app.py`; `tests/server/test_prompt_offload.py`; `tests/unit/test_zmq_backend.py`

### 2026-08-06 — [progress] CPU microbenchmark gate retries timing-ratio noise
- What: the same-run CPU gate re-runs a benchmark (≤3 attempts total) when its only failing checks are timing ratios; structural failures and benchmark errors still fail immediately, and thresholds plus the report schema consumed by the nightly series are unchanged.
- Why: shared-runner jitter tripped `scheduler.priority_speedup` (0.87 vs the 0.9 floor) on a docs-only PR; a real hot-path regression keeps failing on every attempt, so retries preserve the smoke alarm while removing the flake.
- Refs: PR #422 CI run 31094215441; `scripts/cpu_microbench_gate.py`; `tests/verification/test_cpu_microbench_gate.py`

### 2026-08-06 — [design] PROGRESS.md restructured for minimal session-start tokens
- What: archived the verbose Current Status to `docs/progress/archive/status-2026-08-06.md` and all older Change Log entries to `docs/progress/archive/change-log.md` (both verbatim); PROGRESS.md now holds a compact Product/Current Status/Change Log. Size budgets and archiving live in `.claude/rules/progress-log.md` + `docs/progress/archiving.md`, enforced by a SessionStart hook running `scripts/check_progress_size.py`.
- Why: PROGRESS.md and the progress rules are loaded every session; at ~514 KB they burned the context they were meant to save.
- Refs: `docs/progress/archive/`; `.claude/settings.json`; `scripts/check_progress_size.py`

### 2026-08-06 — [progress] Repeated token SSE shapes bypass per-token model construction
- What: added stream-owned byte encoders for repeated role-less Chat content, unfinished legacy Completion text, and Responses output-text deltas. Constant JSON envelope fragments and per-index prefixes are retained per stream, only the dynamic scalar is serialized per chunk, and Starlette receives bytes directly. Strict predicates keep first-role, non-string/non-integer, logprob, tool, finish, usage, trace, error, and terminal shapes on their existing serializers. Hoisted fallback exclusion sets and byte-for-byte golden tests bind field order, usage omission/null, multiple indices, empty/control/Unicode content, SSE line separators in both constants and deltas, malformed boundaries, and operation counts. Same-process 100,000-iteration diagnostics measured 7.804→0.570 µs (13.69x) for Chat, 4.625→0.569 µs (8.13x) for Completions, and 3.847→0.753 µs (5.11x) for Responses.
- Why: constructing and validating nested Pydantic models plus serializing the complete roughly 200-byte envelope for every small token delta imposed avoidable CPU work and allocations on all dominant streaming paths.
- Refs: issue #334; `kairyu/entrypoints/server/{sse_encode,app,responses_service}.py`; `tests/server/test_sse_encode.py`

### 2026-08-06 — [progress] Retained benchmark evidence gains a strict tracked catalog
- What: added `bench/results/index.json` with one path-sorted record for all 63 Git-tracked top-level artifacts, including nullable recorded gate/date/source-commit/hardware/verdict metadata and authoritative bundle summaries. A package-owned strict parser and checkout validator reject duplicate/non-finite/non-canonical JSON, schema or slug drift, unsafe or symlinked paths, untracked summaries, and missing/stale Git coverage while excluding ignored and untracked runtime output. The read-only verifier now runs in portable CI alongside the entrypoint and wheel-boundary checks.
- Why: retained flat files and bundles were discoverable only through narrative documentation, so trend tooling and evidence audits could silently miss new roots or invent provenance for heterogeneous historical artifacts.
- Refs: issue #383; `bench/results/index.json`; `kairyu/bench/results_index.py`; `scripts/verify_bench_results_index.py`; `tests/bench/test_bench_results_index.py`; `bench/README.md`; `docs/benchmarks.md`

### 2026-08-06 — [progress] Benchmark profiling gains shared, bound trace artifacts
- What: added a lazy `kairyu.bench.profiling` context that preserves caller-owned warm-up, synchronization, and measured scopes while mapping explicit CPU/CUDA activities without fallback. Migrated the two checkout benchmark users and all nine GPU-test users from direct profiler construction. `serving_bench.py --profile` now records one CPU-only local-client range and publishes a private, strict-JSON, 64 MiB-bounded `*.client.pt.trace.json` sidecar through same-filesystem temporary export and exclusive hard-link publication; the paired UTC-microsecond result binds its relative name, format, size, SHA-256, activity, local scope, target exclusion, and diagnostic-only status. Both members publish without overwriting a concurrent winner, and result-publication failure rolls back only its exact untampered trace. The example serving reporter rejects profiled diagnostics rather than mixing them into timing comparisons. Disabled/help/core-wheel paths do not import torch, and missing torch, unavailable CUDA, unsafe or colliding paths, invalid exports, and publish races fail closed.
- Why: ad hoc profiler setup had divergent options and no common artifact identity or overwrite policy, while a serving-client trace could otherwise be misrepresented as remote server or GPU evidence.
- Refs: issue #380; `kairyu/bench/profiling.py`; `bench/{serving_bench,future_token_bench,batched_prefill_qwen}.py`; `tests/{unit/test_bench_profiling.py,gpu}`; `docs/benchmarks.md`; `bench/README.md`; `docs/gpu-runbook.md`

### 2026-08-06 — [design] Structured output gains paired schema-conformance evidence
- What: added the dedicated `structured` benchmark suite with a package-owned, SHA-256-addressed Draft 2020-12 corpus spanning nested, recursive local-reference, enum, regex-pattern, and union schemas. Each selected item and sampling seed retains one counterbalanced constrained/control pair whose wire bodies differ only by `response_format`; strict outer-response and completion JSON parsing, an independently pinned `jsonschema` evaluator, and type-sensitive exact-answer scoring separate request acceptance, JSON validity, schema conformance, task success, malformed output, endpoint-reported token coverage/deltas, and diagnostic latency. Explicit structured-schema 400/422 responses remain measured conformance failures, while unrelated HTTP/transport/malformed-envelope failures withhold the complete claim. Valid HTTP 200 refusals, content-filtered empty messages, and unexpected call payloads remain accepted non-JSON/task-failure evidence instead of becoming environment faults. Raw arms and all derived metrics are cross-validated; exact config-selected corpus IDs, rejected control constraints, fixed cache/source identities, installed evaluator distributions, and subset incomparability are rebound at runner, report, and fresh-history boundaries. Fresh scoreboards recompute the detailed claim, schema-1 history proves then strips it, and stored detached claims are rejected. The suite, wheel fixture, cache manifest, evaluator resources/dependencies, docs, and example config are isolated from the frozen Core/quantization matrices.
- Why: the previous one-object smoke assertion did not exercise common schema features, compare enforced generation with a matched control, quantify malformed output or token cost, or retain replayable evidence capable of detecting semantic-quality and reporting regressions.
- Refs: issue #375; `kairyu/bench/{structured,types,aggregate,history}.py`; `kairyu/bench/adapters/structured_output.py`; `kairyu/bench/fixtures/structured-output.jsonl`; `examples/bench_structured.yaml`; `tests/bench/test_bench_structured_output.py`; `docs/benchmarks.md`

### 2026-08-06 — [design] Chat evaluations gain grouped sampling-sensitivity evidence
- What: extended generative benchmark adapters so `attempts > 1` runs one ordered consecutive-seed sweep per source item, retaining strict child outcomes beneath that item rather than flattening correlated repeats. Complete rows recompute seed means, sample SD, range, and unbiased `1-C(n-c,k)/C(n,k)` pass@k for declared binary adapters; incomplete or tampered matrices withhold the sensitivity summary, and ordinary Wilson intervals plus configuration A/B are disabled for multi-attempt evidence. Raw source-item identities, canonical parent/pair status and reason, denominators, realizable binary pass@k margins, and fresh history labels are cross-validated. Runner, report, and history boundaries also bind target seed/mode/temperature plus the attempt budget to retained methodology; schema-1 history omits the derived label after proving it against the raw pair, while a new protocol marker preserves old agentic multi-trial records. SciCode independently reruns each seed's whole sequential problem chain. Targets can select an explicit temperature or `sampling_mode: recommended`, which omits temperature/top-p/top-k/min-p/repetition-penalty so endpoint generation defaults may apply while explicitly declining remote attestation. External harnesses fail closed when those chat-only policies cannot be forwarded, and SWE-Bench Pro requires one attempt. The default single-attempt target serialization, target-config fingerprint input, temperature-zero request bytes, and absent seed remain byte-shape compatible.
- Why: one deterministic chat call cannot quantify sampling noise or pass@k, but flattening repeated seeds would understate uncertainty and corrupt paired statistics. Explicit omission is also required to measure the model-owned generation defaults added by issue #351 without falsely claiming that a remote server applied them.
- Refs: issue #371; `docs/benchmarks.md`; `examples/bench_{fugu,core}.yaml`; `kairyu/bench/{sampling,types,aggregate,config,cli,config_ab,history}.py`; `kairyu/bench/adapters/{base,ifeval,scicode,swebench_pro,terminal_bench,tau_bench}.py`; `tests/bench/test_bench_{sampling_sensitivity,config,config_ab,history,agentic_conditions,scicode_sequential,runner}.py`

### 2026-08-06 — [design] Quantization formats gain task-level accuracy gates
- What: added the fixed `quantization` suite and `kairyu bench quant-sweep` command for dense BF16, FP8, INT8, AWQ, GPTQ, NVFP4, and dense BF16 with FP8-E4M3 KV across GSM8K, MMLU, IFEval, and GPQA Diamond. Every target declares an exact weight/compute/KV classification plus a distinct served-manifest digest; six complete configuration A/B artifacts provide the sole paired statistical core, and all 24 independently toleranced Newcombe lower-bound gates must pass without task or scheme averaging. The dedicated atomic JSON/Markdown artifact binds the clean source, hash-chain record, raw pairs, deployment declarations, comparator runtime, and versioned protocol. Reports explicitly distinguish operator declaration from remote attestation and keep FP8-E4M3 KV scoped to external or experimental deployments because native Kairyu still rejects it after the failed quality bake. Optional quantization identity serializes only when declared, preserving existing Fugu/Core fingerprints and judge-calibration reloads.
- Why: kernel parity and throughput measurements do not establish downstream task quality, while a flat score table without exact source, deployment, item, and pair bindings could hide missing formats, incomparable requests, or post-hoc tolerance choices.
- Refs: issue #372; `docs/design/issue-372-quantization-sweep.md`; `examples/bench_quantization.yaml`; `kairyu/bench/{quant_sweep,config_ab,cli,store,types}.py`; `tests/bench/test_bench_{quant_sweep,cli_quant_sweep,config,aggregate,runner,store}.py`

### 2026-08-06 — [design] Configuration A/B gates bind deployments to paired task evidence
- What: added `kairyu bench compare` for non-inferiority gates between two explicit served configurations. The command validates the complete scoreboard hash chain, reopens and content-binds every selected raw pair, requires identical full-data methodology, runtime, request policy, and item sets, then applies paired Newcombe method-10 intervals to independent binary outcomes, deterministic paired bootstrap intervals to bounded scores, or whole-problem clustered bootstrap intervals to SciCode's dependent sequential sub-steps. Versioned binary/cluster declarations live in the run fingerprint, and the fixed SplitMix64 sampler removes Python `randrange()` drift. Each benchmark has an independent percentage-point tolerance and one-sided 95% lower-bound verdict; candidate-owned JSON/Markdown artifacts retain the deployment, source, comparator runtime, protocol, policy, and raw-evidence hashes without changing the history index. Runs can declare the operator-owned served configuration label and SHA-256 in YAML or paired CLI flags, with the non-attestation boundary stated explicitly.
- Why: token parity and published-number comparisons cannot quantify whether a quantization, KV-cache, expert-parallel, or speculation configuration causes a task-level regression. Paired item evidence reduces avoidable variance, while source, methodology, deployment, and raw-result bindings keep a toleranced PASS from silently comparing different experiments or anonymous redeployments.
- Refs: issue #365; `docs/design/issue-365-config-ab.md`; `kairyu/bench/{config_ab,config_compare,history,store,cli,types,runner}.py`; `kairyu/bench/adapters/{base,scicode}.py`; `tests/bench/test_bench_{config_ab,config_compare_stats,cli_config_compare,history,store}.py`

### 2026-08-06 — [design] Nightly CPU ratios gain trailing-median history
- What: added a scheduled/manual main-only workflow that reuses the portable CPU microbenchmark report, records eight fixed same-process ratios in a canonical SHA-256-chained JSONL series, and alerts when any current ratio is at least 15% below the median of up to seven preceding records after five compatible warmup observations. Exact method/workload/runtime/policy segmentation, strict JSON and chain validation, attempt-specific Actions provenance, newest-valid recovery including failed regression runs, and upload-before-enforcement keep the baseline fail closed. Absolute timings, router/process-wire data, and sampler overlay measurements stay outside the cross-run comparator while the unchanged same-run source gate remains binding.
- Why: one-shot CI artifacts could satisfy absolute smoke bounds while a persistent cross-commit regression went unnoticed. Same-process ratios are the portable signal available on shared CPU runners; the 15% alert intentionally remains a reproducibility prompt rather than formal product-performance or TTFT evidence, including for the observed noisy sampler-append ratio.
- Refs: issue #377; `scripts/cpu_perf_series.py`; `.github/workflows/nightly-cpu-perf.yml`; `docs/design/issue-377-nightly-cpu-perf.md`; `tests/unit/test_cpu_perf_series.py`; `tests/unit/test_nightly_cpu_perf_workflow.py`

### 2026-08-06 — [amendment] IDE tool-call compatibility fails closed without rolling back public APIs
- What: hardened the pending IDE integration so the single-call hint merges into one existing system message, Responses does not inject it twice, and native Llama/Qwen parsing is selected by the executed server-owned template rather than the request model name while request-controlled Llama syntax branches are rejected. JSON arguments reject duplicate keys and non-finite values; Qwen XML additionally rejects duplicate parameters, residual markup, and violations of the supported top-level type/required/additional-property constraints. Only supported final upstreams receive typed `parallel_tool_calls`. The documented legacy `SamplingParams.extra_args.parallel_tool_calls` path remains supported when it is the sole source, all new public dataclass/function parameters are appended to preserve positional binding, and the unauthenticated Docker examples publish on host loopback only. Documentation now scopes multimodal and schema claims to the implemented IDE boundary.
- Why: the original branch could manufacture a second system role rejected by valid tokenizer templates, reinterpret ordinary bare JSON as a tool call for unrelated models, accept malformed model output as executable arguments, expose unauthenticated examples beyond localhost, misstate existing multimodal support, and accidentally roll back positional and legacy Python compatibility while adding typed propagation.
- Refs: PR #409; `kairyu/entrypoints/{chat_template,server/chat_service,server/responses_service}.py`; `kairyu/{engine,orchestration}/`; `tests/server/test_{openai_api,tool_call_protocol}.py`; `tests/unit/test_{kairyu_backend,openai_backend,orchestration_request,ide_client_examples}.py`; `docs/{deployment,ide-clients}.md`

### 2026-08-05 — [progress] IDE agents complete model-native tool loops
- What: added OpenAI-compatible `parallel_tool_calls=false` enforcement, model-native Llama and Qwen3-Coder tool-call parsing, isolated gpu02 deployment examples, and Cline/Continue configuration guidance. A Qwen3-Coder-30B-A3B-Instruct TP1 replica on gpu02 completed a real Cline 4.1.3 Act-mode `read_file` → `attempt_completion` loop through an SSH local forward; both requests returned 200. The implementation was rebased onto current `main`, retaining its tokenizer-owned template auto-discovery, and 396 focused API/template/deployment tests pass.
- Why: IDE clients need deterministic single-call policy enforcement and must receive structured OpenAI tool calls even when a checkpoint emits its documented model-native format. Using the checkpoint's own template and protocol avoids model-ID spoofing and prompt-only heuristics, while the local forward separates API compatibility from editor SOCKS transport behavior.
- Refs: `kairyu/entrypoints/server/{protocol,chat_service,app}.py`; `tests/server/test_openai_api.py`; `docs/ide-clients.md`; `examples/ide-client/{gpu-replica,qwen3-coder-gpu-replica}.yaml`; gpu02 request IDs `051659ecd29746a0`, `13780c2db86d437f`

### 2026-08-05 — [design] B3 scoreboards become source-bound cross-commit history
- What: added a canonical, append-only, hash-chained `scoreboards.jsonl` per benchmark suite. Each record snapshots validated run metadata, source-attested complete PairResult digests/summaries, and the complete scoreboard under `(local harness git_commit, run fingerprint)` with dirfd-pinned fixed-lock concurrent append, atomic replace, directory durability, strict JSON/chain/schema/status/tamper validation, canonical-byte idempotence, and conflicting-evidence rejection. Resume rejects stable environment/source drift before download; observed source or evaluator drift is durably tainted before evidence rewrite. Fingerprints bind adapter, runner, cache, aggregation, judge, referenced assets, score-time dependencies, and resolved external-harness content/PATH ownership; opaque request bodies persist only as hashes. Full-suite records retain unresolved agentic/code cells under structured per-cell withholding policies, while an available inspected Docker image ID/platform can make generated-code cells eligible. The public `kairyu bench compare-runs BASE CANDIDATE` command remains read-only and renders candidate-minus-baseline scores only across exact suite/fingerprint/Python/runtime/target/benchmark/methodology/denominator matches with `allowed` policies, revalidates Wilson intervals, suppresses synthetic-fixture and withheld deltas, and does not turn a negative delta into a policy exit code.
- Why: per-run directories alone cannot expose regressions across commits, while CWD-derived commits, mutable path references, source or evaluator drift, executable shadowing, runtime mismatch, credential-bearing tracked config, or structurally mismatched cells can produce unsafe artifacts or confident but false trend claims. A local harness commit also cannot attest a redeployed target at the same URL, so artifacts and reports state that boundary explicitly.
- Refs: issue #369; `docs/design/issue-369-cross-commit-scoreboards.md`; `kairyu/bench/{history,store,runner,aggregate,compare,cli}.py`; `tests/bench/test_bench_{history,run_compare,cli_compare,runner}.py`

### 2026-08-05 — [progress] A12 passes the real Qwen3-32B TP8 gate
- What: executed the strict direct-path batch-invariance operator on the pinned Qwen3-32B checkpoint and eight RTX PRO 6000 Blackwell GPUs. The retained 38-row raw stream passed all 28 independently derived checks, including exact four-arm answers, all-rank FlashInfer/CUDA-graph execution, scheduler/cache causality, and complete source/checkpoint/hardware identity; retained verification and raw replay also passed. Portable CI then exposed a constructor-bypassing TP follower fixture that had not mirrored the new shared decode capability field, so the fixture now preserves the production alias and a regression proves capability-gapped passive runners fall back before the batch call.
- Refs: issue #360; PR #411; source commit `d5044c2`; raw SHA-256 `c42797f18b8db7b9c87ab9203a3abc2bf0b26aaa2264d9ce42024fbcc5bf8b88`; `docs/design/issue-360-batch-invariance.md`; `tests/unit/test_tp_sampling_authority.py`

### 2026-08-05 — [design] A12 binds greedy answers across batch, cache, and chunk shapes
- What: added a checkout-only, path-only Qwen3-32B TP8 gate that runs one fixed 129-token prompt cold in a real 32-request cohort, cold alone after six-request deterministic radix eviction, warm alone after an independent 128-token page-aligned cache probe, and cold alone in a fresh runtime with exact `32, 32, 32, 32, 1` prefill chunks. All four native 32-token responses must match exactly by token ID, raw vocabulary piece, and final text. The raw contract additionally binds all-rank FlashInfer prefill/decode counters, CUDA-graph capture/replay, scheduler cohorts, cache usage, source/checkpoint/hardware identity, strict isolated startup, canonical replay, and exclusive no-overwrite artifacts. Ordinary decode now shares the existing model/layer/backend capability gate with speculative verification, so unsupported MLA/custom stacks serialize before model or KV mutation while complete list-only implementations remain batched.
- Why: identical prompts can otherwise cross shape-dependent attention, graph, batching, cache, and chunked-prefill paths and return different greedy answers, making production incidents and load-dependent quality comparisons unreproducible. Causal native schedule/cache evidence distinguishes a real shape comparison from copied configuration or scenario labels, and pre-call capability selection avoids unsafe retry after partial KV writes.
- Refs: issue #360; `docs/design/issue-360-batch-invariance.md`; `bench/batch_invariance_bench.py`; `kairyu/bench/batch_invariance.py`; `kairyu/engine/core/model_runner.py`; `docs/gpu-runbook.md` §9.8d

### 2026-08-05 — [amendment] B7 formal evidence requires isolated direct-path startup
- What: formal `run-native`, `assemble`, `verify`, and `replay` now require a fresh direct checkout-script process whose initial interpreter is launched with `python -I -B` (or the runbook's `uv run --frozen` equivalent); a plain interpreter is rejected instead of receiving an unsafe late restart. Before exposing the checkout on `sys.path`, the wrapper creates an unpredictable, exclusive 0700 `tempfile.TemporaryDirectory` under `/tmp`, assigns it to `sys.pycache_prefix`, and performs its tracked-clean/import-artifact preflight. Module invocation is excluded from the declared inventory and remains only as an unsupported, non-evidence `--help` convenience; its evidence output is unsupported and non-publishable. The selected interpreter, locked `.venv`, site bootstrap, and installed packages are explicit TCB. The contract also distinguishes the 151,669-entry tokenizer ID domain from the model's padded 151,936-wide logits head and validates every prompt/response ID against `TOKENIZER_VOCAB_SIZE = 151669`.
- Why: module startup, ambient import paths, or repository bytecode could execute before an ordinary in-process source check, while conflating tokenizer IDs with padded logits rows could admit 267 non-tokenizer IDs as answer evidence. Isolated direct-path bootstrap, a wrapper-owned fresh cache namespace, and pre-import clean/artifact checks close the former gap without pretending to attest the interpreter; the explicit tokenizer boundary closes the latter.
- Refs: issue #373; `docs/design/issue-373-kv-answer-equivalence.md`; `bench/README.md`; `docs/gpu-runbook.md` §9.8c; `kairyu/bench/kv_equivalence.py`

### 2026-08-05 — [amendment] B7 fails closed on sealed F4b quality and executable drift
- What: review hardened the B7 companion without changing a parent gate. F4b now replays its owning sealed performance-plus-quality verifier and content-binds separate performance manifest/raw and quality manifest/raw snapshots. F2d's router sibling is also snapshotted around replay; every parent row index, schema-fixed numeric/boolean value, and F2d raw/router row schema is validated with exact JSON types. Every repository bytecode/import artifact outside the locked `.venv` is rejected, loaded checkout namespaces must resolve to tracked `HEAD` files, and source/checkpoint/profile identities are checked again after shutdown. The pure contract additionally enforces the exact 1,024-token prompt, pinned vocabulary bounds, nonempty native pieces, JSON type-sensitive frozen values, immutable single-read raw replay, and exclusive no-overwrite output with post-write byte verification.
- Why: a performance-only F4b parent, crafted cached bytecode, JSON bool/int/float coercion, out-of-vocabulary evidence, or a write/read replacement race could otherwise manufacture a passing child without proving the reviewed native cache path and answer.
- Refs: issue #373; `docs/design/issue-373-kv-answer-equivalence.md`; `bench/kv_answer_equivalence_bench.py`; `kairyu/bench/kv_equivalence.py`; `tests/verification/test_kv_answer_equivalence_bench.py`; `tests/verification/test_kv_equivalence.py`

### 2026-08-05 — [design] B7 binds cache-warm answers to exact cache-cold answers
- What: added an additive five-cell companion gate in the exact order `f2c-tp2`, `f2d-tp2`, `f4a-tp4`, `f4a-tp8`, `f4b-tp4`. `run-native --cell` selects fixed geometry and `assemble` requires one raw for every named cell. Before GPU construction each child replays and content-binds immutable snapshots of its exact parent through the owning verifier. The cells require five distinct nonces but one clean-source identity and one fully hashed Qwen3-32B checkpoint: all 17 shards, both weight rollups, architecture/config, six exact metadata files, and the separate weight index. F2d uses `aggregate-common-f2c` because its parent contains no checkpoint assertion. Each persistent native runtime retains and rehashes exact prompt IDs and requires identical IDs, pieces, and text from cold and complete-hit 32-token greedy requests. Radix cells require an independently probed exact full-prompt hit; DRAM cells separately prove pressure-interval offload and following warm-interval restore with zero fallback/ownership failures. Clean-source preflight rejects import shadows and non-`HEAD` loaded repo modules; source, checkpoint, and profile identities are rechecked after shutdown. Canonical replay hashes and parses the same immutable raw bytes; assembly, manifest verification, strict no-overwrite behavior, and the checkout boundary fail closed. All existing parent schemas, gates, diagnostics, and verdicts remain unchanged. Portable tests make no real-model, GPU-cache, or GPU-DRAM claim.
- Why: performance-only KV gates can pass while radix reuse, page-table state, or DRAM spill/restore silently corrupts an answer. Same-runtime cold/warm equality isolates that corruption signal from the known BF16 near ties and topology/shape differences in existing cross-arm diagnostics.
- Refs: issue #373; `docs/design/issue-373-kv-answer-equivalence.md`; `bench/kv_answer_equivalence_bench.py`; `kairyu/bench/kv_equivalence.py`; `docs/gpu-runbook.md` §9.8c

### 2026-08-05 — [design] Core MMLU ranks exact teacher-forced continuations
- What: added a native `kairyu_continuation` completions mode and an ordered forced-token engine path that returns each candidate token's raw pre-processor log probability while genuinely conditioning later positions on the forced prefix. Token boundaries and the processed prompt/continuation IDs are tokenizer-owned and exact; EOS can be scored as data, process-isolated and in-process engines preserve the intent, malformed backend evidence fails closed, and remote/OpenAI/vLLM paths cannot silently drop it. A separate benchmark request/response protocol validates finite aligned evidence, probes capability before dataset fan-out, and distinguishes explicit unsupported targets from authentication, endpoint, retry, schema, and fixed-tokenizer failures. Core MMLU now ranks `" A"` through `" D"` with a stable tie break, requires four distinct single-token candidates, and retains every candidate score and token proof. A full core run is consequently 58,028 calls. The final portable collection is 5,436 selected tests; exact full-suite runs plus all 19 post-run hardening selections are green, with 83.86% measured combined coverage, all 64 benchmark entrypoints, and the isolated wheel boundary verified.
- Why: generated letters and generated top-k membership cannot provide the conditional probability of an arbitrary candidate and can hide quantization or cache quality changes. Exact teacher forcing makes multiple-choice ranking auditable and supplies the reusable primitive required by later likelihood-based quality gates without weakening the ordinary completion contract or manufacturing scores on incapable targets.
- Refs: issue #368; `docs/design/issue-368-loglikelihood.md`; `docs/design/issue-367-core-evals.md`; `kairyu/engine/core/sampler.py`; `kairyu/entrypoints/server/app.py`; `kairyu/bench/adapters/{base,mmlu}.py`; `docs/benchmarks.md`

### 2026-08-05 — [design] Deterministic core suite adds GSM8K, MMLU, and IFEval
- What: added a separate `core` benchmark suite with the immutable full GSM8K test set, the full 57-subject MMLU test set, and all 541 IFEval prompts/834 instructions/25 checker IDs. Fugu remains the unchanged default 11-row suite and alone receives its published comparison. Core uses deterministic exact-match or programmatic scoring, retains IFEval's four strict/loose aggregates, pins the Google checker and English Punkt resources, repairs missing Punkt assets without rewriting valid normalized data, and records per-instruction evidence. Pinned IFEval keys 1122 (`#`) and 1129 (`!`) receive one documented dataset-consistency amendment because upstream replaces their punctuation with independently random ASCII letters; Kairyu scores the exact requested character. Dataset/resource readiness is unified across every cache consumer, and package-resource SHA-256 identities bind each core prompt/parser/scorer plus every vendored IFEval module before resume. Real pinned downloads yielded exactly 1,319/14,042/541 rows, all 834 IFEval instructions replayed with random fallbacks disabled, 64 checkout-only entrypoints and 12 wheel fixtures verified, and the 5,301-test portable CI sequence passed with 83.75% coverage.
- Why: frequent quality regression checks need a cheap, judge-free, Docker-free, vision-free suite without changing Fugu methodology or presenting zero-shot generated-letter MMLU as canonical. Exact data, scorer, dependency, auxiliary-resource, and implementation identities prevent silent drift or reuse of evidence scored under different semantics.
- Refs: issue #367; `docs/design/issue-367-core-evals.md`; `docs/benchmarks.md`; `kairyu/bench/adapters/{gsm8k,mmlu,ifeval}.py`; `kairyu/bench/_vendor/ifeval/`; `examples/bench_core.yaml`

### 2026-08-05 — [amendment] Issue #364 withdraws FP32 final logits after the paired A1/A2 result
- What: completed the clean-commit `ac589fb` paired experiment. A1 passed for both `model` and `float32`; A2 failed for both against the immutable 1004/1024 shared-reference floor, with `model` TP2/4/8 agreement of 1009/1004/999 and `float32` agreement of 1008/1003/1002. All cells retain zero substantive disagreements and zero missing observations. A `main@6cff10f` TP8 control has raw tokens, logprobs, and reference rows exactly equal to the `model` arm. The retained conclusion is `evidence_valid=true`, `feature_ready=false`, and `quality_classification=mixed`; the public `logits_dtype` construction/deployment option is withdrawn and is not shipped, while the evidence operator and negative result remain.
- Why: the existing A2 readiness floor must not be weakened or bypassed to ship an option whose paired result is mixed. The exact main-control match distinguishes pre-existing baseline drift from an experiment regression without turning a formal A2 failure into a pass.
- Refs: issue #364; measurement commit `ac589fb67452173f45f23d9107af313c2b79cc17`; main control `6cff10f9f39c5d114d9a21875ea0c6e460d4cf32`; G2 A1/A2; `docs/design/issue-364-fp32-logits.md`; `bench/results/issue-364-fp32-logits-a1-a2-2026-08-05/`

### 2026-08-05 — [design] FP32 final logits are a build-time opt-in measured against A1/A2
- What: implemented `logits_dtype="model"|"float32"` for every native real-model construction path. The default retains the exact existing output-head operation; the opt-in produces FP32 at the final dense GEMM before greedy selection, raw-logprob capture, penalties, and filtering while preserving model/KV/weight dtype, tied weight identity, state dictionaries, distributed ownership, and the no-host-sync boundary. Process/rank attestation, health reporting, synthetic BF16 near-ties, sampling boundaries, and CUDA graph/profiler tests bind the option. A narrow same-reference paired A1/A2 matrix, separate from issue #365's generic A/B framework, will retain per-mode and per-TP evidence and report positive, mixed, unchanged, or negative quality results without rewriting the historical formal artifacts. Real-model measurement remains in progress.
- Why: BF16 final logits are compatible rather than defective, but their roughly 0.125-nat resolution can move near-tied argmax and filtering decisions. Only an opt-in final-GEMM experiment measured against the existing self-agreement and tie-gap gates can determine whether FP32 output improves those decisions without weakening compatibility or correctness.
- Refs: issue #364; M12 D2/D5/D6; M8 D2; M14 D3; G2 A1/A2; `docs/design/issue-364-fp32-logits.md`

### 2026-08-05 — [amendment] Gumbel sampling keeps full seed width and a 52-bit open uniform
- What: retained the complete unsigned 64-bit output of the per-position seed mixer and replaced the 32-bit additive Wang counter with an odd-stride, XOR-keyed SplitMix64 finalizer implemented through signed-int64 wraparound and explicit logical shifts. Both CPU and CUDA now transform 52 random mantissa bits through a float64 midpoint in the strictly open interval `(0, 1)`. The stochastic CPU/CUDA sequence intentionally changes together; greedy, filtering, penalties, grammar support, raw logprobs, replay ownership, and the no-host-sync CUDA result boundary are unchanged. Independent uint64 word oracles, high-bit and known-low32-collision fixtures, 151,936-word uniqueness, adjacent-seed lag correlations, a deterministic 100,000-position three-category distribution, open-endpoint tail checks, CPU/CUDA raw-word and token parity, and a CUDA profiler gate bind the new stream. On SM120 the 151,936-vocabulary helper measured 0.169 → 0.201 ms and the complete filtered sampler 0.580 → 0.618 ms; single-thread CPU full sampling measured about 20% slower. The bounded CPU cost is accepted to preserve the same genuinely 52-bit transform and tail on both execution paths.
- Why: truncating the 64-bit position seed to 32 bits created identical noise vectors in long generations, adding seed and vocabulary index made adjacent seeds shifted views, and the 24-bit midpoint limited the flipped-Gumbel winning tail to `2^-25`. A keyed full-width permutation removes both structural aliases while the 52-bit open uniform extends that tail to `2^-53` without mutable RNG state or a device-to-host scalar read.
- Refs: issue #354; m8 D2; `kairyu/engine/core/sampling_types.py`; `kairyu/kernels/sampling_gpu.py`; `tests/unit/test_sampling_rng.py`; `tests/gpu/test_sampling_rng_parity_gpu.py`

### 2026-08-05 — [amendment] CPU and CUDA sampling share one stateless random draw
- What: made the existing stateless Gumbel-max tensor sampler canonical for CPU, structured-output, and CUDA execution after one shared filtering implementation. CPU alone materializes the selected scalar, skips normal-path fallback construction, and reuses a bounded immutable offset cache; CUDA retains its branchless device result and established stochastic sequence. The former CPU/structured `torch.multinomial` sequence intentionally changes, while grammar masking and acceptance remain on CPU, greedy behavior is unchanged, and fixed-logit filter, penalty, minimum-token, structured-support, raw-logprob, replay, and CPU/CUDA parity gates cover the boundary.
- Why: selecting an execution path or enabling a grammar whose current legal set is unchanged must not silently switch RNG algorithms and alter the sampled trajectory. One stateless draw also keeps replay, TP ownership, overlap, and P-D behavior independent of mutable generator offsets without folding the separate seed-quality work into this change.
- Refs: issue #353; m8 D2; `kairyu/engine/core/sampler.py`; `kairyu/kernels/sampling_gpu.py`; `tests/unit/test_sampler.py`; `tests/gpu/test_sampling_rng_parity_gpu.py`

### 2026-08-05 — [amendment] Special-token visibility is per request and logprobs expose raw pieces
- What: threaded `skip_special_tokens` through native in-process and process-split request state, native/fallback detokenization, and the Kairyu upstream capability profile while preserving old custom-tokenizer signatures. Opposite policies are isolated and the actual EOS/stop-terminating ID remains invisible under both. Native selected and top rich-logprob token strings now use a lazily cached sparse-ID-safe raw vocabulary table with pre-allocation amplification bounds, negative-index exclusion, and decoded fallback for valid padded LM-head IDs while remote adapters preserve provider token strings; native bytes remain the flag-sensitive single-ID decode, and legacy completion offsets advance by those decoded contributions rather than raw marker lengths.
- Why: callers explicitly requesting special-token text must not lose it through native admission, shared tokenizer state, a default-policy stream, or process transport, while logprob labels must retain exact vocabulary markers without weakening terminal no-retraction semantics or crashing on padded output-head IDs.
- Refs: issue #362; D1 in `docs/design/m8-engine-cpu.md`; D3 in `docs/design/m9-truthful-api.md`; `docs/deployment.md`; `kairyu/engine/tokenizer.py`; `kairyu/engine/engine_loop.py`; `kairyu/engine/openai_capabilities.py`; `kairyu/engine/core/engine_service.py`

### 2026-08-05 — [amendment] Final decoding never rewrites streamed text
- What: made accumulated incremental detokenizer output authoritative at request completion. Native streams may append an optional `finalize_suffix()` delta; the HF adapter excludes special IDs and decodes only its un-emitted replacement-bearing terminal window, while Toy appends nothing. If Rust rejects a later token because malformed held bytes make its reconstructed prefix disagree with published text, the adapter appends only the held replacement suffix and resumes in a fresh native stream. The full-prefix compatibility path reuses its last safe cached candidate without another decode, and legacy push-only custom streams retain a guarded historical terminal decode. Finalization is idempotent, and the engine independently preserves the published prefix before terminal stop matching. Valid multilingual, emoji, contextual replacement-token, CTC, special-token, and byte-fallback sequences retain full-decode parity whenever the full decode extends published text; malformed bytes can only append their held replacement suffix.
- Why: a final full-history re-decode could disagree with already-delivered incremental chunks, duplicating or dropping SSE text or causing the no-retraction stop matcher to fail the request.
- Refs: issue #361; D1 in `docs/design/m8-engine-cpu.md`; `kairyu/engine/tokenizer.py`; `kairyu/engine/engine_loop.py`

### 2026-08-05 — [progress] Portable CI catches large CPU hot-path regressions
- What: added one Python 3.12 CPU-only CI job that sequentially runs the six existing scheduler queue, radix eviction, operation queue, sampler penalty-state, process-wire, and router-latency benchmarks. Fixed short workloads, pinned native thread counts, same-process legacy ratios, structural coalescing checks, deterministic wire growth, and a strict 10 ms router p99 budget feed one fail-closed JSON report with seven-day diagnostic retention. The ratio limits are intentionally loose and absolute shared-runner timings remain non-binding.
- Why: these hot paths already had fast source-checkout benchmarks but no pull-request coverage, allowing large CPU regressions to land without consuming any GPU budget.
- Refs: issue #378; `scripts/cpu_microbench_gate.py`; `.github/workflows/ci.yml`; `bench/README.md`

### 2026-08-05 — [progress] Bounded serving profiles expose Python and CUDA stalls
- What: added a source-checkout `profile_server.py` launcher for finite, non-overwriting py-spy speedscope and Nsight Systems process-tree captures of `kairyu serve`. It preserves argv boundaries, follows process-isolated engines and TP ranks, separates Nsight load delay from the capture window, discards the recorded environment, and emits an exact dependency-free dry run. The GPU runbook now pins profiler preflight, workload, lifecycle, provenance, privacy, and interpretation rules; profiled runs remain diagnostic rather than formal performance evidence.
- Why: serving regressions could be attributed to high-level stages after #379, but there was still no reproducible way to inspect Python/GIL scheduling or correlate host gaps with CUDA work without ad hoc commands and unsafe attachment/overwrite behavior.
- Refs: issue #381; `scripts/profile_server.py`; `tests/unit/test_profile_server.py`; `docs/gpu-runbook.md` §0.1

### 2026-08-05 — [amendment] Direct native stage traces make serving regressions attributable
- What: extended trace v2 with opt-in, cumulative request-observed `tokenize`, `queue_wait`, `schedule`, `prefill`, `decode_step`, `detokenize`, and `sse_write` scalar events across in-process, process-split, and Kairyu-replica paths. The serving benchmark now explicitly requests the trace, validates terminal SSE/envelope/identity/scalar structure without retaining raw detail, reports nearest-rank stage p50/p99, and preserves complete/partial/missing/invalid plus per-stage coverage denominators. Trace-off avoids clock reads and per-request stage state; older or external targets remain missing/partial rather than zero-valued.
- Why: TTFT/TPOT alone could identify a regression but could not distinguish prompt preparation, scheduler delay, model execution, detokenization, or HTTP delivery. The measurements are deliberately non-exclusive request observations so batching and overlap are not misrepresented as GPU ownership or additive end-to-end time.
- Refs: issue #379; `docs/design/observability-trace-contract.md`; `bench/serving_bench.py`; `kairyu/engine/{backend,engine_loop,kairyu_backend,zmq_backend,openai_backend}.py`; `kairyu/entrypoints/server/app.py`

### 2026-08-05 — [progress] Issue #333 valid v2 diagnostic does not support dominant GIL contention
- What: completed the sole fresh v2 Qwen3-32B TP4 ABBA diagnostic from clean source `65ba4779c118d534f1e34a1a4dcb1b579cbcfe73`. All 15 binding checks pass independent raw replay, 512/512 synchronized measurement requests succeeded once, all four containers exited gracefully with code zero and were removed without force, and the selected GPUs returned exactly to the run-start idle baseline. Paired process/in-process TTFT-p99 ratios were 0.9189755344057482 and 0.9219867334510442; their 0.9204811339283963 median is above the predeclared ≤0.90 material line, so the report-only classification is `no_material_reduction` and the dominant process/GIL-contention hypothesis is `not_supported`. Median goodput and TTFT-p50 ratios were 1.086201492163829 and 0.9282808350389853. The raw/manifest SHA-256s are `21ba4789cf34fcf4beab5ab5862952574693f80c8b608c2cede002da05088925` and `bd321c32654c0602d6e799167d16a7b375b8edf5dea5bd95faa899acde24286c`.
- Why: the issue requested one real TP4 process-isolation diagnostic to determine whether API/engine GIL contention was the dominant TTFT-p99 cause. The valid net-backend movement was beneficial but smaller than the declared material magnitude, so the evidence does not support that dominant-cause hypothesis. The result does not isolate pure GIL cost and does not claim an A6 verdict.
- Refs: issue #333; m8 D6; `bench/results/issue-333-proc-http-qwen3-32b-rtxpro6000-2026-08-05/`; `bench/issue_333_proc_http_bench.py`; `docs/{gpu-runbook.md,design/m8-engine-cpu.md}`

### 2026-08-05 — [amendment] Issue #333 separates concurrent repeatability from evidence integrity
- What: retained the first fail-closed four-cell trial strictly as an invalid observation: source `ad11a322b77337547ac34dc1717586c40f76fd8b`, paired-median process/in-process TTFT-p99 ratio 0.9454156693989547 with no classification, raw SHA-256 `6b726d0076f55226b9d50370a27f8f06e486ca9c53088192f003891507b95a13`, and manifest SHA-256 `15526fc3efec8b5a12ae5dba3e7c5c54c9cc3729987e005dd5a838cf8f94d393`. Before any rerun, bumped the operator evidence schema to v2, retained strict A6 HTTP/SSE/prompt/usage/terminal structure and well-formed per-row digest metadata, and added binding per-cell uniqueness for all 163 response IDs plus all-four output parity for each of the four serialized ShareGPT warm-ups. All six c128 measurement-cell output-hash agreement counts and rates are explicit non-binding diagnostics; the predeclared 0.90 report-only TTFT-p99 interpretation is unchanged. The next fresh full ABBA is the sole v2 rerun and is retained regardless of direction; v1 raw rows are never rewritten as v2.
- Why: the two same-arm repeat pairs in the discarded trial agreed on only 29/128 and 41/128 completion hashes, so exact all-four equality could not distinguish backend semantics from fresh-server concurrent scheduling variation. Serializing the 128-request burst would remove the very contention being diagnosed. TTFT ends at first-token arrival, while the full 128-token continuation hash is downstream of the backend treatment; conditioning the TTFT result on that hash was post-treatment selection. Raw rows retain digest metadata but not decoded bytes, so the contract does not claim independent output-hash recomputation.
- Refs: issue #333; m8 D6; source commit `ad11a32`; `bench/issue_333_proc_http_bench.py`; `tests/verification/test_issue_333_proc_http_bench.py`; `bench/results/issue-333-proc-http-qwen3-32b-rtxpro6000-discarded-v1-2026-08-05/`; `docs/{gpu-runbook.md,design/m8-engine-cpu.md}`

### 2026-08-05 — [amendment] Issue #333 rejects force-cleaned or non-idle GPU cells
- What: changed the four-cell TP4 diagnostic lifecycle to stop each measured container with a bounded graceful timeout, re-attest its immutable launch ID and zero non-OOM exit, retain shutdown logs, and remove it without force. Pre-start and post-removal evidence now requires no selected-GPU compute applications, zero utilization, and memory exactly equal to a stable per-GPU run-start idle baseline, with bounded retry for transient NVML query failures. Stop or removal failures still trigger best-effort forced recovery but invalidate the cell and prevent a shard from being written.
- Why: an interrupted trial demonstrated that `docker rm -f` could remove every visible process and allocation while two GPUs continued executing orphaned kernels at 100% utilization. Process-list-only quiescence therefore admitted contaminated subsequent cells, and a successful `docker stop` return code alone could conceal Docker's timeout SIGKILL.
- Refs: issue #333; m8 D6; `bench/issue_333_proc_http_bench.py`; `tests/verification/test_issue_333_proc_http_bench.py`; `docs/{gpu-runbook.md,design/m8-engine-cpu.md}`

### 2026-08-04 — [progress] Serving prompt preparation leaves the event loop
- What: added separate bounded, cancellation-safe request-body and prompt CPU lanes; moved chat/route JSON and Pydantic construction, the fused content walk and render, backend tokenization/serialization, orchestration planning, and dynamic role/MoA prompt construction off-loop. Exact prepared requests now survive direct, role, MoA, replica, cancellation, and ZMQ generation boundaries without duplicate validation or stale publication. Pre-placement metrics expose ingress, validation, routing, and admission phases while typed FastAPI/OpenAPI/error and synchronous entrypoint contracts remain unchanged.
- Why: request-sized synchronous preparation serialized concurrent serving on the event loop and inflated ingress-to-selection and TTFT p99; sharing one worker queue also allowed later body parses to convoy earlier TTFT-critical render/tokenizer work.
- Refs: issue #336; `kairyu/async_thread.py`; `kairyu/entrypoints/server/{app,chat_service,metrics,request_body}.py`; `kairyu/engine/{backend,kairyu_backend,openai_backend,zmq_backend}.py`; `kairyu/orchestration/{orchestrator,replica,conductor,moa}.py`; `tests/{unit,server,compat}`

### 2026-08-04 — [amendment] Process-isolated TP owns and attests its complete rank tree
- What: enabled real-model tensor parallelism in `kairyu-proc`; delayed public topology until child startup attested the configured degree; placed the non-daemon service and ranks in one private POSIX session with an API-parent lease; made the Linux API a child subreaper that confirms every forced descendant is reaped; added fatal launcher heartbeat/readiness and a 120-second worst-step timeout; bounded add/abort/heartbeat/shutdown sends; and added a fail-closed TP4 fresh-server ABBA diagnostic with exact A6 traffic, imported-source, checkpoint, GPU, backend, config, and process-tree evidence plus a predeclared report-only 0.90 TTFT-p99 ratio interpretation.
- Why: the proposed separate-GIL diagnostic could not run at the A6 TP4 topology, and merely spawning a service without positive topology, failure, and ownership contracts could strand GPU ranks or publish a healthy endpoint after a follower died. The diagnostic must also distinguish integrity from performance and prevent a partial run from confirming or denying the hypothesis.
- Refs: issue #333; m8 D6; `kairyu/engine/{config_validation,kairyu_backend,zmq_backend}.py`; `kairyu/engine/core/{engine_service,worker}.py`; `bench/issue_333_proc_http_bench.py`; `tests/{unit,dist,bench}/`; `docs/gpu-runbook.md`

### 2026-08-04 — [amendment] Checkpoint chat templates become fail-closed defaults
- What: added metadata-only loading of dedicated and tokenizer-config HF chat templates plus named special tokens; resolved the effective tokenizer and compiled every deployment renderer before constructing backends; required a template or model-scoped `legacy_chat_models` membership at every production, lower-level chat, and offline-chat boundary; made completion-only low-level construction log an explicit missing-policy warning while rejecting chat before dispatch; rejected unverified special-token variables instead of silently dropping BOS/EOS; preserved rendered-template ownership through an in-process/ZMQ typed prompt marker and disabled vLLM's completion special-token insertion only for that marker; rejected remote/discovery/orchestration and direct Conductor/MoA derivation paths that cannot preserve the marker; limited automatic legacy selection to DeploymentSpec-built deterministic mocks; made the checked-in VLM overlay opt in explicitly while image requests bypass its text renderer; and replaced the Qwen example's temporary template extraction with direct checkpoint auto-loading. Offline validation now reports the same missing/invalid policy, while Llama-3 and Qwen bytes and token IDs match Transformers exactly. The final portable split passed 4,908 tests (1,343 benchmark and 3,565 non-benchmark; 201 deselected).
- Why: the implicit role-prefix fallback and empty BOS/EOS context could silently serve prompts in a format the model was never trained on, even when its checkpoint already supplied the authoritative template. Preflight resolution prevents quality corruption and fails before allocating GPU/backend resources.
- Refs: issue #350; m9 D2; `kairyu/engine/{tokenizer,prompt,vllm_backend,zmq_backend}.py`; `kairyu/entrypoints/{chat_template,llm,async_engine}.py`; `kairyu/entrypoints/server/{chat_service,app,responses_service}.py`; `kairyu/orchestration/{orchestrator,conductor,moa}.py`; `kairyu/deploy/{spec,builder,validation}.py`; `kairyu/batch/worker.py`; `deploy/{compose,kind,helm}/`; `examples/serve.py`; `tests/unit/test_{chat_template,tokenizer}.py`; `tests/server/test_chat_template_policy.py`

### 2026-08-04 — [progress] Judge evidence binds prompts and calibration
- What: registered and hashed the exact HLE/CharXiv judge templates plus their parser/generation protocol into the run identity; added concurrent strict-majority judge panels with ordered per-member evidence and fail-closed aggregation; packaged 12 paired, published-gold LLMBar Natural calibration prompts (24 labeled responses) with pinned source and MIT notice; measured both response orders and self/non-self behavior; and bound headline eligibility to fixed promotion gates plus the complete canonical identity of the evaluated run. The complete benchmark suite passes 1,343 tests with no selected skips, and an isolated wheel contains all nine fixtures, the LLMBar notice, and the calibration CLI.
- Why: a benchmark could previously retain the same fingerprint after its frozen judge prompt changed, and one uncalibrated judge could silently introduce position or self-preference bias. Exact identity binding and auditable calibration make judge-backed scores reproducible without allowing weaker exploratory thresholds or stale run artifacts to qualify as headline evidence.
- Refs: issue #376; `kairyu/bench/{calibration,config,judge,judge_prompts,runner,types}.py`; `kairyu/bench/adapters/{base,charxiv,hle}.py`; `kairyu/bench/fixtures/{judge-calibration.jsonl,LLMBAR_LICENSE}`; `tests/bench/test_bench_{aggregate,config,judge,judge_calibration,runner,sampling,tau}.py`; `docs/benchmarks.md`

### 2026-08-04 — [progress] Binary benchmark cells expose Wilson uncertainty
- What: declared the eight Bernoulli benchmark adapters explicitly and added structured two-sided 95% Wilson score intervals only to completed cells when that metric contract, retained 0/1 item outcomes, successes, equal scored/total denominators, total item count, and point estimate all agree; rendered the interval and sample count in Markdown; showed scored/total counts for partial cells; and kept MRCR/Terminal/τ reward metrics, incomplete or inconsistent counts, and legacy artifacts without item evidence interval-free. Stored-run report regeneration and old scoreboards remain compatible through an additive optional cell field. The complete benchmark suite passes 1,285 tests with no selected skips.
- Why: a 20-item smoke score and a 500-item full score previously rendered as the same bare percentage, inviting readers to over-interpret small-sample deltas. Inferring binomial trials from an aggregate score alone would instead attach false confidence bounds to continuous and partial-reward metrics.
- Refs: issue #370; `kairyu/bench/adapters/{base,charxiv,gpqa,hle,livecodebench,livecodebench_pro,longbench_v2,scicode,swebench_pro}.py`; `kairyu/bench/aggregate.py`; `tests/bench/test_bench_aggregate.py`; `docs/benchmarks.md`

### 2026-08-04 — [amendment] Min-token sampling masks stops before selection
- What: added a shared min-token logits processor that masks model EOS and deduplicated valid request/model stop IDs through every CPU and device sampling path, including direct-argmax, batched, overlap, P-D, TP, EP, and speculative rows at their logical output positions; retained raw pre-processor logprobs; excluded only an actual terminating stop token from visible incremental detokenization while preserving token IDs, usage, logprobs, scheduler/KV history, and radix accounting; and rejected `min_tokens > max_tokens`. Focused integration, process-wire, and real-CUDA overlap gates pass. The exact portable CI-equivalent run passed 1,259 benchmark plus 3,441 non-benchmark tests with no selected skips and 86.90% combined coverage.
- Why: EOS or a configured stop token could previously become ordinary model context before the minimum length and drive off-distribution repetition because the scheduler discovered termination only after appending the sampled token. A non-special terminating stop token could also leak into user-visible text even though the token must remain part of authoritative engine accounting.
- Refs: issue #352; m8 D1/D2; `kairyu/engine/core/{sampler,step_input,model_runner,torch_runner}.py`; `kairyu/engine/engine_loop.py`; `kairyu/sampling_params.py`; `tests/{unit,gpu}/`

### 2026-08-04 — [amendment] Model generation defaults preserve request omission
- What: extended model generation defaults from EOS-only parsing to validated `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty`; preserved omitted-versus-explicit intent through HTTP, offline, native distributed/process, OpenAI-compatible, and vLLM adapters; added strict `auto`/`vllm`/`none` deployment and CLI policy; made real-model process restarts reject missing or stale policy metadata; eagerly started process-backed orchestrator workers before deployment readiness; and exposed direct, homogeneous-local-pool, sampled-remote, and per-orchestrator-worker audit records through `/backends`. The final portable Python 3.12 gate passed 1,259 benchmark plus 3,428 non-benchmark tests with no selected skips and 86.88% combined coverage.
- Why: modern instruction checkpoints rely on `generation_config.json`, but concrete API defaults previously erased whether callers omitted a field and native Kairyu therefore diverged from checkpoint and compatible-engine generation behavior. Model defaults must apply only to genuine omissions, while explicit values—including neutral values—remain authoritative and invalid or heterogeneous policy must fail closed.
- Refs: issue #351; m12 D5; `kairyu/models/generation.py`; `kairyu/sampling_params.py`; `kairyu/engine/{kairyu_backend,vllm_backend,zmq_backend}.py`; `kairyu/entrypoints/{cli,llm}.py`; `kairyu/entrypoints/server/{app,chat_service,responses_service,health}.py`; `tests/{unit,compat,server}/`

### 2026-08-04 — [amendment] Native streams bound and coalesce backpressured snapshots
- What: replaced each in-process request's unbounded cumulative-update queue with a terminal-sealed conflating mailbox, retained the first token-bearing snapshot per choice, and drained consecutive non-terminal backlog before later public yields. Physical mailbox backlog is bounded to two snapshots and the pre-consumer error edge to three; successful terminals replace cumulative state, while payload-free errors follow the latest state. Single and `n > 1` tests bind 128-update backlog, TTFT cadence, full text/token/logprob/usage preservation, terminal ordering, and sibling error propagation. A direct ASGI gate blocks the first body send through engine completion and binds one queued terminal, two content chunks, exact usage/finish/DONE, and a constant body-send bound. ZMQ v2 remains unchanged because its wire events are sequenced deltas. The final portable Python 3.12 two-phase run passed all 4,595 selected tests in 10m20s with zero selected skips and 86.77% combined coverage.
- Why: a slow SSE client previously forced one JSON encode/socket write per backlogged engine step and retained an unbounded O(T²)-payload sequence of cumulative text/token/logprob snapshots. Blindly taking the last item would instead lose first-token latency, metering state before errors, or FIFO success when a later unrelated pump failure appended an error.
- Refs: issue #335; m8 D1; `kairyu/engine/kairyu_backend.py`; `tests/unit/test_kairyu_backend.py`; `tests/server/test_openai_api.py`

### 2026-08-04 — [amendment] Dynamic replica clients retain one saturated serving wave
- What: dynamic origin-local clients now snapshot the live replica count when first used and retain `min(64, max(1, ceil(gateway concurrency / live replicas)))` idle connections, with an explicit fallback of eight when gateway concurrency is unbounded. Active connections remain transport-unqueued, expiry remains 30 seconds and same-origin activity driven, and replica removal still closes its client immediately. A real HTTP/1.1 TCP regression gate holds two consecutive 64-request waves at a barrier, requires exact socket reuse, then removes the real backend from its pool and waits for every server handler to observe close.
- Why: with the issue's 128 clients across DP=2, a one-socket idle cap closed 63 connections per replica after every saturated wave and forced the next wave to reconnect. A fixed 64 cap would instead permit 12,800 retained sockets across F1a's 200 replicas; capacity-proportional lazy sizing gives the cited DP case 64 while F1a retains two per origin.
- Refs: issue #345; m10 A18; `kairyu/deploy/builder.py`; `tests/unit/test_k8s_builder.py`; `tests/unit/test_openai_backend.py`

### 2026-08-04 — [amendment] Batched paged-KV live bounds reuse one Triton kernel
- What: made batched row count, page-table width, and contiguous table stride runtime non-specialized arguments while retaining layout, dtype, head geometry, and write mode as compile-time constants; a real-GPU gate now changes both live dimensions, checks exact K/V output, and requires one compilation.
- Why: specializing either the explicit width or its equal contiguous stride kept request batch/page-width changes in Triton's cache key and caused avoidable serving-time JIT cliffs.
- Refs: issue #321; `docs/design/m2-engine.md` §2.3; `kairyu/kernels/paged_kv_write_gpu.py`; `tests/gpu/test_paged_kv_write_gpu.py`

### 2026-08-04 — [progress] Portable CPU coverage gate removes redundant bench work
- What: retained the full Python 3.11/3.12 portable matrix, all 4,579 selected tests, fail-on-skip policy, and the 80% combined coverage gate; only `tests/bench` uses two file-isolated xdist workers, while repeated formal F4b input/seal generation, an unrelated SciCode network fetch, and production-sized intentional waits are reused or bounded without removing their assertions.
- Why: the prior GitHub test step spent about 80% of 16m33s in benchmark tests, while the Qwen/Llama parity file itself was sub-second there. The exact final two-phase Python 3.12 run passed in 10m20s with 86.74% coverage and no selected skips.
- Refs: issue #321; `.github/workflows/ci.yml`; `pyproject.toml`; `tests/verification/test_agentic_kv_tier_f4b_bench.py`; `tests/bench/test_bench_code_adapters.py`; `tests/bench/test_bench_pins.py`; `tests/unit/test_qwen_gpu_detection.py`

### 2026-08-04 — [amendment] Sequential prefill removes per-layer CUDA host drains
- What: Dense and MLA sequential attention now receive one scheduler-owned chunk boundary and writable-suffix decision through an explicit model capability. Cached-prefix writes use a contiguous suffix slice instead of CUDA boolean indexing; auxiliary target capture can opt in and the MTP decoder supplies the same metadata. Models without the capability and arbitrary-position direct calls retain the original mask path. CPU gates forbid tensor scalar reads, preserve partial/full cached KV, and pin the arbitrary-position fallback; a real-CUDA profiler gate compares the complete KV/output result to the mask oracle while rejecting `aten::nonzero`, `_local_scalar_dense`, and any stream synchronization parented by those scalar/mask operations.
- Why: Reading `writable.any()` and `positions[0]` on every layer caused two explicit device-to-host synchronizations, while boolean indexing the writable mask introduced another dynamic-shape `nonzero` synchronization. Qwen3-32B TP4 therefore paid up to 128 explicit drains per single-chunk prefill step before the indexing cost, directly inflating TTFT.
- Refs: issue #319; m12 D2; m17 A1; `kairyu/engine/core/model_runner.py`; `kairyu/models/{attention,llama,mla,mtp}.py`; `bench/draft_quant_qwen.py`; `tests/{unit/test_batched_prefill,unit/test_eagle_mtp,gpu/test_batched_prefill_gpu}.py`

### 2026-08-03 — [amendment] Agentic Fugu adapters match live harness dataset/output contracts
- What: Changed SWE-Bench Pro generation to pass mini-swe-agent an output directory, require its `<output>/preds.json` file before evaluation, and pass that file to the official swebench evaluator. Replaced the nonexistent legacy-registry `terminal-bench@2.1` selector with the official Harbor Hub organization/package `terminal-bench/terminal-bench-2-1`. A real one-task Terminal-Bench 2.1 oracle smoke completed at 1/1 with zero exceptions; all 1,259 benchmark tests pass with 9 deselected.
- Why: The first real Qwen3-32B TP8 suite attempt exposed two contracts that mocked unit tests had accepted incorrectly: mini-swe-agent 2.4.4 created a directory named `preds.json`, causing swebench to raise `IsADirectoryError`, and Harbor 0.17 could not find `terminal-bench@2.1`. Both were development failures rather than model accuracy results or environment skips.
- Refs: `kairyu/bench/adapters/{swebench_pro,terminal_bench}.py`; `tests/bench/test_bench_agentic*.py`; `docs/benchmarks.md`

### 2026-08-03 — [amendment] Qwen TP8 Fugu path preflights its real request contract
- What: Added native, reserved-variable-safe `chat_template_kwargs`, wired the Qwen3-32B example to the exact checkpoint-owned HF template, replaced unsupported `reasoning_effort` fields with Qwen `enable_thinking` controls for target and judge requests, and added a one-token request preflight before any benchmark item. FlashInfer now resolves its exact prefill/decode AOT or JIT modules against live model/KV geometry before readiness; direct virtual-environment entrypoints expose a sibling `ninja`, and FlashAttention delegates its decode preflight. The complete portable suite passes 4,573 tests with 197 deselected. A real RTX PRO 6000 run reported FlashInfer prefill/decode and tensor parallel size 8, accepted both thinking modes, and released all eight GPUs to zero allocated memory after teardown.
- Why: Post-merge validation of PR #314 found three integrated startup gaps: the example had no configured chat template, sent provider-specific fields that native Kairyu does not implement, and could publish readiness before a first-request FlashInfer module/JIT failure. The correction rejects an unusable request contract before measurement without changing benchmark cases or converting dataset-specific skips into framework policy.
- Refs: PR #314; m9 D2; `examples/qwen3-32b-multi-gpu/`; `kairyu/entrypoints/{chat_template,server/}`; `kairyu/engine/core/{model_runner,attention/}`; `tests/{server,unit}/`

### 2026-08-03 — [amendment] Fugu operator owns setup and recognizes official τ³ v1.x
- What: Made the Qwen3-32B Fugu wrapper start or reuse serving, provision `bench`/`bench-agentic` plus a commit-pinned official τ³ v1.0.1 package and task-data checkout, select an immutable Docker code-execution image, and apply Qwen/Fugu sampling defaults so `HF_TOKEN` is its only required per-run setting. Corrected the adapter to treat the official v1.x `tau2` package/CLI identity as τ³ while retaining pre-1.0 τ² as an explicitly incomparable fallback, and to skip before execution when the official alltools sandbox binaries are unavailable.
- Why: Requiring operators to pre-sync harnesses and manually wire the sandbox contradicted the one-command quality-suite contract, while upstream τ³ deliberately retained the `tau2` distribution/CLI name and was therefore being mislabeled as a substitute.
- Refs: `examples/qwen3-32b-multi-gpu/fugu-benchmark.sh`, `kairyu/bench/adapters/tau_bench.py`, `docs/benchmarks.md`, `tests/unit/test_qwen_fugu_example.py`, `tests/bench/test_bench_tau.py`

### 2026-08-03 — [amendment] G4 M-A3 isolates measurement connections and closes by explicit deviation
- What: The M-A3 operator now completes and closes the entire model-probe/serial/graph-warmup HTTP pool before creating a distinct zero-history measurement pool, records exact pool/request ordering, and rejects lifecycle violations during assembly, verification, and raw replay. Same-device int64 sampled tokens now enter persistent decode slots through one vectorized batched D2D copy instead of per-row scalar copies. The earlier 571.542867-versus-449.965–481.865 diagnostic is withdrawn because the arms used incompatible client lifecycles. A corrected fresh-server/fresh-pool diagnostic measured Kairyu 536.690626 versus SGLang 551.731445 completion tok/s/GPU (0.972739x), with a TTFT-p99 ratio of 0.868731. A full-server CUTLASS override reached only 530.616804 tok/s/GPU, so the retained production choice remains FlashInfer `auto`.
  Destination-aliasing compatibility views are staged first rather than entering the no-temporary production fast path.
- Why: Connection-pool history can alter a short synchronized HTTP measurement and must not cross the warmup/measurement boundary. The corrected diagnostic leaves a 2.73% throughput residual while Kairyu has lower TTFT; the product owner explicitly accepts that residual so development can proceed. This closes the M-A3 issue scope as a deviation only: the formal throughput/TTFT thresholds remain 1.0/1.0, the retained formal results remain FAIL, and no diagnostic is relabelled as a formal PASS.
- Refs: issue #168; PR #311; G4 D8; `bench/g4_ma3_sglang_bench.py`; `kairyu/engine/core/model_runner.py`; `tests/{bench,unit,gpu}/`

### 2026-08-03 — [amendment] G4 M-A3 removes replay-side host drains
- What: Replaced the deferred CUDA status sidecar with an unconditional one-scalar CPU/Gloo failure reduction after every rank has enqueued the fixed NCCL token packet; only failures enter the all-rank object diagnostic gather. CUDA-graph replay now carries authoritative scheduler sequence lengths, packs FlashInfer's persistent page metadata with one Triton kernel, and invokes `fast_decode_plan`; compiler/API/layout incompatibility rewrites the buffers through stock planning. The real FlashInfer suite passes 11/11, the status profiler passes 2/2 without `cudaEventSynchronize`, 154 focused CPU checks pass, and a same-trace Qwen3-235B live diagnostic completed 128/128 requests at 571.542867 tok/s/GPU with 1.025656-second TTFT p99, seven fixed captures, and zero eager fallback.
- Why: Rank traces showed the former sidecar synchronizing one CUDA event for about 28.273 ms at every next control boundary, collapsing configured depth 5 to effective depth 1. FlashInfer's stock replay planner independently performed `nonzero` and device-to-host schedule reads that drained queued graph work. The new CPU control reduction overlaps the already-running graph, while device-only replay planning preserves current page metadata without either host drain. The retained clean formal matrix remains FAIL until the complete replacement matrix passes.
- Refs: issue #168; G4 D8; parent `da83bce`; `kairyu/engine/core/{attention/flashinfer_gpu,model_runner,step_executor,worker}.py`; `kairyu/kernels/flashinfer_decode_plan_gpu.py`; `tests/{unit,gpu}/`

### 2026-08-03 — [progress] G4 M-A3 defers fast-packet status validation without rank divergence
- What: The pure-greedy attention-DP status vector now transfers to pinned host memory on the same copy stream and event as deferred public tokens. Driver and passive ranks retain the preceding status sidecar and all resolve exactly one FIFO entry after the next shared control broadcast; shutdown drains the tail. Public commit and later control resolution are idempotent. The CUDA packet path no longer calls `.tolist()`, `.item()`, or `.cpu()`, and the final Gloo reply gather remains removed. Focused CPU protocol tests and a real-CUDA test that forbids eager device scalar/list conversion pass.
- Why: Adversarial review found that directly converting the gathered CUDA status column to a Python tuple synchronized every rank's current stream once per decode step, replacing the removed Gloo wait with another steady host stall. The first deferred draft then used rank-local event readiness to decide whether to validate, which could let one rank raise while another entered the next model collective. Delaying the already-enqueued copy to the next common control boundary removes the eager synchronization while preserving rank-symmetric failure propagation.
- Refs: issue #168; G4 D8; candidate base `a223ff6`; `kairyu/engine/core/{model_runner,worker}.py`; `tests/unit/test_ep_attention_dp_worker.py`; `tests/gpu/test_ep_attention_dp_status_gpu.py`

### 2026-08-03 — [amendment] G4 M-A3 bounds admission overlap and removes one steady Gloo reply
- What: The first clean-commit ten-generation M-A3 matrix completed all 4,630 raw rows with exact request, graph, checkpoint, source, container, and runtime evidence, but retained a formal performance FAIL: the exact median Kairyu/SGLang completion-throughput ratio was 0.783818 and the TTFT-p99 ratio was 1.352633. The next candidate retains configured pipeline depth 5 for pure decode while limiting unresolved waiting, unfinished-prompt, and pending-prefill work to two forwards; a producer arrival during fill stops further schedule-ahead. On the pure-greedy attention-DP path, every rank now carries an explicit success/failure status beside its existing NCCL token slots, so the final per-step Gloo reply gather is removed without weakening all-rank failure detection. Non-fast sampling retains the former Gloo reply.
- Why: All four paired throughput ratios were below one (0.741839/0.798127/0.829296/0.769510), so the result cannot be attributed to OS jitter. Raw structure showed admission fragmentation and repeated depth-five snapshots while pinned SGLang and vLLM retain at most previous/current unresolved batches. The steady path also performed three Gloo object transactions per step even though selected tokens already used a fixed NCCL packet. The amendment removes those two identified differences without changing the formal threshold or relabelling the failed result.
- Refs: issue #168; G4 D8; source commit `55f3a8ca4513e158182d4b9b4a818c24f5ae7b34`; raw SHA-256 `c404507238a33fb520a13995821f6fcfde26fd120fd0766c006608ad3aadbd19`; `kairyu/engine/{engine_loop,core/worker}.py`; `tests/unit/test_{unified_engine_loop,pd,ep_attention_dp_worker}.py`

### 2026-08-03 — [amendment] F1a decouples membership convergence from evidence cadence
- What: Added m10 A35. F1a now gives the 500 ms gateway EndpointSlice loop two discovery intervals after the independent observer first sees old-UID withdrawal—one for worst-case polling phase and one for bounded API-read/reconciliation scheduling—while also capping gateway withdrawal at five seconds from Pod DELETE start. The formal one-second deadline is unchanged; smoke no longer reuses its unrelated 500 ms evidence capture interval as a zero-slack one-poll convergence deadline. Retry-free traffic, zero failures, exact placement/membership joins, no late old-UID placement or reappearance, and offline fail-closed replay remain binding.
- Why: PR #311 run `30760700561` served 420/420 requests with zero retries or failures and 0.879256 ms placement p99, but smoke epoch 0 failed only because complete gateway withdrawal was observed 599.934 ms after the EndpointSlice observer. The same run's epoch 1 took 133.546 ms; the two preceding successful runs measured -173.136/195.990 ms and 482.666/237.569 ms. The former 500 ms gate allowed exactly one configured poll and no Kubernetes-read or scheduling time, making the verdict depend on polling phase. The failed epoch still withdrew from gateway eligibility 2.107497 seconds after DELETE start, well inside the unchanged five-second product bound.
- Refs: m10 D5/A24/A35; PR #311; Actions runs `30759734751`, `30760046332`, `30760700561`; `bench/fleet_churn_bench.py`; `tests/verification/test_fleet_churn_bench.py`; `deploy/kind/f1a/base/gateway.yaml`

### 2026-08-03 — [design] G4 M-A3 adopts request-owned attention-DP and a fixed depth-5 candidate
- What: implemented the bounded Qwen3-235B NVFP4 TP1/attention-DP4/EP4 production path with request-owned attention, KV, sampling, and token packets; grouped direct-NCCL all-gather/reduce-scatter; compatible packed-QKV projection; and an all-rank coordinated CUDA-graph decision. A real-checkpoint four-GPU smoke proved direct NCCL plus capture/replay with no fallback. A same-checkpoint diagnostic measured pipeline depth 5 at 1,971.873 aggregate completion tok/s versus 1,697.000 for depth 1, so the formal Kairyu arm is fixed to depth 5. The fail-closed operator pins SGLang v0.5.16, binds complete start/end checkpoint captures to the observed Docker volume, captures and re-observes Docker/source/GPU/process/runtime state around every live generation, binds a 128-request ShareGPT trace and seven graph buckets, executes two preflights plus four fresh-server pairs, and independently replays exact ratio medians; the formal verdict remains pending.
- Why: M-A1's replicated-attention correctness topology cannot provide the request ownership or overlap needed for a credible saturation comparison. Fixing the selected production candidate before formal traffic, matching per-owner cache/prefill capacity, and binding every timing row to its fresh-server capture interval prevent diagnostic reuse, topology relabelling, warmup leakage, or post-hoc candidate selection from changing the M-A3 result.
- Refs: issue #168; G4 D8; `bench/g4_ma3_{kairyu_server,sglang_bench}.py`; `kairyu/engine/core/{direct_nccl,worker}.py`; `kairyu/models/{moe_parallel,packed_linear}.py`; `docs/gpu-runbook.md` §9.13

### 2026-08-02 — [progress] G4 M-A2 proves EP4 preserves radix reuse
- What: ran the fixed Qwen3-235B NVFP4 EP4 trace from clean commit `d2d33e0472fb3101b680f4085d22e80a4ac7ceca` on 4× RTX PRO 6000. All 512 requests completed, the one logical engine rate was 491,008 cached / 557,056 prompt tokens (0.8814338235), all four ranks reported those exact totals and identical page identities, and the raw trace retained 512 exact `BlockStored` events and 4,128 unique blocks. All 12 binding checks, retained manifest verification, and raw-only replay pass; all eight GPUs returned to zero allocated memory after teardown.
- Refs: issue #167; G4 D7; source commit `d2d33e0`; raw SHA-256 `3920cf30dd7633905594f144d05f34727c641a06b0edc54bde8d2ff3fc5cb532`; `bench/results/g4-ma2-ep-kv-qwen3-235b-rtxpro6000-2026-08-02/`

### 2026-08-02 — [design] G4 M-A2 binds one logical radix truth to every EP rank
- What: wired the replicated-attention EP2/EP4 runner into the production Kairyu backend and status lifecycle, and added an opt-in bounded all-rank first-prefill accounting probe that preserves the inference collective order on diagnostic errors. The formal M-A2 operator fixes Qwen3-235B EP4 at 4,129 BF16 KV pages and serializes the A7-lineage 512-request trace through one persistent radix/scheduler/engine path. It counts terminal engine cache usage once, treats four rank rows only as allocation/page invariance witnesses, retains exact raw radix events and runtime provenance, and provides manifest verification plus raw-only replay. The implementation and deterministic tamper gates are complete; the clean-commit real EP4 capture remains to run.
- Why: replicated attention gives every EP rank the same logical `StepDelta`, but a driver-only hit-rate value could hide rank-dependent page/accounting drift. Separating the one logical rate from bounded rank receipts proves the current EP contract without multiplying tokens, adding a timing/jitter gate, claiming physical KV byte readback, or pre-claiming M-A3 attention-DP throughput. The retained M-A1 formal FAIL remains unchanged.
- Refs: issue #167; G4 D7; `bench/g4_ma2_qwen3_235b_ep_kv_bench.py`; `kairyu/engine/{config_validation,kairyu_backend}.py`; `kairyu/engine/core/worker.py`; `kairyu/entrypoints/server/health.py`; `docs/gpu-runbook.md` §9.12

### 2026-08-02 — [amendment] G4 makes fused operands lifecycle-safe and quantifies memory debt
- What: prepared fused-MoE operands are now invalidated before any module device/state mutation, and a forward is marked successful only after its kernel, all-reduce, and shared expert complete. Review also quantified the deliberately retained scale-layout duplicate at 6.609375 GiB/EP2 rank and 3.3046875 GiB/EP4 rank, plus the current validated `o_proj` loader's 854 MiB/1,273 MiB maximum temporary allocation above final EP2/EP4 residency. The earlier claim that inner-axis slicing itself stalled was withdrawn: an isolated 188-slice EP2 probe completed in 4.105 seconds, while prior incomplete full loads did not isolate a cause.
- Why: prevent stale device or scale generations from executing, retain canonical checkpoint/state-dict correctness, report measured memory costs honestly, and defer block-owned scale serialization and source-at-a-time GPU staging as independent P2 optimizations rather than extending this progress PR.
- Refs: issue #166; G4 D2/D3/D6; `kairyu/models/moe_parallel.py`; `tests/unit/test_ep_nvfp4_lifecycle.py`

### 2026-08-02 — [progress] G4 M-A1 validates the adopted EP4 execution path
- What: ran a diagnostic one-prompt, one-token Qwen3-235B NVFP4 EP4 forward on four RTX PRO 6000 GPUs. All ranks loaded their 32 contiguous experts and K=2,048 attention-output shard, produced one token, and reported 94/94 successful fused-MoE forwards, 94/94 successful row-parallel BF16 attention-output forwards, and zero unresolved meta tensors. The launcher shut down cleanly and all eight GPUs returned to zero allocated memory.
- Why: validate the implemented EP4 loading, FlashInfer fused expert ABI, NCCL reductions, and runtime provenance path without spending another full run on an EP2 formal gate that the fixed position-0 diagnostic already proves cannot pass.
- Refs: issue #166; G4 D2/D3/D6; diagnostic artifact `/tmp/kairyu-issue166-ep4.MLipVZ/ep4-smoke.json`

### 2026-08-02 — [amendment] G4 separates implementation progress from formal closure
- What: clarified that a non-zero M-A1 arm remains a binding formal failure and can never support closing issue #166, while an implementation-only progress PR may merge with the issue left open when all applicable repository and GitHub CI checks are green, the failed formal result is reported, and no threshold or fail-closed condition is relaxed.
- Why: retain useful measured implementation progress without misrepresenting the unchanged 1,014/1,024, zero-substantive acceptance gate or conflating that product gate with the pass/fail status of the PR's executable tests.
- Refs: issue #166; G4 D5/D6; `docs/gpu-runbook.md` §9.11

### 2026-08-02 — [amendment] G4 M-A1 adopts fused local experts and reference TP reduction boundaries
- What: corrected the earlier 2026-08-02 progress entry's implied single global MoE output cast. TensorRT-LLM 1.2.1 finalizes each owner rank's routing slots in fp32, casts that rank partial to BF16, then performs a BF16 all-reduce. Kairyu now implements that boundary, stable BF16-logit top-k with lower expert-ID tie resolution, selected-only fp32 routing softmax, layer-global FC1/FC2 input scales, FlashInfer CUTLASS fused local NVFP4 experts with no all-to-all, and a row-parallel NVFP4 attention `o_proj` whose BF16 rank partials are all-reduced. The real EP2 position-0 diagnostic improved from 54/64 exact with 9 substantive disagreements before fused compute, to 57/64 with 6 after fused compute, and to 60/64 with 3 after the attention-output boundary. The fixed 1,014/1,024 and zero-substantive formal criteria are unchanged; because the short diagnostic already cannot pass them, a new full capture is not being misreported or repeated. This implementation advances issue #166 but does not close it.
- Why: direct TensorRT-LLM source inspection and isolated real-checkpoint A/Bs showed that global fp32 expert combination and a full-K attention output GEMM introduced different rounding boundaries, while expert-by-expert execution was not an acceptable throughput implementation. The public fused kernel has no setting that removes its remaining rounding difference. Retaining the measured best fused path and explicitly leaving the residual open preserves product throughput, truthful evidence, and the existing quality gate.
- Refs: issue #166; G4 D2/D3/D5/D6; M15 A8; `kairyu/models/{moe,moe_parallel,parallel}.py`; `kairyu/kernels/nvfp4_moe_gpu.py`; `bench/g4_ma1_qwen3_235b_nvfp4_{capture,bench}.py`

### 2026-08-02 — [progress] G4 M-A1 formal EP2 exposes and corrects NVFP4 MoE math drift
- What: the first full 64-prompt × 16-position formal EP2 arm completed without runtime failures but failed the fixed gate at 917/1,024 exact selected tokens, 83 substantive disagreements, and selected-logprob bounds above 0.25 nat. Prefix hashes, canonical teacher chaining, position association, top-64 capture, and fresh-prefix execution all replay exactly. Position-0 torch attention scored 58/64 against the reference versus FlashInfer's 60/64, excluding the FlashInfer attention kernel as the principal cause. Kairyu now applies the fused NVFP4 MoE contract's layer-global maximum input scale across all experts for FC1 and FC2, reads remote EP scale scalars without loading remote weights, and retains Qwen router weights/final expert accumulation in FP32 until one BF16 output cast.
- Why: TensorRT-LLM 1.2.1 and current vLLM fused NVFP4 MoE paths quantize shared FC1/FC2 activations with layer-global scales and finalize FP32 router-weight accumulation before casting the output. The previous per-expert activation scales varied by up to 2,367.5× in the pinned checkpoint, and premature BF16 router-scale rounding introduced a non-reference precision boundary at every sparse layer. Acceptance thresholds remain unchanged and formal EP4 stays stopped until corrected EP2 passes.
- Refs: issue #166; G4 D5; `kairyu/models/{moe,moe_parallel,loader}.py`; `tests/unit/test_{moe_precision,ep_model_loader}.py`; current vLLM `flashinfer_fp4_moe.py`

### 2026-08-02 — [amendment] G4 M-A1 binds the fresh-prefill teacher rollout
- What: the reference capture now constructs each world-specific canonical continuation as 16 autoregressive one-token fresh-full-prefix waves, and Kairyu scores those exact frozen prefixes. Ordinary 16-token retained-KV continuations remain complete diagnostics but no longer supply the binding prefix. The gate independently rebuilds the teacher chain from raw prefix hashes and selected/canonical token IDs, while the fixed 1,014/1,024 agreement, zero-substantive-disagreement, reciprocal top-64, 0.125-nat near-tie, and 0.25-nat logprob bounds remain unchanged.
- Why: the pre-formal TensorRT-LLM TP2/EP2 diagnostic retained all 1,024 old free-decode-prefix teacher positions and measured only 933 exact matches, with 91 differences, no canonical top-64 omissions, but a 6.125-nat maximum teacher deficit and 6.199-nat canonical logprob path delta. Position 0 was 64/64 exact and prompt/prefix/position audits ruled out capture indexing or association errors. Treating those substantive retained-KV-decode versus fresh-prefill differences as noise, or requiring Kairyu to reproduce an internally inconsistent cross-path invariant, would both be incorrect. The amended gate makes the issue's same-prefix next-token claim explicit and does not claim cross-stack retained-KV decode parity.
- Refs: G4 D5; issue #166; `bench/g4_ma1_qwen3_235b_nvfp4_{capture,bench}.py`; `tests/bench/test_g4_ma1_qwen3_235b_nvfp4_{capture,bench}.py`; `docs/{goals/g4-moe-engine.md,gpu-runbook.md}`

### 2026-08-01 — [design] G4.1 M-A1 mid-MoE correctness design reviewed
- What: reviewed the real Qwen3-235B NVFP4 correctness path as an L1-only replicated-attention EP2/EP4 operator. The fixed gate uses per-world TensorRT-LLM teacher prefixes, 1,014/1,024 agreement, a 0.125-nat reference near-tie bound, reciprocal 0.25-nat logprob bounds, BF16 KV, native packed NVFP4 kernels, no radix/block reuse, and raw-only replay. The implementation now loads only rank-owned routed experts while validating the complete 146,549-tensor checkpoint contract and exact 27-shard/tokenizer file identities. Its all-rank error-envelope probes retain the actual gloo/NCCL topology, rank-0 sampler, loaded expert partition, native kernel inventory, and zero unresolved meta state without stranding peers on a local probe error. Real Kairyu EP2 and EP4 one-prompt forwards agree exactly, and the immutable TensorRT-LLM 1.2.1 TP2/EP2 and TP4/EP4 smokes selected the same token within tolerance.
- Why: M-A1 needs a reproducible correctness anchor without conflating replicated-attention correctness mode with the attention-DP/overlap throughput topology required by M-A3. Same-prefix scoring avoids treating all positions after one valid near-tie as mismatches, while the unchanged fixed thresholds and fail-closed provenance prevent observed noise from weakening the gate.
- Refs: issue #166; `docs/design/g4-mid-moe.md` D1–D6; `docs/goals/g4-moe-engine.md` M-A1; `kairyu/models/moe_parallel.py`; `kairyu/engine/core/worker.py`; `bench/g4_ma1_qwen3_235b_nvfp4_bench.py`

### 2026-08-01 — [progress] Quantized EAGLE/MTP draft-head support reaches formal closure
- What: the clean source-`d8dbdba` public Qwen3-32B/EAGLE-3 run passed all 13 formal checks on one RTX PRO 6000 Blackwell with FlashInfer target verification. Dense and four offline-packed dynamic-FP8 arms each accepted 90/270 proposals (33.33%). The selected `fp8_dense_fc` arm used 861,854,720 bytes versus dense 1,565,296,640 (55.06%), and retained 25.842/26.172 = 98.74% standalone-cycle goodput; its 5.218/4.287 ms draft median is an explicit 1.2171x latency cost, so it remains opt-in rather than a default. Every teacher prefix was exact. Corrections were exact in 87/90 repeated rows per arm; the one unique cross-shape selection differed by 0.13118 nat and the maximum correction delta was 0.16395 nat, both within the fixed 0.25-nat reciprocal bound. CPU draft coverage passes 80 checks, attention coverage passes 55, the complete CPU/benchmark suite passes 3,573 tests with 30 deselected, and the fused CUDA draft suite passes 3/3 without skips.
- Why: this supplies the issue's required real trained-head latency, memory, acceptance, correction, and target-corrected cycle comparison without converting a measured draft slowdown into a speedup claim. The environment has no compatible trained public MTP target/checkpoint, so MTP closure is deliberately limited to canonical packed-checkpoint loading, numerical tolerance, and real fused-CUDA execution; native trained EAGLE/MTP serving-state integration remains G4.
- Refs: issue #234; implementation commits `bacd23f`, `d8dbdba`; m14 §9; m17 A21–A25; `bench/results/issue-234-draft-quant-qwen3-32b-rtxpro6000-2026-07-31.json` (SHA-256 `850191a039edd6e3ff5ae4bf974eadeef3227b3700b1747d281c595daad63c59`)

### 2026-08-01 — [progress] F4b proves agentic DRAM-tier fleet value without TPOT regression
- What: closed F4b with a retained six-container Qwen3-32B TP4 artifact. Four sequential AB/BA performance arms raised the pooled engine prefix-hit rate from 47.7941% to 60.2338% (+12.4397 points); the pooled tier-on/off TPOT p99 ratio was 1.03721 and the cohort-ratio geometric mean was 1.04488, both within 1.10. Synchronous step evidence observed the decode allocation control with no tier transfer after first content. Two fresh sequential cohort-A quality arms exactly reproduced their parent performance output, cache usage, and per-request tier counters. The maximum selected-logprob difference across 3,968 comparable positions was 0.195256 nat; the maximum reciprocal difference at four first divergences was 0.213124 nat, with every tier-on divergence restoring 160 pages and no fallback or ownership failure. Seal, retained-copy byte verification, and independent raw replay passed.
- Refs: G5 F4b; issue #188; performance source `9af5d7a4b30502c8ca95bcbafc6e6ae65d521ba1`; quality source `6dd97da1bc147331161d6d7d6f34cabbe2af224e`; `bench/results/g5-f4b-agentic-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`; performance raw SHA-256 `63ee8bc89bd19e331354419e1f1511428b90b60c2785f71c218c7df113637e05`; quality raw SHA-256 `aaa989e790aa3c857048fd4ab6d6be9e47a1e05c61f136774a8a182f07492109`; performance manifest SHA-256 `54269beea4b24fee099eeca58ec48eaa4e267408d57ed9446125464dcc6f157f`; quality manifest SHA-256 `a4e258aeceb9cd1b248cf6f5d912617ab460f8c43d3d65dc61684f54f966881e`

### 2026-08-01 — [amendment] F4b separates performance from distribution quality
- What: F4b keeps the original four-container performance raw byte-identical and amends its verdict contract instead of requiring exact free-running output equality across tier-off/on cache-prefill shapes. Prompt/order, cache gain, TPOT, tier activity, decode-path isolation, runtime, and Docker lifecycle remain binding. A separate timing-nonbinding pair of fresh cohort-A containers records top-64 logprobs and must exactly reproduce each corresponding performance arm's output, cached-token usage, and per-request tier state. Selected-token logprobs on the common generated prefix and reciprocal selected-token logprobs at the first divergence must differ by at most 0.25 nat; later positions have different generated prefixes and are not compared. Both quality container lifecycles are retained and independently replayed. The raw-shard schema remains v1 while the amended verdict revision is versioned separately.
- Why: the formal run produced four repeatable tier-off/on differences at the first token of turn 5 while both same-arm cross-cohort pairs remained 128/128 exact. A DRAM-disabled HBM-cache rerun produced a third valid greedy choice at the same positions, and the distribution diagnostic measured maximum common-prefix and reciprocal first-divergence differences of 0.195256 and 0.213124 nat. This identifies normal BF16 near-tie/cache-prefill-shape sensitivity, not stale DRAM restore. Requiring exact free-running equality would be an over-strong, shape-dependent gate; removing all quality control would be too weak, so the amended test binds comparable probability evidence without contaminating TPOT or comparing post-divergence inputs.
- Refs: G5 F4b; issue #188; raw SHA-256 `63ee8bc89bd19e331354419e1f1511428b90b60c2785f71c218c7df113637e05`; `bench/agentic_kv_tier_f4b_bench.py`; `tests/verification/test_agentic_kv_tier_f4b_bench.py`; `docs/gpu-runbook.md` §9.8b

### 2026-08-01 — [amendment] F4b separates calibrated runtime and measured-source provenance
- What: F4b pins the exact retained F4a execution image (`sha256:25543ae9cbc9d2e80f1b4be2193d138486adb91c89a02cdbd0be0e62a1cc67be`, OCI revision `edd535f7018695fc03c479a86fbd690174cca5ef`) as its compiled-runtime authority while independently binding the clean read-only source commit and all executed Kairyu Python bytes. All four arms must use that same image and source, and the tier-on arms still require the retained profile's exact runtime and engine identities.
- Why: rebuilding the locked environment produced a different FlashInfer `RECORD` identity because its installer metadata carried a different timestamp. The runtime correctly rejected that uncalibrated image. Reusing the content-addressed calibration image preserves the fail-closed profile contract without treating its historical build-source label as the source currently mounted and measured.
- Refs: G5 F4a/F4b; issue #188; `bench/agentic_kv_tier_f4b_bench.py`; `docs/gpu-runbook.md` §9.8b

### 2026-08-01 — [progress] F4b agentic DRAM-tier gate reaches formal-run readiness
- What: added a fail-closed Qwen3-32B TP4 operator for F4b's tier-off/on agentic trace. The fixed 16-session × 8-turn schedule uses a 2,048-token fleet prefix, 512 appended agent/tool tokens per turn, and 32 deterministic output tokens. Four fresh containers execute a sequential two-cohort AB/BA crossover. Replay derives engine cached-token rates, nearest-rank request TPOT p99 noninferiority, exact prompt/output parity, and step-level proof that every tier counter remains unchanged after first content while the observed output-16-to-17 page allocation makes that proof non-vacuous. Source, checkpoint, immutable image, retained F4a TP4 profile, runtime/backend/dtype, hardware, container command, and Docker lifecycle identities fail closed; synthetic tests exercise each negative gate without claiming a GPU result.
- Why: F4b needs production cache/timing/tiering events from a representative multi-turn workload, while the prior F4a crossover proves only transfer correctness and restore economics. Predeclaring the workload and binding four non-overlapping arms prevents a one-shot latency result, environment drift, or benchmark-side cache estimate from becoming fleet evidence.
- Refs: G5 F4b; issue #188; `bench/agentic_kv_tier_f4b_bench.py`; `tests/verification/test_agentic_kv_tier_f4b_bench.py`; `docs/gpu-runbook.md` §9.8b

### 2026-08-01 — [progress] F4a publishes measured DRAM restore crossovers
- What: closed F4a with separate, non-overlapping Qwen3-32B TP4 and TP8 schema-v2 shards from clean commit `edd535f7018695fc03c479a86fbd690174cca5ef` and immutable image `sha256:25543ae9cbc9d2e80f1b4be2193d138486adb91c89a02cdbd0be0e62a1cc67be`. TP4's first stable restore-winning suffix begins at 1,024 tokens: 512 failed at a 1.021531 median paired restore/recompute ratio and 2/9 wins, while 1,024 passed at 0.975449 and 8/9 and all larger cells passed. TP8 passed all ten cells from the 16-token measurement floor with 9/9 wins, so its threshold is recorded as at or below 16 tokens. All correctness and provenance checks, assembly, retained-copy verification, and independent raw replay passed. The complete 5.1-MiB artifact retains both raw shards, profiles, manifest, immutable-image inspect, full container IDs, and created/exited container records.
- Why: this replaces an assumed DRAM threshold with policy-consumable measurements from the exact production implementation while retaining the failed 512-token TP4 cell and every other sample, so runtime enables restore only from a complete, independently reproducible crossover profile.
- Refs: m7 D6 amendment; G5 F4a; issue #187; operator source commit `edd535f7018695fc03c479a86fbd690174cca5ef`; `bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`; manifest SHA-256 `0680333d06bf6d06ea91fbd12ef5b88732c936d1446a06d631674dcb15946fd6`; TP4 raw SHA-256 `609ff6bb1b951a7d4f70bb5948e1f9dcd68e55b0523e2993effa76a3b750cf01`; TP8 raw SHA-256 `2947d27dcb51a227ec9e21c72a4e435e693dbb675ff683632dd285cfc86d6611`

### 2026-08-01 — [amendment] F4a corrects its comparator and coalesces DRAM transfers
- What: the complete FlashInfer TP4/TP8 schema-v1 collection passed KV/output correctness and provenance but correctly failed because neither TP degree had a stable restore-winning suffix. Audit found two independent defects: cold recompute split the final prompt token into an extra model invocation, biasing the comparison toward restore, and the page-oriented DRAM path built 131,072 Python fragment views and submitted 65,536 four-KiB copies per rank at 8,192 tokens. Schema v2 now runs the complete prompt through production cold-prefill chunks and samples the final hidden state directly. The CUDA tier retains one NUMA-attested pinned owner but views it as `[fragment, slot, bytes]`, caches matching host/device plane owners, and copies each jointly contiguous extent once. Logical page ordering, SHA-256 checksums, fingerprints, legacy CPU/injected backends, and fail-closed ownership remain unchanged; raw profiles and runtime policy bind the exact versioned backend identity. Focused benchmark/unit coverage passes 106 tests, real CUDA roundtrip coverage passes on GPU 7, and a diagnostic 256-MiB probe reduced the longest-prefix transfer from 65,536 submissions and 0.73–0.97 s to 128 submissions and 4.97–5.21 ms while preserving exact bytes.
- Why: the first raw curve cannot be repaired or relabelled: its comparator did not match production cold prefill, and its transfer implementation measured Python submission overhead rather than the intended coalesced DRAM path. Preserving the strict stable-suffix gate and checksum/ownership ABI while requiring fresh schema-v2 TP4/TP8 shards keeps the eventual crossover honest and deployable. No crossover is claimed until those clean-commit shards pass assembly, verify, and independent raw replay.
- Refs: m7 D6 amendment; G5 F4a; issue #187; `kairyu/engine/core/kv_tier.py`; `kairyu/engine/core/kv_tier_policy.py`; `bench/dram_kv_tier_qwen.py`; `tests/unit/test_kv_tier.py`; `tests/gpu/test_kv_tier_gpu.py`; `docs/gpu-runbook.md`

### 2026-08-01 — [progress] F4a rejects incorrect Qwen metadata and Torch fallback
- What: the post-RoPE formal TP4 run completed all 100 restore/recompute pairs, then the retained validator rejected its self-inconsistent checkpoint identity: Qwen3-32B explicitly declares `head_dim=128`, while the collector incorrectly derived `5120/64=80`. In-memory audit found every remaining ownership, KV, output, timing, and paired-sample check valid, but also showed that non-root FlashInfer initialization had tried to create `/.cache` and auto-fallen back to Torch. The collector now reads and validates the explicit head dimension, pins the known config-file SHA, and rejects any non-FlashInfer formal path before model loading. The container contract supplies a writable `FLASHINFER_WORKSPACE_BASE`, the image defaults that workspace under `/tmp`, and the Docker revision label now occupies a metadata-only final stage so benchmark-only commits do not invalidate the large dependency payload. Focused benchmark and Dockerfile tests pass 39/39.
- Why: the TP4 raw measured a real Torch-bound crossover, but production auto-selection on SM120 is FlashInfer and the retained policy binds the attention implementation. Publishing the Torch result or weakening the AOT validator would therefore misstate production behavior. Fail-fast selection plus exact config provenance preserves the formal quality gate and removes repeated multi-minute runs for setup errors.
- Refs: issue #187; `bench/dram_kv_tier_qwen.py`; `Dockerfile.cuda`; `tests/verification/test_dram_kv_tier_qwen.py`; `tests/unit/test_dockerfile_cuda_aot.py`; `docs/gpu-runbook.md`

### 2026-08-01 — [progress] F4a formal TP4 exposes and fixes a fused RoPE race
- What: the first Qwen3-32B TP4 F4a run correctly stopped when independently recomputed KV bytes differed from the offloaded source on every rank. The first corruption was rank 2, layer 4, token 11, KV head 1, upper half-vector; retained SM120 PTX confirmed that the four-warp in-place RoPE kernel stored one half before another warp necessarily loaded its partner, with no barrier. RoPE now assigns each lower/upper pair to one lane, loads both values before either store, and uses one/two/four warps for head dimensions 64/128/256. A production T=15, QH=16, KVH=2, D=128 GPU regression passes exact Q/K parity across 32 storage remaps; the patched full Qwen3-32B TP4 diagnostic produced bitwise-identical source and two consecutive recomputations on all four ranks with identical output token 892. The strict KV SHA gate remains unchanged. The Docker runbook also fixes GPU device quoting, retains the container-ID hostname through the default bridge rather than host networking, asserts that provenance before launch, and gives non-root Triton an explicit writable cache.
- Why: treating the SHA failure as harmless BF16 nondeterminism would have hidden real cross-warp memory corruption. Pair ownership removes the race structurally without weakening lossless DRAM or recompute correctness, while the corrected container contract makes the formal operator executable under the retained non-root runtime.
- Refs: issue #187; `kairyu/kernels/rope_gpu.py`; `tests/gpu/test_fused_rope_gpu.py`; `bench/dram_kv_tier_qwen.py`; `docs/gpu-runbook.md`

### 2026-08-01 — [amendment] F4a native DRAM tier reaches formal-run readiness
- What: added an opt-in bounded pinned-DRAM tier behind RadixKV eviction and restore, with a dedicated CUDA copy stream/events, full prefix and page integrity identities, exact-once transfer handles, quarantined unknown ownership, GPU-local NUMA first-touch plus scoped checksum affinity, and unanimous TP control over a separate Gloo group. Startup now loads only a retained crossover profile that independently replays nine paired samples per length and exactly matches the model, TP/KV layout, complete engine source, attention implementation, batching, CPU/NUMA/PCIe, GPU, and software runtime; capacity below the measured threshold and P-D use fail before serving. The Qwen3-32B TP4/TP8 formal operator, raw replay, profiles, Docker procedure, benchmark registry, and health metadata are implemented. The full unit suite passed 2,406 tests, the final benchmark/surface set passed 44, and the real CUDA pinned-tier roundtrip passed on GPU 7.
- Why: F4a requires a published measured restore-versus-recompute crossover, not an assumed threshold. Binding policy to complete retained evidence prevents a stale hardware or implementation cohort from silently changing the decision, while fail-closed host/HBM ownership prevents cancellation, rank disagreement, callback reentry, or ambiguous DMA from publishing corrupt KV or recycling a page still in use. F4a deliberately remains open until the same clean commit and immutable image complete separate, non-overlapping TP4 and TP8 runs.
- Refs: m7 D6 amendment; G5 F4a; issue #187; `kairyu/engine/core/kv_tier.py`; `kairyu/engine/core/kv_tier_policy.py`; `kairyu/engine/core/numa.py`; `kairyu/engine/core/radix_kv.py`; `kairyu/engine/core/worker.py`; `bench/dram_kv_tier_qwen.py`; `docs/gpu-runbook.md`

### 2026-07-31 — [amendment] Draft correction parity binds target distributions across shapes
- What: the formal EAGLE gate still requires every accepted proposal prefix to exactly match the independently generated teacher trace. Sequential teacher and multi-token verification corrections are additionally compared under both complete target distributions: an exact token difference is valid only when the reciprocal selected-token log-probability delta is finite and at most the fixed, non-operator-adjustable 0.25 nat bound. Every row records both selected IDs, all four cross-distribution log-probabilities, both individual deltas, and the exact-match flag; summaries retain the exact count and maximum delta. Validation work is excluded from the measured speculative-cycle wall time.
- Why: the first five-prompt formal run found one unique all-accepted correction where the same target checkpoint chose a different token across sequential and multi-token execution shapes; all accepted proposal tokens, KV prefixes, and the other correction rows remained exact. Requiring bit-identical argmax across legitimate target shapes is stronger than product distribution parity, while merely dropping the mismatch would not protect quality. The reciprocal 0.25-nat bound is the project's established target-parity criterion and will distinguish a harmless near tie from a material distribution change.
- Refs: issue #234; m14 §9; m17 A24; diagnostic schema-v1 artifact `/tmp/issue-234-draft-quant-qwen3-32b-rtxpro6000-2026-07-31.json`; `bench/draft_quant_qwen.py`

### 2026-07-31 — [design] Quantized draft heads own a fail-closed checkpoint policy
- What: EAGLE and MTP now construct eligible canonical projections through a draft-owned compressed-tensors dynamic-FP8 policy, load packed checkpoints without online conversion or dense substitution, and reject unsupported or ambiguous metadata before tensor loading. Public Qwen3-32B EAGLE support adds explicit 64Q/8KV GQA geometry, `(1, 31, 60)` auxiliary captures, and the trained target-root contract pairing `aux[t]` with `embedding(token[t+1])`; target verification consumes `[root, *proposals]`. The formal operator rotates five dense/FP8 FC/head arms across multiple prompts and anchors, records exact correction, latency, memory, acceptance, committed-token cycle goodput, and complete source/checkpoint provenance. MTP's available checkpoint, tolerance, and fused-CUDA coverage is synthetic because no compatible trained public MTP target/checkpoint is present locally; native EAGLE/MTP serving-state integration remains the existing G4 boundary.
- Why: inheriting target quantization or silently dequantizing a draft can load the wrong checkpoint semantics while hiding memory and acceptance regressions. The original unshifted diagnostic also compared a draft proposal against an already-produced root; upstream SGLang, vLLM, SpecForge, and SpecJAX agree on the shifted contract, and the public Qwen3-32B draft measured 6/12 accepted tokens across four diagnostic anchors after the correction versus 0/12 under the old input.
- Refs: issue #234; m14 §9; m17 A21–A25; `kairyu/models/draft_quant.py`; `kairyu/models/eagle.py`; `kairyu/models/mtp.py`; `bench/draft_quant_qwen.py`

### 2026-07-31 — [progress] G2 A9 DP-versus-TP crossover report passes
- What: completed the fresh Qwen3-32B TP8 arm against the post-SSE-fix A8 comparator. All 984 TP8 requests completed with HTTP 200, exact 32-token output and usage, `[DONE]`, zero retry, and one exact gateway placement; all 14 report-integrity checks passed. `verify --assert-gate` and raw-only `replay --assert-gate` independently reproduced the report. Median DP/TP8 goodput was 3.884/3.902, 7.383/7.313, 12.948/8.994, 16.042/11.707, and 19.612/12.440 req/s across 4/8/16/32/64 offered req/s. The only ordering transition is bracketed at 4–8 req/s with no interpolation; DP is first noninferior at 8 req/s. The portable test job now checks out full history so its fixed-ancestor `git archive`/helper-blob contract runs in Actions instead of failing under a depth-1 checkout.
- Why: this supplies the missing matched-runtime evidence after fixing the one real Unicode SSE transport failure instead of retrying or ignoring it. It also keeps throughput, TTFT, and TPOT separate: TP8's per-token latency is lower at 16–64 req/s, but DP sustains 1.37–1.58× its goodput at 16–64 req/s and much lower TTFT under load. The report is intentionally threshold-free and describes the production topology with its doubled DP aggregate capacity.
- Refs: issue #159; G2 A9; operator source commit `d4c8e121d285a08843e5aa69d2de4a747d574c95`; runtime source commit `86d49223ffcdba6052428474bf0d9094c6791fed`; `bench/results/g2-a9-dp-tp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`; raw SHA-256 `6c6d11deeb5735ecf37de48663e553a88a90e606196bb432111fe70772d4a664`; manifest SHA-256 `799813e087bd9aca9d62c45a995e7eed6471534f18fdaa09fca0666412786bdc`; placements SHA-256 `0276059cd9822ce9e046fce088102b372a324689d620375ce23e13829f4ec020`

### 2026-07-31 — [progress] G2 A9 pins a fresh post-SSE-fix A8 comparator
- What: reran the complete A8 single-TP4 versus DP=2×TP4 sweep on merged source `86d4922` and repinned A9 to its immutable source archive, 199-file runtime rollup, and new raw/manifest/placement digests. All 2,992 requests completed with HTTP 200, `[DONE]`, exact terminal usage, and zero retry; all 1,496 DP placements correlated. Verify and raw-only replay reproduced every value. The sole false check remains the explicitly accepted 1.9× scaling target, with repeat ratios 1.7810×/1.7711×/1.6422× and median 1.7711×. A9 now accepts either a fully passing A8 artifact or exactly that one accepted false check, while rejecting any other failure or verdict/check inconsistency.
- Why: the first TP8 run exposed and then led to fixing Unicode SSE framing. Rerunning the A8 comparator on the same repaired runtime removes a transport/version confound without retrying or discarding any failed request, so the forthcoming TP8 arm will compare the same source, image, checkpoint, trace, and configuration.
- Refs: issue #159; issue #303; source commit `86d49223ffcdba6052428474bf0d9094c6791fed`; `bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`; raw SHA-256 `b024d71741eea71f9aa698fd0bf44e27c245584b524835c79acd4e72426f22d4`; manifest SHA-256 `833a7ef269ad2f427f8e88bf6f91773de1e876f5baa441fc01cfce04b1cd3b28`; placements SHA-256 `3be490d1a17a7d20a8cb98fa3eee8cb8bee60e5f6b8624dfea187f45394ab320`

### 2026-07-31 — [progress] SSE framing preserves Unicode model output
- What: Replaced `httpx.aiter_lines()` in the OpenAI-compatible upstream path with a byte-oriented CR/LF SSE decoder, including BOM, comments, fragmented transport, and repeated `data:` fields. Kairyu's Chat Completions, Completions, and Responses writers also escape U+0085/U+2028/U+2029 so legacy universal-line clients retain one physical JSON event line. The regression suite covers all three separators and reconstructs their exact content.
- Why: The first G2 A9 TP8 sweep failed one of 984 retry-free requests when a valid generated separator split the JSON string at the exact logged column. Retrying or ignoring it would hide a product bug; both sides of the eventual DP/TP8 comparison must run the same fixed transport.
- Refs: issue #303, `kairyu/sse.py`, `kairyu/engine/openai_backend.py`, `kairyu/entrypoints/server/app.py`, `kairyu/entrypoints/server/responses_service.py`

### 2026-07-31 — [progress] G2 A9 reuses A8 DP evidence and reaches TP8-run readiness
- What: added the report-only A9 operator, a dedicated all-eight-GPU TP8 engine plus one-replica gateway stack, immutable A8 raw/manifest/placement replay, exact request/placement verification, and independent goodput/TTFT/terminal-TPOT/crossover recomputation. The operator measures only the missing 24 warmups and 960 TP8 requests. It uses A8 image `sha256:2c73b577...` plus the exact source tree materialized from A8 commit `4924b4d`, verifies the fixed source archive, all 196 mounted runtime files, and the delegated A8 helper, and repeats both source and full-checkpoint attestation after traffic. TP8 uses the retained DP one-token namespace, and the formal config discloses both topologies' per-engine and aggregate KV/sequence/batch/graph capacity.
- Why: rerunning the already complete DP sweep would waste GPU time, while using current checkout code only for TP8 or changing its request namespace would make the comparison unmatched. A8 retained every timestamp needed by `serving_bench.py`'s terminal-stream TPOT, so fixed-SHA replay plus an identical-runtime/request TP8 arm supplies the missing evidence without estimating samples or weakening provenance. Matched per-engine limits intentionally leave two independent DP replicas with twice TP8's aggregate configured capacity; the report records observed concurrency so this production-topology tradeoff is not misread as an equal-capacity kernel comparison. Production PP stage sharding is not relabeled from `pipeline_depth`; the separate PCIe DP/PP/TP layout report remains dependent on real PP wiring.
- Refs: issue #159; G2 A9; `bench/g2_a9_dp_tp_crossover_bench.py`; `examples/qwen3-32b-multi-gpu/a9-tp8-{compose,replica,gateway}.yaml`; `docs/gpu-runbook.md`

### 2026-07-31 — [progress] P-C4 closes on Qwen3-VL-32B TP8 and the real Open WebUI upload path
- What: the clean `b8971cb` stack passed every A18 gate on 8× RTX PRO 6000. RED and BLUE fixtures produced their corresponding distinct answers; unary RED/BLUE and streamed RED each reported exact processor usage of 1,060 input and 2 output tokens; a metadata-service URL failed pre-dispatch with `400 invalid_image`; and pinned Open WebUI uploaded the PNG through `/api/v1/files/`, sent its owned file ID/type/URL without a browser data-URL shortcut, and rendered RED. The vLLM command retained the pinned model/image/TP8 bounds and enabled the `hermes` parser matching the model's JSON-in-`<tool_call>` template. Correction to the preceding progress entry: the browser need not explicitly write `tool_choice=auto`; Open WebUI can inject built-in tools and vLLM normalizes an omitted choice to `auto`.
- Why: the gate proves the issue's complete standard image-chat path without weakening media security, token accounting, or OpenAI request intent. A native Kairyu VLM and multimodal native-tool continuation remain outside A18 rather than being implied by this closure.
- Refs: issue #203; source commit `b8971cbe89e14002749e72f40fa1f33bf65bbc87`; `bench/results/issue-203-vlm-image-chat-qwen3-vl-32b-tp8-2026-07-31.json` (SHA-256 `6c8c653e4fba977226f3eca5dc6540d38ccc0d090cb44a27572bcbc26da7ef2c`); `deploy/compose/docker-compose.webui-vlm.yaml`; `scripts/webui_vlm_{api_smoke.py,browser_smoke.mjs,smoke.sh}`

### 2026-07-31 — [progress] P-C4 real browser gate fixes the stock-vLLM tool parser contract
- What: the clean `79292b2` TP8 API gate passed RED/BLUE unary, RED streaming, exact 1,060/2-token usage, and remote-URL rejection. Open WebUI's owned-file browser path then exposed one real integration gap: WebUI preserves its built-in tools with `tool_choice=auto`, while the stock-vLLM process had no auto-tool parser enabled and returned 400. The overlay now enables auto tool choice with vLLM's `hermes` parser, which exactly matches the pinned Qwen3-VL template's `<tool_call>{JSON}</tool_call>` format; the Compose contract test pins both flags. A clean-commit browser rerun remains pending.
- Why: stripping Open WebUI's tool intent inside Kairyu would violate transparent OpenAI request semantics. Parser selection must follow the immutable model template rather than the model-family name: vLLM 0.26's `qwen3_xml` parser targets a different Qwen3-Coder tag grammar.
- Refs: issue #203; implementation commit `79292b2`; `deploy/compose/docker-compose.webui-vlm.yaml`; `tests/unit/test_webui_vlm_compose.py`

### 2026-07-31 — [amendment] P-C4 gains a fail-closed stock-vLLM image-chat path
- What: added role- and part-preserving multimodal prompt transport through direct and pooled OpenAI-compatible backends, exact processor-usage ownership, backend-specific admission, off-event-loop raster preparation, strict inline PNG/JPEG/WebP validation, bounded chat bodies, and consistent interactive/batch failure semantics. A GPU-only Compose overlay pins stock vLLM and Qwen3-VL-32B-Instruct at TP8 while retaining the production embedding endpoint; its reproducible gate checks RED/BLUE semantics, unary/SSE usage, remote-URL rejection, and Open WebUI's normal owned-file upload path. The portable CI-equivalent suite passes 3,746 tests with no selected skips and 87.98% coverage; the clean-commit real TP8 gate remains the final closure step.
- Why: the former content-part parser intentionally rejected every image because it had no lossless role/part transport, processor owner, media security boundary, exact token accounting, or deployable VLM. Keeping Kairyu responsible for orchestration/admission while delegating model-specific processing and templating once to pinned stock vLLM preserves the L3/L2/L1 product boundary and avoids a second Qwen preprocessing implementation.
- Refs: issue #203; G6 P-C4; m11 D5/D7/A18; `kairyu/engine/{prompt,vision,openai_backend}.py`; `kairyu/entrypoints/server/{chat_service,middleware,metering}.py`; `deploy/compose/{config-vlm.yaml,docker-compose.webui-vlm.yaml}`; `scripts/webui_vlm_{smoke.sh,api_smoke.py,browser_smoke.mjs}`; `tests/{unit/test_vision.py,unit/test_webui_vlm_compose.py,server/test_openai_api.py}`

### 2026-07-31 — [amendment] P-C3 closes on production offline embeddings and Kairyu-only RAG
- What: added a production `fastembed` backend that loads a revision-, ONNX-, and manifest-SHA-pinned all-MiniLM-L6-v2 snapshot entirely offline; validates every recorded bundle file; warms one CPU session; reports exact tokenizer usage; rejects excess work before dispatch; and participates in readiness, fatal health, cancellation-safe startup, and draining shutdown. The default image remains lean while the Open WebUI image opts into the dependency and model. Its mandatory browser smoke now proves two-input 384-dimensional normalized embeddings, document ingestion, vector retrieval of a query-only canary, a citation-bearing Kairyu answer that requires retrieved context, visible outage, and retrieval/answer recovery after restarting only Kairyu. Optional reranking remains disabled and deferred. A source allowlist reduces the root Docker context from 7.5 GB of checkout state to 7.07 MB.
- Why: the former mock-only route proved wire compatibility but not a deployable embedding engine, immutable model semantics, bounded CPU ownership, lifecycle health, or an actual Open WebUI RAG flow. A pinned CPU ONNX path closes those product requirements without adding the multi-GB PyTorch runtime or making CI depend on GPUs/network-time model resolution; a retrieval-only canary prevents a plain chat success from masquerading as RAG.
- Refs: issue #202; G6 P-C3; m11 D4/D7/A17; `kairyu/engine/embedding.py`; `kairyu/deploy/{spec,builder}.py`; `deploy/compose/{config.yaml,docker-compose.webui.yaml}`; `scripts/{prefetch_embedding_model.py,webui_smoke.sh,webui_browser_smoke.mjs}`; `tests/{unit/test_embedding_backend.py,unit/test_compose_configs.py,server/test_embeddings_models.py}`; `docs/{deployment.md,goals/g6-product-surface.md,design/m11-product.md}`

### 2026-07-31 — [progress] Noisy-neighbor schedule test binds raw clock inputs
- What: Replaced the noisy-neighbor schedule test's exact equality on subtracted floating-point timestamps with exact checks of tenant/phase/index order, one shared origin, and each raw target as that origin plus its specified cadence. The test fixes the origin immediately below the 512-second binary exponent boundary and verifies that the benchmark reads the clock exactly once, making the former CI-only failure deterministic while strengthening the named shared-origin contract.
- Why: Python 3.12 job `91101025612` crossed that exponent boundary and produced `0.49999999999994316` for a mathematically 0.5-second offset. The one-ULP cancellation is not a product timing error; exact subtraction was both numerically invalid and unable to prove that the two tenants shared one origin. Broad approximation, skip, retry, or test deletion would provide weaker coverage than asserting the raw schedule inputs directly.
- Refs: PR #299, `tests/verification/test_noisy_neighbor_gpu_bench.py`, GitHub run `30613391675`

### 2026-07-31 — [progress] F1b freezes build images before kind startup
- What: Changed the F1b rollout gate to save each clean-head gateway and mock image to a guarded temporary archive immediately after its build and load those archives after kind startup. The gate still fails closed on image revision and Docker/CRI/containerd digest mismatches and still runs the unchanged zero-failure rollout and independent replay. A policy regression test fixes the build/save/create/load ordering and rejects a return to delayed `kind load docker-image` resolution.
- Why: GitHub run `30611974651` built and inspected both images successfully, then kind's delayed internal `docker save` lost `kairyu:dev` before any Kubernetes rollout ran. F1a/F1b/F1c were on separate hosted runners, so cross-workflow cleanup was not the cause. Freezing the exact build output before cluster creation removes the observed mutable-tag TOCTOU without deleting, retrying, skipping, or weakening the real rollout test.
- Refs: issue #158, PR #299, `scripts/kind_rollout_gate.sh`, `tests/unit/test_ci_workflow_policy.py`, GitHub job `91096580771`

### 2026-07-31 — [amendment] G2 A8 closes on an accepted 1.7993× scaling deviation
- What: Ran the complete Qwen3-32B DP=2×TP4 versus TP4 gate on all eight RTX PRO 6000 GPUs. All 2,992 requests succeeded without retry, every DP response correlated to one of 1,496 placement rows, router p99 was 3.723 ms, and DP retained 99.53% of the single-replica cache-hit rate. The three peak-goodput ratios were 1.9988×, 1.7342×, and 1.7993×; their 1.7993× median missed the original 1.9× threshold, so the retained manifest remains `passed: false`. The product owner explicitly accepted this measured median for closure without rewriting the threshold or claiming a formal PASS.
- Why: The real artifact proves the routing, affinity, integrity, and near-linear scaling value while preserving the remaining 5.3% target shortfall as visible evidence. Treating 1.7993× as 1.9× or moving the threshold after measurement would hide the residual; an explicit accepted deviation records the product decision without falsifying the gate.
- Refs: issue #158, PR #299, source commit `4924b4d71b8fae0af087979908819aed6939a871`, `bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31/`, raw SHA-256 `b637489302a9b818a0c34790c4059946994ff6070f76ac9e1bc7d128bbbd803f`, manifest SHA-256 `da439f153c04d05178ddf96c489aca7fb1cc270ba982736c1bc98197730e3946`, placements SHA-256 `76ca1a5f709ccf8492238bdcc4193776f94fdb7a88a40a7e970a517db30270c3`

### 2026-07-31 — [progress] G4 E-KV rejects unit-scale FP8 KV and stays BF16
- What: ran the formal 8K/16K/32K Qwen3-32B BF16-versus-unit-scale-E4M3 bake
  on one RTX PRO 6000 Blackwell. Source, checkpoint, runtime, GPU, compilers,
  and all five FlashInfer shared objects were stable. All 7,522,091,008 K/V
  values passed finite/range/SATFINITE-byte/error audits, but output tokens
  diverged at 8K and 32K, common-prefix selected-logprob deltas were
  0.3099/0.4518/0.2656 against a 0.25 limit, and cache NRMSE reached 0.1047
  against a 0.05 limit. Public `fp8_e4m3` startup now fails closed with this
  bake result; `auto`/BF16 remain unchanged. The replay verifier ignores only
  GCC's nondeterministic timing footer and compares logprobs only while both
  arms share the same generated-token prefix.
- Why: issue #170 explicitly requires retaining a failed bake and keeping FP8
  KV disabled. Post-hoc threshold relaxation or enabling an uncalibrated
  candidate would violate that product-quality contract. Per-layer K/V static
  calibration needs representative calibration data, headroom, a bound scale
  artifact, and another independent long-context bake.
- Refs: issue #170; G4 E-KV;
  `bench/fp8_kv_g4_ekv_bench.py`;
  `bench/results/g4-ekv-fp8-kv-qwen3-32b-sm120-fail-2026-07-31/`;
  raw SHA-256
  `f759fa3308f90f70c26e04e51ebf82515a2891d1b183ef3a8bbfa67acbada305`;
  manifest SHA-256
  `4c213ebfb7376755e98bddb7c16ad508ee8ac56ef88fd69c57971e78c2224a64`;
  `kairyu/engine/core/kv_cache_dtype.py`

### 2026-07-31 — [progress] G4 E-KV is ready for the real Qwen3-32B bake
- What: implemented an opt-in `fp8_e4m3` KV cache without changing the BF16
  default. Single-rank, TP, process-service, YAML validation, and `/backends`
  reporting now carry requested and resolved cache dtype and reject unsupported
  or cross-rank/cross-process mismatched identities. The SM120/FlashInfer-only
  path stores K/V with unit-scale SATFINITE E4M3, passes unit K/V scales through
  every FlashInfer prefill/decode path, and keeps speculative, MLA, unsupported
  hardware/backend, and P-D combinations fail closed. FlashInfer v0.6.14 AOT
  packaging now includes E4M3 KV for FP16/BF16 queries at heads 64/128 and has
  a static non-empty-FP8 regression guard. Read-only inspection of the existing
  `kairyu-depth-ab:20260731-issue156` image (`4dc6af1ddc97`) found 24 FA2
  attention modules but zero FA2 E4M3-KV modules; that image was not rebuilt.
- Why: host JIT and unit/GPU kernel tests can prove the implementation contract
  but cannot satisfy G4 E-KV's binding long-context product gate or prove that a
  nvcc-free production image contains the new AOT modules. The real Qwen3-32B
  prompts, outputs, cache metadata, environment, checkpoint identity, and
  BF16-vs-FP8 tolerance result must be retained before issue #170 can close.
- Refs: issue #170; G4 E-KV; `Dockerfile.cuda`;
  `kairyu/engine/core/{kv_cache_dtype.py,kv_pool.py,worker.py}`;
  `kairyu/engine/core/attention/flashinfer_gpu.py`;
  `kairyu/kernels/paged_kv_write_gpu.py`;
  `kairyu/engine/{kairyu_backend.py,zmq_backend.py}`;
  `bench/fp8_kv_g4_ekv_bench.py`;
  `docs/design/flashinfer-sm120-aot.md`;
  `tests/{unit/test_kv_cache_dtype.py,unit/test_kv_pool_fp8.py,gpu/test_flashinfer_gpu.py}`

### 2026-07-31 — [fix] G2 A8 packaged benchmark count follows its new operator
- What: Raised the isolated-wheel registry assertion from 51 to 52 when A8 added the 52nd owned benchmark entrypoint. The real wheel now proves that the complete registry and CLI are installed instead of failing on the intentionally added wrapper.
- Refs: issue #158, PR #299, `scripts/verify_bench_wheel.py`

### 2026-07-31 — [progress] G2 A8 formal DP scaling gate reaches live-run readiness
- What: Added the fixed Qwen3-32B DP=2×TP4 versus TP4 open-loop operator, the forced-recreate two-replica/gateway stack, exact request/placement/cache evidence replay, and current-runtime container/source/checkpoint attestation. Portable tests cover semantic and integrity tampering but cannot claim the real eight-GPU result. The formal preflight currently rejects an unrelated 26.8 GiB GPU 0 owner, so A8 remains open and unpassed.
- Refs: issue #158, `bench/dp_scaling_g2_a8_bench.py`, `examples/qwen3-32b-multi-gpu/a8-*`, `tests/verification/test_dp_scaling_g2_a8_bench.py`, `docs/goals/g2-multi-gpu.md`

### 2026-07-31 — [design] Benchmark ownership becomes package-enforced
- What: made `kairyu.bench` the installed owner of reusable target,
  credential, percentile, and atomic-reporting contracts; moved Fugu's
  `ResultStore` and the general serving/frontier wrappers onto those
  primitives; and added a packaged 51-entry registry plus exact allowlist for
  the 14 retained wrapper-composition edges. Both historical wrapper invocation
  forms are exercised 102/102, all installed benchmark subcommands and eight
  fixtures are verified from a real isolated wheel, and checkout-only
  wrappers/results/tests are rejected from that wheel. URL and credential
  construction is identical across combined flags, split flags, direct models,
  and YAML; configured missing credentials fail closed and secret inputs are
  redacted. New serving/frontier artifacts identify
  `nearest-rank-v1`; older percentile semantics are explicitly migrated.
- Why: installable quality benchmarks and repository-only formal/performance
  operators had no enforceable ownership boundary and had already diverged on
  credentials, URL normalization, percentile definitions, and atomic output.
  Keeping stable wrapper/result paths preserves G2/G4/G5/G6 replay provenance,
  while package-owned reusable semantics and CI-enforced inventories prevent
  that drift from recurring.
- Refs: issue #232;
  `kairyu/bench/{entrypoints.toml,ownership.py,targets.py,reporting.py}`;
  `bench/README.md`; `scripts/verify_bench_{entrypoints,wheel}.py`;
  `docs/{benchmarks.md,gpu-runbook.md}`

### 2026-07-31 — [amendment] F4c defers a global KV pool after exact F2 replay
- What: added a fail-closed F4c analyzer and retained decision artifact that
  pin and replay the exact F2a/F2c manifests, raw rows, source commits, trace,
  logical residency, paired Qwen3-32B token usage, and performance ratios.
  F2a reconstructs 997 redundant session-HRW family copies and 513,809
  duplicate family-copy/request-steps versus zero under prefix routing; F2c
  reconstructs 319,696 avoided prompt tokens and a 1.5608% residual gross
  upper bound. The m7 D6 amendment retains per-replica RadixKV/F2 routing,
  rejects LMCache wholesale and a global `KVTransport`, and conditionally
  selects Mooncake Store behind a separate Kairyu-owned global-KV object-store
  adapter only after one exact trigger branch holds across three consecutive
  10,000-request windows.
- Why: routing already eliminates the duplicate-prefix mass visible in the
  retained representative evidence, while F2 cannot distinguish the small
  residual uncached mass from novel suffixes. Taking on a distributed store
  now would add correctness and operational ownership without measured
  incremental value; the exact window formulas preserve an objective revisit
  path after native DRAM tiering is measured.
- Refs: issue #189; G5 F4c; m7 D6;
  `bench/global_kv_pool_decision.py`;
  `bench/results/f4c-global-kv-pool-decision-2026-07-31.json`;
  `tests/verification/test_global_kv_pool_decision.py`;
  `docs/{design/m7-productionization.md,goals/g5-fleet-scale.md,gpu-runbook.md,roadmap.md}`

### 2026-07-31 — [amendment] P-B3 closes on a pinned real Open WebUI browser flow
- What: replaced the mutable, Kairyu-only demo check with health-gated Kairyu
  and Open WebUI v0.11.0-slim services exposing `default` and `kairyu-auto`.
  Open WebUI and the Playwright 1.60.0 base are immutable linux/amd64 pins; a
  lockfile-built derivative supplies the JS package omitted by the official
  browser image. A bounded three-phase Playwright gate proves fresh signup,
  both model selections and SSE bodies, both UI chats, reload/WebSocket
  persistence, real gateway outage at the proxy and UI, and recovery after
  restarting only Kairyu. Assistant-specific response elements distinguish
  completed replies from submitted user messages, and an isolated unique
  Compose project makes fresh-user state deterministic while protecting the
  normal demo volume. It runs in its own mandatory CI job.
- Why: the previous smoke intentionally started only Kairyu and therefore
  could not support P-B3's one-command fresh-user chat, streaming, or
  reconnect/error claims. Exact release pins and browser-visible evidence make
  the supported product surface reproducible without building a custom chat
  frontend.
- Refs: issue #197; G6 P-B3; m11 D7;
  `deploy/compose/{docker-compose.webui.yaml,config.yaml,webui-orchestrator.yaml,Dockerfile.webui-browser}`;
  `scripts/{webui_smoke.sh,webui_browser_smoke.mjs}`;
  `.github/workflows/ci.yml`; `tests/unit/test_compose_configs.py`

### 2026-07-31 — [progress] PostgreSQL CI waits for final external readiness
- What: replaced in-container `pg_isready` polling with a bounded psycopg
  connection and `SELECT 1` against the exact published host-port DSN used by
  the integration suite. Shell-level regression fixtures prove a transient
  failed external query is retried to success and that exhaustion stops at the
  exact bound, emits container diagnostics, and still cleans up; no
  in-container readiness probe is used.
- Why: the pinned official image briefly exposes an initialization-only
  Unix-socket server, then shuts it down before starting the final TCP server.
  The former probe could pass during that transition and race the first real
  test connection.
- Refs: issue #295; PR #294 Actions run `30593765947`;
  `scripts/postgres_integration.sh`; `tests/unit/test_ci_workflow_policy.py`

### 2026-07-31 — [amendment] F1d closes the cross-process OTel trace
- What: amended m10 D4/A34 and completed an optional W3C-only trace from the
  gateway SERVER request through routing, actual pool selection, replica CLIENT
  call, remote replica SERVER request, and each Conductor stage. The request
  span stays open through the final body; failures retain only exception type,
  cancellation state, and description-free ERROR status. A deterministic
  in-memory fixture and a mandatory separate-container Compose validator prove
  the tree by request/trace/span/parent IDs, service identity, response
  completion, and prompt/output-canary absence.
- Why: isolated local spans did not prove propagation through a deployed
  gateway/replica boundary, while OpenTelemetry's default exception event and
  status description can copy application text into telemetry.
- Refs: issue #178; G5 F1d; m10 D4/A34; `kairyu/telemetry.py`;
  `kairyu/entrypoints/server/middleware.py`;
  `kairyu/orchestration/{orchestrator,conductor,replica}.py`;
  `scripts/verify_otel_trace.py`; `scripts/compose_smoke.sh`

### 2026-07-31 — [progress] Ragged KV writes remove per-request shape compilation
- What: made ragged token, row, page-table-width, and leading-stride bounds
  non-specializing Triton runtime arguments while retaining pool geometry,
  page size, head shape, and payload block as compile-time constants. A
  structural GPU regression protects that boundary. In fresh-cache probes,
  the former kernel compiled again for unseen table widths 61, 64, and 65 in
  67.525/65.437/63.807 ms; the selected kernel handled them after one initial
  compile in 0.199/0.126/0.112 ms. Every K/V write was exact.
- Why: a real 1,024-token prefill trace contained a 70.487 ms GPU-idle interval
  immediately before the first ragged write. The steady kernel itself is about
  0.11 ms, so specializing request-dependent shapes made compile latency, not
  write execution, the dominant cost. Runtime bounds preserve Kairyu's native
  L1 path without copying vLLM's scheduler or execution architecture.
- Refs: issue #156; G2 A6; `kairyu/kernels/paged_kv_write_gpu.py`;
  `tests/gpu/test_paged_kv_write_gpu.py`

### 2026-07-31 — [progress] G2 A6 depth 5 removes the serialized-run comparison confound
- What: the exact TP4 direct-engine ShareGPT token trace measured 674.181
  output token/s at pipeline depth 5 versus the prior 495.811 at depth 1
  (+35.98%). Pinned vLLM remains 798.057 token/s, so the diagnostic ratio is
  now 0.845x and still below the 0.95 target. Kairyu steady decode is already
  approximately equal to vLLM; the measured remaining candidate is L1 prefill
  compile/launch. The HTTP TP4/TP8 matrix remains the binding A6 verdict.
- Why: depth 1 measured a serialized host/device execution policy against
  vLLM's asynchronous in-flight execution, so it could not support a causal
  claim that Kairyu's L3/L2/L1 responsibility split caused the whole gap.
  Matching a useful in-flight depth preserves the product architecture and
  limits further convergence to independently proven L1 mechanisms.
- Refs: issue #156; G2 A6; `bench/g2_a6_vllm_bench.py`;
  `examples/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml`

### 2026-07-30 — [progress] G2 A6 first matched TP4 pair isolates a steady-decode gap
- What: The first clean Qwen3-32B TP4 ShareGPT pair completed all 128 requests
  per arm. Kairyu measured 1.452571 SLO-goodput versus vLLM 3.117408
  (0.466x), TTFT p99 29.614074 s versus 18.188697 s (1.628x), and a
  33.044848 s measurement window versus 20.529874 s. The remaining matrix was
  stopped before spending GPU time on a result that could not meet the pooled
  gate. Fixed-B16 evidence attributes the largest controlled gap to the GPU
  model path rather than tokenization, HTTP, attention, or an OS-jitter rule.
  A stride-aware joint Q/K RMSNorm now preserves the existing BF16 boundary in
  one CUDA-graph-safe launch; focused GPU tests are bit-exact and the real TP4
  teacher-forced Qwen gate remains 252/256 with zero substantive
  disagreements. A packed gate/up candidate was discarded after it reduced
  whole-graph time by only about 0.008 ms and failed the same quality gate at
  249/256 with one substantive disagreement.
- Why: The issue's pooled performance thresholds remain the only binding
  verdict. Stopping a clearly failing matrix and rejecting a numerically unsafe
  low-yield transform preserves both GPU time and product quality without
  inventing a single-run or OS-jitter completion condition.
- Refs: issue #156; G2 A6; `kairyu/kernels/rms_norm_gpu.py`;
  `kairyu/models/attention.py`; `tests/gpu/test_rms_norm_gpu.py`;
  `tests/unit/test_joint_qk_rms_norm.py`

### 2026-07-30 — [progress] G2 A6 live preflight corrects the immutable vLLM identity
- What: The first real formal preflight stopped before any vLLM measurement
  because the immutable CUDA image installs `vllm==0.26.0+cu129` while its
  release tag and startup label are `v0.26.0`. The next fresh vLLM startup
  exposed its exact resolved-backend message as
  `Using AttentionBackendEnum.FLASH_ATTN backend.` alongside the separately
  retained `Using FlashAttention version 2` marker. The operator and replay
  schema now pin those observed immutable-image values; regression fixtures use
  the real message shape. The completed first Kairyu shard remains
  hash-protected for safe resume, and no failed vLLM measurement row exists.
- Why: Package local-version suffixes and enum-qualified logger output are
  provenance facts, not performance requirements. Binding the actual image
  content prevents both false rejection and a weaker tag-only claim without
  changing any of A6's three pooled performance thresholds.
- Refs: issue #156; G2 A6;
  `bench/run_g2_a6_formal.py`; `bench/g2_a6_vllm_bench.py`;
  `tests/verification/test_run_g2_a6_formal.py`

### 2026-07-30 — [progress] G2 A6 performance candidate reaches formal-run readiness
- What: Added canonical-name-preserving packed dense Q/K/V execution, guarded
  SM120 fused RMSNorm, RoPE, and paged-KV-write kernels with exact fallbacks,
  and a long-lived native engine step worker. Native HTTP and process backends
  now tokenize each request once while rejecting unsupported fields and
  `max_model_len` overflow before response headers. The serving stack pins
  uvloop/httptools on Linux, propagates the context limit through TP and P-D
  construction, and preserves shutdown, cancellation, checkpoint, quantized,
  and replacement-module behavior.
- Why: The earlier native path paid repeated projection launches, Python
  thread-pool and tokenization overhead, and avoidable elementwise/KV-write
  launches that stock vLLM does not pay. The guarded execution views retain the
  public module and checkpoint contracts while allowing the measured Qwen3-32B
  path to remove those costs.
- Refs: issue #156; G2 A6; m2 D1; m5 D6;
  `kairyu/models/packed_linear.py`; `kairyu/kernels/rms_norm_gpu.py`;
  `kairyu/kernels/rope_gpu.py`; `kairyu/kernels/paged_kv_write_gpu.py`;
  `kairyu/engine/engine_loop.py`; `kairyu/engine/kairyu_backend.py`;
  `kairyu/engine/zmq_backend.py`

### 2026-07-30 — [amendment] G2 A6 formal comparison becomes replayable and fail closed
- What: Added the committed `bench/run_g2_a6_formal.py` operator and hardened
  `bench/g2_a6_vllm_bench.py` to regenerate the complete pinned trace, verify
  the full live checkpoint before and after execution, retain 31 unique
  synchronized graph warmups per cell, and bind raw environment-session,
  post-start launch/backend/package/GPU, logging, CUDA Graph, and cache-capacity
  evidence. Both arms allocate 8,193 pages: Kairyu reserves one graph scratch
  page and stock vLLM reserves its mandatory null page, leaving 8,192 usable
  pages on each at Kairyu `pipeline_depth=1`. Stock vLLM has explicit async
  scheduling, multiprocessing TP, compile mode 3, disabled custom
  all-reduce/access/request logging, and an actual FlashAttention 2 startup
  marker retained from the SM120 process.
- Why: A performance ratio is not attributable or replayable when trace
  construction, checkpoint bytes, effective KV capacity, runtime HTTP stack,
  resolved backend, graph preparation, or measurement-session identity can
  differ without invalidating the artifact.
- Refs: issue #156; G2 A6; m5 D6;
  `bench/run_g2_a6_formal.py`; `bench/g2_a6_vllm_bench.py`;
  `examples/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml`;
  `tests/verification/test_g2_a6_vllm_bench.py`;
  `tests/verification/test_run_g2_a6_formal.py`;
  `bench/results/env-2026-07-30.json`

### 2026-07-30 — [amendment] M13 pins upstream execution identities and actual process reporting
- What: Corrected the public FA3 capability to the officially supported SM90
  profile and FA4 direct paged KV to SM90/SM100/SM110, retaining device-side
  materialization on SM120. Startup now validates the exact FA3/FA4 public
  versions, function signatures, FA3 paged/variable-length build flags, the
  selected GPU's real capability, and FA4 beta24's environment/global
  architecture cache. The CUDA 13 FA4 extra pins upstream beta24 and its exact
  prerelease CUTLASS dependency; every delegated FlashInfer dependency failure
  names the required `gpu` extra. Selection decisions include architecture; TP
  ranks compare their canonical full execution identity and P-D reports both
  role decisions. The official deploy lifespan eagerly starts `kairyu-proc`
  before serving; its optional versioned startup frame propagates the child's
  actual decision with legacy compatibility, while generation-bound queue
  failure and cancellation-safe process ownership close restart/shutdown races.
- Why: A resolved label alone can hide incompatible paged-KV modes or
  heterogeneous rank capabilities, while parent-side re-selection can
  misreport a child fallback. FA4 beta24 caches one architecture process-wide,
  so declaration-only validation would also let a heterogeneous role fail on
  its first prefill. Exact upstream/device preflight, eager child construction,
  and end-to-end decision propagation fail before serving instead of silently
  running a different kernel contract or leaving an in-flight request hung.
- Refs: issue #277; `docs/design/m13-attention-backend.md`;
  `kairyu/engine/core/attention/flashattention_gpu.py`;
  `kairyu/engine/core/attention_selector.py`;
  `kairyu/engine/core/engine_service.py`; `kairyu/engine/zmq_backend.py`;
  `kairyu/deploy/builder.py`

### 2026-07-30 — [amendment] M13 adds strict, reportable FA3/FA4 phase adapters
- What: Extended the public attention switch and Helm schema to `auto`, torch,
  FlashInfer, FA3, and FA4. FA3/FA4 own prefill while FlashInfer retains paged
  decode and CUDA graphs; `/backends` reports the actual prefill/decode/KV
  composition, versions, architecture rationale, and selection source.
  Explicit choices fail before serving on missing dependencies, unsupported
  architecture, dtype, or shape. Only `auto` may use a reported construction
  fallback. Retained SM120 Qwen3-32B parity and 24 interleaved AB/BA pairs per
  TP4/TP8-local shape show FlashInfer faster in all 48 pairs, so `auto` remains
  FlashInfer rather than promoting FA4. FA3 has API-contract coverage and
  fails closed on this unsupported SM120 host; no FA3 hardware result is
  claimed.
- Why: Kernel availability alone is not a performance policy, and a monolithic
  label would hide the graph-critical decode owner. Profile-specific raw paired
  samples expose order and host/OS jitter without arbitrary timing thresholds,
  while unsupported hardware is neither run nor counted as passed.
- Refs: issue #277; `docs/design/m13-attention-backend.md`;
  `bench/results/attention-backends-qwen3-32b-sm120-2026-07-29.json`;
  `bench/results/attention-backends-serving-qwen3-32b-sm120-2026-07-29.json`

### 2026-07-30 — [progress] Issue #227 passes the clean-source Qwen3-32B TP8 gate
- What: Retained
  `bench/results/issue-227-typed-prompt-qwen3-32b-tp8-2026-07-30.json`
  from clean implementation commit
  `62bd57fa782154455dbaa6445a1e48e373601122` on 8× RTX PRO 6000
  Blackwell. All ten binding checks pass: non-null clean commit provenance,
  exact direct-source/checkpoint/hardware identity, preservation of all 264
  caller token IDs, deliberately non-authoritative display text, identical
  eight-token greedy output IDs/text, exact 264-token usage on both paths, and
  the expected 256-token RadixKV replay hit. Artifact SHA-256:
  `4fa2c57fe2dd7f8723f9dc60ea28972d63832754c72eb74d5ec8badfaf910920`.
- Why: Commit-bound structural evidence closes the provenance gap left by the
  earlier pre-commit diagnostic and proves token ownership, accounting, and
  cache identity on the requested Qwen3-32B TP8 deployment. Wall-clock timing
  remains non-binding because OS and host jitter do not affect these invariants.
- Refs: issue #227; commit
  `62bd57fa782154455dbaa6445a1e48e373601122`;
  `bench/typed_prompt_qwen.py`;
  `bench/results/issue-227-typed-prompt-qwen3-32b-tp8-2026-07-30.json`

### 2026-07-30 — [amendment] Backend requests gain strict text, token, and multimodal prompt types
- What: Added frozen `TextPrompt`, `TokensPrompt`, ordered
  `MultimodalPrompt`/`MultimodalItem`, one strict tagged codec, and
  domain-separated token fingerprints. Native Kairyu, the process service, the
  vLLM adapter, offline APIs, and `/v1/completions` now preserve caller-owned
  token IDs without encoding or stringification and report their exact count.
  Native IDs are validated against the backend vocabulary before scheduler
  mutation. Multimodal data is retained by the codec but rejected before every
  current backend dispatch. OpenAI Chat capabilities remain explicitly
  text-only. Replica placement bypasses its text-only prefix index for non-text
  prompts while retaining session affinity; a text prefix fingerprint on
  another prompt domain is rejected. Prompt carriers/cache identity in
  `SamplingParams.extra_args`, nested Chat carriers, unsupported content
  parts/images, Mapping-shaped offline prompts, and validator-less backend
  paths now fail before dispatch instead of being flattened, iterated, or
  dropped. Portable unit/compat/server validation passes 2,487 tests with zero
  failures and 20 environment-specific marker cases deselected. A pre-commit
  Qwen3-32B TP8 run on 8× RTX PRO 6000 Blackwell preserves all 264 caller IDs,
  reports 264 prompt tokens on both paths, produces identical eight-token
  greedy output, and records the expected 256-token native cache hit. The
  retained gate now requires a non-null clean source commit, expanded direct
  source hashes, and that exact cache-hit count; its commit-bound rerun remains
  pending. Timing is explicitly non-binding.
- Why: A loose `str | list[int] | dict` boundary would make tokenization
  ownership ambiguous, let Python booleans alias integer RadixKV keys, and
  permit adapters to silently flatten or drop media. Nominal immutable values,
  exact validation, and explicit backend capability failures keep the existing
  token-native core unchanged while making every conversion auditable.
- Refs: issue #227; m1 D1; m8 D1/D6; m9 D3;
  `kairyu/engine/{prompt,backend,engine_loop,kairyu_backend,zmq_backend,vllm_backend,openai_backend}.py`;
  `kairyu/entrypoints/{llm,async_engine,chat_template}.py`;
  `kairyu/entrypoints/server/{protocol,app,chat_service,metering,responses_service}.py`;
  `kairyu/sampling_params.py`;
  `bench/typed_prompt_qwen.py`;
  `tests/{unit,compat,server}/`

### 2026-07-30 — [progress] m17 A18-A20 close decode page-table reuse on real Qwen3-32B TP8
- What: Retained the 243,071-byte formal artifact from clean implementation commit `78d87a5a902edf4876901d2fd1c3ae3880393cc6` on 8× RTX PRO 6000 Blackwell. All 11 live gates and all 15 independent replay gates pass: exact source/checkpoint/hardware/topology provenance, identical outputs and structural stats on every rank, legacy allocation/full-copy counts, bounded cache upload/copy reductions, zero graph fallback, and zero live ownership after release. Per 31-step run, legacy performs 31 outer plus 248 row allocations, writes 9,024 elements, and copies 248 graph rows/15,872 elements; cache performs zero steady storage allocations, uploads 37 rows/165 elements, and copies 23 graph rows/527 elements. Artifact SHA-256: `e9476591f4c64b319bcdcbf8658870db8c88857d82afe64b5ddeae9a11a70a70`.
- Why: Binding structural evidence proves the selected bounded cache removes steady decode page-table allocation and most H2D/D2D metadata traffic without stale ownership or output changes. The balanced-order diagnostic median improved from 0.823584 to 0.814709 seconds wall time and from 301.12 to 304.40 token/s, but timing remains non-binding because host and OS jitter are outside the correctness contract.
- Refs: issue #229; m17 A18-A20; commit `78d87a5a902edf4876901d2fd1c3ae3880393cc6`; `bench/results/issue-229-page-table-cache-qwen3-32b-tp8-2026-07-30.json`

### 2026-07-30 — [amendment] m17 A18-A20 bound decode page-table reuse to row ownership
- What: Added one geometrically grown page-table tensor per runner/device, stable request/lane signatures, minimal safe dirty-range H2D updates, and independently counted graph-static D2D updates. Ordinary rows use lane 0 and flattened verification rows use request-local lanes 1..k. Growth preserves known data D2D; shape shrink/regrow treats hidden columns as unknown. Release clears every cache signature and leaves an ordered graph-row tombstone, so the next real row rewrites fully and a padding row is filled with the reserved scratch page before replay. Cache OFF omits both storage and signatures, preserving the exact former allocation/full-copy path. All-rank TP mode/stats probes use bounded tensor gathers and fail closed. The formal Qwen3-32B TP8 runner binds clean source/checkpoint/hardware/topology, exact output parity, allocation/upload/copy reductions, zero graph fallback, and zero live ownership after release; timing is diagnostic and its artifact remains pending.
- Why: Rebuilding one device tensor plus one temporary tensor per row and then copying a full graph rectangle made steady decode metadata work proportional to batch and width. A request-keyed tensor map would retain unbounded/stale ownership, while clearing only graph metadata after release can replay old page IDs into newly freed KV pages. One bounded storage object plus explicit ownership/tombstone state removes steady allocation and copy work without weakening CUDA Graph address or page-reuse safety.
- Refs: issue #229; m17 A18-A20; `kairyu/engine/core/{step_executor,model_runner,worker,spec_runner,tp_runner}.py`; `bench/decode_page_table_cache_qwen.py`; `tests/unit/test_{step_executor,model_runner_page_table_cache,decode_page_table_cache_bench,tp_worker}.py`

### 2026-07-30 — [amendment] Batch retention gate binds object lifetime, not allocator bytes
- What: Corrected the preceding local-only validation record after the initial GitHub Python 3.11 job reported one failure while Python 3.12 was cancelled by matrix fail-fast. Replaced the fixed 4 MiB `tracemalloc` threshold with a direct bound on simultaneously live parsed input rows across the fixed producer queue and consumer pool, and disabled matrix fail-fast so both supported Python versions always report an outcome. The replacement and workflow-policy tests pass locally under Python 3.11.15 and 3.12 with coverage; the new GitHub run remains the merge authority.
- Why: Allocated-byte peaks vary with the Python runtime and coverage instrumentation and cannot be a deterministic correctness gate. Parsed-row lifetime directly proves that the streaming pipeline does not retain input-size-proportional line objects, while the separately tested duplicate-ID set remains intentionally input-sized.
- Refs: issue #286; PR #287; `.github/workflows/ci.yml`; `tests/server/test_batches.py`; `tests/unit/test_ci_workflow_policy.py`

### 2026-07-30 — [amendment] CI outcomes become locally reproducible and fail closed
- What: Replaced label/provenance-only F1b/F1c/F2a/F2b decisions with direct locally runnable commands; made environment suites explicit and prerequisite-checked; added repository-wide selected-skip rejection; restored distributed-test collection; and separated Helm, PostgreSQL, vLLM, Docker, and GPU applicability from the portable CPU suite. F2a/F2b now bind clean committed source, replay integrity, workload coverage, routing, lease-safety, and state invariants while recording OS-scheduling-dependent latency/goodput checks as integrity-protected non-binding diagnostics. Compose now proves killed-replica ejection, unhealthy convergence, ten consecutive successes, restart, and healthy reintegration without requiring a slower-prober 502 race. Local validation passed 3,060 portable CPU tests at 87.85% coverage, 130 GPU tests, six real Docker-runner tests, real PostgreSQL and Helm suites, Compose recovery, and WebUI smoke with zero selected skips.
- Why: GitHub-specific labels, run identity, stale artifact selection, hidden pytest directory exclusions, and successful all-skip exits could report green without product evidence. Conversely, hosted-runner timing jitter and requiring an observable failure before a faster health ejection could reject correct behavior. The revised contract makes applicable failures binding, unavailable prerequisites explicitly not executed, and timing visible without treating host noise as product correctness.
- Refs: issue #286; `.github/workflows/{ci,f1b-rollout,f1c-gateway,f2a-prefix-routing,f2b-kv-event-chaos}.yml`; `tests/conftest.py`; `scripts/test_prerequisites.py`; `scripts/{compose_smoke,helm_integration,postgres_integration}.sh`; `scripts/gpu_gates/`; `bench/{prefix_routing_f2a_bench,kv_event_f2b_bench}.py`

### 2026-07-30 — [progress] m17 A13-A17 close on real Qwen3-32B TP8 evidence
- What: Retained the 220,596-byte formal artifact from clean implementation commit `5dc7dd1591b37b8685fa7c6df6a94c8b8481574d` on 8× RTX PRO 6000 Blackwell. All 16 live gates and all 21 independent replay gates pass: exact source/checkpoint/hardware/tokenizer provenance, all-rank target/model/backend counts, eager and CUDA Graph OFF/ON ABBA/BAAB completeness, zero graph fallback, end-to-end plus fixed-target token parity, and poisoned-slot KV write coverage. Artifact SHA-256: `58ec81de2a7a1e89dbf7ced1d6f223039037c80be34cf743c1f4939aa10e66c9`.
- Why: Structural counts prove that 32 fixed target positions collapse from 32 model calls to one on every rank without weakening correctness. Diagnostic end-to-end throughput rose from 12.95 to 260.16 token/s in eager and 12.40 to 387.48 token/s with CUDA Graph. The objective eager microcomparison retained all 32 selected tokens and measured 59.004 ms flattened versus 75.561 ms native ragged CUDA time (0.7809x); none of these timing values decides the verdict.
- Refs: issue #215; m3 §4.1; m17 A13-A17; commit `5dc7dd1591b37b8685fa7c6df6a94c8b8481574d`; `bench/results/issue-215-batched-spec-verify-qwen3-32b-tp8-2026-07-30.json`

### 2026-07-30 — [amendment] m17 A13-A17 close batched speculative target verification
- What: Compatible `ModelRunner`s now score every draft plus bonus/correction position across a scheduler step in one flattened decode-shaped model invocation. The paged runner validates unique physical KV writes, uses total target rows for CUDA Graph buckets, and carries every target position through a fixed variable-width rank-0 TP packet. MLA/DeepSeek and undeclared custom runners select the former sequential path before execution. Acceptance tests bind logical KV/rollback/shortfall behavior, while structural counters and a clean-source Qwen3-32B TP8 runner bind target/output parity, all-rank call counts, and zero graph fallback; timing and cross-kernel BF16 distance are diagnostic.
- Why: The previous Python loop performed one complete target model call per draft position, eliminating most speculative-decoding value. Flattened tensor decode is also the objective eager choice on this hardware: over the identical full-Qwen 8-request/32-position geometry it preserved all selected tokens and measured 59.004 ms median CUDA time versus 75.561 ms for native ragged prefill (0.7809x). Explicit capability selection avoids regressing MLA or caller-supplied one-token runners, while non-binding timing prevents OS jitter from deciding correctness.
- Refs: issue #215; m3 §4.1; m17 A13-A17; `kairyu/engine/core/{spec_runner,model_runner,torch_runner,worker,step_executor}.py`; `kairyu/engine/core/attention/flashinfer_gpu.py`; `kairyu/engine/kairyu_backend.py`; `bench/batched_spec_verify_qwen.py`; `tests/{unit,gpu,dist}/`

### 2026-07-30 — [amendment] Fugu code scoring gains a content-addressed container boundary
- What: Added a pluggable execution-runner contract and explicit benchmark config, fingerprint, CLI, methodology, and run-environment metadata. Local execution retains the trusted-development behavior while reaping descendants and bounding output. The Docker runner accepts only immutable image identities, completes creation and cleanup ownership transfer before starting code, times an attached execution, and removes the exact returned container ID on normal, failed, cancelled, and timed-out catchable paths. Its boundary disables network and daemon logging, uses a read-only root and input mount, drops capabilities and privileges under UID/GID 65534, and bounds CPU, memory/swap, PIDs, `/work`, `/dev/shm`, output, numerical-library threads, and wall time. The supplied hash-pinned Python 3.12 NumPy/HDF5 image and mandatory CI job run the real six-case cross-runner/security/resource/cleanup conformance gate without skips.
- Why: The previous host-Python subprocess contained ordinary runaway code but was explicitly not a security boundary for unattended model-generated benchmark programs. Content identity, fail-closed runner selection, effective-state measurements, exact lifecycle ownership, and disclosed supervisor/daemon/host-user assumptions make the stronger boundary auditable without calling an unavailable or skipped environment a pass.
- Refs: issue #210; `kairyu/bench/{execution,sandbox,types,config,runner}.py`; `kairyu/bench/adapters/{base,livecodebench,scicode}.py`; `deploy/bench/`; `tests/bench/test_bench_exec_{runners,docker}.py`; `.github/workflows/ci.yml`; `docs/benchmarks.md`

### 2026-07-30 — [evidence] m13 D1 batched prefill closes on Qwen3-32B TP8
- What: Retained `bench/results/issue-224-batched-prefill-qwen3-32b-tp8-2026-07-30.json` from clean implementation commit `2fcd9be` on 8× RTX PRO 6000 Blackwell. One fixed B=8 cold-prefill group produced the exact same eight first tokens in sequential and batched modes. Every rank records model calls 8→1, FlashInfer plans 8→1, layer runs 512→64, and sequential rows 8→0; rank-0 CUDA events fall 48,373→6,037. All 11 live checks and the independent stored/source/hardware/checkpoint replay pass. Artifact SHA-256: `96c2450cea3711fc941ee44a7f0aec9202323cc4759fb97ac8f0d21ab65927d8`.
- Why: The exact checkpoint, clean start/end source binding, eight-rank NCCL topology, all-rank counters, and raw token parity prove that the production TP path removes request-proportional prefill chains without crossing KV ownership. The diagnostic wall reduction from 15.153 to 2.907 seconds is not a verdict, so scheduler/OS jitter cannot create a false pass or failure.
- Refs: issue #224; m13 D1; GPU-day C4; commit `2fcd9be`; `bench/batched_prefill_qwen.py`; `bench/results/issue-224-batched-prefill-qwen3-32b-tp8-2026-07-30.json`

### 2026-07-30 — [amendment] m13 D1 completes native cross-request prefill batching
- What: Added a validated ragged `PrefillBatch`, vectorized request-owned KV writes, a native FlashInfer `attend_prefill` contract, one flat dense-model execution for compatible scheduled chunks, strict sequential fallback, and all-rank TP mode/counter diagnostics over the bounded model communicator. CPU tests cover mixed lengths, shared cached prefixes, chunk boundaries, real Scheduler preemption/page reuse, single-request fallback, and KV/token parity. Skip-free SM120 and real NCCL TP2 gates bind FlashInfer plan/run reduction and exact first-token parity. The formal Qwen3-32B TP8 harness pins clean committed source, the exact 17-shard checkpoint at start/end, eight UUID/PCI identities, raw schedules/tokens, all-rank counters, and stored-verdict replay; its artifact remains pending.
- Why: Per-request prefill launched a complete projection/attention/MLP chain for every concurrent prompt and left bursty GPU capacity unused. An explicit native capability keeps compatibility backends honest, page-granular ownership prevents cross-request KV corruption, and structural launch counts avoid turning OS jitter into a false performance verdict.
- Refs: issue #224; m13 D1; GPU-day C4; `kairyu/engine/core/{prefill,kv_pool,model_runner,worker}.py`; `kairyu/engine/core/attention/`; `kairyu/models/{attention,llama}.py`; `bench/batched_prefill_qwen.py`; `tests/{unit,gpu}/test_batched_prefill*`

### 2026-07-30 — [amendment] m16 A12 preserves canonical names through parallel binding
- What: Replaced TP/SP parameter-owning wrappers with execution bindings on the original canonical modules, and changed EP ownership to a global-index `ModuleList` whose remote experts are `None` holes. Post-bind `state_dict`, `named_*`, `get_*`, tied weights, adapter lookup, quantization context, and checkpoint loading now share native HF paths. Rank-local state round-trips under the same topology; requesting a complete HF export from one sharded rank fails with deterministic topology detail. Checkpoint slicing now combines contextual TP placement with exact physical member layouts declared by dense, FP8, INT8, AWQ, GPTQ, NVFP4, and auxiliary router state instead of parsing name suffixes. Row-parallel bias remains registered canonically but is omitted from each local GEMM and added exactly once after reduction. Skip-free gates pass for the 14-case naming contract, seven deployment-preflight formats/architectures, three CPU multi-process EP/SP cases, eight real EP/SP/TP/FP8 NCCL cases, one real TP CUDA-graph capture/replay case, and 28 real SM120 quantized kernel/full-model cases.
- Why: Wrapper-owned `local`, `local_experts`, `embedding`, and `norm` paths made working serving models incompatible with checkpoint tooling, adapters, post-wrap inspection, and contextual quantization. Canonical ownership removes that split identity without increasing rank-local tensor ownership or changing collective behavior; typed physical layouts also prevent packed formats and DeepSeek router metadata from being misvalidated. Omitting bias before the local BF16/FP16 GEMM preserves the established reduce-then-bias numerical contract, which post-hoc subtraction cannot reconstruct after rounding.
- Refs: issue #233; m16 D2/A12; `kairyu/models/{parallel,moe,moe_parallel}.py`; `kairyu/quant/linear.py`; `kairyu/kernels/quant_gemm_gpu.py`; `kairyu/deploy/validation.py`; `tests/unit/test_{parallel_names,parallel_shard,deployment_validation}.py`; `tests/{dist,gpu}/`

### 2026-07-30 — [amendment] m14 binds projection identity and hardware at construction
- What: Added an immutable contextual linear-construction contract carrying checkpoint-canonical name, semantic role, target/EAGLE/MTP scope, layer/expert identity, logical device/dtype, TP rank and shard geometry, and a one-time kernel-capability snapshot. Dense, attention, MLA, routed/shared MoE, output-head, and draft projection sites now provide that context without changing local module or state-dict names. The default policy preserves checkpoint `ignore` rules and established dense/excluded roles; specialized policies may select only an explicitly compatible checkpoint format and concrete kernel. Runtime factories bind fused CUDA callables before serving and report deterministic context-rich errors instead of silently falling back. The public three-argument factory remains compatible. The skip-free contextual CPU gate passes 23/23, and the actual RTX PRO 6000 SM120 kernel/full-model gate passes 28/28 across FP8, INT8, AWQ, GPTQ, and NVFP4.
- Why: Projection shape alone cannot safely select heterogeneous checkpoint formats or hardware kernels. Binding stable identity and actual placement/capabilities once prevents checkpoint-name drift, per-forward policy work, accidental CPU-oracle/dense fallback on CUDA, and unsupported format execution while keeping legacy callers and checkpoints loadable.
- Refs: issue #228; m14 §8; `kairyu/quant/linear.py`; `kairyu/models/{attention,layers,mla,moe,eagle,mtp}.py`; `kairyu/models/{loader,parallel}.py`; `tests/unit/test_linear_factory_context.py`; `tests/gpu/{test_quant_kernels,test_quant_full_model_gpu}.py`

### 2026-07-30 — [progress] m18 D3 P-D overlap validates on real Qwen3-32B
- What: Retained `bench/results/issue-223-pd-overlap-qwen3-32b-rtxpro6000-2026-07-30.json` from clean implementation commit `a08c416` on two peer-accessible RTX PRO 6000 Blackwell GPUs. The blocking and deferred production backends each completed two rounds, eight outputs, and 128 generated tokens with exact token/text parity. A Qwen3-32B-layout 64-layer P2P probe copied 134,217,728 bytes with identical source-before, source-after, blocking-destination, and deferred-destination SHA-256 `438ce11a438299fd87132d874a96871e5383d274a7b84e13085069e55edcc2de`. Blocking role work begins after the copy interval; deferred source and destination role work both overlap it. Deferred returns with one incomplete completion and no publication, then finishes with zero pending or settled outcomes. All ten live checks and the independent stored replay pass. Artifact SHA-256: `215b3f5068ee0e8ba39c048fecfd01f5d4478120e60f86dd5cb2a69aa9923748`.
- Why: The real checkpoint and raw KV evidence establish output safety, consumer dependency, physical completion ownership, and useful device overlap without treating host timing as correctness. Across the two fixed-order rounds, the diagnostic-only median deferred/blocking ratios are 0.9942 wall time, 1.2065 TTFT, 0.9948 completion latency, and 1.0059 output token/s; their mixed direction is exactly why OS/runtime/clock/thermal-sensitive short timing is recorded but non-binding.
- Refs: issue #223; m18 D3; commit `a08c416`; `bench/pd_overlap_qwen.py`; `bench/results/issue-223-pd-overlap-qwen3-32b-rtxpro6000-2026-07-30.json`

### 2026-07-30 — [amendment] m18 D3 retains physical P-D completion ownership
- What: Replaced stream-ordering-as-completion with an opaque retained completion queried by the production P-D coordinator before destination KV publication, decode adoption, source commit, or page release. Cross-device CUDA handoff now uses dedicated source-copy and destination dependency/completion streams, requires peer access, and exposes independently probed `pd_prefill_device`, `pd_decode_device`, and a serialized `pd_defer_handoff=false` control. Every mutation boundary reconciles partial success; uncertain completion or cleanup poisons/retains ownership and fails closed. Added exact byte, timeline, role/backend, retry, cancellation, partial-publication, and completion-loss coverage plus a clean-source Qwen3-32B evidence harness. The formal artifact remains pending and is not counted as passing evidence.
- Why: A CUDA stream wait orders future device work but does not prove physical completion to host-side ownership code, and a destination-only stream serializes source work rather than creating useful overlap. Publication or page reuse before a queried completion can expose incomplete KV; exception paths that lose an allocation or source lease can make retry unsafe.
- Refs: issue #223; m18 D3; `kairyu/engine/core/{handoff_stream,pd,pd_factory,pd_remote}.py`; `kairyu/engine/kairyu_backend.py`; `bench/pd_overlap_qwen.py`; `tests/{unit,gpu,bench}/`

### 2026-07-30 — [evidence] m16 A3 TP sampling ownership closes on real TP2/TP8 and Qwen3-32B
- What: Retained two independently replayed artifacts from clean source commits on 8× NVIDIA RTX PRO 6000 Blackwell Server Edition. The TP8 NCCL protocol artifact from `dcd641a` records 256 worst-rank samples per B=1/8/16 cell, exact raw overwrite/equality/divergence behavior, unique UUID/PCI provenance, and rank-0 broadcast p95 of 0.078400/0.070816/0.075648 ms against the fixed 1 ms ceiling. The production Qwen3-32B artifact from `70b887f` records complete TP1/TP8 output and raw logprob evidence for six mixed requests and 43 tokens per degree, plus the post-generation topology: TP1 local owner/sampler; TP8 gloo control, NCCL model group, rank 0 owner/sampler, and seven passive sampler-free followers. Free-running equality is 41/43 and remains diagnostic. The binding 41-token common-prefix selected-logprob maximum is 0.148189 and the first-divergence direct reciprocal cross-selected maximum is 0.101015, both below 0.25. Every binding check and stored replay passes. Artifact SHA-256: protocol `6b9e9df9b5f67b4542c9529abc08bc5b6bfd3a2f4cef08d0db455a2e5436de90`; Qwen `2180ef16cd2c5891b43abdfac99e7161ef3c1afa9ae44668cc986901dfa5e6a0`.
- Why: These artifacts prove the chosen single-owner protocol on the actual eight-rank NCCL topology and the deployment checkpoint without turning scheduler jitter or BF16 reduction-order near-ties into false failures. The separate real TP2 injected test remains the binding proof that every rank adopts the canonical packet and consumes it on the next decode.
- Refs: issue #225; m16 A3/D4; `bench/results/issue-225-tp-sampling-{comm-rtxpro6000,qwen3-32b-rtxpro6000}-2026-07-30.json`; `tests/gpu/test_tp_sampling_owner_nccl.py`

### 2026-07-29 — [amendment] m16 A3 makes rank 0 the sole TP sampling owner
- What: Replaced all-rank sampling and sampled-result comparison with one protocol in both the production SPMD runner and in-process facade. Rank 0 alone owns RNG, penalties, grammar, logprobs, and public output materialization. Followers execute a passive model/KV path, receive one fixed int64 token slot per scheduled chunk on the model communicator, and adopt its device scalar before returning to the next control receive. Eager followers skip the replicated `lm_head`; every follower skips sampler state and public D2H. The packet retains partial-prefill sentinels, seeded/mixed/structured/speculative state, and fatal protocol diagnostics. A model-subgroup abort plus peer-first reap bounds post-model/pre-packet rank-0 failure teardown. The real TP2 injection gate binds packet adoption and use by the next decode; TP8 evidence independently binds NCCL overwrite plus complete rank-0-owner/passive-follower topology. Cross-degree Qwen evidence treats free-running TP1/TP8 equality as diagnostic and instead binds complete finite raw records, verified rank/ownership topology, common-prefix distribution compatibility, and direct reciprocal cross-selected logprob tolerance at the first divergence. Its direct-venv launch path discovers `ninja` beside the active Python executable before FlashInfer JIT, avoiding a shell wrapper that changes GPU visibility.
- Why: Independent rank sampling duplicates stateful work and can advance future KV/model state from a locally divergent token before a later Python result comparison detects it. Rank-0 authority makes one token canonical by construction, removes sampled-result object traffic, keeps the decode dependency device-side, and remains compatible with a future vocab-sharded head. Cross-degree free-running equality is not a correctness invariant because one reduction-order near-tie changes every later prefix; distribution comparisons are meaningful only while prefixes remain aligned. A worst-rank p95 sanity ceiling is binding for the tiny collective; p99/max/tail counts remain visible but non-binding because late host launch jitter affects every aligned CUDA event and is not protocol correctness.
- Refs: issue #225; m16 A3/D4; `kairyu/engine/core/{model_runner,worker,tp_runner}.py`; `bench/tp_sampling_owner_{bench,qwen}.py`; `tests/{unit,dist,gpu}/`

### 2026-07-29 — [evidence] m8 D6 process wire is linear in output length
- What: Retained `bench/results/proc-wire-delta-2026-07-29.json` from clean implementation commit `5c634ee`. Across 128/256/512/1,024 output tokens, legacy cumulative frames total 504,382/1,972,335/7,798,943/31,012,271 bytes while wire v2 totals 43,419/87,559/177,099/356,199 bytes. Legacy doubling ratios are 3.910/3.954/3.976 (empirical exponent 1.97–1.99); v2 ratios are 2.017/2.023/2.011 (exponent 1.01–1.02). Every token, text character, id logprob, and rich logprob item appears exactly once in v2. Artifact SHA-256: `02054d9def30281f493c50ac6a774069b51f9eaca6178374609e3aa31021fb0f`.
- Why: Exact msgpack frame bytes prove the requested O(output length) boundary independently of wall-clock scheduling and OS jitter, while the legacy production encoder remains the quadratic control. The source/dirty checks and exact payload oracle prevent a smaller-but-incomplete delta from passing.
- Refs: issue #212; m8 D6; `bench/proc_wire_bench.py`; `bench/results/proc-wire-delta-2026-07-29.json`; `tests/verification/test_proc_wire_bench.py`

### 2026-07-29 — [amendment] m8 D6 negotiates a linear process-result wire
- What: Replaced `kairyu-proc`'s per-step cumulative output/text/logprob retransmission with per-request wire v2: one sequence-0 snapshot followed by offset-checked deltas. Empty non-terminal events are suppressed; terminal exact detokenization may replace one suffix; generate reconstructs all deltas without materializing cumulative text until terminal, while stream materializes only at its cumulative public yield. A client-generated internal wire request ID and `stream_id` bind add/abort/result/error generations and prevent stale v1 or v2 events from crossing immediate public-ID reuse. An omitted sampling seed is made explicit from that public ID, retaining deterministic output. Missing version retains the cumulative v1 path in both rolling-upgrade directions. Added long process parity, malformed sequence/offset/version/metadata, cancellation/reuse, and deterministic msgpack byte-volume coverage.
- Why: Retransmitting every cumulative prefix makes serialized bytes quadratic in output length and consumes the same host/process boundary needed by streaming and TPOT-sensitive serving. Version negotiation avoids a flag day, while sequence, offsets, and stream generations fail closed instead of silently reconstructing corrupt or stale output.
- Refs: issue #212; m8 D6; `kairyu/engine/core/engine_service.py`; `kairyu/engine/zmq_backend.py`; `bench/proc_wire_bench.py`; `tests/unit/test_{proc_wire_protocol,zmq_backend}.py`; `tests/verification/test_proc_wire_bench.py`

### 2026-07-29 — [progress] G2 A7 closes on real Qwen3-32B TP4/8 evidence
- What: Retain the four-cell Qwen3-32B result on 8× RTX PRO 6000: TP4 direct/gateway 87.6725%/87.3531% and TP8 direct/gateway 87.6725%/87.3531%, each with 512/512 successful requests. The independent verifier re-hashes 2,058 raw rows and passes all eight binding trace, usage, topology, and provenance checks.
- Why: Matching engine-token accounting at TP4 and TP8 proves the issue #157 KV-hit invariant through both the replica and production gateway without importing latency, OS-jitter, output, affinity, or repeat-count criteria.
- Refs: issue #157; G2 A7; m5 D1/D6; `bench/results/g2-a7-kv-hit-qwen3-32b-rtxpro6000-2026-07-29/`; measurement commit `b5bcaf10df99ee60e92a291afd4a2764c232a1f2`

### 2026-07-29 — [progress] G2 A7 gateway admits its declared trace
- What: Pin the example gateway's default tenant to a 600,000-token one-run burst after the first formal TP4 gateway arm reached the generic 200,000-token capacity at request 406; retain its sustained 200,000-token/minute rate and rerun from a fresh engine.
- Why: A7 must measure all 512 fixed requests through the gateway, not turn the unrelated generic tenant quota into a partial KV result or hide the rejection with a client retry.
- Refs: issue #157; `examples/qwen3-32b-multi-gpu/auto-gateway.yaml`; `bench/tp_kv_hit_g2_a7_bench.py`

### 2026-07-29 — [amendment] G2 A7 binds to real engine usage at TP4/8
- What: Replaced the label-only CPU A7 procedure with a deterministic Qwen3-32B real-engine harness over TP4/8 direct and single-replica-gateway paths. The 64-session × 8-turn trace retains exact 512-token shared and 128-token appended geometry. The offline verifier pools only engine-originated `cached_tokens / prompt_tokens`, requires the four written strict >80% results plus complete HTTP usage, exact `/backends` topology, raw integrity, and stable source/config/GPU provenance. The CPU RadixKV result remains a geometry diagnostic. Gateway session-affinity counters are retained beside the engine metric but are non-binding; latency, OS jitter, output equality, and repeat counts are outside this deterministic accounting gate. P-D remains an A10 concern rather than an issue #157 A7 condition.
- Why: `bench/multiturn_prefix.py --tensor-parallel N --pd` only recorded labels around one CPU RadixKVCache and could neither exercise TP sharding nor traverse the gateway. Binding to response usage from the real sharded engine proves the stated invariant without importing unrelated timing or output-quality requirements.
- Refs: G2 A7; m5 D1/D5/D6; issue #157; `bench/tp_kv_hit_g2_a7_bench.py`; `bench/multiturn_prefix.py`; `examples/qwen3-32b-multi-gpu/compose.yaml`; `tests/verification/test_tp_kv_hit_g2_a7_bench.py`

### 2026-07-29 — [amendment] D3 adds credential-free offline deployment preflight
- What: Added `kairyu validate <deployment.yaml>` with stable 0/1 exit status and deterministic aggregation of DeploymentSpec, backend capability, linked orchestrator/DAG, `.jinja`, native model metadata/checkpoint tensor-shape, and tokenizer errors. The validation-only loader context retains tenant topology checks without resolving API-key environment values; network and hardware remain explicit skipped/indeterminate classes. Backend constructors, server startup, tensor materialization, model execution, subprocesses, and probes are never invoked. The command covers only artifact links the current DeploymentSpec can declare; standalone adapter, grammar, and benchmark links remain not applicable rather than being invented.
- Why: Operators otherwise discovered cross-file and backend mismatches only after serving began to construct resources. A credential-free, side-effect-free preflight catches the safely knowable failures before rollout while preserving runtime ownership of secrets, remote reachability, and hardware acceptance.
- Refs: m7 D3; issue #230; `kairyu/deploy/validation.py`; `kairyu/deploy/spec.py`; `kairyu/entrypoints/cli.py`; `docs/deployment.md`; `tests/unit/test_{deployment_validation,validate_cli}.py`

### 2026-07-29 — [amendment] Deployment server schema stops inheriting runtime settings
- What: Amended m7 D3 so the versioned DeploymentSpec `ServerSection` independently declares its ten existing YAML fields, forbids unknown keys, and translates the eight runtime fields explicitly to `ServerSettings`. Builder and tenant preflight use that one mapping. Schema/default snapshots, full-key YAML round-trip, default/runtime parity, unknown-key rejection, and builder lifecycle tests preserve current artifacts while making future runtime-only additions an explicit design choice.
- Why: Inheriting the runtime settings model coupled an external deployment artifact to internal API evolution: a runtime-only field could silently change the accepted/generated YAML schema. An independently owned model plus enumerated conversion preserves compatibility while forcing any future public configuration change to be deliberate and reviewable.
- Refs: m7 D3; issue #231; `kairyu/deploy/spec.py`; `kairyu/deploy/builder.py`; `tests/unit/test_deployment_spec.py`; `tests/server/test_serve_builder.py`

### 2026-07-29 — [progress] F2d closes unbiased prefix-weight replay
- What: The exact-source A33 formal artifact and independent offline verifier passed every source, retained-input, split-isolation, complete-policy replay, production-route join, balanced-panel, train-only selection, frozen-held-out, success, and integrity check. Seven normalized policies replayed every request in 48 training families; the tuner selected `λ=1.0` before held-out execution. Against the declared `λ=0.25` baseline on 16 disjoint held-out families (256 requests per arm), mean TTFT was 4.43359375 versus 8.5 deterministic virtual ticks. All 5,888 production placement rows joined one-to-one to successful outcomes. p95 and 176/256 action differences remain diagnostic only.
- Refs: G5 F2d; m10 D8/A33; issue #182; source `86dde278d0f2a093bde64f5d1d9cba9aca9e1221`; artifact `bench/results/f2d-prefix-weight-replay-2026-07-29/`; manifest SHA-256 `3205721922fd8c013ae6336aaa4ffcb0a1938a40059e70acb500b5acba86ac3c`; raw SHA-256 `1ccc5ab012e5ee6677f96709ec60cc15ea5db32cefb72360941238ca505c75eb`; router SHA-256 `3296fdd000aede574ea5c3a152ff1ef0f54e204545bfb1f9aa61f7b47c83546f`

### 2026-07-29 — [amendment] F2d replaces chosen-action agreement with full-policy replay
- What: Added m10 D8/A33 and froze the pre-results F2d method. Production `JsonlRouterLog` `kind=replica` decisions join exactly once to `placement_outcome` TTFT rows through `learning/dataset.py`. The policy grid now fixes `α=1`, tunes only the identifiable `λ=β/α`, and declares `λ=0.25` as the baseline. Every candidate replays complete episodes from the same frozen initial cache/background-load state over disjoint training families; the minimum-mean-TTFT candidate is frozen, then only it and the baseline receive identical held-out traces from that state under deterministic virtual time. Closure binds strict held-out mean-TTFT improvement plus complete, zero-failure, balanced-work, one-to-one joined, split-isolated, hash-bound evidence and independent raw replay. Arm execution order is neither binding nor diagnostic. The former extra 10% threshold and p95 gate do not apply; p95 and action differences are diagnostic. F2d remains in progress with no result claimed.
- Why: Candidate-specific chosen-action agreement changes the evaluated request subset and observes only the logging policy's outcomes, so it is selection-biased and cannot estimate unchosen placement rewards; a coverage penalty cannot restore the missing counterfactual state. Queue, load, and cache decisions also alter later requests, making the complete stateful episode—not an isolated placement row—the valid replay unit. Positive common scaling preserves the score ordering, threshold, and ties, so separate `(α, β)` magnitudes are unidentifiable. Family-level train/held-out isolation and a frozen winner prevent evaluation leakage, while deterministic virtual-time replay from the same frozen initial state and evidence-integrity gates preserve an objective CPU acceptance criterion without imposing an unsupported effect size or tail-latency requirement.
- Refs: G5 F2d; m10 D8/A33; issue #182; `kairyu/orchestration/learning/dataset.py`; `kairyu/orchestration/replica.py`; `kairyu/orchestration/router.py`

### 2026-07-29 — [progress] F2c closes real-engine KV-aware TTFT routing
- What: The exact-source Qwen3-32B TP2×4 formal run and independent offline verification passed every F2c check over 512 binding requests with zero failures. Control-to-candidate pooled TTFT p95 fell from 527.957623 ms to 134.357747 ms; candidate/control ratios were 0.2544858548 pooled, 0.2550841404 at the seventh ordered round, and 0.2530080045 by geometric mean. Engine-token cache rate rose from 0.4994645560 to 0.9843917326 with every round noninferior. SLO-goodput ratios were 0.9999979014 pooled, 0.9998437390 at the second ordered round, and 0.9999978783 by geometric mean. Output agreement remained diagnostic at 239/256 (0.93359375); maximum paired receipt skew and schedule lateness were diagnostic at 5.182959 ms and 7.470463 ms.
- Refs: G5 F2c; m10 D6/A32; issue #181; source `80b039b5d429c656871a480c2740740951b29b97`; image `kairyu-f2c@sha256:d2c01580964f461a3d3d2a02ced5303e69c681696d4a38179162084e1624121f`; artifact `bench/results/f2c-kv-aware-ttft-qwen3-32b-2026-07-29/`; raw SHA-256 `4cfcdeba2b7473aa6c2b28409dbf21de23d775d9b08e971beed6bdab875abe64`; trace SHA-256 `51d188671432bf791c02d66d91e6a7d785eb2bd01f64e29a41a62e74f9957dad`

### 2026-07-29 — [amendment] F2c replaces post-treatment output equality with a frozen transcript
- What: The first formal F2c execution stopped at round 1 family 0 on the former exact cross-arm output assertion. A targeted fixed-endpoint reproduction reported fully warm prompt caches (2,544/2,546 cached tokens): each endpoint repeated its own continuation twice, yet B0 and A1 differed; a separate longer family matched between warm and cold arms. The corrected trace now predeclares one family-specific canonical assistant continuation, binds its digest and the resulting turn-2 prompt digest, waits for every turn-1 arm to succeed, and supplies that same frozen transcript to both turn-2 arms without consuming either observed output. Every output digest remains raw evidence, while cross-arm match count, total, and rate are diagnostic only. Prompt identity, paired prompt/completion work, routing, engine usage, provenance, and all formal TTFT/goodput/cache thresholds remain binding.
- Why: The fixed-endpoint result is consistent with a BF16/TP near-tie under different cross-endpoint and cache-population execution shapes; A1/B0 prefix KV may have been formed through different chunk/prefill histories, so it neither establishes semantic cache corruption nor isolates a physical GPU-pair effect. G2 already establishes that free-running greedy sequence equality is not a correctness gate because one moved near-tie changes every later autoregressive prefix. Reusing the observed turn-1 output also made a post-treatment result part of later workload construction. A predeclared common transcript preserves identical turn-2 work and the routing-performance causal comparison without conflating it with free-running numerical identity.
- Refs: m10 D6/A32; G5 F2c; G2 free-running correctness amendment; issue #181; `bench/kv_aware_ttft_f2c_bench.py`; `tests/verification/test_kv_aware_ttft_f2c_bench.py`; `docs/gpu-runbook.md`

### 2026-07-29 — [amendment] F2c freezes a real-engine paired crossover
- What: Added m10 A32 and the pre-results F2c method. The production `ReplicaPool`/`OpenAICompatBackend` path compares `PrefixIndex` with session HRW on four independent Qwen3-32B TP2 endpoints across all eight GPUs. Two disjoint two-replica cohorts run simultaneously and exchange policies for eight rounds; a recorded single-use namespace separates smoke/retry cache roots, and each round uses 16 unique 2,048-word RAG families, cold seeds opposite the measured HRW target, identical paired hints/prompts/outputs, actual turn-1 output in turn 2, and exact eight-token work. Nearest-rank TTFT p95 must improve by at least 30% pooled, at the seventh ordered round ratio, and by geometric mean. SLO-goodput must retain at least 0.99 pooled, at the second ordered ratio, and by geometric mean; engine-token cache rate must improve strictly pooled without a per-round regression. Raw router decisions, engine usage, topology, configuration, model/source hashes, and timing diagnostics are independently replayed.
- Why: Simultaneous cache-disjoint arms plus cohort crossover separate the routing policy from fixed GPU/NUMA and time-order effects, while exact seven-of-eight order statistics and full-sample geometric means remain robust without deleting observations or imposing an arbitrary OS scheduling-skew cutoff. TP2 preserves a validated Qwen3-32B weight/KV/runtime memory margin and still supplies two independently cached replicas per policy. Direct L2 execution proves the D6 production placement path; adding DeploymentSpec exact-KV hash-provider and subscriber lifecycle wiring would conflate the separate D7 product responsibility with this performance gate.
- Refs: m10 D6/A32; G5 F2c; issue #181; `bench/kv_aware_ttft_f2c_bench.py`; `examples/qwen3-32b-multi-gpu/f2c-compose.yaml`; `docs/gpu-runbook.md`

### 2026-07-29 — [progress] F2b closes KV-event freshness and chaos fallback
- What: Exact-source Actions run `30417507859` at `f383806` passed every independently replayed source, runner/Actions provenance, 200-replica ten-by-twenty churn, epoch/sequence, atomic routing, freshness, stream-kill fallback, same-object recovery, and final-state gate. All 500 offered routes were observed: 175 fresh exact, 140 stale approximate, and 185 restored exact. Maximum exact truth age was 232.314498 ms under the 250 ms lease; maximum logical apply, live-wire apply, heartbeat apply, route lateness, selected-route gap, and chaos-action lateness were respectively 0.035925, 51.454668, 50.806312, 3.608193, 21.536138, and 1.166648 ms. The first stale approximate route arrived 251.339950 ms after pause, and complete replay restored the same process and objects in 50.740933 ms after resume. The original Actions artifact and its completed-success run metadata independently replayed green; its 2,196 raw rows are retained byte-identically under `bench/results/f2b-kv-event-retained/`.
- Why: This is the one binding F2b measurement. It proves the corrected strict timing contract without excluding scheduler-delayed observations and binds the retained bytes to the original successful Actions artifact. F1a run `30374404150` and F2a run `30411111758` supplied only their declared prior context and were not rerun; adding the retained artifact is evidence-only and does not repeat F2b.
- Refs: G5 F2b; m10 D7/A31; issue #180; PR #271; Actions run `30417507859`; source commit `f383806`; raw SHA-256 `6a43544f672af438b7eb2d6cb74e35dd907cf357a0191bf42a54e8093ac0a6bb`; `bench/results/f2b-kv-event-retained/`

### 2026-07-28 — [amendment] F2b makes exact KV-event routing recoverable and atomic
- What: Added m10 A31 and the F2b implementation candidate. KV sources now carry cache-lifetime epochs, contiguous sequences, high-water heartbeats, a bounded replay journal, and authoritative snapshot fallback. Retired epochs are held for the process lifetime, and inactive member tombstones reject delayed frames without allowing unknown inactive epochs to displace the active tombstone; cache replacement rotates one publisher's epoch without restarting its object or socket thread, while generation-bound callbacks reject an old cache event that races rotation. Failed exact membership mutations quarantine that replica until a successful forget/register reset. Production exact placement requires one synchronized, lease-valid `route_overlaps` vector computed for every eligible replica under one lock and one clock sample; every gap, expiry, malformed vector, optional-index failure, or unsynchronized feed makes the complete request use the existing approximate trie. The replayable formal fixture reuses F1a's exact 200-replica ten-by-twenty churn identities and F2a's routing precedent, but repeats neither measurement; it adds only 199 sequenced logical feeds, one binding physical ZMQ feed, actual-time kill/restore/replay evidence, and independently reconstructed routing/state oracles. Fresh evidence is bound to the current Actions runner context; retained replay resolves the recorded completed-success run through the Actions API and byte-compares both files with the original Actions artifact.
- Why: Recency-only two-frame PUB/SUB could silently trust a dropped removal, revive an old epoch after replica-ID reuse, or mix exact scores observed at different points during churn. A 500 ms `>` comparison also did not prove the written strict `<500 ms` contract and was vulnerable to scheduler-jitter arguments. The 250 ms route lease, actual monotonic timestamps, and fail-closed atomic bulk observation preserve explicit headroom without excluding or relabeling delayed samples; pause/resume action lateness, offered-route lateness, and selected-route blind spots at or above 500 ms now fail instead of being hidden by catch-up execution.
- Refs: m10 D7/A31; G5 F2b; issue #180; `kairyu/orchestration/kv_index.py`; `kairyu/orchestration/kv_routing.py`; `kairyu/orchestration/replica.py`; `bench/kv_event_f2b_bench.py`; `.github/workflows/f2b-kv-event-chaos.yml`

### 2026-07-28 — [progress] F2a closes prefix routing at 500 replicas
- What: Exact-source run `30411111758` at `c067cb8` passed every frozen source, independent replay, shared-cache, exact-median, full-sample geometric-mean, sign, and placement-p99 gate. Prefix routing delivered a 37.9259x backend-truth cached prompt-work ratio over the non-zero HRW baseline. Across 21 blank-root paired uniform rounds, the goodput-ratio median was 1.002142, the exact 96.0823%-coverage median lower bound was 0.999512, the geometric mean over every round was 1.008610, and 21/21 ratios met the 0.99 floor. Worst-trace placement p99 was 0.145979 ms. Independent replay accepted all 24,709 raw rows and exact provenance. The immutable artifact is retained under `bench/results/f2a-prefix-routing-500-2026-07-28/`, and the evidence-only closure commit does not repeat the measurement.
- Refs: G5 F2a; m10 D6/A30; issue #179; PR #269; Actions run `30411111758`; source commit `c067cb8`; `bench/results/f2a-prefix-routing-500-2026-07-28/`

### 2026-07-28 — [amendment] F2a uses exact median inference for hosted-runner timing
- What: Exact-source run `30410293786` at `918c4fa` passed source, replay, 37.9259x shared cached-work, 0.298553 ms worst-trace placement-p99, and the pre-existing 18/21 sign gate. Blank-hint uniform requests had byte-identical prompts, sessions, HRW selections, reasons, and cache outcomes across all 10,752 pairs; their paired median dispatch difference fell to 0.241 us and median goodput ratio rose to 0.999610. The former Student-t log-mean LCB remained 0.977150 because round 7 alone contained a time-local prefix-arm plateau. F2a now binds an exact distribution-free one-sided median lower bound: for 21 pairs the seventh ordered ratio has 96.0823% binomial coverage and must remain at least 0.99, exactly equivalent to the already frozen 15/21 sign requirement. The geometric mean over all 21 unmodified ratios must also remain at least 0.99 as a magnitude guard. The Student-t LCB remains diagnostic; no round is removed, trimmed, or winsorized.
- Why: Hosted-runner timing was demonstrably non-Gaussian and correlated inside a time-local round: round 7 contributed 86.67% of the apparent all-round dispatch excess, affected placement and common post-processing together for roughly the first 320 requests, and then recovered. Raw evidence cannot distinguish OS steal, throttling, GC, or another runner/runtime cause, so the round remains a complete failure observation rather than being manually classified away. The exact inference treats each paired round, not each request, as an independent experimental unit and makes no Gaussian or symmetry assumption across round ratios; the full-sample geometric mean prevents a resistant location statistic from hiding a few large losses. Applying both frozen rules retrospectively rejects every systematic-tax candidate (`304037`: exact LCB 0.937210 / geometric mean 0.934120; `304069`: 0.978835 / 0.978948; `304091`: 0.980374 / 0.989176) and accepts only the bypass candidate (`304102`: 0.994246 / 0.990814), so the 0.99 floor is not weakened.
- Refs: m10 D6/A30; G5 F2a; issue #179; PR #269; Actions run `30410293786`; source commit `918c4fa`; `bench/prefix_routing_f2a_bench.py`; `docs/goals/g5-fleet-scale.md`

### 2026-07-28 — [amendment] F2a makes blank hints explicitly session-only
- What: Exact-source run `30409184848` at `967b003` passed source, replay, 37.9259x shared cached-work, and 0.248605 ms worst-trace placement-p99 checks, but failed the unchanged uniform gates at 0.983770 median, 0.974901 one-sided LCB, and 6/21 passing rounds. The next candidate makes a blank `CacheHint.prefix_fingerprint` an explicit session-only contract: native `PrefixIndex` preparation and successful publication are both skipped, leaving the existing HRW and queue-depth path unchanged. Non-empty exact roots still enable cross-session prefix placement; requests without any CacheHint retain local prefix discovery; legacy index subclasses retain their original path. F2a shared traffic carries exact prompt-derived roots, while uniform calibration and binding traffic carry blank hints and independently replay the bypass. All 43 changed-scope routing/replay checks pass.
- Why: All 10,752 uniform pairs had identical prompts, sessions, selected replicas, cache misses, and `session_affinity` decisions. Removing the two opposing host plateaus still leaves prefix placement 2.121 us slower, post-selection work 1.238 us slower, total dispatch 3.360 us slower, geometric goodput ratio 0.984896, LCB 0.981748, and only 5/19 rounds at or above 0.99. This is a systematic tax, not OS jitter. A session hint already preserves same-session KV locality; without a declared root, speculative cross-session hashing/publication adds cost with no asserted reuse. Bypassing that work is both faster and a clearer responsibility boundary than weakening the 1% gate.
- Refs: m10 D6/A30; G5 F2a; issue #179; PR #269; Actions run `30409184848`; source commit `967b003`; `kairyu/engine/backend.py`; `kairyu/orchestration/replica.py`; `bench/prefix_routing_f2a_bench.py`

### 2026-07-28 — [amendment] F2a removes the residual uniform cold-path tax
- What: Exact-source run `30406943237` at `9478f40` passed every source, replay, shared-cache, and placement-p99 check but failed the unchanged uniform goodput gates: median ratio 0.982195, one-sided 95% LCB 0.973463, and 2/21 rounds at or above 0.99. Across all 10,752 identical paired requests, prefix placement cost 2.557 us more and success publication cost 2.148 us more on average. The next candidate uses versioned XXH3-64 keys, carries Conductor's root only when a complete 256-character shared chunk makes it byte-identical to local hashing, preserves local fallback for empty/short prefixes and the binding uniform trace, preallocates native replica stores at membership time, and sends successful cold roots through a dedicated fast path. One fully retained/replayed `binding=false` uniform calibration pair precedes the unchanged 21 binding rounds; warm promotion, legacy subclass overrides, failure/cancellation truth, concurrent completion merge, and remove/re-add identity guards remain unchanged.
- Why: Round 0 contained a host episode, but removing it still left a 4.114 us mean dispatch delta, 0.977964 LCB, and only 2/20 passing pairs; the final 16 rounds still differed by 3.595 us, and reversing arm order did not reverse the result. This is implementation work rather than OS jitter, so rerunning the same source or weakening the predeclared 1% criterion would be invalid. After the implementation changes, a four-CPU targeted trace isolated a distinct first-prefix-arm CPython arena episode: prefix-first paid 36.234 us/request, while a fresh prefix pool after one completed prefix arm paid only 0.568 us/request total and 0.228 us/request in placement. A predeclared calibration pair removes that process-start allocation from a steady-state claim; the rejected source still fails even with its first round excluded, so this cannot hide its systematic regression. A pre-contract dirty diagnostic with uniform family hints passed at 37.9259x shared cached work, 0.996050 median, 0.994464 LCB, 20/21 sign rounds, and 0.216723 ms worst p99, but those uniform numbers are optimization diagnostics rather than evidence for the corrected blank-hint local-fallback trace. The corrected contract passes all 58 changed-scope functional and replay checks; its one clean formal run remains. XXH3 follows the vLLM Router class of non-cryptographic routing keys; this index is neither persisted nor used for output correctness, and its 64-bit width remains unchanged.
- Refs: m10 D6/A30; G5 F2a; issue #179; PR #269; Actions run `30406943237`; source commit `9478f40`; `kairyu/orchestration/conductor.py`; `kairyu/orchestration/prefix_index.py`; `kairyu/orchestration/replica.py`

### 2026-07-28 — [amendment] F2a carries one lazy prefix hash chain through success
- What: Exact-source run `30403762900` rejected the first F2a candidate despite a 37.9259x shared cached-prompt-work ratio and 0.326926 ms worst placement p99: the uniform median goodput ratio was 0.944327, its one-sided 95% lower bound was 0.921098, and 0/21 paired rounds met 0.99. The corrective candidate now prepares only a root key for a cold lookup, reuses it on successful cold publication, and materializes one bounded request-local lazy cumulative-hash chain only for a warm candidate before successful full-depth promotion. Native method capabilities are frozen at pool initialization, while subclasses overriding the legacy hash/publication seam retain that path. Singleton reverse-map candidates and insertion-ordered per-replica stores avoid cold-only container work. The 500-replica cold path still performs no overlap fleet scan and one session HRW, while failure, cancellation, concurrent cold completion, and remove/re-add identity guards remain unchanged. Each binding arm now executes a declared 512-request run-in on that same pool and policy immediately before its clock; the verifier binds its deterministic trace digest, actual completed count, positive interval, and placement before the measured arm while no run-in sample enters a gate.
- Why: The retained 23,682 raw rows isolate 7.548 us/request of uniform placement overhead and another 6.798 us after selection to duplicate five-chunk hashing and cold five-key index mutation. Every paired round regressed, and the robust median remained about 5.5% slower after the isolated host pauses were excluded, so OS jitter could not justify weakening the predeclared 1% non-inferiority gate. A root key is sufficient to discover the backend holding the complete successful prompt; deeper approximate keys become useful only after a related warm route. Later changed-hot-path diagnostics exposed a separate fresh-pool run-in asymmetry: hundreds of requests stayed on a slower code/CPU-frequency plateau, and running a different warmup pool did not survive the next pool build. Same-pool per-arm run-in removes that order confound without changing the 1% threshold. The final dirty-source diagnostic then passed every performance gate (37.9259x cached-work ratio, 0.993453 median, 0.992096 one-sided LCB, 17/21 sign rounds, and 0.217870 ms worst p99); only clean exact-source provenance remains for the binding run.
- Refs: m10 D6/A30; G5 F2a; issue #179; PR #269; Actions run `30403762900`; source commit `ededdb1`; `kairyu/orchestration/prefix_index.py`; `kairyu/orchestration/replica.py`

### 2026-07-28 — [amendment] F2a compares identical sessions against successful cache truth
- What: Added m10 A30 and the F2a implementation candidate. Prefix-enabled pools score reverse-indexed warm candidates against a conservative zero cold baseline, prefer reuse at a zero tie, fall back when the warm maximum is negative, use session HRW only to break equal warm scores, publish approximate cache state only after successful completion, and retain reachable root chunks at the per-replica cap. The replayable CPU gate sends identical prompts and per-request session IDs through both 500-entry arms, takes cached prompt-work from the selected mock backend, uses 21 alternating paired uniform rounds with a one-sided 95% log-ratio lower bound plus a 15/21 sign guard, computes <10 ms SLO-goodput from every offered raw dispatch interval and whole-population shared/uniform placement p99, and hashes raw rows plus exact run/source provenance for independent replay.
- Why: The former session-first implementation bypassed prefix scoring for production requests carrying affinity hints, while placement-time `observe` could advertise a failed or cancelled request as warm. A benchmark that removed session hints only from the treatment would therefore manufacture the desired hit-rate gain. F1a already proves the kind deployment path and is not rerun; F2a's distinct claim is selector quality and cost at 500 logical replicas.
- Refs: m10 D6/A30; G5 F2a; issue #179; `kairyu/orchestration/replica.py`; `kairyu/orchestration/prefix_index.py`; `bench/prefix_routing_f2a_bench.py`; `.github/workflows/f2a-prefix-routing.yml`

### 2026-07-28 — [progress] F1c separates retained evidence from its live workspace
- What: Moved F1c's disposable default and CI result directory to `bench/results/f1c-three-gateway-live/` while keeping the closed, independently replayable artifact under `bench/results/f1c-three-gateway/`. The artifact name and verifier contract are unchanged.
- Why: The evidence-only closure commit made the former live output directory tracked. Diagnostic run `30399758298` then failed in 0.18 seconds before any build or measurement because startup cleanup correctly deleted the old result names and clean-source attestation correctly rejected those tracked deletions. Separate directories preserve both immutable evidence and fail-closed source provenance.
- Refs: issue #177; PR #268; Actions run `30399758298`; `.github/workflows/f1c-gateway.yml`; `scripts/kind_gateway_gate.sh`; `bench/results/f1c-three-gateway/`

### 2026-07-28 — [progress] F1c closes three-gateway affinity and shared batch failover
- What: Closed G5 F1c on exact-head source commit `be40b97` with Actions run `30399229234`. All 26 independent replay checks passed: six baseline sessions distributed as two per gateway and stayed sticky, all gateways reconstructed the same 12-replica affinity decisions and PostgreSQL identity, a fence-1 owner on gateway B was killed, gateway A reclaimed only after that exact lease expired with fence 2, 200/200 batch lines completed with no failures, and all three gateways returned byte-identical output. The complete raw/replay artifact is committed under `bench/results/f1c-three-gateway/`; the evidence-only commit intentionally does not rerun F1c.
- Refs: m10 A29; G5 F1c; issue #177; PR #268; Actions run `30399229234`; source commit `be40b97`; `bench/results/f1c-three-gateway/`

### 2026-07-28 — [amendment] F1c attests the registry-to-kind image identity chain
- What: F1c now retains the raw Docker inspection of the pinned PostgreSQL image and independently replays the registry manifest pin → Docker config ID → kind CRI config ID → Pod runtime digest chain. A Pod may report either the original registry pin or a CRI-reported import digest; any digest outside that attested set fails. The live script also fail-fast compares the Docker and CRI config IDs before deployment.
- Why: Exact-head run `30398220981` loaded the same pinned `postgres@sha256:f3bd19…` content as run `30397522862`, but kind emitted a generated `import-2026-07-28@sha256:1b3460…` Pod/CRI repo digest instead of retaining the registry manifest digest. OCI manifest digests and image config IDs are different identities, and a local import may legitimately rewrite the former while preserving the latter. Requiring only the registry digest is therefore unstable; trusting every runtime digest would be too weak.
- Refs: m10 A29; G5 F1c; issue #177; PR #268; Actions runs `30397522862` and `30398220981`; `scripts/kind_gateway_gate.sh`; `bench/fleet_gateway_bench.py`; `tests/verification/test_fleet_gateway_bench.py`

### 2026-07-28 — [amendment] F1c separates causal lease proof from asynchronous observations
- What: Corrected A29 replay without weakening its fencing proof. Digest-pinned Pods accept Kubernetes' tag-only display image only when the runtime `imageID` equals the source pin exactly. Owner-Pod absence remains required, but its polling observation is no longer treated as the actual stop timestamp; replay instead binds the kill request to the active old lease, forbids later old-fence renewals, requires expiry of that exact lease before the higher-fence reclaim, and requires one new-fence terminal commit. The immutable raw artifact from run `30397522862` passes all 26 corrected offline checks.
- Why: The live drill exposed two invalid cross-domain assumptions: kind's local CRI config ID is not an OCI registry manifest digest, and a 250 ms Kubernetes polling observation can occur after a valid PostgreSQL-clock reclaim. The source-pinned Pod `imageID` matched exactly, and the observed sequence was kill request under fence 1, exact lease expiry, fence-2 reclaim 71.306 ms later, Pod-absence observation 45.868 ms after reclaim, renewals, and one in-lease terminal commit.
- Refs: m10 A29; G5 F1c; issue #177; PR #268; Actions run `30397522862`; `bench/fleet_gateway_bench.py`; `tests/verification/test_fleet_gateway_bench.py`

### 2026-07-28 — [amendment] F1c separates gateway affinity from fenced batch ownership
- What: Added A29 and an F1c implementation candidate. Three independently restartable gateways sit behind an explicit `X-Session-ID` SHA-256 HRW load balancer while retaining ReplicaPool's existing HRW placement. A PostgreSQL BatchStore stores ordered file chunks and jobs, claims work with `FOR UPDATE SKIP LOCKED`, renews DB-clock leases on a separate connection, keeps claim-produced outputs pending, and atomically publishes only the active fencing token's referenced terminal files. The one F1c kind driver hashes and independently replays raw LB, placement, membership, Pod, HTTP, file, job, claim, runtime-image, and rendered-manifest evidence.
- Why: A shared filesystem or process-local queue cannot preserve cross-gateway ownership after a gateway Pod disappears, and retrying inference without a fencing token can expose stale or duplicate terminal output. F1a and F1b already prove shared-runner capacity, dynamic replica membership, image/source provenance, and zero-failure rollout; rerunning them would add cost without testing F1c's distinct contract.
- Refs: m10 A29; G5 F1c; issue #177; Actions runs `30374404150` and `30387260062`; `kairyu/batch/postgres_store.py`; `bench/fleet_gateway_bench.py`; `scripts/kind_gateway_gate.sh`; `deploy/kind/f1c/`

### 2026-07-28 — [amendment] F1b sizes its whole-run safety cap from formal evidence
- What: Added A28. The formal whole-rollout timeout is now a 1,500-second stuck-run safety cap; the binding five-second withdrawal and 60-second per-replacement limits, retry-free zero-failure rules, exact 100-replica lineage, and all replay checks are unchanged.
- Why: Exact-head formal run `30385162649` completed 88 exact sequential replacements in 890.208 seconds with no recorded lifecycle-condition failure while recording 46,500/46,500 valid retry-free 2xx responses, then the former 900-second cap cancelled ordinal 11. The observed mean, p95, and maximum full-cycle durations were 10.116, 11.165, and 12.134 seconds. Extrapolating the maximum to 100 steps, adding 20% shared-runner jitter margin, and rounding up yields 1,500 seconds; issue #176 defines no rollout-latency SLO.
- Refs: m10 D5/A27/A28; G5 F1b; issue #176; PR #267; Actions run `30385162649`; `bench/fleet_rollout_bench.py`

### 2026-07-28 — [amendment] F1b attests pre-status Pods by immutable UID
- What: F1b now retains the frozen Pod spec image when Kubernetes has created a replacement UID but has not yet emitted its first ContainerStatus. Replay also accepts that single Pending, not-Ready pre-status form only when the exact same immutable UID is later observed with the expected runtime reference and pinned CRI digest; missing eventual attestation or any visible mismatch still fails.
- Why: Exact-head PR smoke run `30384219955` served 458/458 retry-free requests and passed every rollout, identity, readiness, placement, and provenance check except one normal pre-ContainerStatus snapshot. Replaying its immutable sidecars with the UID-completion rule passes every check while preserving fail-closed image provenance.
- Refs: issue #176; PR #267; Actions run `30384219955`; m10 D5/A27; `bench/fleet_rollout_bench.py`; `tests/verification/test_fleet_rollout_bench.py`

### 2026-07-28 — [amendment] F1b replays server IDs and eventual image attestation
- What: F1b now joins each offered request to the gateway placement by the non-empty server-assigned response ID while separately retaining the unique client schedule ID. Lifecycle provenance permits only a Pending, not-Ready observation with a temporarily empty image ID, rejects every visible mismatch immediately, and still requires every observed Pod UID to later match the pinned CRI digest.
- Why: PR smoke run `30383243344` completed 412/412 retry-free requests and all four replacements, but exposed two verifier false negatives: the gateway intentionally assigns its own request ID, and Kubernetes can publish a replacement Pod before CRI materializes `status.containerStatuses[].imageID`. Replaying that immutable artifact under the corrected rules yields no failed checks without weakening ID or image provenance.
- Refs: issue #176; PR #267; Actions run `30383243344`; m10 D1/D5/A27; `bench/fleet_rollout_bench.py`; `tests/verification/test_fleet_rollout_bench.py`

### 2026-07-28 — [amendment] F1b freezes a drain-first partitioned rollout
- What: Added A27 and a separate F1b gate. One `kubectl rollout restart` is held at partition N, then a checked-in driver drains each old UID, proves EndpointSlice and gateway withdrawal plus zero outstanding work, releases exactly one ordinal, and proves the new UID/revision/Ready/eligibility state before continuing. Retry-free traffic and hashed request, placement, membership, rollout, Pod, and EndpointSlice sidecars are independently replayed. Pull requests run a reduced smoke; only one clean exact-head 100-replica formal artifact can close F1b.
- Why: F1a's `OnDelete` batch churn proves shared capacity and dynamic discovery but cannot prove drain-before-termination ordering, a partition-controlled image restart, exact old/new UID lineage, or unattended zero-failure completion for 100 replicas.
- Refs: m10 D5/A27; G5 F1b; issue #176; `bench/fleet_rollout_bench.py`; `scripts/kind_rollout_gate.sh`; `.github/workflows/f1b-rollout.yml`; `deploy/kind/f1b/`

### 2026-07-28 — [progress] F1a merges the retained formal result
- What: PR #266 merged as `68f43f4` and closed #175 after focused validation and independent A26 replay. The retained Actions run `30374404150` remains the binding 200-replica formal artifact; no duplicate formal run was needed.
- Refs: issue #175; PR #266; commit `68f43f4`; Actions run `30374404150`; m10 D5/A26

### 2026-07-28 — [amendment] F1a restores the written measurement-wide p99
- What: Added A26. F1a now applies the strict 10 ms nearest-rank placement p99 once across all requests in the frozen ten-minute measurement. Every epoch and ten-second post-delete window remains mandatory, hashed, published, and independently replayed diagnostic evidence, but its local p99 is not a second SLO. This supersedes only A16's later all-window threshold; the 200 replicas, 50 requests/s, ten churn batches, retry prohibition, zero-error rules, 99% pacing, lifecycle/identity joins, five-second withdrawal, provenance, sidecars, and fail-closed replay contract are unchanged.
- Why: G5 F1a specifies one ten-minute placement p99. A 500-sample window at a true 1% exceedance rate has only a 61.60% chance to satisfy nearest-rank p99 <10 ms, and ten such windows all pass only 0.79% of the time. The repeated-window rule therefore made ordinary shared-runner scheduling jitter a much stricter unstated SLO. Actions run `30374404150` proves the written target with 33,000/33,000 2xx responses, 5.221 ms p99 over 30,000 measurement samples, 128 samples (0.427%) at or above 10 ms, and maximum 1.117-second withdrawal. Its four locally elevated diagnostic windows correlate with replacement container startup rather than membership transitions. Replaying the immutable sidecars under A26 passes every amended gate check; the legacy manifest differs only in the published check map and `passed` value that A26 intentionally changes.
- Refs: m10 D5/A16/A26; G5 F1a; issue #175; PR #266; Actions run `30374404150`; `bench/fleet_churn_bench.py`; `tests/verification/test_fleet_churn_bench.py`; `deploy/kind/f1a/README.md`

### 2026-07-28 — [amendment] F1a removes remaining shared-node lifecycle contention
- What: Added A25. Formal Pod deletion now uses the lifecycle-owned persistent Kubernetes proxy/client with bounded eight-way REST DELETE instead of measurement-time kubectl subprocesses, preserving background propagation and the declared Pod grace period. Kubelet evidence requests the CPU-and-memory-only Summary API at the unchanged cadence and schema. EndpointSlice parsing preserves exact historical selection while validating once and retaining one best candidate per replica. The formal gateway's matching 2 CPU / 256 MiB requests and limits freeze Guaranteed QoS. All traffic, latency, withdrawal, pacing, evidence, and replay thresholds remain unchanged.
- Why: Exact-head Actions run `30370270740` served 33,000/33,000 requests without errors and reached 7.554 ms overall placement p99, but nine of ten churn windows were 10.787–15.147 ms; the remaining window was 9.920 ms. Tail correlation placed 29/30 early samples inside Pod DELETE process execution and 58 later samples inside replacement container/kubelet startup, while exact membership transitions explained too few samples to justify a wider watcher execution-context refactor. Shared-node CPU reached 3.459/4 cores while the gateway remained below 300m CPU and 56 MiB.
- Refs: m10 D2/D5/A24/A25; G5 F1a; issue #175; PR #266; Actions run `30370270740`; `bench/fleet_churn_bench.py`; `kairyu/deploy/registry.py`; `deploy/kind/f1a/base/gateway.yaml`; `tests/verification/test_fleet_churn_bench.py`; `tests/unit/test_k8s_discovery.py`

### 2026-07-28 — [amendment] F1a removes observer process churn and exact-HRW repetition
- What: Added A24. All F1a Kubernetes sampling reads now use one lifecycle-owned `kubectl proxy` and persistent bounded-timeout HTTP client; Pod DELETE remains the only measurement-time kubectl subprocess. The formal gateway polls EndpointSlices every 500 ms instead of four times per second. ReplicaPool preserves every historical SHA-256 rendezvous winner while reusing the session-prefix hash state and mutation-boundary encoded-ID, ordinal, and eligible-ring caches; membership audit capture now builds one internally consistent state in one traversal. All frozen traffic, placement, withdrawal, evidence, and replay thresholds remain unchanged.
- Why: Exact-head Actions run `30365961550` returned 33,000/33,000 2xx with 7.998 ms overall placement p99, but every post-delete window reached 10.761–26.315 ms. Simultaneous EndpointSlice/resource fetches correlated with 36.025 ms p99 versus 4.188 ms outside either fetch while gateway CPU/memory had ample headroom. Exact-allocation microbenchmarking over 200 replicas and 10,000 requests improved 1.541 s → 0.808 s (1.91x). Even the historical 3.618 s maximum old-UID withdrawal leaves at least 1.13 s worst-phase margin at the new cadence under the unchanged five-second limit.
- Refs: m10 D1/D2/D5/A24; G5 F1a; issue #175; PR #266; Actions run `30365961550`; `bench/fleet_churn_bench.py`; `kairyu/orchestration/replica.py`; `kairyu/deploy/registry.py`; `deploy/kind/f1a/`

### 2026-07-28 — [amendment] F1a makes sparse lifecycle evidence fail-closed
- What: Clarified A23's reduced-control-plane evidence contract without changing the frozen gate. Ready, non-terminating EndpointSlice `targetRef` rows must form the exact expected Pod-name-to-UID bijection with no duplicate, alias, fallback, missing, unexpected, or contradictory same-timestamp identity. The initial 200 Pods, each epoch's twenty replacement Pods, the final 200 Pods, and the single anchored gateway Pod must each prove Ready state, lifecycle identity, and pinned container/runtime image identity. Raw fetch brackets make initial, EndpointSlice recovery, targeted Pod corroboration, final capture, and derived recovery times causally replayable. Periodic rows retain their schema/kind/payload, scheduled deadline, exact skipped-slot count, and traffic-start/end coverage; independent replay re-derives every deadline/skip transition and rejects catch-up, truncated, empty, duplicated, or backdated evidence.
- Why: Removing intrusive full-fleet polling is safe only if the smaller evidence set cannot admit identity aliases, hide missed sampling deadlines, treat EndpointSlice readiness as Pod/image proof, or claim recovery before the corroborating Pod fetch completed.
- Refs: m10 D5/A21/A23; G5 F1a; issue #175; PR #266; Actions run `30359098798`; `bench/fleet_churn_bench.py`; `tests/verification/test_fleet_churn_bench.py`

### 2026-07-28 — [amendment] F1a makes control-plane evidence deadline-paced
- What: Superseded A22's per-second full-fleet recovery LIST with EndpointSlice-first exact-200 recovery followed by one label-selected collection LIST for the twenty target Pods. Removed periodic full-200 Pod serialization in favor of initial, per-epoch target, and final captures; periodic samplers now record and skip overdue intervals instead of issuing catch-up bursts. Post-traffic placement copies retain bounded admission but flush and retry on a worker thread, and the formal probe interval is one second. All frozen traffic, churn, latency, withdrawal, identity, and replay thresholds are unchanged.
- Why: Exact-head public run `30359098798` preserved all ten 20-Pod replacements and 2.429–3.618 s withdrawals but exposed two independent defects. A 32,179-row gateway audit overflowed the 4,096-row nonblocking bulk-copy queue before manifest evaluation. More importantly, slow all-200 Pod and kubelet evidence fetches caused the absolute sampler to replay missed slots immediately: 570 periodic Pod LISTs consumed 645.67 s, the runner delivered only 30,808/33,000 2xx with 67 502s, 56 transport failures, and 2,069 unsent arrivals, and measurement placement p99 reached 234.374 ms. The clean cooldown returned to 1,500/1,500 2xx and 6.044 ms p99 while gateway CPU remained only 285m average/361m maximum, identifying intrusive control-plane catch-up rather than insufficient gateway CPU.
- Refs: m10 D5/A23; G5 F1a; issue #175; PR #266; Actions run `30359098798`; `bench/fleet_churn_bench.py`; `deploy/kind/f1a/`

### 2026-07-28 — [amendment] F1a removes measurement and recovery amplification
- What: Moved benchmark JSONL encoding and flushing to the bounded lifecycle-owned writer, replaced each recovery cycle's twenty per-name Pod GETs with one label LIST plus local filtering, and limited post-withdrawal recovery polling to one second while retaining the independent absolute 250 ms EndpointSlice observer. Removed duplicate Uvicorn and successful-httpx request logs while retaining the authoritative structured Kairyu access record, and raised the formal gateway CPU request from 100m to an evidence-backed 500m. The 50 requests/s load, ten 20-Pod batches, retry prohibition, 10 ms placement limit, 99% pacing requirement, and all raw replay checks are unchanged.
- Why: The final-head public formal run served 33,000/33,000 requests with zero 5xx, 429, transport failures, or unsent arrivals, but the shared four-vCPU runner reached only 97.658% one-interval pacing and 13.191 ms placement p99. The recovery command amplified twenty names into roughly twenty Kubernetes GETs per cycle; 96.1% of placement violations and 86.7% of pacing misses occurred during those recovery intervals. The same traffic event loop also encoded and flushed 107 MiB of raw Pod/EndpointSlice evidence. Gateway samples topped out at 233m while shared-node churn reached 2.876 cores, so 500m provides more than 2x measured headroom without the unsupported full-core reservation.
- Refs: m10 D5/A22; G5 F1a; issue #175; PR #266; Actions run `30355794934`; `bench/fleet_churn_bench.py`; `deploy/kind/f1a/`; `kairyu/entrypoints/cli.py`

### 2026-07-28 — [amendment] F1a pins OCI config identity across Docker stores
- What: Replaced the store-specific Docker-ID/containerd-target equality assumption with an explicit image chain. The gate now records Docker's image ID and canonical config digest, retains the raw containerd manifest and rehashes it to the target descriptor, reads that manifest's config, and reads the CRI status ID; all config identities must match, while Docker's image ID must name either the config or manifest target. Artifact replay enforces the same chain, alongside the existing source-revision, raw CRI hash, and per-pod runtime checks.
- Why: PR smoke on GitHub exposed a cross-store semantic difference: classic Docker reports the OCI config as `.Id`, whereas the local containerd-backed Docker 29 reports the manifest descriptor. Both loaded the correct image, but comparing `.Id` directly with containerd's manifest target falsely rejected the classic runner before Kubernetes apply.
- Refs: m10 D5/A21; G5 F1a; issue #175; PR #266; Actions run `30353624027`; `scripts/kind_churn_gate.sh`; `bench/fleet_churn_bench.py`

### 2026-07-28 — [progress] F1a closes the 200-replica churn gate
- What: Completed the clean-commit formal gate with one gateway, 200 mock replicas, ten disjoint 20-replica churn epochs over ten minutes, and 33,000/33,000 successful requests. There were zero 5xx, 429, transport failures, or unsent arrivals; overall placement p99 was 2.966 ms and all churn-window p99 values were 2.680–4.082 ms. Every raw sidecar hash and request/placement/membership/pod/EndpointSlice join replayed, all ten withdrawal observers were provenance-consistent, and the maximum old-UID withdrawal was 2.510 s. Two fresh-cluster smoke runs and the independent artifact replay also passed.
- Refs: issue #175; commit `9c83913`; m10 D5/A16–A20; `bench/results/f1a-replica-churn-kind-final3-2026-07-28/manifest.json`

### 2026-07-28 — [amendment] F1a observes withdrawal concurrently with deletion
- What: Added an endpoint-only observer that starts with each pod deletion and samples on absolute 250 ms deadlines until the old UID set is disjoint, before slower pod readiness recovery. Raw rows now carry a contiguous sequence plus scheduled, fetch-start, and observation timestamps. Artifact replay verifies those fields, the final disjoint claim, and a conservative last-old-fetch-start to first-disjoint-observation bracket against the unchanged one-second limit.
- Why: The corrected full run passed every traffic, placement, membership, lifecycle, and provenance condition—33,000/33,000 2xx, zero failures, 3.071 ms overall placement p99, and 2.705–4.083 ms across all ten churn windows—but epoch 4 lacked one raw causality bracket. `kubectl delete` took 2.159 s while the detailed recovery observer started only after it returned; a simultaneous 291 ms driver/host pause stretched the sole 1 Hz sampler gap to 1.249 s. Kubernetes and the gateway withdrew every old UID normally in 2.256/2.169 s with no reappearance, but the raw criterion correctly remained unproven. Starting the causal observer before awaiting delete removes phase luck without weakening the gate.
- Refs: m10 D5/A20; G5 F1a; issue #175; `bench/fleet_churn_bench.py`; `tests/verification/test_fleet_churn_bench.py`; `bench/results/f1a-replica-churn-kind-final2-2026-07-28/manifest.json`

### 2026-07-28 — [amendment] F1a removes replica-count validation and audit encoding from placement
- What: Added explicit immutable request-validation identities so a ReplicaPool validates one representative for each exact backend-type/capability pair while retaining per-replica validation for unknown or malformed keys. Kept membership snapshotting and ordered queue admission synchronous, but deferred JSON serialization to the existing bounded audit writer thread with a 64 KiB encoded-batch ceiling. At 200 equivalent OpenAI backends validation measured 1.785 ms → 0.143 ms median (12.48x); membership admission measured 0.012 ms median. In a 22-row burst of real-sized 37 KiB membership records, median event-loop heartbeat delay fell from 2.407 ms to 0.902 ms.
- Why: The first complete formal run served 33,000/33,000 requests with zero 5xx, 429, transport failures, or unsent arrivals and 6.924 ms overall placement p99, but epoch windows 2/3/5 reached 10.038/10.299/11.958 ms. Capability validation was redundantly repeated for all 180–200 eligible replicas before every placement; vLLM's comparable rendezvous router also scans healthy workers, so changing HRW would add remapping semantics without addressing this Kairyu-specific work. Validation grouping removes the dominant deterministic cost, while lifecycle-owned deferred encoding adds tail headroom without weakening the fixed gate or reordering evidence.
- Refs: m10 D5/A19; G5 F1a; issue #175; `kairyu/orchestration/replica.py`; `kairyu/engine/openai_backend.py`; `kairyu/audit_io.py`; `kairyu/orchestration/router.py`; vLLM Router `d60711d`

### 2026-07-28 — [amendment] F1a supersedes A17 with origin-local HTTP pools
- What: Replaced the fleet-wide httpx data/probe pool with cheap lazy origin-local data clients built from one eagerly shared TLS context. Each backend owns and closes its client; one idle socket per origin bounds retained FDs, active connections remain admission-bounded, and readiness has an independent 16-connection pool.
- Why: The second formal attempt failed before churn: latency rose from 12.6 ms to 929 ms over the first 1,000 requests, then 502s ejected healthy replicas and starved the single event loop. httpcore 1.0.9 scans a flat cross-origin connection list and recomputes idle counts quadratically; at 456 connections its measured assignment cost was 5.93 ms, enough to consume about 593 ms CPU/s at 50 requests/s before other gateway work. Current vLLM aiohttp/reqwest clients instead index connections by origin. Separate transports retain httpx compatibility without either this fleet-wide scan or the first attempt's repeated CA-bundle loading.
- Refs: m10 D5/A17/A18; G5 F1a; issue #175; `kairyu/deploy/builder.py`; `kairyu/engine/openai_backend.py`; httpx 0.28.1/httpcore 1.0.9; vLLM Production Stack `a0f4980`; vLLM Router `d60711d`

### 2026-07-28 — [amendment] F1a shares transport at pool scope and measures through NodePort
- What: Replaced per-replica lazy HTTP clients with one lifecycle-owned data/probe connection pool per dynamic ReplicaPool, preserving per-operation timeouts and cancellation-safe exactly-once close. Replaced `kubectl port-forward` in the kind gate with a provenance-pinned localhost-to-NodePort mapping. A 200-replica diagnostic completed 1,000/1,000 cold requests and another 1,000/1,000 after replacing 20 Pods, with zero 5xx, transport errors, or gateway restarts.
- Why: The first formal attempt exposed two independent measurement blockers before epoch zero: synchronously constructing 181 per-replica clients stalled the event loop for roughly 4.7 seconds and caused 124 real upstream 502s, while the 1,024-FD `kubectl port-forward` process later lost its tunnel after 258 connections. Pool-scoped reuse matches current vLLM router practice and removes O(replicas) TLS/client setup; direct NodePort keeps the transport harness outside the measured gateway path.
- Refs: m10 D5/A17; G5 F1a; issue #175; `kairyu/deploy/builder.py`; `kairyu/engine/openai_backend.py`; `scripts/kind_churn_gate.sh`; vLLM Production Stack PR #767

### 2026-07-28 — [amendment] F1a makes Kubernetes churn evidence production-owned and replayable
- What: Added in-cluster EndpointSlice discovery with UID-stable generations, readiness and termination filtering, dynamic gateway lifecycle wiring, bounded-state metrics, ingress-to-placement timing, and ordered membership transition evidence. Added a fixed-seed kind protocol for one gateway and 200 mock replicas whose independent verifier rehashes and replays raw requests, placements, EndpointSlices, pod identities, resources, and all ten churn epochs.
- Why: The previous fake-source reconciliation tests and small kind smoke could not prove the G5 F1a envelope or causally join zero-error traffic and placement latency to the exact replicas removed and recreated.
- Refs: m10 D2/D5/A16; G5 F1a; issue #175; `kairyu/deploy/registry.py`; `bench/fleet_churn_bench.py`; `deploy/kind/f1a/`

### 2026-07-28 — [progress] F5c closes with matched queue-and-hope evidence
- What: Completed the 50-check clean-commit F5c gate. Underload stays at 200/400 SLO-successful requests; smooth exact-2x improves queue-and-hope 3 → 397 and bursty exact-2x improves 2 → 198. The independent forced-admit oracle records zero false positives and 4/800 and 6/800 false negatives; every admitted/deferred request, lease, and scheduler queue drains.
- Refs: issue #192; G5 F5c; commit `dac6980`; `bench/results/f5c-slo-admission-cpu-2026-07-28.json`

### 2026-07-28 — [amendment] F5c separates interactive prediction from deferred backlog
- What: Replaced split SLO decision/start bookkeeping with an atomic, exactly-once admission lease; froze concurrency at admission for TTFT EMA feedback; separated interactive in-flight prediction from a total-active defer bound; and added matched production-Scheduler traces plus an independent forced-admit FP/FN oracle.
- Why: The first matched smooth-2x diagnostic exposed 788 false positives in 800 requests: intentionally deferred lower-priority work remained in the equal-priority predictor and caused self-reinforcing rejection even though it could not advance ahead of new interactive work. Class-separated state removes that causal error while retaining a finite batch backlog.
- Refs: m11 D6; issue #192; `kairyu/entrypoints/server/slo.py`; `bench/slo_admission_bench.py`; `tests/verification/test_slo_admission_bench.py`

### 2026-07-28 — [progress] F5b closes on real Qwen3-32B TP8
- What: Completed the matched-comparator GPU gate on 8x RTX PRO 6000
  Blackwell. All 47 checks pass. At 7.2697 requests/s calibrated capacity,
  good-only TTFT p99 is 0.1741 s, compliant-neighbor controls are
  0.2783/0.2792 s, and 10x treatment is 0.2801 s. The treatment admits 66 and
  rejects 639 noisy requests before shared dispatch, while all 256 good
  requests complete. Its accepted noisy work is 1.03125x the control average.
  Gateway/replica/scheduler/usage counts reconcile; TP8 is reported at runtime;
  in-flight and reserved-token gauges drain to zero with no bound violations.
  The deterministic CPU gate also passes all 11 checks at the same clean
  source commit.
- Refs: issue #191; G5 F5b; commit `0d8d091`;
  `bench/results/f5b-noisy-neighbor-cpu-2026-07-28.json`;
  `bench/results/f5b-noisy-neighbor-qwen3-32b-tp8-2026-07-28.json`.

### 2026-07-28 — [amendment] F5b uses a causal compliant-neighbor comparator
- What: Strengthened the real Qwen3-32B TP8 gate from good-only controls to
  bracketed controls that run the same good load plus a noisy tenant offering
  0.9x quota. The 10x treatment must match control accepted noisy work within
  10%, keep good p99 within the unchanged 50 ms non-inferiority margin, and
  retain the fixed 2 s SLO. Every arm begins after a full request-bucket
  wash-in; the oversize pre-dispatch probe now runs before that wash-in.
  Good-only latency remains a secondary reference.
- Why: The initial clean TP8 run passed 35/36 checks, including all admission,
  dispatch, usage, topology, and drain checks, but compared good-only 0.5x
  capacity against treatment containing another 0.92 requests/s of legitimate
  GPU work. Its 0.177 -> 0.564 s p99 therefore mixed allowed 1x work with the
  effect of rejecting excess 9x traffic. A matched accepted-work comparator
  isolates that causal excess-load effect without relaxing the latency margin.
- Refs: issue #191; G5 F5b; `bench/noisy_neighbor_gpu_bench.py`;
  `tests/verification/test_noisy_neighbor_gpu_bench.py`.

### 2026-07-28 — [amendment] F5b closes every compute-ingress lease boundary
- What: Extended pre-dispatch reservation to Embeddings using a conservative
  non-refundable input-work ceiling. AUTO now reserves the maximum configured
  route rather than relying on a preview whose stateful route choice could
  diverge at execution; its final candidate fan-out now includes tools and
  response-schema bytes, while private stages normalize both `n` and `best_of`
  to one candidate. Every settlement debits and reports observed work
  above the reservation even when usage is non-refundable, while surplus is
  still returned only from exact single-candidate usage. Empty Completion
  prompt arrays are rejected before admission, and a failing array element
  cancels and joins every sibling before the HTTP request can release its
  tenant lease.
- Why: Any compute ingress that bypasses reservation can become a
  noisy-neighbor path. Releasing a lease while sibling generation is still
  running likewise understates in-flight work and defeats the isolation bound.
- Refs: issue #191; m11 D3/A7; `kairyu/entrypoints/server/app.py`;
  `kairyu/entrypoints/server/extra_routes.py`;
  `kairyu/orchestration/orchestrator.py`.

### 2026-07-28 — [amendment] F5b reserves worst-case work before shared dispatch
- What: Corrected the initial #191 design from post-execution token debt plus
  request-count leases to two-stage admission. Every execution surface now
  atomically reserves a finite worst-case compute ceiling after validation and
  before shared engine placement. Candidate prefill fan-out, prompt arrays,
  tools/response metadata, and AUTO's maximum internal DAG are included.
  Exact single-candidate terminal usage can refund surplus; missing or
  approximate usage, failure/disconnect, and multi-candidate work consume the
  full reservation. Outstanding reservation and in-flight gauges must drain
  to zero. The refill comparison also tolerates floating-point ulps, correcting
  the deterministic 6 RPM result from 110 to the exact 120 admissions.
- Why: Request counts alone allow one long prompt, large output allowance,
  hidden `best_of`, or AUTO fan-out to occupy scheduler/KV capacity before
  post-completion charging. Gateway reservation keeps this portable across
  Kairyu, vLLM, and SGLang without adding tenant logic to the per-token
  scheduler hot path.
- Refs: corrects the earlier 2026-07-28 F5b amendment; issue #191; m11 D3/A7;
  `kairyu/engine/backend.py`; `kairyu/entrypoints/server/tenancy.py`;
  `kairyu/orchestration/orchestrator.py`; `bench/noisy_neighbor_bench.py`.

### 2026-07-28 — [amendment] F5b bounds 10x noisy-neighbor admission
- What: Added independent request/token burst capacities and optional
  per-tenant in-flight leases acquired before global concurrency and held
  through the final unary/SSE byte. Batch lines now acquire the same lease
  before shared replica dispatch. Bounded admission/in-flight metrics expose
  tenant, source, decision, and reason; token debt is charged even if optional
  ledger admission fails. The deterministic production-component gate drives
  a noisy tenant at exactly 10x its 6 RPM quota on the same native scheduler
  as a 15 RPM good tenant. It rejects 1,090/1,200 noisy requests before
  enqueue, completes all 300 good requests, bounds good TTFT p99 to 2 seconds
  versus control 1, and holds queue high-watermark to 2. The unprotected
  matched trace reaches good p99 298 seconds and queue high-watermark 301.
- Why: A one-minute-capacity request bucket plus post-completion token charging
  allowed cold concurrent bursts to enter ReplicaPool/GPU capacity before any
  429. Gateway leases provide constant-time isolation without adding
  tenant-WFQ work to every scheduler token; `max_in_flight` bounds the
  remaining post-settlement token exposure.
- Refs: issue #191; m11 D3; G5 F5b;
  `kairyu/entrypoints/server/tenancy.py`; `kairyu/batch/worker.py`;
  `bench/noisy_neighbor_bench.py`.

### 2026-07-28 — [amendment] F5a holds interactive TTFT under exact 2x overload
- What: Added signed-int64 priority to Chat Completions, legacy Completions,
  Responses, offline APIs, replica transports, vLLM, native, and process-split
  engines. Authenticated gateways assign interactive and batch priorities from
  trusted tenant/source context and carry an explicit bounded class through
  Kairyu HTTP transport. Native schedulers use exact signed-int64 indexed
  priority admission with nanosecond-resolution aging and may
  recompute-preempt only strictly lower-priority, output-free incomplete
  prefills; decode remains protected. Local vLLM construction forces its
  priority policy rather than silently retaining FCFS. Replica metrics publish
  class queue depth/high-watermark and enqueue/admit/preempt/complete counters.
  The deterministic
  CPU gate records interactive TTFT p99 of 1 tick versus 400 under FCFS at
  exact 2x load. A production-shaped Qwen3-32B TP8 gateway gate calibrated
  7.6304 requests/s and, under planned 0.5x interactive plus 1.5x batch load,
  measured interactive TTFT p99 of 1.3025 s with 100% attainment of the fixed
  2 s SLO. During the mixed window, batch recorded 54 HTTP dispatches, 53
  scheduler admissions, 46 scheduler completions, and one queued request; it
  drained 192/192 with zero failures afterward. The full CPU suite reports
  2,213 passed. The production
  artifact is generated only after an untimed
  warmup from a clean implementation commit and pins benchmark/config hashes,
  image digest, model revision, `/backends` topology, and GPU inventory.
  The gate requires a positive batch queue at the measurement boundary and a
  batch high-watermark at least as large as that depth; merely publishing
  zero-valued gauge series cannot satisfy the queue-evidence assertion.
- Why: FIFO admission lets queued batch prefills occupy every active slot and
  makes a feasible interactive SLO fail during overload. Gateway-owned
  classification prevents clients from self-promoting, while work-conserving
  priority plus bounded aging lets batch consume residual capacity without a
  static reservation. Protecting decode and restricting victims bounds
  recomputation and is stronger for TTFT than vLLM current main's
  allocation-failure-only priority victim selection.
- Refs: issue #190; m11 D6/A11; G5 F5a;
  `kairyu/engine/core/scheduler.py`;
  `bench/priority_overload_{bench,gpu_bench}.py`;
  `bench/results/f5a-priority-overload-{cpu,qwen3-32b-tp8}-2026-07-28.json`

### 2026-07-28 — [amendment] Responses API closes typed streaming and Codex tool loops
- What: Replaced the unary-only Responses subset with canonical gapless text/function SSE, completed/incomplete/failed terminals, flat and namespace function tools, linked `function_call_output` history, structured text formats, explicit unsupported-field rejection, and bounded tenant-scoped successful-response continuation. The adapter now reuses Chat Completions validation/execution, model templates, tool-choice enforcement, upstream capability preflight, and usage ownership. Official OpenAI SDK unary/sync-stream/async-stream and failure/state/tool tests are green; the full CPU suite reports 2,186 passed. An unmodified Codex CLI 0.145.0 completed a text turn and a real `pwd` namespace command/result loop against Qwen3-32B TP8 on 8x RTX PRO 6000, with every latest-code Responses POST returning 200.
- Why: A second Responses-specific generation stack would drift from the already hardened chat contract. Normalizing wire items at L3 preserves one truthful execution boundary, while direct pull-through for irrevocable text and buffering only retractable tool decisions matches the measured responsibility split used by orchestration streaming. Tenant-scoped protocol-item storage supports SDK continuation without making stateless Codex turns depend on gateway-local history.
- Refs: issue #201; m11 D4/A16; `kairyu/entrypoints/server/responses_service.py`; `kairyu/entrypoints/chat_template.py`; `tests/server/test_responses_api.py`; `bench/results/responses-codex-qwen3-32b-tp8-2026-07-28.json`

### 2026-07-28 — [amendment] OpenAI-compatible forwarding becomes capability-aware
- What: Added immutable `generic`/`openai`/`vllm`/`anthropic`/`gemini`/`kairyu` request profiles, canonical max-token wire names, configurable sampling capabilities and vendor-extension allowlists, load-time configuration validation, and direct/ReplicaPool/AUTO pre-dispatch 400s for unsupported or unsafe intent. The Kairyu HTTP boundary now types and preserves `top_k`, `min_p`, `repetition_penalty`, `stop_token_ids`, `min_tokens`, and `ignore_eos`, while explicitly rejecting unimplemented `best_of`, prompt-logprob, skip-special-token, vendor, and strict-tool intent; dynamic and documented Kairyu replica configurations select that profile explicitly. Exact request-body, negative mismatch, configuration, HTTP-boundary, and live Qwen3-32B TP8 gateway-to-replica checks are green; the full CPU suite reports 2,161 passed.
- Why: OpenAI-compatible endpoints do not share one truthful field contract—Anthropic explicitly ignores several accepted fields, while vLLM and Kairyu execute non-standard sampling controls. An explicit constructor-resolved policy prevents silent intent loss and URL heuristics without adding provider-specific registries or request-hot-path schema work.
- Refs: issue #209; `kairyu/engine/openai_capabilities.py`; `kairyu/engine/openai_backend.py`; `kairyu/entrypoints/server/protocol.py`; `kairyu/entrypoints/server/chat_service.py`; `docs/deployment.md`

### 2026-07-28 — [amendment] AUTO preserves OpenAI intent at a bounded final-output boundary
- What: Issue #208 replaces prompt-only AUTO dispatch with an immutable
  per-request intent for sampling, deterministic `n`, choice-scoped logprobs,
  tools/tool choice, and response format. Direct, Conductor, and MoA retain
  complete final `CompletionOutput` choices in unary and pull-through streams;
  private stages use `n=1`, no logprob/tool/grammar expansion, and the explicit
  1024-token orchestration policy while the final stage receives the exact
  public output limit and intent. Native OpenAI tool calls are normalized at
  the existing L3 parser boundary, prompt-only engines receive one tool
  instruction, and AUTO chat templates are valid deployment targets.
  Qwen3-32B TP8 passed schema-valid `n=2` with logprobs, a required named tool,
  and a post-gate plain request. The tiered rerun measured 1.0123x / 0.7666x
  p50/p99 standard/direct TTFT and max 3/8 versus standard 1/8, with max using
  32 versus 38 calls and 1,499.274 versus 2,567.389 allocated GPU-seconds.
- Why: The old L3/L2 seam validated these OpenAI fields and then either rejected
  or discarded them, while fixed MoA sampling made the advertised model
  contract route-dependent. Applying a public 8192-token answer allowance to
  every private stage was also objectively worse: one stage took 76.44 s and a
  MoA proposal exceeded the upstream 60 s timeout, returning 502. The selected
  Kairyu-native boundary exposes the private cap through `/routing`, preserves
  scalar intent without shared-orchestrator mutation, and leaves the final
  public limit exact.
- Refs: m11 D1/D2, m8 D1/D2, G6 P-B4, issue #208;
  `kairyu/orchestration/request.py`,
  `kairyu/{orchestration,entrypoints/server,engine}/`,
  `bench/auto_params_bench.py`,
  `bench/results/auto-params-qwen3-32b-tp8-2026-07-28.json`,
  `bench/results/tiered-auto-qwen3-32b-tp8-2026-07-28.json`

### 2026-07-28 — [amendment] xgrammar receives tokenizer-native vocabulary metadata on every TP rank
- What: Structured sampling now carries a serializable `GrammarVocabulary`
  containing RAW/BYTE_FALLBACK/BYTE_LEVEL type, prefix-space behavior, encoded
  token strings, and the model lm-head vocabulary width. Local, P-D, and every
  spawned TP sampler receive the same metadata. The Qwen3-32B TP8 native engine
  completed `n=1` and `n=2` JSON-schema requests with logprobs and remained
  ready afterward.
- Why: Qwen stores a space token as byte-level `Ġ`, but Kairyu constructed
  xgrammar's `TokenizerInfo` as RAW and padded the tokenizer vocabulary with
  fabricated empty strings. After the valid prefix `{"answer":`, the RAW FSM
  could expose zero legal tokens; all masked logits became `-inf`, argmax chose
  token 0, and the TP runner became fatal when the matcher rejected it. vLLM's
  independently reviewed xgrammar path likewise passes tokenizer type and
  model vocabulary width; Kairyu now preserves that correctness contract
  through its own lighter tokenizer/sampler boundary.
- Refs: m8 D1/D2, issue #208;
  `kairyu/engine/{tokenizer.py,core/structured.py,core/worker.py}`,
  `tests/unit/{test_tokenizer.py,test_structured_output.py}`

### 2026-07-27 — [evidence] P-B4 proves distinct AUTO tiers on Qwen3-32B TP8
- What: The declarative orchestrator spec now wires a bounded `moa_samples`
  value into `Orchestrator`, making `kairyu-auto-max` a real three-proposal MoA
  tier while standard AUTO remains Conductor. One production DeploymentSpec
  lists Qwen3-32B, standard, and max together. A reproducible benchmark records
  model discovery, alternating TTFT, fixed LiveCodeBench sandbox scores,
  structured-trace call counts, internal tokens, response hashes, latency, and
  allocated GPU-seconds. Twelve latency pairs measured 1.0207x p50 / 0.9674x
  p99. The fixed eight-item multi-agent slice scored max 2/8 versus standard
  0/8; max used 32 calls and 45,759 tokens versus 44 calls and 60,369 tokens.
- Why: A larger budget in YAML did not select the already-implemented MoA path,
  so the two advertised tiers were not operationally distinct. The new gate
  compares the same checkpoint, hardware, dataset items, route class, and
  alternating request order, isolating orchestration depth. The result is
  deliberately scoped to the fixed subset and records that request sampling
  propagation remains issue #208.
- Refs: m11 D2, G6 P-B4, issue #198;
  `bench/tiered_auto_bench.py`,
  `bench/results/tiered-auto-qwen3-32b-tp8-2026-07-27.json`,
  `examples/qwen3-32b-multi-gpu/{auto-gateway,auto-orchestrator,auto-max-orchestrator}.yaml`

### 2026-07-27 — [contract] P-B2 closes truthful orchestration usage and trace
- What: AUTO unary and usage-enabled streaming responses now expose
  `orchestration_input_tokens` / `orchestration_output_tokens` reconciled to
  cumulative backend-reported internal calls. Direct, Conductor, and MoA cover
  retries, verifier calls, fallback resolution, proposals, and synthesis.
  Streaming trace opt-in returns the same versioned route/DAG/verifier/timing
  envelope as unary on one terminal metadata chunk. A synchronous cumulative
  accounting observer preserves completed pre-final and partial-final usage on
  disconnect without a queue; unary and streaming partial failures return
  known usage plus sanitized typed failure events.
- Why: The prior result counters were internally summed but indistinguishable
  from standard usage on the public wire, streaming exposed only a legacy
  comment, and exceptions discarded partial MoA/final-stream accounting. The
  unified terminal contract makes orchestration spend auditable without
  exposing prompts, intermediate generations, exception messages, credentials,
  or changing ordinary engine response shapes.
- Refs: m11 D1, G6 P-B2, issue #196;
  `kairyu/entrypoints/server/{protocol,app}.py`,
  `kairyu/orchestration/{conductor,moa,orchestrator}.py`,
  `tests/server/test_orchestration_usage_trace.py`

### 2026-07-27 — [design] P-B1 final-stage pull-through streaming closes issue #195
- What: Conductor and MoA now own their final backend iterators and expose typed
  deltas/results to Orchestrator; Orchestrator emits pre-final SSE-comment
  keep-alives and assembles route/trace/accounting, while the HTTP layer only
  serializes OpenAI chunks. Direct, Conductor, and MoA finals stream live;
  cancellation/failure closes the backend iterator, releases budget
  reservations, and finalizes usage once. A final role with its own
  post-generation verifier is rejected explicitly; supported DAGs verify the
  draft before an unverified final boundary. Production Qwen3-32B TP8 configs,
  a paired TTFT gate, raw samples, and cancellation/failure/usage/trace tests
  are committed.
- Why: Matching another framework's component names is not a performance
  requirement. A controlled transport A/B measured pull-through at 161.14
  ns/event versus 7,045.2 ns/event for a task plus bounded queue (43.721x), so
  Kairyu keeps the execution-owned iterator. The 24-pair real-model gate
  measured AUTO/direct TTFT at 1.0096x p50 and 1.0122x p99, below P-B1's 1.5x
  ceiling.
- Refs: m11 D1/A5, G6 P-B1, issue #195;
  `kairyu/orchestration/{conductor,moa,orchestrator}.py`,
  `bench/orchestration_stream_bench.py`,
  `bench/results/orchestration-stream-qwen3-32b-tp8-2026-07-27.json`

### 2026-07-27 — [amendment] P-C5 emits versioned cached-token invoices
- What: usage rows and Prometheus totals now expose explicit uncached input in
  addition to cached input. DeploymentSpec validates a versioned blended
  price sheet with cached-input and tenant discount fractions. The
  caller-scoped `/admin/usage.csv` endpoint exports deterministic period
  invoices with separated quantities, Decimal rates/charges, source SHA-256,
  and invoice ID.
- Why: prompt totals alone cannot apply cache discounts, and computing charges
  during inference would couple availability and latency to billing. Immutable
  usage plus an explicit price-sheet version makes recalculation deterministic
  while preserving the faster gateway-local request path.
- Verification: pricing/reconciliation/server tests cover sync and streaming
  chat, Responses, embeddings, batch, old rows, three gateways, periods,
  tenant scope, discounts, rounding, contradictory fields, corrupt JSON, and
  truncated tails; the full CPU suite passes 2,054 tests. The uncached-aware
  five-run A/B still selects local async: producer p99 19.696 vs 28.026 us and
  throughput 95,029 vs 72,374 rows/s.
- Refs: m11 D3/A15; G6 P-C5; Issue #204; `kairyu/pricing.py`;
  `tests/server/test_pricing_invoice.py`.

### 2026-07-27 — [amendment] F5e fleet usage stays gateway-local and reconciles offline
- What: cached prompt-token counts now flow through chat, completions,
  Responses, batch, direct orchestration, Conductor, and MoA into tenant
  ledgers, `/admin/usage`, and dedicated Prometheus usage counters. A strict
  offline reconciler aggregates independently owned gateway ledgers against
  request audit logs. The committed fixed replay covers nine cross-endpoint
  outcomes across three gateways, including disconnect, upstream failure, and
  batch rollback, and records the raw logs plus report.
- Why: a shared ledger in the request path adds a fleet-wide serialization and
  availability dependency. Among candidates preserving raw rows and exact
  totals, five 30,000-row runs measured gateway-local async at 19.062 us
  producer p99 and 99,474 rows/s versus 27.970 us and 72,272 rows/s for a
  favorable same-host/no-network synchronous shared store. The local boundary
  is both faster and failure-isolated.
- Verification: maximum request-log/ledger error is 0.0% against the strict
  `<0.1%` F5e gate; focused usage/server/orchestration tests pass.
- Refs: m11 D3/A14; G5 F5e; Issue #194;
  `kairyu/usage_reconciliation.py`; `bench/fleet_usage_replay.py`;
  `bench/results/{fleet-usage-reconciliation,usage-architecture}-2026-07-27.json`.

### 2026-07-27 — [amendment] P-B5 usage counters reconcile from the ledger
- What: successful metering events now increment tenant-bounded Prometheus
  counters for executions, prompt tokens, and completion tokens at the same
  seam as ledger admission. Batch workers use the same sink. A
  single-gateway restart restores counter totals from the app-owned ledger
  before serving. The JSONL writer terminates a preserved truncated tail before
  appending the first post-restart row.
- Why: generic HTTP request counters include rejections and unknown models, so
  they cannot reconcile to successful billable executions. Process-local
  counters also reset on restart unless the append-only source of truth seeds
  them. Separating a crash tail prevents the first recovered request from being
  concatenated into—and lost with—the malformed row.
- Verification: the nine-mode usage matrix covers sync/stream chat,
  completions, AUTO, Responses, embeddings, and batch with exact ledger/counter
  equality. The supported DeploymentSpec path proves independent two-key 429s,
  and a restart gate proves counter restoration, malformed-tail recovery, and
  shutdown drain with 0% reconciliation error. Focused server/audit regression:
  127 passed; the complete CPU suite passed 2,030 tests (12 skipped,
  118 deselected).
- Refs: issue #199; m11 D3/A13; G6 P-B5;
  `kairyu/entrypoints/server/metrics.py`;
  `kairyu/entrypoints/server/metering.py`;
  `tests/server/test_m11_product.py`;
  `tests/server/test_serve_builder.py`.

### 2026-07-27 — [amendment] audit JSONL filesystem work leaves request loops
- What: usage-ledger and router-log producers now admit complete JSON rows to a
  bounded queue without waiting for filesystem calls. One lifecycle-owned
  thread batches up to 128 rows per append+flush. Ordered `flush()` barriers
  preserve read-after-write, `/admin/usage` runs its barrier and scan through a
  worker thread, and `close()` drains all accepted rows. Queue saturation fails
  immediately; filesystem failures are sticky and surface through admission,
  flush, or close rather than claiming durability. Flush reaches the OS for
  reader visibility but deliberately does not claim `fsync` power-loss safety.
- Why: persistent handles removed open/close overhead but still performed
  write+flush synchronously in request handlers and stream finalizers. Slow or
  failing storage could therefore stall every coroutine sharing the event loop.
  Bounded admission keeps latency independent of disk while preserving explicit
  backpressure and accounting integrity.
- Verification: 305 focused server/router tests pass. Six fault-oriented gates
  cover an intentionally blocked filesystem behind a real SSE response,
  immediate queue-full failure, sticky disk errors, concurrent JSONL integrity,
  ordered visibility, batching, restart, and shutdown drain. With 10,000
  records and an injected 0.1 ms delay per write/flush, request-path admission
  measured 3.1821 s → 0.02077 s (153.21×), full flush/drain 3.1821 s →
  0.05842 s (54.47×), and writes 10,000 → 79. The final complete CPU suite
  passed 2,027 tests (12 skipped, 118 deselected).
- Refs: issue #213; m11 D3/A7; m4 §2.1;
  `kairyu/audit_io.py`; `kairyu/entrypoints/server/tenancy.py`;
  `kairyu/orchestration/router.py`; `bench/audit_io_bench.py`.

### 2026-07-27 — [amendment] sampler penalties use measured incremental effective counts
- What: penalty-active sampler states lazily allocate prompt membership,
  repetition membership, output-count, and output-membership rows for the
  active device/vocabulary. A private committed shadow plus scheduler-owned
  output epoch removes normal retained-prefix copies/comparisons. Exact pending
  commit advances the boundary without recounting; rollback/correction takes
  the exceptional rebuild path. CPU sparse active IDs use growing buffers and
  position maps. P-D migration releases the prior device row, and normal
  sampler release reclaims the state.
- Why: rebuilding prompt/output sets, tensors, zero rows, and `bincount` from
  the full history at every token made penalty sampling scale with generation
  length. Current vLLM main
  (`5f89a03dcb52702a62644e15b93f766765d06b28`) still converts complete output
  lists to a padded tensor per penalty application and marks that path
  inefficient. Kairyu's persistent request-state/lifecycle boundary is native
  to its overlap/P-D ownership, so we also measured a fairly optimized
  committed-count/transient-pending alternative: effective counts win the
  complete normal step by 1.31× on CPU and 1.76× on CUDA.
- Verification: 2,020 CPU tests pass (12 skipped, 118 deselected), plus 14/14
  focused CUDA/full-model tests. All 52 non-trivial
  repetition/presence/frequency combinations preserve exact processed logits
  and seeded samples against the pre-change oracle through append and
  rollback. Same-length and epoch-signalled correction, pending commit,
  repeated-token counts, lifecycle release, same-device handoff, cross-GPU
  handoff, TP epoch propagation, and eager/overlap GPU sampling pass. The
  production-shaped 151,936-vocabulary/32,768-history benchmark commits one
  token and shifts pending each iteration. The alternative receives the same
  CPU direct-update optimization and prior pending CUDA scalar, and preserves
  the legacy total-count floating-point order. Every transition plus duplicate
  pending is bitwise-checked outside the timer. CPU measured 9,265.6 → 513.9
  µs (18.03×; alternative 672.8 µs), while CUDA measured 6,879.2 → 190.7 µs
  (36.07×; alternative 334.7 µs). A 32-token repetition-only step also wins
  501.1 → 138.7 µs (alternative 176.0 µs).
- Refs: issue #216; m8 D2; `kairyu/engine/core/sampler.py`;
  `tests/unit/test_sampler_incremental_state.py`;
  `tests/gpu/test_sampler_incremental_state_gpu.py`;
  `bench/sampler_penalty_state_bench.py`.

### 2026-07-27 — [amendment] RadixKV eviction selects indexed live leaves
- What: every radix node carries an eviction generation. A min-heap orders
  leaf candidates by LRU stamp and stable sequence, while an index identifies
  the one live generation. Insert, lock/unlock, touch, split, and delete
  transitions refresh eligibility. Stale entries are skipped and compacted at
  a bounded ratio; a failed BlockRemoved delivery requeues the selected victim.
- Why: collecting every refcount-zero leaf and taking `min` for each reclaimed
  page repeated an O(nodes) traversal per eviction. The indexed heap makes
  selection O(log leaves) without changing leaf-only, refcount, LRU, page, or
  event ownership semantics.
- Verification: 1,960 CPU tests pass (12 skipped, 115 deselected). Ten seeded
  1,000-operation traces match the legacy scanner after every allocation on
  page IDs, cached-token counts, free pages, hit rate, and complete event
  order. Repeated-hit compaction and BlockRemoved failure retry are pinned. A
  reproducible 100,000-leaf/100-eviction/3-repeat A/B run measured 1.7183 s →
  0.000609 s median selection time (2,823×).
- Refs: issue #214; m2 §2.3/§5; `kairyu/engine/core/radix_kv.py`;
  `tests/unit/test_radix_eviction_index.py`;
  `bench/radix_eviction_bench.py`.

### 2026-07-27 — [amendment] RadixKV event hashes extend a node-local chain
- What: event-enabled radix nodes now store the open canonical tuple SHA-256
  state, prefix token count, and local block digests. Child creation copies the
  parent state and feeds only new tokens; digest snapshots add exact tuple
  punctuation without mutating the continuation. Splits partition existing
  block hashes and derive only the upper continuation. Stored and removed
  events read these node-local values directly; event-disabled caches skip all
  hash state and SHA work.
- Why: reconstructing every root-to-node prefix and hashing every growing tuple
  slice made long-prefix event publication quadratic. A streaming continuation
  makes node creation linear and emission proportional only to the output list,
  while preserving the existing wire hashes so gateway migration is unnecessary.
- Verification: 1,948 CPU tests pass (12 skipped, 115 deselected). Randomized
  branches and splits across four page sizes and five seeds, singleton tuple
  punctuation, decode extensions, and randomized eviction all match the legacy
  SHA formula exactly. A reproducible 32,768-token/2,048-block/3-repeat A/B run
  measured 2.3222 s → 0.00871 s median hash generation (266.67×).
- Refs: issue #220; m10b D7/A13; `kairyu/engine/core/radix_kv.py`;
  `tests/unit/test_kv_event_hash_chain.py`;
  `bench/kv_event_hash_bench.py`.

### 2026-07-27 — [amendment] scheduler waiting admission uses indexed queues
- What: the Scheduler waiting list is replaced by an ID-indexed queue. FIFO
  mode uses `OrderedDict` for O(1) append, head removal, and cancellation.
  Priority mode uses a stable sequence-numbered heap, an ID index, and
  amortized tombstone compaction. Algebraically factoring the common
  `now/age` term makes the priority key immutable; recompute-preempted requests
  retain front-of-tie placement.
- Why: `pop(0)`, `remove(request_id)`, and a full stable sort on each admission
  made queue churn scale linearly or worse with waiting depth. Indexed
  ownership removes those host-path costs without allowing skip-ahead around a
  KV-blocked head or changing priority-aging and starvation contracts.
- Verification: 1,921 CPU tests pass (12 skipped, 115 deselected). Five seeded
  randomized traces for FIFO, priority-only, and aging modes match the legacy
  list model; stable ties, preemption-front, 100,000-ID stress, rejection, and
  head-of-line blocking are pinned. A reproducible 100,000-request/3-repeat A/B
  run measured 13.81× faster full FIFO drain, 94.88× faster distributed removal
  of 10,000 IDs, and 4.79× faster full priority drain.
- Refs: issue #219; m11 D6; `kairyu/engine/core/scheduler.py`;
  `tests/unit/test_scheduler_waiting_queue.py`;
  `bench/scheduler_queue_bench.py`.

### 2026-07-27 — [amendment] engine producer operations drain as safe batches
- What: `EngineLoop` now protects request reservation and producer queue
  mutation with one lock, groups consecutive adds and aborts, and coalesces
  lifecycle-duplicate aborts. The step thread drains a frozen snapshot;
  `Scheduler.add_requests_atomic` consumes compatible add batches, while
  generic adapters retain ordered per-request admission. Partial failures
  restore only untouched suffixes before concurrent batches. Purge filters the
  same structures under lock, and close seals the queue before reclamation.
- Why: per-request Python tuples and repeated abort operations amplified
  allocation/drain work under churn, while the old compound set-check/append
  sequence did not make concurrent duplicate reservation atomic. Explicit
  batches reduce queue objects without erasing add-then-abort terminal
  semantics or weakening failure ownership.
- Verification: 1,903 CPU tests pass (12 skipped, 115 deselected). Concurrent
  duplicate, 8-producer unique-add, 16-producer repeated-abort, 10,000-op,
  purge, close, atomic rollback, and partial-failure recovery tests pass. A
  reproducible 100,000-operation/5-repeat A/B run measured add throughput at
  95,712 → 100,463 op/s (1.05×, 100,000 → 1 queue containers) and duplicate
  abort throughput at 7.30M → 11.95M op/s (1.64×, 100,000 → 1 abort IDs).
- Refs: issue #218; m8 D1; `kairyu/engine/engine_loop.py`;
  `kairyu/engine/core/scheduler.py`; `tests/unit/test_engine_op_queue.py`;
  `bench/op_queue_bench.py`.

### 2026-07-27 — [amendment] stop matching scans only the newly stable tail
- What: each engine request now owns an incremental multi-pattern stop matcher.
  It remembers the prior stable-text length, searches only the new suffix plus
  `max_stop_length - 1` overlap, caches the first observed minimum match, and
  rejects retracted input. Final detokenizer bytes enter through the same path.
  Randomized tests compare every update against the historical full scan, and a
  long-output/many-stop test pins the search-work bound.
- Why: scanning every stop from the beginning of cumulative output on each
  update made a correctness-sensitive streaming path quadratic. A bounded
  overlap is sufficient because any newly completed pattern can start no
  earlier than its length minus one before the previously searched tail.
- Verification: 1,895 CPU tests pass (12 skipped, 115 deselected). For 32,768
  characters, 64 absent stops, and 1,024 updates, the search-window upper bound
  fell from 1,074,790,400 to 3,079,232 character-pattern checks (349.04×);
  median runtime fell from 0.1666 s to 0.0080 s (20.73×).
- Refs: issue #217; m8 D1; `kairyu/engine/engine_loop.py`;
  `tests/unit/test_stop_matcher.py`.

### 2026-07-27 — [amendment] streaming detokenization becomes native and linear
- What: `IncrementalDetokenizer` now consumes an optional per-request
  `TokenDecodeStream`. `HFTokenizer` uses the Rust `DecodeStream`, Toy processes
  only arriving IDs, and one final full decode retains byte-exact output.
  Tokenizers without the capability, old `tokenizers` versions, and subclasses
  whose `decode()` no longer matches the inherited stream remain on the exact
  full-prefix fallback. Tests cover ByteLevel, WordPiece, Metaspace, special
  tokens, multi-token commits, custom fallback, and 4,096-token linear work.
- Why: decoding the entire accumulated token history on every generated token
  made streaming detokenization O(n²). A tokenizer-owned state machine handles
  UTF-8 and decoder context without guessing a universal look-behind window,
  while explicit capability fallback preserves correctness for arbitrary
  programmatic tokenizers.
- Verification: 1,889 CPU tests pass (12 skipped, 115 deselected). The real
  Qwen3-32B tokenizer preserved prefix/final-byte parity across 19 sequences and
  5,792 tokens; 4,096-token decode measured 2.038 s full-prefix versus 0.0149 s
  native (137.25×).
- Refs: issue #211; m8 D1; `kairyu/engine/tokenizer.py`;
  `tests/unit/test_tokenizer.py`.

### 2026-07-27 — [amendment] production generation converges on one pipeline-depth loop
- What: `EngineLoop` now owns immutable snapshot submission, bounded
  schedule-ahead, synchronous/native-async runner handles, oldest-first commit,
  streaming, and late-result reclamation. `pipeline_depth=1` is the compatible
  default and depth 2+ activates overlap through the same Kairyu and ZMQ
  production builders. Speculative variable-length steps use commit barriers;
  stop/grammar/abort terminals emit immediately while scheduler/runner state is
  retained until surplus work drains. The old overlap and pipelined cores are
  explicitly compatibility-only, and the Qwen benchmark now enters through
  production `EngineLoop`.
- Why: independent run-to-completion cores could demonstrate overlap or PP but
  could not safely compose production streaming, stop holdback, grammar,
  speculation, preemption, chunked prefill, P-D adoption, and failure cleanup.
  Mandatory frozen `StepInput` boundaries remove live-state races, while one
  commit path prevents those semantics from diverging again.
- Verification: 1,881 CPU tests pass; the production CUDA image passes all 95
  runnable 1-GPU tests (16 topology/environment skips); TP2/NCCL passes 3/3
  runnable gates (2 conditional skips). Qwen3-32B TP8 on 8× RTX PRO 6000
  retained an identical depth-1/depth-2 output digest and measured 66.951 →
  67.814 token/s (+1.29%, wall −1.27%) over 8×32-token requests.
- Refs: issue #222; m2 §2.2/E3; m6 D5; `kairyu/engine/engine_loop.py`;
  `kairyu/engine/core/step_input.py`; `tests/unit/test_unified_engine_loop.py`;
  `bench/results/unified-loop-qwen3-32b-tp8-2026-07-27.json`

### 2026-07-27 — [amendment] m2 §2.2 future-token feedback is device-side
- What: grammar-free CUDA sampling now covers greedy, seeded stateless
  Gumbel-max with min-p/top-k/top-p, presence/frequency/repetition penalties,
  and raw logprobs. Uncommitted tokens remain device scalars, patch persistent
  decode slots D2D, and materialize as one batched asynchronous D2H only at the
  one-step-late EOS/stop/streaming boundary. The profiler GPU gate records zero
  `.item()`/`aten::_local_scalar_dense`/event waits in the isolated feedback
  interval; EOS, stop, min_tokens, penalties, logprobs, seeded replay, and
  overlap equivalence pass on CUDA. Stateful xgrammar keeps its reviewed CPU
  fallback. Qwen3-32B TP8 produced the same output digest as pre-fix main,
  removed the future-token event wait, and measured 68.161 → 68.686 token/s
  (+0.77%, wall −0.76%) over 8×32-token requests.
- Why: host sampling created a Python token ID, forced a per-step D2H/H2D
  round trip, and synchronized the pinned staging event before the next decode
  input could be reused. A stateless device RNG makes replay and TP rank
  agreement independent of host generator state, while late public
  materialization preserves stop semantics without putting the host back on
  the next-token dependency.
- Verification: 1,866 CPU tests pass; the production CUDA image passes all 93
  then-existing runnable 1-GPU tests (16 topology/environment skips), and the
  expanded #206 target passes 15/15; CUDA graph/tensor decode passes 9/9;
  TP2/NCCL passes 3/3 runnable gates (2 conditional skips).
- Refs: issue #206; m2 §2.2; m8 D2; `kairyu/engine/core/sampler.py`;
  `kairyu/engine/core/model_runner.py`; `kairyu/engine/core/overlap.py`;
  `kairyu/kernels/sampling_gpu.py`; the overlap/decode-slot GPU tests;
  `bench/results/future-token-qwen3-32b-tp8-2026-07-27.json`

### 2026-07-27 — [progress] G2 A2 closes on Llama-3.3-70B FP8 TP2/4/8
- What: the formal 64-prompt × 16-position gate passes all ten checks. HF
  self-agreement is 1005/1024 with a measured 0.5-nat tie gap; TP2, TP4, and
  TP8 achieve 1006, 1005, and 1006 agreements respectively, all with zero
  substantive disagreements and complete logprob evidence. Direct TP4/8
  comparisons against TP2 are each 1004/1024 with zero substantive
  differences. The self-contained result embeds the four raw envelopes,
  fifteen full weight digests, pinned Hub revision, CUDA/NCCL runtime,
  physical topology, and one clean measurement commit.
- Why: rank-local dynamic activation scales in row-parallel FP8 made the
  quantization contract depend on TP degree; TP4 missed the reference floor by
  one position. A per-token MAX reduction restores the unsharded W8A8 scale
  and brought TP4 to the binding reference floor without relaxing either
  amended criterion.
- Refs: issue #152; m16 D2/A10; G2 A2 and §7; GPU runbook §6;
  `bench/results/g2-a2-llama33-70b-fp8-rtxpro6000-2026-07-27.json`

### 2026-07-27 — [progress] G2 A2 70B FP8 TP gate is executable
- What: dense tensor parallel loading now accepts compressed-tensors FP8,
  shards projection weights and output-channel scales with their column
  projections, replicates row-parallel output scales, and value-preservingly
  widens serialized BF16 scales to the fused scaled-MM FP32 ABI. Dynamic FP8
  row-parallel projections MAX-reduce one amax per token so all ranks use the
  unsharded activation's scale instead of a TP-degree-dependent local scale.
  The shared HF
  reference can use Accelerate multi-GPU placement and batching. Candidate
  outputs retain all per-position tokens/logprobs, complete checkpoint and
  topology provenance, and the new `gate_a2.py` independently recomputes the
  amended HF-relative and TP4/8-vs-TP2 verdicts.
- Why: A2 could not previously load its 70B FP8 anchor at any TP degree, nor
  create a reference that exceeded one GPU's memory, and reported summary
  verdicts alone could not fail closed on partial or fabricated evidence.
- Refs: issue #152; m16 D2/A10; G2 A2 and §7; GPU runbook §6;
  `kairyu/models/{loader,parallel}.py`; `bench/{parity_hf,gate_a2}.py`

### 2026-07-27 — [amendment] m14 D2/D4: quantized CUDA forward is fused and production-wired
- What: CUDA `QuantizedLinear.forward` now dispatches every advertised scheme
  to a validated kernel: selected scaled-MM/Triton FP8, int32-accumulating
  Triton INT8, register-tiled AWQ/GPTQ W4A16, and FlashInfer native NVFP4 W4A4.
  No CUDA path calls `dequantize()` or constructs a floating `[out, in]`
  weight. Static/dynamic FP8 checkpoint semantics are preserved, ModelOpt
  scales remain fp32/fp8 at load, invalid capability/layout/dtype combinations
  fail closed, and quantized MLA is rejected until its direct projection-weight
  access has a quantized contract.
- Why: the previous helpers were direct-test-only fallbacks; production model
  calls still materialized full weights and silently forfeited quantization's
  latency and memory benefit. Kernel selection must be reachable from the model
  and unsupported combinations must never disguise themselves as a slow
  success.
- Refs: issue #205; m14 §7; roadmap §2; `kairyu/kernels/quant_gemm_gpu.py`;
  `tests/gpu/test_quant_{kernels,full_model_gpu}.py`;
  `bench/results/quant-{gemm,vllm}-rtxpro6000-2026-07-27.json`

### 2026-07-26 — [progress] G2 A1 closes on Llama-3.1-8B TP1/2
- What: the formal A1 result contains all 64 fixed prompts and token IDs, all
  four 16-token continuations (TP1/2, overlap OFF/ON), the full HF reference,
  TP1/TP2 teacher-forced results, exact checkpoint digests, CUDA/NCCL config,
  and clean-code provenance. Overlap is exact at both degrees. HF self-agreement
  is 1010/1024 (0.9863); TP1 and TP2 each achieve 1014/1024 (0.9902), zero
  substantive disagreements or missing samples, and 0.10440/0.10331 maximum
  agreeing-position logprob deltas against the 0.25 bound. All 14 assembler
  checks pass.
- Why: A1 is the foundational TP correctness anchor for later G2 performance
  gates. The first assembly attempt correctly rejected a hidden BOS mismatch;
  reference schema 4 now uses the production Kairyu no-special-token prompt
  contract, so the free-running and teacher-forced evidence uses identical
  prefixes rather than merely identical text.
- Refs: issue #151; G2 A1 and §7 amendment; GPU runbook §6;
  `bench/results/g2-a1-llama31-8b-rtxpro6000-2026-07-26.json`

### 2026-07-26 — [progress] G2 A1 has a fail-closed evidence assembler
- What: `parity_tp.py` now retains the fixed prompt text/token IDs and every
  TP1/2 overlap ON/OFF continuation, plus checkpoint architecture and the
  CUDA/NCCL runtime. `parity_hf.py` records code provenance. The new
  `gate_a1.py` command verifies the Llama-3.1-8B contract, exact weight and
  prompt identity, complete raw evidence, overlap transparency, both amended
  teacher-forced verdicts, and one clean commit before producing a
  self-contained result.
- Why: the prior diagnostics could report the component measurements but could
  neither retain the raw free-running outputs nor fail the formal gate when
  evidence was partial, stale, or came from another checkpoint.
- Refs: issue #151; G2 A1 and §7 amendment; GPU runbook §6;
  `bench/{parity_tp,parity_hf,gate_a1}.py`

### 2026-07-26 — [amendment] m2/m13/m17: Qwen3-32B TP8 LiveCodeBench finishes below the request timeout
- What: the Qwen3-32B example now has 8,192 KV pages, hardware-selected
  attention, and a 512-page CUDA-graph width. Empty scheduler plans with running
  requests raise instead of hot-spinning; an adapter can preserve a real
  control-only transition such as P-D prefill/KV handoff. Pure-greedy batches
  argmax on-device and transfer one token-id vector rather than one
  full-vocabulary row per request. FlashInfer prefill and decode share a 394
  MiB workspace, matching vLLM's serving default. Static regressions bind
  KV/graph capacity and the attention default.
- Measurement: on 8x RTX PRO 6000 Blackwell, Qwen3-32B TP8, the pinned
  LiveCodeBench 20-item subset at concurrency 8 and 8,192 max output tokens
  completed 20/20 with zero failed or timed-out inference requests. Maximum
  request latency was 460.681 s under the 600 s timeout; pair elapsed time was
  1,049 s because twenty requests ran in waves of eight. The model scored 40.0
  (twelve successful generations had no code block and scored zero).
- Why: the original 1,024-page pool held only 16,384 tokens, so all running
  requests could consume it and wait forever for pages none could release.
  After capacity was fixed, the forced torch reference backend and 64-page
  graph width left long decode unnecessarily eager. The first full FlashInfer
  run then exposed a second hard limit: chunked prefill required 190,840,832
  workspace bytes but only 134,217,728 were reserved. The 394 MiB shared
  workspace removes that failure without separate prefill/decode reservations.
- Refs: issue #150; m2 §2.2/§2.4; m13 D4; m17 A12;
  `bench/results/issue-150-qwen3-32b-tp8-livecodebench-2026-07-26.json`;
  `kairyu/engine/core/{engine_core,model_runner,sampler}.py`,
  `kairyu/engine/engine_loop.py`,
  `kairyu/engine/core/attention/flashinfer_gpu.py`,
  `examples/qwen3-32b-multi-gpu/`

### 2026-07-26 — [amendment] m16 D4: idle control receives do not inherit the model timeout
- What: TP serving now gives the gloo control group an effectively
  process-lifetime idle timeout while keeping the model group at 120 s.
  `DistTPModelRunner` marks any failed distributed step fatal; readiness requests
  process replacement, request purge stops issuing collectives, the pump does
  not hot-retry a broken group, and shutdown aborts/reaps instead of entering a
  mismatched barrier. A two-rank regression waits three model-timeout intervals
  inside the worker control receive before rank 0 delivers the next step.
- Why: the #150 Qwen3-32B TP8 rerun began after more than 120 idle seconds.
  Every worker was correctly blocked waiting for the next `StepDelta`, but the
  shared operational timeout treated that normal idle receive as a deadlock.
  The subsequent graph-teardown barrier timeout obscured the original gloo
  failure, and the backend then retried the permanently broken group in a tight
  loop. Model collectives have no pending operation while idle and retain the
  fail-fast bound.
- Refs: issue #148 follow-up / #150; m16 D4/A6;
  `kairyu/engine/core/worker.py`, `kairyu/engine/kairyu_backend.py`,
  `tests/{unit/test_tp_worker.py,dist/test_distributed.py}`

### 2026-07-26 — [amendment] m2 §2.2: batched attention has no per-row host synchronization
- What: supported eager and captured batched decode now share the tensor-only
  model/attention path. `write_from` is a fifth static device input; KV rows
  below it retain their existing cached/shared value through a tensor mask.
  Ragged eager page-table tails repeat each row's owned last page and remain
  length-masked, so eager needs no reserved graph page. FlashInfer performs its
  allowed plan once at the step boundary. A GPU profiler gate reports zero
  `aten::_local_scalar_dense` events at B=1 and B=8 for the tensor path and
  host-metadata compatibility fallback; the audited pre-fix path grew with B.
- Measurement: on Qwen3-32B TP8, 8 concurrent synthetic requests x 32 output
  tokens, list eager -> tensor eager changed wall 16.722 -> 8.844 s, TPOT
  441.831 -> 192.075 ms/token, and throughput 0.48 -> 0.90 req/s (47.1%,
  56.5%, and 87.5% improvements). Re-running Graph on the tensor path measured
  7.196 s, 130.297 ms/token, and 1.11 req/s, separating a further 18.6% wall,
  32.2% TPOT, and 23.3% throughput graph gain from the row-sync removal.
- Why: `forward_decode_batch` converted every CUDA `positions[i]` into Python
  twice per row per layer, serializing decode in proportion to batch size and
  undermining both overlap and graph launch savings. Simply routing eager
  through the graph tensor path was not safe until cached-prefix KV writes were
  masked without a data-dependent shape or Python predicate.
- Refs: issue #207; m2 §2.2; m17 D1/A5;
  `kairyu/engine/core/{kv_pool,model_runner,step_executor}.py`,
  `kairyu/models/{attention,llama}.py`,
  `tests/{unit/test_tensor_decode_path.py,gpu/test_tensor_decode_gpu.py}`,
  `bench/results/decode-row-sync-qwen3-32b-tp8-2026-07-26.json`

### 2026-07-26 — [amendment] m17 D1/D2: CUDA graph decode is a production serving mode
- What: `backend: kairyu` now accepts explicit `decode_mode: eager|cuda_graph`
  plus capture batch/page/warmup bounds. The production builder constructs
  `CudaGraphBackend` only for a real model on CUDA, retains eager fallback
  outside captured shapes, and preserves one scheduler-reserved scratch page
  across every TP rank. Real single-GPU and TP2 gates prove capture/replay,
  eager token parity, scratch capacity, and clean teardown. Qwen3-32B TP8 on
  8x RTX PRO 6000 measured eager vs graph at 16.722 vs 8.928 s wall and
  441.831 vs 194.200 ms/token TPOT; the graph service then shut down normally
  and released all eight GPUs.
- Why: m17's CUDA backend and capture contract existed but no deployment could
  construct them, so production always ran eager and the intended launch-latency
  reduction was neither selectable nor measurable. TP graph teardown also
  exposed a hidden ownership bug: a nested-class closure retained each
  `CUDAGraph`, leaving captured NCCL communicators alive through process-group
  destruction.
- Refs: m17 D1/D2/A5 + A10/A11; issue #221;
  `kairyu/engine/{kairyu_backend.py,core/{cuda_graph_gpu,model_runner,step_executor,worker}.py}`,
  `tests/gpu/test_{cuda_graph_decode_gpu,tp_cuda_graph_serve}.py`,
  `bench/results/cuda-graph-qwen3-32b-tp8-2026-07-26.json`

### 2026-07-26 — [amendment] m16 D4: TP control traffic leaves the model NCCL group
- What: TP serving now creates two 120 s operational groups in a fixed order after
  the startup handshake. `StepDelta`, release, and shutdown object broadcasts use
  gloo; `RowParallelLinear` and the other model tensor collectives use a separate
  placement-backend group (NCCL on CUDA). A permanent two-GPU gate alternates 512
  control broadcasts and model all-reduces and asserts the backend split.
- Why: the exact Qwen3-32B TP=8 / Fugu LiveCodeBench concurrency-8 workload
  reproduced issue #148: rank 1–7 timed out in BROADCAST sequence 4163 while rank 0
  had advanced to ALLREDUCE sequence 4165. Python object broadcast is a multi-stage
  metadata/payload/host-deserialization protocol and must not be interleaved with
  model tensors on the same NCCL group. With the split, the same 20-item workload
  completed its 654.53 s harness run without a watchdog, readiness remained 200,
  and normal shutdown returned all eight GPUs to 0 MiB / 0% without reset. The
  observed 600 s empty completions remain separate issues #149/#150.
- Refs: m16 D4 (amended), issue #148, `kairyu/engine/core/worker.py`,
  `tests/unit/test_tp_worker.py`, `tests/dist/test_distributed.py`,
  `tests/dist/dist_targets.py`, `tests/gpu/test_tp_control_plane_nccl.py`

### 2026-07-26 — [amendment] Empty benchmark completions are missing evidence, not zero accuracy
- What: the shared generative benchmark path now classifies a successful HTTP
  response with empty/whitespace-only completion content as a failed item,
  preserves latency for both response and transport failures, and counts only
  completed items with a numeric score in `n_scored`. If every target call is
  empty, the pair is failed with `score: null`, `n_scored: 0`, and every item
  carries the controlled empty-completion reason.
- Why: Issue #149 captured a LiveCodeBench run where all 20 requests returned no
  generated answer, yet the artifact said `completed`, `score: 0.0`,
  `n_scored: 20`, and `n_failed: 0`. That made absent evidence
  indistinguishable from 20 attempted solutions that all graded incorrect and
  could publish a false zero in a full-suite comparison.
- Refs: Issue #149; G6 P-C1 evidence rules;
  `kairyu/bench/adapters/base.py`,
  `tests/bench/test_bench_code_adapters.py`

### 2026-07-26 — [amendment] m18 D3: token 0 keeps its sampling identity across the P-D handoff
- What: the P-D prefill clone sampled under its INTERNAL id (`r#p0`) on a different
  runner and `Sampler` from decode's. Three corrections. (1) `EngineRequest.sampling_id`
  / `sampling_identity`: the clone keeps its own `request_id` for scheduler bookkeeping
  but samples under the PUBLIC id, and `PagedModelRunner` / `TorchModelRunner` key the
  sampler by that identity. (2) `Sampler.hand_over()` moves a request's sampling state
  from the prefill half to the decode half at adoption, carrying the grammar matcher.
  (3) `PDCoordinator` keeps token 0's full `SampledToken` and exposes it through
  `drain_carried_tokens()`, which `PDLoopAdapter` forwards and `EngineLoop` prepends to
  the request's logprob metadata; a token 0 that terminates the grammar finishes the
  request at adoption, before decode is planned.
- Why: [P1] on PR #144. `Sampler._state_for` derives the base seed from the request id
  when `seed is None`, so with default stochastic sampling token 0 and token 1 onward
  came off different RNG streams — greedy parity tests cannot see it, because argmax
  never consults the seed. The grammar enforcer was rebuilt decode-side from its initial
  state, having never accepted token 0, so every later mask was computed against the
  wrong grammar position. And `resume_with_kv` commits token 0 straight into the decode
  outputs, so no `execute()` reports it: its logprob/top_logprobs were dropped, and at
  `max_tokens=1` there is no decode step at all, so a one-token completion reported no
  logprobs whatsoever.
- Refs: m18 D3 (amended), m5 D5, m8 D2, `kairyu/engine/core/scheduler.py`,
  `kairyu/engine/core/sampler.py`, `kairyu/engine/core/model_runner.py`,
  `kairyu/engine/core/torch_runner.py`, `kairyu/engine/core/pd.py`,
  `kairyu/engine/core/pd_loop.py`, `kairyu/engine/engine_loop.py`,
  `tests/unit/test_pd_factory.py`, `tests/unit/test_pd.py`

### 2026-07-26 — [amendment] m18 D3: the deferred KV copy's gate is pipelined one producer step
- What: `PDCoordinator` no longer settles a deferred transfer in the step that started it.
  `_release_source_pages` is replaced by `_settle_handover`, which completes a whole
  step's transfers at once — the prefill commit, the abort on a failed copy, AND the
  decode-side `resume_with_kv` — behind one `gate_pending()`. `_step_prefill` plans, runs
  its forward, and only then settles the PREVIOUS step's `_Handover`, so the gate lands
  with that forward (and the decode step queued before it) already in front of it. A step
  with nothing to schedule settles up front instead, so the leased pages come back rather
  than deadlocking admission. Blocking handoffs are unaffected: they settle in their own
  step, as before. Correction to the entry below, which claimed the gate gave the producer
  overlap; it did not.
- Why: [P1] on PR #142. The gate was stream-ordered rather than a host block, but it sat
  immediately before every subsequent kernel — the decode step and the next prefill
  forward were both queued after it — so the device timeline stayed exactly as serial as
  the blocking form. Measured on the 8× RTX PRO 6000 host before the fix: copy
  `[11.651, 12.342] ms`, next prefill forward `[12.898, 13.369] ms`; after it the two
  intervals overlap. Adoption had to move with the release because it is the other half of
  the m6 D4 rule (it is what puts the destination pages in front of the decode runner).
  The cost is one extra step of prefill-side KV lease — capacity, never correctness, since
  a page the prefill scheduler still owns cannot be handed to anyone else.
- Refs: m18 D3 (amended), `kairyu/engine/core/pd.py`, `tests/unit/test_pd.py`
  (`test_a_deferred_copy_has_engine_work_queued_alongside_it`,
  `test_a_blocking_handoff_settles_inside_its_own_step`),
  `tests/gpu/test_handoff_stream_gpu.py`
  (`test_the_copy_overlaps_the_next_prefill_forward_on_the_coordinator`)

### 2026-07-26 — [amendment] one copying KV handoff in the P-D stack, not two
- What: corrects two claims in the entry "the `pd_separation` serving path, corrected
  after review" below. (1) It named `pd_factory.PagedCopyKVHandoff` as the handoff that
  moves the KV bytes. The same [P1] had already been fixed one PR down the stack as
  `pd.LocalCopyKVHandoff` (see "a production P-D constructor, and a handoff that actually
  copies KV" below), on a branch this one had forked before. The duplicate is deleted and
  `build_kv_handoff` / `build_pd_coordinator` wire `LocalCopyKVHandoff`, now the single
  copying handoff in the stack. (2) It said the serving path takes the blocking handoff
  because nothing settled a deferred copy. The entry "the deferred KV copy keeps its
  source lease, and gets a caller" landed exactly that consumer, so the serving path now
  inherits `build_pd_coordinator(defer_handoff=True)`: `PDLoopAdapter.schedule()` drives
  `PDCoordinator.step_prefill`, whose `_release_source_pages` gates every prefill-side
  release on the copy's completion event.
- Why: two branches of one stack fixed the same P1 independently and only met at the
  merge. Two implementations of the same Protocol are a place for the two to drift, and
  the serving path would otherwise have used whichever one the merge happened to leave
  wired. `LocalCopyKVHandoff` is the one kept because it validates more at the seam: pool
  geometry AND cache/pool page-size agreement at construction, plus a source page count
  that matches the prompt at transfer, where the duplicate only checked that some pages
  were passed. Its tests subsume the duplicate's, so no coverage is dropped with it.
  Correcting rather than editing the entries below, per the append-only rule.
- Refs: m18 D3, `kairyu/engine/core/pd.py`, `kairyu/engine/core/pd_factory.py`,
  `kairyu/engine/core/pd_loop.py`, `tests/unit/test_pd_factory.py`,
  `tests/gpu/test_handoff_stream_gpu.py`, PRs #140 / #142 / #144

### 2026-07-26 — [amendment] the `pd_separation` serving path, corrected after review
- What: corrects the entry below it, which claimed the P-D serving chain was complete.
  It was not: the loop it described could not execute even its first request, and would
  have decoded from empty KV if it had. Three fixes. (1) `engine/core/pd_loop.py`
  `PDLoopAdapter` — `EngineLoop._drain_ops` adds submissions to the scheduler it was
  given, so wiring the loop to the coordinator's decode scheduler inserted requests into
  DECODE with no prompt KV, never called `PDCoordinator.add_request`, and then called
  `execute()` on a coordinator that has no such method. The adapter is the loop's
  scheduler AND runner: submissions enter at prefill, each `schedule()` runs a prefill
  step (prefill → KV transfer → commit) before planning decode, and abort/forget/release
  reach both halves. (2) `pd_factory.PagedCopyKVHandoff` — the two halves own two POOLS,
  and `LocalKVHandoff` only does the accounting (allocate + mark computed), so the
  destination pool stayed zero-initialised; the paged handoff copies the non-cached
  pages, all layers at once, with the same receiver-side prefix dedup as
  `RemoteKVReceiver`. (3) `_build_pd_loop` returns the cache and scheduler the loop
  actually drives (decode's cache, the adapter) instead of a third, unrelated pair.
- Why: the previous entry also read as if production overlapped the copy via
  `StreamCopyKVHandoff(defer=True)`. It does not, and should not yet: the serving path
  takes the blocking default so the commit point cannot run ahead of the copy (m6 D4).
  The deferred form stays opt-in with no production caller until a consumer-side
  `wait_for_pending()` and source-page lifetime ordering land. Correcting rather than
  editing, per the append-only rule.
- Refs: m18 D3 (amended), `kairyu/engine/core/pd_loop.py`,
  `kairyu/engine/core/pd_factory.py`, `kairyu/engine/core/pd.py`,
  `kairyu/engine/kairyu_backend.py`, `tests/unit/test_pd_factory.py`, PR #144 review

### 2026-07-26 — [progress] P-D disaggregation is reachable from a deployment (G2 stage 5.3)
- What: `pd_factory.build_pd_coordinator()` assembles a prefill/decode pair from a
  checkpoint and `build_kv_handoff()` picks the handoff from where the KV lives;
  `backend: kairyu` accepts `pd_separation: true`, so a deployment YAML can serve through
  the pair. `StreamCopyKVHandoff(defer=True)` records a completion event instead of
  blocking, so the producer can queue its next step while the copy runs.
- Why: `PDCoordinator` had no production constructor at all — it existed only in
  `tests/unit/test_pd.py`, which is why `CudaStreamProvider` had no caller and why m2
  §2.4's reserved `pd_separation` surface was never wired. TP > 1 and speculative decoding
  are rejected with pd_separation rather than silently serving a different topology.
- Refs: m18 D3 (amended), `kairyu/engine/core/pd_factory.py`,
  `kairyu/engine/core/handoff_stream.py`, `kairyu/engine/kairyu_backend.py`

### 2026-07-26 — [amendment] m18 D3: the deferred KV copy keeps its source lease, and gets a caller
- What: `StreamCopyKVHandoff(defer=True)` records the copy's completion event instead of
  blocking, and `PDCoordinator` is now the consumer that settles it.
  `_release_source_pages` became the single point where prefill-side pages go back —
  commit and abort alike — and it calls `gate_pending()` first. The gate is stream-ordered
  (`event.wait(current_stream)`), not a host block. Deferred events now ACCUMULATE rather
  than occupying one slot. `PDCoordinator` refuses a deferring handoff that exposes no
  `gate_pending()`. `pd_factory.build_pd_coordinator(defer_handoff=True)` is the default
  and the only caller that enables deferring; `build_kv_handoff(..., defer=...)` stays off
  for everyone else.
- Why: two [P1]s on PR #142. (1) Deferring returned while the copy was still READING the
  prefill-side pages, and the coordinator's m6 D4 release ran immediately after — so the
  next prefill step could allocate the same page and overwrite it on the caller's stream
  under the running copy. Waiting on the destination does not prevent that; it is a
  source-side read/write race, and the failure path (`abort()` → `release_preempted`) had
  it too, since a raising transfer may already have queued part of the copy. A single
  `pending_event` slot compounded it: a prefill step transfers every prompt that completed
  in it, so all but the last copy went unordered. (2) Nothing in production enabled or
  settled the deferred path — `build_kv_handoff` constructed the blocking form and no
  caller used `wait_for_pending` — so "overlap landed" was not true of any shipped path.
- Refs: m18 D3 (amended), `kairyu/engine/core/handoff_stream.py`,
  `kairyu/engine/core/pd.py`, `kairyu/engine/core/pd_factory.py`,
  `tests/unit/test_pd.py`, `tests/unit/test_pd_factory.py`,
  `tests/gpu/test_handoff_stream_gpu.py`

### 2026-07-26 — [amendment] m18 D3: a production P-D constructor, and a handoff that actually copies KV
- What: `kairyu/engine/core/pd_factory.py` gives `PDCoordinator` its first production
  constructor — `build_pd_coordinator()` assembles a prefill/decode pair from one
  checkpoint, and `build_kv_handoff()` selects the handoff from where the KV lives (host
  pool → plain, device pool → `StreamCopyKVHandoff` over a `CudaStreamProvider` bound to
  that pool's device). Review [P1] then found the constructor wrapped `LocalKVHandoff`,
  which copies no bytes; `pd.LocalCopyKVHandoff` replaces it as the inner handoff.
  Review [P2]: placement (`profile`/`device`/`dtype`/`attention_backend`) is now
  injectable, so unit tests no longer probe hardware or require optional FlashInfer.
- Why: the stream seam had no production caller because nothing could reach a `KVHandoff`
  at all — `PDCoordinator` existed only in `tests/unit/test_pd.py`. Correctness: the two
  halves own separate `PagedKVPool`s, so the accounting-only `LocalKVHandoff` published
  the decode pool's untouched (zeroed) pages as *computed*. Decode continued from KV that
  was never written and the radix tree served that wrong prefix to later matches —
  silently, since nothing raises and the token counts are unchanged.
  `LocalCopyKVHandoff` does a direct pool-to-pool page copy (D2D on a device pair, rather
  than `RemoteKVHandoff`'s D2H+H2D serde round-trip, since in-process there is no wire),
  skipping only the destination's already-cached prefix pages, and releases the allocation
  rather than publishing a half-written one if the copy fails. `LocalKVHandoff` is now
  documented as a test double, not a deployment option. Tests assert destination KV byte
  parity with the source and P-D/single-engine greedy token parity, both of which fail
  without the copy. Still open: overlap with the next forward (needs a completion event
  instead of the host-wide wait), and the serving-layer `pd_separation` option (G2 5.3).
- Refs: m18 D3 (amended twice), `kairyu/engine/core/pd_factory.py`,
  `kairyu/engine/core/pd.py`, `tests/unit/test_pd_factory.py`,
  `tests/gpu/test_handoff_stream_gpu.py`

### 2026-07-26 — [amendment] A1's overlap ON/OFF equality is measured, not inferred
- What: corrects the entry below it. `bench/parity_tp.py` compared each overlap mode
  against its OWN TP1 base and dropped the outputs when the next mode overwrote them,
  so ON and OFF were never compared to each other; "ON reproduces OFF exactly" was read
  off two aggregate rows agreeing. Outputs are now retained per (degree, mode) and
  compared directly, recording the first disagreeing request id, position and token
  pair, and the harness exits non-zero when they differ. Re-measured on the 8x
  RTX PRO 6000 host: TP1/2/4/8 all 64/64 exact, token rate 1.0, no first mismatch.
- Why: two runs can diverge on DIFFERENT prompts at the same depth and land on identical
  exact_match, tokens, token_match_rate and median_first_divergence — equal aggregates
  are not sequence equality. Unlike the cross-TP rates (reduction order, orientation
  only per G2 §7), ON vs OFF is the same ranks in the same order, so a difference is the
  pipeline changing an answer. That makes it a verdict rather than a report.
  The evidence also carries the corrected checkpoint provenance: the previous digest
  hashed only safetensors headers plus file sizes, which a base model and a fine-tune of
  it share, so it identified layout rather than weights.
- Refs: G2 A1, m2 §2.2, `bench/parity_tp.py`,
  `bench/results/parity-tp-qwen3-32b-2026-07-26.json`

### 2026-07-26 — [progress] G2 A1's overlap-ON half is measured, and it matches OFF
- What: `bench/parity_tp.py` no longer forces `overlap_modes` to OFF when a real model is
  loaded, so the A1 sweep runs both halves. On the 8x RTX PRO 6000 host, Qwen3-32B, 64
  fixed prompts x 8 new tokens, overlap ON reproduces overlap OFF exactly at every TP
  degree: TP2 57/64 (token 0.9277), TP4 58/64 (0.9473), TP8 53/64 (0.9023) in both modes.
  Evidence: `bench/results/parity-tp-qwen3-32b-2026-07-26.json`.
- Why: A1 requires parity with the overlap pipeline ON *and* OFF, and only OFF had ever
  been measured — `PagedModelRunner` read `state.outputs[position - 1]`, which an overlap
  snapshot is one entry short of, so a real runner raised IndexError and the harness
  recorded the gap instead of a number. The in-flight token buffer removed that
  precondition; this run is what shows the pipeline changes no output rather than
  asserting it. The rates themselves are orientation only, per G2 §7 (amended
  2026-07-25): free-running greedy equality is not the correctness bar, because one
  flipped token fails a prompt and every token after it. What A1 gets from this run is
  the ON-vs-OFF equality, which is exact.
- Refs: G2 A1, m2 §2.2, `bench/parity_tp.py`, `bench/results/parity-tp-qwen3-32b-2026-07-26.json`

### 2026-07-26 — [amendment] Growing the decode slots waits for the in-flight staging DMA
- What: `PagedModelRunner._allocate_decode_slots` now calls `_retire_decode_slots()`
  first, which synchronizes the OLD `_slot_copy_done` event and drops the old pinned
  staging buffer only afterwards. A new CUDA gate
  (`test_growing_the_slots_waits_for_the_in_flight_staging_dma`) keeps a real transfer
  outstanding (`torch.cuda._sleep` ahead of the H2D) and asserts that no replacement
  buffer is allocated while the old event is unfinished.
- Why: PR #143 review [P2] on the staging lifecycle. `_decode_input_slots` grew capacity
  BEFORE it synchronized, and the growth overwrote `_slot_staging` and `_slot_copy_done`
  together — so the `synchronize()` that followed was on a freshly recorded, empty event
  and the handle on the outstanding DMA was already gone. Freeing pinned host memory is
  not stream-ordered, so the source rows could be returned to the allocator while the copy
  engine was still reading them. Ordinary generation only survived because the host
  sampler pulls logits to CPU every step — exactly the dependency the previous entry
  claimed the event had removed. An active decode batch can grow (8 → 9+) mid-run, so the
  claim in the 2026-07-26 [progress] entry below ("ordered against reuse by a CUDA event")
  did not hold across a growth until this change; it holds now.
- Refs: m2 §2.2 + §5, PR #143 review, `kairyu/engine/core/model_runner.py`,
  `tests/gpu/test_decode_input_slots_gpu.py`

### 2026-07-26 — [progress] m2 §2.2: persistent decode input slots, and the honest scope
- What: the decode inputs (token ids and positions) live in persistent device tensors
  allocated once and written IN PLACE every step
  (`PagedModelRunner._decode_input_slots`), used by the batched AND the single-request
  decode path. No decode step allocates a device tensor, and the one-request tail of a
  workload no longer takes a different, host-rebuilding path. On CUDA the ids are staged
  through pinned memory and copied as one async DMA per step, ordered against reuse by a
  CUDA event.
  NOT done, and now stated as OPEN in `overlap.py`, `model_runner.py` and m2 §2.2 instead
  of claimed: filling those slots DEVICE-to-device, and §2.2's "step loop never blocks on
  `.item()`/`.cpu()`" invariant.
- Why: the sampler decides on the CPU because m8 D2 pins reproducibility (incl. spec ≡
  greedy) to the CPU RNG stream. The chosen id therefore exists only as a Python int and
  there is nothing on the device to copy from; one batched H2D per step is the floor until
  the sampling DECISION itself moves onto the device, which redefines what those pins mean.
  An earlier revision of this entry claimed the device-to-device patch was done, on the
  strength of a `SampledToken.device_token` that was `torch.as_tensor(int, device=cuda)` —
  a fresh scalar H2D per row, i.e. the same round trip B times over rather than once. That
  claim and that field are withdrawn (PR #143 review, [P1]); the single-request path gap is
  [P2] of the same review. A second, independent violation of the same invariant remains in
  `models/attention.py::forward_decode_batch` (`int(positions[i])` per row per layer).
- Refs: m2 §2.2 + §5, PR #143 review, `kairyu/engine/core/model_runner.py`,
  `kairyu/engine/core/sampler.py`, `tests/unit/test_decode_input_slots.py`,
  `tests/gpu/test_decode_input_slots_gpu.py`

### 2026-07-26 — [amendment] m16 D6 records sequence parallelism; the §3 call-site non-goal is lifted
- What: the entry below landed sequence parallelism but updated PROGRESS only, leaving the
  binding design doc contradicting it — m16's D1 amendment still said SP was "a design
  change this milestone does not specify" and §3 still listed USING `reduce_scatter` at the
  `RowParallelLinear` call site as a non-goal, which is exactly what shipped. Reconciled in
  `docs/design/m16-distributed.md`: new **D6** records the `SequenceParallelContext`
  contract (`scatter`/`gather`/`reduce_scatter`), the wrapper placement over D2's tree, the
  rule that padding lives at the shard boundary ONLY (attention builds its mask from the
  real length), the activation-memory-not-latency framing, and the scope — dense
  `build_tp_model` only, NOT EP/PP/the SPMD worker. §3's non-goal is narrowed to those
  unwired paths and to making SP the default; D1's closing paragraph now points at D6; the
  Status line and §5 verification list the gates. No code change.
- Why: repo rule — a design change must move the D-IDs in `docs/design/` and PROGRESS in the
  SAME change. A design doc that denies what the code does is worse than silence: the next
  agent reads §3, believes the call site is untouched, and reasons from a false premise.
  Recorded as an amendment rather than by editing the entry below, which stays as written.
- Refs: m16 D1/D2/D6 + §3/§5 (`docs/design/m16-distributed.md`), PR #139 review [P2],
  commit 4d1f9f0

### 2026-07-26 — [design] Sequence parallelism (Megatron TP+SP) behind an opt-in flag
- What: `build_tp_model(..., sequence_parallel=True)` shards the residual stream between
  blocks along TOKENS. The norms run on the shard, their output is all_gathered into the
  TP region, and the row-parallel `o_proj`/`down_proj` exit with a reduce_scatter instead
  of an all_reduce. Ragged token counts are padded at the shard boundary and trimmed on the
  way out, so the TP region always sees the real sequence length (attention builds its mask
  from it). Off by default; `tp >= 2` required.
- Why: m16 §3 listed reduce_scatter at the RowParallelLinear call site as a non-goal
  because a bare swap loses — measured at ~0.96x an all_reduce (m16 D1 amendment,
  2026-07-25). The 1.90x that `reduce_scatter` alone shows is only reachable if the
  consumer accepts a shard, which is what this makes true. The honest framing, recorded
  here so nobody enables it for the wrong reason: all_gather + reduce_scatter moves what
  one all_reduce moves, so this does NOT reduce comm time. The gain is ACTIVATION MEMORY —
  the norms and the inter-block residual hold S/tp rows instead of S.
- Refs: m16 D1/D2 (+2026-07-25 amendment), `kairyu/models/parallel.py`,
  `tests/dist/test_distributed.py`, `tests/gpu/test_sequence_parallel_nccl.py`

### 2026-07-26 — [amendment] FlashInfer declares graph capture and is planned by the runner
- What: `FlashInferBackend` now sets `supports_graph_capture = True`, so the
  `GraphDecodeBackend` gate accepts it and `PagedModelRunner` will build a graph path
  over it. Its `plan_decode()` already had the contract's signature and is now called
  by production code — `GraphStepExecutor` -> `DenseDecoder.plan_decode_tensors()` ->
  the backend — instead of only by tests. Gated on the 8× RTX PRO 6000 host by four
  new integration tests in `tests/gpu/test_flashinfer_tensor_decode.py` that drive
  real capture and replay through `PagedModelRunner` (growing seq_lens, pages swapped
  mid-run, and a `warmup_iters=0` capture that only the pre-capture hook can save),
  plus a CPU gate that FlashInfer satisfies `graph_capture_gap()`.
- Why: PR #141's review found `plan_decode` had no production caller at all — the
  decode path reached `attend_decode` per layer and the executor had no step-boundary
  hook — so combined with #138's graph path the first capture raised "no live plan",
  and a replay after `_copy_in` would have attended over the pages that were in the
  static buffers at capture time. Removing the post-copy-in plan reproduces exactly
  that: two steps over different pages return byte-identical logits.
- Refs: PR #141 review [P1], PR #138 review [P1]; m17 D1, m13 D4;
  `kairyu/engine/core/attention/flashinfer_gpu.py`,
  `tests/gpu/test_flashinfer_tensor_decode.py`, `tests/unit/test_attention_backend.py`.

### 2026-07-26 — [design] GraphDecodeBackend: the CUDA-graph decode capability contract
- What: a backend is capture-eligible only if it DECLARES it, and the decode step
  boundary now reaches it. Defined once in `kairyu/engine/core/attention/__init__.py`
  as the `GraphDecodeBackend` protocol plus its single enforcement point
  `graph_capture_gap()`: `supports_graph_capture` (declared, not inferred),
  `plan_decode(kv_pool, page_tables, seq_lens, *, num_qo_heads, q_dtype)` (the
  step-boundary HOST phase, a no-op where there is none), and the capture-safe
  `attend_decode`. `GraphStepExecutor` gained an optional `plan_fn`, called before
  each capture and after every `_copy_in` BEFORE `replay()`; `PagedModelRunner`
  supplies it via the new `DenseDecoder.plan_decode_tensors()`, which plans once per
  backend INSTANCE per step (not per layer). `_tensor_decode_gap()` now asks
  `graph_capture_gap()` instead of `hasattr(backend, "attend_decode")`.
- Why: method presence is not capture-safety. FlashInfer (PR #141) owns
  `attend_decode` and would have passed the old check, but its `plan()` copies
  `indptr` to the host and cannot run under capture at all — so the runner would
  construct successfully and the FIRST capture would die with a D2H `RuntimeError`,
  defeating the fail-fast the gate exists for. And planning outside capture is
  useless if nothing calls it: `forward_decode_tensors` reaches `attend_decode` per
  layer with no seam for a host phase, and only the executor knows which static
  buffers the next replay will read — the plan must therefore be taken after
  copy-in, or the replay attends over the previous step's pages.
- Refs: m17 D1/D2, PR #138 review [P1], PR #141 review [P1];
  `kairyu/engine/core/attention/{__init__,torch_backend}.py`,
  `kairyu/engine/core/{model_runner,step_executor}.py`,
  `kairyu/models/{attention,llama}.py`, `tests/unit/test_step_executor.py`,
  `tests/unit/test_graph_decode_wiring.py`.

### 2026-07-26 — [amendment] CUDA-graph decode: reserved scratch page + backend fail-fast
- What: `PagedModelRunner(graph_backend=...)` now wires the m17 D1 capture seam into
  batched decode (PR #138), with two review blockers fixed before merge. [P1] the
  scratch page the graph's padding rows write KV to is now RESERVED out of the
  allocator via the new `RadixKVCache.reserve_scratch_page()`, and a graph backend
  without a cache is rejected; `GraphStepExecutor`/`build_decode_batch` no longer
  default `scratch_page` to 0. [P2] the runner now checks at construction that every
  layer's attention implements the tensor decode contract
  (`forward_decode_tensors` + `backend.attend_decode`) and raises `ValueError`
  naming the gap. The seam stays OFF unless a backend is passed.
- Why: the scratch page defaulted to 0, which `PagePool` hands out as the FIRST
  ordinary page — so the capture warmup and every partial-bucket replay wrote K/V
  into a live request's page 0 slot 0, silently corrupting its cache whenever the
  damage did not cross an argmax boundary. And a FlashInfer or MLA model constructed
  fine and then died with `AttributeError` on the first batched decode, arbitrarily
  deep into a run, rather than at build time. Capacity now drops by exactly one page
  for the graph's lifetime — the documented cost of the reservation.
- Refs: m17 D1/D2/A5, `docs/gpu-runbook.md` §6.3, PR #138 review [P1]/[P2];
  `kairyu/engine/core/{model_runner,step_executor,radix_kv}.py`,
  `tests/unit/test_graph_decode_wiring.py`, `tests/gpu/test_cuda_graph_decode_gpu.py`.

### 2026-07-26 — [amendment] FlashInfer decode is split into a host plan and a capture-safe run
- What: `FlashInferBackend.attend_decode` no longer derives-and-plans inline. The adapter
  now owns `plan_decode()` (the HOST phase: device-derived indptr/indices/last_page_len
  handed to a `use_cuda_graph=True` wrapper whose paged buffers are persistent, one
  wrapper per (batch, max_pages) shape and never replaced) and `attend_decode()` (a bare
  `run()` over those buffers — no `.tolist()`, no `.cpu()`, no `plan()`). Inside a capture
  the adapter refuses to plan and refuses to run unplanned; eagerly it still plans lazily,
  now once per step instead of once per layer. Gated by a real `torch.cuda.CUDAGraph`
  capture on the 8× RTX PRO 6000 host whose replay reflects an in-place page-table/seq-len
  change after the step is re-planned, plus a CPU gate that forbids host synchronization
  inside the capture region.
- Why: the first cut of PR #141 claimed the m17 D1 tensor contract but converted the page
  table and lengths with `.tolist()`/`.cpu()` on every layer, so capture died with
  `cudaErrorStreamCaptureInvalidated` (reproduced on hardware) and the path was eager-only.
  FlashInfer's `plan()` "cannot be used in Cuda Graph" by its own documentation — it builds
  the split-KV schedule on the CPU — so the honest decomposition is plan-outside /
  run-inside, which is what the wrapper's cudagraph buffers exist for.
- Refs: PR #141 review [P1]; m17 D1, m13 D4; `kairyu/engine/core/attention/flashinfer_gpu.py`,
  `tests/unit/test_attention_backend.py`, `tests/gpu/test_flashinfer_tensor_decode.py`.

### 2026-07-26 — [amendment] Checkpoint provenance is an exact content digest, not a sampled one
- What: `bench/parity_hf.py` now fingerprints a checkpoint by the complete SHA-256 of every
  weight file, recorded per file as `checkpoint_weight_files` with a rollup in
  `checkpoint_weights_sha256` (reference schema 2 → 3). The reference and both TP results
  were regenerated under it and the pre-schema-3 files removed; the corrected paths are
  `bench/results/gate1-hf-parity-tp{1,8}-2026-07-26.json`, superseding the
  `-2026-07-25` names in the Refs of the entry below. All 17 recorded shard digests are
  byte-identical to the LFS oids Hugging Face publishes for `Qwen/Qwen3-32B@main`, so the
  evidence identifies its checkpoint against an upstream immutable revision. Numbers are
  unchanged: TP=1 253/256 = 0.9883, TP=8 251/256 = 0.9805, 0 substantive, all samples
  collected — and the HF reference regenerated bit-identically outside the provenance block.
- Why: the fingerprint it replaces hashed each shard's safetensors header plus four fixed
  4 KB windows. A weight edit anywhere between those windows left it unchanged, so a
  reference cache built from DIFFERENT weights was accepted while the field claimed to pin
  the bytes (review [P2] on #131). A sampled fingerprint cannot be the basis for cache
  safety: the bytes it skips are exactly the ones a swap changes. G2 §8 requires a number a
  decision rests on to be reviewable next to the config that produced it, which a
  machine-local path plus a partial hash does not deliver. Reading all 64 GB costs ~20 s
  against a run that loads the same bytes onto a GPU anyway.
- Refs: `bench/parity_hf.py`, `tests/verification/test_parity_hf_gates.py`,
  `bench/results/hf-reference-qwen3-32b.json`,
  `bench/results/gate1-hf-parity-tp{1,8}-2026-07-26.json`,
  `docs/goals/g2-multi-gpu.md` §8 + A1/A2 amendment

### 2026-07-25 — [amendment] A1/A2 and m2 §2.5 restated against measured quantities
- What: G2 A2's "greedy output-match rate >=99%" and m2 §2.5's "greedy-decode token-level
  parity with HF transformers" are replaced by two measured criteria, both computed by the
  new `bench/parity_hf.py` (a DIAGNOSTIC — the formal gate additionally needs full
  continuations with overlap ON, blocked on the unimplemented m2 §2.2 future-token
  patch): (a) zero substantive disagreements — every disagreement inside
  the reference's top-k and within a tie gap measured from the reference's own
  self-disagreements; (b) agreement at or above the reference's self-agreement rate. The
  gate is teacher-forced (identical prefix at every position, next token only); free-running
  greedy sequence equality is explicitly no longer a correctness gate.
- Why: the fixed 99% is not achievable by any implementation, including the reference's own.
  On Qwen3-32B/bf16/8x RTX PRO 6000, HF transformers agrees with ITSELF — `generate()`
  against a teacher-forced forward over the same sequence — on only 251/256 = 0.9805
  positions, while kairyu agrees with HF on 253/256 = 0.9883 at TP=1 and 251/256 at TP=8.
  A gate demanding an engine match a reference more closely than the reference matches
  itself measures the reference's instability. The same applies to the logprob half: a 0.1
  nat tolerance sits below bf16's ~0.125 quantization of these gaps, and one observed gap
  was negative (HF's forward scoring kairyu's pick above HF's own choice). Separately,
  free-running comparison scored the same engine at 0.786 against 0.988 teacher-forced —
  once one token differs, every later token is compared against a prefix the other side
  never produced, so a single moved near-tie is indistinguishable from a broken shard.
- Refs: G2 §7 amendment (2026-07-25), `docs/design/m2-engine.md` §2.5, `bench/parity_hf.py`,
  `bench/results/gate1-hf-parity-tp{1,8}-2026-07-25.json`,
  `bench/results/hf-reference-qwen3-32b.json`

### 2026-07-25 — [progress] overlap ON works with a real runner (host-side in-flight tokens)
- What: `PagedModelRunner` keeps the token it just sampled, so a decode can read
  `position - 1` before that token is committed. `OverlapEngineCore` takes the snapshot
  for step N+1 before step N commits, so `state.outputs` is one short — every real-model
  overlap run raised `IndexError: tuple index out of range`. Committed outputs still win
  over the in-flight value, so a speculative rollback is not shadowed, and only the newest
  position is retained (a decode reads exactly one).
- Why: `overlap.py` already specified this — "decode chunks carry an explicit position so
  the runner never needs previously-committed token values from the host (on GPU, the
  last-token slot is patched device-side)" — and nothing implemented it. The toy runner
  honours the contract by ignoring outputs entirely, which is why no CPU test could see
  the gap. It blocked the overlap-ON half of G2 A1 and runbook §1 Gate 1, both of which
  require overlap ON and OFF.
  Scope: this is the HOST-SIDE half of m2 §2.2. That section specifies patching the
  placeholder slot device-to-device from the sampled tensor with no host sync in the hot
  path; this keeps Python ints and rebuilds the input tensor each step. Correctness is
  restored — overlap ON now runs and matches OFF — but the device-side technique and the
  zero-host-sync invariant remain OPEN, along with the perf gate that would show them.
- Refs: m2 §2.2 (partially), G2 A1, `kairyu/engine/core/model_runner.py`,
  `tests/unit/test_overlap_future_token.py`, `tests/gpu/test_overlap_future_token_gpu.py`

### 2026-07-25 — [amendment] m18 D3: CudaStreamProvider landed; KV serde can read a device pool
- What: `CudaStreamProvider` implemented (the deploy-day half of the m18 D3 stream seam),
  and `kv_serde._to_bytes` now copies to host before `.numpy()`.
- Why: the seam had never run against a CUDA pool. `_to_bytes` called `.numpy()` on the
  tensor directly, so `extract_page` on a device `PagedKVPool` raised `TypeError: can't
  convert cuda:0 device type tensor to numpy` — the real handoff could not read GPU KV at
  all, which a side stream does not help with. Scope is recorded honestly: the extraction
  copy is now isolated on a side stream that waits on the caller's stream FOR THE DEVICE
  THE PROVIDER WAS BUILT FOR (an argument-less `current_stream()` follows the thread's
  current device instead). End-to-end overlap with the next forward is NOT delivered —
  `StreamCopyKVHandoff` blocks the host before returning and `PDCoordinator` commits
  before stepping decode — and neither is production wiring; both need a completion event
  handed to the consumer rather than a host-wide wait.
- Refs: m18 D3 (amended), `kairyu/engine/core/handoff_stream.py`,
  `kairyu/engine/core/kv_serde.py`, `tests/gpu/test_handoff_stream_gpu.py`

### 2026-07-25 — [amendment] m16 D1: reduce_scatter implemented; "same-call-site optimization" withdrawn
- What: `TorchDistCommunicator.tensor_reduce_scatter` added — NCCL's real collective, and
  all_reduce + a local slice under gloo, which has none. The D1 note calling NCCL's
  reduce_scatter "a same-call-site optimization recorded for deploy day" is withdrawn as
  incorrect, and the m16 §3 non-goal is narrowed to USING it at the `RowParallelLinear`
  call site (which needs sequence parallelism), not to the primitive.
- Why: measured, not assumed. On 8x RTX PRO 6000 Blackwell, 8192x5120 bf16, torch
  2.12.1+cu130 / NCCL 2.29.7 — per-trial worst-rank elapsed via CUDA events,
  barrier-bounded, MAX-reduced across ranks, buffers outside the timed region, paths
  interleaved, 120 samples each (6 rounds x 20 trials) — `all_reduce` medians 3.784 ms while
  `reduce_scatter`+`all_gather` medians 3.944 ms. Swapping one for the other at the same
  call site moves the same bytes and adds a launch, so it LOSES; all_reduce's p95 sits
  below rs+ag's MINIMUM, so this is not straggler noise (the full supports do overlap — a
  few all_reduce samples land above rs+ag's floor — so the claim is about the bulk). `reduce_scatter`
  alone medians 1.988 ms (1.90x), but its output is a shard, so realising that win means
  sequence parallelism — a design change m16 does not specify and one that should be
  argued on activation memory as much as on comm time. The call site is deliberately
  unchanged.
- Refs: m16 D1 + §3 (amended), `kairyu/engine/core/dist_comm.py`,
  `bench/reduce_scatter_bench.py`, `bench/results/reduce-scatter-2026-07-25.json`
  (raw per-trial samples committed)

### 2026-07-25 — [progress] Multi-process TP places its shards on the GPU
- What: `build_engine_loop` returns into `_build_dist_tp_loop` for `model_path` +
  `tensor_parallel_size > 1`, which happens BEFORE the `probe()` block that selects
  `compute_device`/`compute_dtype` and calls `model.to(...)`. Every spawned rank therefore
  kept the CPU/fp32 defaults of `DenseDecoder` and `PagedKVPool`, so the
  `examples/qwen3-32b-multi-gpu` deployment ran Qwen3-32B on the host in fp32 over gloo —
  8 GPUs at 0 MiB, ~153 GB of host RAM, and generation that never returned. Added
  `TPPlacement`/`tp_placement()` in `engine/core/worker.py`: one CUDA device per rank
  (`cuda:<rank>`), bf16, and the NCCL backend, threaded through `build_tp_runner` →
  `build_tp_model` → `PagedKVPool`. `torch.cuda.set_device(rank)` now precedes
  `init_process_group`; `TorchDistCommunicator` takes the rank's device so its own
  `all_reduce`/`barrier` do not hand NCCL host tensors; the attention backend is selected
  from the PLACEMENT rather than the raw probe (a CPU-placed rank on a GPU box was getting
  the flashinfer kernel and fp32 tensors); and the rendezvous timeout is raised from the
  CI-tuned 120 s, which a cold multi-GB shard read trips before anything is deadlocked.
- Why: the GPU half of M5 assumed the placement the single-process path performs; nothing
  performed it for the multi-process path, and no CPU test could observe the difference
  because on a CPU box the defaults are correct. This was the first real-hardware use of
  `kairyu serve --tp N`.
- Refs: m5 D1/D3, m16 D1/D2, `docs/gpu-runbook.md` §6.1; `kairyu/engine/core/worker.py`,
  `kairyu/models/parallel.py`, `kairyu/engine/core/dist_comm.py`,
  `tests/dist/test_distributed.py` (CPU parity now pins `force_cpu=True`).

### 2026-07-25 — [amendment] Review remediation across the Fugu bench alignment PRs
- What: Addressed the review findings on the nine bench PRs. Highlights: pinned revisions
  are now passed to every fetch (they were recorded but unused, so the cache and run
  fingerprint attested unpinned bytes as pinned); LiveCodeBench Pro fails closed on any
  partial fetch instead of caching a shrunken denominator; MRCRv2 selects the official
  `o200k_base` prompt+answer bins (exactly 500 rows) rather than a chars/4 approximation;
  SciCode scores the official 288 sub-steps, supplies the three the evaluator skips, and
  pins `test_data.h5` by content hash; Harbor/τ commands match the real 0.17/tau2
  contracts (`trial_results` + `verifier_result.rewards`, imported `DATA_DIR`, unique
  `--save-to`); `extra_body` can no longer override built request or sampling fields; the
  progress reporter can no longer abort a run and heartbeats agentic slots; comparability
  is decided per cell so subset/fixture/dynamic-substitution runs withhold their deltas;
  and the Qwen example declares its text-only target so vision slots skip honestly.
- Why: Each finding let a number look like something it was not — a false provenance
  attestation, a shrunken denominator, a different population, or an unmarked delta
  against a published full-suite score.
- Refs: PRs #115–#123 review comments; `kairyu/bench/**`, `tests/bench/**`,
  `tests/unit/test_qwen_fugu_example.py`, `examples/qwen3-32b-multi-gpu/**`,
  `docs/benchmarks.md`.

### 2026-07-25 — [progress] One-command Fugu quality benchmark for the Qwen3-32B example
- What: Added `examples/qwen3-32b-multi-gpu/{run-,}fugu-benchmark.sh`. The first starts the
  all-GPU service (reusing one already running), waits for readiness, and chains into the
  second, which preflights the served model by exact id, declares the text-only target
  with `--no-vision`, runs `kairyu bench run` from the repository with the `bench` extra,
  and points the operator at both `scoreboard.md` and `comparison.md`. Every Fugu
  condition is reachable by environment variable (`REASONING_EFFORT`,
  `JUDGE_REASONING_EFFORT`, `EXTRA_BODY`, `ATTEMPTS`, `BENCH_ONLY`); `BENCH_LIMIT` defaults
  to 20 items per slot and announces itself as a subset run, with `BENCH_LIMIT=0` for the
  full suite. `PORT` reaches the Compose mapping. A static test pins the properties that
  would otherwise silently produce a misleading number.
- Why: The existing example only measured throughput. The quality suite runs on the host
  rather than in the serving image because the image carries no dataset dependencies —
  documented rather than worked around.
- Refs: `examples/qwen3-32b-multi-gpu/{fugu-benchmark.sh,run-fugu-benchmark.sh,README.md,
  compose.yaml,run.sh}`, `kairyu/bench/{cli,config}.py`,
  `tests/unit/test_qwen_fugu_example.py`. Verified end-to-end against a mock gateway
  (all 11 slots, progress, vision skips, subset banner, accuracy report); the GPU host run
  is still pending.

### 2026-07-25 — [amendment] MRCRv2 scores Fugu's 8-needle / 128K slice
- What: The MRCR adapter averaged all 2,400 published rows, which mix 2-, 4- and
  8-needle items across context lengths up to 1M tokens. It now selects only
  `n_needles == 8` rows whose estimated prompt tokens are <= 131,072 — Fugu's reported
  conditions — preferring the dataset's own `n_chars` over the chars/4 heuristic,
  recording the per-row estimate, printing how many rows each filter excluded, and
  failing closed if the slice is empty. The `n_needles` field was already carried in the
  payload but never used.
- Why: The scoreboard cell claimed comparability with Fugu's MRCRv2 number while
  measuring an easier, shorter population; needle count and context length are the two
  variables this benchmark exists to vary.
- Refs: `kairyu/bench/adapters/mrcr.py`, `kairyu/bench/fixtures/mrcr-v2.jsonl`,
  `tests/bench/test_bench_mcq_adapters.py`, `docs/benchmarks.md`.

### 2026-07-25 — [amendment] Agentic bench harnesses use real flags and Fugu's turn/trial limits
- What: All three agentic wrappers built commands the installed harnesses reject, so those
  cells could only report `failed`. `harbor run` has no `--output-dir` (that belongs to
  `harbor jobs download`) and selects datasets as `name@version`, not
  `terminal-bench/terminal-bench-2-1`; the τ harness has no `--output` (results are
  addressed by `--save-to <name>` under its `TAU2_DATA_DIR`) and has no `banking` domain
  (only `banking_knowledge`). Fixed all of those, then pinned Fugu's conditions:
  `agent.step_limit=1000` for SWE-Bench Pro (harness default 250, restating
  `swebench.yaml` because `-c` discards the default config), `--ak max_turns=500` for
  terminus-2, and `--retrieval-config alltools` plus `--user-llm-args` carrying the
  judge's reasoning effort for τ³. New `--attempts` (default 1) drives Harbor `-k` and τ
  `--num-trials`, with annotations recording that Fugu reports τ³ as pass@4 and the
  Terminal-Bench leaderboard wants ≥5.
- Why: A `failed` cell was indistinguishable from a genuinely bad score, and the
  published turn budgets are the difference between a truncated trace and the reported
  condition.
- Refs: `kairyu/bench/adapters/{swebench_pro,terminal_bench,tau_bench}.py`,
  `kairyu/bench/{types,cli,config,runner}.py`, `kairyu/bench/adapters/base.py`,
  `tests/bench/{test_bench_agentic_conditions,test_bench_tau,test_bench_agentic}.py`,
  `docs/benchmarks.md`.

### 2026-07-25 — [amendment] Bench targets own a sampling policy (reasoning effort)
- What: `BenchTarget` and `JudgeConfig` now share a `SamplingOptions` base carrying
  `reasoning_effort`, `top_p`, `seed`, and `extra_body_json`; `call_chat` merges those
  into every request body, and the judge client forwards its own. New flags:
  `--reasoning-effort`, `--top-p`, `--sampling-seed`, `--extra-body`,
  `--judge-reasoning-effort`, `--judge-extra-body`, all of which also override
  YAML-declared targets. `extra_body_json` is validated at load time (JSON object; may
  not override `model`/`messages`/`stream`) and stays a string so the frozen models
  remain hashable. Every field is part of the run fingerprint.
- Why: Fugu reports its table at each model's maximum reasoning effort and ran the τ³
  user simulator at `low`. The request body previously carried only
  model/messages/temperature/stream/max_tokens, so neither condition could reach the
  wire — and for Qwen3 the thinking toggle lives in `chat_template_kwargs`, which needs
  the same escape hatch.
- Refs: `kairyu/bench/{types,config,cli,judge}.py`, `kairyu/bench/adapters/base.py`,
  `tests/bench/test_bench_sampling.py`, `docs/benchmarks.md`, `examples/bench_fugu.yaml`.

### 2026-07-25 — [amendment] Bench dataset revisions are pinned in one registry
- What: Added `kairyu/bench/pins.py` mapping adapter name → (dataset id, commit sha) for
  HLE, GPQA Diamond, CharXiv, SciCode, MRCRv2, LongBench-v2, and LiveCodeBench Pro;
  `all_adapters()` fills each adapter's unset `hf_revision` from it per instance. A pin
  applies only when the recorded dataset id still matches, and an adapter that declares
  its own revision keeps it. Agentic slots stay unpinned because their harnesses fetch
  their own data with no revision knob — recorded as a limitation in `docs/benchmarks.md`.
- Why: `openai/mrcr` was corrected in December 2025 and HLE's item count has shifted since
  release, so unpinned cells were comparable to neither Fugu's numbers nor to earlier
  kairyu runs. Revisions are already part of the run fingerprint, so repinning now refuses
  resume instead of reinterpreting stored evidence.
- Refs: `kairyu/bench/pins.py`, `kairyu/bench/adapters/__init__.py`,
  `tests/bench/{test_bench_pins,test_bench_runner}.py`, `docs/benchmarks.md`.

### 2026-07-25 — [amendment] LiveCodeBench and LiveCodeBench Pro datasets are actually reachable
- What: Both code slots could never load their data, so their scoreboard cells were
  permanently blank. LiveCodeBench passed the *config name* `release_v6` as a git
  revision to a repo that has only `main` and no tags, and its script-loader path also
  needs `trust_remote_code` (removed in `datasets` 4.x); it now reads the
  `test.jsonl`…`test6.jsonl` shards directly at a pinned commit and fails closed unless
  `release_v6` yields exactly 1,055 problems. LiveCodeBench Pro asked for a `train`
  split that does not exist and expected a tabular testcase repo; it now pins Fugu's
  2025 Q2 slice (`quater_2025_4_6`, 167 problems), is declared `gated` (it needs
  HF_TOKEN), and joins each `problem_id` to a `<problem_id>.zip` of
  `testdata/<n>.in`/`.ans`. Stdin grading became per-line whitespace-normalized so a
  correct solution emitting trailing spaces or CRLF is no longer a false negative.
- Why: A permanently blank cell is worse than a documented approximation — the suite
  claimed to cover 11 Fugu slots while silently covering 9. The Pro archives ship a
  per-problem testlib `checker.cpp` that kairyu does not compile, so that cell is now
  annotated as a LOWER BOUND rather than presented as the official Accepted rate.
- Refs: `kairyu/bench/hub.py` (`load_jsonl_files`, `revision` on `download_file`),
  `kairyu/bench/adapters/{livecodebench,livecodebench_pro}.py`,
  `tests/bench/test_bench_lcb_datasets.py`, `docs/benchmarks.md`.

### 2026-07-25 — [amendment] SciCode runs sequentially and can reach its golden data
- What: The SciCode slot could not produce a meaningful number. `SciCode1/SciCode` ships
  no reference code at all (every `ground_truth_code` and `general_solution` is null), so
  the adapter's "gold prior-step code" was always the empty string and any sub-step
  calling an earlier step's helper could only raise `NameError`. Separately, 288 of the
  291 test-split sub-steps compare against `target` golden data from `test_data.h5`,
  which the HF export does not contain, so those items were all `unjudged` — leaving
  three scoreable sub-steps. Sub-steps now run SEQUENTIALLY per problem with the model's
  own earlier code carried into both the prompt and the executed program; `--limit`
  selects whole problems; the golden data is fetched from upstream first and otherwise
  from a pinned public mirror, accepted only on HDF5 magic bytes; and prompts now include
  problem-level and step-level background (Fugu's with-background condition). Extracted
  `attempt_item()` in `adapters/base.py` so the sequential loop and the shared generative
  loop classify request failures identically.
- Why: 288 is exactly the denominator Fugu reports, and the sequential setting is
  SciCode's main one — evaluating steps in isolation without any prior implementation
  measures nothing the benchmark is about.
- Refs: `kairyu/bench/adapters/{scicode,base}.py`, `kairyu/bench/fixtures/scicode.jsonl`,
  `tests/bench/test_bench_scicode_sequential.py`, `docs/benchmarks.md`.

### 2026-07-25 — [progress] Live progress display for bench runs
- What: Added `kairyu/bench/progress.py` with three reporters behind one protocol —
  `TqdmProgress` (suite bar + per-pair item bar on a TTY), `LineProgress` (one
  self-contained, throttled line per event for logs), and `NullProgress` — selected by
  `make_reporter()` from TTY-ness and the new `--no-progress` flag. `RunContext.progress`
  defaults to silence, the shared generative loop advances it for every item including
  skips and failures, and the runner announces each benchmark×target (labelling agentic
  slots, which have no item count until their harness returns). `progress` is excluded
  from the run fingerprint. `tqdm` joins the `bench` extra and its absence degrades to
  lines instead of raising.
- Why: A full Fugu run is thousands of judged items over hours; with no output, a working
  run and a hung one looked identical, and a redrawing bar is unusable in
  `docker compose logs`.
- Refs: `kairyu/bench/{progress,runner,types,cli,config}.py`,
  `kairyu/bench/adapters/base.py`, `pyproject.toml`,
  `tests/bench/{test_bench_progress,test_bench_runner}.py`, `docs/benchmarks.md`.

### 2026-07-25 — [progress] Accuracy report compares a run against the published Fugu scores
- What: Added `kairyu/bench/reference.py` (the sakana.ai/fugu-release table and
  per-benchmark figure as committed constants, with source URLs, both asset paths, the
  2026-07-25 retrieval date, the page's own footnotes, and the HLE text-only variant) and
  `kairyu/bench/compare.py`, which renders measured-vs-published with `Δ` against
  published `Fugu`. Every run now writes and prints `comparison.md`/`comparison.json`
  alongside the scoreboard; `kairyu bench report` rebuilds it (`--no-comparison` to skip).
- Why: G6's frontier scoreboard needs the published numbers next to ours, but a bare delta
  invites an apples-to-apples reading the data does not support. The report therefore
  refuses to print 0 for an unmeasured cell, marks partial denominators, withholds the
  delta entirely for the substituted Long Context Reasoning row, and repeats the release
  page's own statement that all non-Fugu scores are provider-reported. The page renders
  its table as a PNG, so the values are transcribed, not fetched — recorded as such.
- Refs: `kairyu/bench/{reference,compare,store,runner,cli}.py`,
  `tests/bench/test_bench_compare.py`, `docs/benchmarks.md`.

### 2026-07-23 — [progress] Versioned structured orchestration trace for evaluation tooling
- What: Added additive, opt-in `kairyu_trace_v2` on unary orchestrated chat responses while
  preserving `kairyu_trace`. Direct, Conductor, and MoA paths emit a common versioned envelope
  with sequenced route/role events, status/attempt, timestamps, resolved worker/engine/model,
  backend token usage, budget deltas, and exception class only. The Pydantic response schema
  declares both structured route and trace for capability discovery while unrequested extension
  fields remain absent. MoA preserves proposal/synthesizer resolution even when both fall back
  to one engine. A contract document pins compatibility, timing, privacy, and extension rules.
- Why: The verification app needs to overlay the actual execution path on the configured role
  DAG and diagnose latency/refinement/budget behavior without parsing strings or collecting
  prompts and generated text.
- Refs: `docs/design/observability-trace-contract.md`,
  `kairyu/orchestration/{trace,conductor,orchestrator}.py`,
  `kairyu/entrypoints/server/{protocol,app}.py`.

### 2026-07-23 — [progress] All-GPU Qwen3-32B TP serve and benchmark workflow implemented
- What: Replaced the single-GPU Qwen3-32B example with a Compose workflow that
  reserves every visible NVIDIA GPU, derives `tensor_parallel_size` at startup,
  and rejects GPU counts that cannot evenly shard Qwen3-32B. Added a one-command
  start-and-benchmark script targeting host port 8001; it records timestamped
  JSON and regenerates a Markdown summary of all saved runs.
- Refs: `examples/qwen3-32b-multi-gpu/`.

### 2026-07-23 — [progress] GPU gate-integrity issues #102–#104 fixed and hardware-verified
- What: Pinned the CPU-only synthetic model fixture to torch attention so the
  CUDA-visible default suite is green (1409 passed, 12 skipped, 92% coverage);
  removed the no-op `KAIRYU_DIST_BACKEND=nccl` rerun from the multi-GPU gate
  while retaining the explicit 2-GPU NCCL EP test; and added a real §0 recorder
  that writes the dated `EnvRecord` with driver, CUDA, library versions, GPU
  topology, MIG/vBIOS inventory, and explicit null/unmeasured P2P fields. The
  explicit NCCL test and real SM120 environment-record generation both pass.
- Why: Gate 0 must be host-independent, and hardware gates must produce the
  evidence they claim rather than returning a false green result.
- Refs: PR #105; Issues #102, #103, #104; `scripts/gpu_gates/{00_env,06_multigpu,record_env.py}`;
  `tests/unit/{test_model_loader_backend,test_gpu_gate_regressions}.py`.

### 2026-07-23 — [progress] First RTX PRO 6000 GPU validation exposed three gate-integrity defects
- What: Provisioned the locked dev/GPU/engine environment on an 8× RTX PRO 6000
  Blackwell Server Edition host (driver 595.71.05, CUDA 13.2). Ruff passed; the
  CUDA-hidden full suite passed with 1407 tests, 12 skips, and 92% coverage; the
  single-GPU marked suite passed 10 tests; the explicit 2-GPU NCCL EP test
  passed; and the 8-test distributed suite passed on gloo. With CUDA visible,
  two ordinary model-loader/backend tests failed because the SM120 profile
  auto-selected FlashInfer for an unsupported tiny fixture. Audit also found
  that the §6 script’s `KAIRYU_DIST_BACKEND=nccl` rerun remains on gloo and the
  §0 script never writes its required environment JSON.
- Why: These failures make Gate 0 host-dependent and allow the environment and
  multi-GPU gate scripts to report success without producing the evidence they
  claim.
- Refs: Issues #102, #103, #104; `tests/unit/test_model_loader_backend.py`;
  `kairyu/engine/kairyu_backend.py`; `scripts/gpu_gates/{00_env,06_multigpu}.sh`;
  `tests/{dist,gpu}/`; `bench/results/env-2026-07-23.json`.

### 2026-07-19 — [amendment] Routing verification is non-mutating and deployment-visible
- What: Added built-in Router `preview`/safe descriptor contracts, including RNG-cloned bandit
  preview; exposed rendered-prompt `/v1/route`, authenticated `/routing`, and opt-in structured
  actual decisions. GPU compose now mounts a routing spec consumed by inventory-driven gateway
  orchestration, while the verification BFF/UI separates preview from actual distribution/history.
- Why: A direct `route()` dry-run advanced bandit RNG, raw-query preview diverged from templated
  chat execution, and deployed GPU gateways did not serve any orchestrator even though local mock did.
- Refs: M1 D3/D6, M4 router learning; `kairyu/orchestration/`,
  `kairyu/entrypoints/server/app.py`, `deploy/compose/`.

### 2026-07-16 — [progress] GET /backends introspection endpoint (m13)
- What: Added an open `GET /backends` reporting the resolved attention backend
  (torch/flashinfer), library versions, and the per-engine backend map. New pure
  `select_backend_name(profile)` names the backend without importing flashinfer
  (`select_backend` delegates to it). In the gateway+replica topology the gateway
  runs no local attention, so for `ReplicaPool` engines it aggregates one replica's
  `/backends` (L1 `OpenAICompatBackend.fetch_backends` → L2 `ReplicaPool.probe_backends`,
  cached + best-effort) and surfaces the replica's kernel under `via_replica`; `role`
  distinguishes gateway vs engine-host.
- Why: tooling (the verify web app) must show "what am I running on" without
  deep-walking private engine internals. The aggregation was needed because the
  proxy gateway — the process callers actually reach — otherwise only ever reports
  its own CPU/torch, hiding the replicas' flashinfer.
- Refs: engine/core/attention/selector.py, entrypoints/server/health.py,
  entrypoints/server/middleware.py (`_OPEN_PATHS`), engine/openai_backend.py,
  orchestration/replica.py, tests/server/test_backends.py; PR #100.

### 2026-07-16 — [progress] FlashInfer AOT covers head_dim 128; multi-model serving is inventory-driven
- What: Extended the FlashInfer AOT jit-cache build in `Dockerfile.cuda` from head_dim 64 only to
  `fa2_head_dim=[(64,64),(128,128)]`, so head_dim-128 models (Llama-3.x) run on the SM120 FlashInfer
  kernels alongside head_dim-64 (Qwen2.5). Previously a 128 model hit a request-time JIT that fails on the
  slim `-runtime` image (no nvcc; `ninja` exit 127). Kept the stock `gpu-replica.yaml`/`gateway-gpu.yaml`
  single-model and refreshed `gpu-replica.multimodel.example.yaml`; the deployment (kairyu-iac) renders the
  multi-model compose from its `kairyu_models` inventory, so no deployment-specific model set is hardcoded here.
- Why: Serve several eval models (Qwen2.5-0.5B + Llama-3.2-3B + Llama-3.1-8B) on one RTX PRO 6000 (SM120)
  replica on FlashInfer, without baking a model list into the OSS compose.
- Refs: PR #99; `Dockerfile.cuda`, `deploy/compose/gpu-replica.multimodel.example.yaml`; kairyu-iac develop
  (compose templating from `kairyu_models`). Adding a model with a new head_dim needs the `fa2_head_dim`
  scope extended + an image rebuild (the one dependency that stays in this repo).

### 2026-07-14 — [amendment] Usage ledger shutdown and recovery are app-owned (m11 D3/A7)
- What: `create_app` now wraps its optional caller lifespan and flushes/closes the
  app-created usage ledger after all inner worker/backend cleanup, even when that cleanup
  raises. Ledger scans validate records independently, ignore whitespace, warn on a
  non-newline malformed tail, error with line numbers for complete malformed records,
  retain valid totals, and expose the latest malformed-record count.
- Why: Issue #90 showed that the persistent O_APPEND handle was never closed and that one
  truncated or schema-skewed JSONL line made the entire administrator usage endpoint fail.
- Refs: Issue #90; m11 D3/A7; `kairyu/entrypoints/server/{app,tenancy}.py`;
  `kairyu/deploy/builder.py`; `tests/server/{test_m11_product,test_serve_builder}.py`.

### 2026-07-14 — [amendment] Embeddings use explicit served model IDs (m11 D4)
- What: `create_app` and `DeploymentSpec` now accept model-ID-to-embedding-backend
  registries, reject IDs that collide with engines, pools, or orchestrators, and include
  configured embedding IDs in `/v1/models`. The embeddings route resolves before validation
  or execution, shares the 404 `model_not_found` response, routes multiple backends, and
  records response, metric, and ledger identity only from the resolved key while charging
  the limiter only after resolution.
- Why: Issue #89 showed that the anonymous global backend accepted and echoed arbitrary IDs,
  omitted its model from discovery, and admitted attacker-controlled metric and metering
  identities despite executing the same backend.
- Refs: Issue #89; m11 D4/A12; `kairyu/entrypoints/server/{app,extra_routes}.py`;
  `kairyu/deploy/{spec,builder}.py`; `tests/server/test_embeddings_models.py`.

### 2026-07-14 — [amendment] Required tool choice is enforced per response choice (m1 D6)
- What: required and named tool choice now succeeds only when every returned choice retains
  a permitted tool call after per-choice filtering. Mixed and empty results use the existing
  controlled 502 before buffered SSE emission, without regeneration, and their consumed
  generation is recorded exactly once.
- Why: Issue #88 showed that the existential satisfaction check accepted multi-choice
  responses when only one choice complied, violating the request contract for the remaining
  choices.
- Refs: Issue #88; m1 D6; `kairyu/entrypoints/server/chat_service.py`;
  `tests/server/{test_openai_api,test_m11_product}.py`.

### 2026-07-14 — [amendment] OpenAI-compatible streams preserve empty choices (m1 D1)
- What: streamed choice state is now initialized whenever an upstream index is observed,
  independently of text content, and partial/final completions are built from the union of
  text and finish indexes with empty defaults. Empty single choices and mixed `n > 1`
  results retain their indexes and finish reasons; streams with no choices still fail.
- Why: Issue #87 showed that truthy-content-only tracking converted valid immediate-EOS or
  content-filtered empty responses into upstream errors and silently dropped empty siblings
  from multi-choice results, contradicting non-streaming behavior.
- Refs: Issue #87; m1 D1; `kairyu/engine/openai_backend.py`;
  `tests/unit/test_openai_backend.py`.

### 2026-07-14 — [amendment] Batch and HTTP share the chat request boundary (m7 D7)
- What: batch JSONL rows now require a frozen envelope with non-blank, per-job-unique
  `custom_id`, `POST`, the owning job endpoint, and an object body. A new transport-neutral
  chat service owns tool, stream/logprob, response-format, image, model, `supports_n`,
  sampling, and backend preflight validation plus buffered dispatch and tool-choice
  satisfaction for both regular HTTP and batch. Controlled failures retain the same
  message/type/code without dispatch; unexpected backend errors expose only the exception
  class in both transports while their full tracebacks remain server-side.
- Why: Issue #86 showed that batch ignored its method and URL, accepted missing or duplicate
  IDs, skipped the public request checks, executed invalid work, and persisted arbitrary
  backend exception strings containing internal topology or secrets.
- Refs: Issue #86; m7 D7; `kairyu/batch/{envelope,worker}.py`;
  `kairyu/entrypoints/server/{chat_service,errors,app}.py`;
  `tests/{unit/test_batch_envelope,server/test_chat_parity,server/test_batches}.py`.

### 2026-07-14 — [amendment] Batch uploads use bounded storage transactions (m10a D3/A8)
- What: `BatchStoreProtocol` expands from ten to eleven methods with
  `save_file_streaming`. `/v1/files` now supplies fixed 1 MiB chunks, enforces the existing
  512 MiB limit incrementally before writing an over-limit chunk, and publishes content plus
  owner metadata only after the stream completes. Rejection, iterator failure, cancellation,
  close failure, and metadata failure remove partial artifacts.
- Why: Issue #85 showed that the route read up to the complete 512 MiB allowance into memory
  per request on top of Starlette's multipart spool, so concurrent accepted uploads could
  exhaust the gateway even though each individual request obeyed the size cap.
- Refs: Issue #85; m10a D3/A8; `kairyu/batch/store.py`;
  `kairyu/entrypoints/server/batch_routes.py`; `tests/unit/test_batch_store_streaming.py`;
  `tests/server/test_batches.py`.

### 2026-07-14 — [amendment] Async abort owns active request lifecycles
- What: `AsyncLLMEngine` replaced persistent abort markers with a registry of active
  request-local events. Generation now races backend progress against abort, rejects only
  concurrently active duplicate IDs, and centralizes deregistration, pending-task
  cancellation, and backend iterator closure across completion, abort, and consumer close.
- Why: unknown aborts previously retained attacker-controlled IDs and suppressed a future
  request reusing that ID, while an active stream blocked on its backend could not observe
  abort until another partial arrived.
- Refs: Issue #84; `kairyu/entrypoints/async_engine.py`;
  `tests/{compat/test_async_engine_compat,unit/test_async_engine_abort}.py`.

### 2026-07-14 — [progress] FlashInfer SM120 (Blackwell) attention enabled: GPU device placement + AOT image
- What: The single-process GPU serve path had no device placement — `build_engine_loop` loaded the
  model and `PagedKVPool` in fp32 on CPU, so a "GPU" deployment ran inference on CPU via the
  device-agnostic torch backend, and FlashInfer (the sm_120 default) failed with
  `KeyError: torch.float32` (its prefill `plan()` has no fp32 kernel). Fixes:
  - `build_engine_loop`: probe-driven device/dtype (cuda → bf16 on-device, cpu → fp32 on host;
    guarded so every CPU path is byte-identical). `load_model(dtype=…)` + `model.to(device)` +
    `PagedKVPool.for_cache(dtype, device)`.
  - `PagedModelRunner` / `PagedKVPool`: device-correct input and index tensors.
  - `Sampler`: sample on CPU (seeded RNG + penalties/enforcer) so the m8-D2 determinism pins hold
    while logits arrive on cuda.
  - `FlashInferBackend.attend`: pass the `[1,H,D]` decode query (0.6.x `decode.run` rejects a 2D
    `[H,D]` slice); the batched `attend_batched` path already passes 3D.
  - `app.py`: backend-exception handlers log the full traceback (client still gets only the class name).
  - `Dockerfile.cuda` → multi-stage AOT: a CUDA 13.0 `-devel` build stage compiles
    `flashinfer-jit-cache` (`FLASHINFER_CUDA_ARCH_LIST=12.0f`, bf16/head_dim-64/FA2); slim
    `13.0.1-runtime`, no runtime nvcc, no first-request JIT.
- Why: run the FlashInfer paged-attention kernels on RTX PRO 6000 Blackwell (sm_120). The torch
  backend was a correct-but-slow interim and — as found — CPU-bound. AOT removes the ~20s
  first-request JIT so a cold replica is not ejected by the gateway readiness probe.
- Refs: docs/design/flashinfer-sm120-aot.md; kairyu/engine/kairyu_backend.py,
  engine/core/{model_runner,kv_pool,sampler}.py, engine/core/attention/flashinfer_gpu.py,
  entrypoints/server/app.py, Dockerfile.cuda.

### 2026-07-14 — [progress] Responses, embeddings, and batches complete usage accounting
- What: successful `/v1/responses` and `/v1/embeddings` calls now record one usage
  event for the authenticated tenant using the same counts returned on the wire.
  The bounded batch worker receives optional ledger/limiter sinks from the deployment
  builder and records each successful backend line for its persisted owner immediately
  after execution, independently of later spool, finalization, or cancellation outcomes.
  Backend usage wins when present; Responses and batch fall back to the shared derived
  approximation, while embeddings charge input tokens with zero output tokens.
- Why: these three surfaces bypassed the shared tenant recorder, leaving successful work
  unaccounted; delaying batch accounting until file publication would also lose already
  executed calls whenever transactional output was rolled back.
- Refs: Issue #45 Task 3; `kairyu/{batch/worker,deploy/builder}.py`;
  `kairyu/entrypoints/server/{extra_routes,tenancy}.py`;
  `tests/server/{test_batches,test_m11_product,test_serve_builder}.py`.

### 2026-07-14 — [progress] Stream metering survives completion, failure, and disconnect
- What: engine chat, orchestrated chat, and legacy completions streams now share one
  idempotent finalization owner. Each dispatched stream records exactly one tenant usage
  event on normal completion, missing backend usage, partial client disconnect, or a
  partial upstream error; backend counts win when present and otherwise the existing wire
  approximation is derived from the rendered prompt and latest cumulative completions.
- Why: accounting after the normal stream loop skipped orchestrated/completions streams
  entirely and bypassed billing whenever a client disconnected or an upstream failed
  after doing partial backend work.
- Refs: Issue #45 Task 2; `kairyu/entrypoints/server/{app,metering}.py`;
  `tests/server/{test_openai_api,test_m11_product}.py`.

### 2026-07-13 — [amendment] Batch execution is bounded and failure-terminal (m7 D7)
- What: batch execution now uses one streaming producer, a bounded input queue, and a
  fixed consumer pool instead of whole-file materialization and one task per line.
  Output/error rows spool incrementally; unexpected line, append, or finalization
  failures roll back every partial file and persist a controlled `failed` state, while
  explicit cancellation wins and task cancellation still propagates.
- Why: Issue #44 demonstrated that valid large uploads could multiply into unbounded
  memory/tasks and that post-admission storage errors could leave jobs `in_progress` or
  expose half-published result files.
- Refs: Issue #44 Tasks 2–3; `kairyu/batch/worker.py`; `kairyu/batch/store.py`;
  `tests/server/test_batches.py`.

### 2026-07-13 — [amendment] Batch storage adds streaming transaction seams (m10a D3/A8)
- What: `BatchStoreProtocol` expands from eight to ten methods with owner-scoped
  `iter_file_lines` and `create_jsonl_writer`. The store now supports lazy binary-line
  input and a lazy JSONL transaction that writes one flushed line at a time, publishes
  owner-scoped metadata only on commit, and removes partial data on abort.
- Why: the existing batch worker materializes the full accepted upload and all output
  rows, so Issue #44 needs bounded storage primitives before Task 2 can replace that
  worker path without exposing partial result files or weakening tenant isolation.
- Refs: Issue #44 Task 1; m10a D3/A8; `kairyu/batch/store.py`;
  `tests/unit/test_batch_store_tenancy.py`.

### 2026-07-13 — [amendment] Deployment auth shares one preflight key snapshot
- What: the deployment builder now resolves both data-plane and administrator key
  sets before constructing owned backends, then passes those immutable snapshots
  through `create_app`; tenant validation and authentication therefore consume the
  same data-plane snapshot. Direct programmatic `create_app` calls retain their
  existing settings-based resolution behavior.
- Why: the initial Issue #46 Task 3 integration re-read data-plane keys during app
  construction and deferred administrator-key resolution until after backend
  ownership, allowing environment changes to desynchronize tenant mapping from
  authentication or to fail after resources had been created.
- Refs: Issue #46 Task 3 review; `af6e2fa`; `kairyu/deploy/builder.py`;
  `kairyu/entrypoints/server/app.py`; `tests/server/test_serve_builder.py`.

### 2026-07-13 — [progress] Deployment YAML tenants wired into runtime isolation
- What: `build_app_from_spec` now preflights the optional deployment `tenants:`
  section before constructing owned backends, converts its limit profiles into
  runtime `TenantLimits`, and passes one validated `TenantConfig` into the server.
  Two-key end-to-end coverage pins independent request buckets, role-auth scoped
  `/admin/usage`, and tenant-named ledger records while tenant-less deployments
  retain their legacy app state.
- Why: The typed schema and mapping validation from Issue #46 Tasks 1–2 were not
  yet connected to `kairyu serve`, so deployment files could describe tenants
  without activating runtime isolation or per-tenant accounting.
- Refs: Issue #46 Task 3; `kairyu/deploy/builder.py`;
  `tests/server/test_serve_builder.py`.

### 2026-07-13 — [design] D4 defines atomic orchestration budget admission
- What: D4 now requires every Conductor generation to reserve its step synchronously
  before dispatch, gives result-priced work one exclusive unknown-cost admission slot,
  and releases complete reservations on failure or cancellation. MoA reserves all
  proposal plus synthesis steps as one operation. An admitted generation's eventual
  actual cost remains fully accounted and queryable even when it crosses the cap.
- Why: Parallel DAG roots and MoA previously admitted work from stale completed-only
  state, multiplying strict step/cost limits. Pre-dispatch reservations close that race
  while preserving the truthful result-priced behavior: an unknowable actual charge is
  never clamped or hidden after admission.
- Refs: issue #43; D4 in `docs/design/m1-orchestration-and-interface.md`;
  `kairyu/orchestration/{budget,conductor,orchestrator}.py`

### 2026-07-13 — [amendment] Remote readiness requires generation-safe probes
- What: Remote replicas with declared readiness URLs now start unknown and stay out of placement and `/readyz` until a successful `/readyz` probe. The serve prober runs immediately at startup, validates unknown/ejected replicas with bounded concurrency, isolates failures, and binds results to the entry generation; URL-less local/programmatic pools remain trusted.
- Why: Treating a newly constructed remote backend as healthy allowed readiness and request placement before any live endpoint check, while serial or ID-only recovery could multiply startup latency or validate a replacement from a stale response.
- Refs: issue #52; `docs/design/m7-productionization.md` D4; `docs/design/m10-fleet-cpu.md` D1/D2/A15; `kairyu/orchestration/replica.py`, `kairyu/deploy/prober.py`, `kairyu/deploy/builder.py`

### 2026-07-13 — [amendment] Elastic ownership follows same-ID entry generations
- What: `ReplicaPool` now gives every backend entry an opaque generation token, and `PoolReconciler` binds applied identities and drain leases to that generation. An external remove/re-add of the same ID discards old tracking; desired absence acquires a fresh lease on the new entry, while desired presence baselines it without factory, replacement, or shutdown side effects. Fresh-entry manual drains remain authoritative.
- Why: Comparing only replica ID sets missed a complete entry replacement between reconciliation ticks, so an old lease could suppress draining of the fresh entry and prevent removal from converging.
- Refs: issue #41; `docs/design/m10-fleet-cpu.md` D1/D2 and A14; `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`

### 2026-07-13 — [amendment] Manual drains follow same-ID backend replacement
- What: A successful identity replacement now carries the manual drain owner from the old pool entry to the new backend entry while discarding the reconciler lease. The pool exposes a manual-only drain query; the replacement backend starts with fresh health and outstanding state but remains non-eligible until manual undrain.
- Why: Manual ownership was stored on the backend entry, so deleting the old entry and adding the replacement silently made an operator-drained logical replica eligible.
- Refs: issue #41; `docs/design/m10-fleet-cpu.md` D1/D2 and A14; `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`

### 2026-07-13 — [amendment] Elastic drains preserve overlapping owners
- What: Corrected the preceding reversible-drain amendment by splitting each pool entry's manual drain owner from opaque drain leases. Reconciliation now records and releases only its own lease, so a manual drain asserted before or after reconciliation remains active, and manual undrain cannot cancel reconciliation work.
- Why: ID-only ownership over one boolean could not represent overlapping owners and allowed a desired-state revert or retry factory failure to undrain a replica that an operator had drained afterward.
- Refs: issues #41 and #42; `docs/design/m10-fleet-cpu.md` D1/D2 and A14; `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`

### 2026-07-13 — [amendment] Elastic reconciliation owns reversible drains
- What: Discovery and applied state now carry complete typed replica identities; same-ID changes replace backends construct-before-drain with async ownership cleanup. The reconciler separately tracks drains it initiated and cancels them when replacement/removal intent reverts or a retry factory fails, without overriding manual drains.
- Why: Address/model/auth changes were previously invisible, while an in-flight replacement or removal could leave the old replica permanently non-eligible after desired state returned to the applied identity.
- Refs: issues #41 and #42; `docs/design/m10-fleet-cpu.md` D1/D2, A6, A14; `kairyu/deploy/registry.py`, `kairyu/orchestration/replica.py`

### 2026-07-13 — [amendment] Open WebUI Compose demo + Kairyu-only CI smoke (m11 D7)
- What: The checked-in WebUI topology now mounts a standalone valid
  `deploy/compose/config.yaml` serving keyless mock model `default`; all literal
  Compose binds and mounted DeploymentSpecs are validated before startup. The new
  `scripts/webui_smoke.sh` also pins the rendered internal WebUI endpoint, starts only
  Kairyu, and gates bounded readiness, exact `/v1/models`, and one non-streaming
  completion after the existing default Compose drill.
- Why: m11 D7 previously claimed only that the container config rendered while the
  checked-in bind target did not exist, so a clean checkout could not start the demo.
  Keeping the smoke Kairyu-only proves the broken startup and API contract without
  pulling or browser-testing the large mutable third-party Open WebUI image.
- Refs: m11 D7; `deploy/compose/{docker-compose.webui.yaml,config.yaml}`;
  `scripts/{validate_compose_binds.py,webui_smoke.sh}`; `.github/workflows/ci.yml`;
  `tests/unit/test_compose_configs.py`.

### 2026-07-13 — [amendment] Preflight the production benchmark model
- What: Amended m19 D3 so gate 09 checks `/v1/models` after `readyz` and before
  `serving_bench.py`, requires the requested model ID by exact equality, and uses
  the same `KAIRYU_BENCH_MODEL` value for both steps. Added safe failure handling
  for absent IDs, malformed responses, and non-2xx responses, plus source and
  default/override dry-run pins for ordering and propagation.
- Why: A healthy gateway can pass `readyz` while not serving the model selected
  for the production benchmark, which otherwise makes the benchmark fail late or
  exercise the wrong deployment contract.
- Refs: m19 D3; `scripts/gpu_gates/{09_production.sh,check_served_model.py}`;
  `tests/verification/test_gpu_gates_scripts.py`.

### 2026-07-13 — [amendment] Blackwell Helm profile pins the supported attention backend
- What: Added a strict Helm `attentionBackend` seam that renders
  `KAIRYU_ATTENTION_BACKEND`; CPU defaults omit it, the checked-in
  `pcie-gddr`/SM120 overlay pins `torch`, and operators can select `flashinfer`
  on supported hardware. Extended static and render contracts plus chart docs.
- Why: The automatic SM120 `fa2` tier selects FlashInfer, but the current build
  has no Blackwell kernels. Without an environment seam, the documented overlay
  could render successfully yet fail when starting the real backend.
- Refs: Issue #49 final independent review; `deploy/helm/kairyu/{values.yaml,
  values-gpu.yaml,values.schema.json,templates/deployment.yaml,README.md}`;
  `docs/design/m19-deploy-packaging.md` D2 clarification.

### 2026-07-13 — [amendment] GPU Helm overlay becomes a mandatory CI render gate
- What: `scripts/kind_smoke.sh` now runs fail-fast default/GPU `helm lint` and
  `helm template` gates before cluster creation, with a `--helm-check` mode used
  by an explicit CI schema/GPU-template step. The script remains the single
  command source; CI does not duplicate the four Helm invocations. Appended an
  M19 D2/D3 amendment recording the placement/runtime/storage/real-backend gate
  and its template-only, no-GPU execution boundary.
- Why: The GPU overlay was statically covered but not a mandatory CI input, so
  schema or rendering regressions could merge while the CPU kind smoke remained
  green. Fail-fast rendering makes both chart profiles release-gating without
  pretending ordinary CI can run a GPU workload.
- Refs: Issue #49 Task 3; `scripts/kind_smoke.sh`, `.github/workflows/ci.yml`,
  `tests/unit/test_fleet_elastic.py`, `docs/design/m19-deploy-packaging.md` D2/D3
  amendment.

### 2026-07-13 — [progress] Helm GPU overlay wires real model storage and engine
- What: Added strict chart values schema and a conditional, read-only model volume
  backed by exactly one absolute host path or existing PVC. The checked-in GPU
  values request one NVIDIA GPU, preserve runtime/node placement, mount `/models`,
  and replace the mock DeploymentSpec with backend `kairyu` at
  `/models/checkpoint`; CPU defaults keep model storage disabled. Added semantic
  render/schema regressions and operator documentation for hostPath/PVC use.
- Refs: Issue #49 Task 2; `deploy/helm/kairyu/values.yaml`,
  `values-gpu.yaml`, `values.schema.json`, `README.md`, `templates/deployment.yaml`,
  `tests/unit/test_fleet_elastic.py`. Helm-backed render/lint execution remains
  pending on a Helm-enabled host; local pure/static gates pass.

### 2026-07-13 — [progress] Backend ownership closes across replica and app lifecycles
- What: Replica removal is now an async ownership boundary that closes the removed backend exactly once. Shared shutdown aggregation attempts every unique backend, and orchestrator/application lifespan teardown cascades through separately owned workers even when another shutdown fails.
- Why: Removed/replaced replicas and DSL-built orchestrators leaked clients and worker tasks; one shutdown exception also skipped every later resource.
- Refs: issue #42; `kairyu/engine/backend.py`, `kairyu/orchestration/{replica,orchestrator}.py`, `kairyu/deploy/{registry,builder}.py`

### 2026-07-09 — [progress] Single-node GPU compose: dedicated gateway config + attention-backend env
- What: `docker-compose.gpu.yaml` now mounts a new `deploy/compose/gateway-gpu.yaml`
  (single `replica` upstream, forwards `model: default`) instead of the shared
  `gateway.yaml`, and passes `KAIRYU_ATTENTION_BACKEND` through to the replica
  (empty → auto-select; set `torch` to bypass FlashInfer). `gateway.yaml` is left
  in its CPU-smoke form (three `replica-1/2/3` upstreams, `model: llama`).
- Why: `gateway.yaml` was mounted by BOTH `docker-compose.yaml` (CPU smoke: three
  `replica-N` services serving engine id `llama`) and `docker-compose.gpu.yaml`
  (single `replica` service serving engine id `default`). The two topologies have
  different service names and engine ids, so one file cannot serve both — pointing
  it at the single GPU replica breaks the CPU compose and the CI `compose_smoke.sh`
  drill. Splitting into `gateway-gpu.yaml` lets each topology stand alone. The
  attention env exists because FlashInfer has no Blackwell/sm_120 kernels yet, so a
  Blackwell/RTX PRO 6000 replica can pin `torch` (selector: `KAIRYU_ATTENTION_BACKEND`,
  honored on the single-process `model_path` engine path).
- Refs: `deploy/compose/{gateway-gpu.yaml,docker-compose.gpu.yaml,gateway.yaml}`,
  `kairyu/engine/core/attention/selector.py`, `scripts/compose_smoke.sh` (m19 D2).

### 2026-07-09 — [progress] GPU image base bumped to CUDA 12.8.1 / Ubuntu 24.04
- What: `Dockerfile.cuda` base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  → `nvidia/cuda:12.8.1-runtime-ubuntu24.04`. Ubuntu 24.04 ships `python3.12`
  in its default repos, so the existing `apt-get install python3.12` line now
  resolves natively; nothing else in the image changed.
- Why: The GPU execution host runs Ubuntu 24.04, so the deployment image should
  match the host OS. The CUDA 12.8 runtime also adds SM120 (Blackwell /
  RTX PRO 6000) support that the roadmap's PCIe fleet targets and that the 12.4
  runtime lacked.
- Refs: `Dockerfile.cuda`; supersedes the base image recorded in the
  2026-07-03 M19 deploy-packaging entry (m19 D1).

### 2026-07-05 — [progress] Real multi-process TP wired into `kairyu serve --tp N`
- What: `build_engine_loop(model_path=…, tensor_parallel_size>1)` no longer
  raises "not yet wired" — it spawns a `DistTPLauncher` group (rank 0 in the
  serve process, ranks 1.. as workers running `worker_step_loop`) and drives it
  through `DistTPModelRunner`. The loop carries a `.tp_launcher` handle that
  `KairyuBackend.shutdown()` calls to stop the workers and destroy the group.
  Added `load_generation_defaults` (public eos/stop loader for the sharded path).
- Why: M16's distributed TP was spawn-tested only in `tests/dist` and unreachable
  from the serve entrypoint — so real tensor-parallel models could not be
  deployed. Now `kairyu serve --tp 2` runs end to end.
- Refs: `kairyu/engine/kairyu_backend.py` (`_build_dist_tp_loop`),
  `kairyu/engine/core/worker.py` (`DistTPLauncher`, `_tp_worker_entry`),
  `kairyu/models/loader.py`, test
  `tests/dist/test_distributed.py::test_dist_tp_launcher_serve_path_matches_single_process`.

### 2026-07-04 — [design] Review remediation Phase 6: GPU-day seam changes (CPU design + C5 contract test)
- What: Captured the five GPU-day seam changes from the full-repo review in
  `docs/design/gpu-day-seams.md` (C5 CUDA-graph static buffers, C4 batched
  execution, E3 engine-loop unification, TP delta-broadcast + sampling ownership,
  KVTransport region ownership), and landed the **C5 contract test**: a faithful
  `SnapshotGraphBackend` that freezes page_tables/seq_lens at capture (as a real
  CUDA graph does), plus `test_graph_replay_reflects_current_page_tables`
  (`xfail(strict=True)`) that concretely proves `GraphStepExecutor` currently
  rebinds page tables as Python attributes a real graph never sees. The test
  flips to pass when the static-device-buffer fix lands.
- Why: These CPU-pinned abstractions silently break (C5) or make a perf gate
  unreachable (C4) when real kernels/NCCL/FlashInfer replace the CPU references;
  they must be designed + contract-tested on CPU before GPU time, but can only be
  fully validated on hardware. The full implementations are a scheduled GPU-day
  design milestone (land before the runbook perf gates), not a same-session edit.
- Refs: review report; `docs/design/gpu-day-seams.md`,
  `kairyu/engine/core/step_executor.py` (`SnapshotGraphBackend`),
  `tests/unit/test_step_executor.py`.

### 2026-07-04 — [progress] Review remediation Phase 8: packaging + doc accuracy
- What: Fixed the cross-cutting packaging/doc defects from the full-repo review.
  Added an **`[engine]` extra** (torch + xgrammar + tokenizers + safetensors) so
  real models run WITHOUT the dev group, and pointed **`Dockerfile.cuda`** at it
  (`--extra engine` replaces `--extra hf`) so the production GPU image ships
  xgrammar and can serve `response_format: json_schema` (was missing). Fixed the
  misplaced comment above the `otel` extra (it described the fleet transports).
  `build_engine_loop`'s TP>1 error/docstring now state the truth — the
  multi-process `DistTPModelRunner` exists (m16, tests/dist) but is not yet wired
  into the single-process serve path — instead of "arrives in M16". Refreshed
  `docs/gpu-runbook.md` §0/§1: corrected the stale "177 tests" count, the
  `--group gpu`/`uv sync --dev` command errors (now `--extra gpu`/`--group dev`),
  and the "replace KairyuBackend._tokenize / TorchPagedRunner" instructions that
  M8/M12/M13 already delivered, with a note that the seams exist and GPU-day is
  enabling/tuning them.
- Why: The GPU image couldn't serve structured outputs, there was no non-dev
  install path for real models, and the runbook (the artifact GPU day executes
  from) contradicted the codebase.
- Refs: review report; `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`,
  `kairyu/engine/kairyu_backend.py`, `docs/gpu-runbook.md`.
  **Deferred follow-up:** `kairyu validate` cross-artifact command, typed
  `GenerationRequest.prompt` (token-ids/multimodal), `deploy/spec.py`
  ServerSection compose-not-inherit, and the `kairyu/bench/` package boundary.

### 2026-07-04 — [progress] Review remediation Phase 7: host-path performance (safe subset)
- What: Fixed the provably-safe, output-preserving host-path hot spots from the
  full-repo review. **P5**: `prompt_chunks` re-hashed the whole prompt prefix per
  256-char chunk (O(L²) sha256 on the placement path, event-loop-blocking) and
  the pool called `overlap()` twice per replica; replaced with ONE streaming
  sha256 chain (byte-identical keys, proven equivalent over random trials) and a
  single `overlap()` per replica. **P-perf (completions)**: `/v1/completions`
  ran a prompt array serially (`await` per prompt = sum of latencies); now
  `asyncio.gather` runs them concurrently with order restored by index (response
  byte-unchanged).
- Why: Both are event-loop-blocking / latency costs on the request path that
  survive the GPU swap; both are output-identical so they carry no correctness
  risk.
- Refs: review report; `kairyu/orchestration/{prefix_index,replica}.py`,
  `kairyu/entrypoints/server/app.py`; tests `tests/unit/test_kv_routing.py`.
  **Deferred (risk/complexity, need care or their own change):** P1 incremental
  detokenization (correctness-sensitive output path — a subtle detok bug corrupts
  generation, and CPU tests can't cover every tokenizer edge, so not worth a
  perf-only rewrite), P3 (process-split delta wire), P4 (async ledger/router I/O
  — file-handle lifecycle), P6 (eviction leaf heap), P7 (batched spec verify),
  and the MEDIUM-perf items (sampler penalty state, stop-string offset, queue
  coalescing, scheduler deque, KV-event hash chain, page-table cache).

### 2026-07-04 — [progress] Review remediation Phase 5: bench scoring correctness + security
- What: Fixed the scoring-integrity and security defects in the Fugu bench suite.
  **B1**: the MCQ answer-extraction regex matched "answer" + the first letter of
  the following word (so "Answer: B, because the answer depends…" extracted D)
  and the fallback picked lone lowercase articles/pronouns — tightened to a
  bounded letter after the marker and an uppercase-only fallback. **B2**: an
  un-typed `normalize()` error (schema drift KeyError, image/codec, unpickling)
  crashed the whole suite run; it now degrades THAT dataset to `unavailable`
  ("degradation is data, not control flow"). **M6**: the dataset cache is now
  invalidated when the pinned dataset/revision changes, so bumping `hf_revision`
  re-downloads instead of scoring stale rows. **M7**: private-test blobs unpickle
  through a `_RestrictedUnpickler` that blocks class/global loading (was a
  download-time arbitrary-code vector); the judge response fed into the prompt is
  length-capped. **M8**: LCB solutions that start with `from __future__ import`
  no longer become a SyntaxError when the import header is prepended (the future
  import is hoisted). **M10**: the judge verdict regex accepts markdown-emphasized
  labels (`**correct:** yes`) and the judge token budget was raised so a reasoning
  judge is not truncated before its verdict.
- Why: Each silently corrupts the scoreboard (wrong scores, crashed runs, stale
  data) or is a security hole (ACE at download time).
- Refs: review report; `kairyu/bench/{adapters/base,adapters/livecodebench,cache,judge}.py`;
  tests under `tests/bench/`. **Deferred follow-up (design/policy):** B3 (resume
  per-pair config hash), B4 + denominator policy (skipped/unjudged as 0 or n/a,
  show per-target n_scored), LCB per-line/tolerant scoring, sandbox NPROC/session
  hardening, self-judge (judge==target) scoreboard flag, judge prompt delimiters.

### 2026-07-04 — [progress] Review remediation Phase 4: model + quant parity
- What: Fixed the parity-affecting model/quant defects from the full-repo review.
  **M3 (rope)**: unsupported `rope_scaling` kinds (linear/dynamic/longrope) now
  raise instead of silently dropping to None — a silent parity break vs
  hf.generate. **M4 (fp8 load)**: `Fp8Linear` adopts the checkpoint's
  weight_scale shape, so static per-tensor `(1,)` FP8 (and modelopt FP8) load
  instead of a size-mismatch crash. **M1 (nvfp4 oracle)**: the RNE tie table
  applied LUT *values* as indices and dropped two boundaries, corrupting the
  GPU-kernel packing oracle by up to 60%; replaced with the correct even-index
  table for all seven boundaries (and the test that pinned the wrong behavior).
  MEDIUM: DeepSeek MoE config now falls back to HF defaults
  (norm_topk_prob=True, routed_scaling=2.5, first_k_dense=3, n_group/topk_group
  8/4) for trimmed configs, and a missing expert count raises clearly instead of
  `int(None)`; GPTQ/AWQ `group_size=-1` (single whole-input group) normalizes to
  in_features instead of a negative buffer count; `tp_view` fails fast on MoE
  (no dense down_proj to row-parallelize) like it already does for MLA; bare
  `quant_method: "fp8"` rejects block-wise FP8 (weight_block_size) loudly
  instead of mis-routing DeepSeek block-FP8 to the per-channel path.
- Why: Each is a silent wrong-output or load-time failure on real checkpoints;
  all are CPU-validatable and covered by new tests.
- Refs: review report; `kairyu/models/{config,parallel}.py`,
  `kairyu/quant/{linear,nvfp4}.py`, `kairyu/engine/core/quant_config.py`;
  `tests/unit/test_config_and_fp8_load.py`, `test_quant_compute.py`.
  **Deferred (needs GPU + SpecForge reference to validate):** EAGLE-3 midlayer
  RoPE (H1) and KV-cached rollout feedback (H2) — both affect draft ACCEPTANCE
  RATE only, not output correctness (verification is by the target), so no CPU
  test can validate a fix; plus the design items (linear_factory context,
  forward_fused wiring, HF-name-preserving TP/EP wrappers, draft-head quant).

### 2026-07-04 — [progress] Review remediation Phase 3: orchestration + fleet reliability
- What: Fixed the L2 fleet/orchestration HIGH defects from the full-repo review.
  **O1**: request errors were all counted as replica failures — a new
  `UpstreamClientError` (4xx) is raised by the openai backend and excluded from
  `consecutive_failures`, so one misbehaving client can no longer cascade-eject
  the pool. **O2**: the HealthProber was ordinal-keyed against a dynamic
  id-keyed pool (wrong-replica restore / IndexError / silent prober death);
  it is now id-keyed, resolves URLs per id, and `run()` swallows a bad tick.
  **O3**: the prober now probes `/readyz` (readiness) not `/health` (liveness),
  so a drained/wedged node stays ejected — O1+O3 together kill the flap loop.
  **O4**: the Conductor wraps each unit so a transient backend error records a
  trace event and returns best-so-far instead of raising and discarding every
  completed output. MEDIUM: **M2** orchestrator direct calls no longer mint a
  random per-request session_id (which defeated prefix + least-outstanding
  routing); **M4** KvEventIndex stamps freshness only after a valid apply,
  handles vLLM `AllBlocksCleared`, and the ZMQ drain drops malformed frames
  instead of aborting; **M5** `remove_replica` calls `prefix_index.forget_replica`
  so a re-added id can't inherit phantom prefixes; **M7** lifespan shutdown
  isolates a crashed background task and shuts every engine down independently.
- Why: These are DoS / flap-loop / cost-and-routing-correctness defects the
  single-node CPU tests could not exercise.
- Refs: review report; `kairyu/orchestration/{replica,conductor,orchestrator,kv_index}.py`,
  `kairyu/engine/{backend,openai_backend}.py`, `kairyu/deploy/{prober,registry,spec,builder}.py`;
  tests under `tests/unit/`. Deferred follow-up: M1 (verifier non-target deps +
  _SafeDict masking), M3 (MoA path Budget/cost wiring), M8 (run_chat periodic
  keep-alive), and the KvEventIndex↔ReplicaPool integration (design item).

### 2026-07-04 — [progress] Review remediation Phase 2: API security + tenant isolation
- What: Fixed the CRITICAL/HIGH L3-server defects from the full-repo review.
  **C3 (CRITICAL) batch/file tenant isolation**: File/Batch objects gained an
  `owner`; the store scopes every get/read/list/cancel and cross-tenant access
  reads as not-found — a tenant can no longer enumerate or read another's batch
  prompts/outputs (worker output/error files inherit the batch owner). **S1**:
  a non-object JSONL line becomes a per-line error instead of wedging the job
  in_progress forever. **S2**: invalid sampling params (top_p=0, n=0,
  temperature<0) return 400, not a 500/mislabeled-502. **S3**: streaming chat
  and /v1/completions are now metered (were a billing bypass) — usage flows to
  the ledger; orchestrator-stream/responses/embeddings metering still TODO.
  **S4**: `tokens_per_minute` is enforced via a per-tenant token bucket charged
  post-response. **S5**: `/admin/drain` requires an admin key when configured
  (was any data-plane key = one-request DoS) and gains `/admin/undrain`.
  **S6**: streamed `delta.tool_calls[]` carry the required `index` (SDK
  accumulation). **S7**: `/v1/files` upload is size-capped (413) to prevent
  gateway OOM.
- Why: These are cross-tenant disclosure, billing bypass, and DoS holes that
  the single-tenant CPU test suite could not see.
- Refs: review report; `kairyu/batch/{store,worker}.py`,
  `kairyu/entrypoints/server/{batch_routes,app,health,settings,tenancy,protocol}.py`;
  tests under `tests/server/` + `tests/unit/test_batch_store_tenancy.py`.
  Deferred to a Phase 2 follow-up: MEDIUM items (Prometheus label cardinality,
  /v1/responses store bounds + tenant scope, error-body leak scrub, AUTO-model
  param handling, non-ASCII bearer 401, embeddings validation) and full S3
  metering coverage.

### 2026-07-04 — [progress] Repo-wide review remediation Phase 1: engine-core correctness
- What: Fixed the CRITICAL/HIGH engine-core defects found in the 2026-07-04
  full-repo review (report in job scratch). **C1 radix cache poisoning**:
  `commit_and_release` folded the final sampled token's page as computed even
  though the decode loop never writes that token's KV — a page-boundary
  completion poisoned the next multi-turn prefix (silent wrong output ~1/16 of
  requests). Now caps committable length below the unwritten final token
  (`radix_kv.py`). **C2 oversized-prompt permanent death**: a prompt larger
  than the whole KV cache blocked the head of line forever, turning every empty
  schedule into a fatal engine stall that killed all concurrent requests. The
  scheduler now rejects unadmittable prompts at admission (finish_reason
  "length", drained via `drain_rejected`), and all four engine cores
  (EngineCore/OverlapEngineCore/PipelinedEngineCore/EngineLoop) replace the
  fatal stall with `reject_waiting_head`. **E1 ZMQ receiver death**: a dead
  receiver left every subsequent request hanging; `_ensure_started` now respawns
  a fresh child over a crashed one and per-frame errors no longer kill the loop.
  **E2 state leaks**: `Scheduler.forget` + runner `release` reclaim finished
  per-request state (output lists, sampler seeds, grammar enforcers) — wired
  into `EngineLoop`. MEDIUM: engine_service per-message fault isolation,
  `resume_with_kv` honors ignore_eos/min_tokens/stop_token_ids/finish_reason,
  `RemoteKVReceiver.adopt` frees the allocation on failure, `zmq generate()`
  aborts on cancel, NIXL send yields instead of busy-spinning. LOW: PagePool
  rejects duplicate free ids, torch attention builds indices on the query
  device.
- Why: The CPU test suite was single-turn/single-tenant and could not see these
  multi-turn / long-running / crash-path failures; each is output-corrupting,
  a DoS, or an unbounded leak on the deploy-day paths.
- Refs: review report; `kairyu/engine/core/{radix_kv,scheduler,engine_core,
  overlap,pipeline,pd_remote,pages,model_runner,spec_runner,engine_service}.py`,
  `kairyu/engine/{engine_loop,zmq_backend}.py`; tests under `tests/unit/`

### 2026-07-03 — [progress] Fugu benchmark suite: one-command quality scoreboard (G6 P-C1)
- What: 646 → 730+ tests. New `kairyu/bench/` package + `kairyu bench
  run/download/report/list` CLI. All 11 rows of the Fugu release table
  (sakana.ai/fugu-release) implemented as adapters: GPQA Diamond, HLE,
  LiveCodeBench(+Pro community mirror), SciCode, CharXiv Reasoning, MRCRv2,
  LongBench-v2 (annotated substitute for the unpublished "Long Context
  Reasoning"), τ³-Bench Banking / SWE-Bench Pro (mini-swe-agent scaffold) /
  Terminal-Bench 2.1 (Harbor) as official-harness wrappers. One command
  downloads missing datasets (normalized JSONL cache under
  ~/.cache/kairyu/benchmarks, $KAIRYU_BENCH_CACHE), runs every benchmark ×
  every target, and writes bench/results/fugu/<run_id>/ with per-item
  evidence, methodology (dataset revisions, judge model, truncation policy)
  and a footnoted Fugu-layout scoreboard (JSON+MD). Degradation is data:
  docker/gated-dataset/judge/vision/context-length preconditions produce
  skipped/partial cells with reasons — exit 1 only on hard failures; same
  --run-id resumes. Configurable LLM judge endpoint (HLE free-form, CharXiv,
  τ user-simulator; unjudgeable items recorded, never guessed). Execution
  scoring in an rlimit subprocess sandbox (documented as not a security
  boundary). Orchestration measured as plain model names (kairyu-auto,
  kairyu-auto-max) via the new `orchestrators:` DeploymentSpec map. New
  extras: [bench] (datasets/hub/pillow/h5py), [bench-agentic]
  (mini-swe-agent/swebench/harbor; tau3 documented as git install). Offline
  fixtures keep the default CPU suite and --offline-fixtures runs hermetic;
  networked download tests are hf_hub-marked.
- Refs: goal G6 P-C1/P-B4, roadmap §6 evidence rules; `kairyu/bench/`,
  `kairyu/entrypoints/cli.py`, `docs/benchmarks.md`,
  `examples/{deploy_multi_orchestrator,bench_fugu,agent_pool_max}.yaml`,
  `tests/bench/`

### 2026-07-03 — [design] DeploymentSpec gains named `orchestrators:` (m7 D3 / m11 D2 amendment)
- What: `DeploymentSpec.orchestrators: dict[name, OrchestratorSection]` serves any
  number of named orchestrations (e.g. `kairyu-auto` + `kairyu-auto-max`) from one
  YAML; the legacy single `orchestrator:` key stays and is still served as
  `kairyu-auto`. Validators: name collisions with engines/pools rejected at spec
  load; `orchestrator:` + `orchestrators["kairyu-auto"]` double-declaration
  rejected. Builder passes the named map to `create_app(orchestrators=)` — the
  m11 tiered-auto path was already server-side, just not YAML-expressible.
- Why: The Fugu-suite benchmark work (G6 P-C1) needs "orchestration with an
  arbitrary model composition" to be deployable, then benchmarked as just another
  model name on the same endpoint. Previously `kairyu-auto-max` was reachable
  only via the `create_app` kwarg in tests, never from `kairyu serve`.
- Refs: `kairyu/deploy/{spec,builder}.py`,
  `tests/unit/test_deployment_spec.py`, `tests/server/test_serve_builder.py`

### 2026-07-03 — [progress] M19 complete: deploy-ready — the local-complete plan is DONE
- What: 627 → 646 tests. Dockerfile.cuda (nvidia/cuda 12.4 + gpu/hf/fleet
  extras), GPU compose (device reservations, model volume), Helm
  values-gpu.yaml (per-profile nodeSelector: pcie-gddr / nvlink-hbm),
  scripts/gpu_gates/ covering runbook §0/1/2/3/6/7/9 + G4/G5 — every script
  --dry-run capable, with a CPU suite pinning that dry-runs emit command
  plans AND every referenced path exists today. [gpu] extra with
  sys_platform=='linux' markers (macOS uv sync clean — verified).
  **All 13 milestones of the local-complete plan (M8–M19 + M10a/b) are
  implemented.** Remaining work is strictly the hardware list: performance
  gates, kernel selection/tuning, fabric bring-up, and `pytest -m gpu` /
  scripts/gpu_gates execution.
- Refs: `docs/design/m19-deploy-packaging.md`, `Dockerfile.cuda`,
  `scripts/gpu_gates/`, `deploy/helm/kairyu/values-gpu.yaml`

### 2026-07-03 — [progress] M11 complete: Fugu-class product surface + tenancy
- What: 610 → 627 tests. Usage threaded through MoA/Conductor/Orchestrator
  (was dropped at three layers) — the AUTO path now returns REAL summed
  usage (the m9 usage=None fallback removed at that call site only).
  run_chat streaming: direct route streams live token deltas; multi-stage
  routes emit SSE COMMENT keep-alives (data: lines would break the OpenAI
  SDK) then a buffered final; X-Kairyu-Trace: 1 opts into a kairyu_trace
  field. Tiered auto models (orchestrators dict; kairyu-auto-max routes
  multi_agent through MoA — previously dead code). Tenancy v1: auth stores
  the matched key in scope state, TenantLimitMiddleware runs INSIDE auth
  (401 wins; unauthenticated never drains buckets), per-tenant token
  buckets, O_APPEND JSONL usage ledger written from handlers,
  /admin/usage; isolation + exact reconciliation gates. /v1/responses
  (reviewed subset: exact output-item shapes, input/output_tokens usage
  names, instructions, previous_response_id store; stream descoped) and
  /v1/embeddings (base64 = the SDK default) — both OpenAI SDK round-trip
  tested (openai>=1.66). Vision wire format (content-parts flattening
  everywhere incl. batch-worker path; image parts 400 on non-vision
  engines). F5 CPU: priority admission with aging (injectable clock,
  effective priority at sort time, head still blocks on KVCacheFull),
  AdmissionController (gateway-observable TTFT EMA; admit/defer/shed),
  autoscale_decision hysteresis. Open WebUI compose + frontier_compare
  bench harness (scoreboard schema pinned).
- Refs: `docs/design/m11-product.md` (Status: Implemented);
  `kairyu/entrypoints/server/{tenancy,extra_routes,slo}.py`,
  `kairyu/orchestration/{orchestrator,conductor,moa}.py`,
  `bench/frontier_compare.py`, `deploy/compose/docker-compose.webui.yaml`

### 2026-07-03 — [progress] M10b complete: KV-aware routing
- What: 594 → 610 tests. PrefixIndex (text-chunk approximate trie — the
  gateway has no token ids, review A12; prefix-chained keys, LRU-capped);
  ReplicaPool opt-in prefix scoring (α·overlap − β·outstanding, session
  affinity still first, prefix_match decision reason; disabled default keeps
  m5 behavior byte-identical). RadixKVCache(event_sink=): BlockStored on the
  computed False→True transition + decode-extension nodes (allocate never
  emits; _release double-fire guarded), BlockRemoved ONLY from eviction —
  removed hashes proven identical to stored hashes. KvEventIndex (precise
  per-replica block hashes, staleness > 500 ms → None = fall back to the
  trie) + ZMQ PUB/SUB transport with a chaos gate (publisher killed →
  staleness fallback). Offline (α, β) grid tuner over PlacementRecords.
  Security-review hardening: /admin/drain pinned auth-protected when keys
  are configured (keyless = trusted-mesh mode by explicit m7 D5 choice).
- Refs: `docs/design/m10-fleet-cpu.md` (M10a+M10b Implemented);
  `kairyu/orchestration/{prefix_index,kv_index}.py`,
  `kairyu/engine/core/radix_kv.py`, `kairyu/orchestration/learning/dataset.py`

### 2026-07-03 — [progress] M10a complete: elastic fleet base
- What: 584 → 594 tests. ReplicaPool reworked to id-keyed dynamic membership
  (legacy sequences auto-id "0".."N-1" so HRW mappings AND Prometheus labels
  are unchanged — zero existing-test edits beyond one error message).
  add/drain/remove lifecycle: drain stops NEW placements (HRW runs over
  eligible = healthy ∧ not-draining), remove refuses in-flight unless forced,
  late completion on removed ids is a no-op. HRW remap property gates:
  removal moves ONLY the departed replica's sessions; addition moves ~1/N.
  deploy/registry.py: TTL-heartbeat ReplicaRegistry (injected clock),
  DiscoverySource protocol (static + registry; k8s-endpoints is a deploy-day
  adapter), PoolReconciler (drain-then-remove, tolerates in-flight refusal
  across ticks). POST /admin/drain flips /readyz 503 (node role).
  kairyu/telemetry.py traced_span (L2-safe, no-op without the otel extra) +
  pure-ASGI TracingMiddleware + pool-placement span; opentelemetry-sdk in
  dev group + otel extra. BatchStoreProtocol (full 8-method surface). Helm
  chart (readiness /readyz, config at /etc/kairyu/config.yaml) +
  scripts/kind_smoke.sh + CI kind-smoke job + helm-template render test.
- Refs: `docs/design/m10-fleet-cpu.md` (M10a Implemented);
  `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`,
  `kairyu/telemetry.py`, `deploy/helm/kairyu/`

### 2026-07-03 — [progress] M18 complete: real-byte KV transfer + two-process P-D
- What: 571 → 584 tests. kv_serde (PagedKVPool ⇄ PageFrame, layer-major
  fragments, MLA empty-v contract, loud mismatch errors, pool_fingerprint
  handshake). KVHandoff seam widened to carry source page ids (a
  byte-extracting handoff cannot recover the tail page from tokens — the
  freed tail gets reallocated). RemoteKVHandoff/RemoteKVReceiver over the m6
  transport protocol: copy-before-commit ordering, receiver-side dedup skips
  injection of radix-cached pages, sender page ids remapped to
  new_full_pages+(tail). StreamCopyKVHandoff (side-stream copy window;
  synchronize even on failure). NIXL adapter (deferred import;
  registration-once + descriptor math pinned via fake module). FLAGSHIP:
  two REAL processes over TCP — prefill extracts page bytes between
  execute() and update(), decode adopts via resume_with_kv and decodes;
  outputs == single-engine greedy AND per-page sha256 byte parity.
- Refs: `docs/design/m18-kv-transport.md` (Status: Implemented);
  `kairyu/engine/core/{kv_serde,pd_remote,handoff_stream,kv_transport_nixl_gpu}.py`,
  `tests/dist/test_pd_two_process.py`

### 2026-07-03 — [progress] M17 complete: graph-capture seam + EAGLE-3/MTP draft heads
- What: 553 → 571 tests. StepExecutor seam: decode_buckets policy
  (vLLM-style sizes), GraphStepExecutor (capture-once-per-bucket, static
  buffer copy-in, padding to scratch page with outputs dropped, invalidate(),
  oversize→eager) fully pinned against FakeGraphBackend; cuda_graph_gpu.py
  holds the only CUDA lines (side-stream warmup, shared pool). DraftSource
  protocol: n-gram default byte-identical; ModelDraftSource e2e gate — a
  perfect draft through the FULL spec pipeline == plain greedy with >0.9
  acceptance. EAGLE-3 head per corrected review pins (2H midlayer, pre-norm
  residual, fc [H,3H] once per cycle, TRAINED reduced-vocab lm_head + d2t
  offset map, target-aliased embeddings) + SpecForge loader with format-drift
  guards. DeepSeek MTP head (embedding-first eh_proj, separate physical
  head/embed tensors, MoE decoder block at layer_index=num_hidden_layers) +
  extra-layer checkpoint loader. Scope honesty: batched decode capture rides
  FlashInfer's decode wrapper on deploy day (A1); grammar-rollback spec
  stays deferred.
- Refs: `docs/design/m17-graphs-drafts.md` (Status: Implemented);
  `kairyu/engine/core/{step_executor,graph_buckets,draft,cuda_graph_gpu}.py`,
  `kairyu/models/{eagle,mtp}.py`

### 2026-07-03 — [progress] M16 complete: TP/EP/PP run over real multi-process collectives
- What: 547 → 553 tests (incl. 5 gloo spawn gates that run in the default
  suite). TorchDistCommunicator (m5 protocol + tensor extension; NCCL is a
  constructor argument on deploy day). TP: pre-sharded-config scheme
  (tp_view divides heads/kv/intermediate — modules and kv pools come out
  rank-local automatically), get_slice per-rank loading with FULL-config
  bounds, RowParallelLinear (bias once, after the all_reduce), embed/lm_head
  replicated (every rank holds full logits → every rank samples identically,
  m5 D1 kept). TP=2 spawn gate: EngineCore on rank 0 via DistTPModelRunner
  (snapshot broadcast + A11 handshake), worker_step_loop on rank 1 — greedy
  output IDENTICAL to single-process for llama AND qwen2 (bias) tinies.
  EP: EpMoeBlock over uneven all_to_all_single (counts exchange first);
  EP=2 ≡ single-block to 1e-5. PP: PpStageModel stage seam (embed/mid/final,
  rebased per-stage pools) + hidden send/recv; PP=2 greedy ≡ single-process.
  RequestSnapshot finally extended per the m12 mandate (outputs/sampling/
  num_cached_tokens + allocation aliases). Quantized × TP rejected loudly.
- Refs: `docs/design/m16-distributed.md` (Status: Implemented);
  `kairyu/engine/core/{dist_comm,worker,pp_worker,step_input}.py`,
  `kairyu/models/{parallel,moe_parallel}.py`, `tests/dist/`

### 2026-07-03 — [progress] M15 complete: Qwen3-MoE and DeepSeek-V3 with full parity
- What: 530 → 547 tests. Sparse MoE blocks (Qwen3 softmax top-k with fp32
  routing; DeepSeek sigmoid + correction-bias grouped top-k matched exactly —
  bias affects selection only, top-2 group scores, +1e-20 renorm eps,
  routed_scaling on routed only, shared experts, first_k_dense_replace).
  MlaAttention over the latent pool (post-kv_a_layernorm c_kv ‖ roped k_pe as
  ONE kv head, v width 0 — M18 serde contract), q-LoRA and plain-q paths,
  INTERLEAVED rope (DeepSeek default; half-split is wrong), decompress form
  for prefill / absorbed for decode, HF's hardcoded 1e-6 MLA norm eps. yarn
  rope (inv_freq ramp + attention factor + mscale_all_dim² softmax scale).
  Config: dual-alias expert counts, MLA head_dim pinned to qk dims (never
  hidden//heads), kv-pool props (1 head, r+d_rope wide, v=0). Flagship gates:
  logits < 1e-4 AND full-engine greedy == hf.generate for Qwen3-MoE and
  DeepSeek-V3 (q_lora int/None, yarn on/off). Fixture note: random tiny gates
  produce near-tied routing that fp32 noise flips (block itself matches to
  1e-9 on identical inputs) — gates scaled for decisive margins.
- Refs: `docs/design/m15-moe-mla.md` (Status: Implemented);
  `kairyu/models/{moe,mla}.py`, `kairyu/models/{config,layers,llama}.py`,
  `kairyu/engine/core/kv_pool.py`

### 2026-07-03 — [progress] M14 complete: quantized checkpoints load and RUN on CPU
- What: 514 → 530 tests. kairyu/quant/ reference implementations with formats
  verified against AutoAWQ/AutoGPTQ/vLLM/compressed-tensors source and LIVE
  Hub safetensors headers: FP8-E4M3 (clamp-before-cast — torch CPU cast is
  non-saturating), INT8 W8A8 (exact int32 accumulation — the GPU kernels'
  bit-exact oracle), AWQ (out-axis nibble ORDER [0,2,4,6,1,3,5,7], no +1),
  GPTQ (sequential in-axis packing, z-1 storage offset, g_idx always), NVFP4
  (low-nibble-even packing, bit-3 sign, fp8 block scales × fp32 global, RNE
  boundaries). QuantizedLinear modules hold packed buffers under checkpoint
  names; forward_fused is the Triton seam (kairyu/kernels/ stubs, gpu-marked).
  Loader: linear_factory hook live, state_dict-based iteration (non-persistent
  buffers excluded), quantized payloads verbatim + assign=True + lm_head
  re-tie. Guards: AWQ non-gemm, GPTQ v2/non-4bit, compressed-tensors FP4
  (different names + inverted scale) all rejected loudly. Flagship gate: all
  five schemes quantize the tiny llama, write HF-format checkpoints, load,
  and generate through the FULL engine on CPU (8-bit ≥50% greedy agreement;
  4-bit non-degenerate at hidden-64).
- Refs: `docs/design/m14-quant-compute.md` (Status: Implemented);
  `kairyu/quant/{fp8,int8,awq,gptq,nvfp4,linear}.py`, `kairyu/kernels/`,
  `tests/gpu/test_quant_kernels.py`

### 2026-07-03 — [progress] M13 complete: AttentionBackend seam + FlashInfer adapter + MLA reference
- What: attention extracted into a swappable seam (501 → 514 tests, all M12
  parity suites unchanged — the extraction is behavior-free). Backends are
  plain objects (never nn.Module; state_dict safety), ONE instance shared
  across layers (FlashInfer workspace/plan-cache is per-instance).
  FlashInfer adapter written locally with the reviewed API pins (head_dim_qk
  spelling, workspace buffers, explicit q/kv dtypes, int32 host/device index
  arrays, bottom-right causal assertion, per-chunk plan cache) — logic
  CPU-pinned against an injected fake module, kernels mirrored in tests/gpu/
  (7 deselected until deploy day). MLA reference math (decompress ≡ absorbed
  ≡ naive oracle at the pinned (d_nope+d_rope)^-0.5 scale; shared single-head
  k_pe; post-RoPE cache layout) — M15's trusted oracle for the highest-risk
  kernel work. Selector: env override + hw-profile kernel tier;
  build_engine_loop(model_path=) picks the backend from probe() — deploy day
  is config-free.
- Refs: `docs/design/m13-attention-backend.md` (Status: Implemented);
  `kairyu/engine/core/attention/{__init__,torch_backend,mla_torch,flashinfer_gpu,selector}.py`,
  `tests/gpu/test_flashinfer_gpu.py`

### 2026-07-03 — [progress] M12 complete: real dense models with transformers parity
- What: all five m12 phases landed (471 → 501 tests, 95% cov). ModelConfig
  parses both config.json generations; DenseDecoder (HF-exact module tree)
  covers Llama-3.x / Qwen2 / Qwen3 with verified numerics (rotate_half RoPE,
  llama3 scaling, Qwen3 per-head qk-norm, rectangular chunk masks — SDPA
  is_causal measured wrong over cached prefixes); layer-major PagedKVPool;
  PagedModelRunner behind the m8 ModelRunner protocol with the canonical
  state-access contract, KV-write skip below num_cached_tokens, and
  SpeculativeRunner-compatible decode reads. Flagship gates: fp32 logits
  < 1e-4 vs transformers AND full-engine greedy == hf.generate through
  chunked prefill / radix reuse / page-crossing decode / EOS, per arch.
  Loader (tied embeddings mandatory — safetensors omits lm_head; eos LISTS
  from generation_config; quantized checkpoints fail fast until M14);
  KairyuBackend(model_path=) + kairyu-proc model_path (port reported before
  model load). pytest markers gpu/hf_hub/dist (+strict, default-deselected);
  scripts/parity_real_model.py is the opt-in pre-deploy real-model gate.
- Refs: `docs/design/m12-model-zoo.md` (Status: Implemented);
  `kairyu/models/{config,layers,attention,llama,loader}.py`,
  `kairyu/engine/core/{kv_pool,model_runner}.py`, `scripts/parity_real_model.py`

### 2026-07-03 — [progress] M9 complete: the API is truthful (usage, templates, logprobs, n>1)
- What: all five m9 phases landed (437 → 471 tests, 94% cov; goal G6 gates
  P-A1..P-A5 CPU-green). D1 usage truth — GenerationUsage reported by every
  backend (kairyu/proc/mock/openai-passthrough incl. cached_tokens), OpenAI
  include_usage chunk contract exact, batch JSONL outputs truthful. D2 HF Jinja
  chat templates with transformers byte-match parity (trim/lstrip blocks,
  loopcontrols, HF tojson), per-model DeploymentSpec.chat_templates threaded to
  HTTP AND batch identically. D3 logprobs surfaced (TokenLogprob with bytes,
  chunk-choice placement), /v1/completions (legacy four-array logprobs), real
  n>1 via engine sub-request fan-out (seed identity at i=0, cumulative merged
  streams, sibling aborts, prompt counted once). D4 response_format validated
  (400 not crash) + server-level schema-valid-JSON gate with grammar-stop.
  D5 serving_bench: bearer auth, token-granularity TPOT via include_usage with
  labeled chunk fallback, timestamped results JSON.
- Refs: `docs/design/m9-truthful-api.md` (Status: Implemented);
  `kairyu/entrypoints/server/{app,protocol}.py`, `kairyu/entrypoints/chat_template.py`,
  `kairyu/outputs.py`, `kairyu/engine/kairyu_backend.py`, `bench/serving_bench.py`

### 2026-07-03 — [progress] M8 complete: engine CPU core is real (tokens, sampling, spec decode, process split)
- What: all six m8 phases landed (328 → 437 tests, 95% cov). D1 tokenizer seam
  (HF `tokenizers` + incremental detokenizer, SSE-safe stop-string holdback,
  `finish_early` radix-commit path, finish_reason). D2 real sampling
  (SampledToken/StepOutput protocol ripple across every runner and bench;
  grammar-mask-first with xgrammar stop-token termination; raw-logits logprobs;
  sha256+splitmix64 seeding). D3 scheduler multi-token commit (capped spec
  reservation via chunk.num_tokens, capacity degrade-to-1, budget-accurate,
  exact shortfall release via recorded reservation). D4 n-gram
  SpeculativeRunner (overlay-state scoring, spec ≡ greedy pinned with measured
  acceptance > 0, per-request bypass gating). D5 NVFP4/modelopt/INT8 detection,
  HardwareProfile capability matrix + env-record writer, safetensors
  CheckpointReader with get_slice (M16 seam). D6 process split: shared
  `EngineLoop` extracted; ZMQ ROUTER `engine_service` child process (msgpack,
  ephemeral-port pipe handshake) + `kairyu-proc` backend (lazy zmq.asyncio,
  death detection, shutdown escalation, atexit); parity/stop/abort/usage-fields
  pinned across the process boundary. New deps: tokenizers/safetensors ([hf]
  extra), pyzmq/msgpack ([fleet] extra); coverage configured for the spawned
  service.
- Refs: `docs/design/m8-engine-cpu.md` (Status: Implemented);
  `kairyu/engine/{tokenizer,engine_loop,zmq_backend}.py`,
  `kairyu/engine/core/{sampler,sampling_types,spec_runner,hw_profile,weights,engine_service}.py`

### 2026-07-03 — [design] M8 engine-CPU-core designed and reviewed (local-complete program begins)
- What: `docs/design/m8-engine-cpu.md` — real tokenizer/incremental detokenizer
  (toy stays default), real sampling (SampledToken, StepOutput protocol ripple,
  grammar-mask-first, raw-logits logprobs, sha256 seeds), scheduler multi-token
  commit (capped reservation, degrade-not-stall, scheduler-enforced spec
  precondition), n-gram SpeculativeRunner (overlay-state scoring, per-request
  gating), quant/NVFP4 detection + HardwareProfile + safetensors reader, and the
  ZMQ/msgpack API↔engine process split. 3-reviewer panel APPROVE-WITH-AMENDMENTS;
  amendments applied inline (§6): stop-string SSE holdback + `finish_early`
  radix-commit path, step-thread op discipline (fixes a pre-existing add/abort
  race), budget/watermark accounting for spec chunks, loud update() validation.
- Why: The local-complete mandate (implement everything before GPU hardware;
  only measurement/tuning waits) starts with the engine core. Implementation
  milestones M8–M19 continue the m1..m7 numbering and map to roadmap tracks:
  M8/M9→E1-E2/P-A, M10→F1-F2, M11→P-B/P-C/F5, M12–M18→E-track local halves
  (model zoo, attention backends, quant compute, MoE/MLA, gloo/NCCL distributed,
  CUDA-graph/EAGLE seams, KV transport), M19→deploy packaging.
- Refs: `docs/design/m8-engine-cpu.md`; roadmap §4 Track E/P

### 2026-07-03 — [amendment] G2 hardware contract widened to capability profiles (A100+); fleet-scale decisions amended
- What: G2 §7 gains 2026-07-03 amendments: the goal now spans capability profiles
  covering all NVIDIA GPUs from A100 (SM80) onward — original NVLink arithmetic and
  gates A1–A10 stand on NVLink-HBM profiles; the PCIe-GDDR profile (RTX PRO 6000,
  96 GB, no NVLink) uses TP=1/DP as the 70B scaling base and replaces A3–A5 with a
  placement-crossover report; B2/A10 fabric budgets restated against measured link
  rates; the §6 MoE, autoscaling, and H100-only non-goals are lifted. Related
  amendments: m7 D2 no-k8s → k8s as machine layer only (its own revisit triggers fire
  at thousands of GPUs); m5 D4/m7 D6 session-hash affinity → two-step prefix-aware
  placement then KV tiering; m6 D1 static-only topology relaxed (no-Ray stands);
  ClusterSpec coherence-domain cap 2 → 8; m7 D8 no-OTel flipped. Status notes added
  to m5/m6/m7 design docs and `docs/gpu-runbook.md` (§ header note, §6.1 NVLS scoped
  to NVLink profile).
- Why: The product target is an on-prem DC of thousands of GPUs across BOTH fleet
  shapes (8×H100-class NVLink nodes remain possible; the volume fleet is PCIe-only
  RTX PRO 6000, where 96 GB flips the 70B memory arithmetic and PCIe all-reduce
  latency makes TP-first the wrong default), serving all four model classes
  including MoE — the single-hardware, dense-only, static-fleet assumptions no
  longer hold. Original entries are preserved per progress-log rules.
- Refs: `docs/goals/g2-multi-gpu.md` §7, `docs/roadmap.md` §2/§5,
  `docs/design/m5-*.md` / `m6-*.md` / `m7-*.md` status notes, `docs/gpu-runbook.md`

### 2026-07-03 — [design] Master roadmap + goals G4/G5/G6 defined (gap analysis vs frontier serving)
- What: `docs/roadmap.md` — three-track improvement roadmap (E: own-L1 engine to
  SOTA — real runner/sampling/quant per SM, CUDA graphs + EAGLE-3/MTP via a scheduler
  multi-token commit, profile-aware multi-GPU, MoE/EP, frontier MoE over RDMA;
  F: fleet control plane — dynamic ReplicaPool + registry + k8s machine layer,
  prefix/KV-aware routing fed by RadixKV events, NIXL-candidate KV transport + P/D
  pools, DRAM/NVMe KV tiering, tenancy/SLO admission/autoscaling; P: product
  surface — tokenizer-true usage + cached_tokens, HF Jinja chat templates, streaming
  `kairyu-auto` with orchestration-usage/trace disclosure, Open WebUI integration,
  Responses API/embeddings, nightly frontier-API scoreboard). New goal docs:
  `docs/goals/g4-moe-engine.md`, `g5-fleet-scale.md`, `g6-product-surface.md`.
  Grounding research recorded in roadmap §3/§7: Sakana Fugu product facts (GA
  2026-06, orchestration-as-a-model, Responses API, orchestration token accounting,
  no latency win — Kairyu's wedge is orchestration quality at direct-call latency
  plus trace transparency), SM120 kernel-support gotcha list, and the fleet
  control-plane convergence (Dynamo/llm-d/SGLang gateway/Mooncake/AIBrix:
  prefix-cache-aware routing is the top lever; Kairyu's own RadixKV enables native
  KV-event routing; learned multi-model routing is uncovered white space).
- Why: The product goal (Fugu-class orchestration API + chat UI on an on-prem
  multi-thousand-GPU DC, beating Claude/GPT on TTFT/TPOT/goodput) needed a
  comprehensive gap analysis: the engine compute is placeholder, MoE/quant/spec-decode
  paths are absent, the control plane is static, and the API surface cannot yet
  support billing or honest benchmarks. The roadmap sequences the gaps by impact
  (E1+P-A first) while preserving every existing protocol seam.
- Refs: `docs/roadmap.md`, `docs/goals/g4-moe-engine.md`,
  `docs/goals/g5-fleet-scale.md`, `docs/goals/g6-product-surface.md`

### 2026-07-02 — [progress] M7 Phase 5: deployment guide, runbook §9, README — M7 CPU half complete
- What: `docs/deployment.md` (DC topology, security duty split with the managed cloud
  edge, systemd + compose node setup with documented k8s revisit triggers, config
  walkthrough, rolling model-update drill, observability, interconnect sizing, untested
  k8s appendix); `docs/gpu-runbook.md` §9 (production bring-up on real GPUs: real-engine
  compose smoke, affinity/radix hit-rate measurement through the gateway, rolling-update
  and batch-under-load drills on hardware); README M7 row + serving quickstart. With
  this, every CPU-verifiable G3 gate is implemented and tested (328 tests, 95% cov).
- Refs: `docs/deployment.md`, `docs/gpu-runbook.md` §9, README.md;
  m7 status line updated to Implemented — CPU half

### 2026-07-02 — [progress] M7 Phase 4: OpenAI-compatible batch API
- What: `/v1/files` (multipart upload/metadata/content) and `/v1/batches`
  (create/get/list/cancel) backed by a filesystem `BatchStore` (atomic JSON job state,
  JSONL input/output/error files) and an in-gateway `BatchWorker` lifespan task that
  drains jobs through the same served engines/pools under its own semaphore — strictly
  below the server's global concurrency guard, so interactive traffic stays admitted
  (gate C4, pinned by test). Cancel skips remaining lines; restart recovery marks
  in-flight jobs failed with an explicit resubmit message (single-gateway scope, m7 D7).
  Server helpers `sampling_params_from` / `completion_response` made public for reuse.
  New dep: python-multipart (FastAPI form uploads).
- Refs: m7 D7, G3 gate C4; `kairyu/batch/{store,worker}.py`,
  `kairyu/entrypoints/server/batch_routes.py`, `kairyu/deploy/builder.py`;
  tests `tests/server/test_batches.py`

### 2026-07-02 — [progress] M7 Phase 3: container image, compose topology, CI smoke drill
- What: Multi-stage uv `Dockerfile` (one image for every role; the mounted
  DeploymentSpec decides gateway vs replica), `deploy/compose/` (1 gateway + 3 mock
  replicas with healthchecks, gateway/replica YAML configs), `scripts/compose_smoke.sh`
  (readiness → completion → SSE → affinity-by-metrics → replica kill/eject/zero-5xx →
  prober recovery), and a `compose-smoke` CI job separate from the coverage-gated
  pytest job. The full drill was verified end-to-end with the same configs as local
  processes (kill/eject: 10/10 subsequent 200s; prober auto-restore observed in the
  gateway JSON log); the container build itself runs in CI — this dev environment's
  network policy blocks registry CDNs.
- Refs: m7 D1/D2, G3 gates C1–C3; `Dockerfile`, `deploy/compose/`,
  `scripts/compose_smoke.sh`, `.github/workflows/ci.yml`

### 2026-07-02 — [progress] M7 Phase 2: `kairyu serve` CLI, DeploymentSpec, pool wiring, prober, HTTP affinity
- What: `kairyu serve <deployment.yaml>` console entrypoint builds gateway or replica
  from one YAML: `DeploymentSpec` (new, composes with — does not extend — ClusterSpec,
  m7 D3) declares engines, pools (N remote `openai` members, keyless node-to-node),
  server settings, optional DSL orchestrator, batch section. Builder wraps pool members
  in `ReplicaPool` and passes it into `create_app` unchanged (the pool IS an
  EngineBackend); lifespan starts a `HealthProber` per pool (GETs ejected replicas'
  `/health`, restores via existing `probe()`) and shuts engines down gracefully.
  HTTP affinity gap closed: OpenAI `user` field / `X-Session-ID` header now map to
  `CacheHint(session_id=...)`, so external multi-turn traffic reaches the radix-KV
  warm replica (previously cache_hint was never set on the HTTP path).
- Refs: m7 D3/D4/D6; `kairyu/deploy/{spec,builder,prober}.py`,
  `kairyu/entrypoints/cli.py`, `kairyu/entrypoints/server/{app,protocol}.py`;
  tests `tests/unit/test_{deployment_spec,prober,cli}.py`,
  `tests/server/test_serve_builder.py`

### 2026-07-02 — [progress] M7 Phase 1: server hardening landed
- What: `/health`, `/readyz` (pool-aware: 503 unless every ReplicaPool has ≥1 healthy
  replica), `/metrics` (per-app Prometheus registry; request counts/latency histograms,
  scrape-time pool collector for outstanding/health/decision counts), optional static
  API-key auth (env-sourced, constant-time, health endpoints exempt), global concurrency
  guard (429 + Retry-After on /v1/*), JSON access log with X-Request-ID — all pure-ASGI
  middleware so SSE streams hold their concurrency slot to the last byte. `create_app`
  gains an optional `ServerSettings`; defaults preserve pre-M7 behavior. `ReplicaPool`
  gains read-only `healthy`/`replica_count`/`decision_counts` accessors (still no
  background tasks, m5 D4). New dep: prometheus-client.
- Refs: m7 D4/D5/D8; `kairyu/entrypoints/server/{health,metrics,middleware,settings}.py`,
  `kairyu/orchestration/replica.py`; tests `tests/server/test_{health_metrics,auth,limits}.py`

### 2026-07-02 — [design] M7 productionization designed (G3 goal, D1–D8); G2 2-node scope clarified
- What: Wrote `docs/goals/g3-production-deployment.md` (gates C1–C7) and
  `docs/design/m7-productionization.md`. Decisions: D1 on-prem-DC topology (managed
  cloud WAF/LB front → private interconnect → stateless CPU gateway tier running
  `create_app` + Orchestrator + ReplicaPool of remote `openai`-backend replicas → N GPU
  replica nodes running the same artifact); D2 no Kubernetes — systemd + docker compose
  with everything containerized and documented k3s revisit triggers; D3 new
  `DeploymentSpec` (ClusterSpec untouched — it binds the TP/PP coherence domain, not
  fleet size); D4 health/readyz/metrics + serve-layer background prober (pool stays
  passive); D5 edge-owned WAF/TLS, gateway static API keys + concurrency guard, keyless
  node-to-node; D6 cache layer = per-replica radix KV + pool session affinity, no Redis
  (revisit trigger recorded) — includes fixing the gap that the HTTP path never set
  `cache_hint`; D7 minimal filesystem-backed `/v1/files` + `/v1/batches`; D8
  prometheus-client + stdlib JSON logs, no OTel.
- Why: The product-infrastructure review (LB/scaling, WAF, k8s, GPU pool + API layer,
  cache layer, batch orchestrator, DC–cloud interconnect) found all deployment
  machinery absent: components exist in-process (ReplicaPool, remote-replica backend)
  but nothing wires, launches, secures, observes, or packages them.
- Refs: `docs/goals/g3-production-deployment.md`, `docs/design/m7-productionization.md`;
  amendment: g2 §6 "exactly 2 nodes" clarified as TP/PP coherence-domain cap, not a
  ReplicaPool fleet-size cap (`docs/goals/g2-multi-gpu.md` §6, G3 §5).

### 2026-07-02 — [progress] Repo renamed to `ytworks/kairyu`; README refreshed for M5/M6
- What: GitHub repository renamed from `ytworks/rLLM` to `ytworks/kairyu` (local origin
  updated). README brought up to date: M5/M6 rows in the roadmap, TP / P-D / KV-transport /
  PP components in architecture, engine-core and project-layout sections, test count 290+,
  clone URL. Remaining `rLLM` references in CLAUDE.md / AGENTS.md / gpu-runbook fixed.
- Refs: README.md, CLAUDE.md, AGENTS.md, docs/gpu-runbook.md

### 2026-07-02 — [progress] M5/M6 GPU-independent halves implemented (177 → 289 tests)
- What: All CPU-testable pieces of both designs landed with tests (95% coverage):
  M5 — `Communicator`/`FakeCommunicator`, typed immutable `StepInput`, `TPModelRunner`
  (divergence-checked driver protocol; TP=2 greedy-equivalent to TP=1 through
  KairyuBackend), `tensor_parallel_size` plumbed end-to-end (no-op resolved),
  `ReplicaPool` (rendezvous-hash affinity, queue-depth valve, health ejection,
  `record_replica` JSONL), `PDCoordinator` + `LocalKVHandoff` + `Scheduler.resume_with_kv`
  (copy-before-commit ordering, preemption shield, P-D greedy-equivalence).
  M6 — `ClusterSpec` (topology validation), `KVTransport` protocol + `LocalFabric` +
  TCP-loopback transport, `bench/kv_transfer_bench.py` (CPU-runnable, real fragment
  layout), `openai_backend` replica fixes (real SSE streaming, pooled client, optional
  auth, token counts), async submit/handle runner contract + `PipelinedModelRunner`/
  `PipelinedEngineCore` (inter-step pipelining, bubble accounting: depth-2 <0.2 vs
  depth-1 ≈0.5 pinned by test). GPU-runbook §6/§7 added for the GPU days.
- Refs: `kairyu/engine/core/{comm,step_input,tp_runner,pd,kv_transport,pipeline}.py`,
  `kairyu/orchestration/{replica,cluster}.py`, `docs/gpu-runbook.md` §6–7

### 2026-07-02 — [design] M5/M6 designs written and reviewed (APPROVE-WITH-AMENDMENTS)
- What: `docs/design/m5-intra-node-parallelism.md` (TP runner with non-rank driver +
  typed StepInput prerequisite, ReplicaPool with session affinity, P-D copy-on-handoff
  with copy-before-commit protocol and `resume_with_kv`) and
  `docs/design/m6-inter-node-parallelism.md` (static ClusterSpec — no Ray, KVTransport
  with fragment aggregation, streamed P-D with layer-group final chunk, PP=2 via
  inter-step pipelining on an async ModelRunner handle). Three-reviewer agent panel
  fixed 6 blockers: zero-copy P-D donation withdrawn (dual-tree pool accounting unsound
  + incompatible with disjoint-GPU roles); PP intra-step micro-batching withdrawn
  (bounded at ~1.33× vs B4's 1.6×); `openai_backend` "no change" claim corrected (fake
  streaming, per-request client, mandatory auth, empty token_ids all block B1).
- Why: G2 requires reviewed design docs before implementing each milestone; review
  against the real scheduler/radix code caught mechanisms that could not work as drafted.
- Refs: `docs/design/m5-intra-node-parallelism.md` §7, `m6-inter-node-parallelism.md`
  §7, `docs/goals/g2-multi-gpu.md` §7 Amendments

### 2026-07-02 — [amendment] m1 "Ray arrives with multi-node" superseded
- What: M6 D1 uses a static ClusterSpec + torchrun-style rendezvous for the 2-node
  topology; Ray is not adopted.
- Why: G2 excludes elasticity; a dynamic-placement framework for two static nodes fails
  YAGNI. m1 §3/D4's note was forward-looking, not a binding decision.
- Refs: `docs/design/m6-inter-node-parallelism.md` D1;
  `docs/design/m1-orchestration-and-interface.md` §3

### 2026-07-02 — [design] Multi-GPU goal (G2) defined — drives M5/M6
- What: Wrote `docs/goals/g2-multi-gpu.md`, the acceptance contract for intra-node
  (M5: TP, DP replicas via L2 Router, P-D intra-node) and inter-node (M6: 2-node DP,
  page-granular KV transfer plane, P-D inter-node, PP=2) multi-GPU serving.
  Targets: Llama-3.3-70B FP8 on 8×H100 + 2 nodes (IB/RoCE); vLLM-parity-or-better plus
  absolute scaling-efficiency gates (A1–A10, B1–B5); TP=2 is the scaling base (70B FP8
  cannot run TP=1). MoE/expert parallelism is an explicit non-goal.
- Why: Multi-GPU support existed only as a no-op `tensor_parallel_size` arg and the
  P-D admission-policy half; M5/M6 need a G1-style evidence-first goal to drive
  autonomous development. `docs/goals/` created since the original G1 goal was never
  filed as a document.
- Refs: `docs/goals/g2-multi-gpu.md`; seams: `kairyu/engine/core/engine_core.py`
  (ModelRunner), `scheduler.py` (`pd_separation`), `kairyu/orchestration/router.py`

### 2026-07-02 — [progress] Design-change memory harness added
- What: Added PROGRESS.md, `.claude/rules/progress-log.md`, CLAUDE.md, and AGENTS.md so
  Claude Code and Codex sessions share the same record of design changes and progress.
- Why: Design decisions were scattered across design docs, review-amendment commits, and
  session context; new sessions had no single place to recover project state.
- Refs: `.claude/rules/progress-log.md`

### 2026-07-02 — [progress] README enriched; GPU-day runbook added
- What: README expanded with architecture, roadmap, usage guides, and open-model setup
  (Kimi, Qwen). GPU-day runbook consolidates all remaining GPU-gated work into ordered,
  gated execution steps.
- Refs: commits 9d35360, cc45b08; `docs/gpu-runbook.md`

### 2026-07-02 — [progress] xgrammar structured output integrated (M3)
- What: Token-bitmask enforcer and `response_format` plumbing through the engine and
  OpenAI server.
- Refs: commits ad4e18c, a0851f9; `docs/design/m3-spec-decode-and-graphs.md`

### 2026-07-02 — [progress] Engine core validated end-to-end on CPU (M2 CPU half)
- What: `kairyu` EngineBackend exposed and wired through the OpenAI server; paged-KV
  attention proven greedy-equivalent with real torch tensors on CPU; pre-GPU robustness
  items landed (EOS under overlap, preemption, watermark, abort, pin TTL).
- Refs: commits 991832b, aa382f8, e977e1c; `docs/design/m2-engine.md`

### 2026-07-02 — [amendment] Design-review amendments applied (M2/M4)
- What: Compute-skip, computed gating, output caching, and bandit-router fixes applied
  across M2 KV/scheduler and M4 learning pipeline, per the agent design-review panel.
- Why: Review found gaps in the original D-decisions; docs updated with APPROVE-WITH-
  AMENDMENTS status and amendment sections (§5/§6).
- Refs: commits c14f035, 22b0b53; `docs/design/m2-engine.md` §6,
  `docs/design/m4-router-learning.md` §5

### 2026-07-02 — [design] M2–M4 designs written and reviewed; M4 pulled forward
- What: Design docs for M2 (overlap scheduler + Radix-Paged KV), M3 (spec decode, CUDA
  graphs, P-D separation), M4 (router learning) written and agent-review-approved with
  amendments. M4 was pulled ahead of schedule because it is GPU-independent.
- Why: M2's remaining half needs GPU; GPU-independent work (M3 CPU-side, M4) proceeds
  first to keep momentum.
- Refs: `docs/design/m2-engine.md`, `docs/design/m3-spec-decode-and-graphs.md`,
  `docs/design/m4-router-learning.md`; commits d2675a7, c976c64, 4d83229

### 2026-07-02 — [progress] M1 complete: orchestration + vLLM-compatible interface
- What: L2 orchestration (rule-based Router with JSONL decision log, Conductor role-DAG
  with verifier-gated refinement, MoA, immutable budget) and L3 interface
  (vLLM-signature `LLM`, `AsyncLLMEngine`, OpenAI-compatible server with SSE streaming
  and tool calls, YAML/decorator DSL) built on the `EngineBackend` protocol
  (mock / vLLM / external-OpenAI backends).
- Refs: commit 633fa37; `docs/design/m1-orchestration-and-interface.md`

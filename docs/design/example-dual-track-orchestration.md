# Tiered Example Dual-Track Policy-Ensemble Orchestration

Status: **Accepted; implemented and GPU-verified** (2026-08-25; runs
`20260825T161729Z` coding + `20260825T173343Z` generic — both `verify.sh`
serving gates green on the DTO-D8..D14 served config, digest
`69702ab5af0ab3c0…`). The judge routes every coding-gate request to the
ungated `qwen_think_medium` route (all coding TTFT rows `not_applicable`);
the generic matrix passes its route-aware stage validation, including
primary samples with the full internal DAG and the audit verdict. Manual
Chat UI and Terminal-Bench passes remain on the next-window list. The
previous green run (2026-08-18, `20260818T025710Z`) measured the pre-DTO-D8
nine-role DAG.
Applies to: `examples/qwen3.8-deepseek-v4-8gpu/` and the L2 mechanisms in
`kairyu/orchestration/` + `kairyu/dsl/` that it consumes.
Supersedes the ECO-D2/D3/D5/D6 role graphs, profiles, and profile judge in
`example-coding-orchestration.md` (owner decision, 2026-08-18: the coding DAG,
the general ensemble profile, the LLM profile judge, and the sandbox execution
*stages* are discarded). ECO-D1 (deployment-owned sandbox executors) remains
in force as deployed-but-unreferenced infrastructure; ECO-D4's TTFT gate and
head-streaming mechanism are inherited as DTO-D3. EO-D1/D2/D4/D5/D6 remain in
force.

## Goal

Replace the tiered example's two-profile coding/general orchestration with
one dual-track ensemble DAG for every request, per the owner's specified
process, while keeping the latency/throughput requirements unchanged:
**semantic TTFT (first public `content` token) ≤ 2× the measured DeepSeek L1
direct row at the same concurrency** (paired-row denominator; pinned
`example.json` fallbacks c1 779.22 / c8 833.16 / c16 2710.53 / c32
12399.45 ms p50), `max_concurrency: 256` admission, and the agent-turn
latency envelope of issue #495/#509. E2E latency remains unconstrained by
design; only TTFT is gated.

## Decisions

### DTO-D1 — One dual-track DAG for every request

Three waves under the level-synchronous Conductor scheduler:

- **Wave 1 (dependency-free)**: `head` (Qwen, non-thinking, T=0.7, 256
  tokens; sampling amended 2026-08-19, DTO-D6, and completed to the official
  non-thinking values 2026-08-20, DTO-D8) streams the committed public
  opening from t=0; `draft` (Qwen, thinking at low effort, T=1.0, 1024
  tokens; amended 2026-08-19, DTO-D6, and 2026-08-20, DTO-D8) writes a quick
  complete internal draft (Track B input); `policies` (DeepSeek thinking,
  T=0.9/top_p 0.95, 4096 tokens; cap amended 2026-08-19, DTO-D7; T=1.0 and
  effort-graded caps 2026-08-20, DTO-D8) writes four maximally different
  answer policies in one call (Track A source).
- **Wave 2 (two tracks in parallel)**: `answer_1..answer_4` (Qwen, thinking
  at low effort, T=1.0, 2048 tokens, seed_offset 1..4; amended 2026-08-19,
  DTO-D6, and 2026-08-20, DTO-D8) each answer following one policy — the four
  concurrent tier1 roles spread one-per-replica through
  `queue_depth_threshold: 0`; `critique` (DeepSeek thinking, T=0.6/top_p
  0.95, 4096 tokens; T=1.0 and effort-graded caps 2026-08-20, DTO-D8)
  critically analyzes the UNTRUSTED draft and emits one
  improved complete answer.
- **Wave 3 (merge)**: `compose` (DeepSeek thinking, `role_type: publisher`,
  no sampling — it carries the caller's public intent; amended 2026-08-19,
  DTO-D7) builds the best final answer from the refined answer plus the four
  UNTRUSTED candidates and streams the remainder after the committed
  opening; `prompt_headless` covers head-disabled (tool/format) turns.
  Amended 2026-08-20 (DTO-D10): renamed `synthesis`, weighs the five
  candidates as peers, and is verified inline by the `audit` role before
  its remainder is published; wave 1 additionally runs `image_description`
  on image requests only (DTO-D11).

There is no general profile and no profile judge: dropping `general_roles`
requires dropping `profile_judge` (spec validation), removes a serial
pre-admission DeepSeek call from every turn, and the single DAG's role
contracts are reply-format-faithful (the #495/#496/#509 agent-turn language —
"a downstream publisher emits the actual tool call", demanded-format
obedience — is carried by draft/answers/critique/compose).

Budget: `max_steps: 10` (9 generation calls + 1 headroom for the bounded
empty-final-output re-dispatch), `moa_samples: 0`,
`internal_max_tokens: 4096` (raised to 131072 by DTO-D8, 2026-08-20, then
halved to 65536 by DTO-D12), `expose_intermediate_outputs: true`. DTO-D10
raises the budget to `{max_steps: 19, max_refine_depth: 2}`.

Rationale: owner-specified process (2026-08-18) — diversity through four
policy-differentiated answers, plus a critically refined quick draft, merged
by the strongest model. The scheme is fully expressible in the existing L2
DSL; no core L2 change was required.

### DTO-D2 — Policy fan-out is prompt-bound, not mechanically split

The L2 DSL has no output-splitting mechanism (`{role}` placeholders
substitute whole outputs). The four policies are therefore emitted as one
`policies` output containing `POLICY 1:`..`POLICY 4:` sections; every
answerer receives the whole list and is bound to its own policy by prompt
plus a distinct `seed_offset`. The REQUEST and POLICY LIST blocks are
byte-identical across the four answerers so only the trailing role
instruction differs; this preserves deterministic policy binding, but does
not produce cross-replica cache reuse. The width-four wave intentionally
lands one answer on each independent vLLM replica, and no earlier Qwen call
contains the newly generated policy list, so every replica prefills that
list independently. With `shared_prefix` unset, `prefix_index` does not
route these calls by a prefix fingerprint; session affinity and the
queue-depth valve determine placement. The leading REQUEST block can still
receive replica-local native prefix-cache reuse when an answer lands on a
replica warmed by `head` or `draft`. The policy list steers HOW to answer;
the request text alone defines WHAT — answerers ignore
policy-list content that conflicts with the request or is not answer
guidance (prompt-injection hygiene: policies are derived from untrusted
conversation text).

`policies` originally ran on the non-thinking DeepSeek endpoint for wave-1
latency and the measured cap-burning risk of thinking on composite
deliberative prompts (ECO measurements); DTO-D7 (owner amendment,
2026-08-19) supersedes that split — every DeepSeek role now thinks.
`critique`'s T=0.6/top_p=0.95 sampling keeps the measured anti-looping
rationale (greedy thinking burned the full cap before any verdict on 52% of
calls, issue #509). Amended by DTO-D8 (2026-08-20): T=1.0 — even further
from greedy — with the same rationale preserved.

### DTO-D3 — TTFT gate and head streaming inherited unchanged

The first public byte must come from the dependency-free Qwen `head`
(DeepSeek-first public bytes pay a nondeterministic `<think>` tax that
forfeits the gate — ECO-D4 measurement). The head role, its prompt, its
sampling (256-token cap), the committed-opening/`NO_CONTINUATION`
continuation contract, and the `serving-auto-max-coding` TTFT gate
(≤ 2× paired DeepSeek-direct per concurrency, pinned fallbacks) are carried
over byte-compatibly (head sampling later amended to T=0.7 by DTO-D6,
2026-08-19; the head stays non-thinking); only the continuation node is now
named `compose`.
c32 remains the binding row: 1.87× on the previous DAG, measured 0.67× on
the dual-track DAG (run `20260818T025710Z`) — wave 1 places only two small
Qwen calls and the serial pre-admission judge call is gone, so head TTFT
stays clear of Qwen TP1 saturation.

### DTO-D4 — No verifier, no refinement loop; critique is the quality control

**Superseded by DTO-D10 (owner decision, 2026-08-20).** Kept for the record.

The owner's process has no PASS/FAIL verification stage:
`max_refine_depth: 0` and no `verifies:` role. Track B's `critique` performs
the critical analysis (of the quick draft) and `compose` independently
verifies all candidate material (UNTRUSTED framing) before merging. This
also removes the ~30 s per refinement round of the previous DAG from the
agent-turn envelope. If measured quality regresses materially, the
designated revisit is a verifier attached to `critique` (the final unit
cannot carry one under the streaming contract).

### DTO-D5 — Sandbox infrastructure retained, unreferenced

Owner decision (2026-08-18): the compose `executor` service, the
`kairyu.yaml` `executors: sandbox-python` registry, and `sandbox/` sources
stay deployed (ECO-D1 remains in force) but no role references them — the
DAG has no executor worker (an executor worker without an executor role
fails spec validation). Execution grounding can be re-adopted by a future
DAG without redeploying. `verification.py` no longer asserts execution
stages; the deployment-spec unit test continues to pin the registry, and
compose health-gating keeps the service proven at startup.

### DTO-D6 — Reasoning-effort policy (owner amendment, 2026-08-19)

Owner-specified sampling and effort contract, replacing DTO-D1's original
Qwen sampling:

- The L2 DSL gains role-level `reasoning_effort` (`low|high|max` fixed, or
  `inherit` = the caller's L3 effort) and a spec-level
  `default_reasoning_effort` applied when the caller sends none. The
  orchestrator resolves the effective effort once per call and the Conductor
  sends it on every attempt of an `inherit` role.
- Qwen roles are fixed policy, never caller-derived: `draft` and
  `answer_1..4` declare `low` (thinking at low effort; the shared Qwen
  template clamps any level to low) with T=1.0; `head` declares no effort and
  stays non-thinking with T=0.7, preserving the DTO-D3 TTFT mechanism.
  (Qwen sampling completed to the official per-mode values by DTO-D8,
  2026-08-20; the effort/thinking policy itself is unchanged.)
- DeepSeek roles (`policies`, `critique`, `compose`) declare `inherit`, and
  the example sets `default_reasoning_effort: high`. The DeepSeek vLLM
  service's chat template (`deepseek-role-effort.jinja`) is a scaffold
  passthrough that splices the graded high/max reasoning preamble (the
  deepseek-thinking.jinja texts) after `<｜begin▁of▁sentence｜>` only for
  thinking-scaffold calls — under DTO-D7, every DeepSeek role; `low`/absent
  effort passes through byte-identically.

Rationale: owner decision (2026-08-19) — expose one L3 effort knob
(defaulting to high, settable per request from the API or the Chat UI's
Reasoning Effort advanced parameter) that grades the DeepSeek quality-control
stage, while Qwen's role economics stay pinned in the spec.

### DTO-D7 — Every DeepSeek v4 flash role thinks (owner amendment, 2026-08-19)

Owner decision: DeepSeek v4 flash was always intended to run in thinking
mode, so `policies` and `compose` join `critique` on the thinking worker
(`tier2`; the `tier2-direct` worker is removed) and all three role scaffolds
end `<｜Assistant｜><think>`. `compose` drops `reasoning_closed`: its private
deliberation is separated by the deepseek_v4 reasoning parser and the public
remainder streams after it, behind the committed head opening, so the DTO-D3
TTFT gate is unaffected. This supersedes DTO-D2's non-thinking `policies`
placement and DTO-D1's direct `compose`; the previously measured risks —
wave-1 latency ahead of the fan-out and cap burning inside `<think>` on
composite prompts (each role's token cap now bounds think + output together,
and compose's span shares the caller's public allowance) — are explicitly
accepted, with the bounded empty-final-output re-dispatch as the compose
backstop; for `policies`, the cap-burning risk is resolved by raising its
cap from 1024 to 4096 (owner instruction), the same allowance and rationale
as `critique` (both caps effort-graded by DTO-D8, 2026-08-20). Combined with
DTO-D6, the inherited L3 effort (default high) now
grades every DeepSeek deliberation, not only `critique`.

### DTO-D8 — Vendor-official sampling and effort-graded token budgets (owner amendment, 2026-08-20)

Owner decision, based on model-vendor research (Qwen3.8-27B official
sampling values; DeepSeek-V4-Flash local-serving recommendations; per-effort
token budgets). Sampling is effort-invariant for both models; the parameter
that varies with the resolved L3 effort is the token budget, which bounds
private thinking and the answer together.

- **Qwen sampling completed to the official per-mode values.** Thinking
  roles (`draft`, `answer_1..4`): T=1.0, top_p=0.95, top_k=20. Non-thinking
  `head`: T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5 (the penalty
  applies only to the ≤256-token opener). min_p 0.0 and repetition_penalty
  1.0 equal the engine defaults and stay omitted from the spec. DTO-D6's
  Qwen effort/thinking policy is **unchanged**: fixed `low` thinking on
  draft/answers, effort-less non-thinking head, clamp template intact.
- **DeepSeek roles move to the official local recommendation** T=1.0 /
  top_p=0.95 (the official DeepSeek API ignores T/top_p in thinking mode,
  but this deployment serves the checkpoint locally on vLLM, where they
  apply). For `critique` this replaces the 0.6 chosen against greedy-looping
  (issue #509); T=1.0 sits even further from greedy, so that rationale is
  preserved, not contradicted.
- **Effort-graded token budgets.** The L2 DSL gains
  `RoleSamplingSpec.max_tokens_by_effort` (`{low, high, max}`, all tiers
  required, valid only on `reasoning_effort: inherit` roles — fixed levels
  are constants and effort-less roles never consult the map; this
  deliberately forecloses caller-graded budgets on the fixed-low Qwen roles
  unless the rule is relaxed later). The Conductor resolves the role's
  effort as before and selects the tier, still min()'d against
  `internal_max_tokens` and the caller's public `max_tokens`; the plain
  `max_tokens` is the null-effort fallback. `policies` and `critique`
  declare the vendor-recommended starting budgets
  `{low: 16384, high: 65536, max: 131072}` (fallback 65536 = the
  default-effort tier; owner decision 2026-08-20, superseding an initial
  4096/16384/32768 grading — the TTFT gate is head-based with a paired
  same-concurrency DeepSeek-direct denominator and E2E latency is
  unconstrained by design, so the "≤2× DeepSeek-direct" budget does not
  constrain DeepSeek role budgets). New DSL sampling knobs `top_k`, `min_p`,
  `presence_penalty`, `repetition_penalty` thread through the existing
  role-override path (the engine and OpenAI-compat backend already forward
  them).
- **`internal_max_tokens` 4096 → 131072** — the ceiling must admit the max
  tier; grading stays per-role (the DSL bounds on `max_tokens`,
  `max_tokens_by_effort`, and `internal_max_tokens` rise 32768 → 131072
  accordingly). Consequence: `admission_upper_bound` charges the internal
  ceiling per private step, so the AUTO admission bound grows ~32× and
  admission under tenant quotas/SLO becomes markedly more conservative for
  every request regardless of effort. Accepted.
- **`/v1/responses` forwards `reasoning.effort`** (previously silently
  dropped): OpenAI-style levels are normalized onto the L3 knob —
  minimal/low→low, medium/high→high, xhigh/max→max — so Codex can grade the
  DeepSeek stages, including the max tier via `xhigh`; unknown levels are
  rejected 400. /v1/chat/completions accepts the same aliases and
  normalizes them onto low|high|max before L3 dispatch.
- The caller's public `max_tokens` still min()s over every internal budget
  (existing product semantics): a client sending 4096 clamps the high/max
  tiers back to 4096. The Chat UI default (32768 → 131072) and the
  verification harness (`auto_max_combined_max_tokens` 4096 → 131072) are
  sized so every tier is reachable on the product paths.
- GPU consequences: the recorded green gates (run `20260818T025710Z`) bind
  to the previous served-config digest; the default-effort (`high`) budget
  grows 16× over the previous 4096 allowance and head sampling changed, so the
  TTFT/goodput gates and opener quality must be re-verified in the next GPU
  window. Accepted by the owner (2026-08-20) together with the
  research-scale tiers.

### DTO-D9 — Public-output floor for the always-thinking final unit (owner amendment, 2026-08-20)

Status: accepted (issue #542, option 1 chosen by the owner)

The DTO-D7/D8 interaction left a hard failure at small caller caps: the
caller's public `max_tokens` min()s onto `compose`'s combined think+answer
budget, an all-thinking compose can spend the whole clamp inside `<think>`,
and the byte-identical empty-output re-dispatch (issue #496) fails the same
way — a 502 after paying full dual-track cost (reproduced at `max_tokens:
512`). The fix reserves a floor of public-answer tokens:

- **Config**: spec-level `public_output_floor` (the final unit cannot declare
  sampling), plus role-level `reasoning_close_tag` — the literal that closes
  the span the role's `prompt_suffix` opens. A floor fails Conductor
  validation at deployment startup when the final unit has no close tag or
  in MoA mode (`moa_samples > 0` never consumes it); direct
  `Conductor`/`Orchestrator` construction enforces the DSL `1..131072`
  bounds (review fixes #546–#549). A verifier on the final unit was
  rejected by #547 and is accepted again since 2026-08-20 (DTO-D10 engine
  work): the forced-close retry runs *before* the verdict, also for a
  committed-head call, and a refinement attempt reopens the scaffold with
  the role's own budget. The example sets floor 256 (the head opener's
  sizing of a meaningful public chunk; 0.2% of the Chat UI default cap) and
  `</think>` on `compose`.
- **Phase A**: the final unit's attempt 0 dispatches with `max_tokens =
  B − floor` (B = caller cap minus committed-head spend), capping thinking.
- **Phase B**: on empty public output, the existing single bounded
  re-dispatch — not a new retry slot — re-prompts with the rendered scaffold
  plus the captured attempt-0 reasoning plus the forced close tag, at
  `max_tokens = min(floor, B)`, dispatched as a `reasoning_closed` role so
  parser-classified output is reclaimed as public (issue #496 mechanism).
  The trace keeps `retry:empty_output` with metadata `continuation:
  think_close`.
- **Boundary**: at `B ≤ floor` attempt 0 runs unclamped and phase B gets the
  whole remaining budget; worst-case spend stays within the pre-existing
  ≤2× bounded-retry envelope. Unlimited caller budget, floor-unset specs,
  the committed-head exemption, and internal roles (which tolerate empty
  output) are unchanged.
  Multi-choice calls (`n > 1`) also retain the byte-identical retry without a
  floor: one continuation prompt cannot preserve independent per-choice
  reasoning branches.
- **Trade-off**: an answer that finishes thinking inside phase A may
  truncate up to `floor` tokens earlier than before (public output exists,
  so no retry). Accepted; bounded by the floor.
- GPU consequences: none re-run — recorded gates run at cap 131072 where
  the floor shifts the thinking budget 0.2%; behavior changes only for
  small caller caps and the empty-output retry path.

### DTO-D10 — Peer synthesis + inline audit loop on the final unit (owner amendment, 2026-08-20)

Status: accepted; implemented; serving gates GPU-verified (2026-08-25, runs 20260825T161729Z/20260825T173343Z).

- `compose` → `synthesis`: the final unit no longer anchors on critique's
  refined answer. It examines five UNTRUSTED CANDIDATE answers as peers —
  `answer_1..4` and `critique` (CANDIDATE 5) — compares them, verifies their
  claims independently, and writes one answer that is better than every
  candidate. The committed-opening contract (never repeat the opening,
  `NO_CONTINUATION`, blank-line continuation, reply-format obedience, no
  pipeline mention), `prompt_headless`, the thinking scaffold, and
  `reasoning_close_tag` are unchanged.
- New `audit` role: `role_type: verifier`, `verifies: synthesis`, DeepSeek
  thinking (tier2, `reasoning_effort: inherit`), T=1.0/top_p 0.95 — never
  greedy (issue #509) — with the same effort-graded budget as
  policies/critique — amended 2026-08-25 (DTO-D14): the audit now runs on
  Qwen tier1 at fixed medium effort (spec `high`) with one 16384-token cap — `depends_on: [synthesis, head, image_description]` so
  it judges the committed opening plus the candidate remainder as one public
  answer. Verdict protocol is the Conductor's: first line `PASS` or `FAIL`;
  anything else is inconclusive and triggers one bounded re-verification; a
  FAIL verdict is appended verbatim to the synthesis re-dispatch as
  "Verifier feedback:" (`max_refine_depth: 2`); exhaustion publishes the
  last attempt. `n > 1` skips the audit (one verdict cannot judge
  independent choices).
- Engine prerequisite (same day, `kairyu/orchestration`): a verified final
  unit is published *deferred* in `Conductor.stream` — the head still
  streams from t=0, the verify/refine loop runs to completion, and the
  remainder is emitted once; the DTO-D9 floor retry runs before the verdict
  (also with a committed head) and refinements reopen the scaffold.
  Consequence: time-to-remainder grows by one DeepSeek verdict plus any
  refinement rounds; TTFT (head) is unaffected; the remainder arrives as one
  burst rather than progressively.
- Budget arithmetic (image request, worst case; every `_generate` reserves
  one step): 10 generation units + 1 empty-output re-dispatch + 3 audit
  verdicts + 3 bounded inconclusive re-verifies + 2 refinements = 19 →
  `budget: {max_steps: 19, max_refine_depth: 2}`; text requests leave one
  spare step. `example.json`: `product_normal_calls: 10`,
  `product_max_calls: 19`, `product_max_refinements: 2`.
- Gates: `expected_route: synthesis`; the serving rows additionally require
  an `audit` stage of kind `verification` (`expected_verification_nodes`).
- Rationale: owner decision — audit the synthesized answer before the user
  sees it, bounded; supersedes DTO-D4.

### DTO-D11 — Image-only `image_description` stage (owner amendment, 2026-08-20)

Status: accepted; implemented; serving gates GPU-verified (2026-08-25, runs 20260825T161729Z/20260825T173343Z).

- New Qwen role `image_description` (tier1 — the vision-capable pool,
  `requires: image`, `reasoning_effort: low`, Qwen thinking-mode sampling,
  2048 tokens, REQUEST-first framing — amended 2026-08-25 (DTO-D14): fixed
  medium effort, spec `high`, 4096 tokens): describes every attached image in
  precise objective text (verbatim on-image text, diagrams, UI, layout,
  anything the request refers to) and never answers the request.
- `policies`, `critique`, `synthesis`, and `audit` (all text-only DeepSeek
  roles, which otherwise see only `<image:N>` placeholders) depend on it and
  render an `IMAGE DESCRIPTION (empty when the request has no image)` block;
  the Qwen answerers see the image natively and are unchanged. Under the
  level-synchronous scheduler only wave 1 grows by one Qwen call, on image
  requests only.
- Engine prerequisite (`requires: image`, same day): on text requests the
  role is excluded entirely — no model call, no budget step, trace
  `skipped:condition` (`reason: no_image`) — dependents run as if it were
  absent and the slot renders as ""; head/final/verifier/executor roles
  cannot be conditional; the Orchestrator rejects a conditional role on a
  worker that does not accept images. The serving gates' expected
  generation nodes therefore exclude it (text datasets); admission keeps
  charging it (conservative).
- Rationale: owner decision — give the text-only DeepSeek stages a faithful
  textual view of the image without touching text-request latency.

### DTO-D12 — DeepSeek budgets halved for the Terminal-Bench turn envelope (owner amendment, 2026-08-20; supersedes part of DTO-D8)

Status: accepted; implemented; serving gates GPU-verified (2026-08-25, runs 20260825T161729Z/20260825T173343Z).

- `max_tokens_by_effort` on `policies`/`critique`/`audit` (audit until
  DTO-D14 moved it to Qwen with a single cap): `{16384, 65536,
  131072}` → `{8192, 32768, 65536}` (fallback `max_tokens` 65536 → 32768);
  `internal_max_tokens` 131072 → 65536 (the ceiling still admits the max
  tier); Chat UI `DEFAULT_MODEL_PARAMS.max_tokens` and the harness
  `auto_max_combined_max_tokens` 131072 → 65536; `public_output_floor` 256
  unchanged (0.4% of the new default cap); Qwen role caps unchanged.
- Rationale: Terminal-Bench turns increasingly time out before completion
  against the 900 s agent budget; the serial thinking-DeepSeek chain is now
  policies → critique → synthesis → audit plus refinement rounds, and the
  harness calls with `max_output_tokens: 32768`, so the DeepSeek tiers — not
  the Qwen caps — are the lever. `/v1/responses` still defaults
  `max_output_tokens` to 1024 when a client omits it (pre-existing, not
  changed here). Admission upper bound halves accordingly.
- GPU consequences: the served-config digest changes; both `verify.sh` gates
  and the Terminal-Bench-style Codex run must be re-measured in the next GPU
  window.

### DTO-D13 — Qwen-judged five-route selection (owner amendment, 2026-08-22)

Status: accepted; implemented; serving gates GPU-verified (2026-08-25, runs 20260825T161729Z/20260825T173343Z).

Owner decision (2026-08-22): every request first pays one bounded Qwen
non-thinking judge call that reads the request and picks the route expected
to answer correctly and well at the lowest latency and cost, among five
routes: (1) the dual-track ensemble (unchanged; highest cost, best quality),
(2) Qwen3.8 non-thinking direct, (3) Qwen3.8 thinking at fixed `low`, (4)
DeepSeek-V4-Flash-0731 non-thinking direct, (5) DeepSeek thinking at the
caller's L3 effort. Decisions taken with the owner in planning: routes 2–5
fix the example's official per-mode sampling (DTO-D8) on their final unit;
their `max_tokens` are the vendor-official output lengths (Qwen 131,072 —
the Qwen3.8 card's "Final Response: 131,072"; DeepSeek 393,216 — the
DeepSeek-V4-Flash-0731 card's / API's "maximum output length 384K"); the
ensemble route and everything it owns (DeepSeek tiers, `internal_max_tokens`
65536, Chat UI default 65536, floor 256) are untouched; tool/format turns
are judged too; judge failure falls back to the ensemble.

- **Mechanism (engine prerequisite, same day, `kairyu/dsl` +
  `kairyu/orchestration`)**: the issue #509 two-way profile judge
  (`general_roles`, verdicts `CODE|GENERAL`, keyword fallback, head-disable
  short-circuit) is generalized to N named **profiles**
  (`OrchestratorSpec.profiles: [{name, roles}]`, the primary `roles` DAG is
  the implicit profile `primary`) and a judge with spec-defined
  `choices: [{profile, label, criteria}]` and `fallback`. Kairyu builds the
  verdict prompt from the choices (fastest → most thorough, plus a
  deterministic `tool calling yes/no; image attached yes/no` context line;
  bounded 4,000-character latest-user view; ≤8 tokens, T=0). The verdict is
  attached to the immutable request at the serving boundary before
  preflight/admission (pure-function selection preserved); admission
  reserves the judge plus the most expensive profile. Image requests are
  offered only profiles with an image-capable worker; a verdict outside the
  offered set is unparseable → fallback. Head-disable signals no longer pin a
  profile and the keyword `code_task_signal` fallback is deleted. `/routing`
  reports `profiles` and the judge (`choices`, `fallback`); the trace notes
  `role profile: …` / `profile judge: …` and the `profile_judge`
  classification event are unchanged, and `verification/…/serving_bench.py`
  now retains `classification` stages.
- **Final-unit sampling** (Conductor): a final-unit `sampling` block is
  accepted — style fields (temperature/top_p/top_k/penalties/seed/stop) are
  deployment policy layered over the caller's public params, `max_tokens`
  (or the effort tier) is a cap min()'d with the caller's allowance; the
  caller's `n`/`logprobs`/`response_format`/tools still apply (same pattern
  as the head). The DTO-D9 floor is passed only to profiles whose final unit
  declares `reasoning_close_tag`; a floor no profile can consume still fails
  startup. DSL `max_tokens` / `max_tokens_by_effort` bounds rise 131072 →
  393216.
- **Example config**: worker `tier2-direct` (`deepseek-v4-flash-0731`, the
  non-thinking pool; passthrough template leaves the `</think>`-closed
  scaffold byte-identical); `profile_judge` on `tier1` (`QWEN`/`QWEN_THINK`/
  `DEEPSEEK`/`DEEPSEEK_THINK`/`ENSEMBLE`, fallback `primary`); profiles
  `qwen_direct` (`qwen_answer`: no effort, T=0.7/top_p 0.8/top_k 20/
  presence 1.5, 131072), `qwen_think_low` (`qwen_think_answer`: `low`;
  amended 2026-08-25 (DTO-D14): renamed `qwen_think_medium`, spec `high`,
  T=1.0/top_p 0.95/top_k 20, 131072), `deepseek_direct` (`deepseek_answer`
  on `tier2-direct`, `reasoning_closed`, `<｜Assistant｜></think>`, T=1.0/
  top_p 0.95, 393216), `deepseek_think` (`deepseek_think_answer` on `tier2`,
  `inherit`, `<｜Assistant｜><think>` + `reasoning_close_tag`, T=1.0/top_p
  0.95, 393216). Each direct role is one publisher with REQUEST-first (Qwen)
  or scaffold-inline (DeepSeek) framing that writes the complete answer in
  the demanded reply format, uses a trailing tool result directly, and emits
  the actual tool call when required. Budget `{19, 2}` is spec-level; a
  direct route spends 1 step + the bounded re-dispatch.
- **Gates**: `serving-auto-max` / `serving-auto-max-coding` are
  route-aware: every sample must trace the `profile_judge` classification
  stage and exactly one profile's final unit; ensemble samples keep the
  head/internal/audit contract; each row writes `routes.json` (route
  distribution, per-route TTFT p50, judge latency p50). The TTFT gate
  measures the TTFT-gated routes (`primary`, `qwen_direct`,
  `deepseek_direct`; the conservative max of their per-route p50s) against
  2× the paired DeepSeek-direct row; thinking direct routes are reported,
  and a row with no gated sample records `not_applicable`. `example.json`:
  `product_policy: judged-five-route-dual-track`, `profiles`,
  `profile_judge`, `profile_final_roles`, `direct_route_max_tokens`,
  `ttft_gated_profiles`; `control.py::_validate_ready` asserts the profiles,
  judge choices, and `tier2-direct` binding.
- **Amendment (2026-08-22, live verification)**: on the deployed vLLM
  v0.23 Qwen service the judge returned an empty verdict on every request
  and 100% of traffic fell back to the ensemble — vLLM's Qwen3 reasoning
  parser defaults to "thinking enabled" and drops a non-streamed answer that
  never emits `</think>` unless the request carries
  `chat_template_kwargs: {enable_thinking: false}` (the head survived only
  because it streams). Fix: the judge and every effort-less role whose worker
  accepts `enable_thinking` now send that switch explicitly (Conductor
  `_worker_chat_template_kwargs`, judge request, and the exact final
  preflight intent); `GenerationRequest` accepts template kwargs on any
  upstream-templated chat prompt (text or multimodal; still rejected on
  pre-rendered/token prompts). No example-config change.
- **Supersedes**: ECO-D6's "there is no single-engine route in auto-max"
  and its head-disable profile short-circuit; DTO-D1's "one DAG for every
  request" now describes the `primary` profile.
- **Known limits / GPU consequences**: the judge is a serial pre-head Qwen
  call on every turn and lands on the DTO-D3 TTFT path; the direct routes
  render the role-tagged conversation inside one user turn like every other
  role; from the Chat UI the 65536 caller cap binds before the official
  route caps; vLLM's behavior for `max_tokens` exceeding the remaining
  context with the 131072/393216 caps must be confirmed (add a clamp if vLLM
  returns HTTP 400). Both `verify.sh` gates are green on the current digest,
  and their `routes.json` artifacts record judge latency. A manual image chat,
  a manual per-route Chat UI pass, and a Terminal-Bench-style agent run remain
  outstanding.

### DTO-D14 — Qwen medium tier and audit on Qwen (owner amendment, 2026-08-25; amends the Qwen side of DTO-D6/D8 and the audit worker of DTO-D10)

Status: accepted; implemented; serving gates GPU-verified (2026-08-25, runs 20260825T161729Z/20260825T173343Z).

- Owner decision (2026-08-25): the Qwen thinking roles move from the fixed
  `low` tier to a medium tier, and the audit verifier moves from DeepSeek to
  Qwen. Non-thinking Qwen surfaces (head, `qwen_direct`, the route judge)
  and every DeepSeek generation role are unchanged.
- The L2 ladder stays `low|high|max` (no core change): medium is expressed
  as spec level `high`, mirroring the L3 alias normalization (medium→high).
  `draft`, `image_description`, `answer_1..4`, and the renamed
  `qwen_think_medium` route (ex-`qwen_think_low`) declare
  `reasoning_effort: high`.
- The example gets a local Qwen chat template (`qwen3.8-chat.jinja`; the
  compose mounts leave `../qwen3.8-27b-1gpu/chat_template.jinja`, which is
  untouched): `low` renders byte-identically to the shared template, `high`
  prepends a medium reasoning instruction to the system prompt, and `max`
  clamps to `high`.
- Sampling stays effort-invariant (DTO-D8). Qwen thinking budgets double for
  the medium tier: draft 1024→2048, image_description 2048→4096, answers
  2048→4096 (DeepSeek's low→high ladder is 4x; medium takes the log
  midpoint).
- The audit (DTO-D10) now runs on `tier1` at fixed medium effort with a
  Qwen-native prompt (REQUEST-first plain text, no DeepSeek scaffold, no
  `prompt_suffix`), T=1.0/top_p=0.95/top_k=20, never greedy (issue #509). A
  fixed level cannot carry `max_tokens_by_effort` (DTO-D8 rule), so one
  16384-token cap (the geometric midpoint of the halved DTO-D12 low/high
  ladder) bounds think + verdict. The verdict leaves the serial DeepSeek
  chain; `synthesis` stays on tier2 and the caller's effort still grades
  `policies`/`critique`/`synthesis`.
- Risk accepted by the owner: a 27B Qwen now audits a frontier DeepSeek
  synthesis (previously peer-strength). Both serving gates and the
  served-config digest re-pin are complete (2026-08-25, runs
  `20260825T161729Z`/`20260825T173343Z`).


### DTO-D15 — Public-output floor for chat-template final units (owner amendment, 2026-08-26; extends DTO-D9 to the DTO-D13 `qwen_think_medium` route)

Status: accepted; implemented; serving gates GPU re-verify and digest re-pin pending.

- Trigger (2026-08-26): a SWE-bench Pro run sending `max_tokens: 8192`
  received 502s from `qwen_think_medium` after ~2 × 8192 tokens: the Qwen
  medium tier (DTO-D14) routinely deliberates ~7k tokens, the whole caller
  cap went inside `<think>`, and — because the route's final unit declared
  no `reasoning_close_tag` — `_profile_output_floor` passed no floor, so the
  issue-#496 re-dispatch was byte-identical (prefix-cache hit, same outcome)
  and the run ended `EmptyFinalOutput`. The caller cap is legitimate; the
  server must answer within it.
- Same window, second defect fixed en route: the DTO-D9 forced-close retry on
  a *streamed* call (`/v1/messages`, which always streams internally)
  produced `completion text delta offset mismatch … got 0` → 502. The
  reasoning-only OpenAI stream leaves `text_delta=""`/`text_offset=0` on the
  completion that `_unit_public_output` reclaims as public text, while the
  reclaimed text is already published as a tail delta. The reclaim now
  returns the cumulative form (`text_delta`/`text_offset` cleared) so the
  server slices by its own offset; the unit fixture `ThinkExhaustedBackend`
  is delta-native so the floor tests exercise the real backend shape.
- Third defect found by the live re-check on the deployed Qwen worker: vLLM
  v0.23 returns the private span under the OpenAI-compatible key
  `reasoning`, not `reasoning_content`; the OpenAI backend read only the
  latter, so every Qwen-pool call silently lost its reasoning (the chat
  continuation carried an empty think and the `reasoning_closed` reclaim
  had nothing to publish). The backend now accepts either key on messages
  and stream deltas.
- **Config**: role-level `reasoning_continuation: prefix | chat` (default
  `prefix`, today's behavior) and `reasoning_open_tag`. `chat` declares
  that the worker's upstream chat template opens the reasoning span
  (Qwen-family `<think>`), so the empty-output re-dispatch cannot extend a
  text scaffold; it requires both tags. DSL/Conductor validation rejects
  `chat` without them and an open tag outside `chat`.
- **Phase A** is unchanged: attempt 0 dispatches with `max_tokens = B −
  floor`.
- **Phase B (`chat`)**: the same single bounded re-dispatch sends the
  rendered role prompt unchanged plus a final assistant message
  `"<open>\n<attempt-0 reasoning>\n<close>\n\n"` with vLLM
  `continue_final_message: true` / `add_generation_prompt: false`
  (`GenerationRequest.assistant_prefill`; OpenAI backend, vLLM upstream and
  template-owned chat prompts only). The Qwen template re-renders that
  message as a closed thinking turn (`<|im_start|>assistant\n<think>\n…\n</think>\n\n`)
  and the model continues with the reserved `min(floor, B)` public tokens;
  vLLM requires the prefill to appear verbatim in the rendered chat, hence
  the exact shape (probed on the deployed vLLM v0.23 Qwen worker). The
  dispatch is `reasoning_closed` so parser-classified output is reclaimed
  (issue #496); the trace keeps `retry:empty_output` with
  `continuation: think_close` and adds `mode: chat`.
- **Example**: `qwen_think_answer` declares `reasoning_close_tag: "</think>"`,
  `reasoning_continuation: chat`, `reasoning_open_tag: "<think>"`; floor 256
  now applies to the `qwen_think_medium` profile (0.4% of the Chat-UI cap;
  0.2% of 131072). The other profiles are unchanged.
- **Trade-off**: as DTO-D9 — an answer that finishes thinking inside phase A
  may truncate up to `floor` tokens earlier; accepted. The continuation
  prompt shape is model-family knowledge declared as config, like
  `prompt_suffix`.
- GPU consequences: served config changed → both serving gates and the
  digest re-pin must be re-run before the next status claim.

## Acceptance

- CPU suite green with the rewritten example pinning test
  (`tests/unit/test_tiered_frontier_examplectl.py`): workers `tier1`,
  `tier2`, `tier2-direct`, the ordered eleven-role dual-track primary list
  (DTO-D10/D11) cross-checked against `example.json`, the four direct-route
  profiles and the Qwen judge with its five choices (DTO-D13),
  budget `{19, 2}`, `synthesis` on the thinking `tier2` worker with
  non-empty `prompt_headless` (DTO-D7) and CANDIDATE 5 = critique, `audit`
  verifying `synthesis` with a non-greedy PASS/FAIL prompt,
  `image_description` on `tier1` with `requires: image` feeding every
  DeepSeek role, UNTRUSTED delimiters in `critique`/`synthesis`, per-policy
  binding (`POLICY n` in `answer_n`), distinct answerer seed offsets, ensemble
  ceiling/Chat UI/harness caps 65536 (DTO-D12), and direct-route official
  sampling/output caps 131072/393216 (DTO-D13).
- Launcher `_validate_ready` requires the eleven-role dual-track primary DAG,
  `stream_head: head`, the four named direct-route profiles, the Qwen judge's
  five choices with fallback `primary`, `max_steps: 19`,
  `max_refine_depth: 2`, and the tier1/tier2/tier2-direct engine bindings.
- Both serving gates are route-aware: every sample traces the `profile_judge`
  classification and exactly one profile's final unit; primary samples retain
  the head/synthesis public stream and audit contract: `require_head`,
  `expected_route: synthesis`, and `expected_verification_nodes: audit`.
- `serving-auto-max-coding` passes its TTFT gate at every concurrency row that
  contains a TTFT-gated profile (c32 is the watch row); thinking-direct-only
  rows report `not_applicable`. Results are recorded in the example's
  MEASUREMENTS.md.
- GPU verification status for DTO-D10..D14: both `verify.sh` gates are green
  on digest `69702ab5af0ab3c0…`, and the per-row `routes.json` artifacts record
  judge latency. Outstanding manual checks are an image chat (the
  `image_description` stage appears in the internal-work item;
  `skipped:condition` on a text chat), a Chat UI pass for every route, and a
  Terminal-Bench-style Codex run over `/v1/responses` checking per-turn time
  against 900 s and that tool-call turns PASS the audit.
- Test-policy accounting: the executor-status gate tests and retired general
  profile assertions were deleted with the features they protected; DTO-D13
  reintroduced profile/judge assertions and route-aware gate tests;
  base→head collection counts are reported in the change.

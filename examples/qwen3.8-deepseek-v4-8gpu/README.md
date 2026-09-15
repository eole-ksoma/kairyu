# Qwen3.8 + DeepSeek V4 judged five-route orchestration on 8 x RTX PRO 6000

This example starts one layered quality-first product path with one command:

```text
Open WebUI
    -> Kairyu L3 product API (:8003; model kairyu-auto-max)
        -> Kairyu L2 route judge: ONE bounded Qwen non-thinking call reads the
           request and picks the fastest route expected to answer correctly
           and well (DTO-D13):
             QWEN           -> qwen_direct     Qwen3.8 non-thinking, one call
             QWEN_THINK     -> qwen_think_medium  Qwen3.8 thinking (medium), one call
             DEEPSEEK       -> deepseek_direct DeepSeek non-thinking, one call
             DEEPSEEK_THINK -> deepseek_think  DeepSeek thinking at the L3 effort
             ENSEMBLE       -> primary         the dual-track ensemble DAG below
        -> Kairyu L2 primary profile: the dual-track ensemble DAG
           (11 roles: 10 generation + 1 audit verifier)
            Track A: DeepSeek writes 4 maximally different answer policies;
                     4 Qwen replicas answer in parallel, one policy each
            Track B: a Qwen quick draft, critically refined by thinking
                     DeepSeek — concurrently with Track A
            Merge:   thinking DeepSeek synthesizes one better answer from
                     the 5 peer candidates (refined answer + 4 policy
                     answers); a Qwen audit verifies it (PASS/FAIL,
                     <= 2 refinement rounds) before the remainder streams
            Images:  a Qwen image_description stage runs on image requests
                     only and feeds the text-only DeepSeek roles
        -> deployment-owned L1 pools: 4 x Qwen3.8-27B-FP8 TP1 (GPU 0-3),
           DeepSeek-V4-Flash-0731 TP4+EP4 (GPU 4-7)
    -> Kairyu L3 final answer

Embedding clients
    -> Kairyu L3 embeddings API (:8003; model embed-small)
        -> pinned offline FastEmbed MiniLM bundle (384 dimensions)
```

On the ensemble route the head role commits the public answer opening within
a small-prompt Qwen TTFT (~0.3 s measured at c1), so the product's semantic
TTFT (first public `content` token) is gated at **<= 2x the DeepSeek L1
direct row at the same concurrency** while both tracks run behind the
committed opening (DTO-D3, inherited from ECO-D4). The non-thinking direct
routes answer from t=0 and share the gate; the thinking direct routes pay a
deliberate think tax before their first public byte and are reported, not
gated.

## Routing: one Qwen judge picks exactly one of five routes (DTO-D13)

**In one sentence:** every request is first shown to a small, fast Qwen call
(the *route judge*), which answers with one of five labels; the profile named
by that label is the **only** thing that runs for the request — the other
four routes do not execute at all.

How a request flows:

1. **Judge** — one non-thinking Qwen3.8 call (greedy, ≤8 output tokens,
   5 s timeout) reads the latest user turn (a 4,000-character head+tail
   view) plus a one-line context flag (`tool calling yes/no; image attached
   yes/no`) and answers exactly one label: `QWEN`, `QWEN_THINK`, `DEEPSEEK`,
   `DEEPSEEK_THINK`, or `ENSEMBLE`.
2. **Dispatch** — Kairyu attaches the verdict to the request once (before
   preflight and admission, so all three agree) and builds the Conductor DAG
   of that profile only. Four of the profiles are a single model call; the
   fifth is the eleven-role ensemble below.
3. **Fallback** — if the judge times out, errors, or answers anything that is
   not exactly one offered label, the request runs the ensemble (`primary`),
   i.e. the quality-safe route.

| label | profile | what runs | thinking | sampling (fixed, DTO-D8) | max_tokens cap (vendor-official) |
|---|---|---|---|---|---|
| `QWEN` | `qwen_direct` | one Qwen3.8 call (`qwen_answer`) | no | T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5 | 131,072 |
| `QWEN_THINK` | `qwen_think_medium` | one Qwen3.8 call (`qwen_think_answer`) | fixed `medium` (spec `high`) | T=1.0, top_p=0.95, top_k=20 | 131,072 |
| `DEEPSEEK` | `deepseek_direct` | one DeepSeek call on the non-thinking pool (`deepseek_answer`) | no | T=1.0, top_p=0.95 | 393,216 (384K) |
| `DEEPSEEK_THINK` | `deepseek_think` | one DeepSeek call on the thinking pool (`deepseek_think_answer`) | caller's L3 effort (default `high`) | T=1.0, top_p=0.95 | 393,216 (384K) |
| `ENSEMBLE` | `primary` | the eleven-role dual-track DAG below | as before | as before | as before |

### What the judge optimizes — and what it is not

The judge is asked, in plain words, to pick *the fastest and cheapest route
that will still answer correctly and completely, escalating only when the
request genuinely needs it*. The routes are presented to it fastest → most
thorough, each with a short description of the kind of request it is for:

- `QWEN` — a small fast model answers correctly at once: chit-chat, short
  facts, rewording/translation/formatting, simple lookups, trivial one-liners,
  exact fixed outputs.
- `QWEN_THINK` — moderate reasoning a small model handles with medium
  thinking: short math/logic, small well-specified coding tasks, step-by-step
  explanations, routine agent tool-call turns.
- `DEEPSEEK` — needs frontier knowledge, breadth, or long-context
  comprehension but not deep deliberation.
- `DEEPSEEK_THINK` — hard problems where careful deliberation decides
  correctness (competition math, complex algorithms, multi-file coding or
  debugging, proofs, planning).
- `ENSEMBLE` — the hardest, highest-stakes open-ended work where comparing
  several independent approaches and auditing the merged answer materially
  improves quality; by far the slowest and most expensive route.

This is a **prompt-driven heuristic**, not a measured cost model or a learned
router: there is no latency/price table behind the decision, only the
instruction above and the per-route descriptions, which live in
`auto-max.yaml` (`profile_judge.choices[*].criteria`) and can be re-tuned
without code changes. The judge's verdict, the labels it was offered, and its
token usage are recorded in every trace (`profile_judge` classification
stage; `role profile: …` / `profile judge: …` notes), and `/routing`
reports the profiles and judge configuration, so the route mix can be
measured and the criteria adjusted from evidence.

### Contract details

- **Images**: image requests are offered only image-capable profiles (the two
  Qwen routes and the ensemble); the DeepSeek-only routes are withheld, and a
  verdict naming one is treated as unparseable (→ fallback).
- **Agent turns**: tool-calling, `response_format`, and plain-text
  structured-format turns are judged like every other turn. The direct
  routes have no head, so they answer headless by construction: the one
  publisher role writes the complete answer in the demanded reply format and
  emits the actual tool call when the conversation requires one; the
  ensemble keeps its `prompt_headless` behavior.
- **Sampling on the direct routes is fixed deployment policy** on the final
  unit: the caller's `temperature`/`top_p` are overridden for those routes,
  while the caller's `n`, `logprobs`, `response_format`, tools, and public
  `max_tokens` still apply. The route `max_tokens` is a **cap** min()'d with
  the caller's public allowance (Qwen3.8's official "Final Response:
  131,072"; DeepSeek-V4-Flash-0731's official "maximum output length 384K").
  From the Chat UI (default `max_tokens` 65536, unchanged by DTO-D13) the
  caller cap binds first; send a larger `max_tokens` over the API to reach the
  official caps.
- **Non-thinking calls are declared explicitly**: the judge and every
  effort-less role on a worker that accepts `enable_thinking` (the Qwen pool)
  are sent with `chat_template_kwargs: {enable_thinking: false}`. vLLM's Qwen3
  reasoning parser defaults to "thinking enabled" and drops a non-streamed
  answer that never emits `</think>` unless told otherwise — without this the
  judge's verdict arrived empty and every request fell back to the ensemble.
- **The ensemble is unchanged**: its roles, budgets (DeepSeek tiers
  8192/32768/65536, `internal_max_tokens` 65536, floor 256), head streaming,
  and audit loop are exactly DTO-D1..D12. The spec-level `budget {19, 2}`
  applies to every profile; a direct route spends one step plus the bounded
  empty-output re-dispatch.
- **The Qwen thinking route keeps the floor too (DTO-D15)**: the vLLM chat
  template opens Qwen's `<think>` span, so `qwen_think_answer` declares
  `reasoning_close_tag`/`reasoning_open_tag` with
  `reasoning_continuation: chat`. When a small caller cap (an agent harness
  sending `max_tokens: 8192`) is eaten by medium-tier deliberation, the one
  bounded re-dispatch continues the captured reasoning as a closed assistant
  turn (vLLM `continue_final_message`) with the reserved 256 public tokens
  instead of re-running the same request and 502-ing after 2x the cost.
- **Cost**: the judge is one serial small Qwen call on every turn (~0.2 s
  measured on this deployment for a short turn); the next GPU window
  measures its TTFT contribution on the ensemble route and the route
  distribution of the harness datasets.

## The ensemble profile: the dual-track DAG (DTO-D1)

The `primary` profile is the full eleven-role ensemble in three waves. It runs
only when the judge answers `ENSEMBLE` (or as the fallback); **inside the
profile no role is skipped** except the image-only stage on text requests.

```mermaid
flowchart LR
    subgraph W1["Wave 1 — dependency-free"]
        H["head (Qwen non-thinking, T=0.7)<br/>streams public opening at t=0"]
        DR["draft (Qwen thinking-medium, T=1.0)<br/>quick internal draft"]
        ID["image_description (Qwen thinking-medium)<br/>image requests only; skipped otherwise"]
        PO["policies (DeepSeek thinking, T=1.0)<br/>4 maximally different answer policies"]
    end
    subgraph W2["Wave 2 — two tracks in parallel"]
        A1["answer_1 (Qwen thinking-medium, T=1.0)<br/>follows POLICY 1"]
        A2["answer_2 (Qwen thinking-medium, T=1.0)<br/>follows POLICY 2"]
        A3["answer_3 (Qwen thinking-medium, T=1.0)<br/>follows POLICY 3"]
        A4["answer_4 (Qwen thinking-medium, T=1.0)<br/>follows POLICY 4"]
        CR["critique (DeepSeek thinking, T=1.0)<br/>critical analysis of the draft<br/>-> improved answer"]
    end
    subgraph W3["Wave 3 — merge + audit"]
        CO["synthesis (DeepSeek thinking)<br/>one better answer from 5 peer<br/>UNTRUSTED candidates; remainder is<br/>published after the audit"]
        AU["audit (Qwen thinking-medium, verifier)<br/>PASS -> stream; FAIL -> refine (<= 2)"]
    end
    ID -.image requests.-> PO
    ID -.image requests.-> CR
    ID -.image requests.-> CO
    ID -.image requests.-> AU
    PO --> A1
    PO --> A2
    PO --> A3
    PO --> A4
    DR --> CR
    H --> CO
    CR --> CO
    A1 --> CO
    A2 --> CO
    A3 --> CO
    A4 --> CO
    CO --> AU
    AU -.refine.-> CO
```

Role contracts:

| role | worker | what it does |
|---|---|---|
| `head` | Qwen TP1 | dependency-free; streams the committed public opening from t=0 (the TTFT gate) |
| `draft` | Qwen TP1 | quick complete internal draft — Track B's input, never published |
| `image_description` | Qwen TP1 | image requests only (`requires: image`; skipped entirely otherwise): precise textual description of the attached image for the text-only DeepSeek roles |
| `policies` | DeepSeek thinking | ONE call emitting POLICY 1..4, each a substantively different angle/method/priority set |
| `answer_1..4` | Qwen TP1 x4 | four policy-bound answers in parallel, one per replica; the policy list steers HOW, the request alone defines WHAT |
| `critique` | DeepSeek thinking | deliberates privately over the UNTRUSTED draft with critical thinking, then emits one improved complete answer |
| `synthesis` | DeepSeek thinking | the selected final unit: examines the five UNTRUSTED candidates (critique's refined answer + the four policy answers) as peers, verifies them, and writes one better answer; the remainder after the committed opening is published once the audit passes |
| `audit` | Qwen thinking (medium) | verifier on `synthesis`: judges opening + remainder as one public answer (correctness, completeness, consistency, reply format); first line `PASS`/`FAIL`; FAIL feedback drives up to 2 refinement rounds, after which the last attempt is published |

Design notes (see
[`docs/design/example-dual-track-orchestration.md`](../../docs/design/example-dual-track-orchestration.md)):

- The L2 DSL has no output-splitting mechanism, so each answerer receives the
  whole policy list and is bound to its own policy by prompt plus a distinct
  `seed_offset` (DTO-D2). The REQUEST and POLICY LIST blocks are
  byte-identical for deterministic policy binding, not for cross-replica
  cache reuse: the four answers run on independent replicas, and each
  replica prefills the newly generated policy list. With `shared_prefix`
  unset, placement follows session affinity and the queue-depth valve rather
  than `prefix_index`; only the leading REQUEST block may receive
  replica-local reuse from a preceding `head` or `draft` call.
- Every DeepSeek v4 flash role thinks (owner decision, DTO-D7): `policies`,
  `critique`, and `synthesis` all end their scaffolds with
  `<｜Assistant｜><think>` and deliberate privately before their public text.
  The TTFT gate stays on the non-thinking Qwen head, and each role's token
  cap now bounds its `<think>` span plus its output together.
- Qwen sampling and thinking are fixed per role in `auto-max.yaml`, never
  derived from the caller's request: `draft`, `image_description`, and
  `answer_1..4` declare `reasoning_effort: high` (T=1.0) — the medium tier
  under the L3 medium→high alias rule (DTO-D14) — so they think at medium
  effort: the example-local Qwen template (`qwen3.8-chat.jinja`) enables
  thinking for any explicit effort, prepends the medium reasoning preamble
  for `high`, and clamps `max` to `high` — while `head` (T=0.7) declares no
  effort and stays non-thinking so the public opening streams immediately.
- The caller's L3 `reasoning_effort` reaches the DeepSeek roles: `policies`,
  `critique`, and `synthesis` declare `reasoning_effort: inherit`,
  and a request without an explicit effort runs at the spec's
  `default_reasoning_effort: high`. The DeepSeek service's chat template
  (`deepseek-role-effort.jinja`) is a scaffold passthrough that splices the
  graded high/max reasoning preamble after `<｜begin▁of▁sentence｜>` into
  thinking-scaffold calls — all three DeepSeek roles — so `low` yields plain
  thinking and `high`/`max` strengthen every DeepSeek deliberation. The
  effort also grades the DeepSeek token budgets (DTO-D8, halved by
  DTO-D12): `policies` and `critique` declare
  `max_tokens_by_effort` `{low: 8192, high: 32768, max: 65536}` — half the
  vendor-recommended starting budgets, so the serial DeepSeek chain fits
  Terminal-Bench's 900 s per-turn agent envelope — bounding thinking +
  answer together, still clamped by `internal_max_tokens` (65536) and the
  request's public `max_tokens` — send a generous public `max_tokens` (the
  Chat UI default is 65536) or the higher tiers are clamped away.
- The audit verifier gates publication (DTO-D10; since DTO-D14 on Qwen at
  fixed medium effort with one 16384-token think+verdict cap): after every
  `synthesis`
  attempt the audit judges the committed opening plus the candidate
  remainder and answers `PASS` or `FAIL` on its first line; a FAIL verdict
  is fed back verbatim into a `synthesis` refinement (`max_refine_depth: 2`),
  and when the rounds are exhausted the last attempt is published. The
  remainder therefore reaches the stream only after the audit — the head
  still commits at t=0, so the TTFT gate is unaffected. Budget:
  `max_steps: 19` (10 generation units + 1 empty-final-output re-dispatch
  + 3 audit verdicts + 3 bounded inconclusive re-verifies + 2 refinements),
  `moa_samples: 0`.
- Per request the DeepSeek engine sees at most one in-flight call per wave
  (policies -> critique -> synthesis, plus refinement rounds — the audit
  verdict now lands on the Qwen pool, DTO-D14); the four concurrent Qwen
  roles spread one-per-replica through `queue_depth_threshold: 0`.

On agent turns (declared tools, a plain-text structured-format demand, API
`response_format`, `n>1`, or `logprobs`) the head is disabled for the call
(issue #495): there is no committed opening, `synthesis` renders its
`prompt_headless` body, and the whole answer comes from the synthesized
final unit in exactly the demanded reply format, using a trailing tool
result directly when one is present (issue #496). Every internal role states
an intended tool call in one sentence instead of emitting tool-call syntax —
the publisher emits the actual call, and the audit judges the call itself.
`n>1` requests skip the audit (one verdict cannot judge independent
choices) and publish the synthesized choices unverified.

Qwen fits one 96 GB card, so four independent TP1 replicas provide more
aggregate memory bandwidth and lower queueing TTFT than spreading one dense
model over PCIe with TP4 — and the width-4 policy fan-out executes one
proposal per replica. Each Qwen replica retains the checkpoint's vision
encoder. Kairyu validates one inline PNG/JPEG/WebP image up to 8 MiB and
2,097,152 pixels, passes it to every image-capable Qwen role, and gives
the text-only DeepSeek roles the same role-tagged conversation with explicit
image placeholders plus, on image requests only, the Qwen
`image_description` stage's verbatim textual description (`requires: image`
skips the stage entirely — no call, no budget step — on text requests,
DTO-D11). DeepSeek is sharded TP4+EP4 for capacity and retains the
measured eight-GPU example's FP8 KV, DSpark-5, SM120 fallbacks, prefix caching,
chunked batching, and full/piecewise CUDA Graphs.

The Qwen replicas carry the single-GPU winner
(`max_num_batched_tokens=32768`, `max_num_seqs=32`, FP8 KV, FP16
Gated-DeltaNet state, piecewise CUDA Graphs) with speculative decoding disabled.
MTP-3 remains a measured candidate: it improved the role-shaped c1/c4/c8 rows,
but this public deployment admits c16/c32 matrices and the matching Qwen TP1
saturation rows regressed. Re-enable it only after the deployed high-concurrency
envelope passes without regression (see [MEASUREMENTS.md](MEASUREMENTS.md)).
Qwen runs on official vLLM v0.23.0. DeepSeek intentionally stays on the measured
`aa0d513027` SM120 build because v0.23.0 does not support this checkpoint's
DSpark path and its generic MTP loader cannot load the 0731 MTP weights.

Kairyu exposes one public chat model, `kairyu-auto-max`, and one public
embedding model, `embed-small`. A chat request enters L3 once, then L2 borrows
the deployment-owned L1 pools through `engine_ref`; L2 never calls the public
L3 endpoint recursively. The CPU sandbox execution service from the previous
coding DAG stays deployed (compose service, `kairyu.yaml` executor registry,
`sandbox/` sources) but is no longer referenced by any role — execution
grounding can be re-adopted by a future DAG without redeploying.

In the same assistant response, completed L2/L1 stages are sent as
model-attributed `reasoning_content` and rendered by pinned Open WebUI in a
separate expandable internal-work item. The synthesis role's L3 final
answer alone is sent in `content`, so opening the item reveals each role,
attempt, worker, engine, model, and audit verdict without mixing
intermediate work into the answer.

The composed L1 services still use pinned vLLM. This proves the L3/L2/L1 object
boundary and UI behavior, but does **not** close the native-Kairyu L1 production
gate; native full-checkpoint correctness, recovery, soak, and performance gates
remain open. See
[`docs/design/example-layered-orchestration.md`](../../docs/design/example-layered-orchestration.md).

## Start

```sh
./run.sh
```

The command validates the exact eight-card inventory and NUMA affinity, pulls
the pinned Qwen vLLM release, reuses or builds the pinned DeepSeek SM120 image,
verifies or downloads both exact model revisions, builds Kairyu with the
pinned offline MiniLM bundle, waits for all eight services, verifies
`/routing`, sends a two-input embedding smoke, and prints:

```text
OpenAI API: http://127.0.0.1:8003/v1
Chat UI:    http://<outward-facing-host>:3000 (no authentication)
Chat model:      kairyu-auto-max (the only Chat UI model)
Embedding model: embed-small
```

Open WebUI listens on all host interfaces, requires no login, calls only
Kairyu L3, and is explicitly limited to `kairyu-auto-max`. The public
`/v1/models` endpoint additionally returns `embed-small`; the L1 pools are not
public IDs or Chat UI choices. The launcher validates that exact public
inventory, the explicit eleven-role dual-track primary DAG (including the
streamed head and the `{max_steps: 19, max_refine_depth: 2}` budget), the
four direct-route profiles and the Qwen route judge with its five choices
(DTO-D13), the `tier1`/`tier2`/`tier2-direct` engine bindings, and two
ordered finite 384-dimensional embedding vectors with positive usage before
printing the URL.

### Choosing the reasoning effort from the Chat UI

Requests default to `reasoning_effort: high` (`default_reasoning_effort` in
`auto-max.yaml`). To pick a different level per chat, open Open WebUI's Chat
Controls → Valves and choose **Reasoning Effort** from the dropdown
(`default`/`low`/`high`/`max`; `default` leaves the server default in
charge). The dropdown is a globally active Open WebUI filter
(`webui-reasoning-effort-filter.py`) that the launcher installs and
self-verifies on every `up`, and it owns the field: an explicit level
overrides any `reasoning_effort` set elsewhere — including the stock
v0.11.0 Advanced Params free-text input, which stays visible but never
reaches Kairyu — and `default` removes such values so the server default
applies. The selection is forwarded as the OpenAI-compatible
`reasoning_effort` body field.
The chosen level flows through the `inherit`-declared DeepSeek roles and
grades every DeepSeek deliberation (`policies`, `critique`, `synthesis`)
as well as their thinking+answer token budgets
(`max_tokens_by_effort`, DTO-D8/D12), and it is the effort of the
`deepseek_think` direct route (DTO-D13); the Qwen roles and routes keep
their fixed per-role declarations regardless of the UI setting. The API
equivalent:

```sh
curl -sS http://127.0.0.1:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"kairyu-auto-max","reasoning_effort":"max","messages":[{"role":"user","content":"Prove it."}]}'
```

`/v1/responses` clients (Codex among them) set the same knob through
`reasoning.effort`; OpenAI-style levels are normalized onto Kairyu's —
`minimal`/`low`→`low`, `medium`/`high`→`high`, `xhigh`/`max`→`max` — so a
Codex `xhigh` profile reaches the max tier.

The embedding model is the truthfully named
`sentence-transformers/all-MiniLM-L6-v2` FastEmbed deployment, not an alias for
OpenAI's `text-embedding-3-large`. Probe it directly with:

```sh
curl -sS http://127.0.0.1:8003/v1/embeddings \
  -H 'Content-Type: application/json' \
  --data '{"model":"embed-small","input":["first","second"],"encoding_format":"float"}'
```

Selecting this model from tau2's pinned `banking_knowledge/alltools` consumer
is tracked separately in `ytworks/kairyu-bench#5`; this deployment does not
mislabel MiniLM to satisfy tau2's historical OpenAI model default.

All persistent state is bind-backed below `/mnt/nvme`:

- Qwen weights reuse `/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/models`.
- DeepSeek's external Docker volume is verified to bind
  `/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-8gpu`.
- Four independent Qwen compilation caches, the DeepSeek compilation cache,
  and Open WebUI data live below
  `/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4-8gpu/`.

`NVME_STORAGE_ROOT` may select a different root only when it is still under
`/mnt/nvme`; non-NVMe roots fail closed. `VERIFY_MODEL=1 ./run.sh` rehashes both
checkpoint trees. Lifecycle commands are `./run.sh up`, `./run.sh status`,
`./run.sh logs`, and `./run.sh down`.

## Serving verification

```sh
./verify.sh list
./verify.sh serving-auto-max
./verify.sh serving-auto-max-coding
```

`serving-auto-max` records the generic-workload product serving matrix and
proves, for every request, the route judge classification stage and exactly
one profile's final unit; requests the judge sends to the ensemble must
additionally trace the head/synthesis public stream, every internal stage,
and the audit verdict (DTO-D13). Each row writes `routes.json` (route
distribution, per-route TTFT p50, judge latency p50).
`serving-auto-max-coding`
runs a deterministic self-contained Python-task dataset at c1/8/16/32,
measures the paired DeepSeek-direct row on the same dataset through the
loopback L1 endpoint, and
**fails unless the product's semantic TTFT p50 over the TTFT-gated routes
(ensemble, `qwen_direct`, `deepseek_direct`) stays within 2x the direct
row** (pinned `example.json` denominators are the fallback ceiling); samples
the judge sent to a thinking direct route are reported per route but not
gated, and a row with no gated-route sample records `not_applicable` in
`ttft-gate.json`. The
last green runs are the dated 2026-08-25 section of `MEASUREMENTS.md`
(`20260825T161729Z` coding + `20260825T173343Z` generic, both gates green
on the DTO-D8..D14 served config): the judge routed every coding request
to `qwen_think_medium`, so all coding TTFT rows record `not_applicable`,
and the generic matrix passed its route-aware stage validation at every
concurrency. Every older section predates this config and does not
transfer. ChatUI continues
to call only Kairyu
L3. Raw artifacts go to the configured NVMe
`verification-results/<UTC-run-id>/` directory. Model and product evaluations
are invoked explicitly through `python -m evals`; coding accuracy versus
frontier APIs is owned by the external `kairyu-bench` repository.

## Reproducibility pins

- Qwen revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Qwen tree SHA-256: `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- DeepSeek revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- DeepSeek tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- Qwen vLLM release/source: `v0.23.0` /
  `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- Qwen vLLM image digest:
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- DeepSeek vLLM source: `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2`
- DeepSeek vLLM image digest:
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Set `KAIRYU_RESPONSES_COMPACTION_SECRET` to at least 32 random bytes before
`docker compose up` (e.g. `export KAIRYU_RESPONSES_COMPACTION_SECRET=$(openssl
rand -hex 32)`); Codex remote-compaction tokens on `/v1/responses` are sealed
with it, and a stable value keeps them valid across gateway restarts.
Override API/UI/tokenizer-oracle ports with `API_PORT`, `CHAT_UI_PORT`, and
`DEEPSEEK_L1_PORT`. Both L3 endpoints bind all host interfaces by default, so
the API and UI remain reachable through both `127.0.0.1` and the outward-facing
host address. The launcher discovers that address for its printed URLs; set
`PUBLIC_HOST` when clients must use a DNS name, public NAT address, or reverse
proxy. Kairyu's L3 API and the UI are intentionally unauthenticated, so restrict
ports 8003 and 3000 at the firewall or place appropriate TLS/access controls in
front of them when exposure beyond a trusted network is not intended. Set an
explicit bind address when either endpoint must be restricted. Override the
two L1 images independently with `QWEN_VLLM_IMAGE` and
`DEEPSEEK_VLLM_IMAGE`; non-default overrides must already exist locally.

See [MEASUREMENTS.md](MEASUREMENTS.md) for the historical runtime-selection
and serving-performance analysis.

# deepseek-v4.1-openjev-verified-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. DeepSeek-V4.1
DP6/EP6 on GPUs 0-5 (image `sha256:119afb09…`, the six-GPU example's SM120
overlay), OpenJev 0.5.1 (`sha256:65f88680…`, unmodified) on GPU 6 and GPU 7.
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-openjev-verified-8gpu/`
(`calibration/`, `results/`).

## Readiness (2026-10-01)

`run.sh up` probes: every DeepSeek DP rank answers the `{"answer": 323}`
json_schema probe in thinking and chat mode; each OpenJev replica answers a
two-question System One read (billing yes, weather no); one verified request
returns `kairyu_verification`.

## Jev input/output (2026-10-01)

OpenJev (Jev wire API) builds a system prompt "Answer a fixed set of
questions about the state the user provides", lists each question as
`Question qN: <instructions>` with `yes: <criteria.true>` / `no:
<criteria.false>`, sends the state as the user turn, and reads one label
token per question from a 64-token canvas (about 10 questions per read;
larger sets are split into parallel reads; uncertain slots re-read 3 times).

Request form A/B on all 1,129 InFoBench expert labels (OpenJev only, the
model answers as given):

| Form | AUROC | acc@0.5 | Brier | accepted p>=0.9 | violations |
|---|---:|---:|---:|---:|---:|
| free-text state, declarative statement, no criteria | 0.814 | 0.840 | 0.141 | 894 | 98 (11.0 %) |
| **JSON state, `{question, requirement}` object, yes/no criteria** | **0.830** | **0.850** | **0.133** | 860 | **80 (9.3 %)** |
| InFoBench question as-is | 0.821 | 0.841 | 0.139 | 876 | 88 |
| any of the above with `think: 512` | 0.789-0.802 | | worse | | more |

The second form is what m1 D8's conversion produces.

Two exchange defects found on the GPUs and fixed:

- The extractor read Kairyu's L2 `{query}`, whose wrapper instructs the
  answer writer ("Return only the assistant response body", ...); DeepSeek
  extracted those sentences as the user's instruction units, so curation
  replaced real requirements with padding and the probe ended at
  `refinement_limit`. Roles that analyse the request now read
  `{conversation}` (the role-tagged messages only). After the fix the probe
  "Name the capital of France in one word, then explain why in one sentence"
  extracts R1-R4 (one word / order / explains why / one sentence) and is
  guaranteed on attempt 1 (162 s, 26,719 internal output tokens).
- DeepSeek-V4.1 thinks unless `enable_thinking: false` is sent (120-340
  reasoning tokens on a "non-thinking" call, occasionally an empty answer).
  The pool now allows that kwarg, so the state builder runs in chat mode.

## tau_hi calibration (2026-10-01)

`./verify.sh calibrate`, full production checklist path (DeepSeek rewrites
each InFoBench question as a condition, the state builder lists claims,
OpenJev reads through the two-replica System One backend). Split by
instruction (25 / 25, seed 20261001).

| | labels | violations | AUROC |
|---|---:|---:|---:|
| all | 1,129 | 239 | 0.850 |

| tau | requirements accepted | violation rate | 95 % upper bound | answers with every requirement >= tau | of which violated |
|---:|---:|---:|---:|---:|---:|
| 0.99 | 74 % | 8.8 % | 10.6 % | 123 / 249 | 31 |
| 0.995 | 71 % | 7.5 % | 9.2 % | 113 / 249 | 25 |
| 0.999 | 65 % | 5.9 % | 7.5 % | 97 / 249 | 20 |

alpha = 0.05 is reachable only at p = 1.0 exactly (about 10 % of
requirements). Variants measured on the same cached DeepSeek outputs
(violations among p >= 0.999):

| Variant | AUROC | accepted | violations | upper bound |
|---|---:|---:|---:|---:|
| production form | 0.851 | 736 | 46 | 0.079 |
| state = conversation + answer | 0.826 | 651 | 45 | 0.088 |
| state = answer only | 0.808 | 621 | 46 | 0.094 |
| one question per read | 0.826 | 606 | 47 | 0.098 |
| stricter yes/no criteria | 0.842 | 647 | 40 | 0.080 |
| all three combined | 0.803 | 287 | 20 | 0.100 |
| `steps: 4` | 0.846 | | | |
| `samples: 8` | 0.851 | | | |
| atomized requirements, min | 0.804 | 475 | 31 | 0.087 |

Label noise floor (two expert annotators per answer, 1,123 pairs): the
annotators disagree on 10.2 %; when annotator 1 says "satisfied" the official
label says "violated" 9.1 % of the time (annotator 2: 10.0 %). Owner decision:
alpha = 0.10.

Result (alpha = 0.10, 95 % confidence): **tau_hi = 0.9966**.

| Half | accepted | violations | rate | upper bound |
|---|---:|---:|---:|---:|
| calibration | 392 | 29 | 7.4 % | 9.95 % |
| held-out | 398 | 25 | 6.3 % | 8.66 % |

Held-out answers: 54 / 125 pass every requirement; 10 of them carry at least
one labelled violation.

## Per-claim G1 calibration (2026-10-02, VCO-D11; computation and general removed by VCO-D12)

`./verify.sh calibrate-g1`: 600 human-labelled answers per claim kind, the
production state builder (DeepSeek, high effort, claims grammar) and the
production G1 questions read by both OpenJev replicas through `ChecklistRun`.
Split by problem / source document / Wikipedia page (seed 20261001). Raw rows:
`results/g1-calibration-20261001T161504Z.json`, `calibration/g1/tau.json`.

| Kind (data) | answers (violated) | AUROC | tau 0.9 acc / viol | tau 0.99 acc / viol | tau 0.999 acc / viol | calibrated tau | held-out |
|---|---|---:|---|---|---|---|---|
| G1-source (RAGTruth, any span) | 599 (237) | 0.694 | 173 / 41 | 157 / 37 | 141 / 33 | none | — |
| G1-computation (PRM800K phase 2, a -1 step; balanced) | 595 (299) | 0.704 | 198 / 78 | 185 / 74 | 178 / 69 | none | — |
| G1-general (FEVER dev, REFUTES) | 594 (293) | 0.893 | 296 / 53 | 251 / 38 | 197 / 22 | 0.99926 (89 acc, 4 viol, ub 0.0999) | 99 acc, 15 viol, ub 0.224: FAIL |

Answers passing every G1 kind at p >= 0.99: RAGTruth 138 (22 violated),
PRM800K 196 (75), FEVER 254 (38). Accepted RAGTruth violations are mostly
"Baseless Info" (45 spans) rather than "Conflict" (9): true but unsourced
additions that G1's "well-established knowledge" accepts. Accepted PRM800K
violations list the erroneous step as a claim and OpenJev calls it
supported; FEVER violations include "Fringe debuted in 2011" at p 0.9996.
Outcome (owner option A): the G1 questions are advisory (threshold 0).

Cost: 1,800 state-builder calls, 10,557 / 8,404 / 422 median output tokens
(RAGTruth / PRM800K / FEVER; 91 % thinking), p50 142 / 110 / 8 s at 32 in
flight, 74 tok/s per request, 2,370-2,500 tok/s DeepSeek generation, DSpark
acceptance 57-61 %; about 85 minutes in all. 12 state-builder outputs (0.7 %)
were truncated JSON (runaway newlines) and are excluded.

## Two-stage extraction and latency (2026-10-02, VCO-D8 amendment 2, VCO-D12)

Implicit conditions, extraction plus OpenJev only (no answer generation;
dev set of 12 + 4 controls written before tuning; the gate's 20 + 10):

| Configuration | gate-set recall | dev-set recall | controls (implicit kept) | implicit kept per request |
|---|---:|---:|---:|---:|
| one prompt, situational guidance | 0.875 | 0.917-0.958 | 0.0 | 2.2 |
| two-stage | 0.900-0.925 | 0.917-0.958 | 0.0-0.2 | 4.7 |
| two-stage, at most four | 0.925-0.950 | 1.000 | 0.0-0.4 | 2.6-2.75 |

Full-DAG implicit gate runs before the split: recall 0.525 (first run),
0.600, 0.675; re-extraction dropped implicit conditions. The InFoBench
requirements gate on the one-prompt build: gold recall 0.867, p50 471 s
(stated conditions 4.5 per request vs 5.1 without implicit ones).

Latency, 8 InFoBench requests at c8 with traces (seconds; mean per call):

| Configuration | p50 / max | extract | implicit | answer | state builder | guaranteed |
|---|---|---:|---:|---:|---:|---:|
| two-stage | 492 / 587 | 88 | 112 | 34 | 92 (10,491 tok) | 1 / 8 (3 `budget`) |
| + state builder lists source / action claims | 355 / 455 | 82 | 125 | 31 | 43 (4,914 tok) | 2 / 8 (1 `budget`) |
| + step budget 24 | 297 / 432 | 89 | 110 | 23 | 44 (5,109 tok) | 3 / 8 |

The fastest guaranteed request took 166 s (one attempt).

## GPU gates on the final code (2026-10-01, `5b455dd8` plus the G3/report fixes)

Raw rows: `results/{structured,requirements,serving}-2026100*T11*Z.json`,
`results/serving-20261001T124829Z.json`. Every request routes through the
verified DAG (`multi_agent`); tokens are internal orchestration totals
(DeepSeek + System One).

| Gate | Result |
|---|---|
| `l1` (readiness) | PASS: JSON-grammar probe on every DP rank, thinking and chat; System One on each OpenJev replica |
| `repair` (16 constraint requests, c8) | PASS: 16/16 answered, 13 guaranteed, 5 repaired, 0 guaranteed answers violate their stated constraint (independent check); p50 65 s, p95 186 s |
| `structured` (caller json_schema) | PASS: schema-valid and guaranteed |
| `fallback` | PASS: one OpenJev down still guaranteed; both down 200 + `judge_unavailable`; recovered |
| `requirements` (40 InFoBench, c8) | PASS: gold-question recall 0.901 on all 40; 11 guaranteed, 36 repaired; p50 209 s, p95 421 s; 772 output tok/s aggregate |
| `serving` (InFoBench, c1/c4/c8/c16) | PASS: every request answered with a flag (table below) |
| `browser-smoke.sh` | PASS: badge "Guaranteed", answer "red, blue, yellow" |

Serving (each row: requests / ok / guaranteed / repaired / mean attempts /
latency p50 / p95 / per-request internal input / output tokens (median) /
per-request output tok/s median (range) / aggregate output tok/s / req/min):

| c | n | ok | guaranteed | repaired | attempts | latency p50 / p95 s | tokens in / out | tok/s per request | aggregate tok/s | req/min |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| c1 | 8 | 8 | 2 (25 %) | 8 | 2.75 | 141.1 / 221.1 | 11,134 / 27,416 | 180.9 (160.7-206.2) | 185.9 | 0.39 |
| c4 | 16 | 16 | 5 (31 %) | 15 | 2.69 | 153.5 / 293.5 | 11,158 / 21,982 | 141.9 (121.7-153.6) | 525.7 | 1.38 |
| c8 | 16 | 16 | 6 (38 %) | 11 | 2.31 | 213.4 / 346.5 | 10,650 / 24,328 | 110.9 (86.7-226.1) | 808.8 | 1.95 |
| c16 | 32 | 32 | 8 (25 %) | 27 | 2.62 | 260.2 / 443.4 | 13,152 / 23,653 | 87.9 (72.1-130.6) | 1133.0 | 2.91 |

DeepSeek L1 during the gates (30-60 s samples of vLLM `/metrics`, six DP
ranks): 175 tok/s generation at c1, 590-760 tok/s at c4-c8; DSpark
acceptance 51-63 %; prefix-cache hits 15.5 % of prompt tokens.

Findings (open):

- Guarantee rate on long InFoBench requests is 25-38 %; most unguaranteed
  answers fail requirements scored 0.7-0.99 against tau_hi 0.9966, or G1
  (per-claim groundedness, minimum over claims, not separately calibrated).
  Short constraint requests reach 81 %.
- Latency is dominated by internal tokens (thinking-high extraction and
  draft, repairs averaging 2.3-2.8 attempts): p50 141 s at c1, 260 s at
  c16. No latency target has been agreed yet.
- Defects found and fixed during the gates: G3 read JSON keys and code
  string literals as quotations; verbatim matching failed on typography;
  a report omitted requirements an early check failure left unread; the
  extractor read Kairyu's answer-contract wrapper; V4.1 thinks unless
  `enable_thinking: false` is sent.

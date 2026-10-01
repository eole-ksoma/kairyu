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

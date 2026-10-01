# Checklist-Verified Answers (DeepSeek-V4.1 six-GPU + OpenJev x 2)

Status: **Accepted 2026-10-01; implemented and GPU-verified** (every gate PASS; see
`examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md`).
Applies to: `examples/deepseek-v4.1-openjev-verified-8gpu/`. Framework
mechanisms: m1 D8 (checklist verifiers) and the m11 D8 replica amendment.

## Goal

Return an answer with a guarantee flag that means "the request was met":
every requirement of a checklist that is necessary, sufficient and mutually
exclusive (MECE) with respect to the request passed. Otherwise return the
best available answer with the flag off and the reason.

Roles (owner, 2026-10-01):

- DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU
  example's L1): requirement extraction, generator, state builder, repair.
- OpenJev (DiffusionGemma 26B-A4B), one replica each on GPU 6 and GPU 7,
  read through System One: requirement confirmation and per-requirement
  judgments.
- Kairyu L2: Validator, Conductor, transitions, fallback.

## Decisions

### VCO-D1 — DAG

`verified.yaml` (DSL only; no Python orchestration in the example):

1. Wave 1: `extract` (DeepSeek, thinking high, JSON grammar) and
   `generator` (DeepSeek, caller effort, sees only the conversation; a caller
   `response_format` constrains it) run in parallel. `extract` is verified by
   `requirements_check` inline (VCO-D2).
2. Wave 2: `answer` is seeded with the generator's draft (no model call) and
   verified by `checklist`: Validator checks, then `state_builder` (DeepSeek,
   non-thinking, JSON claims with verbatim evidence) re-run on the attempt,
   then OpenJev reads, then the Conductor rule. A FAIL is repaired by
   DeepSeek through `answer.refine_prompt` with the failing items, at most
   twice, and the Validator runs again on the repaired version.
3. A rules router with `multi_step_markers: 0` sends every request to the
   DAG: short requests are not exempt from the guarantee.

### VCO-D2 — Requirement set

The extractor splits the request (system/developer instructions and the
latest user message) into instruction units U1..Un and writes conditions
R1..Rm, each with a proposition, a kind (`deterministic` with one primitive
from the shared check library, or `semantic`), and its source units.
Identifiers and labels are not instructions; an action that cannot be
performed here becomes "states that it cannot be performed".

`requirements_check` confirms it:

- **Sufficiency**: deterministic coverage (every unit id is cited) and one
  `noul` read per unit ("an answer meeting the conditions citing Ui meets
  Ui"), threshold 0.5.
- **Necessity**: one read per condition ("its source units ask for it").
  Threshold 0 for the verdict; curation drops conditions with p < 0.5.
- **Exclusivity**: one read per pair of conditions sharing a source unit
  ("they require the same thing", expect no), threshold 0.5.

A coverage gap or a duplicate re-extracts once with the problems listed.
Afterwards curation drops low-necessity conditions, merges remaining likely
duplicates (first kept, sources united, propositions joined) and pads every
uncovered unit with "the answer responds to this part of the request: <unit>".

Request-independent requirements have their own ids and never overlap the
extracted ones: G1 groundedness (material-based claims quote the material
verbatim — `G1-excerpts`, deterministic — and every claim is supported —
OpenJev per claim, minimum), G2 execution claims (an "action" claim's
evidence must appear in the conversation, i.e. its tool call and result),
G3 quotations in the answer appear in the conversation.

### VCO-D3 — Validator and Conductor

The Validator is the deterministic half of `checklist`: extracted
`deterministic` conditions (an unusable primitive falls back to an OpenJev
read of the proposition), G3 before the state builder, G1-excerpts and G2
after it. Any violation goes to repair without an OpenJev read. The Conductor
passes an attempt only when every item has p >= tau_hi (a check is 1 or 0).
Because the checklist was confirmed sufficient, every item passing means the
request was met.

### VCO-D4 — tau_hi (amended: alpha = 0.10)

tau_hi is calibrated on InFoBench's expert annotation (50 instructions x 5
models, 1,129 labelled requirement judgments). Each decomposed question is
rewritten by DeepSeek as a condition statement and judged through the
production checklist code (same state and question conversion). tau_hi is
the smallest threshold whose accepted judgments have a one-sided 95 %
Clopper-Pearson upper bound on the violation rate <= alpha on the
calibration half (split by instruction, seed 20261001). The held-out half is
reported unchanged (`calibrate.py`).

Result: tau_hi = 0.9966. Calibration: 392 accepted, 29 violations, upper
bound 0.0995. Held-out: 398 accepted, 25 violations (6.3 %), upper bound
0.0866. 54 of 125 held-out answers pass every requirement; 10 of them carry
at least one labelled violation.

Why alpha = 0.10, not 0.05 (owner decision, 2026-10-01): the labels are
noisier than 0.05. When one expert annotator says a requirement is
satisfied, the official label says violated 9.1 % / 10.0 % of the time (the
two experts disagree 5.9-6.6 % of the time). At alpha = 0.05 only p = 1.0
exactly qualified (about 10 % of requirements). Request-form variants did
not lower the violation rate: smaller states, one question per read,
stricter criteria, `steps`/`samples`, atomized requirements, and `think`
all scored equal or worse AUROC (MEASUREMENTS.md).

Limitation: InFoBench has no source material, so G1 uses the same tau_hi
without its own calibration.

### VCO-D5 — Fallback

- Repair limit: the newest attempt whose deterministic checks passed (else
  the draft), `guaranteed: false`, `reason: refinement_limit`.
- Judge unavailable (both OpenJev replicas down or overloaded, in either
  verifier): the generator's draft as-is, `reason: judge_unavailable`.
- Checklist not judgeable (no parseable list, state above 160,000 characters
  for OpenJev's 65,536-token window): the draft, `reason:
  checklist_unavailable`.

### VCO-D6 — Output

The chat response carries `kairyu_verification` (m1 D8) next to the answer;
the answer text is never altered. The example's answer page (nginx on
:3013, Kairyu's API on the same origin) shows the badge, the reason, and the
requirement table with each p.

### VCO-D7 — Jev routing and the think route (2026-10-01)

Two public models: `kairyu-verified` lets Jev (m1 D9) choose between the
verified DAG (VERIFIED) and `deepseek_think`, one thinking DeepSeek answer
(THINK); `kairyu-verified-always` keeps every request on the verified DAG and
backs the answer page and the guarantee gates. VERIFIED criteria: difficult
or specialised questions, many requirements, earlier answers corrected,
a frustrated or pressured user, high-stakes domains, explicit demands for
correctness or sources, output that will be published, sent, signed or
executed, and long multi-part deliverables. Owner decision: accuracy first
(`prefer: VERIFIED` at tau_route; missed accuracy-critical requests < 10 %).
An unjudgeable route falls back to `deepseek_think`, since the verified DAG
could not be judged either. The publisher's private reasoning is withheld
by Kairyu's multi-stage contract on both routes.

### VCO-D8 — Implicit requirements (2026-10-01)

The extractor adds conditions the situation clearly presupposes (origin
`implicit`, attached to the units they serve) next to the stated ones. Each
implicit condition gets its own necessity read ("does the user expect this
even though they did not say it?") and curation drops it below 0.5, so it
enters the guarantee only when Jev judges it expected. Coverage and
exclusivity checks apply unchanged; sufficiency stays on the stated units.

### VCO-D9 — One effort for every DeepSeek step (2026-10-01)

Owner requirement: whatever the route, every DeepSeek role (extract,
generator, repair, state builder, think answer) runs at the caller's
`reasoning_effort` (API field or the Open WebUI dropdown), default high
(75). Open WebUI offers both models and shows the guarantee in the folded
"Verification" section.

### VCO-D10 — Review amendment (PR #616, 2026-10-01)

A requirement checklist still failing sufficiency after re-extraction and
curation blocks the guarantee (`requirements_unconfirmed`); deterministic
conditions are never merged; execution claims need tool-result evidence;
`n > 1` is refused on the verified models.

## Limitations

- A guaranteed answer is not streamed before its checklist finishes (time to
  first token is the whole pipeline).
- Tool-calling turns are not this example's surface: the published answer is
  the generator's text; tools in the request are context only.
- The routing set is author-labelled with clear-cut categories; borderline
  requests are not measured by it.
- Thresholds other than tau_hi (0.5 for necessity, sufficiency and
  exclusivity) are defaults, not calibrated.

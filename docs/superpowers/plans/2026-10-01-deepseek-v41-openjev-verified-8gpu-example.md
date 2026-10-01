# Checklist-verified answers example (DeepSeek-V4.1 six-GPU + OpenJev x 2)

Status: **Approved 2026-10-01; implemented; GPU gates pass.**

Accepted scope (owner, 2026-10-01):

- L1 reuses the existing examples: DeepSeek-V4.1-Flash DP6/EP6 on GPUs 0-5
  (`deepseek-v4.1-flash-6gpu`) and OpenJev DiffusionGemma
  (`openjev-diffusiongemma-26b-1gpu`), here one replica each on GPU 6 and 7,
  used only through System One (no think-first overlay).
- The owner's flow: requirement extraction (DeepSeek) with requirement
  confirmation (OpenJev + deterministic checks, one re-extraction, then
  generic padding), generator in parallel (request only), Validator
  (rule-based), state builder (DeepSeek), OpenJev judgment, Conductor
  (every p >= tau_hi), repair (DeepSeek, at most 2), guaranteed result,
  fallback (repair limit / judge unavailable), output with the flag.
- Framework: add the missing shared mechanisms in `kairyu/`, minimal and
  generic; requirement drop/merge/pad policy stays in the example YAML
  (owner chose "汎用部品として最小限追加").
- tau_hi calibrated on InFoBench expert labels with alpha = 0.10 (amended from 0.05; label noise floor).
- The guarantee flag is a dedicated response field plus a UI display.

## Framework changes (m1 D8, m11 D8 replica amendment)

1. `systemone_ref` workers and checklist verifiers calling System One.
2. Threshold verdicts with failing-item feedback on the existing refine loop.
3. Deterministic check primitives (`kairyu/orchestration/checks.py`).
4. `sampling.response_format` for internal roles (`inherit` for the caller's).
5. `seed_from`, `refine_prompt`, inline-bound generation roles,
   per-verifier `max_refinements`, `on_exhausted: latest_checks_passed`.
6. `on_unavailable: publish_unverified` and `kairyu_verification`.
7. System One `base_urls` (several replicas, one retry on another replica).

## Example

`examples/deepseek-v4.1-openjev-verified-8gpu/`: `compose.yaml`,
`kairyu.yaml`, `verified.yaml`, `example.json`, `control.py`, `run.sh`,
`verification.py`, `verify.sh`, `calibrate.py`, the SM120 overlay files
(copied), `playground/`, `playground-smoke.mjs`, `browser-smoke.sh`,
`README.md`, `MEASUREMENTS.md`. Ports: API 8013, answer page 3013, DeepSeek
L1 loopback 8014.

## Tests

- Framework: `tests/unit/test_conductor_checklist.py` (A->C paths: guarantee,
  deterministic FAIL -> repair -> PASS, exhaustion -> latest checks-passed,
  judge down -> unverified draft, requirement-list curation),
  `tests/unit/test_checks.py` (source-backed primitives),
  `tests/unit/test_dsl.py` (load-time rejection),
  `tests/server/test_systemone_api.py` (replica retry, all down),
  `tests/server/test_orchestration_usage_trace.py` (response field, unary
  and stream).
- Example: `tests/unit/test_deepseek_v41_openjev_verified_example.py`.

## GPU gates (`./verify.sh <gate>`)

`l1`, `calibrate`, `requirements` (InFoBench gold-question recall >= 0.90),
`repair`, `structured`, `fallback`, `serving` (c1/c4/c8/c16), and
`./browser-smoke.sh`. Results: the example's `MEASUREMENTS.md`.

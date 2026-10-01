# Checklist-verified answers: DeepSeek-V4.1 (6 GPUs) + OpenJev (2 GPUs)

Every answer comes back with a guarantee flag. The flag is on only when the
answer passed every requirement of a checklist that was confirmed necessary,
sufficient and mutually exclusive with respect to the request; otherwise the
best available answer is returned with the flag off and the reason.
Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D1..D6);
framework mechanisms: m1 D8 and the m11 D8 replica amendment.

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). OpenJev (DiffusionGemma 26B-A4B NVFP4) on GPU 6 and GPU 7, published image unchanged, read only through System One. |
| L2 | `verified.yaml`: extraction, requirement confirmation, generator, Validator, state builder, OpenJev reads, Conductor, repair (at most 2), fallback. |
| L3 | One public model `kairyu-verified`; `kairyu_verification` on every answer; the answer page on :3013. |

```text
request ──► extract (DeepSeek, JSON) ──► requirements_check (checks + OpenJev) ─┐ re-extract once
        └─► generator (DeepSeek, request only) ─► draft                         │
                                                   ▼                            ▼
             Validator (checks) ─FAIL─► repair (DeepSeek, ≤2) ─► Validator ...   checklist
                 │PASS                                                         (curated)
                 ▼
             state builder (DeepSeek) ─► OpenJev reads ─► Conductor: every p ≥ τ_hi?
                                              │ unavailable        │ yes        │ no
                                              ▼                    ▼            ▼
                                     draft, unverified      guaranteed     repair / limit
```

## Run

```sh
./run.sh            # preflight, images, checkpoints, compose up, readiness probes
./verify.sh list    # GPU gates; evidence lands in model-volumes/<env>/results/
./browser-smoke.sh  # the answer page in a real browser
./run.sh down
```

`DEEPSEEK_MODEL_SEED` / `OPENJEV_MODEL_SEED` may name local copies of the
pinned checkpoints below `/mnt/nvme`; they are hard-linked and re-hashed
against the pins before serving. The DeepSeek image is the six-GPU example's
SM120 overlay (same Dockerfile and patch, copied here; pinned image ID).

## Using it

```sh
curl -s http://127.0.0.1:8013/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "kairyu-verified",
  "messages": [{"role": "user", "content": "List three primary colors, comma-separated."}]
}' | jq '{answer: .choices[0].message.content, verification: .kairyu_verification}'
```

`kairyu_verification`:

- `guaranteed: true` — every requirement passed; `requirements[]` lists each
  one with `p`, its source instruction units and kind.
- `guaranteed: false` with `reason`:
  - `refinement_limit`: two repairs did not pass; the newest version that
    passed the deterministic checks is returned.
  - `judge_unavailable`: neither OpenJev replica answered (down or
    overloaded); the generator's draft is returned as-is.
  - `checklist_unavailable`: the checklist could not be built or exceeded
    OpenJev's input window; the draft is returned.
  - `requirements_unconfirmed`: the requirement checklist still left an
    instruction unit uncovered after re-extraction and curation, so passing
    it would prove nothing; the answer is returned without a guarantee.

The answer page (`http://<host>:3013`) shows the badge, the reason and the
requirement table; internal stages are folded below the answer.

## Limits

- A verified answer is not streamed before its checklist finishes.
- Tool-calling turns are not this example's surface (tools are context only).
- Only τ_hi is calibrated (InFoBench, α = 0.10); the 0.5 thresholds of the
  requirement confirmation are defaults.

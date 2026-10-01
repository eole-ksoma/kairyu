# Measurements: OpenJev DiffusionGemma on one GPU (think = 512)

Status: **GPU-verified (2026-10-01).** Every gate passed on one RTX PRO 6000
Blackwell Server Edition (GPU 0) with served config
`eafbfa94d9376e069a2f39b32b303d13d00c83b7af049d5237ff0c22164893ba`, and
again after both review rounds with served config
`6b5a1807d4abc3a167ec021e1783a301350b4470f2a627b283b40359ed635e2e`
(overlay `sha256:5e8e2e08…`; see "Second review-fix rerun" below). The CPU
evidence below was recorded first, on the development machine.

## Pins

| Item | Value |
|---|---|
| OpenJev source | `dcd20947b5ddad5be4a8f5aed6aa6dd245653823` (package 0.5.0) |
| Base image | `razorback16/openjev:0.5.1@sha256:65f88680e0093c9229d8cc55c0d2f279db27aba78a7ce74ea762b8f2da82d47d`, image ID `sha256:b219be87fdde14896b0029edabe5586b2e99c89380784a9ec73f1405a0453757` |
| vLLM in the base | `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`, with OpenJev's logprob-id cap 512 and vision prefix-LM patch (from the image's build history) |
| Overlay image ID | `sha256:5e8e2e0838c29fecaa868dbd5b41ae123a0ecadf1f7db030725afc404f0aff60` (second review round; earlier passes used `sha256:1c14bdf6…` and `sha256:46530fa7…`) |
| Kairyu gateway image (final runs) | `sha256:69a1ece39192531c04aa6ffd6aa5abb45e8a9b75a18cc8bcb97ac4e482e5e288`, built from this branch (`/v1/systemone`, m11 D8) |
| Playground | `nginx:1.29-alpine@sha256:5616878291a2eed594aee8db4dade5878cf7edcb475e59193904b198d9b830de` |
| Checkpoint | `nvidia/diffusiongemma-26B-A4B-it-NVFP4@ec4ff3df205028f4e81c954c2227f9312b3ec2ea` |
| Checkpoint tree | `e38de908a6b053b2a2781c37251afa41776d26a98b050d1fe7badb19139ea9b7` |
| Chat template | stock `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` → edited `6c0f238239442d0712d48f12421e26b356c4d8f0175c1a2b57cddab426ee69bc` (was `536903c7…` before the GPU fix below) |

The checkpoint tree hash uses the same algorithm as `control.py`'s
attestation: the sha256 of the sorted `{path, size, sha256}` rows. It was
computed from the HF tree API's LFS sha256 values for the two safetensors
files and `tokenizer.json`, and from downloaded bytes for the 11 small
files. The first download on the GPU host re-hashes every file and fails
closed on any difference.

## CPU evidence (development machine, 2026-10-01)

`check_overlay.py` reproduces the template and patched-app evidence below.
It downloads the tokenizer files and the OpenJev source at the pinned
revisions; the run command is in its docstring.

**Template edit**, checked with the checkpoint's own tokenizer
(`tokenizer.json` sha256 `cc8d3a0c…`, transformers 5.12.1):

- All 14 conversations that do not end in an assistant message render byte
  for byte like the stock template, with thinking on and off. The cases are:
  a single user turn, system + user, tools, an image part, multi-turn with a
  stripped thought, a tool result, and Kairyu's flattened legacy transcript.
- Stock template, with `continue_final_message` on both prefills (with and
  without tools): `continue_final_message is set but the final message does
  not appear in the chat`.
- Edited template: the thought-pass prompt ends exactly at
  `<|turn>model\n<|channel>thought\n`, and the answer-pass prompt at
  `…\n<channel|>`.

**Patch** (`patch_openjev.py`) against OpenJev `dcd20947`:

- Both `api.py` anchors match once.
- A second application is refused.
- The patched package compiles.

**Patched OpenJev app** (the real lifespan and the checkpoint tokenizer,
against a local fake vLLM):

| Case | Result |
|---|---|
| Generator in use | `ThinkFirstGenerator`; startup template check passes; `<channel|>` is token 101 (one token; `<|channel>` is 100) |
| Request with `enable_thinking: false`, `reasoning_effort: none` | thought pass sent with thinking on, `max_tokens` 512, `stop_token_ids` [101]; answer pass prefill `<|channel>thought\n…\n<channel|>` with the caller's `max_tokens` 64 |
| Non-stream response | `reasoning` = thought, `content` = answer, usage 20 prompt / 11 completion, of which 9 are reasoning tokens |
| Stream | reasoning, then content, then `[DONE]`; generation slot released |
| Refused thought (upstream 400) | HTTP 400 `invalid_request_error` before the stream starts; slot released |
| Answer pass fails (upstream 500) | stream aborted without `[DONE]`; slot released |
| `OPENJEV_VLLM_ARGS` without the template | startup refused |

**Compose.** `docker compose config` renders the stack from `control.py`'s
environment with no warnings:

- every OpenJev setting comes from `example.json`;
- the OpenJev service has no host port, and its single GPU is `GPU_ID`;
- Kairyu is on `127.0.0.1:8010` and the Chat UI on `127.0.0.1:3010`.

**Tests.** The CPU tests are in `tests/unit/test_openjev_1gpu_example.py`.

**Correction (2026-10-01).** An earlier ad-hoc run of the patched-app check
reported `<channel|>` as token 9. That run loaded the tokenizer without
`tokenizer.json`, an LFS file it had not downloaded. With `tokenizer.json`,
the id is 101. The template rows are unaffected, because rendering is
string-level and both runs agree.

## GPU evidence (2026-10-01)

Host: 8× RTX PRO 6000 Blackwell Server Edition (SM120); the example uses
GPU 0 only. Run artifacts are under
`/mnt/nvme/kairyu/model-volumes/openjev-diffusiongemma-26b-1gpu/verification-results/`
(`final-*`, `final2-*`); each `run.json` records the served-config hash
above, the overlay image ID and the gateway's `kairyu.yaml` sha256
`797228c7b271…`.

### L1-0: overlay build and pin

- **First start failed, then fixed.** vLLM detects the template content
  format as `openai` and passes the assistant prefill as a list of text
  parts. The template edit covered only string content, so `strip_thinking`
  removed `<|channel>thought\n` and vLLM refused every
  `continue_final_message` request with HTTP 400 (`final message does not
  appear in the chat`). The startup check rendered only the string form, so
  it passed. Now both shapes render verbatim and the startup check and
  `check_overlay.py` cover both (8 prefill cases, 14 unchanged cases).
- After the fix: model attestation, the overlay's startup check, readiness,
  and `l1` pass with the pinned image.

### L1-1: generations in flight (adopted 32 + 8)

Mixed generic/coding prompts, 32 requests per row, answer cap 8192, through
Kairyu (`d4-inflight{8,16,32}-serving`):

| In flight / queue | c1 tok/s | c1 E2E p50 / p99 | c32 tok/s | c32 TTFT p50 / p99 | c32 E2E p50 / p99 | Failures |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 32 (OpenJev default) | 482.6 | 2.1 / 12.3 s | 1,303.5 | 6.71 / 17.25 s | 11.3 / 29.5 s | 0 |
| 16 / 24 | 492.1 | 2.1 / 8.6 s | 1,448.6 | 1.17 / 13.49 s | 13.0 / 28.3 s | 0 |
| **32 / 8** | **504.7** | 2.0 / 4.3 s | **1,716.3** | 0.91 / 1.21 s | 12.4 / 21.3 s | 0 |

32 / 8 improves c32 output tok/s by 31.7 % and c1 by 4.6 %, so it passes
the rule (≥ 5 % at c32, no more than 5 % loss at c1) and is adopted.

### Final gates (32 / 8)

| Gate | Result |
|---|---|
| `l1` | PASS. `enable_thinking: false` still thinks (240 reasoning of 244 completion tokens, answer `323`); the long-thought prompt is cut at exactly 512 and answered; stream order reasoning → content → `[DONE]`; two vLLM requests per chat |
| `serving` | PASS, table below |
| `think` | PASS. Default and efforts low/high/max, 4 each: all answer `323` with a thought (152–239 completion tokens); `chat_template_kwargs` gets 400 |
| `tool-calling` | PASS. 4 auto tool calls `bash {"command": "ls"}`, each with a thought; a tool-result turn; a streamed tool call |
| `vision` | PASS. 4 image requests think first and answer `Red` (31–43 completion tokens) |
| `cancellation` | PASS (`final2`). A disconnect during the thought and during the answer: vLLM was active when cut and both Kairyu and vLLM were idle 0.19 s later; a follow-up answers `323`. The first run (`final`) cut nothing during the answer because its prompt ("count upwards") answered in about 40 tokens, which vLLM finished before the first chunk arrived; the gate now asks for 1–2000 (8,192 answer tokens) |
| `restart` | PASS. OpenJev restarts healthy and answers |
| `systemone-live` | PASS. OpenJev's own `tests/test_live.py` (pinned revision) against Kairyu: 12 passed (README example, image, read options, think, 255 options, 30 questions in chunks, unknown model, 64 concurrent reads, chat, chat stream); 4 skipped (encoder models not served) |
| `systemone-serving` | PASS (`final2`), table below |
| `systemone-isolation` | PASS. 640 concurrent System One requests with 8 chats alongside: 512 × 200 and 128 × 429 (Kairyu's 256 forwarded + 256 queued), no 529, in 16.9 s; all 8 chats answer `323`; the chat replica stays healthy |
| Browser | Playground (:3011): the README example answers `urgent` yes 1.000, `team` outage 1.000, `tone` 2.00 ≈ furious (confidence 0.989), 166 input tokens, 80 ms in the browser (Server-Timing model 60.8 ms, server 3.3 ms); the chat column streams a 1,146-character thought, then `urgent: yes / team: outage / tone: furious` (first token 166 ms, 640 ms). Open WebUI (:3010): "Thought for less than a second", then `323` |

**Chat serving** (`final-serving`; mixed prompts, 32 requests per row,
answer cap 8192; every request thinks first and finishes with `stop`):

| Row | OK | Wall | Output tokens | tok/s | TTFT p50 / p99 | Answer TTFT p50 | E2E p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 32/32 | 76.7 s | 39,142 | 510.6 | 0.15 / 0.20 s | 1.10 s | 2.3 / 7.1 s |
| c8 | 32/32 | 23.1 s | 36,365 | 1,575.5 | 0.34 / 0.50 s | 2.54 s | 4.7 / 8.5 s |
| c16 | 32/32 | 24.9 s | 38,426 | 1,540.5 | 0.59 / 2.50 s | 4.11 s | 7.4 / 18.4 s |
| c32 | 32/32 | 21.4 s | 38,273 | 1,785.6 | 0.97 / 1.43 s | 6.98 s | 12.9 / 21.4 s |

TTFT counts the first thought token; "answer TTFT" is the first answer
token, after the thought (about 1,700 thought characters on average).

**System One serving** (`final2-systemone-serving`; OpenJev's README
method: 3 questions per request, cache-busted states, 256 requests per row):

| Concurrency | Through Kairyu req/s | p50 / p95 | Direct to OpenJev req/s | p50 / p95 |
|---|---:|---:|---:|---:|
| 1 | 11.0 | 57 / 140 ms | 10.9 | 127 / 132 ms |
| 16 | 45.4 | 280 / 694 ms | 29.3 | 539 / 819 ms |
| 32 | 39.9 | 739 / 1,382 ms | 37.6 | 826 / 1,478 ms |
| 64 | 37.9 | 1,342 / 3,413 ms | 33.6 | 1,764 / 2,794 ms |

Single-request latency is bimodal: one read when every answer is certain,
four when one is not (OpenJev re-reads at entropy > 0.1). The c1 p50 lands
on different modes; c1 throughput is the same either way (11.0 vs 10.9
req/s), so Kairyu adds no measurable cost. Rows above c16 are 6–9 s long and
vary by run order. OpenJev's README reports 43.3 / 51.7 / 57.4 req/s at
c16/32/64 at 38 % GPU memory; here vLLM has 0.9 of the GPU and serves chat
too.

### Review-fix rerun (2026-10-01, `rev-*`)

The overlay now refuses an empty `stop` or out-of-range `top_logprobs`
before the thought, and Kairyu's System One route reserves the billed upper
bound, checks each model's body limit, validates both usage counts and
answers tenant 429s in Jev's shape (PR #614 review).

| Gate | Result |
|---|---|
| `think` | PASS, including `empty_stop_refused_before_thought`: through Kairyu the stream carries only an error frame (no thought), and the replica stays healthy. Directly, OpenJev answers 400 for `stop: [""]`, streamed or not, with zero vLLM requests |
| `serving` | PASS: c1 489.6 / c8 1,414.6 / c16 1,520.0 / c32 1,645.6 tok/s, 32/32 each, all `stop` |
| `tool-calling`, `vision`, `cancellation`, `restart` | PASS (restart healthy and answering after 105 s) |
| `systemone-live` | PASS (12 passed, 4 skipped) |
| `systemone-serving` | PASS: through Kairyu 13.1 / 39.6 / 42.6 / 39.1 req/s at c1/16/32/64, direct 11.4 / 35.0 / 31.9 / 32.5 |
| `systemone-isolation` | PASS: 512 × 200, 128 × 429, no 529, in 17.5 s; chat answered; replica healthy |
| `l1` | **Intermittent FAIL (2 of 4 runs), fixed in the second round below** in `thought_cut_at_budget`: after the thought is cut at 512, the answer pass sometimes opens a new thought and spends its whole 256-token budget there (`finish_reason: length`, 768 completion tokens). The overlay drops a reopened thought by design, so the answer is empty. The other cases pass in every run. Not caused by the review fixes (the request sets no stop or logprobs); the first GPU pass happened to pass it. Open |

### Second review-fix rerun (2026-10-01, `rev2-*`)

**Empty answer after a cut thought.** Measured on the GPU host by sending the
two passes to vLLM directly, with the same thought for every variant of the
answer pass (`l1`'s long-thought prompt and a coding prompt, 40 thoughts,
all cut at 512, answer cap 1024):

| Answer pass | Empty answers |
|---|---:|
| `enable_thinking: true` (before) | 13 / 40 |
| `enable_thinking: false` | 4 / 40 |
| cut thought ended with the budget sentence, thinking on (adopted) | **0 / 40** |
| cut thought ended with the budget sentence, thinking off | 0 / 40 |

Banning the thought-open token is not possible: vLLM refuses `logit_bias`,
`bad_words` and `allowed_token_ids` for diffusion models (HTTP 400).
Stopping on that token and retrying the answer pass did not converge (about
half of first attempts reopened; 4 of 40 never answered in 4 attempts). The
overlay now ends a thought cut at the budget with "I have reached my
thinking budget, so I will stop thinking and give my final answer now."
before closing it (budget forcing). Only the answer prompt carries the
sentence; the published reasoning stays the model's own thought.

| Gate | Result |
|---|---|
| `l1` | PASS in 8 of 8 runs. The long-thought prompt answers `160000` right after the cut thought in 7 runs (519 completion tokens) and gives the squares list in 1 |
| `think`, `tool-calling`, `vision`, `cancellation`, `systemone-live`, `systemone-isolation` | PASS (isolation: 512 × 200, 128 × 429, no 529, 19.3 s) |
| `serving` | PASS: c1 476.0 / c8 1,477.5 / c16 1,565.2 / c32 1,764.3 tok/s, 32/32 each, all `stop` |
| `systemone-serving` | PASS: through Kairyu 10.1 / 35.7 / 33.9 / 30.3 req/s at c1/16/32/64, direct 9.0 / 31.4 / 29.8 / 28.2 |

Kairyu now also normalizes `samples`/`think`/`steps` the way the upstream
parses them ("32", 32.0) before reserving and forwarding, refuses other
representations with 422, and reserves a `sequential` read's repeated schema.

### Degenerate endings (D2)

The first serving run (8 / 32, 256 requests) had one failure: a coding
answer finished its code, then repeated one character (`助`) until the
8,192-token cap. Its thought had been cut at 512 and the answer began with
the rest of the plan. 1,536 further coding requests at c32, sent directly
to OpenJev, reproduced none:

| L1 | Requests | Degenerate | Thought cut at 512 | tok/s |
|---|---:|---:|---:|---:|
| This overlay (think 512) | 512 | 0 | 474 | 1,639–1,737 |
| Stock OpenJev image, no thinking | 512 | 0 | – | 1,608 |
| Stock OpenJev image, native thinking (no cap) | 512 | 0 | 1 | 1,844 |

The rate is about 1 in 1,800 coding answers and is not attributable to the
overlay. It stays a known limit; `serving` still requires zero failures.

## Limitations

- **Rare degenerate endings:** about 1 in 1,800 coding answers repeats one
  character until the answer cap (see D2 above).
- **KV cache:** OpenJev's vLLM runs an fp8_e4m3 KV cache with scale 1.0,
  because the checkpoint ships no KV scales (OpenJev's setting).
- **Answer quality** is DiffusionGemma 26B-A4B's in this mode; nothing here
  measures it.

# Measurements: OpenJev DiffusionGemma on one GPU (think = 512)

Status: **CPU evidence recorded (2026-10-01); every GPU gate is pending.**
GPU work did not run on the development machine. This file records what
was checked without a GPU, and what remains to run on the GPU host.

## Pins

| Item | Value |
|---|---|
| OpenJev source | `dcd20947b5ddad5be4a8f5aed6aa6dd245653823` (package 0.5.0) |
| Base image | `razorback16/openjev:0.5.1@sha256:65f88680e0093c9229d8cc55c0d2f279db27aba78a7ce74ea762b8f2da82d47d`, image ID `sha256:b219be87fdde14896b0029edabe5586b2e99c89380784a9ec73f1405a0453757` |
| vLLM in the base | `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`, with OpenJev's logprob-id cap 512 and vision prefix-LM patch (from the image's build history) |
| Overlay image ID | **pending** (L1-0) |
| Checkpoint | `nvidia/diffusiongemma-26B-A4B-it-NVFP4@ec4ff3df205028f4e81c954c2227f9312b3ec2ea` |
| Checkpoint tree | `e38de908a6b053b2a2781c37251afa41776d26a98b050d1fe7badb19139ea9b7` |
| Chat template | stock `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` → edited `536903c781831d1b6902ed1c1756217b6cfef133d30aa4854e4728701cfcb9be` |

The checkpoint tree hash uses the same algorithm as `control.py`'s
attestation: the sha256 of the sorted `{path, size, sha256}` rows. It was
computed from the HF tree API's LFS sha256 values for the two safetensors
files and `tokenizer.json`, and from downloaded bytes for the 11 small
files. The first download on the GPU host re-hashes every file and fails
closed on any difference.

## CPU evidence (development machine, 2026-10-01)

**Template edit**, checked with the checkpoint's own tokenizer
(transformers 5.12.1):

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
| Generator in use | `ThinkFirstGenerator`; startup template check passes; `<channel|>` is token 9 (one token) |
| Request with `enable_thinking: false`, `reasoning_effort: none` | thought pass sent with thinking on, `max_tokens` 512, `stop_token_ids` [9]; answer pass prefill `<|channel>thought\n…\n<channel|>` with the caller's `max_tokens` 64 |
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

## GPU gates (pending)

Run on the GPU host, in order. Record run IDs and the served-config hash
from each `run.json`.

1. **L1-0.**
   - Build once: `OPENJEV_ALLOW_UNPINNED_IMAGE=1 ./run.sh up`.
   - Record the printed image ID in `example.json` (`openjev.image_id`) and
     in `kairyu.yaml` (`container_image_digest`), then run `./run.sh up`
     again with the pin.
   - Must pass: the model attestation, the overlay's startup check, the
     readiness probes, and `./verify.sh l1 --no-start`.
2. **L1-1.** Compare `OPENJEV_GEN_MAX_INFLIGHT` at 8, 16 and 32, with
   `OPENJEV_GEN_MAX_QUEUE = 40 − inflight`. For each candidate run
   `./run.sh up`, then `./verify.sh serving --no-start`. Adopt a candidate
   only if it improves c32 output tok/s by at least 5 % without losing more
   than 5 % at c1. Otherwise keep 8 / 32.
3. **Final gates:** `serving`, `think`, `tool-calling`, `vision`,
   `cancellation` and `restart`, plus a manual Chat UI check on :3010 (the
   reasoning is shown before the answer).

## Limitations

- **Unverified until the GPU gates pass:**
  - `continue_final_message` with vLLM's gemma4 reasoning and tool parsers
    on the DiffusionGemma path;
  - whether the thought pass's text arrives as `reasoning` or `content`
    (both are handled);
  - throughput and latency.
- **Answer quality** is DiffusionGemma 26B-A4B's in this mode; nothing here
  measures it.

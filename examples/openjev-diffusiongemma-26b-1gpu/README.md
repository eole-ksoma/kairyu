# OpenJev DiffusionGemma on one RTX PRO 6000 GPU (think = 512)

One [OpenJev](https://github.com/razorback16/openjev) replica on one 96 GB
RTX PRO 6000 Blackwell (SM120) card. OpenJev runs vLLM with DiffusionGemma
26B-A4B NVFP4 inside its container and answers OpenAI chat requests. Kairyu
puts the same L2/L3 structure in front of it as the other single-model
examples: one `ReplicaPool` replica, the OpenAI-compatible API, and Open
WebUI. Kairyu also serves OpenJev's System One API (the Jev wire format,
`POST /v1/systemone`), with a Jev-style playground.

```text
Open WebUI (:3010) -> Kairyu (:8010) -> ReplicaPool        -> OpenJev (:8080) -> vLLM, one GPU
Playground (:3011) -> Kairyu (:8010) -> /v1/systemone + chat -^
```

**Every answer thinks first.** Each chat completion writes a thought of at
most 512 tokens, then the answer. The cap is hard: a longer thought is cut,
with the same meaning as OpenJev's System One `think`. A caller cannot turn
the thought off or change its budget.

## Start

```sh
./run.sh up          # preflight, overlay build, model download + hash check, readiness
./run.sh status
./verify.sh l1 --no-start
./verify.sh serving --no-start
./verify.sh think --no-start
./verify.sh tool-calling --no-start
./verify.sh vision --no-start
./verify.sh cancellation --no-start
./verify.sh restart --no-start
./verify.sh systemone-live --no-start
./verify.sh systemone-serving --no-start
./verify.sh systemone-isolation --no-start
./run.sh down
```

**GPU selection.** `GPU_ID` selects the GPU (default 0). `run.sh up`
refuses to start on a GPU that another workload is using.

**Model download.** The checkpoint (18.9 GB) is downloaded to
`/mnt/nvme/kairyu/model-volumes/openjev-diffusiongemma-26b-1gpu/models` and
hashed against the pinned tree.

**First build.** The first `run.sh up` builds this example's overlay image.
Until its image ID is recorded in `example.json` (`openjev.image_id`) and in
`kairyu.yaml` (`container_image_digest`), serving is refused. Start once
with `OPENJEV_ALLOW_UNPINNED_IMAGE=1` (not evidence), then record the
printed ID. The first start also compiles FlashInfer's NVFP4 MoE kernels
(about 4 minutes); they are cached on NVMe.

**Endpoints.**
- API: `http://127.0.0.1:8010/v1`, model `diffusiongemma-26b`.
- Chat UI: `http://127.0.0.1:3010` (local, no authentication, no effort
  selector).

## How the thought is enforced

OpenJev's chat endpoint has no thinking budget. vLLM's `thinking_token_budget`
does not apply either: DiffusionGemma replaces vLLM's sampler with its own
`DiffusionSampler`, which never enforces that budget. This example therefore
applies OpenJev's own two-pass `think` method to chat, in an overlay on the
published image (`openjev-think.Dockerfile`).

1. **Thought pass.** The conversation is continued from an assistant prefill
   `<|channel>thought\n`, with thinking on, `max_tokens: 512`, and a stop on
   the `<channel|>` token.
2. **Answer pass.** The conversation is continued from
   `<|channel>thought\n{thought}\n<channel|>`. The caller's `max_tokens`,
   `stop`, `logprobs`, `tools` and `tool_choice` apply here, and prefix
   caching reuses the first pass's prompt.

**What the caller sees:**

- **The thought** is returned as reasoning. Kairyu passes it on as
  `reasoning_content`, and a streaming client receives it before the answer.
  If the model opens a second thought during the answer pass, that text is
  dropped, so the published thought is always the capped first pass.
- **Usage** counts the prompt once and both outputs as completion tokens.
  The answer budget (`max_tokens`, OpenJev's cap 8192) comes on top of the
  512-token thought.
- **Errors.** A refused thought is an HTTP error. A request that only the
  answer pass would refuse is checked before any thought is written: a
  required or named `tool_choice` needs structured outputs, which vLLM does
  not support for diffusion models, so it gets a 400 up front. Kairyu counts
  an aborted stream as a failure of the only replica; a 400 is not counted.
  If the answer pass still fails after the thought has streamed (a server
  error), the stream is aborted. Kairyu ignores mid-stream error chunks, so
  ending the stream normally would make a truncated reply look complete.
- **Ignored and rejected fields.** OpenJev ignores `reasoning_effort` and
  `temperature`. Kairyu rejects `chat_template_kwargs` (as for every
  legacy-chat example), and rejects the sampling fields OpenJev would
  silently drop (`n`, `seed`, penalties, `min_p`, `min_tokens`, `ignore_eos`,
  …).

**Code.** `think_core.py` holds the two passes and has no OpenJev imports.
`think_first.py` wires it into OpenJev's chat route. `patch_openjev.py`
installs both at build time, with exactly-once anchors. `chat_template.jinja`
is the checkpoint's template with one edit: a final assistant message is
rendered verbatim, so a thought prefill can be continued. At startup the
overlay refuses to run unless vLLM uses that template and it renders both
prefills.

The published image also keeps its own source tree at `/app/openjev`, and
`/app` is the working directory, so `python -m openjev` would import that
unpatched copy first. The overlay removes it. The build then fails unless an
import from `/app` resolves to the patched package. `check_overlay.py`
reproduces the CPU evidence in `MEASUREMENTS.md`.

## Configuration

- **OpenJev settings** (`example.json` `openjev.settings`): OpenJev's own
  defaults, which are canvas 64, 64 sequences, a 65,536-token context, 0.9
  GPU memory, and an 8,192-token answer cap.
- **Concurrency.** OpenJev runs 32 generations and queues 8 more (its
  default is 8 and 32; 32 in flight measured +32 % output tok/s at c32 and
  no loss at c1, see `MEASUREMENTS.md`). Above that it answers 529, which
  Kairyu counts as a failure of the only replica. Kairyu therefore admits at
  most 40 chat requests and answers 429 first. `OPENJEV_GEN_MAX_INFLIGHT`
  and `OPENJEV_GEN_MAX_QUEUE` may be overridden for tuning, but `control.py`
  refuses any pair whose sum is not 40.
- **System One.** See [System One and the playground](#system-one-and-the-playground).
- **Known limits.**
  - About 1 in 1,800 coding answers repeats one character until the answer
    cap (`MEASUREMENTS.md`, D2); it reproduced on neither the overlay nor
    the stock image in 1,536 direct requests.
  - Kairyu's legacy chat path sends text-only history as one transcript
    message.
  - `/v1/messages/count_tokens` returns 404, because OpenJev has no
    `/tokenize`.
  - `tool_choice` must be `auto` or `none`. A required or named tool call
    is refused with a 400, because it needs structured outputs, which vLLM
    does not support for diffusion models.

## System One and the playground

OpenJev reads typed answers (yes/no, choice, score) straight from the
model's probabilities. Kairyu serves that API as `POST /v1/systemone` under
`openjev-0.1` and the aliases `openjev-latest`, `jev-latest` and
`jev-preview`, so TypeSafe's SDKs work against `http://127.0.0.1:8010`:

```sh
curl http://127.0.0.1:8010/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "openjev-latest", "state": "I was charged twice this month.",
  "questions": {"is_billing": {"type": "noul", "instructions": "Is this a billing issue?"}}}'
```

- **Limits.** OpenJev runs 64 reads at a time and answers 529 once 512
  requests wait. Kairyu forwards at most 256 System One requests and queues
  256 more (`kairyu.yaml` `systemone:`), so a burst gets Kairyu's 429, never
  OpenJev's 529. These requests do not count against chat's 40 and never
  touch the chat replica.
- **Playground** (`http://127.0.0.1:3011`, no authentication): enter a
  state, optional images and typed questions, then see each answer's
  probability bars, confidence, tokens, latency and OpenJev's
  `Server-Timing` split. Next to it, the same state and questions go to the
  chat model, which thinks first (512 tokens at most) and writes its answer.
  The page is `playground/index.html`, served by nginx with Kairyu's API on
  the same origin (`playground/nginx.conf`).

## Verification gates

`./verify.sh list` prints them; the GPU results go in `MEASUREMENTS.md`.

- `l1`: talks to OpenJev directly, inside its container.
  - `enable_thinking: false` still thinks.
  - `reasoning_tokens` is at most 512.
  - A prompt that asks for a long thought is cut at exactly 512 and still
    answered.
  - The stream order is reasoning, then content, then usage.
  - vLLM completes two requests per chat.
- `serving`: completed answers through Kairyu at c1/8/16/32 (32 requests
  each, generic and coding prompts alternating; `--concurrency 1,32` selects
  levels), with placement on the single replica. Every sample must
  think first and finish with `stop`. OpenJev cannot generate a fixed number
  of tokens.
- `think`: the default request and efforts low, high and max all think and
  answer `323`. `chat_template_kwargs` is rejected.
- `tool-calling`: plain and streamed auto tool calls, each with a thought,
  and a tool-result turn.
- `vision`: image requests think first and name the image's color.
- `cancellation`: a disconnect during the thought and during the answer
  frees Kairyu and vLLM, and a follow-up request completes.
- `restart`: OpenJev restarts healthy and passes the readiness probes.
- `systemone-live`: OpenJev's own `tests/test_live.py` at the pinned
  revision, pointed at Kairyu (System One reads, images, options, chat).
- `systemone-serving`: OpenJev's README method, through Kairyu and directly:
  256 cache-busted requests with 3 questions at c1/16/32/64, reporting req/s
  and p50/p95 latency.
- `systemone-isolation`: 640 concurrent System One requests with 8 chats
  alongside. Every read gets 200 or Kairyu's 429 (never 529), chat answers
  correctly, and the chat replica stays healthy.

## Primary references

- OpenJev: https://github.com/razorback16/openjev (`dcd20947`, image
  `razorback16/openjev:0.5.1`).
- Checkpoint: https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
  (`ec4ff3df`).
- vLLM structured reads for DiffusionGemma: vllm-project/vllm#57250
  (`1b3b88ec`).

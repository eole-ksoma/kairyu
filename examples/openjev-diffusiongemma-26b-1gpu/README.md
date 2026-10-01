# OpenJev DiffusionGemma on one RTX PRO 6000 GPU (think = 512)

One [OpenJev](https://github.com/razorback16/openjev) replica on one 96 GB
RTX PRO 6000 Blackwell (SM120) card. OpenJev runs vLLM with DiffusionGemma
26B-A4B NVFP4 inside its container and answers OpenAI chat requests. Kairyu
puts the same L2/L3 structure in front of it as the other single-model
examples: one `ReplicaPool` replica, the OpenAI-compatible API, and Open
WebUI.

```text
Open WebUI (:3010) -> Kairyu (:8010) -> ReplicaPool -> OpenJev (:8080) -> vLLM, one GPU
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
- **Errors.** A refused thought is an HTTP error. If the answer pass fails
  after the thought has streamed, the stream is aborted. Kairyu ignores
  mid-stream error chunks, so ending the stream normally would make a
  truncated reply look complete.
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

## Configuration

- **OpenJev settings** (`example.json` `openjev.settings`): OpenJev's own
  defaults, which are canvas 64, 64 sequences, a 65,536-token context, 0.9
  GPU memory, and an 8,192-token answer cap.
- **Concurrency.** OpenJev runs 8 generations and queues 32 more. Above
  that it answers 529, which Kairyu counts as a failure of the only replica.
  Kairyu therefore admits at most 40 requests and answers 429 first.
  `OPENJEV_GEN_MAX_INFLIGHT` and `OPENJEV_GEN_MAX_QUEUE` may be overridden
  for tuning, but `control.py` refuses any pair whose sum is not 40.
- **System One.** OpenJev's System One API (`/v1/systemone`) stays inside
  the compose network. Kairyu serves the chat model only.
- **Known limits.**
  - Kairyu's legacy chat path sends text-only history as one transcript
    message.
  - `/v1/messages/count_tokens` returns 404, because OpenJev has no
    `/tokenize`.
  - A required or named `tool_choice` needs structured outputs, which vLLM
    does not support for diffusion models.

## Verification gates

`./verify.sh list` prints them; the GPU results go in `MEASUREMENTS.md`.

- `l1`: talks to OpenJev directly, inside its container.
  - `enable_thinking: false` still thinks.
  - `reasoning_tokens` is at most 512.
  - A prompt that asks for a long thought is cut at exactly 512 and still
    answered.
  - The stream order is reasoning, then content, then usage.
  - vLLM completes two requests per chat.
- `serving`: completed answers through Kairyu at c1/8/16/32 for generic and
  coding prompts, with placement on the single replica. Every sample must
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

## Primary references

- OpenJev: https://github.com/razorback16/openjev (`dcd20947`, image
  `razorback16/openjev:0.5.1`).
- Checkpoint: https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
  (`ec4ff3df`).
- vLLM structured reads for DiffusionGemma: vllm-project/vllm#57250
  (`1b3b88ec`).

# Kairyu L3 + vLLM L1 examples

`examples/` contains seven complete environments:

| Environment | GPU layout | Model/context |
|---|---|---|
| [`qwen3.8-27b-1gpu`](qwen3.8-27b-1gpu/README.md) | one selected RTX PRO 6000 Blackwell | official FP8, 262,144 tokens |
| [`deepseek-v4-flash-0731-8gpu`](deepseek-v4-flash-0731-8gpu/README.md) | TP8 + EP8 on eight RTX PRO 6000 Blackwell cards | mixed FP4/FP8, 1,048,576 tokens |
| [`qwen3.8-deepseek-v4-8gpu`](qwen3.8-deepseek-v4-8gpu/README.md) | Qwen TP1 x 4 replicas + DeepSeek TP4/EP4 | Qwen-judged five-route Kairyu L2 (four direct routes + verifier-gated ensemble DAG) |
| [`qwen3.8-27b-dp8-8gpu`](qwen3.8-27b-dp8-8gpu/README.md) | Qwen TP1 x 8 replicas, one per card | one public model with OpenAI tool calling; Kairyu L2 is the replica pool only (even, prefix-aware placement) |
| [`deepseek-v4-flash-0731-dp2-8gpu`](deepseek-v4-flash-0731-dp2-8gpu/README.md) | DeepSeek TP4+EP4 x 2 replicas (GPU 0-3, 4-7) | one public model with OpenAI tool calling; Kairyu L2 is the replica pool only (even, prefix-aware placement) |
| [`deepseek-v4-flash-vision-exp-dp2-8gpu`](deepseek-v4-flash-vision-exp-dp2-8gpu/README.md) | DeepSeek-V4-Flash-Vision-Exp TP4+EP4 x 2 replicas (GPU 0-3, 4-7) | one public text + image model with OpenAI tool calling and a Chat UI reasoning-effort dropdown (default/low/high/max); replica pool only |
| [`deepseek-v4.1-flash-8gpu`](deepseek-v4.1-flash-8gpu/README.md) | DeepSeek-V4.1-Flash, one TP8 replica (GPU 0-7) | text + image, OpenAI tools, default thinking high; V4 ReplicaPool/API/UI structure |
| [`qwen3.8-flash-next-dp2-8gpu`](qwen3.8-flash-next-dp2-8gpu/README.md) | Qwen3.8-Flash-Next-FP8 TP4 x 2 replicas (GPU 0-3, 4-7) | one public text + image model with OpenAI tool calling and a Chat UI reasoning-effort dropdown (default/low/medium/xhigh); replica pool only |

All seven use Kairyu as L3, vLLM as L1, and Open WebUI as the public chat surface.
The four `dp` environments add no orchestration: Kairyu L2 only spreads requests
over identical L1 replicas, and their `verify.sh` proves the per-replica split
and the OpenAI tool-calling agent contract (`tool-calling`); the two vision
`dp2` environments also prove image requests on every replica (`vision`).
Each `run.sh` command prints its API and Chat UI URLs when the stack is ready.
Qwen3.8-27B uses the digest-pinned official vLLM v0.23.0 image. Both
DeepSeek-V4-Flash-0731 deployments share the same measured `aa0d513027` SM120
build, retaining DSpark performance and checkpoint compatibility that v0.23.0
cannot provide. The two vision environments (DeepSeek-V4-Flash-Vision-Exp and
Qwen3.8-Flash-Next, both served only by upstream `main`) share one overlay
image built from upstream's digest-pinned nightly of commit `27a94d1c` plus
FlashInfer `60b49158` (SM120 sparse-MLA prefill fix); whichever runs first
builds it.
The Qwen-hosting and `dp` environments keep persistent model, UI, and cache state
on bind-backed storage below `/mnt/nvme` (the replica examples reuse the sibling
examples' attested checkpoints); the standalone `deepseek-v4-flash-0731-8gpu`
example uses Docker-managed volumes.

Start everything and print the local Chat UI URL:

```sh
./examples/deepseek-v4-flash-0731-8gpu/run.sh
./examples/qwen3.8-27b-1gpu/run.sh
./examples/qwen3.8-deepseek-v4-8gpu/run.sh
./examples/qwen3.8-27b-dp8-8gpu/run.sh
./examples/deepseek-v4-flash-0731-dp2-8gpu/run.sh
./examples/deepseek-v4-flash-vision-exp-dp2-8gpu/run.sh
./examples/qwen3.8-flash-next-dp2-8gpu/run.sh
```

Run serving verification through the Kairyu L3 endpoint:

```sh
./examples/deepseek-v4-flash-0731-8gpu/verify.sh serving
./examples/qwen3.8-27b-1gpu/verify.sh serving
./examples/qwen3.8-deepseek-v4-8gpu/verify.sh serving-auto-max
./examples/qwen3.8-27b-dp8-8gpu/verify.sh serving
./examples/deepseek-v4-flash-0731-dp2-8gpu/verify.sh serving
./examples/deepseek-v4-flash-vision-exp-dp2-8gpu/verify.sh serving
./examples/qwen3.8-flash-next-dp2-8gpu/verify.sh serving
```

List the supported operations with `verify.sh list`. Model and product
evaluations are separate and are invoked explicitly through `python -m evals`;
see [the benchmark documentation](../docs/benchmarks.md).

# DeepSeek V4.1 Flash single-replica example

Accepted scope: one replica on all eight RTX PRO 6000 Blackwell GPUs;
separate PR to main; preserve the V4 vision example's ReplicaPool, OpenAI
chat/tool path, and Chat UI. Default thinking effort is the model author's
`high`, including requests that omit effort.

- [x] Create `codex/deepseek-v4.1-flash-8gpu` from main.
- [x] Resolve checkpoint revision and independently derive its file manifest.
- [x] Pin an SM120-capable vLLM image and reconcile its effort encoding with
  the official checkpoint (`low=50`, `high=75`, `max=100`).
- [x] Implement lifecycle, UI defaults, model attestation, and verification.
- [x] Run CPU contract tests and lint.
- [ ] Save the existing stack's restart information; release its GPUs.
- [ ] Establish a correct TP8 baseline; compare EP, DSpark, batch/sequence
  limits, memory placement, and supported graphs in bounded stages.
- [ ] Run final serving, tool, thinking/default/override, vision, cancellation,
  restart, and long-context checks; retain exact evidence and limitations.
- [ ] Document measurements and design/progress changes; create the PR.

## Sources

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md
- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash

The recipe's GB200 validation does not establish SM120 compatibility.
Its effort aliases differ from the pinned checkpoint. Verify rendered
prompts rather than inferring effort from a nonempty reasoning response.
Fixed 256-token throughput rows measure engine throughput, including
reasoning tokens; they do not establish final-answer quality or latency.

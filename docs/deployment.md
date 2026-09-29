# Deployment Guide — On-Prem DC with a Managed Cloud Front

How to run Kairyu as a product: one container image, one `kairyu serve`
entrypoint, an on-prem GPU fleet behind a stateless gateway tier, and a thin
managed cloud edge. Design rationale: `docs/design/m7-productionization.md`;
acceptance gates: `docs/goals/g3-production-deployment.md`.

## 1. Topology

```
Internet
   │
Cloud edge (managed; no Kairyu code here)
   ├─ WAF            — DDoS, per-client rate limits, bot filtering
   ├─ L7 LB          — TLS termination, health-checked routing to gateways
   │
Private interconnect (Direct Connect / ExpressRoute class; IPsec VPN fallback)
   │
DC gateway tier — 2× CPU nodes, stateless          kairyu serve gateway.yaml
   ├─ OpenAI-compatible API + orchestration (Router/Conductor/MoA)
   ├─ ReplicaPool per served model → remote GPU replicas
   ├─ /health /readyz /metrics, API keys, concurrency guard, batch worker
   │
DC GPU replica tier — N nodes                      kairyu serve replica.yaml
   └─ each node: same image, local engine (kairyu / vllm backend),
      internal TP / P-D / PP layout per docs/gpu-runbook.md §6–7
```

Traffic crossing the interconnect is completions requests/responses only.
KV pages, TP collectives, and P-D transfers never leave the DC fabric
(IB/RoCE stays node-to-node inside the DC).

## 2. Division of security duties

| Concern | Owner | Kairyu's part |
|---|---|---|
| DDoS, bot filtering, per-client rate limits | Cloud WAF | — |
| TLS termination, certificates | Cloud LB (or DC reverse proxy) | serves plain HTTP behind it |
| Client authentication | Gateway | `server.api_keys_env` (static keys, constant-time compare) |
| Process overload | Gateway/replica | `server.max_concurrency`; optional single-local-backend admission queue → 429 + Retry-After |
| TTFT SLO overload | Gateway | optional `server.ttft_slo_s` → admit, batch-defer, or 429 + Retry-After |
| Routing inspection | Gateway | `/v1/route` uses data-plane auth; `/routing` remains inside the configured API-key boundary |
| Node-to-node auth inside the DC | Deployment choice | keyless (`api_key_env: null`) or a shared key env var |
| Audit trail | Gateway | JSON access log with `X-Request-ID`, JSONL router decision log |

## 3. Node setup (systemd + docker compose — design m7 D2)

Kubernetes is deliberately not required: the fleet is small and static, GPU
nodes are pinned to hardware, and the design lineage excludes elasticity
(g2 §6). Everything is containerized, and a CI-tested Helm chart plus kind
manifests already exist for the k8s path (§8). Revisit k8s when any of these
hold: the fleet grows past ~5–8 nodes, multiple teams deploy onto the
cluster, or rolling deploys become weekly toil.

Per node: install Docker, drop the compose file + config, and wrap it in a
systemd unit so the stack survives reboots:

```ini
# /etc/systemd/system/kairyu.service
[Unit]
Description=Kairyu node (gateway or replica; config decides)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/kairyu
ExecStart=/usr/bin/docker compose up -d --wait
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

GPU replica nodes additionally need the NVIDIA container toolkit and a
`deploy.resources.reservations.devices` (or `gpus: all`) stanza in their
compose service.

## 4. Configuration walkthrough

Both roles run the same image; the mounted DeploymentSpec decides what the
process is. CPU nodes use the root `Dockerfile`; GPU nodes use
`Dockerfile.cuda` (CUDA runtime plus the `gpu` extra). Working CPU examples
live in `deploy/compose/` and are exercised by `scripts/compose_smoke.sh` in
CI; the GPU counterpart is `deploy/compose/docker-compose.gpu.yaml` with
`gateway-gpu.yaml` / `gpu-replica.yaml`.
The flat `server:` mapping is a versioned deployment schema and is translated
explicitly to runtime server settings. Runtime-only settings do not
automatically become accepted YAML keys.

Validate a deployment graph before rollout without starting a server:

```bash
kairyu validate /etc/kairyu/deployment.yaml
```

Validation is always deterministic and offline. It checks the DeploymentSpec
schema and declared local filesystem references: orchestrator specs, `.jinja`
chat templates, and local `kairyu`/`kairyu-proc` model and tokenizer artifacts,
including tokenizer-owned chat metadata.
Orchestrator and `.jinja` paths resolve from the deployment file's directory;
native model and tokenizer paths retain the serve process's working-directory
semantics.
It does not read credential environment variables, expose secret values,
construct a backend, materialize model tensors, contact an endpoint, or probe
a GPU. Model compatibility is derived from metadata-only `meta` shapes, and
checkpoint tensor data is never read. Network
checks are therefore skipped and hardware checks remain indeterminate; those
belong to the readiness and deployment acceptance gates. A valid graph exits
`0`, a validation failure exits `1`, and argparse usage errors exit `2`.

The current DeploymentSpec has no standalone adapter, grammar, or benchmark
artifact references, so `validate` does not invent checks for them. Their
request-, benchmark-, and runtime-owned validation remains unchanged.

Chat-template ownership is resolved before any backend can allocate model or
GPU resources. For `kairyu` and `kairyu-proc`, the effective local tokenizer is
`options.tokenizer` when explicitly set and `options.model_path` otherwise;
direct local vLLM uses `options.tokenizer` before `options.model`. At that
tokenizer root, a `chat_template.jinja` default and named
`additional_chat_templates/*.jinja` files collectively take precedence over
the `chat_template` value in `tokenizer_config.json`, matching Transformers.
Kairyu also supplies the tokenizer's named special tokens to the Jinja context.
An explicit per-served-model DeploymentSpec `chat_templates` entry wins over
the auto-loaded template while retaining those tokenizer-owned token values.

There is no silent production fallback. A real text-chat engine or static pool
whose effective local tokenizer has no template fails startup. An explicit
`chat_templates` override is accepted only for local `kairyu`, `kairyu-proc`,
vLLM, or deterministic `mock` engines and compatible static pools, where the pre-rendered prompt can
retain tokenizer ownership through dispatch. Current OpenAI-compatible remote
models, orchestrators, and discovery-backed pools cannot preserve it through
an upstream chat-template boundary or derived planner/worker prompts. They
therefore require the explicit per-model compatibility opt-in:

```yaml
legacy_chat_models: [old-wire-model]
```

That opt-in logs a warning and uses the pre-M9 `role: content` concatenator
only for the listed served models; it cannot be combined with a real template
for the same name. The DeploymentSpec builder automatically selects this path
only for the built-in deterministic `mock` backend because it is a protocol
test double, not a model prompt format. Lower-level app, prompt-validation,
Responses, and batch-worker construction has no mock exception: omitting both
policies logs an explicit construction warning, and chat requests are rejected
before dispatch. Completion-only programmatic apps remain valid, but every
served chat model must provide a `ChatTemplate` or model-scoped legacy
membership. A local path is required for offline auto-loading; a Hub identifier that has not been
materialized locally may use only a self-contained explicit template. If that
template references tokenizer-owned variables such as `bos_token`, preflight
fails until the effective tokenizer metadata is available locally; static
pools require it from every replica before sharing those values.

Rendered HF chat prompts carry a typed ownership marker through Kairyu's
in-process and ZMQ transports. Native tokenization already disables automatic
special-token insertion; the direct vLLM adapter sets
`add_special_tokens=False` only for that marker. Ordinary completion strings
retain vLLM's completion default. Backends that cannot preserve the marker fail
closed instead of converting it to an ordinary string.

Replica node (GPU):

```yaml
server:
  host: 0.0.0.0
  port: 8000
  max_concurrency: 128           # active plus admission waiters
  admission_wait_timeout_s: 30   # explicit opt-in; size for model turnover
engines:
  llama-70b:
    backend: kairyu            # or vllm; mock for CPU smoke
    options:
      model_path: /models/llama-70b  # local checkpoint + tokenizer metadata
      generation_config: auto  # auto | vllm | none; default auto
      tensor_parallel_size: 4
      max_num_seqs: 16       # bound active sequences
      max_num_partial_prefills: 2 # 1 = legacy serial prefill; default 2
      priority_age_s: 60.0   # null = FIFO; 0 = strict priority; >0 = aging
      pipeline_depth: 2        # unified schedule/device overlap; default 1
      # decode_mode: eager      # optional rollback; supported CUDA defaults to graph
      # cuda_graph_max_batch: 8 # optional override; default max_num_seqs
      # cuda_graph_max_pages: 64 # optional override; default context/page_size
```

`pipeline_depth: 1` reproduces the synchronous serving path. Depth 2 or greater
submits immutable request snapshots ahead of the oldest commit while preserving
the same streaming, stop, grammar, speculative, preemption, and chunked-prefill
commit path. A graph-capable real CUDA model defaults to CUDA-graph decode;
CPU, custom/toy, P-D, replicated-attention EP, and current MLA paths default to
eager. Explicit `decode_mode: eager` is the rollback and an explicit
`cuda_graph` request remains fail-closed when unsupported. The default graph
batch tracks `max_num_seqs`; the page width covers `max_model_len / page_size`
(or all usable KV pages when no context limit is set). Dense and TP execution
reserve one scheduler KV page for padding writes; attention-DP keeps its graph
scratch outside the scheduler namespace and can cover every scheduler page.
Every bucket is captured after weights and serving communicators are ready but
before readiness is published. Distributed startup agreement and attention-DP
direct-NCCL/layout agreement use a bounded host control group distinct from the
long-idle request protocol.

`max_num_partial_prefills` shapes equal-priority prompt work into a small,
work-conserving cohort. A single prompt retains the entire token budget; with
two eligible partial prefills, short unused shares are normally reassigned
within the same step. Deferred P-D handoff may retain one peer prompt token so
an asynchronous KV copy overlaps the next engine step. Decode remains first.
If the selected waiting head is blocked by KV
capacity or the decode watermark, only its immediate successor may pass, only
once during that waiting epoch, and only when its prefill completes in the
current share using currently free pages. Setting the knob to `1` restores
serial prefill chunks but does not disable the bounded KV skip. Prompt KV is
still reserved in full at admission; this knob shapes compute, not KV capacity.

Oversized live steps fall back to eager and increment
`kairyu_cuda_graph_eager_fallbacks_total{model=...}`. Local replica pools expose
the monotonic sum across replica generations, including child restarts and
membership replacement. Invalid modes, explicit
CPU graph placement, unsupported attention/model paths, or a page limit that
leaves no scratch page fail during startup rather than after traffic arrives.

For native `kairyu` and `kairyu-proc` engines, `generation_config: auto`
loads `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty` from
the checkpoint's `generation_config.json`. Each value is applied only when the
public request omits that field; an explicit value, including a neutral one,
has precedence. `vllm` retains generation-file stop-token metadata but uses
neutral sampling defaults. `none` ignores the generation file, including its
stop-token override, and uses `config.json` plus neutral sampling defaults.
Malformed or out-of-range defaults fail startup in `auto`; `none` does not read
the file.

The serve-time override applies to every local native engine, native static-pool
replica, and native worker in a linked orchestrator spec:

```bash
kairyu serve /etc/kairyu/deployment.yaml --generation-config vllm
```

It fails when the deployment has no local native target instead of implying
that a remote OpenAI-compatible endpoint was reconfigured. `/backends` exposes
`generation_config`, `generation_config_source`, and the complete resolved
`generation_defaults` map. Local pools publish one top-level record only when
every member agrees. A gateway keeps its single remote audit sample under
`via_replica`, because one reachable replica cannot prove fleet-wide
homogeneity. Orchestrated models report per-worker policies and collapse a
top-level record only when every worker supplies the same complete value.

Gateway:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  api_keys_env: KAIRYU_API_KEYS     # comma-separated client keys
  responses_compaction_secret_env: KAIRYU_RESPONSES_COMPACTION_SECRET
  max_concurrency: 256
  ttft_slo_s: 2.0                  # optional direct-chat predictive admission
pools:
  llama-70b:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "llama-70b", api_key_env: null, upstream: kairyu }
      - backend: openai
        options: { base_url: "http://gpu-1:8000/v1", model: "llama-70b", api_key_env: null, upstream: kairyu }
    unhealthy_after: 3
    queue_depth_threshold: 8
    prefix_index: true                     # optional cross-session KV-aware routing
    probe_interval_s: 5.0
orchestrator: { spec: my-orchestrator.yaml }     # optional: kairyu-auto routing
legacy_chat_models: [llama-70b, kairyu-auto]     # current remote/AUTO compatibility
embeddings:
  embed-test:
    backend: mock                                # deterministic built-in CPU backend
    dimensions: 384
batch: { data_dir: /var/lib/kairyu/batch, max_concurrency: 8 }  # single gateway
tenants:
  default_tenant: default
  limits:
    default:
      request_burst: 8        # initial/refill request burst, not one full minute
      token_burst: 20000      # token-bucket capacity
      max_in_flight: 8        # lease held through final streamed body byte
      interactive_priority: 0  # smaller values run first
      batch_priority: 1        # Batch API overrides client-supplied priority
```

`prefix_index` is off by default. Enabling it gives that pool one bounded,
process-local approximate text-prefix index: successful unary generation or a
stream's first backend result publishes a reusable root, which remains valid if
that stream later fails or is cancelled because prefill already landed. Later
sessionless or explicitly hinted related prompts can prefer the warm replica.
The pool still falls back to its existing session HRW, queue-depth, and
least-outstanding policies when no usable prefix is known. This option does not
start the separate exact KV-event subscriber lifecycle.

`max_concurrency` remains opt-in. By default it retains the historical active
cap and immediate saturation 429. A replica serving exactly one local Kairyu,
process-split Kairyu, or explicitly capacity-configured vLLM backend, with no
orchestrated or embedding models, may also set `admission_wait_timeout_s`;
the backend's sequence budget then caps active
requests and only the remaining `max_concurrency` allowance waits in a bounded
FIFO. A waiter receives 429 after the timeout, while a request beyond the total
bound is rejected immediately. Multiple-model servers, `ReplicaPool`, remote,
and unknown-capacity backends cannot safely map one pre-body global limit to a
model and therefore retain the historical cap; an explicitly requested but
unavailable queue logs a startup warning. Setting a timeout without
`max_concurrency` is rejected during schema validation. This class-blind queue precedes the
native priority scheduler, so interactive work cannot pass an earlier batch
waiter; keep the timeout within the deployment's TTFT policy. Metrics expose
`kairyu_admission_active_requests`, `kairyu_admission_waiting_requests`, and
`kairyu_admission_rejections_total{reason="overflow|timeout"}`.

`ttft_slo_s` is opt-in and applies to direct interactive Chat Completions after
request validation but before backend preparation. The controller admits work
predicted to meet the target, including time already spent inside gateway
ingress. A backend may execute the bounded over-target tail at the lowest
scheduler priority with `scheduling_class="batch"` only by explicitly attesting
`supports_slo_defer`: running deferred decode must be isolated from later
interactive work. Current built-in backends do not provide that isolation, so
they shed the tail instead. Further pressure is also shed with HTTP 429 and
`Retry-After: 1`. Live predictor state is exported under
`kairyu_slo_admission_*`; `null` preserves ordinary admission behavior. Bounded
waiting is a separate policy and is not implied by this setting.

Every tenant profile must keep `interactive_priority < batch_priority`.
For noisy-neighbor isolation, configure `request_burst`, `token_burst`, and
`max_in_flight` explicitly. The tenant lease is acquired outside the global
concurrency guard, released exactly once after unary/SSE completion or failure,
and also enforced per Batch API line. After validation, every generation
surface reserves a worst-case compute ceiling before shared replica placement;
`n`/`best_of`, prompt arrays, and bounded AUTO fan-out are included. Only exact
single-candidate terminal usage refunds unused capacity. Failure, disconnect,
missing usage, and multi-candidate work consume the full reservation.
`kairyu_tenant_in_flight_requests` and `kairyu_tenant_reserved_tokens` must both
return to zero after work drains.

Batch metadata operations and transactional output finalization run outside the
gateway event loop; upload chunk writes do too. Filesystem job start, cancel,
and terminal publication are serialized inside the store. Output/error rows use
a bounded background JSONL writer and publish only after the accepted rows
drain; writer saturation or I/O failure fails and rolls back the batch instead
of dropping a row. File downloads stream fixed-size chunks rather than
materializing the full file in gateway memory.
When both `batch` and `server.ttft_slo_s` are configured, consumers pause before
starting new lines while active interactive work makes the next predicted TTFT
exceed the SLO. Already-running batch generation is not preempted, and without
`ttft_slo_s` the configured `batch.max_concurrency` behavior is unchanged.

For two or more gateways, select the shared PostgreSQL batch backend instead of
mounting the filesystem store over NFS:

```yaml
batch:
  store: postgres
  dsn_env: KAIRYU_BATCH_POSTGRES_DSN
  store_id: production
  max_concurrency: 8
  poll_interval_s: 0.5
  lease_seconds: 30
```

Inject the DSN from a Secret and set `KAIRYU_BATCH_WORKER_ID` to the immutable
gateway Pod UID. PostgreSQL owns file chunks, jobs, claims, leases and fencing
tokens; a process-local submit signal only wakes the local poller. After an
owner crash, inference may run again, but only the current fenced claimant may
publish the terminal job and output.

Durable asynchronous Chat Completions use a separate request state machine and
PostgreSQL schema; they do not reuse Batch jobs or file objects:

```yaml
async_requests:
  dsn_env: KAIRYU_ASYNC_REQUEST_POSTGRES_DSN
  store_id: production
  max_concurrency: 8
  max_body_bytes: 8388608
  max_records_per_tenant: 64
  poll_interval_s: 0.5
  lease_seconds: 30
  # Disabled when omitted. Enable only after the rollout sequence below.
  request_retention_s: 604800
  audit_retention_s: 2592000
  retention_batch_size: 500
```

`POST /v1/async/chat/completions` accepts the normal non-streaming Chat body.
Optional `Idempotency-Key`, `X-Kairyu-Deadline-At` (timezone-aware ISO 8601),
headers control durable submission. Queue ordering uses the authenticated
tenant's trusted `batch_priority`; clients cannot raise their global queue
position. A 202 receipt links to tenant-scoped status, result, and cancel
routes under `/v1/requests/{id}`; list filtering uses `state` and `limit` query
parameters. Receipt, list, and status projections omit persisted input and
output bodies. Pending result reads return 202 with `Retry-After`, successful reads
return the stored Chat Completion, and other terminal states return 409. The
configured body cap applies even when the direct Chat body limit is disabled.
Async control-plane calls are authenticated and tenant-scoped. Submission uses
a separate per-tenant RPM bucket to bound durable queue growth, status polling
does not drain inference quota, and the worker charges inference RPM once when
execution starts.
`max_records_per_tenant` is a hard durable allocation: idempotent replays remain
available at the limit, while new records return 429 until retained terminal
records cross the configured request TTL and the external retention job removes
them. At that point the idempotency guarantee for the deleted request also ends;
reusing its key creates a new request and may execute inference again.
At the default 8 MiB input cap this bounds persisted input to roughly 512 MiB
per tenant; size the PostgreSQL volume/quota for the configured tenant count,
results, indexes, and audit history.

Set `KAIRYU_ASYNC_REQUEST_WORKER_ID` to the immutable gateway Pod UID when it is
available. Workers renew database-clock leases while waiting or executing, use
fencing for terminal publication, and run only publicly served direct Chat
models in v1. Local cancellation aborts generation immediately; another
gateway observes cancellation no later than its next heartbeat. AUTO
orchestration, Responses inputs, and Redis wake-up hints remain separate later
extensions.

AsyncRequest retention is deliberately not run by every gateway. Configure it
in the DeploymentSpec and invoke the bounded external job from one Kubernetes
CronJob or an equivalent scheduler. Omitted request and audit TTLs disable their
respective deletion paths; omitting both makes the job a complete no-op. A
request TTL removes only `succeeded`, `failed`, `cancelled`, and `expired`
records. Before the request row is deleted, its body-free claim audit is moved
in bounded chunks to the audit archive. The audit TTL is measured from the
original event timestamp across both active and archived audit rows, so moving
an event never restarts its retention period. Audit retention may be longer
than request retention. If `audit_retention_s` is omitted, the audit archive is
unbounded and must be capacity-planned separately.

Deletion is eventual within the scheduler interval, not an exact-time erasure
guarantee. Each transaction uses the PostgreSQL clock, locks terminal parents
before their audit rows, skips work owned by another purger, and rechecks state
and cutoff at deletion. Per-event transition and attempt counters survive both
request and audit deletion; only the current state gauge is decremented. Neither
request bodies, results, owners, request IDs, nor the DSN are written to normal
retention-job output.

Existing deployments must use this order because older binaries reject unknown
DeploymentSpec keys:

1. Deploy the new binary to every gateway while leaving all retention fields
   omitted, and verify every old Pod has drained.
2. Complete the existing `sharded-v2` telemetry migration for every store.
3. Build the retention indexes online and activate the store marker:

   ```bash
   python scripts/async_request_retention.py deployment.yaml \
     --mode prepare --apply
   ```

4. Add `request_retention_s`, `audit_retention_s`, and optionally
   `retention_batch_size` to the DeploymentSpec. Preview one capped batch:

   ```bash
   python scripts/async_request_retention.py deployment.yaml --mode purge
   ```

   Preview opens a repeatable-read, database-enforced read-only transaction and
   reports what the next bounded batch would delete or archive. It never creates
   or repairs schema objects. `--mode prepare` without `--apply`, and purge when
   both TTLs are omitted, return before opening a database connection.

5. Apply bounded batches after reviewing the structured preview:

   ```bash
   python scripts/async_request_retention.py deployment.yaml \
     --mode purge --apply --max-batches 100
   ```

Run the scheduled command with `concurrencyPolicy: Forbid`, a finite job
deadline, `restartPolicy: Never`, and a dedicated Secret/service account. Its
database role needs only connection plus the documented AsyncRequest retention
tables and index preparation privileges; do not reuse a superuser credential.
The job exits non-zero after a partial-run error and reports only aggregate
counts plus a post-batch hint that eligible work remains. The hint includes rows
temporarily locked by another purger, but concurrent commits immediately after
the observation may change it. A later CronJob run safely resumes the backlog.

PostgreSQL connect, statement, lock, idle-transaction, and TCP keepalive waits
are bounded; worker shutdown also has a bounded drain before forced task
cancellation. Successful result JSON encoding is offloaded from the gateway
event loop.

Temporary tenant request, token, or in-flight pressure returns the request to
the queue behind a durable tenant cooldown rather than making it terminal or
occupying a worker slot. This allows another tenant to claim work immediately.
A request whose token bound can never fit the tenant burst fails terminally as
a configuration/request mismatch.

Execution is at-least-once across lease takeover. Fencing prevents stale result
publication, but usage metering is not yet exactly-once; a takeover after an old
worker records usage can duplicate accounting. A future ledger schema should
use the request ID as an idempotency key.

The first deployment gate is CPU-only and reuses the disposable F1c kind
topology (three independently restartable gateways and one shared PostgreSQL):

```bash
bash scripts/kind_async_request_gate.sh
```

It runs the existing F1c shared-store gate before checking cross-gateway
idempotency and reads, remote cancellation, deadline expiry, a concurrent
768 KiB submit and responsiveness probe, lease-fenced takeover after killing
the active owner, and persistence plus new work after a PostgreSQL process
restart. The registered live smoke entrypoint is
`verification/fleet/resilience/async_request_gateway_smoke.py`. Use
`--keep-cluster` only for inspection; the default collects the
report, claim audit, Kubernetes state/events, and service logs before bounded
cluster cleanup. Evidence is written below
`bench/results/async-request-kind-live/`, which is an ignored local directory.

This is the per-slice functional gate. Multi-tenant fairness is exercised when
the tenant-policy slice is integrated; GPU model correctness, peak/soak load,
and performance acceptance remain end-of-plan integration gates. This ordering
finds distributed lifecycle defects early without repeatedly paying for GPU
runs while the control plane is still changing.

The formal F5b GPU check is
`verification/fleet/resilience/noisy_neighbor_gpu_bench.py --assert-gate`.
It compares 10x offered noisy traffic against bracketed compliant-neighbor
controls with matched accepted work; good-only latency is retained as a
secondary reference. The pinned Qwen3-32B TP8 result is
`bench/results/f5b-noisy-neighbor-qwen3-32b-tp8-2026-07-28.json`.

### OpenAI-compatible upstream capabilities

Set `options.upstream` explicitly whenever the provider is known. Kairyu
validates non-default request intent before opening the HTTP client, so a
compatibility endpoint cannot return 200 while silently discarding a field.
The `generic` default preserves older deployments but should be treated as a
migration profile.

Every profile currently declares `prompt_kinds={"text"}` because
`OpenAICompatBackend` targets Chat Completions. This is independent from the
local native and vLLM adapters, which accept `TokensPrompt` directly, and from
Kairyu's `/v1/completions` token-array input. A remote Chat-compatible worker
therefore rejects token and multimodal prompts before creating its HTTP client;
it never stringifies IDs or flattens media. Prompt kinds participate in the
same immutable capability key as sampling/tool policy.

Prompt content must never be placed in `extra_args`. Kairyu reserves alternate
text/token/embed/multimodal carrier names and `cache_salt` at the common
`SamplingParams` boundary, independently of a provider allowlist. Unknown
top-level or nested Chat fields are also rejected before dispatch. Use
`PromptInput` for content and `CacheHint` for native affinity; a legacy backend
without `validate_request` remains compatible only with plain string prompts.

`GenerationRequest.parallel_tool_calls` is the typed forwarding path for the
`openai`, `kairyu`, and `vllm` profiles. The `openai` profile also preserves its
older `SamplingParams.extra_args.parallel_tool_calls` compatibility path, but a
request cannot specify both sources. Unsupported upstreams do not receive the
hint; the public Chat/Responses boundary still checks the generated call count
and fails closed when `parallel_tool_calls=false`. A verified custom `generic`
profile may opt in with `capabilities.parallel_tool_calls: true`; named
provider profiles may be narrowed with `false` but not broadened beyond their
built-in contract.

| `upstream` | Portable request controls | Provider-specific notes |
|---|---|---|
| `openai` | OpenAI Chat Completions sampling, logprobs, structured output, tools | Emits the canonical `max_completion_tokens`; forwards typed `parallel_tool_calls` and retains the legacy `extra_args` spelling; allows `reasoning_effort` and `service_tier` in `extra_args`. |
| `anthropic` | temperature (0–1), top-p, one completion, max tokens, stop, non-strict tools | Rejects penalties, seed, logprobs, `response_format`, and strict tool schemas because the [Anthropic compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) documents them as ignored. Anthropic recommends its native API for production features. |
| `gemini` | max tokens, structured output, and non-strict tools | Allows `reasoning_effort` and the documented `extra_body.google` extension object. Sampling controls vary across Gemini model families, so fields not guaranteed by the [Gemini OpenAI compatibility contract](https://ai.google.dev/gemini-api/docs/openai) fail closed unless a pinned deployment declares a verified custom contract. |
| `kairyu` | OpenAI controls plus `top_k`, `min_p`, `repetition_penalty`, `stop_token_ids`, `min_tokens`, `ignore_eos`, `skip_special_tokens`, and signed-int64 `priority` | Use for gateway-to-Kairyu replica traffic. These extensions and the bounded interactive/batch class hint are typed and preserved through the receiving HTTP boundary into native scheduler admission. `skip_special_tokens` defaults to `true` and is isolated per request; `false` exposes otherwise-visible registered specials, but an ID that actually terminates on EOS or a stop token remains hidden under both values. |
| `vllm` | OpenAI controls plus result-preserving [vLLM Chat extensions](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and `priority` | Shares the Kairyu profile's `skip_special_tokens` control. Smaller priority values run first. Kairyu's local vLLM adapter requires `scheduling_policy=priority` so the field cannot be silently ignored; a separately operated remote vLLM server must enable the same policy. `prompt_logprobs` fails closed until Kairyu's result/API types can return the upstream prompt distribution. |

Example provider configurations:

```yaml
engines:
  hosted-openai:
    backend: openai
    options:
      base_url: https://api.openai.com/v1
      model: gpt-4o
      api_key_env: OPENAI_API_KEY
      upstream: openai
  hosted-anthropic:
    backend: openai
    options:
      base_url: https://api.anthropic.com/v1
      model: claude-sonnet-4-5
      api_key_env: ANTHROPIC_API_KEY
      upstream: anthropic
  hosted-gemini:
    backend: openai
    options:
      base_url: https://generativelanguage.googleapis.com/v1beta/openai
      model: gemini-2.5-flash
      api_key_env: GEMINI_API_KEY
      upstream: gemini

# Hosted providers expose no local tokenizer metadata, and the current remote
# adapter cannot preserve a Kairyu pre-rendered prompt through their own chat
# template. This example therefore selects the compatibility renderer explicitly.
legacy_chat_models: [hosted-openai, hosted-anthropic, hosted-gemini]
```

Vendor extensions use `SamplingParams(extra_args=...)`. Each key must be in
the selected preset's allowlist or added by deployment configuration:
for example, Gemini thinking configuration is
`extra_args={"extra_body": {"google": {"thinking_config": {...}}}}`.

```yaml
options:
  upstream: generic
  capabilities:
    allow_extra_args: [vendor_cache]
    allow_sampling_fields: [best_of]  # only after verifying this endpoint executes it
```

Core keys such as `model`, `messages`, `temperature`, `tools`, and `stream`
cannot be allowlisted or overwritten through vendor extensions.
`response_format` remains a reserved canonical intent in the existing
`SamplingParams.extra_args` representation and cannot be reclassified by the
vendor allowlist. Capability overrides are validated while loading deployment
and orchestrator specs; unsupported non-neutral request values return a
pre-dispatch HTTP 400.
Additive sampling overrides are limited to the `generic` custom-provider
profile. Named provider presets may be narrowed but not broadened, preventing
configuration from re-enabling fields that the provider documents as ignored.

### Responses API and Codex

`POST /v1/responses` supports unary and canonical typed SSE responses for text
and function calls. Flat OpenAI function definitions and Codex namespace
definitions are accepted; `function_call_output` and tenant-scoped
`previous_response_id` continue the tool loop without resending earlier items.
Only successful responses are stored. `store: false` disables continuation.
The in-process store is bounded and intentionally not shared across gateways,
so route a continued response to the same gateway or omit
`previous_response_id` and send the full input history in an HA deployment.

Codex can use Kairyu as a custom Responses provider without source changes:

```bash
KAIRYU_BASE_URL=http://127.0.0.1:8000/v1 \
KAIRYU_MODEL=qwen3-32b \
KAIRYU_API_KEY=local \
scripts/codex_responses_smoke.sh
```

The script creates an ephemeral Codex run with `wire_api="responses"` and a
read-only sandbox. Its default `KAIRYU_SMOKE_MODE=tool` requires a real `pwd`
command event, its tool result, and a final message containing `PASS`;
`KAIRYU_SMOKE_MODE=text` selects a text-only wire smoke. Kairyu accepts Codex
function namespaces and its disabled web-search declaration (including its
search configuration fields); enabled hosted web search remains unsupported.
`background`, Conversations API objects, hosted prompt templates, moderation,
automatic truncation, context management, `max_tool_calls`, and response
top-logprobs fail before model dispatch. This explicit rejection boundary keeps
the accepted compatibility surface truthful. `service_tier` supports only the
neutral `auto` selection; explicit paid/priority tiers fail rather than being
echoed as executed. Codex reasoning/include metadata is accepted for wire
compatibility but Kairyu emits no reasoning or encrypted-reasoning output item;
echoed `reasoning` history items and Codex-internal passthrough item fields are
accepted and dropped. `text.verbosity` is applied as a model instruction, not
claimed as a provider quality-of-service tier.

Since issue #530, `/v1/responses` serves every chat model `/v1/models`
advertises: orchestrated (`kairyu-auto*`) models delegate to the Chat
Completions orchestration contract, so `public_models` topologies that hide
every L1 pool behind a single AUTO model work end to end (this includes the
Terminal-Bench gateway shape). Additional Codex-derived behavior:

- **Buffered streams never go silent.** Any streamed request with tools
  replies `response.created`/`response.in_progress` immediately and emits
  `: keep-alive` comments while generation runs; Codex aborts SSE streams
  that stay silent for 300 s. Failures after the stream opens surface as
  `error` + `response.failed` events.
- **WebSocket upgrades get 426.** Harbor/Terminal-Bench repoints Codex's
  built-in `openai` provider (`openai_base_url` in `config.toml`), which
  tries WebSocket first; 426 makes Codex fall back to HTTPS immediately and
  silently. A custom `model_providers` entry defaults to HTTPS-only and
  skips the attempt entirely.
- **Remote compaction v2 is served.** Against an OpenAI-shaped base URL,
  Codex auto-compacts near its assumed context budget by sending a terminal
  `compaction_trigger` input item; Kairyu answers with one `compaction`
  output item whose `encrypted_content` is an opaque self-issued token, and
  echoed compaction items decode back into the summarized context. Tokens are
  AES-256-GCM sealed and bound to the authenticated tenant; forged, modified,
  or cross-tenant tokens fail closed. Set
  `server.responses_compaction_secret_env` to the name of an environment
  variable containing at least 32 random bytes. When unset, each process uses an
  ephemeral key and tokens do not survive a restart or gateway hop.
- **`function_call_output.output` arrays** are accepted (text parts
  concatenated); image/audio tool-output parts are rejected explicitly.
- **Codex never retries 429.** A tenant admission 429 fails the Codex turn
  outright, so size tenant admission generously (or leave it unset) on
  deployments that serve Codex/Terminal-Bench traffic.

Operational notes:

- **Session affinity is the cache layer.** Clients that send the OpenAI
  `user` field (or an `X-Session-ID` header) keep a conversation on the
  replica holding its warm radix-KV prefix. Watch
  `kairyu_pool_decisions_total{reason="session_affinity"}` — a low share on
  multi-turn traffic means clients aren't sending session identity.
- **Gateway HA**: run gateways behind an L7 load balancer that consistently
  hashes `X-Session-ID`. Use `batch.store: postgres` for cross-gateway files and
  jobs. The filesystem store remains single-gateway only and must not be shared
  over NFS. Gateways serving remote compaction must also share the same secret
  environment value so an authenticated token issued by one gateway can be
  opened by another.
- **Two GPU nodes acting as one model** (TP/PP/P-D across nodes) is an
  engine-layer concern configured by `ClusterSpec` per `docs/gpu-runbook.md`
  §7; the gateway still sees one OpenAI endpoint per coherence domain.
- **Embedding model IDs are explicit.** Each `embeddings:` key is listed by
  `/v1/models`, routes only to its configured backend, and must not collide
  with an engine, pool, or orchestrator name. Unknown IDs return
  `model_not_found` without execution or usage accounting.

### Production embeddings and Open WebUI RAG

Install the CPU production backend with `uv sync --extra embeddings`. It uses
FastEmbed/ONNX Runtime rather than PyTorch and owns one warmed model session
plus a bounded worker pool:

```yaml
embeddings:
  embed-small:
    backend: fastembed
    model: sentence-transformers/all-MiniLM-L6-v2
    model_path: /opt/kairyu/models/all-MiniLM-L6-v2
    revision: 5f1b8cd78bc4fb444dd171e59b18f3a3af89a079
    model_sha256: bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5
    provenance_sha256: 57246a4990eb0f08755df06ba57c1fec161032bd588332435e89c7ece244639c
    dimensions: 384
    batch_size: 64
    threads: 2
    max_concurrency: 2
```

`model_path` must be a prefetched local snapshot with
`MODEL_PROVENANCE.json`. Startup verifies the configured repository revision,
FastEmbed catalog dimensions, recorded model hash, and the actual ONNX SHA-256,
then runs one normalized-vector warmup without network access. Until that
finishes `/readyz` is 503; an integrity or load failure is fatal and also makes
`/health` return 503. Shutdown drains accepted inference work before releasing
the executor and model session. Successful requests report tokenizer-derived
exact input usage, so tenant admission refunds the difference between its
pre-dispatch UTF-8 work bound and actual token work.

The repository Dockerfile keeps this dependency and model out of the default
image. Set `KAIRYU_EMBEDDINGS=1` and the four immutable model build arguments
shown in `deploy/compose/docker-compose.webui.yaml` to build the RAG image.
That Compose topology points both chat and embedding traffic at Kairyu:

```bash
docker compose -f deploy/compose/docker-compose.webui.yaml up --build
```

Its Open WebUI configuration selects `embed-small`, bounds embedding batches
and concurrent requests, and disables query generation, full-context bypass,
and reranking. Reranking is optional in P-C3 and is deliberately deferred;
retrieval itself remains mandatory and is exercised by
`scripts/webui_smoke.sh`, which uploads a document, verifies its retrieved
canary, obtains a citation-bearing Kairyu answer, and repeats retrieval after
restarting only Kairyu. Existing Open WebUI data volumes may retain prior RAG
settings in the application database; confirm the admin RAG settings or use a
fresh/migrated volume when adopting this configuration.

### Qwen3-VL image chat through stock vLLM

Image chat is an opt-in GPU topology, not part of the CPU-safe default stack.
The overlay starts the immutable Qwen3-VL-32B-Instruct revision on stock vLLM
with tensor parallelism across eight NVIDIA GPUs, then exposes it as the
`qwen3-vl-32b` Kairyu pool. Its `hermes` tool parser matches the pinned
model's `<tool_call>{JSON}</tool_call>` chat-template format. Open WebUI can
inject built-in tools, and vLLM normalizes an omitted tool choice to `auto`;
the topology accepts that ordinary request instead of silently stripping it.
P-C4 validates standard image answers, not a multimodal native-tool
continuation:

```bash
docker compose \
  -f deploy/compose/docker-compose.webui.yaml \
  -f deploy/compose/docker-compose.webui-vlm.yaml \
  up -d --build --wait
```

VLMs have no deployment-preflight exception. The checked-in
`config-vlm.yaml` explicitly lists `qwen3-vl-32b` in `legacy_chat_models` for
its text-only compatibility path. Image-bearing requests bypass that renderer:
Kairyu preserves message roles and content-part order, and the stock vLLM
replica owns the Qwen processor/template. Configuring a Kairyu
`chat_templates` entry for that remote backend is rejected at startup rather
than applying the model template twice. The gateway accepts only inline PNG,
JPEG, or WebP data URLs and verifies the decoded raster before admission; remote
URLs, local paths, invalid MIME/magic, malformed or animated images,
decompression bombs, and configured byte/pixel/dimension/aspect-ratio overages
return a controlled OpenAI-compatible error without upstream media I/O.

The production overlay permits one image up to 8 MiB and 2,097,152 pixels,
uses Qwen's matching 65,536–2,097,152-pixel processor range, and reserves the
complete 8,192-token model context because image, text, roles, and template
tokens share that context. Set `KAIRYU_VISION=1` when building a custom image
so Pillow is installed for strict raster verification.

Run `scripts/webui_vlm_smoke.sh` on an eight-GPU host. It generates
deterministic RED/BLUE PNGs, proves different correct model answers and exact
unary/stream usage through Kairyu, rejects a metadata-service URL before
dispatch, and uses Open WebUI's normal authenticated file-upload path in the
pinned Playwright browser. The Open WebUI backend owns the uploaded file and
converts it to an inline data URL only when forwarding the completion to
Kairyu.

## 5. Rolling model update (gate C7)

Weights update = rolling replica restart; there is no hot swap (m7 §3).
Do not stop a still-eligible replica and rely on a failed request to eject it.
Use a drain-first, partitioned StatefulSet rollout:

1. Set `updateStrategy.type=RollingUpdate` and hold
   `rollingUpdate.partition` at the current replica count before staging the
   new image or weights. For the F1b restart drill, invoke
   `kubectl rollout restart` once while that partition still holds every Pod.
2. Starting at the highest ordinal, call the replica's `/admin/drain` through
   the authenticated Kubernetes Pod proxy. Wait for both the ready,
   non-terminating EndpointSlice UID to disappear and the gateway membership
   record to mark that UID ineligible. Also wait for already placed work to
   complete; do not drain the remaining replicas.
3. Lower the partition by one. The StatefulSet may now replace only that
   ordinal. Wait for a different Pod UID at the update revision, the replica's
   `/readyz` to return 200, and the gateway to report the new UID eligible.
4. Repeat the same bounded step for the next ordinal. On timeout, stop without
   advancing the partition; do not repair the rollout with direct Pod deletes.

The no-new-work boundary is the later of gateway ineligibility and
EndpointSlice withdrawal. A local `/admin/drain` acknowledgement alone is not
proof that distributed routing has converged. Requests placed before that
boundary may finish normally, but no placement after it may select the old UID,
and the partition is not released until its outstanding count is zero.

The automated F1b rehearsal freezes this sequence in a checked-in driver:
exactly 100 old UIDs become 100 disjoint new UIDs, retry count is zero, every
offered request returns 2xx, and raw request, placement, membership, rollout,
readiness, Pod, and EndpointSlice evidence is independently replayed. Once the
job starts, no human or cluster-installed rollout Operator chooses or repairs a
step. Pull-request smoke validates the state machine at reduced scale; only the
clean exact-head 100-replica formal kind artifact satisfies F1b.

`scripts/compose_smoke.sh` remains the C1–C3 kill/eject/recover prerequisite.
Its intentional fault-trigger request is not a drain-first rolling update and
is not C7/F1b acceptance evidence.

## 6. Observability

- `/metrics` (Prometheus): request counts by model/status, latency, and
  ledger-reconcilable usage executions plus prompt/completion/cached/uncached
  tokens by tenant,
  histograms, per-replica outstanding/health, pool decision counts, batch
  job states. Priority-enabled replicas also expose bounded
  `kairyu_scheduler_priority_events_total`,
  `kairyu_scheduler_queue_depth`, and
  `kairyu_scheduler_queue_high_watermark` series by model and
  interactive/batch class. Tenant-enabled gateways additionally expose
  `kairyu_tenant_admission_total` by bounded tenant/source/decision/reason and
  `kairyu_tenant_in_flight_requests`. AsyncRequest gateways expose shared-store
  `kairyu_async_request_queue_depth`,
  `kairyu_async_request_oldest_queued_age_seconds`, lifecycle state,
  transition, attempt, and snapshot-health series. Queue, state, transition,
  and attempt values are shared PostgreSQL-backed snapshots: de-duplicate those
  series across gateway targets with `max without(instance, pod)` rather than
  `sum`. Snapshot health is gateway-local and must not use `max`; alert per
  target, or when `min without(instance, pod)
  (kairyu_async_request_metrics_snapshot_success) == 0`. During a failed
  refresh the shared series retain that gateway's last good snapshot. Scrape
  every gateway and replica.

  A new empty AsyncRequest store activates sharded telemetry automatically.
  For a pre-existing registry, deploy the compatible code first; the first v2
  Pod atomically advances all registry rows so already-running v1 Pods can
  drain, while a v1 Pod restart fails closed instead of replacing shared
  trigger functions. Telemetry reports snapshot health `0` until the one-time
  backfill. Drain AsyncRequest writers for every store, then run the following
  once per store:

  ```bash
  python scripts/async_request_metrics_migration.py \
    --store-id "$STORE_ID" --mode backfill \
    --maintenance-window-confirmed
  ```

  The DSN is read from `KAIRYU_ASYNC_REQUEST_POSTGRES_DSN`. During a mixed
  rollout an existing legacy trigger remains available to running old Pods while the
  new, separately named trigger writes 64 sharded counters. After every store
  is backfilled and all old Pods are gone, run the same command with
  `--mode finalize-legacy` to remove the legacy single-row writer. Neither
  operation runs automatically during normal Pod startup.
- With a versioned `pricing:` section, `/admin/usage.csv` snapshots the local
  immutable ledger and exports tenant charges for a `[start_ts,end_ts)` period.
  The CSV carries source SHA-256, price-sheet version, Decimal unit rates,
  cached/uncached/output components, tenant discount, and total. Corrupt or
  truncated input fails closed with `invoice_ledger_invalid`.
- JSON logs on stdout (one access line per request, prober/batch events);
  ship with the log collector of your choice.
- `/readyz` is the LB health check for gateways; `/health` is the container
  healthcheck on every node.

## 7. DC–cloud interconnect

Only the gateway tier needs to be reachable from the cloud edge; size the
link for request/response payloads (tokens, not tensors — KV never crosses
it). A Direct Connect / ExpressRoute class link gives predictable latency
for TTFT-sensitive SLOs; keep an IPsec VPN as the fallback path. Restrict
the edge→DC path to the gateway port; replica ports stay DC-internal.

## 8. Appendix: Kubernetes (Helm chart + kind manifests)

If a revisit trigger from §3 fires, the migration shape is: the
DeploymentSpec as a ConfigMap, `/health`→livenessProbe,
`/readyz`→readinessProbe, a Service in front of gateway pods, and the NVIDIA
GPU operator on replica nodes (pin one replica pod per node with
`nodeSelector` + `resources.limits.nvidia.com/gpu`). A Helm chart
(`deploy/helm/kairyu/`, GPU overlay `values-gpu.yaml`) and kind kustomize
manifests (`deploy/kind/f1a|f1b|f1c/`, gateway Deployment + replica
StatefulSet as in §5) are shipped and exercised in CI by
`scripts/helm_integration.sh`, `scripts/kind_smoke.sh`, and the F1a/F1b/F1c
gate workflows. The compose topology remains the recommended small-fleet
path (m7 D2).

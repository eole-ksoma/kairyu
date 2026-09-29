# Kairyu Helm chart

This chart deploys one Kairyu gateway/replica service. The default values are a CPU-safe
smoke configuration backed by the mock engine:

```console
helm install kairyu deploy/helm/kairyu
```

Production and PoC releases should pin the image by digest. When `image.digest` is set,
the chart renders `repository@digest` and ignores the tag for the Pod image reference:

```console
helm upgrade --install kairyu deploy/helm/kairyu \
  --set-string image.repository=harbor.example.internal/ai/kairyu \
  --set-string image.digest=sha256:<64-lowercase-hex-characters>
```

Set `workloadRole` to `gateway` or `replica` on split-role releases. The value is emitted
as the standard `app.kubernetes.io/component` label so NetworkPolicy and monitoring can
select roles without depending on a release name.

The checked-in GPU overlay requests one NVIDIA GPU, selects the `pcie-gddr` node profile,
uses the `nvidia` RuntimeClass, mounts model files read-only, and starts the real `kairyu`
engine:

```console
helm install kairyu deploy/helm/kairyu \
  -f deploy/helm/kairyu/values-gpu.yaml
```

## GPU prerequisites

The cluster must have NVIDIA drivers and the NVIDIA GPU Operator (or equivalent device
plugin) installed. It must expose `nvidia.com/gpu`, provide a RuntimeClass named `nvidia`,
and label the target nodes with `kairyu.dev/gpu-profile=pcie-gddr` (or the selector must be
overridden for the cluster).

Ordinary CI only lints and renders the GPU manifest. It has no GPU node and does not run
the resulting pod.

## Shared batch store

The default filesystem BatchStore is intentionally single-gateway. For a
multi-gateway release, configure `batch.store: postgres` in `config` and inject
the matching DSN environment variable from an existing Secret:

```yaml
batchPostgres:
  enabled: true
  secretName: kairyu-batch-postgres
  secretKey: dsn
  dsnEnvName: KAIRYU_BATCH_POSTGRES_DSN
```

The chart also injects each Pod's immutable UID as
`KAIRYU_BATCH_WORKER_ID`, which binds PostgreSQL claims and fencing audit rows
to the actual gateway instance. The chart does not create or own PostgreSQL.

### Attention backend

The checked-in `pcie-gddr` overlay targets RTX PRO 6000 Blackwell (SM120) nodes and uses
`attentionBackend: auto`. Retained Qwen3-32B TP4/TP8 evidence currently resolves that
profile to FlashInfer; keeping the overlay on `auto` leaves the profile policy in one
place. Operators can render any public backend explicitly:

```console
helm install kairyu deploy/helm/kairyu \
  -f deploy/helm/kairyu/values-gpu.yaml \
  --set-string attentionBackend=flashattention4
```

The strict schema accepts these values:

| value | behavior |
|---|---|
| `""` | Chart default. Omit `KAIRYU_ATTENTION_BACKEND`; runtime selection behaves as `auto`. |
| `auto` | Emit the automatic policy explicitly. It uses the stable profile choice unless retained profile-specific evidence justifies promotion, and falls back to torch if that optional choice cannot be constructed. |
| `torch` | Portable torch implementation for prefill and decode. |
| `flashinfer` | FlashInfer paged prefill and decode. |
| `flashattention3` | Official upstream FA3 SM90 prefill plus FlashInfer paged decode. |
| `flashattention4` | FA4 prefill plus FlashInfer paged decode. |

Explicit selections are strict. A missing package, unsupported GPU, or unsupported tensor
shape makes the replica fail before serving instead of silently choosing another backend.
The `/backends` response exposes the resolved prefill/decode components and selection
source.

FA4 consumes Kairyu's page table directly on SM90/SM100/SM110. Its SM120 path preserves
the same page identities and materializes only the selected pages device-to-device before
prefill. Images built with `Dockerfile.cuda` include the pinned CUDA 13 variant,
`flash-attn-4[cu13]==4.0.0b24`. FA3 images must build the official upstream
`hopper/` package at tag `fa4-v4.0.0.beta24`, commit
`849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`; see the repository README for
the exact build commands. Without representative SM90 hardware, FA3's fake API
contract verifies strict fail-closed behavior only; it is not a performance or
default-selection claim.

## Model storage

Model storage is disabled by default. When enabled, configure exactly one source:

- `hostPath`: an absolute path already present on every eligible GPU node; or
- `pvcName`: the name of an existing PersistentVolumeClaim in the release namespace.

The GPU overlay uses a host path and expects the checkpoint directory to exist at
`/models/checkpoint` on the selected node. It mounts `/models` read-only at `/models` in
the container:

```yaml
modelStorage:
  enabled: true
  pvcName: ""
  hostPath: /models
  mountPath: /models
```

To use an existing PVC instead, keep the same directory layout inside the volume and
override the source without editing the Deployment template:

```console
helm install kairyu deploy/helm/kairyu \
  -f deploy/helm/kairyu/values-gpu.yaml \
  --set-string modelStorage.pvcName=kairyu-models \
  --set-string modelStorage.hostPath=
```

In both cases, the mounted storage must contain `/models/checkpoint` as seen from the
container because the GPU DeploymentSpec sets `model_path: /models/checkpoint`. Both
`hostPath` and `mountPath` must be absolute. The values schema rejects enabled storage
with no source, both sources at once, relative paths, and unknown storage fields.

## Rollout, shutdown, and monitoring

The chart defaults to a long startup probe window for model loading, disables service
account token mounting only when operators set `automountServiceAccountToken: false`,
applies the runtime-default seccomp profile, drops Linux capabilities, and triggers a
rollout whenever the embedded DeploymentSpec changes. The token default remains `true`
for compatibility with `kubernetes_endpoints` discovery; static pools should disable it.

For a one-Pod GPU release pinned to one node, use `strategyType: Recreate`; a rolling
update cannot schedule a second Pod when only one GPU is allocated to that release.
Enable `gracefulDrain.enabled` to POST `/admin/drain` before termination and wait for
endpoint propagation. Keep `terminationGracePeriodSeconds` greater than the propagation
delay by at least 10 seconds; the chart rejects a shorter grace period. The hook uses the
Python runtime already present in both Kairyu images. For a
deployment with `server.admin_keys_env`, set `gracefulDrain.auth.secretName` and
`secretKey` to an existing Secret containing one admin key; the hook sends it as a Bearer
token. The DeploymentSpec must set
`server.admin_keys_env: KAIRYU_HELM_DRAIN_TOKEN` so the server reads the same injected
value. Keyless trusted-mesh deployments leave `secretName` empty.

Set `serviceMonitor.enabled: true` when the Prometheus Operator CRDs are installed. The
chart then scrapes `/metrics` on the named Service port; use `serviceMonitor.labels` for
the cluster's Prometheus selector (for example `release: kube-prometheus-stack`).

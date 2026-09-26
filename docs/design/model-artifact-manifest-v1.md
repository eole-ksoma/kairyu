# Model artifact manifest v1

Status: WP4.1 implemented; WP4.2 consumes this contract for verified node-cache
fills. Deployment wiring and WP4.3+ policy remain open.

## Purpose and boundary

The manifest gives every model tree one content-derived identity before a
deployment, cache agent, or Runner may refer to it. A Runner request and GitOps
configuration use the manifest SHA-256, never a mutable repository tag or local
path.

WP4.1 validates signed metadata. It does not download blobs or claim that local
bytes match the declared per-blob digests. WP4.2 now performs that verification
while filling a temporary tree and publishes the tree atomically only after
every digest matches. See `docs/design/node-model-cache-v1.md`.

## Contracts

`kairyu.artifacts` exposes strict, immutable Pydantic v2 schemas:

- `ModelArtifactManifest`: model/upstream identity, architecture,
  quantization, tokenizer, license, environment and GPU constraints, resource
  estimates, signer identity, and the complete blob tree.
- `SignedModelArtifactManifest`: the manifest, its canonical digest, and an
  Ed25519 signature.
- `ModelArtifactTrustStore`: public signer keys and the environments each key
  may authorize.
- `ModelArtifactAdmissionRequest`: the deployment identity plus exact manifest
  digest, model identity, environment, and GPU profile declared by GitOps.
- `ModelArtifactAdmission`: the narrow verified binding passed to later cache
  and Runner control planes.

All schemas reject unknown fields. Integer resource and blob sizes reject bools
and coercion. Collections with set semantics must be sorted and unique so one
semantic manifest cannot acquire multiple identities through reordering.

### Required manifest fields

The v1 schema requires:

- `model_id` and `model_revision`;
- `upstream_repository` and a 40- or 64-character lowercase hexadecimal immutable
  `upstream_revision`;
- `architecture` and structured `quantization` metadata;
- an immutable tokenizer repository/revision/digest;
- a license identifier and license files present in the declared blob tree;
- sorted, non-empty `required_gpu_profiles` and `approved_environments`;
- disk, RAM, and VRAM estimates, with disk at least the sum of blob sizes;
- the trusted `signer_key_id`;
- sorted, unique safe relative POSIX paths, byte sizes, and SHA-256 for every
  blob; and
- `file_tree_sha256` derived from the complete ordered blob list.

Absolute paths, empty segments, `.` / `..`, backslashes, NUL, duplicate paths,
components over 255 UTF-8 bytes, paths over 4096 UTF-8 bytes, uppercase or
malformed digests, mutable upstream revisions, and inconsistent tree/resource
metadata are rejected.

## Canonical identity and signature

The file-tree digest is SHA-256 over UTF-8 JSON for the blob array with object
keys sorted, no insignificant whitespace, and the already validated path order.

The manifest digest uses the same JSON profile over the complete manifest:

```text
sha256(json.dumps(manifest, sort_keys=true, separators=(",", ":"),
                  ensure_ascii=false).utf8)
```

Ed25519 signs the domain-separated bytes:

```text
"kairyu:model-artifact-manifest:v1\n" || canonical_manifest_bytes
```

The signed envelope also carries `manifest_digest`. Verification recomputes it
before looking up the signer and then verifies the signature with the trust
store public key. Base64 must be canonical and decode to exactly 32 bytes for a
public key or 64 bytes for a signature. Input JSON is bounded and duplicate
object keys are rejected.

Private-key loading and key custody are intentionally outside Kairyu. The
library signing helper accepts an already loaded `Ed25519PrivateKey` so a build
system or HSM integration can produce envelopes without adding private-key file
handling to the serving CLI.

## CLI and GitOps admission

Offline signature verification:

```console
kairyu artifact validate signed-manifest.json --trust-store trust-store.json
```

GitOps stores the deployment intent as a separate versioned JSON object. The
CLI does not accept field-by-field overrides, so policy evaluates one reviewable
desired-state artifact:

```json
{
  "schema_version": "kairyu-model-admission-v1",
  "deployment_id": "production/model-service",
  "manifest_digest": "0123...cdef",
  "model_id": "org/model",
  "model_revision": "release-2026-09",
  "environment": "production",
  "gpu_profile": "h100-sxm"
}
```

Exact deployment-intent admission:

```console
kairyu artifact admit signed-manifest.json \
  --trust-store trust-store.json \
  --request model-deployment.json
```

Admission succeeds only when:

1. canonical digest, signer trust, and Ed25519 signature verify;
2. the GitOps digest, model ID, and model revision exactly match the manifest;
3. both the signed manifest and the signer's trust-store entry authorize the
   target environment; and
4. the requested GPU profile appears in the signed required profile set.

Any mismatch returns exit status 1 and a bounded diagnostic without a
traceback. The admission function is side-effect free, enabling the IaC layer
to call it in CI, a policy job, or a Kubernetes admission webhook without
granting it storage or runtime access.

This is the WP4.1 admission primitive. The GitOps repository must make the
request file part of the rendered deployment and gate its merge or apply with
this command. DeploymentSpec/Helm/controller consumption of the resulting
binding remains explicit wiring work; until that wiring lands, a successful
admission must not be treated as proof that a running workload uses the digest.

## Fail-closed handoff to WP4.2+

Downstream components must accept `ModelArtifactAdmission` plus the signed
envelope or a durably equivalent verified record. They must not reconstruct an
authorization from a model name, repository URL, tag, path, or unverified
manifest object. WP4.2 cache publication re-runs admission, verifies each blob's
size and digest in a private staging tree, and atomically publishes only the
complete tree. Durable residency/index state, eviction protection, quarantine,
and pre-stage status remain WP4.3–WP4.7.

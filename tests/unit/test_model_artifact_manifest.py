"""WP4.1 signed model-artifact manifest and GitOps admission coverage."""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from kairyu.artifacts import (
    InvalidModelArtifactError,
    ModelArtifactAdmissionError,
    ModelArtifactAdmissionRequest,
    ModelArtifactBlob,
    ModelArtifactManifest,
    ModelArtifactQuantization,
    ModelArtifactResourceEstimate,
    ModelArtifactTokenizer,
    ModelArtifactTrustStore,
    SignedModelArtifactManifest,
    TrustedModelSigner,
    admit_model_artifact,
    canonical_model_manifest_bytes,
    load_model_artifact_admission_request,
    load_model_artifact_trust_store,
    load_signed_model_artifact,
    model_file_tree_digest,
    model_manifest_digest,
    sign_model_artifact_manifest,
    verify_model_artifact_manifest,
)
from kairyu.entrypoints import cli

_REVISION = "1" * 40
_TOKENIZER_REVISION = "2" * 40


def _private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def _public_key_base64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _manifest(**updates) -> ModelArtifactManifest:
    blobs = (
        ModelArtifactBlob(path="LICENSE", size_bytes=12, sha256="a" * 64),
        ModelArtifactBlob(
            path="model/model-00001.safetensors",
            size_bytes=1024,
            sha256="b" * 64,
        ),
        ModelArtifactBlob(path="tokenizer.json", size_bytes=256, sha256="c" * 64),
    )
    values = {
        "model_id": "org/model",
        "model_revision": "release-2026-09",
        "upstream_repository": "https://example.invalid/org/model",
        "upstream_revision": _REVISION,
        "architecture": "ExampleForCausalLM",
        "quantization": ModelArtifactQuantization(
            method="fp8",
            format="safetensors",
            bits=8,
        ),
        "tokenizer": ModelArtifactTokenizer(
            repository="https://example.invalid/org/model",
            revision=_TOKENIZER_REVISION,
            sha256="d" * 64,
        ),
        "license_id": "Apache-2.0",
        "license_files": ("LICENSE",),
        "required_gpu_profiles": ("h100-sxm", "rtx-pro-6000-blackwell"),
        "approved_environments": ("production", "staging"),
        "signer_key_id": "release-key-2026",
        "resources": ModelArtifactResourceEstimate(
            disk_bytes=2048,
            ram_bytes=4096,
            vram_bytes=8192,
        ),
        "files": blobs,
        "file_tree_sha256": model_file_tree_digest(blobs),
    }
    values.update(updates)
    return ModelArtifactManifest(**values)


def _trust_store(
    private_key: Ed25519PrivateKey,
    *,
    environments: tuple[str, ...] = ("production", "staging"),
) -> ModelArtifactTrustStore:
    return ModelArtifactTrustStore(
        signers=(
            TrustedModelSigner(
                key_id="release-key-2026",
                public_key_base64=_public_key_base64(private_key),
                approved_environments=environments,
            ),
        )
    )


def _signed() -> tuple[
    SignedModelArtifactManifest,
    ModelArtifactTrustStore,
]:
    private_key = _private_key()
    return (
        sign_model_artifact_manifest(_manifest(), private_key),
        _trust_store(private_key),
    )


def _admission_request(
    envelope: SignedModelArtifactManifest,
    **updates,
) -> ModelArtifactAdmissionRequest:
    values = {
        "deployment_id": "production/model-service",
        "manifest_digest": envelope.manifest_digest,
        "model_id": "org/model",
        "model_revision": "release-2026-09",
        "environment": "production",
        "gpu_profile": "h100-sxm",
    }
    values.update(updates)
    return ModelArtifactAdmissionRequest(**values)


def test_signed_manifest_verifies_and_has_stable_digest():
    envelope, trust_store = _signed()

    verified = verify_model_artifact_manifest(envelope, trust_store)

    assert verified.manifest_digest == model_manifest_digest(envelope.manifest)
    assert verified.signer_key_id == "release-key-2026"
    assert canonical_model_manifest_bytes(envelope.manifest) == json.dumps(
        envelope.manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def test_v1_canonical_json_digest_and_signature_golden_vector():
    blobs = (ModelArtifactBlob(path="LICENSE", size_bytes=0, sha256="3" * 64),)
    manifest = ModelArtifactManifest(
        model_id="組織/模型",
        model_revision='r"1',
        upstream_repository="https://例え.invalid/model",
        upstream_revision="0" * 40,
        architecture="Arch\\Name",
        quantization=ModelArtifactQuantization(method="none", format="safetensors"),
        tokenizer=ModelArtifactTokenizer(
            repository="https://例え.invalid/tokenizer",
            revision="1" * 40,
            sha256="2" * 64,
        ),
        license_id="Apache-2.0",
        license_files=("LICENSE",),
        required_gpu_profiles=("gpu-α",),
        approved_environments=("staging",),
        signer_key_id="key-1",
        resources=ModelArtifactResourceEstimate(
            disk_bytes=0,
            ram_bytes=0,
            vram_bytes=0,
        ),
        files=blobs,
        file_tree_sha256="66e9d1a0d3ac78637b75bba7d403864fef7d6883cf7144afec43d0eebf9a443b",
    )
    expected_canonical = (
        b'{"approved_environments":["staging"],"architecture":"Arch\\\\Name",'
        b'"file_tree_sha256":"66e9d1a0d3ac78637b75bba7d403864fef7d6883cf7144afec43d0eebf9a443b",'
        b'"files":[{"path":"LICENSE","sha256":"3333333333333333333333333333333333333333333333333333333333333333",'
        b'"size_bytes":0}],"license_files":["LICENSE"],"license_id":"Apache-2.0",'
        b'"model_id":"\xe7\xb5\x84\xe7\xb9\x94/\xe6\xa8\xa1\xe5\x9e\x8b","model_revision":"r\\"1",'
        b'"quantization":{"bits":null,"format":"safetensors","method":"none"},'
        b'"required_gpu_profiles":["gpu-\xce\xb1"],'
        b'"resources":{"disk_bytes":0,"ram_bytes":0,"vram_bytes":0},'
        b'"schema_version":"kairyu-model-manifest-v1","signer_key_id":"key-1",'
        b'"tokenizer":{"repository":"https://\xe4\xbe\x8b\xe3\x81\x88.invalid/tokenizer",'
        b'"revision":"1111111111111111111111111111111111111111",'
        b'"sha256":"2222222222222222222222222222222222222222222222222222222222222222"},'
        b'"upstream_repository":"https://\xe4\xbe\x8b\xe3\x81\x88.invalid/model",'
        b'"upstream_revision":"0000000000000000000000000000000000000000"}'
    )

    envelope = sign_model_artifact_manifest(manifest, _private_key())

    assert canonical_model_manifest_bytes(manifest) == expected_canonical
    assert envelope.manifest_digest == (
        "5fce380d56f194b38e5cf65ec84884d28ee4f2e380f9cfc1b05341125b0bf737"
    )
    assert envelope.signature_base64 == (
        "I3B3sccDWQGLDsAZCPTK2CzF4qzh1CaUK1eAkvuB2KrXbla/OR8n5PxF+u3l0amrAWUimOeAstm6GDzKE6raAg=="
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("manifest_digest", "0" * 64, "digest does not match"),
        (
            "signature_base64",
            base64.b64encode(bytes(64)).decode("ascii"),
            "signature is invalid",
        ),
    ],
)
def test_verification_rejects_digest_or_signature_tampering(field, value, error):
    envelope, trust_store = _signed()
    tampered = envelope.model_copy(update={field: value})

    with pytest.raises(InvalidModelArtifactError, match=error):
        verify_model_artifact_manifest(tampered, trust_store)


def test_verification_rejects_unknown_signer():
    envelope, _ = _signed()
    unrelated_key = Ed25519PrivateKey.generate()
    trust_store = ModelArtifactTrustStore(
        signers=(
            TrustedModelSigner(
                key_id="other-key",
                public_key_base64=_public_key_base64(unrelated_key),
                approved_environments=("production",),
            ),
        )
    )

    with pytest.raises(InvalidModelArtifactError, match="not trusted"):
        verify_model_artifact_manifest(envelope, trust_store)


def test_admission_binds_exact_gitops_identity_environment_and_gpu_profile():
    envelope, trust_store = _signed()

    admitted = admit_model_artifact(
        envelope,
        trust_store,
        _admission_request(envelope),
    )

    assert admitted.manifest_digest == envelope.manifest_digest
    assert admitted.model_id == "org/model"
    assert admitted.environment == "production"
    assert admitted.gpu_profile == "h100-sxm"


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"manifest_digest": "0" * 64}, "digest does not match"),
        ({"model_id": "other/model"}, "model ID does not match"),
        ({"model_revision": "other"}, "model revision does not match"),
        ({"environment": "development"}, "environment is not approved"),
        ({"gpu_profile": "a100-sxm"}, "GPU profile is not approved"),
    ],
)
def test_admission_rejects_gitops_binding_mismatch(updates, error):
    envelope, trust_store = _signed()

    with pytest.raises(ModelArtifactAdmissionError, match=error):
        admit_model_artifact(
            envelope,
            trust_store,
            _admission_request(envelope, **updates),
        )


def test_admission_requires_signer_authority_for_environment():
    private_key = _private_key()
    envelope = sign_model_artifact_manifest(_manifest(), private_key)
    staging_only = _trust_store(private_key, environments=("staging",))

    with pytest.raises(ModelArtifactAdmissionError, match="approved for signer"):
        admit_model_artifact(
            envelope,
            staging_only,
            _admission_request(envelope),
        )


def test_manifest_rejects_noncanonical_or_inconsistent_file_tree():
    valid = _manifest()

    with pytest.raises(ValidationError, match="sorted by unique path"):
        _manifest(files=tuple(reversed(valid.files)))
    with pytest.raises(ValidationError, match="does not match files"):
        _manifest(file_tree_sha256="0" * 64)
    with pytest.raises(ValidationError, match="cover all declared blobs"):
        _manifest(
            resources=ModelArtifactResourceEstimate(
                disk_bytes=1,
                ram_bytes=4096,
                vram_bytes=8192,
            )
        )
    with pytest.raises(ValidationError, match="safe relative POSIX path"):
        ModelArtifactBlob(path="../model", size_bytes=1, sha256="a" * 64)
    with pytest.raises(ValidationError, match="component byte limit"):
        ModelArtifactBlob(path="é" * 128, size_bytes=1, sha256="a" * 64)
    with pytest.raises(ValidationError, match="path byte limit"):
        ModelArtifactBlob(
            path="/".join("a" * 255 for _ in range(17)),
            size_bytes=1,
            sha256="a" * 64,
        )


def test_manifest_requires_immutable_revision_and_canonical_sets():
    with pytest.raises(ValidationError, match="lowercase commit digest"):
        _manifest(upstream_revision="main")
    with pytest.raises(ValidationError, match="lowercase commit digest"):
        _manifest(upstream_revision="1" * 41)
    with pytest.raises(ValidationError, match="sorted and unique"):
        _manifest(approved_environments=("staging", "production"))
    with pytest.raises(ValidationError, match="sorted and unique"):
        _manifest(required_gpu_profiles=("h100-sxm", "h100-sxm"))


def test_loaders_reject_duplicate_json_keys(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version":"x","schema_version":"y"}')

    with pytest.raises(InvalidModelArtifactError, match="keys must be unique"):
        load_signed_model_artifact(manifest_path)


def test_loaders_reject_nonstandard_json_constants(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version":NaN}')

    with pytest.raises(InvalidModelArtifactError, match="strict JSON grammar"):
        load_signed_model_artifact(manifest_path)


def test_admission_request_loader_rejects_oversize_before_parsing(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_bytes(b"{" + b" " * (64 * 1024))

    with pytest.raises(InvalidModelArtifactError, match="exceeds the size limit"):
        load_model_artifact_admission_request(request_path)


def test_loaders_reject_invalid_trust_store_without_echoing_values(tmp_path):
    trust_path = tmp_path / "trust.json"
    trust_path.write_text('{"signers":[{"key_id":"secret-value"}]}')

    with pytest.raises(InvalidModelArtifactError) as exc_info:
        load_model_artifact_trust_store(trust_path)

    assert "secret-value" not in str(exc_info.value)
    assert "schema validation failed" in str(exc_info.value)


def _write_cli_inputs(tmp_path):
    envelope, trust_store = _signed()
    manifest_path = tmp_path / "manifest.json"
    trust_path = tmp_path / "trust.json"
    request_path = tmp_path / "deployment-intent.json"
    manifest_path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    trust_path.write_text(trust_store.model_dump_json(indent=2), encoding="utf-8")
    request_path.write_text(
        _admission_request(envelope).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return envelope, manifest_path, trust_path, request_path


def test_artifact_validate_cli_verifies_without_runtime_side_effects(tmp_path, capsys):
    envelope, manifest_path, trust_path, _ = _write_cli_inputs(tmp_path)

    cli.main(
        [
            "artifact",
            "validate",
            str(manifest_path),
            "--trust-store",
            str(trust_path),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out == (f"VALID manifest={envelope.manifest_digest} signer=release-key-2026\n")
    assert captured.err == ""


def test_artifact_admit_cli_emits_bound_identity(tmp_path, capsys):
    envelope, manifest_path, trust_path, request_path = _write_cli_inputs(tmp_path)

    cli.main(
        [
            "artifact",
            "admit",
            str(manifest_path),
            "--trust-store",
            str(trust_path),
            "--request",
            str(request_path),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.startswith(f"ADMITTED manifest={envelope.manifest_digest} ")
    assert "deployment=production/model-service" in captured.out
    assert "model=org/model" in captured.out
    assert "environment=production" in captured.out
    assert captured.err == ""


def test_artifact_cli_fails_closed_without_traceback(tmp_path, capsys):
    envelope, manifest_path, trust_path, request_path = _write_cli_inputs(tmp_path)
    request_path.write_text(
        _admission_request(envelope, gpu_profile="unapproved-gpu").model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "artifact",
                "admit",
                str(manifest_path),
                "--trust-store",
                str(trust_path),
                "--request",
                str(request_path),
            ]
        )

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert captured.out == "INVALID artifact: GPU profile is not approved by manifest\n"
    assert captured.err == ""

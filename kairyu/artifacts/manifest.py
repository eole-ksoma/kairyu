"""Immutable, signed model-artifact manifest and admission contracts.

WP4.1 deliberately validates metadata only.  Blob retrieval, per-blob content
verification, cache population, and eviction remain later Phase 4 concerns.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_MAX_SIGNED_BIGINT = 2**63 - 1
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_TRUST_STORE_BYTES = 1024 * 1024
_MAX_ADMISSION_REQUEST_BYTES = 64 * 1024
_MAX_PATH_BYTES = 4096
_MAX_PATH_COMPONENT_BYTES = 255
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SIGNING_DOMAIN = b"kairyu:model-artifact-manifest:v1\n"


class InvalidModelArtifactError(ValueError):
    """The signed manifest or trust configuration is not valid."""


class ModelArtifactAdmissionError(InvalidModelArtifactError):
    """A valid signed artifact does not match the requested deployment."""


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _validate_sha256(value: str, *, name: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_revision(value: str, *, name: str) -> str:
    if not _REVISION_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a 40- or 64-character lowercase commit digest")
    return value


def _validate_relative_path(value: str, *, name: str) -> str:
    if not value or "\x00" in value or "\\" in value or value.startswith("/"):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    if len(value.encode("utf-8")) > _MAX_PATH_BYTES:
        raise ValueError(f"{name} exceeds the UTF-8 path byte limit")
    if any(len(part.encode("utf-8")) > _MAX_PATH_COMPONENT_BYTES for part in parts):
        raise ValueError(f"{name} exceeds the UTF-8 component byte limit")
    return value


def _validate_canonical_set(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    normalized = tuple(_non_empty(value, name=name) for value in values)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} must be sorted and unique")
    return normalized


class ModelArtifactBlob(BaseModel):
    """One immutable blob declared by a model manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    path: str
    size_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value, name="blob path")

    @field_validator("size_bytes", mode="before")
    @classmethod
    def validate_size_bytes(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("size_bytes must be an integer")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_sha256(value, name="blob sha256")


class ModelArtifactQuantization(BaseModel):
    """Portable quantization identity; values are producer-defined and signed."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    method: str = Field(max_length=128)
    format: str = Field(max_length=128)
    bits: int | None = Field(default=None, ge=1, le=64)

    @field_validator("method", "format")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("bits", mode="before")
    @classmethod
    def validate_bits(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("bits must be an integer")
        return value


class ModelArtifactTokenizer(BaseModel):
    """Immutable tokenizer source bound into the model manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    repository: str = Field(max_length=2048)
    revision: str = Field(max_length=64)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        return _non_empty(value, name="tokenizer repository")

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _validate_revision(value, name="tokenizer revision")

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_sha256(value, name="tokenizer sha256")


class ModelArtifactResourceEstimate(BaseModel):
    """Admission estimates signed by the manifest producer."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    disk_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    ram_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    vram_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)

    @field_validator("disk_bytes", "ram_bytes", "vram_bytes", mode="before")
    @classmethod
    def validate_bytes(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value


def model_file_tree_digest(blobs: Sequence[ModelArtifactBlob]) -> str:
    """Return the deterministic SHA-256 for an already canonical blob tree."""

    payload = [
        type(blob).model_validate(blob.model_dump()).model_dump(mode="json") for blob in blobs
    ]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class ModelArtifactManifest(BaseModel):
    """Canonical metadata for one immutable model artifact tree."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-model-manifest-v1"] = "kairyu-model-manifest-v1"
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    upstream_repository: str = Field(max_length=2048)
    upstream_revision: str = Field(max_length=64)
    architecture: str = Field(max_length=255)
    quantization: ModelArtifactQuantization
    tokenizer: ModelArtifactTokenizer
    license_id: str = Field(max_length=255)
    license_files: tuple[str, ...] = Field(min_length=1, max_length=64)
    required_gpu_profiles: tuple[str, ...] = Field(min_length=1, max_length=64)
    approved_environments: tuple[str, ...] = Field(min_length=1, max_length=64)
    signer_key_id: str = Field(max_length=255)
    resources: ModelArtifactResourceEstimate
    files: tuple[ModelArtifactBlob, ...] = Field(min_length=1, max_length=100_000)
    file_tree_sha256: str = Field(min_length=64, max_length=64)

    @field_validator(
        "model_id",
        "model_revision",
        "upstream_repository",
        "architecture",
        "license_id",
        "signer_key_id",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("upstream_revision")
    @classmethod
    def validate_upstream_revision(cls, value: str) -> str:
        return _validate_revision(value, name="upstream_revision")

    @field_validator("file_tree_sha256")
    @classmethod
    def validate_file_tree_digest(cls, value: str) -> str:
        return _validate_sha256(value, name="file_tree_sha256")

    @field_validator("license_files")
    @classmethod
    def validate_license_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        paths = tuple(_validate_relative_path(value, name="license file") for value in values)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("license_files must be sorted and unique")
        return paths

    @field_validator("required_gpu_profiles", "approved_environments")
    @classmethod
    def validate_canonical_sets(
        cls,
        values: tuple[str, ...],
        info,
    ) -> tuple[str, ...]:
        return _validate_canonical_set(values, name=info.field_name)

    @model_validator(mode="after")
    def validate_tree(self) -> ModelArtifactManifest:
        paths = tuple(blob.path for blob in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("files must be sorted by unique path")
        if not set(self.license_files).issubset(paths):
            raise ValueError("every license file must be declared in files")
        if sum(blob.size_bytes for blob in self.files) > self.resources.disk_bytes:
            raise ValueError("disk estimate must cover all declared blobs")
        if model_file_tree_digest(self.files) != self.file_tree_sha256:
            raise ValueError("file_tree_sha256 does not match files")
        return self


def canonical_model_manifest_bytes(manifest: ModelArtifactManifest) -> bytes:
    """Serialize a validated manifest with the WP4.1 canonical JSON profile."""

    validated = type(manifest).model_validate(manifest.model_dump())
    return json.dumps(
        validated.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def model_manifest_digest(manifest: ModelArtifactManifest) -> str:
    """Return the immutable identity used by GitOps and Runner requests."""

    return hashlib.sha256(canonical_model_manifest_bytes(manifest)).hexdigest()


class SignedModelArtifactManifest(BaseModel):
    """Manifest plus its declared digest and Ed25519 signature."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-model-signature-v1"] = "kairyu-model-signature-v1"
    algorithm: Literal["ed25519"] = "ed25519"
    manifest: ModelArtifactManifest
    manifest_digest: str = Field(min_length=64, max_length=64)
    signature_base64: str = Field(min_length=88, max_length=88)

    @field_validator("manifest_digest")
    @classmethod
    def validate_manifest_digest(cls, value: str) -> str:
        return _validate_sha256(value, name="manifest_digest")

    @field_validator("signature_base64")
    @classmethod
    def validate_signature(cls, value: str) -> str:
        _decode_base64(value, expected_size=64, name="signature")
        return value


class TrustedModelSigner(BaseModel):
    """One Ed25519 signer and the environments it may approve."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    key_id: str = Field(max_length=255)
    algorithm: Literal["ed25519"] = "ed25519"
    public_key_base64: str = Field(min_length=44, max_length=44)
    approved_environments: tuple[str, ...] = Field(min_length=1, max_length=64)

    @field_validator("key_id")
    @classmethod
    def validate_key_id(cls, value: str) -> str:
        return _non_empty(value, name="key_id")

    @field_validator("public_key_base64")
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        _decode_base64(value, expected_size=32, name="public key")
        return value

    @field_validator("approved_environments")
    @classmethod
    def validate_environments(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_canonical_set(values, name="approved_environments")


class ModelArtifactTrustStore(BaseModel):
    """Versioned local trust roots used by validation and admission."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-model-trust-store-v1"] = "kairyu-model-trust-store-v1"
    signers: tuple[TrustedModelSigner, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_unique_signers(self) -> ModelArtifactTrustStore:
        key_ids = tuple(signer.key_id for signer in self.signers)
        if len(set(key_ids)) != len(key_ids):
            raise ValueError("trust store signer key IDs must be unique")
        return self

    def signer_for(self, key_id: str) -> TrustedModelSigner:
        validated = type(self).model_validate(self.model_dump())
        for signer in validated.signers:
            if signer.key_id == key_id:
                return signer
        raise InvalidModelArtifactError("manifest signer is not trusted")


class VerifiedModelArtifact(BaseModel):
    """Evidence produced only after digest, trust, and signature verification."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    manifest: ModelArtifactManifest
    manifest_digest: str = Field(min_length=64, max_length=64)
    signer_key_id: str = Field(max_length=255)
    signer_approved_environments: tuple[str, ...] = Field(min_length=1, max_length=64)


class ModelArtifactAdmissionRequest(BaseModel):
    """Immutable GitOps intent that must match a verified manifest exactly."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-model-admission-v1"] = "kairyu-model-admission-v1"
    deployment_id: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    environment: str = Field(max_length=255)
    gpu_profile: str = Field(max_length=255)

    @field_validator("manifest_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_sha256(value, name="manifest_digest")

    @field_validator(
        "deployment_id",
        "model_id",
        "model_revision",
        "environment",
        "gpu_profile",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)


class ModelArtifactAdmission(BaseModel):
    """Narrow capability passed to later cache and Runner control planes."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-model-admission-result-v1"] = "kairyu-model-admission-result-v1"
    deployment_id: str
    manifest_digest: str
    model_id: str
    model_revision: str
    environment: str
    gpu_profile: str
    signer_key_id: str


def _decode_base64(value: str, *, expected_size: int, name: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name} must use canonical base64") from exc
    if len(decoded) != expected_size or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{name} must use canonical base64")
    return decoded


def sign_model_artifact_manifest(
    manifest: ModelArtifactManifest,
    private_key: Ed25519PrivateKey,
) -> SignedModelArtifactManifest:
    """Create a signed envelope for an already-authorized build producer."""

    canonical = canonical_model_manifest_bytes(manifest)
    signature = private_key.sign(_SIGNING_DOMAIN + canonical)
    return SignedModelArtifactManifest(
        manifest=manifest,
        manifest_digest=hashlib.sha256(canonical).hexdigest(),
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )


def verify_model_artifact_manifest(
    envelope: SignedModelArtifactManifest,
    trust_store: ModelArtifactTrustStore,
) -> VerifiedModelArtifact:
    """Fail closed unless digest, signer trust, and Ed25519 signature all match."""

    validated_envelope = type(envelope).model_validate(envelope.model_dump())
    validated_trust = type(trust_store).model_validate(trust_store.model_dump())
    canonical = canonical_model_manifest_bytes(validated_envelope.manifest)
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != validated_envelope.manifest_digest:
        raise InvalidModelArtifactError("manifest digest does not match content")
    signer = validated_trust.signer_for(validated_envelope.manifest.signer_key_id)
    public_key_bytes = _decode_base64(
        signer.public_key_base64,
        expected_size=32,
        name="public key",
    )
    signature = _decode_base64(
        validated_envelope.signature_base64,
        expected_size=64,
        name="signature",
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            _SIGNING_DOMAIN + canonical,
        )
    except (InvalidSignature, ValueError) as exc:
        raise InvalidModelArtifactError("manifest signature is invalid") from exc
    return VerifiedModelArtifact(
        manifest=validated_envelope.manifest,
        manifest_digest=digest,
        signer_key_id=signer.key_id,
        signer_approved_environments=signer.approved_environments,
    )


def admit_model_artifact(
    envelope: SignedModelArtifactManifest,
    trust_store: ModelArtifactTrustStore,
    request: ModelArtifactAdmissionRequest,
) -> ModelArtifactAdmission:
    """Authorize one exact GitOps model/environment/GPU binding."""

    verified = verify_model_artifact_manifest(envelope, trust_store)
    validated_request = type(request).model_validate(request.model_dump())
    manifest = verified.manifest
    if validated_request.manifest_digest != verified.manifest_digest:
        raise ModelArtifactAdmissionError("GitOps manifest digest does not match")
    if validated_request.model_id != manifest.model_id:
        raise ModelArtifactAdmissionError("GitOps model ID does not match")
    if validated_request.model_revision != manifest.model_revision:
        raise ModelArtifactAdmissionError("GitOps model revision does not match")
    if validated_request.environment not in manifest.approved_environments:
        raise ModelArtifactAdmissionError("environment is not approved by manifest")
    if validated_request.environment not in verified.signer_approved_environments:
        raise ModelArtifactAdmissionError("environment is not approved for signer")
    if validated_request.gpu_profile not in manifest.required_gpu_profiles:
        raise ModelArtifactAdmissionError("GPU profile is not approved by manifest")
    return ModelArtifactAdmission(
        deployment_id=validated_request.deployment_id,
        manifest_digest=verified.manifest_digest,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        environment=validated_request.environment,
        gpu_profile=validated_request.gpu_profile,
        signer_key_id=verified.signer_key_id,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidModelArtifactError("JSON object keys must be unique")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise InvalidModelArtifactError("JSON constants must use the strict JSON grammar")


def _load_json(path: Path, *, max_bytes: int, kind: str) -> Mapping[str, Any]:
    try:
        with path.open("rb") as source:
            raw = source.read(max_bytes + 1)
    except (OSError, ValueError) as exc:
        raise InvalidModelArtifactError(f"cannot read {kind}") from exc
    if len(raw) > max_bytes:
        raise InvalidModelArtifactError(f"{kind} exceeds the size limit")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
        raise InvalidModelArtifactError(f"{kind} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise InvalidModelArtifactError(f"{kind} root must be a JSON object")
    return value


def _schema_error(error: ValidationError, *, kind: str) -> InvalidModelArtifactError:
    detail = error.errors(include_input=False, include_url=False)[0]
    location = ".".join(str(part) for part in detail.get("loc", ())) or "<root>"
    location = "".join(
        character if character.isprintable() else character.encode("unicode_escape").decode("ascii")
        for character in location
    )
    if len(location) > 512:
        location = location[:509] + "..."
    error_type = str(detail.get("type", "invalid"))
    return InvalidModelArtifactError(
        f"{kind} schema validation failed at {location} ({error_type})"
    )


def load_signed_model_artifact(path: Path) -> SignedModelArtifactManifest:
    """Load a bounded, duplicate-key-safe signed manifest JSON file."""

    value = _load_json(path, max_bytes=_MAX_MANIFEST_BYTES, kind="manifest")
    try:
        return SignedModelArtifactManifest.model_validate(value)
    except ValidationError as exc:
        raise _schema_error(exc, kind="manifest") from exc


def load_model_artifact_trust_store(path: Path) -> ModelArtifactTrustStore:
    """Load a bounded, duplicate-key-safe public trust store JSON file."""

    value = _load_json(path, max_bytes=_MAX_TRUST_STORE_BYTES, kind="trust store")
    try:
        return ModelArtifactTrustStore.model_validate(value)
    except ValidationError as exc:
        raise _schema_error(exc, kind="trust store") from exc


def load_model_artifact_admission_request(path: Path) -> ModelArtifactAdmissionRequest:
    """Load a bounded GitOps model deployment-intent JSON file."""

    value = _load_json(
        path,
        max_bytes=_MAX_ADMISSION_REQUEST_BYTES,
        kind="admission request",
    )
    try:
        return ModelArtifactAdmissionRequest.model_validate(value)
    except ValidationError as exc:
        raise _schema_error(exc, kind="admission request") from exc

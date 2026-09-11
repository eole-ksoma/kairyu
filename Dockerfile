# One image for every node role; the mounted DeploymentSpec decides
# gateway vs replica (design m7 D1/D2).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder
WORKDIR /app
# Bound installation/bytecode workers so high-core builders with a conventional
# 1024-descriptor container limit remain reproducible.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_CONCURRENT_INSTALLS=8
ARG KAIRYU_EMBEDDINGS=0
ARG KAIRYU_VISION=0
COPY pyproject.toml uv.lock README.md ./
RUN set -eu; \
    case "$KAIRYU_EMBEDDINGS:$KAIRYU_VISION" in \
      0:0) uv sync --frozen --no-install-project --no-dev --extra fleet --extra otel ;; \
      1:0) uv sync --frozen --no-install-project --no-dev --extra fleet --extra otel --extra embeddings ;; \
      0:1) uv sync --frozen --no-install-project --no-dev --extra fleet --extra otel --extra vision ;; \
      1:1) uv sync --frozen --no-install-project --no-dev --extra fleet --extra otel --extra embeddings --extra vision ;; \
      *) echo "KAIRYU_EMBEDDINGS and KAIRYU_VISION must each be 0 or 1" >&2; exit 2 ;; \
    esac
COPY kairyu ./kairyu
COPY scripts/prefetch_embedding_model.py ./scripts/prefetch_embedding_model.py
COPY scripts/async_request_retention.py ./scripts/async_request_retention.py
RUN set -eu; \
    case "$KAIRYU_EMBEDDINGS:$KAIRYU_VISION" in \
      0:0) uv sync --frozen --no-dev --extra fleet --extra otel ;; \
      1:0) uv sync --frozen --no-dev --extra fleet --extra otel --extra embeddings ;; \
      0:1) uv sync --frozen --no-dev --extra fleet --extra otel --extra vision ;; \
      1:1) uv sync --frozen --no-dev --extra fleet --extra otel --extra embeddings --extra vision ;; \
      *) echo "KAIRYU_EMBEDDINGS and KAIRYU_VISION must each be 0 or 1" >&2; exit 2 ;; \
    esac
ARG KAIRYU_EMBEDDING_MODEL_REPOSITORY=
ARG KAIRYU_EMBEDDING_MODEL_REVISION=
ARG KAIRYU_EMBEDDING_MODEL_SHA256=
ARG KAIRYU_EMBEDDING_PROVENANCE_SHA256=
RUN set -eu; \
    mkdir -p /opt/kairyu/models; \
    if [ "$KAIRYU_EMBEDDINGS" = 1 ]; then \
      test -n "$KAIRYU_EMBEDDING_MODEL_REPOSITORY"; \
      test -n "$KAIRYU_EMBEDDING_MODEL_REVISION"; \
      test -n "$KAIRYU_EMBEDDING_MODEL_SHA256"; \
      test -n "$KAIRYU_EMBEDDING_PROVENANCE_SHA256"; \
      .venv/bin/python scripts/prefetch_embedding_model.py \
        --repository "$KAIRYU_EMBEDDING_MODEL_REPOSITORY" \
        --revision "$KAIRYU_EMBEDDING_MODEL_REVISION" \
        --destination /opt/kairyu/models/all-MiniLM-L6-v2 \
        --model-sha256 "$KAIRYU_EMBEDDING_MODEL_SHA256" \
        --provenance-sha256 "$KAIRYU_EMBEDDING_PROVENANCE_SHA256" \
        --license apache-2.0; \
    else \
      test -z "$KAIRYU_EMBEDDING_MODEL_REPOSITORY"; \
      test -z "$KAIRYU_EMBEDDING_MODEL_REVISION"; \
      test -z "$KAIRYU_EMBEDDING_MODEL_SHA256"; \
      test -z "$KAIRYU_EMBEDDING_PROVENANCE_SHA256"; \
    fi

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
WORKDIR /app
COPY --from=builder /app /app
COPY --from=builder /opt/kairyu/models /opt/kairyu/models
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
ENTRYPOINT ["kairyu", "serve"]
CMD ["/etc/kairyu/config.yaml"]

# FlashInfer SM120 (Blackwell) AOT packaging — Step B draft

Status: **DRAFT** — finalize and wire into `Dockerfile.cuda` only AFTER the
`FlashInferBackend` adapter passes `pytest -m gpu` and a forced-flashinfer
`/v1/chat/completions` returns 200 on a real SM120 box (see the enablement
plan). Do NOT replace the working `Dockerfile.cuda` with this until then —
AOT-baking kernels for an adapter that still 502s buys nothing.

Date: 2026-07-13 (relocated from a session scratchpad into the repo so the
draft is not stranded; see enablement plan §13.3).
Task: FlashInfer (SM120) enablement — Step B (AOT multi-stage build).

## Why AOT, and why CUDA 13.0

- The runtime image is a slim `nvidia/cuda:*-runtime` base with **no nvcc**, and
  `flashinfer-python 0.6.14` is `py3-none-any` (kernels are JIT-compiled at
  runtime). Without a compiler and without a precompiled kernel cache, the first
  request would try to `nvcc`-compile and fail (`NO_NVCC`).
- **A (this draft): build `flashinfer-jit-cache` from source in a `-devel` stage,
  scoped to SM120 only (`FLASHINFER_CUDA_ARCH_LIST=12.0f`), then COPY the wheel
  into the runtime venv.** This supplies the BF16 paged-attention kernels
  ahead-of-time and keeps the runtime image nvcc-free.
- **B (`flashinfer-cubin`) is NOT a substitute**: those cubins are trtllm-gen
  FP4/FMHA, not BF16 paged-attention, and sm_120/sm_121 cubins were reported
  missing (flashinfer #3294).
- **Toolkit is pinned to CUDA 13.0, not 12.8.** torch 2.12.1 is a cu13 build and
  carries its own `libcudart.so.13`; kernels compiled against a 12.8 toolkit link
  `libcudart.so.12` and would double the CUDA runtime in one process. CUDA 13.0
  is officially supported for `12.0f` JIT; avoid CUDA 13.3 (flashinfer #3493
  header/compiler mismatch). Scope the arch list to `12.0f` only — a full-arch
  build can trip unrelated FP4-kernel failures.

## Draft Dockerfile.cuda (multi-stage AOT)

```dockerfile
# ---- Stage 1: compile FlashInfer AOT kernels for SM120 (Blackwell / RTX PRO 6000) ----
# CUDA 13.0 devel: nvcc matching torch's cu130 runtime, official-supported for
# sm120f, avoids the CUDA 13.3 JIT breakage (flashinfer #3493).
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS flashinfer-build
ARG FLASHINFER_REF=v0.6.14                 # keep in sync with uv.lock flashinfer-python
ARG FLASHINFER_CUDA_ARCH_LIST=12.0f        # SM120-only (per decision); FP4 archs skipped
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/fi-build
ENV PATH="/opt/fi-build/bin:$PATH"
# torch must match the runtime venv (2.12.1, cu130) so AOT kernels link the same CUDA.
RUN pip install --no-cache-dir "torch==2.12.1" "setuptools>=77" build ninja
WORKDIR /src
RUN git clone --recursive --branch ${FLASHINFER_REF} https://github.com/flashinfer-ai/flashinfer.git
WORKDIR /src/flashinfer
RUN pip install --no-cache-dir -v .        # flashinfer-python (may be trimmable; per official docs)
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
RUN cd flashinfer-jit-cache && python -m build --no-isolation --wheel && ls -la dist/

# ---- Stage 2: runtime (base from PR #32) ----
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
# UV_COMPILE_BYTECODE=0: uv's parallel compile exhausts FDs (EMFILE) with the full GPU dep set.
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_PYTHON=python3.12
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra gpu --extra engine --extra fleet
COPY kairyu ./kairyu
RUN uv sync --frozen --no-dev --extra gpu --extra engine --extra fleet
# Install the AOT FlashInfer kernel cache into the venv. NOT via `uv sync --frozen`
# (rejects lockfile drift); --no-deps keeps the frozen env intact (flashinfer-python present).
COPY --from=flashinfer-build /src/flashinfer/flashinfer-jit-cache/dist/*.whl /tmp/fi/
RUN uv pip install --python /app/.venv/bin/python --no-deps /tmp/fi/*.whl && rm -rf /tmp/fi
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
ENTRYPOINT ["kairyu", "serve"]
CMD ["/etc/kairyu/config.yaml"]
```

## Open items to confirm on the first Step-B build (needs the SM120 box)

1. Exact `-devel` tag (13.0.1 vs 13.0.0) actually pullable on the host.
2. Is a full `pip install -v .` (flashinfer-python source build) required before the
   `flashinfer-jit-cache` build, or can jit-cache build standalone? Trim to cut build time.
3. **Runtime base version.** This draft keeps `12.8.1-runtime` (torch carries its
   own cu13 libs, and Step A ran torch+13.0-JIT'd kernels fine). If a double-CUDA
   surfaces at runtime, bump the runtime base to `13.0.1-runtime` to unify on
   cu13 (see enablement plan §12.2).
4. Wheel filename/tag (`flashinfer_jit_cache-<ver>+cu130-*`) — the glob COPY handles it.
5. Confirm the baked cache actually eliminates first-request JIT latency (the whole
   point of AOT): a cold replica's first `/v1/chat/completions` must not stall.

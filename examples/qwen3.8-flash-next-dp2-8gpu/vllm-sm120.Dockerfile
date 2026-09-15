# syntax=docker/dockerfile:1.7
# SM120 (RTX PRO 6000 Blackwell) vLLM image shared by the
# deepseek-v4-flash-vision-exp-dp2-8gpu and qwen3.8-flash-next-dp2-8gpu
# examples; the two copies of this file are byte-identical (unit-tested).
#
# Base: upstream vLLM's own CI build of one pinned `main` commit (the nightly
# tag is the source build of exactly that commit, pinned by digest). It
# carries DeepSeek-V4-Flash-Vision-Exp (#54566), Qwen3.8-Flash-Next (#53896)
# and the SM120 DeepSeek-V4 paths (#43477/#53574), which no release tag has.
#
# Overlay: FlashInfer 0.6.18 (the base's pin) lacks the SM120 sparse-MLA
# prefill path the DeepSeek vision checkpoint hits on its first image request;
# the fix is FlashInfer main (#4802). FlashInfer's runtime dependency list is
# unchanged between 0.6.18 and the pinned commit, so the Python package is
# swapped with --no-deps. The 0.6.18 AOT module cache (flashinfer-jit-cache)
# is removed so stale prebuilt kernels cannot shadow the JIT-compiled fix;
# flashinfer-cubin stays (data-only cubins, version-independent). Kernels
# JIT-compile on first use into the per-replica /root/.cache bind mount.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

ARG FLASHINFER_REPOSITORY=https://github.com/flashinfer-ai/flashinfer.git
ARG FLASHINFER_REVISION
RUN test -n "${FLASHINFER_REVISION}" \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git init -q /tmp/flashinfer \
    && git -C /tmp/flashinfer fetch -q --depth 1 "${FLASHINFER_REPOSITORY}" "${FLASHINFER_REVISION}" \
    && git -C /tmp/flashinfer checkout -q FETCH_HEAD \
    && git -C /tmp/flashinfer submodule update -q --init --recursive --depth 1 \
        3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl \
    && uv pip uninstall --system flashinfer-jit-cache \
    && BUILD_NVEP=0 uv pip install --system --no-deps /tmp/flashinfer \
    && rm -rf /tmp/flashinfer \
    && python3 -c "import flashinfer, importlib.metadata as m; print('flashinfer', m.version('flashinfer-python'))" \
    && ! python3 -c "import flashinfer_jit_cache" 2>/dev/null

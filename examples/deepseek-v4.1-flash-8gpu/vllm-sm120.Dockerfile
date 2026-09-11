# syntax=docker/dockerfile:1.7
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

# FlashInfer 0.6.18 in the official image predates the SM120 sparse-MLA
# prefill support required for image inputs. Build a fixed source revision;
# remove the old AOT cache so it cannot shadow the new JIT specializations.
ARG FLASHINFER_REVISION
RUN test -n "${FLASHINFER_REVISION}" \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git init -q /tmp/flashinfer \
    && git -C /tmp/flashinfer fetch -q --depth 1 https://github.com/flashinfer-ai/flashinfer.git "${FLASHINFER_REVISION}" \
    && git -C /tmp/flashinfer checkout -q FETCH_HEAD \
    && git -C /tmp/flashinfer submodule update -q --init --recursive --depth 1 \
        3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl \
    && uv pip uninstall --system flashinfer-jit-cache \
    && BUILD_NVEP=0 uv pip install --system --no-deps /tmp/flashinfer \
    && rm -rf /tmp/flashinfer

# Use the Python frontend whose encoder is checked below. The recipe's
# initial aliases predate the model author's published low/high/max mapping.
ENV VLLM_USE_RUST_FRONTEND=0
COPY patch_runtime.py /opt/kairyu/patch_runtime.py
RUN python3 /opt/kairyu/patch_runtime.py

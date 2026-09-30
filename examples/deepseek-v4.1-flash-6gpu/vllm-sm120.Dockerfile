# syntax=docker/dockerfile:1.7
# SM120 overlay for DeepSeek-V4.1-Flash on six RTX PRO 6000 GPUs.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

# Optional FlashInfer source build: the 0909 base ships FlashInfer 0.6.18,
# which predates the SM120 sparse-MLA prefill used by image inputs. An empty
# revision keeps the base image's FlashInfer. The old AOT cache is removed so
# it cannot shadow JIT specializations of the patched sources.
ARG FLASHINFER_REVISION=""
RUN if [ -n "${FLASHINFER_REVISION}" ]; then \
      apt-get update -y \
      && apt-get install -y --no-install-recommends git \
      && rm -rf /var/lib/apt/lists/* \
      && git init -q /tmp/flashinfer \
      && git -C /tmp/flashinfer fetch -q --depth 1 https://github.com/flashinfer-ai/flashinfer.git "${FLASHINFER_REVISION}" \
      && git -C /tmp/flashinfer checkout -q FETCH_HEAD \
      && git -C /tmp/flashinfer submodule update -q --init --recursive --depth 1 \
          3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl \
      && BUILD_NVEP=0 uv pip install --system --no-deps /tmp/flashinfer \
      && rm -rf /tmp/flashinfer; \
    fi
# Prebuilt JIT caches would shadow the patched FlashInfer sources below with
# unpatched kernels of the same version; remove every architecture's cache.
RUN packages="$(uv pip list --system 2>/dev/null | awk '/^flashinfer-jit-cache/ {print $1}')" \
    && if [ -n "$packages" ]; then uv pip uninstall --system $packages; fi

# The Python frontend renders chats with the encoder checked by patch_sm120.py.
ENV VLLM_USE_RUST_FRONTEND=0
COPY patch_sm120.py /opt/kairyu/patch_sm120.py
RUN python3 /opt/kairyu/patch_sm120.py
# Generated kernels of the patched FlashInfer sources live in their own
# workspace so an unpatched kernel of the same version can never be reused.
ENV FLASHINFER_WORKSPACE_BASE=/root/.cache/flashinfer-kairyu-sm120-v1

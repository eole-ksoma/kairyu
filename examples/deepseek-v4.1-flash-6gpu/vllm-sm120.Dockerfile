# syntax=docker/dockerfile:1.7
# SM120 overlay for DeepSeek-V4.1-Flash on six RTX PRO 6000 GPUs.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

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

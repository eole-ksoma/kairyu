# syntax=docker/dockerfile:1.7
# This example's overlay on the published OpenJev image: every chat completion
# thinks first with a fixed 512-token budget (think_core.py). control.py builds
# it with the pins from example.json and attests the result by image ID.
ARG OPENJEV_BASE_IMAGE
FROM ${OPENJEV_BASE_IMAGE}
ARG OPENJEV_REVISION
ARG OPENJEV_VERSION
ARG VLLM_REVISION

# Fail closed unless the base carries the vLLM this overlay was written for:
# vllm#57250's commit, OpenJev's 512 logprob-id cap and its vision prefix-LM
# patch. The sources are located without importing vLLM (no GPU at build time).
RUN test "$(git -C /opt/vllm rev-parse HEAD)" = "${VLLM_REVISION}" \
    && vllm_dir="$(python -c 'import importlib.util as u; print(next(iter(u.find_spec("vllm").submodule_search_locations)))')" \
    && grep -q '^MAX_LOGPROB_TOKEN_IDS = 512$' "${vllm_dir}/sampling_params.py" \
    && grep -q 'compute_mm_prefix_ranges' "${vllm_dir}/model_executor/models/diffusion_gemma.py"

# The pinned OpenJev source, whatever release the base image was built from:
# patch_openjev.py's anchors are exact for this revision.
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps \
        --reinstall-package openjev \
        "openjev @ git+https://github.com/razorback16/openjev@${OPENJEV_REVISION}"

COPY think_core.py think_first.py patch_openjev.py /opt/kairyu/
RUN python /opt/kairyu/patch_openjev.py --version "${OPENJEV_VERSION}" \
    && python -c "import openjev.api, openjev.think_first"

# Baked in, so the image ID attests the template vLLM renders with.
COPY chat_template.jinja /etc/kairyu/diffusiongemma-think.jinja
ENV KAIRYU_THINK_CHAT_TEMPLATE=/etc/kairyu/diffusiongemma-think.jinja \
    OPENJEV_VLLM_ARGS="--chat-template /etc/kairyu/diffusiongemma-think.jinja"

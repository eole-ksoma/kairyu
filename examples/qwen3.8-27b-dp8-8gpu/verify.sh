#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
venv_bin="$here/../../.venv/bin"
nvme_root="${NVME_STORAGE_ROOT:-/mnt/nvme/kairyu}"
bench_tmp="$nvme_root/model-volumes/qwen3.8-27b-dp8-8gpu/bench-tmp"
mkdir -p "$bench_tmp"
PATH="$venv_bin:$PATH"
TMPDIR="$bench_tmp"
export PATH
export TMPDIR
exec "$venv_bin/python" "$here/verification.py" "$@"

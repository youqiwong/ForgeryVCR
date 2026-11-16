#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${1:-${ADAPTER_PATH:-}}"
if [[ -z "${ADAPTER_PATH}" ]]; then
    echo "Usage: bash experts/train/merge_lora.sh <adapter-checkpoint>" >&2
    exit 1
fi

swift export \
    --adapters "${ADAPTER_PATH}" \
    --merge_lora true

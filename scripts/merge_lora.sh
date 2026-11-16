#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${1:-${ADAPTER_PATH:-}}"
if [[ -z "${ADAPTER_PATH}" ]]; then
    echo "Usage: bash scripts/merge_lora.sh <adapter-checkpoint>" >&2
    echo "   or: ADAPTER_PATH=<adapter-checkpoint> bash scripts/merge_lora.sh" >&2
    exit 1
fi
if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "[ERROR] adapter checkpoint does not exist: ${ADAPTER_PATH}" >&2
    exit 1
fi

swift export \
    --adapters "${ADAPTER_PATH}" \
    --merge_lora true

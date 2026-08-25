#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INFERENCE_MODEL_PATH="${INFERENCE_MODEL_PATH:-weights/ForgeryVCR/GRPO}"
INFERENCE_EXP_NAME="${INFERENCE_EXP_NAME:-forgeryvcr_grpo}"
INFERENCE_OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-inference_outputs/${INFERENCE_EXP_NAME}}"
INFERENCE_DATASETS="${INFERENCE_DATASETS:-all}"
INFERENCE_SEED="${INFERENCE_SEED:-42}"
INFERENCE_USE_SAM2="${INFERENCE_USE_SAM2:-1}"

if [[ ! -d "${INFERENCE_MODEL_PATH}" ]]; then
    echo "[ERROR] inference model does not exist: ${INFERENCE_MODEL_PATH}" >&2
    echo "Set INFERENCE_MODEL_PATH to the downloaded or merged model directory." >&2
    exit 1
fi

read -r -a DATASET_ARGS <<< "${INFERENCE_DATASETS}"
SAM2_ARGS=()
if [[ "${INFERENCE_USE_SAM2}" == "1" ]]; then
    SAM2_ARGS=(--use_sam2)
fi

echo "[Inference] model:   ${INFERENCE_MODEL_PATH}"
echo "[Inference] datasets:${INFERENCE_DATASETS}"
echo "[Inference] output:  ${INFERENCE_OUTPUT_DIR}"

python eval/inference.py \
    --model_path "${INFERENCE_MODEL_PATH}" \
    --output_dir "${INFERENCE_OUTPUT_DIR}" \
    --datasets "${DATASET_ARGS[@]}" \
    --seed "${INFERENCE_SEED}" \
    "${SAM2_ARGS[@]}"

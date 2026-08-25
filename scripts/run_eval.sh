#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INFERENCE_EXP_NAME="${INFERENCE_EXP_NAME:-forgeryvcr_grpo}"
INFERENCE_OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-inference_outputs/${INFERENCE_EXP_NAME}}"
MASK_DIR="${MASK_DIR:-${INFERENCE_OUTPUT_DIR}/pred_masks}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${INFERENCE_OUTPUT_DIR}/eval_reports}"

if [[ ! -d "${INFERENCE_OUTPUT_DIR}" ]]; then
    echo "[ERROR] inference output does not exist: ${INFERENCE_OUTPUT_DIR}" >&2
    echo "Set INFERENCE_OUTPUT_DIR to a completed inference run." >&2
    exit 1
fi

echo "[Evaluation] results: ${INFERENCE_OUTPUT_DIR}"
echo "[Evaluation] reports: ${EVAL_REPORT_DIR}"

if [[ -d "${MASK_DIR}" ]]; then
    echo "[Evaluation] masks:   ${MASK_DIR}"
    python eval/evaluate.py \
        --results_dir "${INFERENCE_OUTPUT_DIR}" \
        --mask_dir "${MASK_DIR}" \
        --output_dir "${EVAL_REPORT_DIR}"
else
    echo "[Evaluation] masks:   not found; Pixel-F1 and Pixel-IoU will be left blank"
    python eval/evaluate.py \
        --results_dir "${INFERENCE_OUTPUT_DIR}" \
        --output_dir "${EVAL_REPORT_DIR}"
fi

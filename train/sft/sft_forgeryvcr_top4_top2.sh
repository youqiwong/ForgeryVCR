#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_PATH="${MODEL_PATH:-weights/Qwen3-VL-4B-Instruct}"
SFT_DATASET_PATH="${SFT_DATASET_PATH:-data/sft/sft_dataset_balanced_top4_top2.json}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-train/sft/output_forgeryvcr_top4_top2_balanced}"
SFT_REPORT_TO="${SFT_REPORT_TO:-wandb}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-3}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to the downloaded Qwen3-VL checkpoint directory." >&2
    exit 1
fi
if [[ ! -f "${SFT_DATASET_PATH}" ]]; then
    echo "[ERROR] balanced dataset does not exist: ${SFT_DATASET_PATH}" >&2
    echo "Run data/sft/generate_SFT_datasets.py and balance_sft_dataset.py first." >&2
    exit 1
fi

mkdir -p "${SFT_OUTPUT_DIR}"

export RCCL_DEBUG="${RCCL_DEBUG:-INFO}"
export ROCM_VISIBLE_DEVICES="${ROCM_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-1}"
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK="${MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK:-1}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

echo "[SFT] model:   ${MODEL_PATH}"
echo "[SFT] dataset: ${SFT_DATASET_PATH}"
echo "[SFT] output:  ${SFT_OUTPUT_DIR}"
echo "[SFT] epochs:  ${SFT_NUM_TRAIN_EPOCHS}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
swift sft \
    --model "${MODEL_PATH}" \
    --dataset "${SFT_DATASET_PATH}" \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --gradient_accumulation_steps 8 \
    --num_train_epoch "${SFT_NUM_TRAIN_EPOCHS}" \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --report_to "${SFT_REPORT_TO}" \
    --max_length 10240 \
    --output_dir "${SFT_OUTPUT_DIR}" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --attn_impl sdpa \
    --gradient_checkpointing false \
    --agent_template hermes \
    --loss_scale hermes

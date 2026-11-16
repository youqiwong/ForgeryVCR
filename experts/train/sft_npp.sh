#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL_PATH="${MODEL_PATH:-weights/Qwen3-VL-4B-Instruct}"
export RCCL_DEBUG=INFO
export ROCM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_VERBOSITY=error
export MIOPEN_LOG_LEVEL=1
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1

NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
swift sft \
    --model "${MODEL_PATH}" \
    --dataset workdir/results/qa_npp.json \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --eval_steps 10 \
    --save_steps 50 \
    --save_total_limit 50 \
    --logging_steps 50 \
    --report_to wandb \
    --max_length 8192 \
    --output_dir output_expert_npp \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-08 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --warmup_steps 0 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --attn_impl sdpa \
    --gradient_checkpointing false \
    --system 'You are a helpful assistant in natural image tampering detection.'

echo "训练命令已完成。"

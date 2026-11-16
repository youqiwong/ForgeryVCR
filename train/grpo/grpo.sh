#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# A merged SFT checkpoint is required. There is deliberately no machine-local
# fallback: callers must select the checkpoint they intend to train from.
GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-}"
GRPO_IMD20_DATASET="${GRPO_IMD20_DATASET:-data/rl/datasets/imd20_rl_data.json}"
GRPO_FR_DATASET="${GRPO_FR_DATASET:-data/rl/datasets/fr_rl_data.json}"
GRPO_PLUGIN_PATH="${GRPO_PLUGIN_PATH:-train/grpo/plugin/plugin.py}"
GRPO_OUTPUT_ROOT="${GRPO_OUTPUT_ROOT:-train/grpo/outputs}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-forgeryvcr_grpo_$(date +%Y%m%d_%H%M%S)}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${GRPO_OUTPUT_ROOT}/${GRPO_RUN_NAME}}"
GRPO_TOOL_MAP_ROOT="${GRPO_TOOL_MAP_ROOT:-datasets/train/tool_maps}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-256}"
GRPO_VLLM_MAX_NUM_SEQS="${GRPO_VLLM_MAX_NUM_SEQS:-256}"
GRPO_VLLM_MM_PROCESSOR_CACHE_GB="${GRPO_VLLM_MM_PROCESSOR_CACHE_GB:-4}"
GRPO_DATALOADER_NUM_WORKERS="${GRPO_DATALOADER_NUM_WORKERS:-4}"
GRPO_DATASET_NUM_PROC="${GRPO_DATASET_NUM_PROC:-4}"
GRPO_RESUME_FROM_CHECKPOINT="${GRPO_RESUME_FROM_CHECKPOINT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GRPO_GLOBAL_BATCH_SIZE=$((4 * NPROC_PER_NODE))

if (( GRPO_GENERATION_BATCH_SIZE % GRPO_GLOBAL_BATCH_SIZE != 0 )); then
    echo "[ERROR] GRPO_GENERATION_BATCH_SIZE must be divisible by global batch ${GRPO_GLOBAL_BATCH_SIZE}." >&2
    exit 1
fi
if (( GRPO_GENERATION_BATCH_SIZE % 8 != 0 )); then
    echo "[ERROR] GRPO_GENERATION_BATCH_SIZE must be divisible by num_generations 8." >&2
    exit 1
fi

for required_file in \
    "${GRPO_IMD20_DATASET}" \
    "${GRPO_FR_DATASET}" \
    "${GRPO_PLUGIN_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] required file does not exist: ${required_file}" >&2
        exit 1
    fi
done

if [[ -z "${GRPO_MODEL_PATH}" ]]; then
    echo "[ERROR] GRPO_MODEL_PATH is required." >&2
    echo "Example: GRPO_MODEL_PATH=train/sft/outputs/checkpoint-N-merged bash train/grpo/grpo.sh" >&2
    exit 1
fi
if [[ ! -d "${GRPO_MODEL_PATH}" ]]; then
    echo "[ERROR] merged SFT model does not exist: ${GRPO_MODEL_PATH}" >&2
    echo "Merge the SFT LoRA checkpoint before starting GRPO." >&2
    exit 1
fi
if [[ -f "${GRPO_MODEL_PATH}/adapter_config.json" ]]; then
    echo "[ERROR] GRPO_MODEL_PATH points to an unmerged LoRA adapter." >&2
    echo "Run: swift export --adapters <checkpoint> --merge_lora true" >&2
    exit 1
fi
for dataset_name in imd20_tp imd20_au fr_tp fr_au; do
    for canonical_dir in ELA FFT NPP; do
        if [[ ! -d "${GRPO_TOOL_MAP_ROOT}/${dataset_name}/${canonical_dir}" ]]; then
            echo "[ERROR] missing ${dataset_name}/${canonical_dir} tool maps under: ${GRPO_TOOL_MAP_ROOT}" >&2
            echo "Generate the RL tool maps before starting GRPO." >&2
            exit 1
        fi
    done
done

mkdir -p "${GRPO_OUTPUT_DIR}"

export RCCL_DEBUG="${RCCL_DEBUG:-INFO}"
export ROCM_VISIBLE_DEVICES="${ROCM_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCM_VISIBLE_DEVICES}}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-1}"
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK="${MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK:-1}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export WANDB_PROJECT="${WANDB_PROJECT:-ForgeryVCR}"
export GRPO_REWARD_LOG_DIR="${GRPO_REWARD_LOG_DIR:-${GRPO_OUTPUT_DIR}/reward_logs}"
export GRPO_TOOL_MAP_ROOT
export GRPO_PLOT_INTERVAL="${GRPO_PLOT_INTERVAL:-20}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RESUME_ARGS=()
if [[ -n "${GRPO_RESUME_FROM_CHECKPOINT}" ]]; then
    if [[ ! -d "${GRPO_RESUME_FROM_CHECKPOINT}" ]]; then
        echo "[ERROR] resume checkpoint does not exist: ${GRPO_RESUME_FROM_CHECKPOINT}" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume_from_checkpoint "${GRPO_RESUME_FROM_CHECKPOINT}")
fi

echo "=========================================="
echo "Starting ForgeryVCR GRPO"
echo "model:          ${GRPO_MODEL_PATH}"
echo "IMD20 dataset:  ${GRPO_IMD20_DATASET}"
echo "FR dataset:     ${GRPO_FR_DATASET}"
echo "plugin:         ${GRPO_PLUGIN_PATH}"
echo "tool maps:      ${GRPO_TOOL_MAP_ROOT}"
echo "output:         ${GRPO_OUTPUT_DIR}"
echo "W&B project:    ${WANDB_PROJECT}"
echo "W&B run:        ${GRPO_RUN_NAME}"
echo "generation bs:  ${GRPO_GENERATION_BATCH_SIZE}"
echo "vLLM max seqs:  ${GRPO_VLLM_MAX_NUM_SEQS}"
echo "vLLM MM cache:  ${GRPO_VLLM_MM_PROCESSOR_CACHE_GB} GiB"
if [[ -n "${GRPO_RESUME_FROM_CHECKPOINT}" ]]; then
    echo "resume:         ${GRPO_RESUME_FROM_CHECKPOINT}"
fi
echo "=========================================="

NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift rlhf \
    --rlhf_type grpo \
    --model "${GRPO_MODEL_PATH}" \
    --dataset "${GRPO_IMD20_DATASET}" "${GRPO_FR_DATASET}" \
    --external_plugins "${GRPO_PLUGIN_PATH}" \
    --reward_funcs forgery_detection_reward \
    --multi_turn_scheduler forgery_detection_scheduler \
    "${RESUME_ARGS[@]}" \
    --max_turns 8 \
    --use_vllm true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --vllm_mode colocate \
    --vllm_use_async_engine true \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 16384 \
    --vllm_max_num_seqs "${GRPO_VLLM_MAX_NUM_SEQS}" \
    --vllm_mm_processor_cache_gb "${GRPO_VLLM_MM_PROCESSOR_CACHE_GB}" \
    --attn_impl flash_attn \
    --per_device_train_batch_size 4 \
    --generation_batch_size "${GRPO_GENERATION_BATCH_SIZE}" \
    --gradient_accumulation_steps 8 \
    --deepspeed zero2 \
    --max_length 8192 \
    --max_new_tokens 2048 \
    --output_dir "${GRPO_OUTPUT_DIR}" \
    --num_train_epoch 2 \
    --dataloader_num_workers "${GRPO_DATALOADER_NUM_WORKERS}" \
    --dataset_num_proc "${GRPO_DATASET_NUM_PROC}" \
    --bf16 true \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 10 \
    --save_total_limit 100 \
    --report_to wandb \
    --run_name "${GRPO_RUN_NAME}" \
    --beta 0.1 \
    --num_generations 8 \
    --top_k 0 \
    --top_p 1.0 \
    --temperature 1 \
    --warmup_ratio 0.05 \
    --log_completions false \
    --log_entropy true \
    --loss_scale last_round \
    --importance_sampling_level sequence \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'

echo "=========================================="
echo "GRPO completed"
echo "output: ${GRPO_OUTPUT_DIR}"
echo "=========================================="

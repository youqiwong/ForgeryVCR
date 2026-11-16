# Training reference

The public launchers use the repository-pinned ms-swift submodule. Environment
variables may override paths and resource settings without editing scripts.

## Agent SFT

Launcher: `train/sft/sft_forgeryvcr_top4_top2.sh`

| Setting | Value |
|---|---|
| backbone | Qwen3-VL-4B-Instruct |
| train type | LoRA |
| LoRA rank / alpha | 32 / 64 |
| epochs | 6 in the public command |
| per-device batch / grad accumulation | 1 / 8 |
| learning rate | `5e-5` |
| max length | 10240 |
| agent template / loss scale | Hermes / Hermes |
| precision | bfloat16 |

```bash
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
SFT_DATASET_PATH=data/sft/sft_dataset_balanced_top4_top2.json \
SFT_NUM_TRAIN_EPOCHS=6 \
SFT_REPORT_TO=wandb \
bash train/sft/sft_forgeryvcr_top4_top2.sh
```

Merge a selected adapter before GRPO:

```bash
bash scripts/merge_lora.sh train/sft/outputs/checkpoint-N
```

## GRPO

Launcher: `train/grpo/grpo.sh`; reward and tool scheduler:
`train/grpo/plugin/plugin.py`.

| Setting | Value |
|---|---|
| train type | LoRA |
| LoRA rank / alpha | 128 / 256 |
| generations | 8 |
| epochs | 2 |
| per-device batch / generation batch / grad accumulation | 4 / 256 / 8 |
| learning rate | `5e-5` |
| max turns | 8 |
| rollout | colocated asynchronous vLLM |
| DeepSpeed | ZeRO-2 |

```bash
python data/rl/generate_RL_datasets.py \
  --data_root datasets/rl/832x \
  --output_dir data/rl/datasets \
  --fr_sample_limit 5000 \
  --seed 42

GRPO_MODEL_PATH=train/sft/outputs/checkpoint-N-merged \
GRPO_TOOL_MAP_ROOT=datasets/train/tool_maps \
WANDB_PROJECT=ForgeryVCR \
bash train/grpo/grpo.sh
```

The reward is:

```text
R = 1.0 * R_classification + 2.0 * R_localization + 0.5 * R_tool-utility
```

`GRPO_MODEL_PATH` is mandatory. The launcher intentionally has no private or
machine-local checkpoint fallback.

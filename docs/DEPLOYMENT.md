# Fresh-clone workflow

The Chinese [README](../README.md) is the canonical end-to-end guide. This
page is a compact command index; all commands run from the repository root.

## 1. Clone and environment

```bash
git clone --recursive https://github.com/youqiwong/ForgeryVCR.git
cd ForgeryVCR

conda create -n forgeryvcr python=3.12 -y
conda activate forgeryvcr
python -m pip install -r requirements.txt
python -m pip install -e ./ms-swift
```

Install platform-compatible PyTorch and FlashAttention builds first when
required by the target system.

## 2. Repository-local layout

```text
weights/Qwen3-VL-4B-Instruct/
weights/ForgeryVCR/GRPO/
weights/sam2-hiera-large/sam2_hiera_large.pt
datasets/train/832x/
datasets/train/tool_maps/{ELA,FFT,NPP}/
datasets/rl/832x/
datasets/test/832x/
```

Download commands and archive layout are documented in the README. Validate
the selected workflow before launching a long job:

```bash
python scripts/check_setup.py
```

## 3. Tools and data generation

Start the forensic services in one terminal:

```bash
INSTANCES=8 bash tools/start.sh
```

Generate the training maps:

```bash
python data/prepare/generate_tool_maps.py \
  --base_dir datasets/train/832x \
  --datasets casia2_tp casia2_au \
  --out_root datasets/train/tool_maps \
  --workers 64
```

Evaluation maps use `--per_dataset_subdir`; see
[DATA_PREPARATION.md](DATA_PREPARATION.md).

## 4. Training and evaluation

```bash
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
SFT_DATASET_PATH=data/sft/sft_dataset_balanced_top4_top2.json \
SFT_NUM_TRAIN_EPOCHS=6 \
bash train/sft/sft_forgeryvcr_top4_top2.sh

bash scripts/merge_lora.sh train/sft/outputs/checkpoint-N

GRPO_MODEL_PATH=train/sft/outputs/checkpoint-N-merged \
GRPO_TOOL_MAP_ROOT=datasets/train/tool_maps \
bash train/grpo/grpo.sh
```

For the released model:

```bash
INFERENCE_MODEL_PATH=weights/ForgeryVCR/GRPO \
INFERENCE_DATASETS="all" \
INFERENCE_EXP_NAME=forgeryvcr_grpo \
bash scripts/run_inference.sh

INFERENCE_EXP_NAME=forgeryvcr_grpo \
bash scripts/run_eval.sh
```

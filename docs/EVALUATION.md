# Evaluation

Evaluation has two stages: visual-agent inference with optional SAM2 mask
generation, followed by metric aggregation.

## Prerequisites

- `datasets/test/832x/eval_dataset.json`
- `datasets/test/832x/tool_maps_{ELA,FFT,NPP}.json`
- rendered maps under `datasets/test/832x/tool_maps/`
- `weights/sam2-hiera-large/sam2_hiera_large.pt` for pixel metrics
- a merged model directory

Validate them with:

```bash
python scripts/check_setup.py
```

## Inference

```bash
INFERENCE_MODEL_PATH=weights/ForgeryVCR/GRPO \
INFERENCE_DATASETS="all" \
INFERENCE_EXP_NAME=forgeryvcr_grpo \
INFERENCE_SEED=42 \
bash scripts/run_inference.sh
```

Set `INFERENCE_USE_SAM2=0` to skip mask generation. `INFERENCE_DATASETS` accepts
a space-separated subset such as `"casia1_tp casia1_au"`.

## Metrics

```bash
INFERENCE_EXP_NAME=forgeryvcr_grpo \
bash scripts/run_eval.sh
```

The evaluator writes `eval_metrics.tsv` with one row per benchmark and a Total
row. Reported columns are Class-F1, ACC, Pixel-F1, Pixel-IoU, and BBox-IoU.

# Data preparation

All published dataset JSON paths are relative to the JSON file's dataset
directory. Generated manifests and prompt files resolve those paths locally.

## Directory layout

```text
datasets/
├── train/
│   ├── 832x/casia2_tp/{forged,gt,casia2_tp.json,casia2_tp.txt}
│   ├── 832x/casia2_au/
│   └── tool_maps/{ELA,FFT,NPP}/
├── rl/832x/{imd20_tp,imd20_au,fr_tp,fr_au}/
└── test/832x/
```

The README contains the Hugging Face download and extraction commands.

## Tool maps

Start the local gateway and workers:

```bash
INSTANCES=8 bash tools/start.sh
```

CASIA2 training maps use a flat layout because SFT and GRPO look up maps by
image stem:

```bash
python data/prepare/generate_tool_maps.py \
  --base_dir datasets/train/832x \
  --datasets casia2_tp casia2_au \
  --out_root datasets/train/tool_maps \
  --workers 64
```

Evaluation maps use per-dataset subdirectories to avoid basename collisions:

```bash
python data/prepare/generate_tool_maps.py \
  --base_dir datasets/test/832x \
  --datasets \
    casia1_tp casia1_au \
    columbia_tp columbia_au \
    coverage_tp coverage_au \
    dso_tp dso_au \
    glide_tp glide_au \
    korus_tp korus_au \
    in_the_wild_tp \
    NIST16_tp NIST16_au \
    SID_Set_tp SID_Set_au \
  --out_root datasets/test/832x/tool_maps \
  --per_dataset_subdir \
  --workers 64
```

Build the evaluation manifest and tool-map indices after rendering:

```bash
python data/prepare/prepare_eval_dataset.py \
  --data_root datasets/test/832x \
  --datasets \
    casia1_tp casia1_au \
    columbia_tp columbia_au \
    coverage_tp coverage_au \
    dso_tp dso_au \
    glide_tp glide_au \
    korus_tp korus_au \
    in_the_wild_tp \
    NIST16_tp NIST16_au \
    SID_Set_tp SID_Set_au
```

## SFT and RL data

Follow the README's expert evaluation and gain-ranking commands, then build
Top-4/Top-2 trajectories:

```bash
python data/sft/generate_SFT_datasets.py \
  --real_csv workdir/results/casia2_au_analysis_threshold_zoomin_top2.csv \
  --fake_csv workdir/results/casia2_tp_analysis_threshold_zoomin_top4.csv \
  --fake_json datasets/train/832x/casia2_tp/casia2_tp.json \
  --ela_dir datasets/train/tool_maps/ELA \
  --fft_dir datasets/train/tool_maps/FFT \
  --npp_dir datasets/train/tool_maps/NPP \
  --zoom_save_dir workdir/zoomin_images/generated_fallback \
  --output_json data/sft/sft_dataset_top4_top2_threshold_unbalanced.json \
  --real_k 2 \
  --fake_k 4 \
  --cpu_cores 32 \
  --seed 42

python data/sft/balance_sft_dataset.py \
  --input_json data/sft/sft_dataset_top4_top2_threshold_unbalanced.json \
  --output_json data/sft/sft_dataset_balanced_top4_top2.json \
  --multi_tool_min 2 \
  --seed 42

python data/rl/generate_RL_datasets.py \
  --data_root datasets/rl/832x \
  --output_dir data/rl/datasets \
  --fr_sample_limit 5000 \
  --seed 42
```

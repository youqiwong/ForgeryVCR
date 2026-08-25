<div align="center">

# ForgeryVCR: Visual-Centric Reasoning via Efficient Forensic Tools in MLLMs for Image Forgery Detection and Localization

### 🔥 ACM Multimedia 2026

**Youqi Wang<sup>1,2&#42;†</sup> · Shen Chen<sup>2&#42;♠</sup> ·
Haowei Wang<sup>2</sup> · Rongxuan Peng<sup>1</sup> ·
Taiping Yao<sup>2</sup> · Shunquan Tan<sup>1✉</sup> ·
Changsheng Chen<sup>1</sup> · Bin Li<sup>1</sup> ·
Shouhong Ding<sup>2✉</sup>**

<sup>1</sup> Shenzhen University　　<sup>2</sup> Tencent Youtu Lab

<sup>&#42;</sup>Equal contribution.　<sup>♠</sup>Project Leader.　
<sup>✉</sup>Corresponding author.　<sup>†</sup>Work done during internship at Tencent Youtu Lab.

[![ArXiv](https://img.shields.io/badge/arXiv-2602.14098-b31b1b.svg)](https://arxiv.org/abs/2602.14098)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb.svg)](https://youqiwong.github.io/projects/ForgeryVCR/)
[![Model](https://img.shields.io/badge/Model-Hugging_Face-ffd21e.svg)](https://huggingface.co/youqiwong/ForgeryVCR)
[![License](https://img.shields.io/badge/License-Apache--2.0-d4a017.svg)](LICENSE)

**English** | [简体中文](README_CN.md)

[Model weights](https://huggingface.co/youqiwong/ForgeryVCR) ·
[Evaluation data preparation](#3-model-and-evaluation-data)

</div>

> **Release status.** This release contains the inference, forensic-map
> preparation, SAM2 mask generation, and evaluation pipeline. Training code
> will be released progressively in future updates.

ForgeryVCR turns a multimodal large language model into an active image
forensics agent. The agent can inspect an image with ELA, FFT, NPP, and Zoom-In
tools, classify it as authentic or tampered, and return tampered-region bounding
boxes. The predicted boxes can then prompt SAM2 for pixel-level masks.

## 1. Method overview

<p align="center">
  <img src="assets/readme/forgeryvcr_method.png" alt="ForgeryVCR method" width="900">
</p>

The inference release contains:

```text
ForgeryVCR/
├── eval/                 # agent inference, SAM2 generation, and metrics
├── data/prepare/         # test manifest and forensic-map preparation
├── tools/                # ELA, FFT, and NPP services
├── scripts/              # inference, evaluation, and setup wrappers
├── requirements.txt      # core inference dependencies
└── ms-swift/             # pinned inference framework submodule
```

## 2. Installation

After cloning this repository, initialize the pinned `ms-swift` submodule from
the repository root:

```bash
git submodule update --init --recursive
```

Run all commands below from the repository root. The public workflow uses
repository-relative paths and does not require a separate path configuration
file.

Create the Python 3.12 environment:

```bash
conda create -n forgeryvcr python=3.12 -y
conda activate forgeryvcr
```

Before installing the repository requirements, install builds of
[PyTorch](https://pytorch.org/get-started/locally/) and FlashAttention that are
compatible with your accelerator, operating system, and Python environment.
Verify both packages first, then install the remaining dependencies:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -m pip install -r requirements.txt
python -m pip install -e ./ms-swift
```

`requirements.txt` pins vLLM `0.12.0` and the remaining inference dependencies.

Install SAM2 for pixel-level mask generation:

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e ./third_party/sam2

mkdir -p weights/sam2-hiera-large
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
  -o weights/sam2-hiera-large/sam2_hiera_large.pt
```

## 3. Model and evaluation data

The released inference checkpoint is stored under `GRPO/`. Download it into the
path used by the scripts:

```bash
huggingface-cli download --resume-download youqiwong/ForgeryVCR \
  --repo-type model \
  --include "GRPO/*" \
  --local-dir weights/ForgeryVCR
```

ForgeryVCR does **not** redistribute third-party evaluation datasets or
preprocessed copies. Obtain the benchmarks from their respective maintainers and
follow their licenses, access conditions, and citation requirements:

| Benchmark | Data source / access page |
| --- | --- |
| CASIA v1 | [Original paper](https://doi.org/10.1109/ChinaSIP.2013.6625374) · [Existing image/GT archive (unofficial)](https://github.com/namtpham/casia1groundtruth) |
| Columbia | [Official dataset page](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) · [download request form](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html) |
| COVERAGE | [Author repository](https://github.com/wenbihan/coverage) · [OneDrive download](https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH) · [Baidu mirror](https://pan.baidu.com/s/11i_swrFveLc9uZr1eR006Q) (code `zduj`) |
| CocoGlide | [Official TruFor repository](https://github.com/grip-unina/TruFor) · [Official ZIP](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip) |
| DSO-1 | [RECOD official data page](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1) · [Original paper](https://doi.org/10.1109/TIFS.2013.2265677) |
| Korus | [Author download page](https://pkorus.pl/downloads/dataset-realistic-tampering) · [Realistic Tampering ZIP](https://drive.google.com/open?id=0B73Fq3C_nT4aOThud0NYWUR2MTQ) |
| In-the-Wild | [Official Huh et al. project page](https://minyoungg.github.io/selfconsistency/) · [Official ZIP (89.2 MB)](https://minyoungg.github.io/selfconsistency/in_wild/in_wild.tar.gz) |
| NIST16 | [OpenMFC](https://mfc.nist.gov/) (open the Data tab) · [NIST description](https://www.nist.gov/itl/iad/mltg/open-media-forensics-challenge) |
| SID-Set | [Official SIDA repository](https://github.com/hzlsaber/SIDA) · [Official test.zip](https://drive.google.com/file/d/1_ivsEV5e14efuv93tJgXjWOondYnEC2G/view?usp=sharing) · [HF train/val](https://huggingface.co/datasets/saberzl/SID_Set) |

Access notes: The original CASIA v1 server is no longer available; use the existing unofficial archive and cite the original paper. Columbia requires the official request form. The RECOD DSO-1 page is still reachable, but its historical archive link is unavailable, so contact the authors for a current copy. In-the-Wild refers to the Huh et al. ECCV 2018 dataset, whose official project page provides a downloadable archive. NIST16 requires registration and data-licence acceptance through the OpenMFC Data page. For SID-Set, use the official SIDA `test.zip`; the Hugging Face link primarily provides train/validation. Follow the licence and citation requirements of every dataset. After download, first arrange
each raw split under `datasets/raw/test` using the same names accepted by the
evaluation scripts. Tampered TXT files contain `image,mask` relative-path pairs;
authentic TXT files contain one relative image path per line:

```text
datasets/raw/test/
├── casia1_tp/
│   ├── forged/
│   ├── gt/
│   └── casia1_tp.txt              # forged/xxx.png,gt/xxx_gt.png
└── casia1_au/
    ├── images/
    └── casia1_au.txt              # images/xxx.png
```

Create the 832×832 portable evaluation copy, BBox JSON files, and the combined
manifest with:

```bash
python data/prepare/prepare_eval_dataset.py \
  --source_root datasets/raw/test \
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
    SID_Set_tp SID_Set_au \
  --img_size 832
```

Each benchmark follows the same portable layout. A tampered split contains
images, ground-truth masks, a text listing, and a bounding-box JSON; an
authentic split contains images and a text listing:

```text
datasets/test/832x/
├── casia1_tp/
│   ├── forged/
│   ├── gt/
│   ├── casia1_tp.txt              # forged/xxx.png,gt/xxx_gt.png
│   └── casia1_tp.json
└── casia1_au/
    ├── *.png
    └── casia1_au.txt              # one relative image path per line
```

`casia1_tp.json` is a JSON array. Paths are relative to the directory that
contains the JSON, and boxes use the 832×832 pixel coordinate system:

```json
[
  {
    "forged_path": "forged/example.png",
    "mask_path": "gt/example_gt.png",
    "bboxes": [[214, 176, 503, 598]]
  }
]
```

To rebuild one JSON from an already resized TXT and its GT masks, run the
command below. Tampered samples whose masks produce no valid bounding box are
excluded from the resulting JSON and reported in the summary.

```bash
python data/prepare/mask_to_bbox.py \
  --txt datasets/test/832x/casia1_tp/casia1_tp.txt \
  --base_dir datasets/test/832x/casia1_tp \
  --out datasets/test/832x/casia1_tp/casia1_tp.json \
  --workers 8
```

## 4. Prepare forensic maps

Start the forensic services in one terminal. Set `INSTANCES` to the number of
accelerator workers assigned to each pooled service. The launcher prints
`All services ready` only after the gateway and every configured worker pass
their health checks:

```bash
INSTANCES=1 bash tools/start.sh
```

Generate ELA, FFT, and NPP maps in another terminal:

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
  --workers 16
```

The NPP backend weight is bundled at
`tools/backends/npp/pretrained_models/noiseprint++/noiseprint++.th`; no
additional NPP checkpoint download is required.

Build the portable evaluation manifest and tool-map indices:

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

This creates the portable sample manifest `datasets/test/832x/eval_dataset.json`
and the three map indices `tool_maps_ELA.json`, `tool_maps_FFT.json`, and
`tool_maps_NPP.json`. A tampered and an authentic manifest entry look like:

```json
{
  "images": ["casia1_tp/forged/example.png"],
  "dataset": "casia1_tp",
  "is_tampered": true,
  "bbox_objects": {"bbox": [[214, 176, 503, 598]]}
}
```

```json
{
  "images": ["casia1_au/example.png"],
  "dataset": "casia1_au",
  "is_tampered": false,
  "bbox_objects": {"bbox": []}
}
```

Each tool index maps the same relative source path to its cached forensic map:

```json
{
  "image_path": "casia1_tp/forged/example.png",
  "tool_map_path": "tool_maps/casia1_tp/ELA/example_ELA.png"
}
```

The `images` order is the model input order; `dataset` selects the benchmark,
while `is_tampered` and `bbox_objects` are evaluation-only ground truth.

Validate the model files, manifest image paths, every indexed forensic map, and
the SAM2 installation:

```bash
python scripts/check_setup.py
```

When SAM2 is intentionally disabled, run the same check with
`INFERENCE_USE_SAM2=0`; only the SAM2 package and checkpoint checks are skipped.

## 5. Inference and evaluation

Run the full visual agent and generate SAM2 masks:

```bash
INFERENCE_MODEL_PATH=weights/ForgeryVCR/GRPO \
INFERENCE_DATASETS="all" \
INFERENCE_EXP_NAME=forgeryvcr_grpo \
INFERENCE_SEED=42 \
bash scripts/run_inference.sh
```

The script uses the accelerator devices exposed by the current execution
environment. To evaluate only a subset, set a space-separated list such as
`INFERENCE_DATASETS="casia1_tp casia1_au"`. Set `INFERENCE_USE_SAM2=0` to skip
pixel-mask generation. ELA, FFT, and NPP are loaded from the cached maps created
in Section 4, so the forensic services can be stopped after map generation.
Zoom-In crops are produced dynamically during agent inference.

Aggregate image-level classification, BBox-IoU, Pixel-IoU, and Pixel-F1:

```bash
INFERENCE_EXP_NAME=forgeryvcr_grpo \
bash scripts/run_eval.sh
```

If the prediction-mask directory does not exist, classification and BBox-IoU
are still evaluated, while Pixel-F1 and Pixel-IoU are left blank. Missing GT
masks are reported explicitly and counted instead of being skipped silently.

Predictions, masks, and the final report are written under
`inference_outputs/forgeryvcr_grpo/`:

```text
inference_outputs/forgeryvcr_grpo/
├── <dataset>.json                 # full multi-turn dialogue per sample
├── pred_masks/<dataset>/*.png     # SAM2 masks for predicted fake samples
└── eval_reports/eval_metrics.tsv  # aggregated evaluation report
```

Each result JSON preserves `images`, `dataset`, `is_tampered`, and
`bbox_objects`, and adds the complete `messages` trajectory. Assistant messages
contain either `<tool_call>...</tool_call>` or the final
`<answer>real|fake, ...</answer>`; tool responses record the selected tool,
arguments, returned visual evidence, and turn index. Runtime result paths may be
resolved to local absolute paths, while all distributed dataset manifests and
tool-map indices remain repository-portable.

A shortened tampered-sample result has the following structure:

```json
{
  "images": ["<resolved-image-path>"],
  "dataset": "casia1_tp",
  "is_tampered": true,
  "bbox_objects": {"bbox": [[214, 176, 503, 598]]},
  "messages": [
    {"role": "user", "content": "Is this image real or fake? ..."},
    {"role": "assistant", "content": "<tool_call>...ELA...</tool_call>", "turn": 1},
    {"role": "tool_response", "tool_name": "ELA", "image_info": "<resolved-map-path>", "turn": 1},
    {"role": "assistant", "content": "<answer>fake, ...</answer>", "turn": 2}
  ]
}
```

## 6. Citation and license

```bibtex
@inproceedings{wang2026forgeryvcr,
  title     = {ForgeryVCR: Visual-Centric Reasoning via Efficient Forensic Tools
               in MLLMs for Image Forgery Detection and Localization},
  author    = {Wang, Youqi and Chen, Shen and Wang, Haowei and Peng, Rongxuan
               and Yao, Taiping and Tan, Shunquan and Chen, Changsheng
               and Li, Bin and Ding, Shouhong},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
```

ForgeryVCR builds on
[ms-swift](https://github.com/modelscope/ms-swift),
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), and
[SAM2](https://github.com/facebookresearch/sam2).

Original code in this repository is released under the
[Apache License 2.0](LICENSE). Datasets, model weights, and third-party
components remain subject to their respective licenses and terms. In
particular, the bundled NPP backend is governed by its
[upstream license](tools/backends/npp/LICENSE.txt), which restricts its use
to informational and nonprofit purposes.

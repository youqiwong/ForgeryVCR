from pathlib import Path as _Path

import json

# ================= 路径配置 =================

REPO_ROOT = _Path(__file__).resolve().parents[2]
WORKDIR = REPO_ROOT / "workdir"
TRAIN_DATA_ROOT = REPO_ROOT / "datasets" / "train"

# CSV 路径 (输入)
REAL_CSV_PATH = str(
    WORKDIR / "results" / "casia2_au_analysis_threshold_zoomin_top2.csv"
)
FAKE_CSV_PATH = str(
    WORKDIR / "results" / "casia2_tp_analysis_threshold_zoomin_top4.csv"
)

# GT Bbox JSON (输入)
FAKE_JSON_PATH = str(TRAIN_DATA_ROOT / "832x" / "casia2_tp" / "casia2_tp.json")

# 工具图根目录 (输入)：<root>/<dataset>/<tool>/<stem>_<tool>.png
TOOL_MAP_ROOT = str(TRAIN_DATA_ROOT / "tool_maps")
ELA_DIR = str(TRAIN_DATA_ROOT / "tool_maps" / "ELA")  # legacy flat layout
FFT_DIR = str(TRAIN_DATA_ROOT / "tool_maps" / "FFT")  # legacy flat layout
NPP_DIR = str(TRAIN_DATA_ROOT / "tool_maps" / "NPP")  # legacy flat layout

# Zoom-in 图片保存路径 (输出 - 自动创建)
ZOOM_SAVE_DIR = str(WORKDIR / "zoomin_images" / "generated_fallback")

# 最终数据集输出路径 (输出)
OUTPUT_JSON_PATH = str(
    REPO_ROOT / "data" / "sft" / "sft_dataset_top4_top2_threshold_unbalanced.json"
)

# ================= 参数配置 =================

# 图片尺寸 (用于 bbox 生成逻辑)
IMG_H = 832
IMG_W = 832

# 并行核心数
CPU_CORES = 32

# 允许的工具集合
ALLOWED_TOOLS = {'ELA', 'FFT', 'NPP', 'Zoom-In'}

# ================= Prompt 配置 =================

USER_PROMPTS = [
    "Classify this image as real or fake. If it is fake, provide the bounding boxes for the altered areas.",
    "Determine if this image is real or fake. If manipulation is found, highlight the tampered regions with bounding boxes.",
    "Is this image real or fake? Please output your judgment and, if forged, specify the bounding boxes.",
    "Assess the authenticity of this image (real or fake). If tampered, enclose the manipulated areas in bounding boxes.",
    "Please analyze whether this image is real or fake. If it's not authentic, show me the bounding boxes of the tampered zones.",
    "I need a verdict: is this picture real or fake? Highlight any manipulated parts using bounding boxes if it is fake.",
    "Check for manipulation. Start by stating if the image is real or fake, and then return the bounding boxes for any forged regions.",
    "Perform a forgery detection. Classify the image as real or fake, and if tampered, output the coordinates of the bounding boxes.",
    "Is there any part of this image that has been tampered with? Answer 'real' or 'fake' first, and if 'fake', provide the bounding box coordinates.",
    "What is the status of this image, real or fake? If you detect manipulation, please provide the bounding boxes."
]

# 工具定义 JSON 字符串
TOOLS_DEF_STR = json.dumps([
    {"type": "function", "function": {"name": "zoom_in", "description": "Zooms in on a suspicious region to check for fine-grained manipulation artifacts like inconsistent textures or noise.", "parameters": {"type": "object", "properties": {"bbox": {"type": "array", "items": {"type": "integer"}, "description": "The bounding box coordinates [x1, y1, x2, y2] of the region to zoom in."}}, "required": ["bbox"]}}},
    {"type": "function", "function": {"name": "ELA", "description": "Performs Error Level Analysis to reveal inconsistencies in the image's compression levels. Tampered regions often exhibit distinct error levels.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "FFT", "description": "Analyzes the image's frequency domain using Fast Fourier Transform. Tampering can introduce periodic artifacts or disrupt natural frequency patterns.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "NPP", "description": "Analyzes the image's noise fingerprints using Noise Print Pattern. Tampered regions often show inconsistent noise variance compared to the authentic background.", "parameters": {"type": "object", "properties": {}, "required": []}}}
])

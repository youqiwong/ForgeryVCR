import os as _os
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
EVAL_DATA_ROOT = _Path(
    _os.environ.get("EVAL_DATA_ROOT", REPO_ROOT / "datasets" / "test" / "832x")
).resolve()
# ==================== 基础配置 ====================
# 默认的随机种子
SEED = 42

# 图像尺寸 (用于 Agent 和 SAM2 坐标转换)
IMG_SIZE = 832

# ==================== 路径配置 ====================

# 汇总的测试数据集路径 (包含所有需要推理的样本)
CONSOLIDATED_DATASET_PATH = _os.environ.get(
    "CONSOLIDATED_DATASET_PATH",
    str(EVAL_DATA_ROOT / "eval_dataset.json"),
)

# SAM2 模型路径
SAM2_CHECKPOINT = _os.environ.get(
    "SAM2_CHECKPOINT",
    str(REPO_ROOT / "weights" / "sam2-hiera-large" / "sam2_hiera_large.pt"),
)
SAM2_CONFIG_NAME = "sam2_hiera_l.yaml"

# 工具图 (ELA, NPP, FFT) 的 JSON 索引文件路径
TOOL_MAP_JSON_PATHS = {
    "ELA": [str(EVAL_DATA_ROOT / "tool_maps_ELA.json")],
    "NPP": [str(EVAL_DATA_ROOT / "tool_maps_NPP.json")],
    "FFT": [str(EVAL_DATA_ROOT / "tool_maps_FFT.json")],
}

# 默认处理的数据集列表
DEFAULT_TARGET_DATASETS = [
    'casia1_tp', 'casia1_au', 
    'columbia_tp', 'columbia_au', 
    'coverage_tp', 'coverage_au', 
    'dso_au', 'dso_tp', 
    'glide_au', 'glide_tp', 
    'korus_tp', 'korus_au',
    'in_the_wild_tp',
    'NIST16_tp', 'NIST16_au',
    'SID_Set_tp', 'SID_Set_au',
]

# ==================== Prompt 与工具定义 ====================

# 用户随机 Prompt 列表
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
TOOLS_JSON_STRING = """[
    {\"type\": \"function\", \"function\": {\"name\": \"zoom_in\", \"description\": \"Zooms in on a suspicious region to check for fine-grained manipulation artifacts like inconsistent textures or noise.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"bbox\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"description\": \"The bounding box coordinates [x1, y1, x2, y2] of the region to zoom in.\"}}, \"required\": [\"bbox\"]}}}, 
    {\"type\": \"function\", \"function\": {\"name\": \"ELA\", \"description\": \"Performs Error Level Analysis to reveal inconsistencies in the image's compression levels. Tampered regions often exhibit distinct error levels.\", \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}, 
    {\"type\": \"function\", \"function\": {\"name\": \"FFT\", \"description\": \"Analyzes the image's frequency domain using Fast Fourier Transform. Tampering can introduce periodic artifacts or disrupt natural frequency patterns.\", \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}, 
    {\"type\": \"function\", \"function\": {\"name\": \"NPP\", \"description\": \"Analyzes the image's noise fingerprints using Noise Print Pattern. Tampered regions often show inconsistent noise variance compared to the authentic background.\", \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}
]"""

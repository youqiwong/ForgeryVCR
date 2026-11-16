import os
import json
from pathlib import Path

# ================= 基础路径配置 =================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Public release layout. Both locations can be overridden from the CLI or env.
BASE_PATH_PREFIX = os.environ.get(
    "RL_DATA_ROOT",
    str(REPO_ROOT / "datasets" / "rl" / "832x"),
)
OUTPUT_DIR = os.environ.get(
    "RL_OUTPUT_DIR",
    str(REPO_ROOT / "data" / "rl" / "datasets"),
)

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Prompt 配置 =================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": (
                "Zooms in on a suspicious region to check for fine-grained "
                "manipulation artifacts like inconsistent textures or noise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "The bounding box coordinates "
                            "[x1, y1, x2, y2] of the region to zoom in."
                        ),
                    }
                },
                "required": ["bbox"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ELA",
            "description": (
                "Performs Error Level Analysis to reveal inconsistencies in "
                "the image's compression levels. Tampered regions often "
                "exhibit distinct error levels."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FFT",
            "description": (
                "Analyzes the image's frequency domain using Fast Fourier "
                "Transform. Tampering can introduce periodic artifacts or "
                "disrupt natural frequency patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "NPP",
            "description": (
                "Analyzes the image's noise fingerprints using Noise Print "
                "Pattern. Tampered regions often show inconsistent noise "
                "variance compared to the authentic background."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# 固定的 System Prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant.\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> "
        "XML tags:\n<tools>\n"
        f"{json.dumps(TOOLS, ensure_ascii=False)}"
        "\n</tools>\n\nFor each function call, return a json object with "
        "function name and arguments within <tool_call></tool_call> XML tags:"
        "\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}'
        "\n</tool_call>"
    ),
}

# User Prompts 池
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

# ================= 数据集详细配置 =================

def build_dataset_configs(
    data_root=BASE_PATH_PREFIX,
    output_dir=OUTPUT_DIR,
    fr_sample_limit=5000,
):
    return {
        "IMD20": {
            "real_txt": os.path.join(data_root, "imd20_au/imd20_au.txt"),
            "fake_txt": os.path.join(data_root, "imd20_tp/imd20_tp.txt"),
            "fake_json": os.path.join(data_root, "imd20_tp/imd20_tp.json"),
            "sample_limit": None,
            "output_file": os.path.join(output_dir, "imd20_rl_data.json"),
        },
        "FR": {
            "real_txt": os.path.join(data_root, "fr_au/fr_au.txt"),
            "fake_txt": os.path.join(data_root, "fr_tp/fr_tp_filtered.txt"),
            "fake_json": os.path.join(data_root, "fr_tp/fr_tp.json"),
            "sample_limit": fr_sample_limit,
            "output_file": os.path.join(output_dir, "fr_rl_data.json"),
        },
    }


DATASET_CONFIGS = build_dataset_configs()

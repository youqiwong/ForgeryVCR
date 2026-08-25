import os as _os
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
EVAL_DATA_ROOT = _Path(
    _os.environ.get("EVAL_DATA_ROOT", REPO_ROOT / "datasets" / "test" / "832x")
).resolve()
# ==================== 基础配置 ====================
# 图像统一尺寸 (必须与推理时的设置一致，用于 BBox 归一化)
IMG_SIZE = 832  

# ==================== 路径配置 ====================

# 原始数据集根目录 (用于查找 GT Mask 路径)
# 结构假设: BASE_DATA_DIR/dataset_name/dataset_name.json
BASE_DATA_DIR = str(EVAL_DATA_ROOT)

# 特殊 GT 文件路径映射 (如果某些数据集的 JSON 不在标准位置)
SPECIAL_GT_PATHS = {}

# ==================== 数据集定义 ====================

# 需要评估的目标数据集
TARGET_DATASETS = [
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

# 只需要评估 Pixel 级指标的数据集 (通常只有篡改样本 _tp 才有 GT Mask)
# 这里会自动从 TARGET_DATASETS 中筛选包含 _tp 的数据集，或者你可以手动指定
PIXEL_EVAL_DATASETS = [ds for ds in TARGET_DATASETS if '_tp' in ds]

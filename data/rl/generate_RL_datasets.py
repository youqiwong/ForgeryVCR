import argparse
import json
import os
import random
from config_RL import (
    BASE_PATH_PREFIX,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
    USER_PROMPTS,
    build_dataset_configs,
)

# ================= 工具函数 =================

def load_lines(file_path):
    """读取文本文件行"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def get_absolute_path(txt_file_path, relative_path):
    """根据txt文件位置和相对路径构建绝对路径"""
    base_dir = os.path.dirname(txt_file_path)
    return os.path.abspath(os.path.join(base_dir, relative_path))


def portable_path(path):
    """Store repository-relative paths when possible."""
    relative = os.path.relpath(path, os.getcwd())
    return relative if relative != ".." and not relative.startswith(f"..{os.sep}") else path

def load_bbox_data(json_path):
    """加载 BBox JSON 并构建 {image_path: bboxes} 映射"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    json_dir = os.path.dirname(os.path.abspath(json_path))
    mapping = {}
    for item in data:
        image_path = item['forged_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(json_dir, image_path)
        # Normalize both portable relative paths and legacy absolute paths to
        # the same key used for TXT entries below.
        mapping[os.path.abspath(image_path)] = item['bboxes']
    return mapping

def generate_dataset(dataset_name, config, seed):
    print(f"正在处理数据集: {dataset_name} ...")
    
    # 随机数生成器初始化
    rnd_data_sampler = random.Random(seed) # 用于数据采样
    rnd_prompt = random.Random(seed)       # 用于选择 User Prompt
    
    final_data = []
    
    # 1. 处理真实数据 (Real)
    real_lines = load_lines(config['real_txt'])
    
    # 如果需要采样
    if config['sample_limit'] is not None and len(real_lines) > config['sample_limit']:
        real_lines = rnd_data_sampler.sample(real_lines, config['sample_limit'])
        
    real_samples = []
    for line in real_lines:
        # 真实数据 txt 只有文件名
        abs_path = get_absolute_path(config['real_txt'], line)
        
        user_content = rnd_prompt.choice(USER_PROMPTS)
        
        sample = {
            "messages": [
                SYSTEM_PROMPT,
                {
                    "role": "user",
                    "content": f"<image>{user_content}"
                }
            ],
            "images": [portable_path(abs_path)],
            "is_tampered": False
        }
        real_samples.append(sample)
    
    # 2. 处理篡改数据 (Fake)
    fake_lines = load_lines(config['fake_txt'])
    bbox_map = load_bbox_data(config['fake_json'])
    
    # 如果需要采样
    if config['sample_limit'] is not None and len(fake_lines) > config['sample_limit']:
        fake_lines = rnd_data_sampler.sample(fake_lines, config['sample_limit'])
        
    fake_samples = []
    missing_bbox_count = 0
    
    for line in fake_lines:
        # 篡改数据 txt 格式: forged/img.png,gt/mask.png
        # 我们只需要第一列
        rel_img_path = line.split(',')[0]
        abs_path = get_absolute_path(config['fake_txt'], rel_img_path)
        
        # 获取 BBox
        if abs_path in bbox_map:
            bboxes = bbox_map[abs_path]
            user_content = rnd_prompt.choice(USER_PROMPTS)
            
            sample = {
                "messages": [
                    SYSTEM_PROMPT,
                    {
                        "role": "user",
                        "content": f"<image>{user_content}"
                    }
                ],
                "images": [portable_path(abs_path)],
                "is_tampered": True,
                "bbox_objects": {
                    "bbox": bboxes
                }
            }
            fake_samples.append(sample)
        else:
            missing_bbox_count += 1
            # print(f"Warning: No bbox found for {abs_path}")

    # 合并数据
    final_data = real_samples + fake_samples
    
    # 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(config['output_file'])), exist_ok=True)
    with open(config['output_file'], 'w') as f:
        json.dump(final_data, f, indent=4)
        
    # 打印统计信息
    print(f"--- {dataset_name} 统计 ---")
    print(f"真实样本数: {len(real_samples)}")
    print(f"篡改样本数: {len(fake_samples)}")
    if missing_bbox_count > 0:
        print(f"警告: 有 {missing_bbox_count} 个篡改样本在 JSON 中未找到 BBox 信息，已跳过。")
    print(f"输出文件: {config['output_file']}")
    print("-" * 30 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IMD20 + FantasticReality GRPO prompt JSON files"
    )
    parser.add_argument("--data_root", default=BASE_PATH_PREFIX)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--fr_sample_limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ================= 主程序 =================

if __name__ == "__main__":
    args = parse_args()
    if args.fr_sample_limit < 1:
        raise ValueError("--fr_sample_limit must be >= 1")
    configs = build_dataset_configs(
        data_root=os.path.abspath(args.data_root),
        output_dir=os.path.abspath(args.output_dir),
        fr_sample_limit=args.fr_sample_limit,
    )
    for name, conf in configs.items():
        generate_dataset(name, conf, seed=args.seed)

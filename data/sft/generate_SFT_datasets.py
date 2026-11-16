import os
import json
import random
import multiprocessing
import argparse
import cv2
import numpy as np
import zlib
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 导入配置
from config_SFT import (
    REAL_CSV_PATH, FAKE_CSV_PATH, FAKE_JSON_PATH,
    TOOL_MAP_ROOT, ELA_DIR, FFT_DIR, NPP_DIR, ZOOM_SAVE_DIR, OUTPUT_JSON_PATH,
    IMG_H, IMG_W, CPU_CORES, ALLOWED_TOOLS,
    USER_PROMPTS, TOOLS_DEF_STR
)

# ================= 辅助函数 =================

def get_tool_path(original_path, tool_name):
    filename = os.path.basename(original_path)
    name_no_ext = os.path.splitext(filename)[0]
    dataset_name = next(
        (
            parent.name
            for parent in Path(original_path).parents
            if parent.name.lower().endswith(("_tp", "_au"))
        ),
        None,
    )
    suffix = {
        "ELA": "_ELA.png",
        "FFT": "_FFT.png",
        "NPP": "_NPP.png",
    }.get(tool_name)
    candidates = []
    if dataset_name and suffix:
        candidates.append(
            os.path.join(TOOL_MAP_ROOT, dataset_name, tool_name, f"{name_no_ext}{suffix}")
        )
    if tool_name == 'ELA':
        candidates.append(os.path.join(ELA_DIR, f"{name_no_ext}_ELA.png"))
    elif tool_name == 'FFT':
        candidates.append(os.path.join(FFT_DIR, f"{name_no_ext}_FFT.png"))
    elif tool_name == 'NPP':
        candidates.append(os.path.join(NPP_DIR, f"{name_no_ext}_NPP.png"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Prefer the public canonical name in diagnostics when no map exists.
    return candidates[0] if candidates else None

def normalize_tool_name(name):
    if name == 'Zoom-In':
        return 'zoom_in'
    return name

def parse_strategies(tools_str, max_tools):
    """Build paper-style trajectories from one gain-ranked tool list.

    Every sample retained by ISR gets exactly one NoTool trajectory. Tool
    trajectories consist of every ranked singleton plus cumulative prefixes
    of the original ranking. We never create arbitrary permutations.
    """
    if not isinstance(tools_str, str) or "hardsample" in tools_str:
        return []

    raw_list = [x.strip() for x in tools_str.replace('"', '').split(',') if x.strip()]
    if not raw_list:
        return []

    filtered = []
    seen_tools = set()
    for tool in raw_list:
        if tool not in ALLOWED_TOOLS:
            continue
        normalized = normalize_tool_name(tool)
        if normalized in seen_tools:
            continue
        filtered.append(normalized)
        seen_tools.add(normalized)
        if len(filtered) == max_tools:
            break

    # NoTool is a trajectory for every ISR-retained sample, independent of
    # whether it is written in Effective_Tools_List.
    strategies = [[]]
    if not filtered:
        return strategies

    for t in filtered:
        strategies.append([t])

    if len(filtered) > 1:
        for prefix_length in range(2, len(filtered) + 1):
            strategies.append(filtered[:prefix_length])

    return strategies

def get_deterministic_seed(path_str):
    """根据文件路径生成固定的随机种子"""
    return zlib.adler32(path_str.encode('utf-8')) & 0xffffffff


def canonical_path(path, base_dir=None):
    value = Path(path)
    if not value.is_absolute():
        value = Path(base_dir or os.getcwd()) / value
    return str(value.resolve())


def portable_path(path):
    relative = os.path.relpath(path, os.getcwd())
    return relative if relative != ".." and not relative.startswith(f"..{os.sep}") else path


def initialize_worker(toolmap_root, ela_dir, fft_dir, npp_dir, zoom_save_dir):
    """Initialize path configuration in both forked and spawned workers."""
    global TOOL_MAP_ROOT, ELA_DIR, FFT_DIR, NPP_DIR, ZOOM_SAVE_DIR
    TOOL_MAP_ROOT = toolmap_root
    ELA_DIR = ela_dir
    FFT_DIR = fft_dir
    NPP_DIR = npp_dir
    ZOOM_SAVE_DIR = zoom_save_dir

def parse_zoom_bbox(value):
    """Parse the optional Scheme-B Zoom-In bbox stored in the gain CSV."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [int(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and len(parsed) == 4:
                return [int(v) for v in parsed]
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
    return None

def generate_real_bbox(img_w, img_h, file_seed):
    # 使用基于文件名的固定种子
    np.random.seed(file_seed)
    
    center_x, center_y = img_w // 2, img_h // 2
    w = np.random.randint(int(img_w * 0.2), int(img_w * 0.5))
    h = np.random.randint(int(img_h * 0.2), int(img_h * 0.5))
    x1 = np.clip(center_x - w//2 + np.random.randint(-50, 50), 0, img_w - w)
    y1 = np.clip(center_y - h//2 + np.random.randint(-50, 50), 0, img_h - h)
    return [int(x1), int(y1), int(x1+w), int(y1+h)]

def generate_fake_bbox(gt_bboxes, img_w, img_h, file_seed):
    # 使用基于文件名的固定种子
    np.random.seed(file_seed)
    
    # 随机选择和随机扰动现在对于同一张图是固定的
    # Python random 模块的种子设置
    random.seed(file_seed)
    target_bbox = random.choice(gt_bboxes)
    
    x1, y1, x2, y2 = target_bbox
    w, h = x2 - x1, y2 - y1
    scale = np.random.uniform(1.1, 1.2)
    new_w, new_h = w * scale, h * scale
    center_x, center_y = x1 + w / 2, y1 + h / 2
    new_x1 = max(0, center_x - new_w / 2)
    new_y1 = max(0, center_y - new_h / 2)
    new_x2 = min(img_w, new_x1 + new_w)
    new_y2 = min(img_h, new_y1 + new_h)
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

# ================= Worker 处理逻辑 =================

def process_data_chunk(args):
    task_list, chunk_idx = args
    results = []
    
    local_stats = {
        'real': defaultdict(int), 
        'fake': defaultdict(int),
        'real_total_entries': 0,
        'fake_total_entries': 0
    }
    
    for item in task_list:
        original_path = item['img_path']
        label = item['label']
        gt_bboxes = item['bboxes']
        strategies = item['strategies']
        prescribed_zoom_bbox = item.get('zoom_bbox')
        prescribed_zoom_crop = item.get('zoom_crop_path')
        
        stat_key = 'fake' if label == 1 else 'real'
        
        # 截断策略：Real <= 2, Fake <= 4
        max_tools = 2 if label == 0 else 4
        valid_strategies = [s for s in strategies if len(s) <= max_tools]
        
        # 计算该图片的固定种子
        file_seed = get_deterministic_seed(original_path)

        for strategy in valid_strategies:
            if label == 0: local_stats['real_total_entries'] += 1
            else: local_stats['fake_total_entries'] += 1
            
            tool_len = len(strategy)
            local_stats[stat_key][f'count_{tool_len}_tool'] += 1
            if tool_len == 0: local_stats[stat_key]['count_NoTool'] += 1
            trajectory_name = "NoTool" if not strategy else " -> ".join(strategy)
            local_stats[stat_key][f'trajectory::{trajectory_name}'] += 1
            
            prompt_seed = get_deterministic_seed(
                f"{item['seed']}|{original_path}|{trajectory_name}|prompt"
            )
            prompt = random.Random(prompt_seed).choice(USER_PROMPTS)
            messages = [
                {"role": "user", "content": f"<image>{prompt}"},
                {"role": "assistant", "content": ""}
            ]
            
            image_paths = [original_path]
            
            for tool_idx, tool_name in enumerate(strategy):
                local_stats[stat_key][f'use_{tool_name}'] += 1
                
                tool_call_content = ""
                tool_res_content = '{"status": "success", "tool_image": "<image>"}'
                tool_img_path = ""
                
                if tool_name == 'zoom_in':
                    # 生成确定性的 bbox
                    if prescribed_zoom_bbox:
                        bbox = prescribed_zoom_bbox
                    elif label == 1 and gt_bboxes:
                        bbox = generate_fake_bbox(gt_bboxes, IMG_W, IMG_H, file_seed)
                    else:
                        bbox = generate_real_bbox(IMG_W, IMG_H, file_seed)
                    
                    tool_call_content = json.dumps({"name": "zoom_in", "arguments": {"bbox": bbox}})
                    
                    # 生成固定的文件名：原名_zoomin.png
                    base_name = os.path.splitext(os.path.basename(original_path))[0]
                    zoom_filename = f"{base_name}_zoomin.png"
                    zoom_save_path = os.path.join(ZOOM_SAVE_DIR, zoom_filename)
                    
                    # 检查文件是否存在，存在则跳过读取和写入（优化 IO）
                    if prescribed_zoom_crop and os.path.exists(prescribed_zoom_crop):
                        tool_img_path = prescribed_zoom_crop
                    elif os.path.exists(zoom_save_path):
                        tool_img_path = zoom_save_path
                    else:
                        try:
                            src_img = cv2.imread(original_path)
                            if src_img is not None:
                                z_x1, z_y1, z_x2, z_y2 = bbox
                                crop_img = src_img[z_y1:z_y2, z_x1:z_x2]
                                if crop_img.size > 0:
                                    cv2.imwrite(zoom_save_path, crop_img)
                                    tool_img_path = zoom_save_path
                                else:
                                    tool_img_path = original_path
                            else:
                                tool_img_path = original_path
                        except:
                            tool_img_path = original_path
                else:
                    tool_call_content = json.dumps({"name": tool_name, "arguments": {}})
                    tool_img_path = get_tool_path(original_path, tool_name)
                
                messages.append({"role": "tool_call", "content": tool_call_content})
                messages.append({"role": "tool_response", "content": tool_res_content})
                image_paths.append(tool_img_path)
                
                if tool_idx < len(strategy) - 1:
                     messages.append({"role": "assistant", "content": ""})

            # Answer
            if label == 1:
                bbox_tokens = "<bbox>" * len(gt_bboxes) if gt_bboxes else "<bbox>"
                final_content = f"<answer>fake, {bbox_tokens}</answer>"
                final_objects = {"bbox": gt_bboxes}
            else:
                final_content = "<answer>real</answer>"
                final_objects = {}
            
            messages.append({"role": "assistant", "content": final_content})
            
            entry = {
                "tools": TOOLS_DEF_STR,
                "messages": messages,
                "images": image_paths,
                "label": label
            }
            if final_objects:
                entry["objects"] = final_objects
            
            results.append(entry)
            
    return results, local_stats

# ================= 主程序 =================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-style SFT trajectories after ISR: one NoTool, "
            "all ranked single tools, and cumulative ranked prefixes."
        )
    )
    parser.add_argument("--real_csv", default=REAL_CSV_PATH)
    parser.add_argument("--fake_csv", default=FAKE_CSV_PATH)
    parser.add_argument("--fake_json", default=FAKE_JSON_PATH)
    parser.add_argument("--toolmap_root", default=TOOL_MAP_ROOT)
    parser.add_argument("--ela_dir", default=ELA_DIR)
    parser.add_argument("--fft_dir", default=FFT_DIR)
    parser.add_argument("--npp_dir", default=NPP_DIR)
    parser.add_argument("--zoom_save_dir", default=ZOOM_SAVE_DIR)
    parser.add_argument("--output_json", default=OUTPUT_JSON_PATH)
    parser.add_argument("--real_k", type=int, default=2)
    parser.add_argument("--fake_k", type=int, default=4)
    parser.add_argument("--cpu_cores", type=int, default=CPU_CORES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.real_k < 1 or args.fake_k < 1:
        raise ValueError("--real_k and --fake_k must be >= 1")
    if args.cpu_cores < 1:
        raise ValueError("--cpu_cores must be >= 1")

    # Keep worker helpers simple while allowing portable CLI paths.
    global TOOL_MAP_ROOT, ELA_DIR, FFT_DIR, NPP_DIR, ZOOM_SAVE_DIR
    TOOL_MAP_ROOT = args.toolmap_root
    ELA_DIR = args.ela_dir
    FFT_DIR = args.fft_dir
    NPP_DIR = args.npp_dir
    ZOOM_SAVE_DIR = args.zoom_save_dir

    # 确保输出目录存在
    os.makedirs(ZOOM_SAVE_DIR, exist_ok=True)
    
    print("Step 1: 加载篡改样本 bbox 映射...")
    fake_bbox_map = {}
    try:
        with open(args.fake_json, 'r') as f:
            fake_json = json.load(f)
            for item in fake_json:
                fake_path = canonical_path(
                    item['forged_path'],
                    os.path.dirname(os.path.abspath(args.fake_json)),
                )
                fake_bbox_map[fake_path] = item['bboxes']
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print("Step 2: 读取 CSV 并生成任务...")
    all_tasks = []
    seen_real_paths = set()
    seen_fake_paths = set()
    duplicate_real_rows = 0
    duplicate_fake_rows = 0
    
    # Real
    try:
        df_real = pd.read_csv(args.real_csv)
        for _, row in df_real.iterrows():
            path = portable_path(canonical_path(row['Image Path']))
            if path in seen_real_paths:
                duplicate_real_rows += 1
                continue
            strategies = parse_strategies(
                row['Effective_Tools_List'], max_tools=args.real_k
            )
            if not strategies:
                continue
            seen_real_paths.add(path)
            all_tasks.append({
                'img_path': path,
                'label': 0,
                'bboxes': [],
                'strategies': strategies,
                'zoom_bbox': parse_zoom_bbox(row.get('ZoomIn_BBox')),
                'zoom_crop_path': (
                    str(row.get('ZoomIn_Crop_Path'))
                    if pd.notna(row.get('ZoomIn_Crop_Path')) else None
                ),
                'seed': args.seed,
            })
    except Exception as e:
        print(f"Error Real CSV: {e}")

    # Fake
    try:
        df_fake = pd.read_csv(args.fake_csv)
        for _, row in df_fake.iterrows():
            path = portable_path(canonical_path(row['Image Path']))
            if path in seen_fake_paths:
                duplicate_fake_rows += 1
                continue
            canonical_fake_path = canonical_path(path)
            if canonical_fake_path not in fake_bbox_map:
                continue
            strategies = parse_strategies(
                row['Effective_Tools_List'], max_tools=args.fake_k
            )
            if not strategies:
                continue
            seen_fake_paths.add(path)
            all_tasks.append({
                'img_path': path,
                'label': 1,
                'bboxes': fake_bbox_map[canonical_fake_path],
                'strategies': strategies,
                'zoom_bbox': parse_zoom_bbox(row.get('ZoomIn_BBox')),
                'zoom_crop_path': (
                    str(row.get('ZoomIn_Crop_Path'))
                    if pd.notna(row.get('ZoomIn_Crop_Path')) else None
                ),
                'seed': args.seed,
            })
    except Exception as e:
         print(f"Error Fake CSV: {e}")

    print(f"  有效任务总数 (Images): {len(all_tasks)}")
    print(
        "  跳过重复 CSV 行: "
        f"real={duplicate_real_rows}, fake={duplicate_fake_rows}"
    )
    random.Random(args.seed).shuffle(all_tasks)

    # Step 3: 并行
    chunk_size = len(all_tasks) // args.cpu_cores + 1
    chunks = []
    for i in range(0, len(all_tasks), chunk_size):
        chunks.append((all_tasks[i:i+chunk_size], i))

    print(f"Step 4: 开始并行生成 (Cores: {args.cpu_cores})...")
    
    global_results = []
    global_stats = {
        'real': defaultdict(int), 'fake': defaultdict(int),
        'real_total_entries': 0, 'fake_total_entries': 0
    }
    
    with multiprocessing.Pool(
        processes=args.cpu_cores,
        initializer=initialize_worker,
        initargs=(TOOL_MAP_ROOT, ELA_DIR, FFT_DIR, NPP_DIR, ZOOM_SAVE_DIR),
    ) as pool:
        for res_list, stat in tqdm(pool.imap(process_data_chunk, chunks), total=len(chunks)):
            global_results.extend(res_list)
            global_stats['real_total_entries'] += stat['real_total_entries']
            global_stats['fake_total_entries'] += stat['fake_total_entries']
            for cat in ['real', 'fake']:
                for k, v in stat[cat].items():
                    global_stats[cat][k] += v

    print(f"Step 5: 保存结果...")
    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(global_results, f, indent=4)

    print("\n" + "="*60)
    print("FINAL STATISTICS REPORT")
    print("="*60)
    
    # Print Real Stats
    s = global_stats['real']
    print(f"Category: REAL SAMPLES (Total: {global_stats['real_total_entries']})")
    print(f"  NoTool: {s['count_NoTool']}")
    print(f"  1 Tool: {s['count_1_tool']}")
    print(f"  2 Tool: {s['count_2_tool']}")
    print(f"  Use Zoom-in: {s['use_zoom_in']}")
    print("  Exact trajectory distribution:")
    for key in sorted(k for k in s if k.startswith("trajectory::")):
        count = s[key]
        print(
            f"    {key.split('::', 1)[1]}: {count} "
            f"({count / global_stats['real_total_entries']:.2%})"
        )
    print("-" * 30)

    # Print Fake Stats
    s = global_stats['fake']
    print(f"Category: FAKE SAMPLES (Total: {global_stats['fake_total_entries']})")
    print(f"  NoTool: {s['count_NoTool']}")
    print(f"  1 Tool: {s['count_1_tool']}")
    print(f"  2 Tool: {s['count_2_tool']}")
    print(f"  3 Tool: {s['count_3_tool']}")
    print(f"  4 Tool: {s['count_4_tool']}")
    print(f"  Use Zoom-in: {s['use_zoom_in']}")
    print("  Exact trajectory distribution:")
    for key in sorted(k for k in s if k.startswith("trajectory::")):
        count = s[key]
        print(
            f"    {key.split('::', 1)[1]}: {count} "
            f"({count / global_stats['fake_total_entries']:.2%})"
        )
    print("="*60)
    print(f"[DONE] SFT dataset -> {args.output_json}")

if __name__ == "__main__":
    main()

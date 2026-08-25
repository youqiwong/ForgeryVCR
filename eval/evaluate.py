# eval_metrics.py
import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from PIL import Image

# 导入配置
import config_eval as cfg

# ==================== 辅助函数: 分类与 BBox 解析 ====================

def parse_model_response_strict(response_text):
    """
    解析模型回复（严格模式）。
    只要出现 fake/manipulated 相关词汇，即判为 Fake。
    """
    if not response_text:
        return False, []

    # 提取 <answer> 内容
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    content = answer_match.group(1).strip() if answer_match else response_text.strip()
    content_lower = content.lower()
    
    # 判定真假
    fake_keywords = ['fake', 'manipulated', 'tampered', 'forged', 'altered']
    is_pred_fake = any(kw in content_lower for kw in fake_keywords)
    
    if not is_pred_fake:
        real_keywords = ['real', 'authentic', 'genuine', 'original']
        is_pred_real = any(kw in content_lower for kw in real_keywords)
        if not is_pred_real:
            is_pred_fake = False # 默认倾向
    
    # 提取 BBox
    bboxes = []
    if is_pred_fake:
        # 匹配 (x1, y1), (x2, y2)
        pattern_list = re.compile(r'\((\d+)\s*,\s*(-?\d+)\)\s*,\s*\((\d+)\s*,\s*(-?\d+)\)')
        matches_list = pattern_list.findall(content)
        for m in matches_list:
            try:
                x1, y1, x2, y2 = map(int, m)
                bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            except: continue
                
    return is_pred_fake, bboxes

def normalize_bbox(bbox, img_size):
    """将 0-1000 的坐标归一化到 img_size"""
    x1, y1, x2, y2 = bbox
    scale = img_size / 1000.0
    return [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)]

# ==================== 辅助函数: 指标计算 ====================

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def calculate_bbox_miou(gt_boxes, pred_boxes):
    if not gt_boxes or not pred_boxes: return 0.0
    num_gt, num_pred = len(gt_boxes), len(pred_boxes)
    iou_matrix = np.zeros((num_gt, num_pred))
    for i in range(num_gt):
        for j in range(num_pred):
            iou_matrix[i, j] = calculate_iou(gt_boxes[i], pred_boxes[j])
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    total_iou_of_matches = iou_matrix[row_ind, col_ind].sum()
    denominator = max(num_gt, num_pred)
    return total_iou_of_matches / denominator if denominator > 0 else 0.0

def calculate_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def calculate_pixel_metrics(pred_mask_path, gt_mask_path, img_size):
    """计算单张图的 Pixel IoU 和 Pixel F1"""
    # 1. 加载 GT Mask
    if gt_mask_path and os.path.exists(gt_mask_path):
        try:
            gt_img = Image.open(gt_mask_path).convert('L')
            if gt_img.size != (img_size, img_size):
                gt_img = gt_img.resize((img_size, img_size), resample=Image.NEAREST)
            gt_mask = np.array(gt_img) > 128
        except:
            gt_mask = np.zeros((img_size, img_size), dtype=bool)
    else:
        gt_mask = np.zeros((img_size, img_size), dtype=bool)

    # 2. 加载 Pred Mask
    if pred_mask_path and os.path.exists(pred_mask_path):
        try:
            pred_img = Image.open(pred_mask_path).convert('L')
            # 预测掩码通常已经是 img_size，但为了保险 resize
            if pred_img.size != (img_size, img_size):
                pred_img = pred_img.resize((img_size, img_size), resample=Image.NEAREST)
            pred_mask = np.array(pred_img) > 128
        except:
            pred_mask = np.zeros((img_size, img_size), dtype=bool)
    else:
        # 如果没有对应的预测掩码文件（说明被判为 Real），则全黑
        pred_mask = np.zeros((img_size, img_size), dtype=bool)

    # 3. 计算指标
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # IoU
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0 # 都是全黑则为 1.0
    else:
        iou = intersection / union

    # F1
    tp = intersection
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return iou, f1

# ==================== 主逻辑 ====================

def get_base_dataset_name(dataset_name):
    return dataset_name.replace('_tp', '').replace('_au', '')

def main():
    parser = argparse.ArgumentParser(description="Unified Agent Evaluation (Class, BBox, Pixel)")
    parser.add_argument('--results_dir', type=str, required=True, help="Dir containing Agent JSON results")
    parser.add_argument('--mask_dir', type=str, default=None, help="Dir containing SAM2 Pred Masks (root/dataset/img.png)")
    parser.add_argument('--output_dir', type=str, required=True, help="Dir to save the TSV report")
    args = parser.parse_args()

    print(f"--- 综合评估脚本 (Class + BBox + Pixel) ---")
    print(f"JSON 结果: {args.results_dir}")
    print(f"Mask 结果: {args.mask_dir if args.mask_dir else '未提供 (跳过 Pixel 评估)'}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 建立 GT Mask 映射 (Image Path -> GT Mask Path)
    print("正在加载 Ground Truth 映射...")
    gt_lookup = {}
    for ds_name in cfg.TARGET_DATASETS:
        json_file = os.path.join(cfg.BASE_DATA_DIR, ds_name, f"{ds_name}.json")
        if ds_name in cfg.SPECIAL_GT_PATHS:
            json_file = cfg.SPECIAL_GT_PATHS[ds_name]
            
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    json_dir = os.path.dirname(os.path.abspath(json_file))
                    def resolve(value):
                        if not value or os.path.isabs(value):
                            return value
                        return os.path.abspath(os.path.join(json_dir, value))
                    for item in data:
                        # 尝试多种键名获取图片路径
                        k = item.get('forged_path') or item.get('image_path') or (item.get('images')[0] if 'images' in item and len(item['images'])>0 else None)
                        if k:
                            mask_p = item.get('mask_path') or item.get('mask')
                            gt_lookup[resolve(k)] = resolve(mask_p)
                except Exception: pass

    # 2. 读取所有 Agent 结果
    all_samples = []
    if os.path.exists(args.results_dir):
        json_files = [f for f in os.listdir(args.results_dir) if f.endswith('.json')]
        for file_name in json_files:
            with open(os.path.join(args.results_dir, file_name), 'r', encoding='utf-8') as f:
                all_samples.extend(json.load(f))
    else:
        print(f"Error: {args.results_dir} not found.")
        return

    # 3. 初始化统计容器
    # Class stats
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'gt_tampered': 0, 'gt_authentic': 0})
    # Metric lists
    bbox_scores = defaultdict(list)
    pixel_iou_scores = defaultdict(list)
    pixel_f1_scores = defaultdict(list)
    missing_gt_masks = defaultdict(int)
    
    # 4. 遍历评估
    for sample in tqdm(all_samples, desc="Evaluating Samples"):
        ds_name_full = sample.get('dataset', 'unknown')
        base_dataset = get_base_dataset_name(ds_name_full)
        
        # Ground Truth Info
        is_tampered_gt = bool(sample.get('is_tampered', False))
        image_path = sample.get('images', [''])[0]
        gt_bboxes = sample.get('bbox_objects', {}).get('bbox', [])
        
        if is_tampered_gt: class_stats[base_dataset]['gt_tampered'] += 1
        else: class_stats[base_dataset]['gt_authentic'] += 1

        # Prediction Info
        last_msg = next((msg for msg in reversed(sample.get('messages', [])) if msg.get('role') == 'assistant'), None)
        content = last_msg.get('content', '') if last_msg else ""
        
        is_pred_tampered, pred_bboxes_raw = parse_model_response_strict(content)
        pred_bboxes = [normalize_bbox(b, cfg.IMG_SIZE) for b in pred_bboxes_raw]

        # --- A. Classification Metrics ---
        if is_tampered_gt:
            if is_pred_tampered:
                class_stats[base_dataset]['tp'] += 1
                # 只有 TP 计算 BBox IoU
                miou = calculate_bbox_miou(gt_bboxes, pred_bboxes)
                bbox_scores[base_dataset].append(miou)
            else:
                class_stats[base_dataset]['fn'] += 1
                bbox_scores[base_dataset].append(0.0) # 漏报 IoU 为 0
        else:
            if not is_pred_tampered:
                class_stats[base_dataset]['tn'] += 1
            else:
                class_stats[base_dataset]['fp'] += 1

        # --- B. Pixel Metrics (仅对 GT 为 Tampered 的样本计算) ---
        # 逻辑：只要 GT 是篡改的，我们就对比生成的掩码和 GT 掩码。
        # 如果模型判为 Real (FN)，则生成掩码视为全黑，IoU/F1 自然较低。
        if is_tampered_gt and args.mask_dir:
            gt_mask_path = gt_lookup.get(image_path)
            if not gt_mask_path or not os.path.isfile(gt_mask_path):
                missing_gt_masks[ds_name_full] += 1
                continue
            
            # 构造预测掩码路径: mask_dir/dataset_name/basename.png
            # 注意：SAM2 生成脚本中的目录结构是按 dataset_name 分文件夹的
            img_basename = os.path.basename(image_path)
            mask_basename = os.path.splitext(img_basename)[0] + ".png"
            pred_mask_path = os.path.join(args.mask_dir, ds_name_full, mask_basename)
            
            p_iou, p_f1 = calculate_pixel_metrics(pred_mask_path, gt_mask_path, cfg.IMG_SIZE)
            
            pixel_iou_scores[base_dataset].append(p_iou)
            pixel_f1_scores[base_dataset].append(p_f1)

    if missing_gt_masks:
        total_missing = sum(missing_gt_masks.values())
        details = ", ".join(
            f"{name}={count}" for name, count in sorted(missing_gt_masks.items())
        )
        print(f"[WARN] skipped Pixel evaluation for {total_missing} tampered samples "
              f"with missing GT masks ({details})")

    # 5. 生成报告
    results_f1 = {}
    results_acc = {}
    results_bbox_iou = {}
    results_pixel_iou = {}
    results_pixel_f1 = {}
    
    sorted_datasets = sorted(class_stats.keys())
    
    # 汇总变量
    g_tp, g_fp, g_tn, g_fn = 0, 0, 0, 0
    all_bbox_vals = []
    all_pixel_iou_vals = []
    all_pixel_f1_vals = []
    
    for ds in sorted_datasets:
        s = class_stats[ds]
        g_tp += s['tp']; g_fp += s['fp']; g_tn += s['tn']; g_fn += s['fn']
        
        total = s['gt_tampered'] + s['gt_authentic']
        
        # Class F1
        results_f1[ds] = [total, calculate_f1_score(s['tp'], s['fp'], s['fn'])]
        
        # Accuracy
        acc = (s['tp'] + s['tn']) / total if total > 0 else 0
        acc_fake = s['tp'] / s['gt_tampered'] if s['gt_tampered'] > 0 else 0
        acc_real = s['tn'] / s['gt_authentic'] if s['gt_authentic'] > 0 else 0
        results_acc[ds] = [total, acc, acc_fake, acc_real]
        
        # BBox IoU
        b_vals = bbox_scores[ds]
        avg_bbox = np.mean(b_vals) if b_vals else 0.0
        results_bbox_iou[ds] = [s['gt_tampered'], avg_bbox]
        all_bbox_vals.extend(b_vals)
        
        # Pixel Metrics
        if args.mask_dir:
            p_iou_vals = pixel_iou_scores[ds]
            p_f1_vals = pixel_f1_scores[ds]
            
            avg_p_iou = np.mean(p_iou_vals) if p_iou_vals else 0.0
            avg_p_f1 = np.mean(p_f1_vals) if p_f1_vals else 0.0
            
            # Count 使用该数据集参与 Pixel 评估的样本数
            results_pixel_iou[ds] = [len(p_iou_vals), avg_p_iou]
            results_pixel_f1[ds] = [len(p_f1_vals), avg_p_f1]
            
            all_pixel_iou_vals.extend(p_iou_vals)
            all_pixel_f1_vals.extend(p_f1_vals)

    # Total Rows
    total_all = g_tp + g_fp + g_tn + g_fn
    results_f1['Total'] = [total_all, calculate_f1_score(g_tp, g_fp, g_fn)]
    results_acc['Total'] = [total_all, (g_tp+g_tn)/total_all if total_all>0 else 0, g_tp/(g_tp+g_fn) if (g_tp+g_fn)>0 else 0, g_tn/(g_tn+g_fp) if (g_tn+g_fp)>0 else 0]
    results_bbox_iou['Total'] = [len(all_bbox_vals), np.mean(all_bbox_vals) if all_bbox_vals else 0.0]
    
    if args.mask_dir:
        results_pixel_iou['Total'] = [len(all_pixel_iou_vals), np.mean(all_pixel_iou_vals) if all_pixel_iou_vals else 0.0]
        results_pixel_f1['Total'] = [len(all_pixel_f1_vals), np.mean(all_pixel_f1_vals) if all_pixel_f1_vals else 0.0]

    # One row per dataset plus a final aggregate row.
    report_rows = []
    report_order = sorted_datasets + ["Total"]
    total_stats = {
        "tp": g_tp,
        "fp": g_fp,
        "tn": g_tn,
        "fn": g_fn,
        "gt_tampered": g_tp + g_fn,
        "gt_authentic": g_tn + g_fp,
    }
    for ds in report_order:
        s = total_stats if ds == "Total" else class_stats[ds]
        pixel_iou = results_pixel_iou.get(ds)
        pixel_f1 = results_pixel_f1.get(ds)
        report_rows.append({
            "Dataset": ds,
            "Samples": results_acc[ds][0],
            "Tampered": s["gt_tampered"],
            "Authentic": s["gt_authentic"],
            "TP": s["tp"],
            "FP": s["fp"],
            "TN": s["tn"],
            "FN": s["fn"],
            "ACC": results_acc[ds][1],
            "Fake_ACC": results_acc[ds][2],
            "Real_ACC": results_acc[ds][3],
            "Class_F1": results_f1[ds][1],
            "BBox_Count": results_bbox_iou[ds][0],
            "BBox_IoU": results_bbox_iou[ds][1],
            "Pixel_Count": pixel_iou[0] if pixel_iou else None,
            "Pixel_IoU": pixel_iou[1] if pixel_iou else None,
            "Pixel_F1": pixel_f1[1] if pixel_f1 else None,
        })

    # Keep the public evaluation report compact and directly comparable across
    # datasets/checkpoints. Detailed confusion-matrix counts are used above to
    # compute these metrics but are intentionally omitted from the TSV.
    report_columns = [
        "Dataset",
        "Class_F1",
        "ACC",
        "Pixel_F1",
        "Pixel_IoU",
        "BBox_IoU",
    ]
    report_df = pd.DataFrame(report_rows)[report_columns]
    report_path = os.path.join(args.output_dir, "eval_metrics.tsv")
    report_df.to_csv(report_path, sep="\t", index=False, float_format="%.4f")
    print("\n" + report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[DONE] evaluation report -> {report_path}")

if __name__ == "__main__":
    main()

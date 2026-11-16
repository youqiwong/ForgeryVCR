"""
compute_gain_csv.py — Build diagnostic-rich gain CSVs from expert results.

Each strategy (NoTool/ELA/FFT/NPP/Zoom-In) contributes:
  * a vLLM JSONL with the predicted class and, for fake predictions, bounding
    boxes normalized to Qwen's [0, 1000] coordinate space;
  * an optional real-token JSONL with normalized P(real).

Predicted boxes are mapped back to the 832x832 image coordinate system before
computing the mean Hungarian-matched bbox IoU. Unmatched GT or predicted boxes
are penalized as zero by dividing the matched IoU sum by
max(num_gt_boxes, num_pred_boxes).

The two output CSVs expose, per strategy:
    <Tool>_Pred_Label       0=real, 1=fake
    <Tool>_Cls_Correct      0/1
    <Tool>_BBox_IoU         mean Hungarian IoU (tampered rows; blank for real)
    <Tool>_Real_Prob        normalized P(real), when --prob is supplied
    <Tool>_Selected         0/1
    <Tool>_Selection_Rank   1-based rank, blank when not selected

Selection:
  * Tampered: a non-baseline strategy must classify correctly, have IoU above
    --iou_threshold, and strictly outperform NoTool's IoU. Qualified strategies
    are ranked by IoU. NoTool is used only if no candidate qualifies.
  * Authentic: a non-baseline strategy must classify correctly, have normalized
    P(real) above --real_prob_threshold. Qualified strategies are ranked by
    P(real), without requiring them to outperform NoTool. NoTool is used only
    if no callable tool qualifies and the baseline itself is valid.

--top_fake/--top_real count only callable tools. NoTool is selected only when
the baseline is valid and no candidate tool qualifies.
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


BBOX_RE = re.compile(
    r"\((-?\d+)\s*,\s*(-?\d+)\)\s*,\s*\((-?\d+)\s*,\s*(-?\d+)\)"
)


def parse_response(text):
    """Return (pred_label, normalized_bboxes) from an expert response."""
    if not text:
        return 0, []
    content = text.lower().replace("\n", "").strip()
    match = re.search(r"<answer>(.*?)</answer>", content)
    inner = match.group(1) if match else content
    pred_label = 1 if "fake" in inner else 0
    bboxes = []
    for x1, y1, x2, y2 in BBOX_RE.findall(content):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        bboxes.append(
            [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        )
    return pred_label, bboxes


def denormalize_bbox(bbox, img_size=832, coord_max=1000):
    """Map one Qwen [0, coord_max] bbox to [0, img_size] pixel coordinates."""
    scale = img_size / float(coord_max)
    values = [
        max(0, min(img_size, int(value * scale)))
        for value in bbox
    ]
    x1, y1, x2, y2 = values
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def hungarian_iou(gt_boxes, pred_boxes):
    """Mean Hungarian IoU with unmatched boxes counted as zero."""
    if not gt_boxes or not pred_boxes:
        return 0.0
    matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            matrix[gt_idx, pred_idx] = bbox_iou(gt_box, pred_box)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return float(matrix[row_ind, col_ind].sum() / max(len(gt_boxes), len(pred_boxes)))


def image_path_of(entry):
    images = entry.get("images")
    if images:
        first = images[0]
        return first.get("path") if isinstance(first, dict) else first
    return entry.get("original_image") or entry.get("image_path")


def canonical_path(path, base_dir=None):
    value = Path(path)
    if not value.is_absolute():
        value = Path(base_dir or os.getcwd()) / value
    return str(value.resolve())


def portable_path(path):
    relative = os.path.relpath(path, os.getcwd())
    return relative if relative != ".." and not relative.startswith(f"..{os.sep}") else path


def response_of(entry):
    if entry.get("response"):
        return entry["response"]
    for message in reversed(entry.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def load_jsonl(path):
    rows = []
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] invalid JSONL skipped: {path}:{line_number}")
    return rows


def parse_kv(items):
    parsed = {}
    for item in items or []:
        name, separator, path = item.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"expected NAME=path, got: {item}")
        parsed[name.strip()] = path.strip()
    return parsed


def load_gt_map(path):
    gt_map = {}
    json_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as file:
        entries = json.load(file)
    for entry in entries:
        image_path = entry.get("forged_path") or entry.get("image_path")
        bboxes = entry.get("bboxes")
        if bboxes is None and isinstance(entry.get("bbox_objects"), dict):
            bboxes = entry["bbox_objects"].get("bbox", [])
        if image_path:
            gt_map[canonical_path(image_path, json_dir)] = bboxes or []
    return gt_map


def set_tool_columns(row, tool, result, real_probability, gt_label):
    pred_label = result.get("label") if result else None
    row[f"{tool}_Pred_Label"] = pred_label
    row[f"{tool}_Cls_Correct"] = (
        int(pred_label == gt_label) if pred_label is not None else 0
    )
    row[f"{tool}_BBox_IoU"] = (
        round(result.get("iou", 0.0), 6)
        if gt_label == 1 and result is not None
        else None
    )
    row[f"{tool}_Real_Prob"] = (
        round(real_probability, 8) if real_probability is not None else None
    )
    row[f"{tool}_Selected"] = 0
    row[f"{tool}_Selection_Rank"] = None


def mark_selected(row, ordered_tools):
    for rank, tool in enumerate(ordered_tools, start=1):
        row[f"{tool}_Selected"] = 1
        row[f"{tool}_Selection_Rank"] = rank
    row["Effective_Tools_List"] = (
        ",".join(ordered_tools) if ordered_tools else "hardsample"
    )
    row["Num_Selected_Tools"] = len(ordered_tools)
    row["Is_HardSample"] = int(not ordered_tools)


def output_columns(tools):
    columns = ["Image Path", "GT Label"]
    for tool in tools:
        columns.extend(
            [
                f"{tool}_Pred_Label",
                f"{tool}_Cls_Correct",
                f"{tool}_BBox_IoU",
                f"{tool}_Real_Prob",
                f"{tool}_Selected",
                f"{tool}_Selection_Rank",
            ]
        )
    columns.extend(
        ["Effective_Tools_List", "Num_Selected_Tools", "Is_HardSample"]
    )
    return columns


def print_summary(name, dataframe, tools):
    total = len(dataframe)
    hard = int(dataframe["Is_HardSample"].sum()) if total else 0
    retained = total - hard
    total_selections = int(dataframe["Num_Selected_Tools"].sum()) if total else 0

    print(f"\n===== {name} selection summary =====")
    print(f"total samples:       {total}")
    print(f"retained samples:    {retained} ({retained / total:.2%})" if total else "retained samples:    0")
    print(f"hardsamples:         {hard} ({hard / total:.2%})" if total else "hardsamples:         0")

    distribution = Counter(dataframe["Num_Selected_Tools"].tolist()) if total else {}
    print("selected-count distribution:")
    for count in sorted(distribution):
        print(f"  {count} strategies: {distribution[count]}")

    print("per-strategy selection:")
    for tool in tools:
        selected = int(dataframe[f"{tool}_Selected"].sum()) if total else 0
        sample_ratio = selected / retained if retained else 0.0
        selection_share = selected / total_selections if total_selections else 0.0
        cls_correct = int(dataframe[f"{tool}_Cls_Correct"].sum()) if total else 0
        cls_accuracy = cls_correct / total if total else 0.0
        metric = ""
        if name.lower().startswith("tampered"):
            mean_iou = dataframe[f"{tool}_BBox_IoU"].fillna(0).mean() if total else 0.0
            metric = f", mean_bbox_iou={mean_iou:.4f}"
        else:
            probabilities = dataframe[f"{tool}_Real_Prob"].dropna()
            mean_probability = probabilities.mean() if len(probabilities) else float("nan")
            metric = f", mean_P(real)={mean_probability:.4f}"
        print(
            f"  {tool}: selected={selected}, "
            f"sample_ratio={sample_ratio:.2%}, "
            f"selection_share={selection_share:.2%}, "
            f"cls_acc={cls_accuracy:.2%}{metric}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build gain-driven CSVs from expert test results"
    )
    parser.add_argument(
        "--gt_json", required=True, help="tampered JSON with forged_path + bboxes"
    )
    parser.add_argument(
        "--tool",
        action="append",
        required=True,
        help="NAME=path.jsonl; repeat for NoTool/ELA/FFT/NPP/Zoom-In",
    )
    parser.add_argument(
        "--prob",
        action="append",
        help="NAME=real-token-score.jsonl; repeat for each strategy",
    )
    parser.add_argument("--out_tp", required=True)
    parser.add_argument("--out_au", required=True)
    parser.add_argument("--top_fake", type=int, default=4)
    parser.add_argument("--top_real", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=832)
    parser.add_argument("--coord_max", type=int, default=1000)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--real_prob_threshold", type=float, default=0.5)
    args = parser.parse_args()

    tool_files = parse_kv(args.tool)
    probability_files = parse_kv(args.prob)
    tools = list(tool_files)
    if "NoTool" not in tools:
        parser.error("--tool must include NoTool=...")
    gt_map = load_gt_map(args.gt_json)

    results = defaultdict(dict)
    real_probabilities = defaultdict(dict)
    all_paths = set()

    for tool, jsonl_path in tool_files.items():
        if not os.path.exists(jsonl_path):
            print(f"[WARN] missing vLLM results for {tool}: {jsonl_path}")
            continue
        for entry in load_jsonl(jsonl_path):
            raw_image_path = image_path_of(entry)
            if not raw_image_path:
                continue
            image_path = canonical_path(raw_image_path)
            all_paths.add(image_path)
            label, normalized_bboxes = parse_response(response_of(entry))
            pixel_bboxes = [
                denormalize_bbox(
                    bbox, img_size=args.img_size, coord_max=args.coord_max
                )
                for bbox in normalized_bboxes
            ]
            iou = (
                hungarian_iou(gt_map[image_path], pixel_bboxes)
                if image_path in gt_map and label == 1
                else 0.0
            )
            results[image_path][tool] = {
                "label": label,
                "iou": iou,
                "pred_bboxes": pixel_bboxes,
            }

    for tool, jsonl_path in probability_files.items():
        if not os.path.exists(jsonl_path):
            print(f"[WARN] missing real-token scores for {tool}: {jsonl_path}")
            continue
        for entry in load_jsonl(jsonl_path):
            raw_image_path = (
                entry.get("original_image")
                or image_path_of(entry)
                or entry.get("image_path")
            )
            if not raw_image_path:
                continue
            image_path = canonical_path(raw_image_path)
            probability = entry.get(
                "normalized_real_prob", entry.get("real_prob")
            )
            if image_path is not None and probability is not None:
                all_paths.add(image_path)
                real_probabilities[image_path][tool] = float(probability)

    columns = output_columns(tools)

    tampered_rows = []
    for image_path in sorted(path for path in all_paths if path in gt_map):
        row = {"Image Path": portable_path(image_path), "GT Label": 1}
        for tool in tools:
            result = results[image_path].get(tool)
            probability = real_probabilities[image_path].get(tool)
            set_tool_columns(row, tool, result, probability, gt_label=1)

        baseline_result = results[image_path].get("NoTool")
        baseline_iou = (
            baseline_result["iou"]
            if baseline_result and baseline_result["label"] == 1
            else 0.0
        )
        baseline_valid = (
            baseline_result is not None
            and baseline_result["label"] == 1
            and baseline_iou > args.iou_threshold
        )
        improved_tools = sorted(
            (
                (tool, results[image_path][tool]["iou"])
                for tool in tools
                if tool != "NoTool"
                and tool in results[image_path]
                and results[image_path][tool]["label"] == 1
                and results[image_path][tool]["iou"] > args.iou_threshold
                and results[image_path][tool]["iou"] > baseline_iou
            ),
            key=lambda item: (-item[1], tools.index(item[0])),
        )
        ordered = [tool for tool, _ in improved_tools[: args.top_fake]]
        if not ordered and baseline_valid:
            ordered = ["NoTool"]
        mark_selected(row, ordered)
        tampered_rows.append(row)

    authentic_rows = []
    for image_path in sorted(path for path in all_paths if path not in gt_map):
        row = {"Image Path": portable_path(image_path), "GT Label": 0}
        for tool in tools:
            result = results[image_path].get(tool)
            probability = real_probabilities[image_path].get(tool)
            set_tool_columns(row, tool, result, probability, gt_label=0)

        baseline_result = results[image_path].get("NoTool")
        baseline_probability = real_probabilities[image_path].get("NoTool")
        baseline_valid = (
            baseline_result is not None
            and baseline_result["label"] == 0
            and baseline_probability is not None
            and baseline_probability > args.real_prob_threshold
        )
        qualified_tools = sorted(
            (
                (tool, real_probabilities[image_path][tool])
                for tool in tools
                if tool != "NoTool"
                and tool in results[image_path]
                and results[image_path][tool]["label"] == 0
                and tool in real_probabilities[image_path]
                and real_probabilities[image_path][tool]
                > args.real_prob_threshold
            ),
            key=lambda item: (-item[1], tools.index(item[0])),
        )
        ordered = [tool for tool, _ in qualified_tools[: args.top_real]]
        if not ordered and baseline_valid:
            ordered = ["NoTool"]
        mark_selected(row, ordered)
        authentic_rows.append(row)

    tampered_df = pd.DataFrame(tampered_rows).reindex(columns=columns)
    authentic_df = pd.DataFrame(authentic_rows).reindex(columns=columns)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tp)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_au)), exist_ok=True)
    tampered_df.to_csv(args.out_tp, index=False)
    authentic_df.to_csv(args.out_au, index=False)

    print(f"[DONE] tampered -> {args.out_tp} ({len(tampered_df)} rows)")
    print(f"[DONE] authentic -> {args.out_au} ({len(authentic_df)} rows)")
    print_summary("Tampered", tampered_df, tools)
    print_summary("Authentic", authentic_df, tools)


if __name__ == "__main__":
    main()

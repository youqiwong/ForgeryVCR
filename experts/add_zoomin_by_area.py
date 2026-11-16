"""
Add Scheme-B Zoom-In decisions to a tampered-sample gain CSV.

Zoom-In is selected when the tampered mask/bbox-union area is smaller than the
configured ratio. Its bbox is a deterministic perturbation around one GT bbox,
and the corresponding crop is saved for SFT generation.

Zoom-In is appended after gain-ranked callable tools. ISR hardsamples remain
hardsamples and are never rescued by this rule-only selector.
"""

import argparse
import json
import os
import random
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def deterministic_seed(path):
    return zlib.adler32(path.encode("utf-8")) & 0xFFFFFFFF


def canonical_path(path, base_dir=None):
    value = Path(path)
    if not value.is_absolute():
        value = Path(base_dir or os.getcwd()) / value
    return str(value.resolve())


def parse_strategy_list(value):
    if not isinstance(value, str) or value in ("", "hardsample"):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def is_hardsample(value):
    return isinstance(value, str) and value.strip().lower() == "hardsample"


def bbox_union_area_ratio(bboxes, img_size):
    if not bboxes:
        return 0.0
    grid = np.zeros((img_size, img_size), dtype=np.uint8)
    for bbox in bboxes:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_size, x2), min(img_size, y2)
        if x2 > x1 and y2 > y1:
            grid[y1:y2, x1:x2] = 1
    return float(grid.mean())


def tamper_area_ratio(entry, img_size):
    mask_path = entry.get("mask_path")
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.size:
            return float((mask > 127).mean())
    return bbox_union_area_ratio(entry.get("bboxes") or [], img_size)


def make_zoom_bbox(bboxes, image_width, image_height, img_size, seed):
    """Return matching (image-coordinate crop bbox, img_size-coordinate bbox)."""
    valid = [
        [float(value) for value in bbox]
        for bbox in bboxes
        if isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and float(bbox[2]) > float(bbox[0])
        and float(bbox[3]) > float(bbox[1])
    ]
    if not valid:
        return None, None

    rng = random.Random(seed)
    x1, y1, x2, y2 = rng.choice(valid)
    x1, x2 = x1 * image_width / img_size, x2 * image_width / img_size
    y1, y2 = y1 * image_height / img_size, y2 * image_height / img_size
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    scale = rng.uniform(1.1, 1.2)
    new_width = min(float(image_width), width * scale)
    new_height = min(float(image_height), height * scale)
    center_x = (x1 + x2) / 2 + rng.uniform(-0.05, 0.05) * width
    center_y = (y1 + y2) / 2 + rng.uniform(-0.05, 0.05) * height
    new_x1 = min(max(0.0, center_x - new_width / 2), image_width - new_width)
    new_y1 = min(max(0.0, center_y - new_height / 2), image_height - new_height)
    crop_bbox = [
        int(new_x1),
        int(new_y1),
        int(new_x1 + new_width),
        int(new_y1 + new_height),
    ]
    tool_bbox = [
        max(0, min(img_size, int(crop_bbox[0] * img_size / image_width))),
        max(0, min(img_size, int(crop_bbox[1] * img_size / image_height))),
        max(0, min(img_size, int(crop_bbox[2] * img_size / image_width))),
        max(0, min(img_size, int(crop_bbox[3] * img_size / image_height))),
    ]
    return crop_bbox, tool_bbox


def update_selection_columns(row, selected_tools):
    for column in [name for name in row.index if name.endswith("_Selected")]:
        row[column] = 0
    for column in [
        name for name in row.index if name.endswith("_Selection_Rank")
    ]:
        row[column] = np.nan
    for rank, tool in enumerate(selected_tools, start=1):
        row[f"{tool}_Selected"] = 1
        row[f"{tool}_Selection_Rank"] = rank
    row["Effective_Tools_List"] = (
        ",".join(selected_tools) if selected_tools else "hardsample"
    )
    row["Num_Selected_Tools"] = len(selected_tools)
    row["Is_HardSample"] = int(not selected_tools)


def main():
    parser = argparse.ArgumentParser(
        description="Add tampered Zoom-In calls using mask-area Scheme B"
    )
    parser.add_argument("--tp_csv", required=True)
    parser.add_argument("--bbox_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--crop_dir", required=True)
    parser.add_argument("--max_area_ratio", type=float, default=0.05)
    parser.add_argument(
        "--max_tools",
        type=int,
        default=4,
        help="maximum callable tools after adding Zoom-In; NoTool is not counted",
    )
    parser.add_argument("--img_size", type=int, default=832)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.max_area_ratio <= 1:
        parser.error("--max_area_ratio must be in [0,1]")
    if args.max_tools < 1:
        parser.error("--max_tools must be >=1")

    with open(args.bbox_json) as file:
        entries = json.load(file)
    bbox_json_dir = os.path.dirname(os.path.abspath(args.bbox_json))
    gt_map = {}
    for entry in entries:
        image_path = entry.get("forged_path") or entry.get("image_path")
        if image_path:
            normalized = dict(entry)
            normalized["forged_path"] = canonical_path(image_path, bbox_json_dir)
            mask_path = entry.get("mask_path")
            if mask_path:
                normalized["mask_path"] = canonical_path(mask_path, bbox_json_dir)
            gt_map[normalized["forged_path"]] = normalized

    dataframe = pd.read_csv(args.tp_csv)
    os.makedirs(args.crop_dir, exist_ok=True)
    ratios, selected_flags, bbox_values, crop_paths = [], [], [], []
    cropped = cached = missing = unreadable = missing_gt = 0

    for _, row in dataframe.iterrows():
        row_image_path = str(row["Image Path"])
        image_path = canonical_path(row_image_path)
        hard_sample = is_hardsample(row.get("Effective_Tools_List"))
        entry = gt_map.get(image_path)
        ratio, selected, bbox_text, crop_path = np.nan, False, "", ""

        if entry is None:
            missing_gt += 1
        else:
            ratio = tamper_area_ratio(entry, args.img_size)
            selected = (
                not hard_sample
                and ratio < args.max_area_ratio
                and bool(entry.get("bboxes"))
            )

        if selected and not os.path.exists(image_path):
            missing += 1
            selected = False
        if selected:
            image = cv2.imread(image_path)
            if image is None:
                unreadable += 1
                selected = False
            else:
                image_height, image_width = image.shape[:2]
                crop_bbox, tool_bbox = make_zoom_bbox(
                    entry.get("bboxes") or [],
                    image_width,
                    image_height,
                    args.img_size,
                    deterministic_seed(image_path),
                )
                selected = crop_bbox is not None
                if selected:
                    digest = deterministic_seed(image_path)
                    crop_path = os.path.join(
                        args.crop_dir,
                        f"{Path(image_path).stem}_{digest:08x}_zoomin.png",
                    )
                    if os.path.exists(crop_path) and not args.overwrite:
                        cached += 1
                    else:
                        x1, y1, x2, y2 = crop_bbox
                        crop = image[y1:y2, x1:x2]
                        if crop.size and cv2.imwrite(crop_path, crop):
                            cropped += 1
                        else:
                            selected, crop_path = False, ""
                    if selected:
                        bbox_text = json.dumps(tool_bbox, separators=(",", ":"))

        selected_tools = parse_strategy_list(row.get("Effective_Tools_List"))
        has_baseline = "NoTool" in selected_tools
        selected_tools = [
            tool for tool in selected_tools if tool not in ("Zoom-In", "NoTool")
        ]
        if selected:
            keep_count = max(0, args.max_tools - 1)
            selected_tools = selected_tools[:keep_count] + ["Zoom-In"]
        else:
            selected_tools = selected_tools[: args.max_tools]
        if has_baseline and not selected:
            selected_tools.append("NoTool")

        update_selection_columns(row, selected_tools)
        for column, value in row.items():
            dataframe.at[row.name, column] = value

        ratios.append(ratio)
        selected_flags.append(int(selected))
        bbox_values.append(bbox_text)
        crop_paths.append(
            os.path.relpath(crop_path, os.getcwd()) if crop_path else ""
        )

    dataframe["Tamper_Area_Ratio"] = ratios
    dataframe["ZoomIn_BBox"] = bbox_values
    dataframe["ZoomIn_Crop_Path"] = crop_paths
    dataframe["Zoom-In_Selected"] = selected_flags
    dataframe["Zoom-In_Selection_Rank"] = [
        (
            parse_strategy_list(value).index("Zoom-In") + 1
            if "Zoom-In" in parse_strategy_list(value)
            else np.nan
        )
        for value in dataframe["Effective_Tools_List"]
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    dataframe.to_csv(args.out_csv, index=False)
    selected_count = int(sum(selected_flags))
    print(f"[DONE] tampered Zoom-In CSV -> {args.out_csv}")
    print(f"total samples:       {len(dataframe)}")
    print(
        f"Zoom-In selected:    {selected_count} "
        f"({selected_count / len(dataframe):.2%})"
        if len(dataframe)
        else "Zoom-In selected:    0"
    )
    print(f"crops written:       {cropped}")
    print(f"crops cached:        {cached}")
    print(f"missing images:      {missing}")
    print(f"unreadable images:   {unreadable}")
    print(f"missing GT entries:  {missing_gt}")


if __name__ == "__main__":
    main()

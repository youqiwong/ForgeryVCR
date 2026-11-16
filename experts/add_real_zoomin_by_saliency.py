"""
Add Scheme-B Zoom-In decisions to an authentic-sample gain CSV.

For every authentic image:
  1. compute a training-free spectral-residual saliency map;
  2. measure how concentrated the saliency is (mass in the top-pixel fraction);
  3. if the score passes --score_threshold, select the strongest connected
     salient region, deterministically perturb/enlarge its bbox, and save a raw
     crop for the Zoom-In tool response;
  4. append Zoom-In after gain-ranked callable tools. ISR hardsamples remain
     hardsamples and are never rescued by this rule-only selector.

The output CSV adds:
    ZoomIn_Saliency_Score
    ZoomIn_BBox
    ZoomIn_Crop_Path
    Zoom-In_Selected
    Zoom-In_Selection_Rank

`data/sft/generate_SFT_datasets.py` consumes ZoomIn_BBox and ZoomIn_Crop_Path
so the recorded tool-call bbox and returned crop remain exactly aligned.
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


def canonical_path(path):
    value = Path(path)
    if not value.is_absolute():
        value = Path(os.getcwd()) / value
    return str(value.resolve())


def spectral_residual_saliency(image):
    """Return a [0,1] spectral-residual saliency map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    log_amplitude = np.log(amplitude + 1e-8)
    average_log_amplitude = cv2.boxFilter(
        log_amplitude,
        ddepth=-1,
        ksize=(3, 3),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    residual = log_amplitude - average_log_amplitude
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency = cv2.resize(
        saliency, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    minimum, maximum = float(saliency.min()), float(saliency.max())
    if maximum <= minimum + 1e-12:
        return np.zeros_like(saliency, dtype=np.float32)
    return ((saliency - minimum) / (maximum - minimum)).astype(np.float32)


def saliency_concentration_score(saliency, top_fraction=0.10):
    """Fraction of total saliency mass contained in the strongest pixels."""
    flat = saliency.reshape(-1)
    total = float(flat.sum())
    if total <= 1e-12:
        return 0.0
    count = max(1, int(round(len(flat) * top_fraction)))
    top_mass = float(np.partition(flat, len(flat) - count)[-count:].sum())
    return top_mass / total


def strongest_component_bbox(saliency, percentile=90.0):
    threshold = float(np.percentile(saliency, percentile))
    if threshold <= 0:
        return None
    binary = (saliency >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_bbox, best_mass = None, -1.0
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if area <= 0:
            continue
        mass = float(saliency[labels == component].sum())
        if mass > best_mass:
            best_mass = mass
            best_bbox = [int(x), int(y), int(x + width), int(y + height)]
    return best_bbox


def perturb_bbox(bbox, width, height, seed, min_crop_ratio=0.15):
    """Deterministically enlarge and slightly translate a salient bbox."""
    rng = random.Random(seed)
    x1, y1, x2, y2 = bbox
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    scale = rng.uniform(1.1, 1.2)
    new_width = max(box_width * scale, width * min_crop_ratio)
    new_height = max(box_height * scale, height * min_crop_ratio)
    center_x = (x1 + x2) / 2 + rng.uniform(-0.05, 0.05) * box_width
    center_y = (y1 + y2) / 2 + rng.uniform(-0.05, 0.05) * box_height

    new_width = min(float(width), new_width)
    new_height = min(float(height), new_height)
    new_x1 = min(max(0.0, center_x - new_width / 2), width - new_width)
    new_y1 = min(max(0.0, center_y - new_height / 2), height - new_height)
    return [
        int(new_x1),
        int(new_y1),
        int(new_x1 + new_width),
        int(new_y1 + new_height),
    ]


def scale_bbox(bbox, source_width, source_height, target_size):
    x1, y1, x2, y2 = bbox
    scale_x = target_size / float(source_width)
    scale_y = target_size / float(source_height)
    return [
        max(0, min(target_size, int(x1 * scale_x))),
        max(0, min(target_size, int(y1 * scale_y))),
        max(0, min(target_size, int(x2 * scale_x))),
        max(0, min(target_size, int(y2 * scale_y))),
    ]


def parse_strategy_list(value):
    if not isinstance(value, str) or value in ("", "hardsample"):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def is_hardsample(value):
    return isinstance(value, str) and value.strip().lower() == "hardsample"


def update_selection_columns(row, selected_tools):
    selected_columns = [
        column for column in row.index if column.endswith("_Selected")
    ]
    rank_columns = [
        column for column in row.index if column.endswith("_Selection_Rank")
    ]
    for column in selected_columns:
        row[column] = 0
    for column in rank_columns:
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
        description="Add authentic Zoom-In calls using saliency Scheme B"
    )
    parser.add_argument("--au_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--crop_dir", required=True)
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--saliency_percentile", type=float, default=90.0)
    parser.add_argument("--min_crop_ratio", type=float, default=0.15)
    parser.add_argument(
        "--max_tools",
        type=int,
        default=2,
        help="maximum callable tools after adding Zoom-In; NoTool is not counted",
    )
    parser.add_argument("--img_size", type=int, default=832)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.top_fraction < 1:
        parser.error("--top_fraction must be in (0,1)")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score_threshold must be in [0,1]")
    if not 0 < args.saliency_percentile < 100:
        parser.error("--saliency_percentile must be in (0,100)")
    if args.max_tools < 1:
        parser.error("--max_tools must be >=1")

    dataframe = pd.read_csv(args.au_csv)
    os.makedirs(args.crop_dir, exist_ok=True)

    scores, selected_flags, bbox_values, crop_paths = [], [], [], []
    missing = unreadable = cropped = cached = 0

    for _, row in dataframe.iterrows():
        row_image_path = str(row["Image Path"])
        image_path = canonical_path(row_image_path)
        hard_sample = is_hardsample(row.get("Effective_Tools_List"))
        score, selected, bbox_text, crop_path = np.nan, False, "", ""

        if hard_sample:
            pass
        elif not os.path.exists(image_path):
            missing += 1
        else:
            image = cv2.imread(image_path)
            if image is None:
                unreadable += 1
            else:
                saliency = spectral_residual_saliency(image)
                score = saliency_concentration_score(
                    saliency, top_fraction=args.top_fraction
                )
                component_bbox = strongest_component_bbox(
                    saliency, percentile=args.saliency_percentile
                )
                selected = (
                    component_bbox is not None
                    and score >= args.score_threshold
                )
                if selected:
                    height, width = image.shape[:2]
                    crop_bbox = perturb_bbox(
                        component_bbox,
                        width,
                        height,
                        deterministic_seed(image_path),
                        min_crop_ratio=args.min_crop_ratio,
                    )
                    tool_bbox = scale_bbox(
                        crop_bbox, width, height, target_size=args.img_size
                    )
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

        scores.append(score)
        selected_flags.append(int(selected))
        bbox_values.append(bbox_text)
        crop_paths.append(
            os.path.relpath(crop_path, os.getcwd()) if crop_path else ""
        )

    dataframe["ZoomIn_Saliency_Score"] = scores
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

    valid_scores = np.asarray([score for score in scores if not np.isnan(score)])
    selected_count = int(sum(selected_flags))
    non_hard_count = int((dataframe["Is_HardSample"] == 0).sum())
    print(f"[DONE] authentic Zoom-In CSV -> {args.out_csv}")
    print(f"total samples:       {len(dataframe)}")
    print(f"non-hardsamples:     {non_hard_count}")
    print(
        f"Zoom-In selected:    {selected_count} "
        f"({selected_count / non_hard_count:.2%} of non-hardsamples)"
        if non_hard_count
        else "Zoom-In selected:    0"
    )
    print(f"crops written:       {cropped}")
    print(f"crops cached:        {cached}")
    print(f"missing images:      {missing}")
    print(f"unreadable images:   {unreadable}")
    if len(valid_scores):
        print(
            "saliency score:      "
            f"mean={valid_scores.mean():.4f}, "
            f"p50={np.percentile(valid_scores, 50):.4f}, "
            f"p90={np.percentile(valid_scores, 90):.4f}, "
            f"max={valid_scores.max():.4f}"
        )


if __name__ == "__main__":
    main()

"""
mask_to_bbox.py — Build the ground-truth bbox JSON from a dataset's forged/mask pairs.

This evaluation data-prep step turns a dataset listing (a .txt of
`forged/xxx.png,gt/xxx_gt.png` pairs) into the portable `<dataset>.json`
consumed by `prepare_eval_dataset.py`.

Each GT mask is binarized, denoised (morphological opening + tiny-component
removal), and every remaining 8-connected component becomes one axis-aligned
bounding box  [x1, y1, x2, y2]  (x2 = x1 + width, y2 = y1 + height).

Input  .txt  (one pair per line, paths relative to --base_dir):
    forged/Tp_D_CND_M_N_ani00018_sec00096_00138.png,gt/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png
    forged/Tp_D_CND_M_N_art00076_art00077_10289.png,gt/Tp_D_CND_M_N_art00076_art00077_10289_gt.png

Output .json:
    [
        {
            "forged_path": "forged/....png",
            "mask_path":   "gt/...._gt.png",
            "bboxes": [[0, 0, 345, 793]]
        },
        ...
    ]

Example
-------
python data/prepare/mask_to_bbox.py \
  --txt      datasets/test/832x/casia1_tp/casia1_tp.txt \
  --base_dir datasets/test/832x/casia1_tp \
  --out      datasets/test/832x/casia1_tp/casia1_tp.json

Notes
-----
* Convention: tampered region is BRIGHT (white, >127) on a black background — the
  usual CASIA2 GT-mask polarity. If your masks are inverted, pass --invert.
* --min_area_ratio drops noise specks (fraction of the whole image, default 0.05%).
"""

import os
import cv2
import json
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kwargs):
        return x


def mask_to_bboxes(mask_path, min_area_ratio=0.0005, open_ksize=3, invert=False):
    """Return a list of [x1, y1, x2, y2] boxes for one mask, or None if unreadable."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = mask.shape[:2]

    _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if invert:
        binm = cv2.bitwise_not(binm)

    # Denoise: morphological opening removes isolated speckles.
    if open_ksize and open_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, kernel)

    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)
    min_area = max(1, int(min_area_ratio * h * w))

    bboxes = []
    for i in range(1, num):  # label 0 is the background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue  # drop noise points
        bboxes.append([int(x), int(y), int(x + bw), int(y + bh)])
    return bboxes


def _process_line(line, base_dir, min_area_ratio, open_ksize, invert):
    line = line.strip()
    if not line or ',' not in line:
        return None
    forged_rel, mask_rel = [p.strip() for p in line.split(',', 1)]
    forged_path = os.path.join(base_dir, forged_rel)
    mask_path = os.path.join(base_dir, mask_rel)

    if not os.path.exists(mask_path):
        return {"_error": f"mask not found: {mask_path}"}

    bboxes = mask_to_bboxes(mask_path, min_area_ratio, open_ksize, invert)
    if bboxes is None:
        return {"_error": f"unreadable mask: {mask_path}"}
    return {
        # Persist dataset-local paths. The output JSON sits in the dataset
        # directory, so it remains valid after the repository is relocated.
        "forged_path": forged_rel.replace(os.sep, "/"),
        "mask_path": mask_rel.replace(os.sep, "/"),
        "bboxes": bboxes,
    }


def main():
    ap = argparse.ArgumentParser(description="Mask -> connected-component bboxes -> dataset JSON")
    ap.add_argument('--txt', required=True, help="dataset listing (forged,gt per line)")
    ap.add_argument('--base_dir', required=True, help="root the txt paths are relative to")
    ap.add_argument('--out', required=True, help="output .json path")
    ap.add_argument('--min_area_ratio', type=float, default=0.0005,
                    help="drop components smaller than this fraction of the image (default 0.05%%)")
    ap.add_argument('--open_ksize', type=int, default=3, help="morphological opening kernel (0 disables)")
    ap.add_argument('--invert', action='store_true', help="tampered region is DARK instead of bright")
    ap.add_argument('--workers', type=int, default=8, help="parallel workers")
    args = ap.parse_args()

    with open(args.txt) as f:
        lines = [ln for ln in f if ln.strip()]
    print(f"[INFO] {len(lines)} entries from {args.txt}")

    worker = partial(_process_line, base_dir=args.base_dir,
                     min_area_ratio=args.min_area_ratio,
                     open_ksize=args.open_ksize, invert=args.invert)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(worker, lines, chunksize=16), total=len(lines)))
    else:
        results = [worker(ln) for ln in tqdm(lines)]

    records, errors, empty = [], 0, 0
    for r in results:
        if r is None:
            continue
        if "_error" in r:
            errors += 1
            continue
        if not r["bboxes"]:
            empty += 1
            continue
        records.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(records, f, indent=4)

    print(f"[DONE] wrote {len(records)} records -> {args.out}")
    print(f"       errors(unreadable/missing): {errors} | empty-bbox samples dropped: {empty}")


if __name__ == "__main__":
    main()

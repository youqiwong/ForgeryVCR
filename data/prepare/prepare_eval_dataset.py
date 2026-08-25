#!/usr/bin/env python3
"""Build portable evaluation manifests using paths relative to the JSON files."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from mask_to_bbox import mask_to_bboxes


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TOOL_LAYOUT = {
    "ELA": ("ELA", "_ELA.png"),
    "FFT": ("FFT", "_FFT.png"),
    "NPP": ("NPP", "_NPP.png"),
}


def relpath(path, start):
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()


def listed_images(dataset_dir, dataset_name):
    txt_path = dataset_dir / f"{dataset_name}.txt"
    if txt_path.is_file():
        return [
            line.split(",", 1)[0].strip()
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [
        path.relative_to(dataset_dir).as_posix()
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def resize_image(source, target, size, is_mask=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("L" if is_mask else "RGB")
        image = image.resize(
            (size, size),
            resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS,
        )
        image.save(target)


def prepare_resized_copy(source_root, data_root, dataset_name, img_size):
    source_dir = source_root / dataset_name
    target_dir = data_root / dataset_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source dataset directory not found: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    txt_source = source_dir / f"{dataset_name}.txt"
    is_tampered = dataset_name.lower().endswith("_tp")
    if is_tampered:
        if not txt_source.is_file():
            raise FileNotFoundError(f"missing tampered listing: {txt_source}")
        lines = [line.strip() for line in txt_source.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        for line in tqdm(lines, desc=f"{dataset_name}: resize"):
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) != 2:
                raise ValueError(f"expected forged,gt pair in {txt_source}: {line}")
            forged_rel, mask_rel = parts
            resize_image(source_dir / forged_rel, target_dir / forged_rel, img_size)
            resize_image(source_dir / mask_rel, target_dir / mask_rel, img_size, is_mask=True)
        (target_dir / f"{dataset_name}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
    else:
        rel_images = listed_images(source_dir, dataset_name)
        for image_rel in tqdm(rel_images, desc=f"{dataset_name}: resize"):
            resize_image(source_dir / image_rel, target_dir / image_rel, img_size)
        (target_dir / f"{dataset_name}.txt").write_text(
            "".join(f"{path}\n" for path in rel_images), encoding="utf-8"
        )


def load_portable_tampered_json(dataset_dir, dataset_name):
    json_path = dataset_dir / f"{dataset_name}.json"
    if not json_path.is_file():
        return None

    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"expected a JSON list: {json_path}")

    normalized = []
    empty_bboxes = 0
    for index, record in enumerate(records):
        forged_value = record.get("forged_path")
        mask_value = record.get("mask_path")
        if not forged_value or not mask_value:
            raise ValueError(
                f"missing forged_path/mask_path in {json_path}, record {index}"
            )

        forged_path = Path(forged_value)
        mask_path = Path(mask_value)
        if not forged_path.is_absolute():
            forged_path = dataset_dir / forged_path
        if not mask_path.is_absolute():
            mask_path = dataset_dir / mask_path
        forged_path = forged_path.resolve()
        mask_path = mask_path.resolve()

        if not forged_path.is_file():
            raise FileNotFoundError(f"missing forged image: {forged_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing GT mask: {mask_path}")

        bboxes = record.get("bboxes")
        if bboxes is None:
            return None
        if not bboxes:
            empty_bboxes += 1
            continue
        normalized.append({
            **record,
            "forged_path": relpath(forged_path, dataset_dir),
            "mask_path": relpath(mask_path, dataset_dir),
            "bboxes": bboxes,
        })

    # Normalize any legacy absolute paths in place so the prepared dataset stays
    # portable after extraction into a different repository location.
    json_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    if empty_bboxes:
        print(f"[WARN] {dataset_name}: dropped {empty_bboxes} tampered samples "
              "with empty bboxes from the existing JSON")
    return normalized


def load_or_build_tampered(dataset_dir, dataset_name, min_area_ratio, open_ksize, invert):
    existing = load_portable_tampered_json(dataset_dir, dataset_name)
    if existing is not None:
        return existing

    txt_path = dataset_dir / f"{dataset_name}.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"missing tampered listing: {txt_path}")

    records = []
    empty_bboxes = 0
    for line in tqdm(txt_path.read_text(encoding="utf-8").splitlines(),
                     desc=f"{dataset_name}: masks"):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            raise ValueError(f"expected forged,gt pair in {txt_path}: {line}")
        forged_rel, mask_rel = parts
        forged_path = dataset_dir / forged_rel
        mask_path = dataset_dir / mask_rel
        if not forged_path.is_file():
            raise FileNotFoundError(f"missing forged image: {forged_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing GT mask: {mask_path}")
        bboxes = mask_to_bboxes(
            str(mask_path),
            min_area_ratio=min_area_ratio,
            open_ksize=open_ksize,
            invert=invert,
        )
        if bboxes is None:
            raise ValueError(f"unreadable GT mask: {mask_path}")
        if not bboxes:
            empty_bboxes += 1
            continue
        records.append({
            "forged_path": Path(forged_rel).as_posix(),
            "mask_path": Path(mask_rel).as_posix(),
            "bboxes": bboxes,
        })

    output_path = dataset_dir / f"{dataset_name}.json"
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if empty_bboxes:
        print(f"[WARN] {dataset_name}: dropped {empty_bboxes} tampered samples "
              "whose GT masks produced no bboxes")
    return records


def load_or_build_authentic(dataset_dir, dataset_name):
    txt_path = dataset_dir / f"{dataset_name}.txt"
    if txt_path.is_file():
        rel_images = listed_images(dataset_dir, dataset_name)
    else:
        rel_images = [
            path.relative_to(dataset_dir).as_posix()
            for path in sorted(dataset_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        txt_path.write_text("".join(f"{path}\n" for path in rel_images), encoding="utf-8")

    for image_rel in rel_images:
        if not (dataset_dir / image_rel).is_file():
            raise FileNotFoundError(f"missing authentic image: {dataset_dir / image_rel}")
    return rel_images


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path,
                        help="optional raw dataset root; datasets are resized into --data_root")
    parser.add_argument("--data_root", type=Path,
                        default=repo_root / "datasets" / "test" / "832x")
    parser.add_argument("--datasets", nargs="+", default=["casia1_tp", "casia1_au"])
    parser.add_argument("--tool_root", type=Path)
    parser.add_argument("--output", default="eval_dataset.json")
    parser.add_argument("--img_size", type=int, default=832)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--open_ksize", type=int, default=3)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    tool_root = (args.tool_root or data_root / "tool_maps").resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    if args.source_root:
        source_root = args.source_root.resolve()
        for dataset_name in args.datasets:
            prepare_resized_copy(source_root, data_root, dataset_name, args.img_size)

    manifest = []
    tool_indices = {tool: [] for tool in TOOL_LAYOUT}
    missing_tools = {tool: 0 for tool in TOOL_LAYOUT}

    for dataset_name in args.datasets:
        dataset_dir = data_root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

        is_tampered = dataset_name.lower().endswith("_tp")
        if is_tampered:
            records = load_or_build_tampered(
                dataset_dir, dataset_name, args.min_area_ratio, args.open_ksize, args.invert
            )
            samples = [(record["forged_path"], record["bboxes"]) for record in records]
        else:
            samples = [(path, []) for path in load_or_build_authentic(dataset_dir, dataset_name)]

        for image_rel, bboxes in samples:
            image_path = dataset_dir / image_rel
            manifest.append({
                "images": [relpath(image_path, data_root)],
                "dataset": dataset_name,
                "is_tampered": is_tampered,
                "bbox_objects": {"bbox": bboxes},
            })
            for tool, (subdir, suffix) in TOOL_LAYOUT.items():
                tool_path = tool_root / dataset_name / subdir / f"{image_path.stem}{suffix}"
                tool_indices[tool].append({
                    "image_path": relpath(image_path, data_root),
                    "tool_map_path": relpath(tool_path, data_root),
                })
                if not tool_path.is_file():
                    missing_tools[tool] += 1

    manifest_path = data_root / args.output
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for tool, records in tool_indices.items():
        path = data_root / f"tool_maps_{tool}.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"[DONE] evaluation manifest: {manifest_path} ({len(manifest)} samples)")
    for tool in TOOL_LAYOUT:
        print(f"       {tool}: {len(tool_indices[tool])} indexed, "
              f"{missing_tools[tool]} tool maps not generated yet")


if __name__ == "__main__":
    main()

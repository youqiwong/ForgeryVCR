#!/usr/bin/env python3
"""Validate the public inference workflow and its repository-local assets."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, message: str) -> None:
        print(f"[ OK ] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")

    def require_path(self, path: Path, label: str, *, directory: bool = False) -> None:
        exists = path.is_dir() if directory else path.is_file()
        if exists:
            try:
                display_path = path.relative_to(REPO_ROOT)
            except ValueError:
                display_path = path
            self.ok(f"{label}: {display_path}")
        else:
            kind = "directory" if directory else "file"
            self.fail(f"missing {kind} for {label}: {path}")

    def require_import(self, module: str) -> None:
        if importlib.util.find_spec(module) is None:
            self.fail(f"missing Python package: {module}")
        else:
            self.ok(f"Python package: {module}")

    def validate_json_paths(
        self,
        json_path: Path,
        label: str,
        path_fields: tuple[str, ...],
    ) -> None:
        try:
            records = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.fail(f"cannot read {label}: {json_path} ({exc})")
            return
        if not isinstance(records, list):
            self.fail(f"{label} must contain a JSON list: {json_path}")
            return

        missing: list[str] = []
        malformed = 0
        for record in records:
            if not isinstance(record, dict):
                malformed += 1
                continue
            for field in path_fields:
                values = record.get(field)
                if field == "images":
                    values = values if isinstance(values, list) else []
                else:
                    values = [values] if isinstance(values, str) else []
                if not values:
                    malformed += 1
                    continue
                for value in values:
                    path = Path(value)
                    if not path.is_absolute():
                        path = json_path.parent / path
                    if not path.is_file():
                        missing.append(str(path))

        if malformed:
            self.fail(f"{label}: {malformed} malformed path field(s)")
        if missing:
            preview = ", ".join(missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            self.fail(f"{label}: {len(missing)} referenced file(s) missing: {preview}{suffix}")
        if not malformed and not missing:
            self.ok(f"{label}: {len(records)} records and all referenced files exist")


def resolved_json_paths(json_path: Path, field: str) -> list[Path]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    resolved: list[Path] = []
    for record in records:
        values = record.get(field, [])
        if isinstance(values, str):
            values = [values]
        for value in values if isinstance(values, list) else []:
            path = Path(value)
            resolved.append(path if path.is_absolute() else (json_path.parent / path).resolve())
    return resolved


def main() -> int:
    report = Report()

    print("\n== Python environment ==")
    for module in (
        "torch", "swift", "vllm", "transformers", "termcolor",
        "PIL", "cv2", "numpy", "pandas", "scipy",
    ):
        report.require_import(module)

    print("\n== Inference model and evaluation data ==")
    model_path = Path(
        os.environ.get(
            "INFERENCE_MODEL_PATH",
            REPO_ROOT / "weights" / "ForgeryVCR" / "GRPO",
        )
    ).expanduser()
    data_root = Path(
        os.environ.get(
            "EVAL_DATA_ROOT",
            REPO_ROOT / "datasets" / "test" / "832x",
        )
    ).expanduser()
    report.require_path(model_path, "released inference model", directory=True)
    if model_path.is_dir():
        report.require_path(model_path / "config.json", "model config")
        report.require_path(model_path / "tokenizer_config.json", "tokenizer config")
        if list(model_path.glob("*.safetensors")):
            report.ok("model weights: safetensors found")
        else:
            report.fail(f"missing model weights (*.safetensors): {model_path}")

    manifest_path = data_root / "eval_dataset.json"
    report.require_path(manifest_path, "evaluation manifest")
    manifest_images: set[Path] = set()
    if manifest_path.is_file():
        report.validate_json_paths(manifest_path, "evaluation manifest", ("images",))
        try:
            manifest_paths = resolved_json_paths(manifest_path, "images")
            manifest_images = set(manifest_paths)
            if len(manifest_paths) != len(manifest_images):
                report.fail("evaluation manifest contains duplicate image paths")
        except Exception as exc:
            report.fail(f"cannot compare evaluation manifest coverage: {exc}")
    for tool in ("ELA", "FFT", "NPP"):
        index_path = data_root / f"tool_maps_{tool}.json"
        report.require_path(index_path, f"{tool} map index")
        if index_path.is_file():
            report.validate_json_paths(
                index_path,
                f"{tool} map index",
                ("image_path", "tool_map_path"),
            )
            try:
                index_paths = resolved_json_paths(index_path, "image_path")
                index_images = set(index_paths)
                if len(index_paths) != len(index_images):
                    report.fail(f"{tool} map index contains duplicate source-image paths")
                missing_entries = manifest_images - index_images
                extra_entries = index_images - manifest_images
                if missing_entries or extra_entries:
                    report.fail(
                        f"{tool} map coverage differs from the evaluation manifest: "
                        f"missing={len(missing_entries)}, extra={len(extra_entries)}"
                    )
                elif manifest_images:
                    report.ok(f"{tool} map coverage matches all manifest images")
            except Exception as exc:
                report.fail(f"cannot compare {tool} map coverage: {exc}")

    print("\n== Optional SAM2 mask generation ==")
    use_sam2 = os.environ.get("INFERENCE_USE_SAM2", "1").lower() not in {
        "0", "false", "no", "off",
    }
    if use_sam2:
        report.require_import("sam2")
        sam2_checkpoint = Path(
            os.environ.get(
                "SAM2_CHECKPOINT",
                REPO_ROOT / "weights" / "sam2-hiera-large" / "sam2_hiera_large.pt",
            )
        ).expanduser()
        report.require_path(sam2_checkpoint, "SAM2 checkpoint")
    else:
        report.ok("SAM2 checks skipped because INFERENCE_USE_SAM2=0")

    if report.failures:
        print(f"\nSetup check failed: {report.failures} required item(s) missing.")
        return 1
    print("\nSetup check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

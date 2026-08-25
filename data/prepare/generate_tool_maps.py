"""
generate_tool_maps.py — Batch-render forensic tool maps for a dataset.

For every image listed in a dataset's `<name>.txt`, this posts to the forensic
API gateway (started by `tools/start.sh`) and caches the returned forensic map in the
inference layout:

    <out_root>/<dataset>/ELA/<stem>_ELA.png
    <out_root>/<dataset>/FFT/<stem>_FFT.png
    <out_root>/<dataset>/NPP/<stem>_NPP.png

Prerequisite: the tool services must be running —
    bash tools/start.sh            # gateway on 127.0.0.1:5000

Example
-------
python data/prepare/generate_tool_maps.py \
    --base_dir datasets/test/832x \
    --datasets casia1_tp casia1_au \
    --out_root datasets/test/832x/tool_maps

Resumable: outputs that already exist are skipped. Failures are logged and the
run continues; re-run to retry only the missing ones.
"""

import os
import time
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Public tool name -> (gateway endpoint, output subdir, filename suffix).
TOOL_OUTPUT_CONFIG = {
    "ELA": ("ela", "ELA", "_ELA.png"),
    "FFT": ("fft", "FFT", "_FFT.png"),
    "NPP": ("npp", "NPP", "_NPP.png"),
}
TOOL_ALIASES = {
    "ela": "ELA",
    "fft": "FFT",
    "npp": "NPP",
}

# These operators depend on the raw pixel grid / high-freq noise; never resized.
NO_PROXY = {"http": None, "https": None}


def call_gateway(task):
    tool, image_path, output_path, gateway, retries, retry_delay, timeout = task
    if not os.path.exists(image_path):
        return ("error", f"input not found: {image_path}")

    url = f"{gateway}/{tool}"
    payload = {
        "image_path": image_path,
        "output_path": output_path,
        "original_basename": Path(image_path).stem,
    }
    err = ""
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout, proxies=NO_PROXY)
            if r.status_code == 200:
                return ("success", r.json().get("output_path", output_path))
            err = f"{tool} {image_path}: HTTP {r.status_code} {r.text[:200]}"
        except requests.exceptions.RequestException as e:
            err = f"{tool} {image_path}: {e}"
        if attempt < retries - 1:
            time.sleep(retry_delay)
    logging.error(err)
    return ("error", err)


def build_tasks(base_dir, datasets, out_root, tools, gateway, retries, retry_delay, timeout):
    # Send absolute paths to the persistent services so their behavior does not
    # depend on the directory from which tools/start.sh was launched.
    base_dir = Path(base_dir).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    tasks, skipped = [], 0
    for ds in datasets:
        ds_dir = base_dir / ds
        txt = ds_dir / f"{ds_dir.name}.txt"
        if not txt.is_file():
            print(f"[WARN] listing not found, skipping: {txt}")
            continue
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rel = line.split(",")[0].strip()          # first column = image path
            img = ds_dir / rel
            stem = img.stem
            for tool in tools:
                endpoint, subdir, suffix = TOOL_OUTPUT_CONFIG[tool]
                out_dir = out_root / ds / subdir
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{stem}{suffix}"
                if out_path.is_file():
                    skipped += 1
                    continue
                tasks.append((endpoint, str(img), str(out_path),
                              gateway, retries, retry_delay, timeout))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser(description="Batch-render forensic tool maps via the API gateway")
    ap.add_argument("--base_dir", required=True, help="dataset root (contains <dataset>/<dataset>.txt)")
    ap.add_argument("--datasets", nargs="+", default=["casia2_tp", "casia2_au"],
                    help="dataset subdir names under --base_dir (may include e.g. test/casia1_tp)")
    ap.add_argument("--out_root", required=True, help="tool-map output root (subdirs created here)")
    ap.add_argument("--tools", nargs="+", default=list(TOOL_OUTPUT_CONFIG.keys()),
                    choices=list(TOOL_OUTPUT_CONFIG.keys()) + list(TOOL_ALIASES.keys()))
    ap.add_argument("--gateway", default="http://127.0.0.1:5000/run")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry_delay", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--log", default="tool_maps_failed.log")
    args = ap.parse_args()
    args.tools = list(dict.fromkeys(TOOL_ALIASES.get(tool, tool) for tool in args.tools))

    logging.basicConfig(filename=args.log, level=logging.ERROR,
                        format="%(asctime)s - %(message)s", force=True)

    # Require the gateway and every configured backend worker to be ready.
    health_url = args.gateway.rsplit("/run", 1)[0] + "/health"
    try:
        response = requests.get(health_url, timeout=10, proxies=NO_PROXY)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise SystemExit(
            f"[ERROR] forensic services are not ready at {health_url}: {exc}\n"
            "Run tools/start.sh and wait for its readiness confirmation."
        ) from exc

    tasks, skipped = build_tasks(args.base_dir, args.datasets, args.out_root, args.tools,
                                 args.gateway, args.retries, args.retry_delay, args.timeout)
    print(f"[INFO] tools={args.tools} datasets={args.datasets}")
    print(f"[INFO] {len(tasks)} to render, {skipped} already cached, {args.workers} workers")
    if not tasks:
        print("[DONE] nothing to do.")
        return

    ok = err = 0
    with Pool(args.workers) as pool:
        for status, _ in tqdm(pool.imap_unordered(call_gateway, tasks), total=len(tasks)):
            if status == "success":
                ok += 1
            else:
                err += 1

    print(f"[DONE] success={ok} failed={err}")
    if err:
        print(f"       see {os.path.abspath(args.log)} for failures; re-run to retry only the missing.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

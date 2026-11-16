"""
filter_done.py — Emit the not-yet-inferred subset of a QA dataset (for resume).

Reads a QA json (list of {messages, images, ...}) and an existing result JSONL
(from a crashed/partial `swift infer`), and writes a new QA json containing only
the samples whose first image path is NOT already present in the results.
With --limit, only the first N remaining samples are written (chunking).

    python filter_done.py --qa qa.json --done tool.jsonl --out todo.json --limit 3000
"""

import os
import json
import argparse


def first_image(entry):
    imgs = entry.get("images") or []
    if not imgs:
        return None
    x = imgs[0]
    return x.get("path") if isinstance(x, dict) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--done", required=True, help="existing result jsonl (may not exist yet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max remaining samples to emit (0 = all)")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.done):
        with open(args.done) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = first_image(json.loads(line))
                    if p:
                        done.add(p)
                except json.JSONDecodeError:
                    pass

    qa = json.load(open(args.qa))
    remaining = [it for it in qa if first_image(it) not in done]
    total_remaining = len(remaining)
    if args.limit and args.limit > 0:
        remaining = remaining[:args.limit]

    json.dump(remaining, open(args.out, "w"))
    print(f"[filter_done] done={len(done)} remaining={total_remaining} writing={len(remaining)}")


if __name__ == "__main__":
    main()

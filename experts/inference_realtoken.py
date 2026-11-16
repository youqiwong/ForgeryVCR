"""
inference_realtoken.py — Score each sample's normalized P(real) for one expert.

For authentic-sample gain ranking (Stage 1), we need how confidently a tool's
expert calls an image "real". This runs the merged expert over a QA dataset and,
at the position right after "<answer>", reads the logits for the "real" and
"fake" tokens, then reports the normalized probability:

    normalized_real_prob = P(real) / (P(real) + P(fake))

Output JSONL (one line per sample) — consumed by compute_gain_csv.py via --prob:
    {"original_image": ..., "tool_type": ..., "is_tampered": bool,
     "real_prob_raw": ..., "fake_prob_raw": ..., "normalized_real_prob": ...,
     "max_prob": ..., "max_token": ..., "dataset_source": ...}

Multi-GPU (torchrun). The input datasets are the per-tool QA jsons built by
build_qa_dataset.py (messages + images + label).

Example
-------
torchrun --nproc_per_node=8 experts/inference_realtoken.py \
    --tool_name ELA \
    --model  <expert_ckpt>-merged \
    --data   casia2_ela_qa.json \
    --output scores_ela.jsonl
"""

import os
import glob
import json
import argparse

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

DEFAULT_SYSTEM = "You are a helpful assistant in natural image tampering detection."


def parse_item(item):
    image_paths = item["images"]
    raw = item["messages"][0]["content"]
    text_prompt = raw.replace("<image>", "").strip()
    content = [{"type": "image"} for _ in image_paths] + [{"type": "text", "text": text_prompt}]
    return image_paths, content


def main():
    ap = argparse.ArgumentParser(description="Normalized P(real) scoring for one expert")
    ap.add_argument("--tool_name", required=True)
    ap.add_argument("--model", required=True, help="merged expert checkpoint dir")
    ap.add_argument("--data", nargs="+", required=True, help="QA json file(s)")
    ap.add_argument("--output", required=True, help="output jsonl")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # load data (possibly several json files)
    full = []
    for dp in args.data:
        try:
            sub = json.load(open(dp))
            src = os.path.basename(dp)
            for it in sub:
                it.setdefault("dataset", src)
            full.extend(sub)
        except Exception as e:
            if rank == 0:
                print(f"[WARN] failed to load {dp}: {e}")
    if not full:
        if rank == 0:
            print("[ERROR] no data loaded");
        dist.destroy_process_group();
        return
    if rank == 0:
        print(f"[{args.tool_name}] total {len(full)} samples")

    my = full[rank::world_size]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    real_id = processor.tokenizer.encode("real", add_special_tokens=False)[0]
    fake_id = processor.tokenizer.encode("fake", add_special_tokens=False)[0]

    results = []
    it = tqdm(my, desc=f"Rank {rank}", position=rank) if rank == 0 else my
    for item in it:
        try:
            image_paths, content = parse_item(item)
            if not all(os.path.exists(p) for p in image_paths):
                continue
            images = [Image.open(p).convert("RGB") for p in image_paths]
            messages = [
                {"role": "system", "content": args.system},
                {"role": "user", "content": content},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text += "<answer>"
            inputs = processor(text=[text], images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                rp = probs[0, real_id].item()
                fp = probs[0, fake_id].item()
                s = rp + fp
                norm = rp / s if s > 0 else 0.0
                top_p, top_id = torch.max(probs, dim=-1)
                max_tok = processor.tokenizer.decode([top_id[0].item()])

            is_tampered = (item.get("label", 0) == 1) if "label" in item else item.get("is_tampered", False)
            results.append({
                "original_image": image_paths[0],
                "tool_type": args.tool_name,
                "is_tampered": bool(is_tampered),
                "real_prob_raw": rp,
                "fake_prob_raw": fp,
                "normalized_real_prob": norm,
                "max_prob": top_p[0].item(),
                "max_token": max_tok,
                "dataset_source": item.get("dataset", "unknown"),
            })
        except Exception:
            continue

    part = f"{args.output}.part{rank}"
    with open(part, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    del model, processor
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        merged = []
        for pf in glob.glob(f"{args.output}.part*"):
            merged += [json.loads(l) for l in open(pf) if l.strip()]
            os.remove(pf)
        with open(args.output, "w") as f:
            for r in merged:
                f.write(json.dumps(r) + "\n")
        print(f"[DONE] {args.tool_name}: {len(merged)} scores -> {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

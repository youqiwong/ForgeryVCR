"""Balance SFT labels while preserving multi-tool trajectories.

The majority label is downsampled to the minority-label count by default.
Trajectories with at least ``--multi_tool_min`` calls are retained first. The
remaining budget is sampled proportionally across NoTool and single-tool exact
sequences. Sampling is deterministic for a fixed seed.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


def label_of(entry):
    label = entry.get("label")
    if label in (0, 1):
        return int(label)
    for message in reversed(entry.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).lower()
        if "<answer>" in content:
            return 1 if "fake" in content else 0
    raise ValueError("entry has neither a binary label nor a final answer")


def trajectory_of(entry):
    tools = []
    for message in entry.get("messages", []):
        if message.get("role") != "tool_call":
            continue
        content = message.get("content", "")
        if isinstance(content, dict):
            payload = content
        else:
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                payload = {}
        name = payload.get("name")
        if name:
            tools.append(str(name))
    return tuple(tools)


def proportional_sample(entries, target, rng):
    """Sample exact-trajectory strata proportionally with minimum coverage."""
    if target >= len(entries):
        return list(entries)
    if target <= 0:
        return []

    groups = defaultdict(list)
    for entry in entries:
        groups[trajectory_of(entry)].append(entry)
    keys = sorted(groups, key=lambda key: (len(key), key))
    quotas = {key: 0 for key in keys}

    if target >= len(keys):
        for key in keys:
            quotas[key] = 1
        remaining = target - len(keys)
    else:
        for key in sorted(keys, key=lambda key: (-len(groups[key]), key))[:target]:
            quotas[key] = 1
        remaining = 0

    while remaining:
        candidates = [
            key for key in keys if quotas[key] < len(groups[key])
        ]
        if not candidates:
            break
        capacity_total = sum(len(groups[key]) - quotas[key] for key in candidates)
        ideal = {
            key: remaining * (len(groups[key]) - quotas[key]) / capacity_total
            for key in candidates
        }
        additions = {
            key: min(
                len(groups[key]) - quotas[key],
                int(ideal[key]),
            )
            for key in candidates
        }
        added = sum(additions.values())
        if added:
            for key, count in additions.items():
                quotas[key] += count
            remaining -= added
            continue
        key = max(
            candidates,
            key=lambda item: (
                ideal[item] - int(ideal[item]),
                len(groups[item]) - quotas[item],
                item,
            ),
        )
        quotas[key] += 1
        remaining -= 1

    selected = []
    for key in keys:
        selected.extend(rng.sample(groups[key], quotas[key]))
    return selected


def downsample_label(entries, target, multi_tool_min, rng):
    if len(entries) <= target:
        return list(entries)

    multi = [
        entry for entry in entries
        if len(trajectory_of(entry)) >= multi_tool_min
    ]
    short = [
        entry for entry in entries
        if len(trajectory_of(entry)) < multi_tool_min
    ]

    if len(multi) <= target:
        selected = list(multi)
        selected.extend(proportional_sample(short, target - len(selected), rng))
        return selected

    selected = []
    remaining = target
    by_length = defaultdict(list)
    for entry in multi:
        by_length[len(trajectory_of(entry))].append(entry)
    for length in sorted(by_length, reverse=True):
        bucket = by_length[length]
        if len(bucket) <= remaining:
            selected.extend(bucket)
            remaining -= len(bucket)
        else:
            selected.extend(proportional_sample(bucket, remaining, rng))
            remaining = 0
        if not remaining:
            break
    return selected


def format_trajectory(trajectory):
    return "NoTool" if not trajectory else " -> ".join(trajectory)


def print_report(title, entries):
    print(f"\n===== {title} =====")
    total = len(entries)
    labels = Counter(label_of(entry) for entry in entries)
    print(f"total trajectories: {total}")
    for label, name in ((0, "REAL"), (1, "FAKE")):
        count = labels[label]
        print(f"{name}: {count} ({count / total:.2%})" if total else f"{name}: 0")

    for label, name in ((0, "REAL"), (1, "FAKE")):
        subset = [entry for entry in entries if label_of(entry) == label]
        if not subset:
            continue
        length_counts = Counter(len(trajectory_of(entry)) for entry in subset)
        exact_counts = Counter(trajectory_of(entry) for entry in subset)
        print(f"\n{name} tool-call lengths:")
        for length in sorted(length_counts):
            count = length_counts[length]
            label_text = "NoTool" if length == 0 else f"{length}-Tool"
            print(
                f"  {label_text}: {count} "
                f"(within-label={count / len(subset):.2%}, "
                f"overall={count / total:.2%})"
            )
        print(f"{name} exact trajectories:")
        for trajectory, count in sorted(
            exact_counts.items(), key=lambda item: (len(item[0]), item[0])
        ):
            print(
                f"  {format_trajectory(trajectory)}: {count} "
                f"(within-label={count / len(subset):.2%}, "
                f"overall={count / total:.2%})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Balance REAL/FAKE SFT trajectories with multi-tool priority"
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--target_per_label",
        type=int,
        help="default: the smaller of the two input label counts",
    )
    parser.add_argument(
        "--multi_tool_min",
        type=int,
        default=2,
        help="retain trajectories at or above this tool-call count first",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.multi_tool_min < 1:
        parser.error("--multi_tool_min must be >= 1")
    with open(args.input_json) as file:
        entries = json.load(file)
    if not isinstance(entries, list):
        raise ValueError("input JSON must be a list of SFT entries")

    by_label = defaultdict(list)
    for entry in entries:
        by_label[label_of(entry)].append(entry)
    if set(by_label) != {0, 1}:
        raise ValueError("input must contain both binary labels 0=real and 1=fake")

    target = args.target_per_label or min(len(by_label[0]), len(by_label[1]))
    if target < 1:
        parser.error("--target_per_label must be >= 1")
    if target > min(len(by_label[0]), len(by_label[1])):
        parser.error(
            "--target_per_label cannot exceed the smaller label count; "
            "this script does not oversample"
        )

    print_report("BEFORE BALANCING", entries)
    rng = random.Random(args.seed)
    balanced = []
    for label in (0, 1):
        balanced.extend(
            downsample_label(
                by_label[label],
                target=target,
                multi_tool_min=args.multi_tool_min,
                rng=rng,
            )
        )
    rng.shuffle(balanced)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    temporary_path = f"{args.output_json}.tmp"
    with open(temporary_path, "w") as file:
        json.dump(balanced, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, args.output_json)

    print_report("AFTER BALANCING", balanced)
    print(f"\n[DONE] balanced SFT dataset -> {args.output_json}")


if __name__ == "__main__":
    main()

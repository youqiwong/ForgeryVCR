"""Preview paper-defined MTS trajectory counts directly from final CSV files."""

import argparse
from collections import Counter

import pandas as pd


ALLOWED_TOOLS = {"ELA", "FFT", "NPP", "Zoom-In"}


def trajectories(value, max_tools):
    if not isinstance(value, str) or value in ("", "hardsample"):
        return []
    ordered = [item.strip() for item in value.split(",") if item.strip()]
    tools = []
    seen = set()
    for tool in ordered:
        if tool not in ALLOWED_TOOLS or tool in seen:
            continue
        tools.append(tool)
        seen.add(tool)
        if len(tools) == max_tools:
            break
    # MTS adds exactly one NoTool trajectory for every ISR-retained sample.
    paths = [()]
    paths.extend((tool,) for tool in tools)
    paths.extend(tuple(tools[:length]) for length in range(2, len(tools) + 1))
    return paths


def summarize(name, csv_path, max_tools):
    dataframe = pd.read_csv(csv_path)
    counts = Counter()
    retained = 0
    for value in dataframe["Effective_Tools_List"]:
        sample_paths = trajectories(value, max_tools)
        if sample_paths:
            retained += 1
            counts.update(sample_paths)

    print(f"\n===== {name} trajectories =====")
    print(f"images:              {len(dataframe)}")
    print(f"retained images:     {retained}")
    print(f"total trajectories:  {sum(counts.values())}")
    print("by tool-call length:")
    length_counts = Counter()
    for path, count in counts.items():
        length_counts[len(path)] += count
    for length in sorted(length_counts):
        label = "NoTool" if length == 0 else f"{length}-Tool"
        count = length_counts[length]
        print(f"  {label}: {count} ({count / sum(counts.values()):.2%})")
    print("exact trajectories:")
    for path, count in sorted(counts.items(), key=lambda item: (len(item[0]), item[0])):
        label = "NoTool" if not path else " -> ".join(path)
        print(f"  {label}: {count} ({count / sum(counts.values()):.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv")
    parser.add_argument("--fake_csv")
    parser.add_argument("--real_k", type=int, default=2)
    parser.add_argument("--fake_k", type=int, default=4)
    args = parser.parse_args()
    if not args.real_csv and not args.fake_csv:
        parser.error("at least one of --real_csv or --fake_csv is required")
    if args.real_csv:
        summarize(
            f"AUTHENTIC (K={args.real_k})",
            args.real_csv,
            max_tools=args.real_k,
        )
    if args.fake_csv:
        summarize(
            f"TAMPERED (K={args.fake_k})",
            args.fake_csv,
            max_tools=args.fake_k,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze the distribution of predicted direction sequences in an eval_results JSON file.
"""

import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def count_sequences(data: dict) -> Counter:
    sequences = [sample["predicted_path"] for sample in data["per_sample_results"]]
    return Counter(sequences)


def plot_distribution(counts: Counter, input_path: str, top_n: int = 30):
    # Sort by frequency descending, take top N
    most_common = counts.most_common(top_n)
    labels, freqs = zip(*most_common)

    total = sum(counts.values())
    unique = len(counts)

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6))

    bars = ax.bar(range(len(labels)), freqs, color="steelblue", edgecolor="white", linewidth=0.5)

    # Annotate bars with count and percentage
    for bar, freq in zip(bars, freqs):
        pct = freq / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{freq}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    title_suffix = f" (top {top_n} of {unique} unique)" if unique > top_n else f" ({unique} unique sequences)"
    ax.set_title(f"Ground-Truth Direction Sequence Distribution{title_suffix}\n{input_path}", fontsize=10)
    ax.set_xlabel("Direction sequence", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    fig.tight_layout()
    out_path = input_path.replace(".json", "_gt_distribution.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    plt.show()


def print_summary(counts: Counter, top_n: int = 20):
    total = sum(counts.values())
    unique = len(counts)
    print(f"\nTotal samples : {total}")
    print(f"Unique sequences: {unique}")
    print(f"\nTop {min(top_n, unique)} sequences:")
    print(f"{'Rank':<6} {'Count':<8} {'Pct':>7}   Sequence")
    print("-" * 60)
    for rank, (seq, cnt) in enumerate(counts.most_common(top_n), 1):
        pct = cnt / total * 100
        print(f"{rank:<6} {cnt:<8} {pct:>6.2f}%   {seq}")


def main():
    parser = argparse.ArgumentParser(description="Plot predicted sequence distribution from eval_results JSON.")
    parser.add_argument("input", help="Path to the eval_results JSON file")
    parser.add_argument("--top", type=int, default=30, help="Number of top sequences to display (default: 30)")
    args = parser.parse_args()

    data = load_json(args.input)
    counts = count_sequences(data)
    print_summary(counts, top_n=args.top)
    plot_distribution(counts, args.input, top_n=args.top)


if __name__ == "__main__":
    main()

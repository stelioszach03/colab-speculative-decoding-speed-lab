"""Plot descriptive cached/uncached throughput ratios from verified cell CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = defaultdict(dict)
    with args.cells.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["hardware"], row["model"], row["prefix"], int(row["concurrency"]), int(row["repeat"]))
            rows[key][int(row["cache"])] = float(row["completion_tokens_per_s"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True, layout="constrained")
    colors = {"shared": "#1C5B84", "unique": "#CB7B37"}
    for ax, hardware, title in zip(axes, ("rtx4090", "a10080"), ("RTX 4090", "A100 80GB")):
        for model_index, model in enumerate(("1.5b", "7b")):
            for prefix, shift in (("shared", -0.17), ("unique", 0.17)):
                ratios = []
                for repeat in (1, 2, 3, 4):
                    values = rows[(hardware, model, prefix, 32, repeat)]
                    assert set(values) == {0, 1}
                    ratios.append(values[1] / values[0])
                avg = mean(ratios)
                ax.bar(model_index + shift, avg, width=0.32, color=colors[prefix], label=f"{prefix} prefix" if model_index == 0 else None)
                ax.errorbar(model_index + shift, avg, yerr=[[avg - min(ratios)], [max(ratios) - avg]], fmt="none", ecolor="#17212F", capsize=3, linewidth=1.0)
                ax.text(model_index + shift, avg + 0.025, f"{avg:.2f}x", ha="center", va="bottom", fontsize=8, color="#17212F")
        ax.axhline(1.0, color="#6C7784", linewidth=0.9, linestyle="--")
        ax.set_xticks((0, 1), ("Qwen 1.5B", "Qwen 7B"))
        ax.set_title(title, fontsize=12, weight="bold", color="#17212F")
        ax.set_ylim(0, 1.67)
        ax.grid(axis="y", color="#E6EBF0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cached / uncached completion tokens per second")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Prefix caching at concurrency 32", fontsize=15, weight="bold", color="#17212F")
    fig.supxlabel("Mean and full range across four predeclared repeats; synthetic prompts; output hashes not always equal", fontsize=8, color="#536273")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

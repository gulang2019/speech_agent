"""Plot success rate versus replan steps for the fixed-condition sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Success rate vs. replan steps")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    x = [int(row["replan_steps"]) for row in rows]
    y = [100.0 * float(row["success_rate"]) for row in rows]
    low = [100.0 * float(row["success_rate_ci95_low"]) for row in rows]
    high = [100.0 * float(row["success_rate_ci95_high"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, y, color="#1769aa", marker="o", markersize=3.5, linewidth=1.8)
    ax.fill_between(x, low, high, color="#1769aa", alpha=0.18, linewidth=0)
    ax.set_xlabel("Replan steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, 51, 5))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

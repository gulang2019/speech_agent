"""Plot success-rate vs injected delay for the completed delay-grid tasks.

Reads ``<root>/<task>/part*/episodes.csv`` (the format written by
``scripts/xiaomi_latency_experiment.py``) and renders one panel per task plus a
combined panel, with Wilson 95% intervals and paired-bootstrap difference
summaries annotated.

The x-axis is the *injected* delay.  Under ``--delay-domain sim`` this is applied
as a fixed number of control steps and is the only source of staleness; under the
legacy ``wall`` domain it was added on top of the real model latency.

Usage:
    .conda-env/bin/python scripts/plot_delay_grid.py --root <dir> --output <png>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.xiaomi_latency_experiment import wilson_interval
except ModuleNotFoundError:  # pragma: no cover
    from xiaomi_latency_experiment import wilson_interval


TASK_TITLES = {
    "PickPlaceCounterToCabinet": "PickPlaceCounterToCabinet\n(horizon 750)",
    "OpenCabinet": "OpenCabinet\n(horizon 1800)",
    "OpenDoubleDoor": "OpenDoubleDoor\n(horizon 2400)",
    "TurnOffMicrowave": "TurnOffMicrowave\n(horizon 900)",
}
TASK_ORDER = list(TASK_TITLES)
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def load(root: Path, task: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for part in sorted((root / task).glob("part*/episodes.csv")):
        with part.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["error"]:
                    continue
                grouped[int(row["delay_ms"])].append(row)
    return dict(sorted(grouped.items()))


def summarise(rows_by_delay: dict[int, list[dict]]):
    delays, rates, lows, highs, counts = [], [], [], [], []
    for delay, rows in rows_by_delay.items():
        successes = sum(row["success"] == "True" for row in rows)
        total = len(rows)
        low, high = wilson_interval(successes, total)
        delays.append(delay)
        rates.append(successes / total)
        lows.append(low)
        highs.append(high)
        counts.append(total)
    return (
        np.array(delays),
        np.array(rates),
        np.array(lows),
        np.array(highs),
        np.array(counts),
    )


def make_overlay(args, tasks) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for task, color in zip(tasks, COLORS):
        data = load(args.root, task)
        if not data:
            continue
        delays, rates, lows, highs, _ = summarise(data)
        ax.fill_between(delays, lows, highs, color=color, alpha=0.12)
        ax.plot(delays, rates, marker="o", color=color, linewidth=2,
                label=task)
    ax.set_xlabel("injected delay (ms)")
    ax.set_ylabel("task success rate")
    ax.set_xticks(range(0, 101, 10))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        "RoboCasa / Xiaomi-Robotics-1: success rate vs injected delay\n"
        "(replan=5, rtc=off, 30 episodes per point, shaded = 95% Wilson CI)",
        fontsize=11,
    )
    fig.tight_layout()
    out = args.output.with_name(args.output.stem + "_overlay" + args.output.suffix)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--overlay", action="store_true",
                        help="also write a single-panel comparison chart")
    args = parser.parse_args()

    tasks = args.tasks or [
        t for t in TASK_ORDER if (args.root / t).is_dir()
    ]
    if args.overlay:
        make_overlay(args, tasks)
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(4.6 * len(tasks), 4.4), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, task, color in zip(axes, tasks, COLORS):
        data = load(args.root, task)
        if not data:
            continue
        delays, rates, lows, highs, _counts = summarise(data)
        ax.fill_between(delays, lows, highs, color=color, alpha=0.18,
                        label="95% Wilson CI")
        ax.plot(delays, rates, marker="o", color=color, linewidth=2,
                label="success rate")
        base = rates[0]
        ax.axhline(base, color="grey", linestyle="--", linewidth=1,
                   label=f"delay=0 baseline ({base:.0%})")
        for delay, rate in zip(delays, rates):
            ax.annotate(f"{rate:.0%}", (delay, rate), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
        ax.set_title(TASK_TITLES.get(task, task), fontsize=10)
        ax.set_xlabel("injected delay (ms)")
        ax.set_xticks(range(0, 101, 20))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("task success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    title = args.title or (
        "RoboCasa / Xiaomi-Robotics-1: success rate vs injected delay"
        "\n(replan=5, rtc=off, 30 episodes per point, legacy5 scenes, pretrain objects)"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

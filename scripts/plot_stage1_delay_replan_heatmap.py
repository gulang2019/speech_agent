"""Plot stage-1 delay x replan success-rate heatmaps for multiple models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_matrix(run_dir: Path, task: str):
    summary_files = [run_dir / "summary.csv"]
    if not summary_files[0].exists():
        summary_files = sorted(run_dir.glob("delay*/summary.csv"))
    if not summary_files or not summary_files[0].exists():
        summary_files = sorted(run_dir.rglob("summary.csv"))
    rows = []
    for summary_file in summary_files:
        with summary_file.open(newline="", encoding="utf-8") as file:
            rows.extend(row for row in csv.DictReader(file) if row["condition"])
    rows = [row for row in rows if row.get("task_name", task) == task]
    if not rows:
        raise ValueError(f"no rows for task {task!r} in {run_dir}")
    delays = sorted({int(row["delay_ms"]) for row in rows})
    replans = sorted({int(row["replan_steps"]) for row in rows})
    matrix = np.full((len(delays), len(replans)), np.nan)
    for row in rows:
        i = delays.index(int(row["delay_ms"]))
        j = replans.index(int(row["replan_steps"]))
        matrix[i, j] = 100.0 * float(row["success_rate"])
    if np.isnan(matrix).any():
        raise ValueError(f"incomplete delay x replan grid in {run_dir}")
    return delays, replans, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("MODEL", "DIR"), required=True)
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int, default=3)
    args = parser.parse_args()

    loaded = [(model, *load_matrix(Path(directory), args.task)) for model, directory in args.run]
    reference_delays, reference_replans = loaded[0][1:3]
    if any(delays != reference_delays or replans != reference_replans for _, delays, replans, _ in loaded):
        raise ValueError("all model runs must use the same delay and replan grid")

    ncols = min(2, len(loaded))
    nrows = (len(loaded) + ncols - 1) // ncols
    fig = plt.figure(figsize=(7.2 * ncols + 0.8, 4.2 * nrows), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [0.055],
        wspace=0.18,
        hspace=0.12,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(
                grid[row, col],
                sharex=axes[0, 0] if row or col else None,
                sharey=axes[0, 0] if row or col else None,
            )
    axes_flat = axes.ravel()
    vmin, vmax = 0.0, 100.0
    image = None
    for ax, (model, delays, replans, matrix) in zip(axes_flat, loaded):
        image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="YlGnBu")
        ax.set_title(model)
        ax.set_xticks(range(len(replans)))
        ax.set_xticklabels(replans, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(delays)))
        ax.set_yticklabels([f"{delay} ms" for delay in delays])
        for i in range(len(delays)):
            for j in range(len(replans)):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] >= 60 else "black")
        ax.set_xlabel("Replan steps")
        ax.set_ylabel("Injected delay")

    for ax in axes_flat[len(loaded):]:
        ax.set_visible(False)
    colorbar_axis = fig.add_subplot(grid[:, -1])
    fig.colorbar(image, cax=colorbar_axis, label="Success rate (%)")
    fig.suptitle(f"Stage 1: delay x replan success rate\n{args.task}, sim-domain delay, n={args.episodes_per_cell}/cell")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

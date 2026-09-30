"""Figures for the cross-model sim-domain delay grid.

Reads ``cross_model_grid.csv`` (written by ``scripts/analyze_sim_delay_grid.py``)
and renders:

* figure 1 - one panel per model, success rate vs injected delay, with a bold
  task-averaged curve plus thin per-task traces and 95% Wilson intervals;
* figures 2-5 - one figure per model, grouped bars of success rate per task with
  the four delay levels side by side (plus a task-averaged group).

All intervals are Wilson 95% intervals over 30 episodes per cell, so a single
bar carries roughly +-16pp of sampling noise; compare shapes, not millimetres.

Usage:
    .conda-env/bin/python scripts/plot_sim_delay_grid.py \
        --csv eval_results/simdelay_grid_analysis/cross_model_grid.csv \
        --output-dir eval_results/simdelay_grid_analysis/figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


MODEL_ORDER = ["gr00t_n1_5", "pi0_5", "xiaomi", "diffusion_policy"]
MODEL_LABELS = {
    "gr00t_n1_5": "GR00T N1.5",
    "pi0_5": "pi0.5",
    "xiaomi": "Xiaomi-Robotics-1",
    "diffusion_policy": "Diffusion Policy",
}
MODEL_COLORS = {
    "gr00t_n1_5": "#1f77b4",
    "pi0_5": "#d62728",
    "xiaomi": "#2ca02c",
    "diffusion_policy": "#9467bd",
}
DELAY_COLORS = ["#4c72b0", "#5aa469", "#e8a33d", "#c44e52"]
DELAY_LABELS = {0: "0 ms", 100: "100 ms", 300: "300 ms", 500: "500 ms"}

SHORT_TASK = {
    "PickPlaceCounterToCabinet": "PnP\nCounterToCabinet",
    "PickPlaceCounterToSink": "PnP\nCounterToSink",
    "PickPlaceCabinetToCounter": "PnP\nCabinetToCounter",
    "PickPlaceCounterToMicrowave": "PnP\nCounterToMicrowave",
    "PickPlaceCounterToStove": "PnP\nCounterToStove",
    "OpenDrawer": "Open\nDrawer",
    "TurnOffStove": "TurnOff\nStove",
    "CoffeeSetupMug": "CoffeeSetup\nMug",
}


def load(csv_path: Path):
    """Return rows[model][task][delay] = row, plus the task order."""
    rows: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    tasks: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            model, task = row["model"], row["task"]
            if model not in MODEL_ORDER:
                continue
            if task not in tasks:
                tasks.append(task)
            rows[model][task][int(row["delay_ms"])] = row
    return rows, tasks


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def macro_series(rows, model: str, delays: list[int]):
    """Episode-pooled success rate per delay for one model, with Wilson CI."""
    out = {}
    for delay in delays:
        successes = 0
        total = 0
        for task in rows[model]:
            row = rows[model][task].get(delay)
            if row is None:
                continue
            successes += int(row["successes"])
            total += int(row["num_episodes"])
        if not total:
            continue
        rate = successes / total
        low = high = rate
        if successes == 0:
            low, high = 0.0, 1 - 0.05 ** (1 / total)
        elif successes == total:
            low, high = 0.05 ** (1 / total), 1.0
        else:
            z = 1.959963985
            p = rate
            denom = 1 + z * z / total
            centre = (p + z * z / (2 * total)) / denom
            half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
            low, high = max(0.0, centre - half), min(1.0, centre + half)
        out[delay] = (rate, low, high, total, successes)
    return out


def significant_cells(rows, model: str, delays) -> set[tuple[str, int]]:
    """Cells whose paired McNemar test survives Holm correction at 0.05."""
    hits = set()
    for task in rows[model]:
        for delay in delays:
            row = rows[model][task].get(delay)
            if row is None or row.get("is_baseline") == "True":
                continue
            p_value = row.get("holm_adjusted_p_value")
            if p_value and float(p_value) < 0.05:
                hits.add((task, delay))
    return hits


def figure1(rows, delays, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.0), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        color = MODEL_COLORS[model]
        # thin per-task traces
        for task, by_delay in sorted(rows[model].items()):
            xs = [d for d in delays if d in by_delay]
            ys = [as_float(by_delay[d], "success_rate") for d in xs]
            ax.plot(xs, ys, color=color, alpha=0.28, linewidth=1.1, zorder=2)

        series = macro_series(rows, model, delays)
        xs = sorted(series)
        rate = np.array([series[d][0] for d in xs])
        low = np.array([series[d][1] for d in xs])
        high = np.array([series[d][2] for d in xs])
        ax.fill_between(xs, low, high, color=color, alpha=0.18, zorder=1)
        ax.errorbar(
            xs, rate, yerr=[rate - low, high - rate], color=color,
            marker="o", markersize=6, linewidth=2.4, capsize=4, zorder=3,
        )
        for x, y in zip(xs, rate):
            ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=color,
                        fontweight="bold", zorder=4)

        sig = significant_cells(rows, model, delays)
        for delay in delays:
            hits = sum(1 for task, d in sig if d == delay)
            if not hits:
                continue
            ax.annotate(
                f"{hits} task{'s' if hits > 1 else ''} sig.",
                (delay, high[list(xs).index(delay)]), textcoords="offset points",
                xytext=(0, 20), ha="center", fontsize=8, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7),
            )

        base = series[delays[0]][0]
        ax.axhline(base, color="grey", linestyle="--", linewidth=1, zorder=0)
        delta = series[delays[-1]][0] - base
        ax.set_title(
            f"{MODEL_LABELS[model]}\n0 ms -> 500 ms: {delta:+.1%}", fontsize=12
        )
        ax.set_xlabel("injected delay (ms)")
        ax.set_xticks(delays)
        ax.set_xlim(-25, max(delays) + 25)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("task success rate (pooled, 8 tasks)")
    handles = [
        Line2D([], [], color="grey", alpha=0.5, linewidth=1.1, label="individual task"),
        Line2D([], [], color="black", marker="o", linewidth=2.4, label="task-averaged (pooled)"),
        Line2D([], [], color="grey", linestyle="--", linewidth=1, label="0 ms baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(
        "Figure 1 - success rate vs injected delay, one panel per model\n"
        "sim-domain delay (100 ms = 2 control steps @ 20 Hz), replan=5, rtc=off, "
        "30 episodes per task/delay, 8 atomic tasks, pretrain20 scenes",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_per_model(rows, model: str, delays, out_path: Path, figure_index: int) -> None:
    tasks = sorted(rows[model], key=lambda t: -float(
        rows[model][t][delays[0]]["success_rate"]))
    significant = significant_cells(rows, model, delays)
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    width = 0.2
    positions = np.arange(len(tasks))
    for offset, (delay, color) in enumerate(zip(delays, DELAY_COLORS)):
        xs, ys, errs, highs = [], [], [], []
        for index, task in enumerate(tasks):
            row = rows[model][task].get(delay)
            if row is None:
                continue
            rate = as_float(row, "success_rate")
            low = as_float(row, "success_rate_ci95_low")
            high = as_float(row, "success_rate_ci95_high")
            xs.append(index - 1.5 * width + offset * width)
            ys.append(rate)
            errs.append([rate - low, high - rate])
            highs.append(high)
        ax.bar(xs, ys, width=width, color=color, label=DELAY_LABELS[delay],
               yerr=np.array(errs).T, capsize=2.5,
               error_kw=dict(elinewidth=0.9, ecolor="#444444"))
        for x, y, task, high in zip(xs, ys, tasks, highs):
            ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
            if (task, delay) in significant:
                ax.annotate("*", (x, y + (high - y) + 0.035), ha="center",
                            va="bottom", fontsize=14, fontweight="bold",
                            color="#111111")

    series = macro_series(rows, model, delays)
    macro_x = len(tasks)
    for offset, delay in enumerate(delays):
        if delay not in series:
            continue
        rate, low, high, _n, _s = series[delay]
        ax.bar([macro_x - 1.5 * width + offset * width], [rate], width=width,
               color=DELAY_COLORS[offset], alpha=0.85,
               yerr=[[rate - low], [high - rate]], capsize=2.5,
               error_kw=dict(elinewidth=0.9, ecolor="#444444"))
        ax.annotate(f"{rate:.0%}", (macro_x - 1.5 * width + offset * width, rate),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    ax.set_xticks(list(positions) + [macro_x])
    ax.set_xticklabels([SHORT_TASK.get(t, t) for t in tasks] + ["ALL TASKS\n(pooled)"],
                       fontsize=8.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("task success rate")
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    if significant:
        handles.append(Line2D([], [], color="none", marker="$*$", markersize=11,
                              label="Holm < 0.05 vs 0 ms"))
        labels.append("Holm < 0.05 vs 0 ms")
    ax.legend(handles, labels, title="injected delay", frameon=False, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.axvline(len(tasks) - 0.5, color="grey", linestyle=":", linewidth=1)
    base = series[delays[0]][0]
    delta = series[delays[-1]][0] - base
    ax.set_title(
        f"Figure {figure_index} - {MODEL_LABELS[model]}: success rate per task by "
        f"injected delay (0 ms -> 500 ms pooled: {delta:+.1%})\n"
        "30 episodes per bar, error bars = 95% Wilson CI, replan=5, rtc=off, "
        "pretrain20 scenes (bars grouped by delay, tasks sorted by 0 ms success)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("eval_results/simdelay_grid_analysis/cross_model_grid.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("eval_results/simdelay_grid_analysis/figures"))
    args = parser.parse_args()

    rows, _tasks = load(args.csv)
    delays = sorted({d for model in rows for task in rows[model] for d in rows[model][task]})
    if len(rows) != 4:
        raise SystemExit(f"expected 4 models, found {sorted(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure1(rows, delays, args.output_dir / "fig1_success_vs_delay_by_model.png")
    for index, model in enumerate(MODEL_ORDER, start=2):
        figure_per_model(
            rows, model, delays,
            args.output_dir / f"fig{index}_success_by_task_{model}.png",
            figure_index=index,
        )


if __name__ == "__main__":
    main()

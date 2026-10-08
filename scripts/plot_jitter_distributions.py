"""Plot reconstructed per-request delay and replan jitter distributions.

The benchmark stores per-episode moments rather than every sampled request.
Because jitter uses a deterministic SeedSequence, this script reconstructs the
exact request-level samples and validates them against the recorded moments.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from scripts.xiaomi_latency_experiment import sample_jitter
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from xiaomi_latency_experiment import sample_jitter


MODELS = ("xiaomi", "gr00t_n1_5")
MODEL_LABELS = {"xiaomi": "Xiaomi", "gr00t_n1_5": "GR00T N1.5"}
COLORS = {"xiaomi": "#2f6f9f", "gr00t_n1_5": "#d97732"}


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in (
            "episode",
            "seed",
            "delay_sample_count",
            "replan_sample_count",
        ):
            row[key] = int(row[key])
        for key in (
            "delay_jitter_variance_ms2",
            "replan_jitter_variance_steps2",
            "mean_sampled_delay_ms",
            "sampled_delay_variance_ms2",
            "mean_sampled_replan_steps",
            "sampled_replan_variance_steps2",
        ):
            row[key] = float(row[key])
        row["error"] = row.get("error") or None
    return rows


def reconstruct_row(row: dict[str, Any], jitter_seed: int) -> tuple[list[float], list[int]]:
    rng = np.random.default_rng(
        np.random.SeedSequence([jitter_seed, row["seed"], row["episode"]])
    )
    delay_values: list[float] = []
    replan_values: list[int] = [
        int(
            sample_jitter(
                10,
                row["replan_jitter_variance_steps2"],
                rng,
                distribution=row["replan_jitter_distribution"],
                lower_bound=1.0,
                integer=True,
            )
        )
    ]
    for _ in range(row["delay_sample_count"]):
        delay_values.append(
            float(
                sample_jitter(
                    100,
                    row["delay_jitter_variance_ms2"],
                    rng,
                    distribution=row["delay_jitter_distribution"],
                    lower_bound=0.0,
                    integer=False,
                )
            )
        )
        replan_values.append(
            int(
                sample_jitter(
                    10,
                    row["replan_jitter_variance_steps2"],
                    rng,
                    distribution=row["replan_jitter_distribution"],
                    lower_bound=1.0,
                    integer=True,
                )
            )
        )
    if len(replan_values) != row["replan_sample_count"]:
        raise ValueError(
            f"replan sample count mismatch for episode {row['episode']}: "
            f"reconstructed={len(replan_values)} recorded={row['replan_sample_count']}"
        )
    return delay_values, replan_values


def collect(root: Path, axis: str) -> dict[str, dict[float, list[float]]]:
    values: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for run_dir in sorted((root / axis / model).glob("var*")):
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            args = summary["args"]
            jitter_seed = int(args["jitter_seed"])
            rows = [row for row in read_rows(run_dir) if not row["error"]]
            if not rows:
                continue
            variance_key = (
                "delay_jitter_variance_ms2"
                if axis == "delay"
                else "replan_jitter_variance_steps2"
            )
            variance = float(rows[0][variance_key])
            for row in rows:
                delay_values, replan_values = reconstruct_row(row, jitter_seed)
                actual = delay_values if axis == "delay" else replan_values
                recorded_mean = row[
                    "mean_sampled_delay_ms" if axis == "delay" else "mean_sampled_replan_steps"
                ]
                recorded_var = row[
                    "sampled_delay_variance_ms2"
                    if axis == "delay"
                    else "sampled_replan_variance_steps2"
                ]
                if actual and not np.isclose(np.mean(actual), recorded_mean, rtol=0, atol=1e-7):
                    raise ValueError(f"mean validation failed: {run_dir}/episode{row['episode']}")
                if actual and not np.isclose(np.var(actual), recorded_var, rtol=0, atol=1e-5):
                    raise ValueError(f"variance validation failed: {run_dir}/episode{row['episode']}")
                values[model][variance].extend(actual)
    return values


def plot_axis(
    axis: str,
    values: dict[str, dict[float, list[float]]],
    output: Path,
) -> None:
    unit = "ms" if axis == "delay" else "control steps"
    title = "Actual delay jitter distribution" if axis == "delay" else "Actual replan jitter distribution"
    ylabel = "Sampled delay (ms)" if axis == "delay" else "Sampled replan interval (steps)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    all_samples = [sample for model_values in values.values() for group in model_values.values() for sample in group]
    lower = min(all_samples)
    upper = max(all_samples)
    span = max(1.0, upper - lower)
    for ax, model in zip(axes, MODELS):
        groups = values[model]
        variances = sorted(groups)
        data = [groups[variance] for variance in variances]
        positions = np.arange(1, len(data) + 1)
        rng = np.random.default_rng(20261008)
        for position, variance, samples in zip(positions, variances, data):
            # Horizontal jitter separates coincident samples while preserving
            # the true sampled value on the y-axis.
            x = position + rng.uniform(-0.22, 0.22, size=len(samples))
            ax.scatter(
                x,
                samples,
                s=10,
                alpha=0.16,
                color=COLORS[model],
                linewidths=0,
                rasterized=True,
            )
            mean = float(np.mean(samples))
            actual_var = float(np.var(samples))
            ax.scatter(
                [position],
                [mean],
                s=70,
                marker="D",
                color="#222222",
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                f"n={len(samples)}\nmean={mean:.1f}\nvar={actual_var:.0f}",
                (position, max(samples)),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{variance:g}" for variance in variances])
        ax.set_xlabel(f"Target variance ({unit}^2)")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(lower - 0.06 * span, upper + 0.16 * span)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(
        f"{title}\n100 episodes per variance; each dot is a reconstructed per-request sample; ◆ = mean"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_normalized_comparison(
    all_values: dict[str, dict[str, dict[float, list[float]]]], output: Path
) -> None:
    """Plot delay and replan jitter on dimensionless relative scales."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    all_normalized = [
        sample / (100.0 if axis == "delay" else 10.0)
        for axis in ("delay", "replan")
        for model_values in all_values[axis].values()
        for samples in model_values.values()
        for sample in samples
    ]
    lower = min(all_normalized)
    upper = max(all_normalized)
    span = max(1.0, upper - lower)
    rng = np.random.default_rng(20261008)

    for row, axis in enumerate(("delay", "replan")):
        nominal_mean = 100.0 if axis == "delay" else 10.0
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            groups = all_values[axis][model]
            variances = sorted(groups)
            positions = np.arange(1, len(variances) + 1)
            for position, variance in zip(positions, variances):
                samples = np.asarray(groups[variance], dtype=float) / nominal_mean
                x = position + rng.uniform(-0.22, 0.22, size=len(samples))
                ax.scatter(
                    x,
                    samples,
                    s=10,
                    alpha=0.16,
                    color=COLORS[model],
                    linewidths=0,
                    rasterized=True,
                )
                mean = float(np.mean(samples))
                actual_cv2 = float(np.var(samples) / mean**2)
                target_cv2 = variance / nominal_mean**2
                ax.scatter(
                    [position], [mean], s=70, marker="D", color="#222222",
                    edgecolor="white", linewidth=0.8, zorder=4,
                )
                ax.annotate(
                    f"CV²={target_cv2:.2g}\nactual={actual_cv2:.2g}",
                    (position, max(samples)), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#333333",
                )
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{variance / nominal_mean**2:.2g}" for variance in variances])
            ax.set_xlabel("Target normalized variance (CV²)")
            ax.set_title(f"{axis.capitalize()} / {MODEL_LABELS[model]}")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylim(lower - 0.06 * span, upper + 0.16 * span)
    axes[0, 0].set_ylabel("Sample / nominal mean")
    axes[1, 0].set_ylabel("Sample / nominal mean")
    fig.suptitle(
        "Normalized jitter distributions\n"
        "delay / 100 ms and replan / 10 steps; each dot is a reconstructed per-request sample; ◆ = mean"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("eval_results/jitter_selected_100ep_20261007"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/jitter_selected_100ep_20261007/analysis"),
    )
    args = parser.parse_args()
    all_values = {}
    for axis in ("delay", "replan"):
        values = collect(args.root, axis)
        all_values[axis] = values
        plot_axis(axis, values, args.output_dir / f"{axis}_jitter_scatter.png")
        sample_count = sum(len(samples) for model in values.values() for samples in model.values())
        print(f"wrote {axis} distribution ({sample_count} reconstructed samples)")
    plot_normalized_comparison(all_values, args.output_dir / "normalized_jitter_scatter.png")
    print("wrote normalized delay/replan comparison")


if __name__ == "__main__":
    main()

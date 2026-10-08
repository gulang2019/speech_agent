"""Summarize episode steps by success/failure for the Xiaomi replan sweep."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPLANS = (1, 6, 11, 16, 21, 26, 31, 36, 41, 46)
DELAYS = (0, 100)


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q))


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replan in REPLANS:
        path = root / f"replan{replan}" / "episodes.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                delay = int(row["delay_ms"])
                if delay not in DELAYS:
                    raise ValueError(f"unexpected delay in {path}: {delay}")
                if int(row["replan_steps"]) != replan:
                    raise ValueError(f"unexpected replan in {path}: {row['replan_steps']}")
                if row["error"]:
                    raise ValueError(f"episode error in {path}: {row['error']}")
                rows.append({
                    "task_name": row["task_name"],
                    "delay_ms": delay,
                    "delay_steps": int(row["delay_steps"]),
                    "replan_steps": replan,
                    "rtc": row["rtc"],
                    "episode": int(row["episode"]),
                    "seed": int(row["seed"]),
                    "layout_id": int(row["layout_id"]),
                    "style_id": int(row["style_id"]),
                    "success": row["success"].lower() == "true",
                    "steps": int(row["steps"]),
                    "inference_requests": int(row["inference_requests"]),
                    "fallback_steps": float(row["fallback_steps"]),
                    "wall_time_s": float(row["wall_time_s"]),
                })
    return rows


def make_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, bool], list[int]] = {}
    for row in rows:
        key = (int(row["delay_ms"]), int(row["replan_steps"]), bool(row["success"]))
        groups.setdefault(key, []).append(int(row["steps"]))

    output: list[dict[str, object]] = []
    for delay in DELAYS:
        for replan in REPLANS:
            for success in (True, False):
                values = groups.get((delay, replan, success), [])
                if not values:
                    continue
                output.append({
                    "delay_ms": delay,
                    "replan_steps": replan,
                    "outcome": "success" if success else "failure",
                    "num_episodes": len(values),
                    "mean_steps": statistics.mean(values),
                    "median_steps": statistics.median(values),
                    "std_steps": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min_steps": min(values),
                    "p25_steps": percentile(values, 25),
                    "p75_steps": percentile(values, 75),
                    "max_steps": max(values),
                })
    return output


def plot(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)
    colors = {0: "#1769aa", 100: "#c05a00"}
    labels = {0: "0 ms", 100: "100 ms"}
    positions = np.arange(len(REPLANS))

    for ax, success in zip(axes, (True, False)):
        for offset, delay in zip((-0.16, 0.16), DELAYS):
            values = [
                [int(row["steps"])
                 for row in rows
                 if int(row["delay_ms"]) == delay
                 and int(row["replan_steps"]) == replan
                 and bool(row["success"]) == success]
                for replan in REPLANS
            ]
            bp = ax.boxplot(
                values,
                positions=positions + offset,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                boxprops={"facecolor": colors[delay], "alpha": 0.55},
                whiskerprops={"color": colors[delay]},
                capprops={"color": colors[delay]},
            )
            for patch in bp["boxes"]:
                patch.set_edgecolor(colors[delay])
            ax.plot([], [], color=colors[delay], linewidth=7, alpha=0.55, label=labels[delay])
        ax.set_title("Success episodes" if success else "Failure episodes")
        ax.set_xticks(positions)
        ax.set_xticklabels(REPLANS)
        ax.set_xlabel("Replan steps")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 780)
    axes[0].set_ylabel("Episode steps")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Episode duration by outcome\n"
        "Xiaomi, PickPlaceCounterToCabinet, sim-domain delay, RTC off"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.root)
    summary = make_summary(rows)
    detail_path = args.output.with_name(args.output.stem + "_episodes.csv")
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        fields = list(rows[0])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, args.output)
    print(f"episodes={len(rows)} groups={len(summary)}")
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

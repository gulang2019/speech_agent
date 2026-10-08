"""Summarize and plot GR00T episode steps by success/failure outcome."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPLANS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
DELAYS = (0, 100, 300, 500)


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q))


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delay in DELAYS:
        for replan in REPLANS:
            path = root / f"delay{delay}ms" / f"replan{replan}" / "episodes.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if int(row["delay_ms"]) != delay or int(row["replan_steps"]) != replan:
                        raise ValueError(f"condition mismatch in {path}")
                    if row["delay_domain"] != "sim" or row["rtc"].lower() != "false":
                        raise ValueError(f"unexpected domain/RTC in {path}")
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


def plot(rows: list[dict[str, object]], output: Path, model_label: str, scene_label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = {"success": "#1769aa", "failure": "#b42318"}
    positions = np.arange(len(REPLANS))
    for ax, delay in zip(axes.flat, DELAYS):
        for outcome, offset in (("success", -0.18), ("failure", 0.18)):
            values = [
                [int(row["steps"])
                 for row in rows
                 if int(row["delay_ms"]) == delay
                 and int(row["replan_steps"]) == replan
                 and ("success" if bool(row["success"]) else "failure") == outcome]
                for replan in REPLANS
            ]
            # Empty groups are represented by a harmless placeholder and hidden.
            safe_values = [value if value else [np.nan] for value in values]
            bp = ax.boxplot(
                safe_values,
                positions=positions + offset,
                widths=0.30,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
                boxprops={"facecolor": colors[outcome], "alpha": 0.55},
                whiskerprops={"color": colors[outcome]},
                capprops={"color": colors[outcome]},
            )
            for patch in bp["boxes"]:
                patch.set_edgecolor(colors[outcome])
            ax.plot([], [], color=colors[outcome], linewidth=7, alpha=0.55, label=outcome)
        ax.set_title(f"Injected delay: {delay} ms ({delay // 50} sim steps)")
        ax.set_xticks(positions)
        ax.set_xticklabels(REPLANS)
        ax.set_xlabel("Replan steps")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 780)
        ax.legend(frameon=False, loc="upper left")
    axes[0, 0].set_ylabel("Episode steps")
    axes[1, 0].set_ylabel("Episode steps")
    fig.suptitle(
        f"{model_label}: episode duration by outcome\n"
        f"PickPlaceCounterToCabinet, {scene_label}, sim-domain delay, RTC off"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="GR00T")
    parser.add_argument("--scene-label", default="pretrain20 scenes")
    args = parser.parse_args()
    rows = load_rows(args.root)
    summary = make_summary(rows)
    detail_path = args.output.with_name(args.output.stem + "_episodes.csv")
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, args.output, args.model_label, args.scene_label)
    print(f"episodes={len(rows)} groups={len(summary)}")
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

"""Analyze a fixed-condition replan-step sweep."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def load_rows(path: Path):
    with (path / "episodes.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def analyze(path: Path, expected_delay: int, expected_task: str, expected_scene: tuple[int, int]):
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"no episode rows in {path}")
    groups = defaultdict(list)
    for row in rows:
        if row["task_name"] != expected_task:
            raise ValueError(f"unexpected task: {row['task_name']}")
        if int(row["delay_ms"]) != expected_delay or row["delay_domain"] != "sim":
            raise ValueError("the sweep must use one fixed sim-domain delay")
        if row["rtc"].lower() != "false":
            raise ValueError("the sweep must use rtc=off")
        scene = (int(row["layout_id"]), int(row["style_id"]))
        if scene != expected_scene:
            raise ValueError(f"unexpected scene: {scene}, expected {expected_scene}")
        if row["error"]:
            raise ValueError(f"episode error at replan={row['replan_steps']}: {row['error']}")
        groups[int(row["replan_steps"])].append(row)

    values = sorted(groups)
    if values != list(range(1, 51)):
        raise ValueError(f"expected replan steps 1..50, got {values}")

    output = []
    for value in values:
        condition_rows = groups[value]
        successes = sum(row["success"].lower() == "true" for row in condition_rows)
        total = len(condition_rows)
        low, high = wilson_interval(successes, total)
        output.append({
            "replan_steps": value,
            "num_episodes": total,
            "successes": successes,
            "success_rate": successes / total,
            "success_rate_ci95_low": low,
            "success_rate_ci95_high": high,
            "mean_steps": sum(int(row["steps"]) for row in condition_rows) / total,
            "mean_fallback_steps": sum(float(row["fallback_steps"]) for row in condition_rows) / total,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-ms", type=int, default=100)
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--layout-id", type=int, default=7)
    parser.add_argument("--style-id", type=int, default=10)
    args = parser.parse_args()

    rows = analyze(args.input, args.delay_ms, args.task, (args.layout_id, args.style_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} replan conditions to {args.output}")


if __name__ == "__main__":
    main()

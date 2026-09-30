"""Merge the archived Xiaomi replan=1..10 sweep with replayed extensions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    if not total:
        return 0.0, 0.0
    p = successes / total
    d = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / d
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / d
    return max(0.0, center - margin), min(1.0, center + margin)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(rows, replan: int):
    successes = sum(row["success"].lower() == "true" for row in rows)
    total = len(rows)
    low, high = wilson(successes, total)
    return {
        "replan_steps": replan,
        "num_episodes": total,
        "successes": successes,
        "success_rate": successes / total,
        "success_rate_ci95_low": low,
        "success_rate_ci95_high": high,
        "mean_steps": sum(float(row["steps"]) for row in rows) / total,
        "mean_fallback_steps": sum(float(row["fallback_steps"]) for row in rows) / total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_rows = read_csv(args.old / "episodes.csv")
    rows = []
    for replan in range(1, 11):
        rows.append(summarize(
            [row for row in old_rows if int(row["replan_steps"]) == replan and not row["error"]],
            replan,
        ))
    extension_dirs = sorted(
        (path for path in args.extension_root.glob("replan*") if (path / "episodes.csv").exists()),
        key=lambda path: int(path.name.removeprefix("replan")),
    )
    for directory in extension_dirs:
        replan = int(directory.name.removeprefix("replan"))
        extension_rows = [row for row in read_csv(directory / "episodes.csv") if not row["error"]]
        rows.append(summarize(extension_rows, replan))

    rows.sort(key=lambda row: row["replan_steps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} conditions to {args.output}")


if __name__ == "__main__":
    main()

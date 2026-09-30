"""Build replay input rows for extending the archived Xiaomi replan sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replans", default="12,15,20,25,30,35,40,45,50")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    with (args.source / "episodes.csv").open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    source_rows = [
        row for row in source_rows
        if row["task_name"] == "PickPlaceCounterToCabinet"
        and int(row["replan_steps"]) == 1
        and not row["error"]
    ][: args.episodes]
    if len(source_rows) != args.episodes:
        raise ValueError(f"source has {len(source_rows)} usable replan=1 rows, expected {args.episodes}")

    replans = [int(value) for value in args.replans.split(",") if value.strip()]
    for replan in replans:
        output = args.output_root / f"replan{replan}"
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for source in source_rows:
            row = dict(source)
            row["condition"] = f"delay0ms_wall_replan{replan}_rtcoff"
            row["replan_steps"] = str(replan)
            row["delay_domain"] = "wall"
            row["delay_steps"] = "0"
            rows.append(row)
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(replans)} replay inputs with {len(source_rows)} rows each")


if __name__ == "__main__":
    main()

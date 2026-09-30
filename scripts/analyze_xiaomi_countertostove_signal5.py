#!/usr/bin/env python3
"""Summarize paired 0 ms versus 100 ms CounterToStove signal-scene runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    probability = successes / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_p_value(only_zero_success: int, only_hundred_success: int) -> float:
    discordant = only_zero_success + only_hundred_success
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(only_zero_success, only_hundred_success) + 1)
    ) / 2**discordant
    return min(1.0, 2 * tail)


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "episodes.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["delay_ms"] in {"0", "100"}]
    if not rows:
        raise ValueError(f"no 0/100 ms rows in {path}")
    if any(row.get("error") for row in rows):
        raise ValueError(f"run contains failed episodes: {path}")
    return rows


def scene_name(row: dict[str, str]) -> str:
    return f"L{row['layout_id']}S{row['style_id']}"


def summarize(rows: list[dict[str, str]], label: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_scene_delay: dict[tuple[str, int], list[bool]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[int, bool]] = defaultdict(dict)
    for row in rows:
        scene = scene_name(row)
        delay = int(row["delay_ms"])
        success = row["success"].lower() == "true"
        by_scene_delay[(scene, delay)].append(success)
        pair = paired[(scene, int(row["seed"]))]
        if delay in pair:
            raise ValueError(f"duplicate {delay} ms result for {scene}, seed={row['seed']}")
        pair[delay] = success

    rates: list[dict[str, object]] = []
    for (scene, delay), outcomes in sorted(by_scene_delay.items()):
        successes = sum(outcomes)
        low, high = wilson_interval(successes, len(outcomes))
        rates.append({
            "replicate": label,
            "scene": scene,
            "delay_ms": delay,
            "episodes": len(outcomes),
            "successes": successes,
            "success_rate": successes / len(outcomes),
            "ci95_low": low,
            "ci95_high": high,
        })

    effects_by_scene: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for (scene, seed), outcomes in paired.items():
        if set(outcomes) != {0, 100}:
            raise ValueError(f"missing paired delay for {scene}, seed={seed}")
        effects_by_scene[scene].append((outcomes[0], outcomes[100]))

    effects: list[dict[str, object]] = []
    for scene, outcomes in sorted(effects_by_scene.items()):
        only_zero_success = sum(zero and not hundred for zero, hundred in outcomes)
        only_hundred_success = sum(hundred and not zero for zero, hundred in outcomes)
        effects.append({
            "replicate": label,
            "scene": scene,
            "paired_episodes": len(outcomes),
            "both_success": sum(zero and hundred for zero, hundred in outcomes),
            "both_failure": sum(not zero and not hundred for zero, hundred in outcomes),
            "only_0ms_success": only_zero_success,
            "only_100ms_success": only_hundred_success,
            "rate_difference_100_minus_0": (only_hundred_success - only_zero_success) / len(outcomes),
            "mcnemar_exact_p": exact_p_value(only_zero_success, only_hundred_success),
        })
    return rates, effects


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replicate_rows: list[tuple[str, list[dict[str, str]]]] = [
        (run_dir.name, read_rows(run_dir)) for run_dir in args.run
    ]
    all_rows = [row for _, rows in replicate_rows for row in rows]
    rate_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for label, rows in replicate_rows:
        rates, effects = summarize(rows, label)
        rate_rows.extend(rates)
        effect_rows.extend(effects)
    combined_rates, combined_effects = summarize(all_rows, "combined")
    rate_rows.extend(combined_rates)
    effect_rows.extend(combined_effects)
    write_csv(args.output_dir / "scene_success_rates.csv", rate_rows)
    write_csv(args.output_dir / "paired_effects.csv", effect_rows)

    combined_rate_map = {(row["scene"], row["delay_ms"]): row for row in combined_rates}
    lines = [
        "# Xiaomi CounterToStove Five-Scene Latency Replication",
        "",
        "The rows below combine independent Slurm replicates. Each scene uses matched seeds for 0 ms and 100 ms.",
        "",
        "| Scene | 0 ms | 100 ms | Difference | 0-only | 100-only | Exact paired p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for effect in combined_effects:
        scene = effect["scene"]
        zero = combined_rate_map[(scene, 0)]
        hundred = combined_rate_map[(scene, 100)]
        lines.append(
            f"| {scene} | {zero['successes']}/{zero['episodes']} ({zero['success_rate']:.1%}) "
            f"| {hundred['successes']}/{hundred['episodes']} ({hundred['success_rate']:.1%}) "
            f"| {effect['rate_difference_100_minus_0']:+.1%} "
            f"| {effect['only_0ms_success']} | {effect['only_100ms_success']} "
            f"| {effect['mcnemar_exact_p']:.4f} |"
        )
    lines += [
        "",
        "`0-only` / `100-only` count discordant matched seeds. The exact paired p-value is a two-sided McNemar/binomial test.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Analyze the three Xiaomi one-factor-at-a-time latency experiments."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.xiaomi_latency_experiment import wilson_interval
except ModuleNotFoundError:
    from xiaomi_latency_experiment import wilson_interval


SWEEPS = {
    "delay": {
        "field": "delay_ms",
        "values": list(range(0, 601, 100)),
        "baseline": 0,
        "fixed": {"replan_steps": 5, "rtc": False},
    },
    "replan": {
        "field": "replan_steps",
        "values": list(range(1, 11)),
        "baseline": 5,
        "fixed": {"delay_ms": 0, "rtc": False},
    },
    "rtc": {
        "field": "rtc",
        "values": [False, True],
        "baseline": False,
        "fixed": {"delay_ms": 0, "replan_steps": 5},
    },
}


def parse_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with (path / "episodes.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ("delay_ms", "replan_steps", "episode", "seed", "steps"):
            row[key] = int(row[key])
        row["rtc"] = parse_bool(row["rtc"])
        row["success"] = parse_bool(row["success"])
        row["error"] = row["error"] or None
        row["layout_id"] = int(row["layout_id"])
        row["style_id"] = int(row["style_id"])
    return rows


def paired_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task_name"],
        row["episode"],
        row["seed"],
        row["layout_id"],
        row["style_id"],
    )


def exact_mcnemar_p(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, int, float]:
    losses = int(np.sum((reference == 1) & (candidate == 0)))
    gains = int(np.sum((reference == 0) & (candidate == 1)))
    discordant = losses + gains
    if discordant == 0:
        return gains, losses, 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return gains, losses, p_value


def paired_bootstrap_interval(
    reference: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    differences = candidate.astype(float) - reference.astype(float)
    if not np.any(differences):
        return 0.0, 0.0
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    tested = [row for row in rows if not row["is_baseline"]]
    previous = 0.0
    for rank, row in enumerate(sorted(tested, key=lambda item: item["mcnemar_p_value"])):
        adjusted = min(1.0, row["mcnemar_p_value"] * (len(tested) - rank))
        previous = max(previous, adjusted)
        row["holm_adjusted_p_value"] = previous
    for row in rows:
        if row["is_baseline"]:
            row["holm_adjusted_p_value"] = 1.0


def analyze_sweep(
    name: str,
    directory: Path,
    episodes_per_condition: int,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    config = SWEEPS[name]
    rows = load_episodes(directory)
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for field, expected in config["fixed"].items():
            if row[field] != expected:
                raise ValueError(f"{name}: expected {field}={expected}, got {row[field]}")
        grouped[row[config["field"]]].append(row)

    if set(grouped) != set(config["values"]):
        raise ValueError(f"{name}: values {sorted(grouped)} do not match {config['values']}")
    for value, condition_rows in grouped.items():
        if len(condition_rows) != episodes_per_condition:
            raise ValueError(
                f"{name}={value}: expected {episodes_per_condition} episodes, got {len(condition_rows)}"
            )
        if any(row["error"] for row in condition_rows):
            raise ValueError(f"{name}={value}: one or more episodes contain errors")

    keyed = {
        value: {paired_key(row): int(row["success"]) for row in condition_rows}
        for value, condition_rows in grouped.items()
    }
    baseline_keys = set(keyed[config["baseline"]])
    if any(set(condition) != baseline_keys for condition in keyed.values()):
        raise ValueError(f"{name}: conditions do not contain identical paired episode keys")
    ordered_keys = sorted(baseline_keys)
    reference = np.array(
        [keyed[config["baseline"]][key] for key in ordered_keys], dtype=np.int8
    )

    analysis = []
    for value in config["values"]:
        candidate = np.array([keyed[value][key] for key in ordered_keys], dtype=np.int8)
        successes = int(candidate.sum())
        rate = successes / len(candidate)
        ci_low, ci_high = wilson_interval(successes, len(candidate))
        gains, losses, p_value = exact_mcnemar_p(reference, candidate)
        difference_low, difference_high = paired_bootstrap_interval(
            reference, candidate, rng, bootstrap_samples
        )
        analysis.append(
            {
                "sweep": name,
                "value": str(value).lower() if isinstance(value, bool) else value,
                "num_episodes": len(candidate),
                "successes": successes,
                "success_rate": rate,
                "success_rate_ci95_low": ci_low,
                "success_rate_ci95_high": ci_high,
                "baseline_value": str(config["baseline"]).lower()
                if isinstance(config["baseline"], bool)
                else config["baseline"],
                "is_baseline": value == config["baseline"],
                "success_rate_difference": rate - float(reference.mean()),
                "difference_ci95_low": difference_low,
                "difference_ci95_high": difference_high,
                "paired_gains": gains,
                "paired_losses": losses,
                "mcnemar_p_value": p_value,
            }
        )
    holm_adjust(analysis)
    return analysis


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ofat_analysis.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Xiaomi OFAT experiment analysis",
        "",
        "Each condition uses paired episode seeds. Success-rate intervals are 95% Wilson intervals; difference intervals use paired bootstrap resampling. P-values use the exact McNemar test with Holm correction within each sweep.",
        "",
    ]
    for sweep in SWEEPS:
        lines.extend(
            [
                f"## {sweep}",
                "",
                "| Value | Success | Rate (95% CI) | Difference vs baseline (95% CI) | Holm p |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in (item for item in rows if item["sweep"] == sweep):
            lines.append(
                f"| {row['value']} | {row['successes']}/{row['num_episodes']} | "
                f"{row['success_rate']:.1%} ({row['success_rate_ci95_low']:.1%}, {row['success_rate_ci95_high']:.1%}) | "
                f"{row['success_rate_difference']:+.1%} ({row['difference_ci95_low']:+.1%}, {row['difference_ci95_high']:+.1%}) | "
                f"{row['holm_adjusted_p_value']:.4g} |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-dir", type=Path, required=True)
    parser.add_argument("--replan-dir", type=Path, required=True)
    parser.add_argument("--rtc-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(20260917)
    rows = []
    for name, directory in (
        ("delay", args.delay_dir),
        ("replan", args.replan_dir),
        ("rtc", args.rtc_dir),
    ):
        rows.extend(
            analyze_sweep(
                name,
                directory,
                args.episodes,
                rng,
                args.bootstrap_samples,
            )
        )
    write_report(args.output_dir, rows)
    print(f"Wrote {len(rows)} conditions to {args.output_dir}")


if __name__ == "__main__":
    main()

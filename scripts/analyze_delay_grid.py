"""Analyze the multi-task 0-100ms delay sweep (fixed replan=5, rtc=off).

Input layout: ``<root>/<task>/part<N>/episodes.csv``.  All parts of a task are
merged, then each delay is compared against that task's delay=0 baseline using
paired episode keys.  Intervals are 95% Wilson intervals, difference intervals
use paired bootstrap resampling, and p-values use the exact McNemar test with
Holm correction inside each task sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_xiaomi_ofat import (
        exact_mcnemar_p,
        holm_adjust,
        paired_bootstrap_interval,
    )
    from scripts.xiaomi_latency_experiment import wilson_interval
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from analyze_xiaomi_ofat import exact_mcnemar_p, holm_adjust, paired_bootstrap_interval
    from xiaomi_latency_experiment import wilson_interval


EXPECTED_DELAYS = list(range(0, 101, 10))
FIXED = {"replan_steps": 5, "rtc": False}
TASK_ORDER = [
    "PickPlaceCounterToCabinet",
    "OpenCabinet",
    "OpenDoubleDoor",
    "TurnOffMicrowave",
    "CleanMicrowave",
    "PrepareCoffee",
]
TASK_LABELS = {
    "PickPlaceCounterToCabinet": "counter -> cabinet (reference)",
    "OpenCabinet": "single-step door",
    "OpenDoubleDoor": "two-door atomic task",
    "TurnOffMicrowave": "press-stop-button atomic task",
    "CleanMicrowave": "simple multi-step (0% baseline)",
    "PrepareCoffee": "complex (0% baseline)",
}


def load_task_rows(task_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parts = sorted(task_dir.glob("part*/episodes.csv"))
    if not parts:
        flat = task_dir / "episodes.csv"
        if flat.exists():
            parts = [flat]
        else:
            raise FileNotFoundError(f"no episodes.csv under {task_dir}")
    for part in parts:
        with part.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                for key in ("delay_ms", "replan_steps", "episode", "seed", "steps"):
                    row[key] = int(row[key])
                row["rtc"] = row["rtc"].lower() == "true"
                row["success"] = row["success"].lower() == "true"
                row["error"] = row["error"] or None
                rows.append(row)
    return rows


def paired_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["episode"], row["seed"], row["layout_id"], row["style_id"])


def analyze_task(
    task: str,
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    bootstrap_samples: int,
    strict: bool = True,
    expected_delays: list[int] | None = None,
) -> list[dict[str, Any]]:
    expected_delays = list(EXPECTED_DELAYS if expected_delays is None else expected_delays)
    for row in rows:
        if row["task_name"] != task:
            raise ValueError(f"{task}: unexpected task_name {row['task_name']}")
        for field, expected in FIXED.items():
            if row[field] != expected:
                raise ValueError(f"{task}: expected {field}={expected}, got {row[field]}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["delay_ms"]].append(row)

    if strict and set(grouped) != set(expected_delays):
        raise ValueError(
            f"{task}: delays {sorted(grouped)} do not match {expected_delays}"
        )

    counts = {len(group) for group in grouped.values()}
    if strict and len(counts) != 1:
        raise ValueError(f"{task}: uneven episode counts across delays: {counts}")
    for delay, condition_rows in grouped.items():
        bad = [row for row in condition_rows if row["error"]]
        if bad:
            raise ValueError(
                f"{task}: delay={delay} has {len(bad)} errored episodes, e.g. {bad[0]['error']}"
            )
        # With ``--delay-domain sim`` the reported arrival age must equal the
        # injected step count, otherwise the sweep is not measuring what it claims.
        # Under the legacy ``wall`` domain the age also contains model latency, so
        # the check only applies when the run declares the sim domain.
        domains = {row.get("delay_domain", "sim") for row in condition_rows}
        if (
            delay
            and domains == {"sim"}
            and "mean_arrival_age_steps" in condition_rows[0]
        ):
            declared = {float(row["delay_steps"]) for row in condition_rows}
            observed = {float(row["mean_arrival_age_steps"]) for row in condition_rows}
            if len(declared) != 1 or max(observed) != declared.pop():
                raise ValueError(
                    f"{task}: delay={delay} has arrival age {sorted(observed)} "
                    f"inconsistent with the injected step count"
                )

    keyed = {
        delay: {paired_key(row): int(row["success"]) for row in condition_rows}
        for delay, condition_rows in grouped.items()
    }
    baseline_keys = set(keyed[0])
    if strict:
        for delay, mapping in keyed.items():
            if set(mapping) != baseline_keys:
                raise ValueError(f"{task}: delay={delay} does not share baseline episode keys")
    else:
        # Pair only on episodes present for every delay so partial runs still analyze.
        for mapping in keyed.values():
            baseline_keys &= set(mapping)
        print(f"  {task}: partial run, pairing on {len(baseline_keys)} shared episodes")

    ordered = sorted(baseline_keys)
    reference = np.array([keyed[0][key] for key in ordered], dtype=np.int8)
    delays = expected_delays if strict else sorted(grouped)
    results = []
    for delay in delays:
        candidate = np.array([keyed[delay][key] for key in ordered], dtype=np.int8)
        successes = int(candidate.sum())
        total = len(candidate)
        ci_low, ci_high = wilson_interval(successes, total)
        gains, losses, p_value = exact_mcnemar_p(reference, candidate)
        diff_low, diff_high = paired_bootstrap_interval(
            reference, candidate, rng, bootstrap_samples
        )
        results.append(
            {
                "task_name": task,
                "delay_ms": delay,
                "num_episodes": total,
                "successes": successes,
                "success_rate": successes / total,
                "success_rate_ci95_low": ci_low,
                "success_rate_ci95_high": ci_high,
                "baseline_delay_ms": 0,
                "is_baseline": delay == 0,
                "success_rate_difference": successes / total - float(reference.mean()),
                "difference_ci95_low": diff_low,
                "difference_ci95_high": diff_high,
                "paired_gains": gains,
                "paired_losses": losses,
                "mcnemar_p_value": p_value,
            }
        )
    holm_adjust(results)
    return results


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    delay_domain: str = "sim",
    task_order: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "delay_grid_analysis.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "delay_grid_analysis.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    steps = sorted({row["delay_ms"] for row in rows})
    domain_note = (
        "Injected delay is realized as a fixed number of control steps "
        "(delay domain `sim`), so staleness is identical across models and does "
        "not depend on native inference latency."
        if delay_domain == "sim"
        else "Legacy `wall` semantics: the worker sleeps for the configured delay "
        "on top of the real model latency, so total delay = model latency + delay."
    )
    lines = [
        "# Delay sweep " + ", ".join(f"{value}ms" for value in steps) + " (fixed replan=5, rtc=off)",
        "",
        domain_note + " Paired episode seeds within each task; Wilson intervals, "
        "paired bootstrap differences, exact McNemar with Holm correction inside each task.",
        "",
    ]
    for task in (task_order or TASK_ORDER):
        task_rows = [row for row in rows if row["task_name"] == task]
        if not task_rows:
            continue
        baseline = next(row for row in task_rows if row["is_baseline"])
        lines.extend(
            [
                f"## {task} ({TASK_LABELS.get(task, '')})",
                "",
                f"Baseline delay=0: {baseline['successes']}/{baseline['num_episodes']} "
                f"({baseline['success_rate']:.1%})",
                "",
                "| Injected delay (ms) | Success | Rate (95% CI) | Difference vs 0ms (95% CI) | Holm p |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in task_rows:
            lines.append(
                f"| {row['delay_ms']} | {row['successes']}/{row['num_episodes']} | "
                f"{row['success_rate']:.1%} "
                f"({row['success_rate_ci95_low']:.1%}, {row['success_rate_ci95_high']:.1%}) | "
                f"{row['success_rate_difference']:+.1%} "
                f"({row['difference_ci95_low']:+.1%}, {row['difference_ci95_high']:+.1%}) | "
                f"{row['holm_adjusted_p_value']:.4g} |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--delays",
        default=None,
        help="Comma-separated expected delays; defaults to 0..100 in 10ms steps.",
    )
    parser.add_argument("--delay-domain", choices=("sim", "wall"), default="sim")
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    expected_delays = (
        [int(value) for value in args.delays.split(",")] if args.delays else list(EXPECTED_DELAYS)
    )
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        task = task_dir.name
        if args.tasks and task not in args.tasks:
            continue
        print(f"analyzing {task}")
        try:
            all_rows.extend(
                analyze_task(
                    task,
                    load_task_rows(task_dir),
                    rng,
                    args.bootstrap_samples,
                    strict=not args.allow_partial,
                    expected_delays=expected_delays,
                )
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"  skipped: {error}")
    write_report(
        args.output_dir,
        all_rows,
        delay_domain=args.delay_domain,
        task_order=args.tasks,
    )
    print(f"wrote {args.output_dir}/report.md")


if __name__ == "__main__":
    main()

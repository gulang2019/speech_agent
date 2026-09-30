"""Cross-model, cross-task analysis of the sim-domain delay grid.

Input layout: ``<root>/<model>/<task>/episodes.jsonl`` (one directory per
(model, task) slice, as produced by ``slurm/submit_sim_delay_grid.sh``).
The flat ``<root>/<model>_<task>/episodes.jsonl`` form is also accepted; the
model is taken as the longest known-model prefix of the directory name.

Every task/model slice is paired on ``(episode, seed, layout_id, style_id)``
against its own ``delay=0`` baseline, so a delay condition is compared with the
exact same scenes and seeds.  Outputs:

* per (model, task, delay) success rate with a 95% Wilson interval,
* the paired success-rate difference against delay=0 with a paired bootstrap
  interval and an exact McNemar p-value (Holm-corrected within each model),
* a task-averaged macro summary per (model, delay), pairing is done per task
  first so wide tasks do not dominate.

The sim-domain arrival-age check mirrors ``analyze_delay_grid.py``: if a run
declares ``delay_domain=sim`` the observed ``mean_arrival_age_steps`` must equal
the declared ``delay_steps``, otherwise the sweep is not measuring what it says.
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
        paired_key,
    )
    from scripts.xiaomi_latency_experiment import wilson_interval
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from analyze_xiaomi_ofat import (
        exact_mcnemar_p,
        holm_adjust,
        paired_bootstrap_interval,
        paired_key,
    )
    from xiaomi_latency_experiment import wilson_interval


FIXED = {"replan_steps": 5, "rtc": False}

KNOWN_MODELS = ("diffusion_policy", "gr00t_n1_5", "pi0_5", "pi0", "xiaomi")


def model_task_from_flat_dir(name: str) -> tuple[str, str] | None:
    """Split a ``<model>_<task>`` directory name, longest model prefix wins."""
    for model in sorted(KNOWN_MODELS, key=len, reverse=True):
        if name == model:
            continue
        if name.startswith(model + "_"):
            return model, name[len(model) + 1 :]
    return None


def load_slice(slice_dir: Path) -> list[dict[str, Any]]:
    flat = slice_dir / "episodes.jsonl"
    if not flat.exists():
        return []
    rows = [json.loads(line) for line in flat.read_text().splitlines() if line.strip()]
    return rows


def validate_slice(model: str, task: str, rows: list[dict[str, Any]], strict: bool) -> None:
    for row in rows:
        for field, expected in FIXED.items():
            if row[field] != expected:
                raise ValueError(f"{model}/{task}: expected {field}={expected}, got {row[field]}")
        if row.get("error"):
            raise ValueError(f"{model}/{task}: errored episode {row.get('error')}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["delay_ms"]].append(row)
    if strict:
        counts = {len(group) for group in grouped.values()}
        if len(counts) != 1:
            raise ValueError(f"{model}/{task}: uneven episode counts across delays: {counts}")

    for delay, condition_rows in grouped.items():
        if delay == 0:
            continue
        if {row.get("delay_domain", "sim") for row in condition_rows} != {"sim"}:
            continue
        declared = {float(row["delay_steps"]) for row in condition_rows}
        observed = {float(row["mean_arrival_age_steps"]) for row in condition_rows}
        if len(declared) != 1 or max(observed) != declared.pop():
            raise ValueError(
                f"{model}/{task}: delay={delay} arrival age {sorted(observed)} "
                f"inconsistent with declared injected steps"
            )


def analyze_slice(
    model: str,
    task: str,
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["delay_ms"]].append(row)
    if 0 not in grouped:
        raise ValueError(f"{model}/{task}: missing delay=0 baseline")

    keyed = {
        delay: {paired_key(row): int(row["success"]) for row in condition_rows}
        for delay, condition_rows in grouped.items()
    }
    shared = set(keyed[0])
    for mapping in keyed.values():
        shared &= set(mapping)
    keys = sorted(shared)
    baseline = np.array([keyed[0][key] for key in keys], dtype=np.int8)

    results = []
    for delay in sorted(keyed):
        candidate = np.array([keyed[delay][key] for key in keys], dtype=np.int8)
        n = len(candidate)
        low, high = wilson_interval(int(candidate.sum()), n)
        if delay == 0:
            gains = losses = 0
            p_value = 1.0
            diff_low = diff_high = 0.0
        else:
            gains, losses, p_value = exact_mcnemar_p(baseline, candidate)
            diff_low, diff_high = paired_bootstrap_interval(
                baseline, candidate, rng, bootstrap_samples
            )
        results.append(
            {
                "model": model,
                "task": task,
                "delay_ms": delay,
                "delay_steps": int(grouped[delay][0]["delay_steps"]),
                "num_episodes": n,
                "successes": int(candidate.sum()),
                "success_rate": float(candidate.mean()) if n else 0.0,
                "success_rate_ci95_low": low,
                "success_rate_ci95_high": high,
                "baseline_successes": int(baseline.sum()),
                "baseline_success_rate": float(baseline.mean()) if n else 0.0,
                "success_rate_delta": float((candidate - baseline).mean()) if n else 0.0,
                "delta_ci95_low": diff_low,
                "delta_ci95_high": diff_high,
                "gains": gains,
                "losses": losses,
                "is_baseline": delay == 0,
                "mcnemar_p_value": p_value,
                "mean_arrival_age_steps": float(
                    np.mean([row["mean_arrival_age_steps"] for row in grouped[delay]])
                ),
                "mean_fallback_steps": float(
                    np.mean([row["fallback_steps"] for row in grouped[delay]])
                ),
                "mean_model_latency_ms": float(
                    np.mean([row["mean_model_latency_ms"] for row in grouped[delay]])
                ),
            }
        )
    holm_adjust(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval_results/simdelay_grid"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    slices: list[tuple[str, str, Path]] = []
    for entry in sorted(path for path in args.root.iterdir() if path.is_dir()):
        if (entry / "episodes.jsonl").exists():
            flat = model_task_from_flat_dir(entry.name)
            if flat is None:
                print(f"  skip (unrecognized flat dir): {entry.name}")
                continue
            slices.append((flat[0], flat[1], entry))
        else:
            for task_dir in sorted(path for path in entry.iterdir() if path.is_dir()):
                slices.append((entry.name, task_dir.name, task_dir))

    for model, task, task_dir in slices:
        if args.models and model not in args.models:
            continue
        if args.tasks and task not in args.tasks:
            continue
        rows = load_slice(task_dir)
        if not rows:
            print(f"  skip (empty): {model}/{task}")
            continue
        delays = {row["delay_ms"] for row in rows}
        if len(delays) < 2 and not args.allow_partial:
            print(f"  skip (only delays {sorted(delays)}): {model}/{task}")
            continue
        validate_slice(model, task, rows, strict=not args.allow_partial)
        print(f"  analyzed {model}/{task}: {len(rows)} episodes")
        all_rows.extend(analyze_slice(model, task, rows, rng, args.bootstrap_samples))

    if not all_rows:
        raise SystemExit("no analyzable slices found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cross_model_grid.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    # Macro average: mean over tasks of each task's paired delta, so a task with
    # many easy episodes cannot dominate the model-level number.
    per_model_delay: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        per_model_delay[(row["model"], row["delay_ms"])].append(row)

    macro = []
    for (model, delay), rows in sorted(per_model_delay.items()):
        rates = [row["success_rate"] for row in rows]
        deltas = [row["success_rate_delta"] for row in rows]
        pool_episodes = sum(row["num_episodes"] for row in rows)
        pool_successes = sum(row["successes"] for row in rows)
        low, high = wilson_interval(pool_successes, pool_episodes)
        macro.append(
            {
                "model": model,
                "delay_ms": delay,
                "num_tasks": len(rows),
                "num_episodes": pool_episodes,
                "success_rate_macro": float(np.mean(rates)),
                "success_rate_pooled": pool_successes / pool_episodes if pool_episodes else 0.0,
                "success_rate_pooled_ci95_low": low,
                "success_rate_pooled_ci95_high": high,
                "mean_task_delta": float(np.mean(deltas)),
                "min_task_delta": float(np.min(deltas)),
                "max_task_delta": float(np.max(deltas)),
                "tasks_worse": int(sum(1 for delta in deltas if delta < 0)),
                "tasks_better": int(sum(1 for delta in deltas if delta > 0)),
                "tasks_tied": int(sum(1 for delta in deltas if delta == 0)),
            }
        )
    with (args.output_dir / "macro_by_model.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)

    lines = ["# Cross-model sim-domain delay grid", ""]
    lines.append("| model | delay | tasks | pooled success | 95% Wilson | mean task delta | tasks worse/tied/better |")
    lines.append("|---|---:|---:|---:|---|---:|---|")
    for row in macro:
        lines.append(
            f"| {row['model']} | {row['delay_ms']}ms | {row['num_tasks']} | "
            f"{row['success_rate_pooled']:.3f} | "
            f"[{row['success_rate_pooled_ci95_low']:.3f}, {row['success_rate_pooled_ci95_high']:.3f}] | "
            f"{row['mean_task_delta']:+.3f} | "
            f"{row['tasks_worse']}/{row['tasks_tied']}/{row['tasks_better']} |"
        )
    strict_rows = [row for row in all_rows if not row["is_baseline"]]
    significant = [
        row for row in strict_rows if row["holm_adjusted_p_value"] < 0.05 and row["gains"] + row["losses"] > 0
    ]
    lines += ["", f"Significant (Holm<0.05) model/task/delay cells: {len(significant)}"]
    for row in sorted(significant, key=lambda item: item["holm_adjusted_p_value"])[:20]:
        lines.append(
            f"- {row['model']}/{row['task']} {row['delay_ms']}ms: "
            f"{row['baseline_successes']}/{row['num_episodes']} -> "
            f"{row['successes']}/{row['num_episodes']} "
            f"(delta {row['success_rate_delta']:+.3f}, Holm p={row['holm_adjusted_p_value']:.4f})"
        )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_dir}/cross_model_grid.csv, macro_by_model.csv, SUMMARY.md")


if __name__ == "__main__":
    main()

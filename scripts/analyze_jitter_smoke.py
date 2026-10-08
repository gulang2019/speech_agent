"""Summarize the two-factor-model jitter smoke experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("xiaomi", "gr00t_n1_5")
AXES = ("delay", "replan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for axis in AXES:
        for model in MODELS:
            model_dir = args.root / axis / model
            for summary_path in sorted(model_dir.glob("var*/summary.json")):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                for condition in summary.get("conditions", []):
                    nominal_mean = 100.0 if axis == "delay" else 10.0
                    actual_mean = (
                        condition["mean_sampled_delay_ms"]
                        if axis == "delay"
                        else condition["mean_sampled_replan_steps"]
                    )
                    actual_variance = (
                        condition["mean_sampled_delay_variance_ms2"]
                        if axis == "delay"
                        else condition["mean_sampled_replan_variance_steps2"]
                    )
                    variance = (
                        condition["delay_jitter_variance_ms2"]
                        if axis == "delay"
                        else condition["replan_jitter_variance_steps2"]
                    )
                    row = {
                        "axis": axis,
                        "model": model,
                        "variance": variance,
                        "target_normalized_variance_cv2": variance / nominal_mean**2,
                        "actual_normalized_variance_cv2": actual_variance / actual_mean**2,
                        "distribution": (
                            condition["delay_jitter_distribution"]
                            if axis == "delay"
                            else condition["replan_jitter_distribution"]
                        ),
                        "num_episodes": condition["num_episodes"],
                        "successes": condition["successes"],
                        "success_rate": condition["success_rate"],
                        "success_rate_ci95_low": condition["success_rate_ci95_low"],
                        "success_rate_ci95_high": condition["success_rate_ci95_high"],
                        "mean_sampled_delay_ms": condition["mean_sampled_delay_ms"],
                        "mean_sampled_delay_variance_ms2": condition["mean_sampled_delay_variance_ms2"],
                        "mean_sampled_replan_steps": condition["mean_sampled_replan_steps"],
                        "mean_sampled_replan_variance_steps2": condition["mean_sampled_replan_variance_steps2"],
                        "errors": condition["errors"],
                    }
                    rows.append(row)

    if not rows:
        raise SystemExit(f"no summary.json files found under {args.root}")
    rows.sort(key=lambda row: (str(row["axis"]), str(row["model"]), float(row["variance"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "jitter_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Jitter smoke results",
        "",
        "Each row uses the configured paired episodes per variance condition. Normalized variance is Var / mean^2 (CV^2), using each axis's nominal mean (100 ms delay; 10 replan steps).",
        "",
    ]
    for axis in AXES:
        lines.extend([f"## {axis} jitter", "", "| Model | Variance | Normalized var (CV^2) | Success | Rate (95% CI) | Errors |", "|---|---:|---:|---:|---:|---:|"])
        for row in rows:
            if row["axis"] != axis:
                continue
            lines.append(
                f"| {row['model']} | {row['variance']} | {float(row['target_normalized_variance_cv2']):.3g} | {row['successes']}/{row['num_episodes']} | "
                f"{float(row['success_rate']):.1%} ({float(row['success_rate_ci95_low']):.1%}, {float(row['success_rate_ci95_high']):.1%}) | {row['errors']} |"
            )
        lines.append("")
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} condition rows to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Plot success rates with 95% Wilson intervals for the Xiaomi OFAT sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWEEP_ORDER = ("delay", "replan", "rtc")
X_LABELS = {
    "delay": "Injected delay (ms)",
    "replan": "Replan steps",
    "rtc": "RTC enabled",
}
TITLES = {
    "delay": "Delay sweep (replan=5, rtc=off)",
    "replan": "Replan sweep (delay=0, rtc=off)",
    "rtc": "RTC sweep (delay=0, replan=5)",
}


def parse_value(sweep: str, raw: str) -> Any:
    if sweep == "rtc":
        return raw.lower() == "true"
    return int(raw)


def load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    sweeps: dict[str, list[dict[str, Any]]] = {name: [] for name in SWEEP_ORDER}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            sweep = row["sweep"]
            if sweep not in sweeps:
                continue
            rate = float(row["success_rate"])
            low = float(row["success_rate_ci95_low"])
            high = float(row["success_rate_ci95_high"])
            sweeps[sweep].append(
                {
                    "value": parse_value(sweep, row["value"]),
                    "rate": rate,
                    "low": low,
                    "high": high,
                    "successes": int(row["successes"]),
                    "episodes": int(row["num_episodes"]),
                    "baseline": row["is_baseline"].lower() == "true",
                    "p_holm": float(row["holm_adjusted_p_value"]),
                }
            )
    for rows in sweeps.values():
        rows.sort(key=lambda entry: entry["value"])
    return sweeps


def plot_sweep(ax: plt.Axes, sweep: str, rows: list[dict[str, Any]]) -> None:
    episodes = rows[0]["episodes"]
    x = list(range(len(rows)))
    rates = [row["rate"] * 100 for row in rows]
    errors = [
        [(row["rate"] - row["low"]) * 100 for row in rows],
        [(row["high"] - row["rate"]) * 100 for row in rows],
    ]

    ax.errorbar(
        x,
        rates,
        yerr=errors,
        fmt="o-",
        color="#1f77b4",
        ecolor="#4c4c4c",
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
        markersize=6,
        linewidth=1.8,
        label="Success rate (95% Wilson CI)",
        zorder=3,
    )

    for xpos, row in zip(x, rows):
        rate = row["rate"] * 100
        marker = "* " if row["p_holm"] < 0.05 and not row["baseline"] else ""
        ax.annotate(
            f"{marker}{rate:.1f}%",
            (xpos, row["high"] * 100),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            color="#d62728" if marker else "#111111",
            fontweight="bold" if marker else "normal",
        )
        if row["baseline"]:
            ax.annotate(
                "baseline",
                (xpos, row["low"] * 100),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=8,
                style="italic",
                color="#1f77b4",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(row["value"]) for row in rows])
    ax.set_xlabel(X_LABELS[sweep])
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"{TITLES[sweep]}\nn={episodes} episodes per value")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path("eval_results/xiaomi_ofat_analysis_50ep")
    parser.add_argument("--input", type=Path, default=default_dir / "ofat_analysis.csv")
    parser.add_argument("--output", type=Path, default=default_dir / "ofat_success_rate.png")
    args = parser.parse_args()

    sweeps = load_rows(args.input)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, sweep in zip(axes, SWEEP_ORDER):
        plot_sweep(ax, sweep, sweeps[sweep])
        if sweep == "delay":
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Xiaomi OFAT: success rate vs. variable value (paired seeds, * = Holm-adjusted p < 0.05)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

"""Summarize and plot the sparse Xiaomi sim-domain replan sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REPLANS = (1, 6, 11, 16, 21, 26, 31, 36, 41, 46)
EXPECTED_EPISODES = {1: 50, **{replan: 25 for replan in DEFAULT_REPLANS if replan != 1}}


def load_rows(root: Path, replans: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replan in replans:
        path = root / f"replan{replan}" / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            condition_rows = list(csv.DictReader(file))
        if len(condition_rows) != 2:
            raise ValueError(f"expected two delay rows in {path}, got {len(condition_rows)}")
        for row in condition_rows:
            if int(row["replan_steps"]) != replan:
                raise ValueError(f"unexpected replan in {path}: {row['replan_steps']}")
            if row["delay_domain"] != "sim" or row["rtc"].lower() != "false":
                raise ValueError(f"unexpected delay domain or RTC setting in {path}")
            delay = int(row["delay_ms"])
            if delay not in {0, 100}:
                raise ValueError(f"unexpected delay in {path}: {delay}")
            episodes = int(row["num_episodes"])
            if episodes != EXPECTED_EPISODES[replan] or int(row["errors"]) != 0:
                raise ValueError(f"incomplete or failed condition in {path}: {row}")
            rows.append({
                "replan_steps": replan,
                "delay_ms": delay,
                "num_episodes": episodes,
                "successes": int(row["successes"]),
                "success_rate": float(row["success_rate"]),
                "success_rate_ci95_low": float(row["success_rate_ci95_low"]),
                "success_rate_ci95_high": float(row["success_rate_ci95_high"]),
                "mean_steps": float(row["mean_steps"]),
            })
    return sorted(rows, key=lambda row: (int(row["delay_ms"]), int(row["replan_steps"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.root, DEFAULT_REPLANS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = args.output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    styles = {
        0: {"color": "#1769aa", "label": "0 ms injected delay"},
        100: {"color": "#c05a00", "label": "100 ms injected delay (2 sim steps)"},
    }
    for delay in (0, 100):
        subset = [row for row in rows if row["delay_ms"] == delay]
        x = [int(row["replan_steps"]) for row in subset]
        y = [100.0 * float(row["success_rate"]) for row in subset]
        low = [100.0 * float(row["success_rate_ci95_low"]) for row in subset]
        high = [100.0 * float(row["success_rate_ci95_high"]) for row in subset]
        ax.plot(x, y, marker="o", markersize=5, linewidth=2, **styles[delay])
        ax.fill_between(x, low, high, color=styles[delay]["color"], alpha=0.14, linewidth=0)

    ax.axvline(11, color="#6b7280", linestyle=":", linewidth=1)
    ax.set_xlim(1, 46)
    ax.set_ylim(0, 100)
    ax.set_xticks(DEFAULT_REPLANS)
    ax.set_xlabel("Replan steps")
    ax.set_ylabel("Success rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        "Xiaomi: replan frequency under sim-domain injected delay\n"
        "PickPlaceCounterToCabinet, legacy5 scenes, RTC off; shaded = 95% Wilson CI"
    )
    ax.text(
        0.01,
        -0.22,
        "n=50/delay at replan=1; n=25/delay at all other sampled replan values.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {csv_output}, {args.output}, and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

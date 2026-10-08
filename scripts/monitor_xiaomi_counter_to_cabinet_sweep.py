"""Monitor the throttled Xiaomi delay/replan array and estimate completion time.

Usage:
    .conda-env/bin/python scripts/monitor_xiaomi_counter_to_cabinet_sweep.py JOB_ID

For continuous terminal updates:
    watch -n 30 '.conda-env/bin/python scripts/monitor_xiaomi_counter_to_cabinet_sweep.py JOB_ID'

The experiment itself uses sim-domain delay semantics. The ETA is necessarily a
wall-clock estimate because it predicts when the Slurm jobs will finish; wall
time is not used as an experimental delay variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


DELAYS = (0, 100, 300, 500)
REPLANS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
DEFAULT_ROOT = Path("eval_results/xiaomi_counter_to_cabinet_sim_replan_100ep")
EPISODES_PER_CELL = 100
TOTAL_CELLS = len(DELAYS) * len(REPLANS)


def run_command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def cell_path(root: Path, delay: int, replan: int) -> Path:
    return root / "PickPlaceCounterToCabinet" / f"delay{delay}ms" / f"replan{replan}"


def read_cell(path: Path) -> tuple[int, int, float, bool]:
    episodes = path / "episodes.csv"
    if not episodes.exists():
        return 0, 0, 0.0, False
    rows = list(csv.DictReader(episodes.open(newline="", encoding="utf-8")))
    successes = sum(row.get("success", "").lower() == "true" for row in rows)
    wall_times = []
    for row in rows:
        try:
            wall_times.append(float(row["wall_time_s"]))
        except (KeyError, TypeError, ValueError):
            pass
    complete = False
    summary = path / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            complete = data.get("num_results") == EPISODES_PER_CELL and len(data.get("conditions", [])) == 1
        except (OSError, json.JSONDecodeError):
            pass
    return len(rows), successes, sum(wall_times) / len(wall_times) if wall_times else 0.0, complete


def job_states(job_id: str) -> dict[str, int]:
    states: dict[str, int] = {}
    output = run_command("squeue", "-h", "-j", job_id, "-o", "%T|%i")
    for line in output.splitlines():
        state, job_expr = line.split("|", 1)
        count = count_array_entries(job_expr)
        states[state] = states.get(state, 0) + count
    if states:
        return states
    output = run_command("sacct", "-X", "-n", "-P", "-j", job_id, "--format=State")
    for state in output.splitlines():
        state = state.split("+")[0].strip()
        if state:
            states[state] = states.get(state, 0) + 1
    return states


def count_array_entries(job_expr: str) -> int:
    """Count one Slurm job-id field, including compressed array ranges."""
    match = re.search(r"\[(.+)\]", job_expr)
    if not match:
        return 1
    count = 0
    for item in match.group(1).split(","):
        item = item.split("%", 1)[0]
        if "-" not in item:
            count += 1
            continue
        start, end = item.split("-", 1)
        count += int(end) - int(start) + 1
    return count


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Slurm array job ID")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    total_done = total_successes = 0
    observed_rates = []
    complete_cells = 0
    print(f"Updated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Job: {args.job_id} | semantics: sim-domain delay, 20 Hz")
    print("Cell                 Progress  Success  Avg/ep  Status")
    print("-------------------  --------  -------  ------  --------")
    for delay in DELAYS:
        for replan in REPLANS:
            count, successes, avg_wall, complete = read_cell(cell_path(args.root, delay, replan))
            total_done += count
            total_successes += successes
            if avg_wall:
                observed_rates.append(avg_wall)
            if complete:
                complete_cells += 1
            status = "DONE" if complete else ("RUN" if count else "PENDING")
            avg = f"{avg_wall:5.1f}s" if avg_wall else "   -  "
            print(f"{delay:>3}ms / r{replan:<2}       {count:>3}/{EPISODES_PER_CELL:<3}    {successes:>3}     {avg}  {status}")

    states = job_states(args.job_id)
    active = sum(states.get(state, 0) for state in ("RUNNING", "COMPLETING"))
    pending = sum(states.get(state, 0) for state in ("PENDING", "CONFIGURING"))
    failed = sum(states.get(state, 0) for state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"))
    avg_episode = sum(observed_rates) / len(observed_rates) if observed_rates else 41.2
    remaining = TOTAL_CELLS * EPISODES_PER_CELL - total_done
    slots = max(1, active)
    eta_seconds = remaining * avg_episode / slots
    eta = datetime.now().astimezone() + timedelta(seconds=eta_seconds)
    success_rate = total_successes / total_done if total_done else 0.0
    print()
    print(f"Overall: {total_done}/{TOTAL_CELLS * EPISODES_PER_CELL} episodes, {complete_cells}/{TOTAL_CELLS} cells complete")
    print(f"Success so far: {total_successes}/{total_done} ({success_rate:.1%})")
    print(f"Slurm states: running={active} pending={pending} failed={failed} raw={states or {'UNKNOWN': 1}}")
    print(f"Observed throughput basis: {avg_episode:.1f}s/episode; ETA: {format_duration(eta_seconds)} (around {eta.strftime('%Y-%m-%d %H:%M:%S %Z')})")


if __name__ == "__main__":
    main()

"""Re-render a finished latency run as video, without re-deciding anything.

The sweep itself recorded only statistics, but each row stores the task,
condition, seed, and scene. This wrapper restores those inputs and writes one
annotated mp4 per episode, so a past result can be inspected frame by frame
instead of inferred from a success rate. Model inference may vary across
runtimes, so the output is a rerun rather than a bit-for-bit trajectory dump.

Usage:
    # one slice
    .conda-env/bin/python scripts/replay_run_video.py \
        --run eval_results/simdelay_grid/gr00t_n1_5/OpenDrawer --limit 4

    # a whole model, successes and failures only, into a dedicated folder
    .conda-env/bin/python scripts/replay_run_video.py \
        --run eval_results/simdelay_grid/gr00t_n1_5 --out eval_results/videos/gr00t

    # a same-seed 0 ms vs 500 ms pair for one task
    .conda-env/bin/python scripts/replay_run_video.py \
        --run eval_results/simdelay_grid/gr00t_n1_5/OpenDrawer \
        --seeds 0 --delays 0,500
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True,
                        help="Finished run: a slice dir, a model dir, or the sweep root.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Video directory (default: <run>/videos).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N episodes, in stored order.")
    parser.add_argument("--seeds", default=None,
                        help="Only replay these seeds, e.g. '0,3'.")
    parser.add_argument("--delays", default=None,
                        help="Only replay these injected delays, e.g. '0,500'.")
    parser.add_argument("--tasks", default=None,
                        help="Only replay these tasks, e.g. 'OpenDrawer,TurnOffStove'.")
    parser.add_argument("--model-env", default=None,
                        help="Override the isolated worker env for the model.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Truncate episodes; useful for a fast preview.")
    parser.add_argument("--control-frequency", type=float, default=20.0)
    parser.add_argument("--camera", default=None, help="Sim camera name.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--scene-mode",
        choices=("auto", "pinned", "sampled"),
        default="auto",
        help="How to restore the scene RNG path for the replay.",
    )
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Extra flags passed through to the experiment script.")
    args = parser.parse_args()

    from scripts.xiaomi_latency_experiment import (
        build_replay_jobs,
        discover_slices,
        load_replay_rows,
    )

    slices = discover_slices(args.run)
    if not slices:
        raise SystemExit(f"no episodes.csv/jsonl under {args.run}")
    print(f"{len(slices)} slice(s) under {args.run}")

    wanted_seeds = _int_set(args.seeds)
    wanted_delays = _int_set(args.delays)
    wanted_tasks = set(args.tasks.split(",")) if args.tasks else None

    kept: dict[Path, list[dict]] = {}
    for slice_dir in slices:
        rows = [
            row for row in load_replay_rows(slice_dir)
            if (wanted_tasks is None or row.get("task_name") in wanted_tasks)
            and (wanted_seeds is None or int(row["seed"]) in wanted_seeds)
            and (wanted_delays is None or int(row["delay_ms"]) in wanted_delays)
        ]
        if rows:
            kept[slice_dir] = rows
    if not kept:
        raise SystemExit("filters selected no episodes")

    total = sum(len(rows) for rows in kept.values())
    print(f"{total} episode(s) selected across {len(kept)} slice(s)")

    # Stage the filtered selection so the downstream script stays a pure replayer.
    # Stable per-run staging name, so a re-run overwrites rather than accumulates.
    slug = "_".join(args.run.resolve().parts[-3:])
    staging = Path("/tmp") / f"replay_selection_{slug}"
    shutil.rmtree(staging, ignore_errors=True)
    for index, (slice_dir, rows) in enumerate(sorted(kept.items())):
        target = staging / f"{index:03d}_{slice_dir.parent.name}_{slice_dir.name}"
        _write_episodes(target, rows)
        _copy_summary(slice_dir, target)

    out_dir = Path(args.out) if args.out else Path(args.run) / "videos"
    command = [
        sys.executable, str(ROOT_DIR / "scripts/xiaomi_latency_experiment.py"),
        "--replay-from", str(staging),
        "--record-video", "--video-dir", str(out_dir),
        "--control-frequency", str(args.control_frequency),
        "--replay-scene-mode", args.scene_mode,
    ]
    for flag, value in (("--horizon", args.horizon), ("--model-env", args.model_env),
                        ("--video-camera", args.camera), ("--video-width", args.width),
                        ("--video-height", args.height), ("--replay-limit", args.limit)):
        if value is not None:
            command += [flag, str(value)]
    command += list(args.extra)

    print("$ " + " ".join(command))
    raise SystemExit(subprocess.call(command, cwd=ROOT_DIR))


def _int_set(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def _write_episodes(directory: Path, rows: list[dict]) -> None:
    import csv

    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_summary(source: Path, target: Path) -> None:
    summary = source / "summary.json"
    if summary.exists():
        (target / "summary.json").write_text(
            summary.read_text(encoding="utf-8"), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

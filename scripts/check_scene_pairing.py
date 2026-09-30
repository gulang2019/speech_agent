"""Check that every (model, task) slice used the same scene for the same seed.

Before ``pin_scene`` the layout depended on how many fixture-placement retries
happened inside ``Kitchen._load_model``, so a seed could silently map to a
different kitchen in different conditions.  This script reports how often that
happened so a sweep can be triaged (rerun vs. keep) without guessing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval_results/simdelay_grid"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    total = 0
    drifted = 0
    per_slice: list[tuple[str, str, int, int]] = []
    for model_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            flat = task_dir / "episodes.jsonl"
            if not flat.exists():
                continue
            by_episode: dict[tuple[int, int], dict[int, tuple[int, int]]] = defaultdict(dict)
            for line in flat.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                by_episode[(row["episode"], row["seed"])][row["delay_ms"]] = (
                    row["layout_id"],
                    row["style_id"],
                )
            bad = sum(1 for scenes in by_episode.values() if len(set(scenes.values())) > 1)
            total += len(by_episode)
            drifted += bad
            if bad:
                per_slice.append((model_dir.name, task_dir.name, bad, len(by_episode)))

    print(f"episode groups: {total}")
    print(f"with scene differing across delays: {drifted} ({100 * drifted / total:.2f}%)")
    for model, task, bad, n in sorted(per_slice, key=lambda item: -item[2]):
        print(f"  {model}/{task}: {bad}/{n}")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "episode_groups": total,
                    "drifted_groups": drifted,
                    "drifting_slices": [
                        {"model": m, "task": t, "drifted": b, "episodes": n}
                        for m, t, b, n in per_slice
                    ],
                },
                indent=2,
            )
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

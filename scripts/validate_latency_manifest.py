"""Validate the cross-model latency benchmark manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.latency_benchmark_config import DEFAULT_MANIFEST, load_manifest
    from scripts.latency_model_adapters import get_model_spec
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from latency_benchmark_config import DEFAULT_MANIFEST, load_manifest
    from latency_model_adapters import get_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    for model in manifest["models"]:
        get_model_spec(model)
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    task_names = sorted({task for tasks in manifest["task_sets"].values() for task in tasks})
    horizons = {task: get_task_horizon(task) for task in task_names}
    print(f"OK: {args.manifest}")
    print(f"  models: {', '.join(manifest['models'])}")
    print(f"  task sets: {', '.join(manifest['task_sets'])}")
    print(f"  scene sets: {', '.join(manifest['scene_sets'])}")
    for name, tasks in manifest["task_sets"].items():
        print(f"  {name}: {len(tasks)} tasks")
    print(f"  registered tasks: {len(horizons)}")
    for task, horizon in horizons.items():
        print(f"    {task}: horizon={horizon}")
    for name, scene_set in manifest["scene_sets"].items():
        print(f"  {name}: {len(scene_set['layout_and_style_ids'])} scenes")


if __name__ == "__main__":
    main()

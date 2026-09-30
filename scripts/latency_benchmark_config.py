"""Configuration helpers for the cross-model latency benchmark.

The benchmark manifest is intentionally JSON so that it can be consumed by
Slurm wrappers, validation tools, and future model adapters without adding a
new runtime dependency.  This module contains validation and lookup logic
only; it does not import MuJoCo or load a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "configs/latency_benchmark_v1.json"


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and validate a benchmark manifest."""
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a manifest violates the benchmark schema."""
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")

    frequency = manifest.get("control_frequency_hz")
    if not isinstance(frequency, (int, float)) or frequency <= 0:
        raise ValueError("control_frequency_hz must be positive")

    delay_domain = manifest.get("delay_domain", "sim")
    if delay_domain not in {"sim", "wall"}:
        raise ValueError("delay_domain must be 'sim' or 'wall'")

    models = manifest.get("models")
    if not isinstance(models, list) or not models or len(set(models)) != len(models):
        raise ValueError("models must be a non-empty list without duplicates")

    task_sets = manifest.get("task_sets")
    if not isinstance(task_sets, dict) or not task_sets:
        raise ValueError("task_sets must be a non-empty object")
    for name, tasks in task_sets.items():
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"task set {name!r} must be a non-empty list")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"task set {name!r} contains duplicate tasks")

    scene_sets = manifest.get("scene_sets")
    if not isinstance(scene_sets, dict) or not scene_sets:
        raise ValueError("scene_sets must be a non-empty object")
    for name, scene_set in scene_sets.items():
        if not isinstance(scene_set, dict):
            raise ValueError(f"scene set {name!r} must be an object")
        object_split = scene_set.get("object_split")
        if object_split not in {"pretrain", "target"}:
            raise ValueError(f"scene set {name!r} has invalid object_split")
        scene_ids = scene_set.get("layout_and_style_ids")
        if not isinstance(scene_ids, list) or not scene_ids:
            raise ValueError(f"scene set {name!r} must have scene ids")
        normalized_ids: list[tuple[int, int]] = []
        for pair in scene_ids:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, int) and value > 0 for value in pair)
            ):
                raise ValueError(f"scene set {name!r} has invalid layout/style id: {pair!r}")
            normalized_ids.append((pair[0], pair[1]))
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"scene set {name!r} contains duplicate layout/style ids")

    delay_sweeps = manifest.get("delay_sweeps_ms")
    if not isinstance(delay_sweeps, dict) or not delay_sweeps:
        raise ValueError("delay_sweeps_ms must be a non-empty object")
    for name, delays in delay_sweeps.items():
        if (
            not isinstance(delays, list)
            or not delays
            or any(not isinstance(delay, int) or delay < 0 for delay in delays)
            or delays != sorted(set(delays))
        ):
            raise ValueError(f"delay sweep {name!r} must be sorted non-negative integers")


def get_task_set(manifest: dict[str, Any], name: str) -> list[str]:
    """Return a copy of a named task set."""
    try:
        tasks = manifest["task_sets"][name]
    except KeyError as error:
        available = ", ".join(sorted(manifest.get("task_sets", {})))
        raise ValueError(f"unknown task set {name!r}; available: {available}") from error
    return list(tasks)


def get_scene_set(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a normalized scene-set configuration."""
    try:
        scene_set = manifest["scene_sets"][name]
    except KeyError as error:
        available = ", ".join(sorted(manifest.get("scene_sets", {})))
        raise ValueError(f"unknown scene set {name!r}; available: {available}") from error
    normalized = dict(scene_set)
    normalized["layout_and_style_ids"] = tuple(
        tuple(pair) for pair in scene_set["layout_and_style_ids"]
    )
    return normalized

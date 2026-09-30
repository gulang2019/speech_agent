import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.latency_benchmark_config import (
    DEFAULT_MANIFEST,
    get_scene_set,
    get_task_set,
    load_manifest,
    validate_manifest,
)
from scripts.latency_model_adapters import get_model_spec


def test_default_manifest_is_valid_and_has_expected_core_suite():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert get_task_set(manifest, "atomic_smoke") == [
        "PickPlaceCounterToCabinet",
        "OpenDrawer",
    ]
    assert len(get_task_set(manifest, "atomic_screening")) == 8
    assert len(get_task_set(manifest, "atomic_final")) == 6
    assert len(get_scene_set(manifest, "pretrain20")["layout_and_style_ids"]) == 20
    assert get_scene_set(manifest, "single_scene_L7S10")["layout_and_style_ids"] == ((7, 10),)
    assert manifest["delay_sweeps_ms"]["final"] == [0, 50, 100, 200, 300, 500]


def test_scene_set_is_normalized_to_tuples():
    manifest = load_manifest(DEFAULT_MANIFEST)
    scene_set = get_scene_set(manifest, "legacy5")
    assert scene_set["layout_and_style_ids"][0] == (1, 1)


def test_manifest_rejects_duplicate_scene_ids():
    manifest = load_manifest(DEFAULT_MANIFEST)
    broken = copy.deepcopy(manifest)
    broken["scene_sets"]["legacy5"]["layout_and_style_ids"].append([1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(broken)


def test_model_registry_records_prepared_backends():
    assert get_model_spec("xiaomi").status == "ready"
    assert get_model_spec("gr00t_n1_5").status == "ready"
    assert get_model_spec("pi0_5").action_chunk_length == 50
    assert get_model_spec("diffusion_policy").action_chunk_length == 8


def test_manifest_declares_sim_delay_domain():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert manifest["delay_domain"] == "sim"


def test_manifest_rejects_unknown_delay_domain():
    manifest = load_manifest(DEFAULT_MANIFEST)
    broken = copy.deepcopy(manifest)
    broken["delay_domain"] = "steps"
    with pytest.raises(ValueError, match="delay_domain"):
        validate_manifest(broken)

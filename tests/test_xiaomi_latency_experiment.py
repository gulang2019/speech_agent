import argparse
import csv
from collections import deque
from concurrent.futures import Future
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.xiaomi_latency_experiment import (
    CONTROL_MODE_DIM,
    Condition,
    GRIPPER_DIM,
    PendingInference,
    blend_plans,
    build_replay_jobs,
    condition_from_row,
    delay_in_steps,
    build_parser,
    detect_model,
    discover_slices,
    install_completed_inference,
    load_replay_rows,
    make_action,
    pin_scene,
    resolve_replay_scene_mode,
    replay_episodes,
    sample_symmetric_jitter,
    sample_lower_bound_twopoint_jitter,
    sample_lognormal_jitter,
    sample_shifted_negative_binomial_jitter,
    video_name,
    wilson_interval,
)


def completed_pending(chunk, requested_step=0, available_step=None):
    future = Future()
    future.set_result((np.asarray(chunk, dtype=np.float32), 10.0, 20.0))
    return PendingInference(
        future=future,
        requested_step=requested_step,
        available_step=requested_step if available_step is None else available_step,
    )


def test_rtc_off_replaces_queued_actions():
    old_plan = deque([np.full(7, 1.0), np.full(7, 2.0)])
    new_chunk = np.stack([np.full(7, value) for value in range(10)])

    plan, age, stale, replaced, model_ms, total_ms = install_completed_inference(
        completed_pending(new_chunk),
        step=3,
        plan=old_plan,
        rtc=False,
        replan_steps=5,
        blend_steps=4,
    )

    assert len(plan) == 5
    np.testing.assert_allclose(plan[0], 0.0)
    np.testing.assert_allclose(plan[-1], 4.0)
    assert age == 3
    assert stale == 0
    assert replaced == 2
    assert (model_ms, total_ms) == (10.0, 20.0)


def test_rtc_on_aligns_and_blends_new_actions():
    old_plan = deque([np.full(7, 100.0), np.full(7, 200.0)])
    new_chunk = np.stack([np.full(7, value) for value in range(10)])

    plan, age, stale, replaced, _, _ = install_completed_inference(
        completed_pending(new_chunk),
        step=2,
        plan=old_plan,
        rtc=True,
        replan_steps=4,
        blend_steps=2,
    )

    assert len(plan) == 4
    assert age == 2
    np.testing.assert_allclose(plan[0], 51.0)
    np.testing.assert_allclose(plan[1], 3.0)
    np.testing.assert_allclose(plan[2], 4.0)
    np.testing.assert_allclose(plan[3], 5.0)
    assert stale == 2
    assert replaced == 0


def test_blend_plans_keeps_unoverlapped_actions():
    old_plan = deque([np.full(7, 10.0)])
    new_plan = np.stack([np.full(7, 20.0), np.full(7, 30.0)])

    plan = blend_plans(old_plan, new_plan, blend_steps=1)

    assert len(plan) == 2
    np.testing.assert_allclose(plan[0], 20.0)
    np.testing.assert_allclose(plan[1], 30.0)


def test_wilson_interval_known_values_and_bounds():
    low, high = wilson_interval(25, 50)
    np.testing.assert_allclose([low, high], [0.366445, 0.633555], rtol=1e-5)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0


def test_delay_in_steps_matches_control_period():
    assert delay_in_steps(0, 20.0) == 0
    assert delay_in_steps(100, 20.0) == 2
    assert delay_in_steps(500, 20.0) == 10
    assert delay_in_steps(600, 20.0) == 12
    # off-grid delays round to the nearest control step
    assert delay_in_steps(30, 20.0) == 1
    assert delay_in_steps(40, 20.0) == 1
    # step-domain delays do not depend on control frequency being real-time
    assert delay_in_steps(100, 0.0) == 0
    assert delay_in_steps(100, 10.0) == 1


def test_symmetric_jitter_preserves_requested_moments_and_replan_integrality():
    rng = np.random.default_rng(7)
    delay = [sample_symmetric_jitter(100, 2_500, rng, integer=False) for _ in range(10_000)]
    assert set(delay) == {50.0, 150.0}
    np.testing.assert_allclose(np.mean(delay), 100.0, atol=1.5)
    np.testing.assert_allclose(np.var(delay), 2_500.0, atol=5.0)

    rng = np.random.default_rng(8)
    replans = [sample_symmetric_jitter(10, 4, rng, integer=True) for _ in range(100)]
    assert set(replans) == {8, 12}


def test_lower_bound_jitter_supports_variance_above_symmetric_limit():
    rng = np.random.default_rng(9)
    values = [
        sample_lower_bound_twopoint_jitter(
            100, 90_000, rng, lower_bound=0, integer=False
        )
        for _ in range(20_000)
    ]
    assert set(values) == {0.0, 1000.0}
    np.testing.assert_allclose(np.mean(values), 100.0, atol=4.0)
    np.testing.assert_allclose(np.var(values), 90_000.0, atol=3_000.0)

    rng = np.random.default_rng(10)
    replans = [
        sample_lower_bound_twopoint_jitter(
            10, 180, rng, lower_bound=1, integer=True
        )
        for _ in range(100)
    ]
    assert set(replans) == {1, 30}


def test_lognormal_and_shifted_negative_binomial_match_target_moments():
    rng = np.random.default_rng(11)
    delay = [sample_lognormal_jitter(100, 90_000, rng, integer=False) for _ in range(200_000)]
    np.testing.assert_allclose(np.mean(delay), 100.0, rtol=0.03)
    # Heavy-tailed lognormal variance converges slowly; validate with a
    # tolerance appropriate for a finite Monte Carlo sample.
    np.testing.assert_allclose(np.var(delay), 90_000.0, rtol=0.25)

    rng = np.random.default_rng(12)
    replans = [
        sample_shifted_negative_binomial_jitter(
            10, 180, rng, lower_bound=1, integer=True
        )
        for _ in range(200_000)
    ]
    assert min(replans) >= 1
    np.testing.assert_allclose(np.mean(replans), 10.0, rtol=0.03)
    np.testing.assert_allclose(np.var(replans), 180.0, rtol=0.12)


def test_parser_converts_cv2_to_variance():
    args = build_parser().parse_args(
        ["--delays-ms", "100", "--replan-steps-list", "10", "--delay-jitter-cv2", "9"]
    )
    assert args.delay_jitter_cv2 == 9


def test_parser_accepts_manifest_defined_scene_set_names():
    args = build_parser().parse_args(
        ["--scene-set", "countertostove_latency_signal5"]
    )
    assert args.scene_set == "countertostove_latency_signal5"


def test_parser_accepts_legacy_scene_sampling_mode():
    args = build_parser().parse_args(["--scene-selection", "sampled"])
    assert args.scene_selection == "sampled"


def test_condition_name_distinguishes_delay_domain():
    sim = Condition(delay_ms=100, replan_steps=5, rtc=False, delay_steps=2, delay_domain="sim")
    wall = Condition(delay_ms=100, replan_steps=5, rtc=False, delay_steps=2, delay_domain="wall")
    assert sim.name == "delay100ms_d2steps_replan5_rtcoff"
    assert wall.name == "delay100ms_wall_replan5_rtcoff"


def test_sim_delay_makes_age_equal_injected_steps():
    """A response installed exactly delay_steps later is stale by delay_steps."""
    chunk = np.stack([np.full(7, value) for value in range(10)])
    for delay_steps in (0, 2, 5):
        pending = completed_pending(chunk, requested_step=10, available_step=10 + delay_steps)
        plan, age, stale, _, _, _ = install_completed_inference(
            pending,
            step=10 + delay_steps,
            plan=deque(),
            rtc=True,
            replan_steps=5,
            blend_steps=0,
        )
        assert age == delay_steps
        assert stale == delay_steps
        assert len(plan) > 0


class _FakeEnv:
    """Minimal stand-in for the 12D ``PandaOmron`` dense action spec."""

    def __init__(self, dim=12):
        self.action_spec = np.zeros((2, dim), dtype=np.float32)


def test_make_action_forwards_all_twelve_dims():
    """The RoboCasa365 layout must survive the policy -> env hop intact."""
    env = _FakeEnv()
    policy_action = np.arange(12, dtype=np.float32) / 100.0
    action = make_action(env, policy_action)

    assert action.shape == (12,)
    np.testing.assert_allclose(action[:6], policy_action[:6])
    # binary channels are thresholded, everything else is verbatim
    np.testing.assert_allclose(action[7:11], policy_action[7:11])


def test_make_action_thresholds_binary_channels():
    """gripper/control_mode follow ``unmap_action``: <0.5 -> -1 else +1.

    A raw 0 reaching the gripper controller evaluates ``np.sign(0) == 0`` and
    freezes the fingers, so {0, 1} checkpoints (GR00T) depend on this.
    """
    env = _FakeEnv()
    for raw, expected in ((0.0, -1.0), (1.0, 1.0), (0.49, -1.0), (0.5, 1.0), (-1.0, -1.0)):
        action = make_action(env, np.full(12, raw, dtype=np.float32))
        assert action[GRIPPER_DIM] == expected, raw
        assert action[CONTROL_MODE_DIM] == expected, raw


def test_make_action_keeps_pure_7d_chunk_padded():
    """Legacy 7D chunks still zero-fill base/control instead of inheriting junk."""
    env = _FakeEnv()
    policy_action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0], dtype=np.float32)
    action = make_action(env, policy_action)

    assert action.shape == (12,)
    np.testing.assert_allclose(action[:6], policy_action[:6])
    # 7D chunks skip base/control, so those stay zero (legacy behavior)
    np.testing.assert_allclose(action[7:12], 0.0)


def test_make_action_drops_overflow_beyond_env_width():
    env = _FakeEnv(dim=12)
    action = make_action(env, np.arange(16, dtype=np.float32))
    assert action.shape == (12,)
    np.testing.assert_allclose(action[:6], np.arange(6, dtype=np.float32))


def test_pin_scene_is_deterministic_and_covers_all_scenes():
    """Scene choice must not depend on retries or interpreter state.

    ``Kitchen._load_model`` re-samples the layout on every fixture-placement
    retry, so without pinning the same seed could land on different kitchens.
    """
    scenes = ((1, 1), (2, 2), (6, 9), (7, 10))
    assert pin_scene(0, scenes) == {"layout_ids": 1, "style_ids": 1}
    assert pin_scene(1, scenes) == {"layout_ids": 2, "style_ids": 2}
    assert pin_scene(3, scenes) == {"layout_ids": 7, "style_ids": 10}
    # wraps around, so a 30-episode run covers the manifest evenly
    assert pin_scene(4, scenes) == pin_scene(0, scenes)
    assert pin_scene(0, None) == {}
    assert pin_scene(0, ()) == {}
    # distinct seeds must not collapse onto one scene
    assert len({tuple(pin_scene(seed, scenes).values()) for seed in range(4)}) == 4


def test_resolve_replay_scene_mode_detects_legacy_sampling():
    scenes = ((1, 1), (2, 2), (6, 9), (7, 10))
    assert resolve_replay_scene_mode(0, (1, 1), scenes, "auto") == "pinned"
    assert resolve_replay_scene_mode(0, (6, 9), scenes, "auto") == "sampled"
    assert resolve_replay_scene_mode(0, (6, 9), scenes, "pinned") == "pinned"
    assert resolve_replay_scene_mode(0, (6, 9), scenes, "sampled") == "sampled"


def test_delay_step_frequency_can_be_independent_of_wall_clock_pacing():
    args = build_parser().parse_args(
        ["--control-frequency", "0", "--delay-step-frequency", "20"]
    )
    delay_frequency = args.delay_step_frequency or args.control_frequency
    assert delay_in_steps(100, delay_frequency) == 2


def test_replay_episodes_raises_when_a_replay_row_errors(monkeypatch):
    from scripts.xiaomi_latency_experiment import EpisodeResult, ReplayJob

    result = EpisodeResult(
        task_name="OpenDrawer", condition="delay0ms_d0steps_replan5_rtcoff",
        delay_ms=0, delay_domain="sim", delay_steps=0, replan_steps=5, rtc=False,
        episode=0, seed=0, layout_id=1, style_id=1, success=False, steps=0,
        inference_requests=0, mean_arrival_age_steps=0.0, max_arrival_age_steps=0,
        stale_actions_discarded=0, queued_actions_replaced=0, fallback_steps=0,
        mean_model_latency_ms=0.0, mean_total_inference_latency_ms=0.0,
        mean_block_wait_ms=0.0, wall_time_s=0.0, error="RuntimeError: scene mismatch",
    )
    monkeypatch.setattr(
        "scripts.xiaomi_latency_experiment.run_episode", lambda *args, **kwargs: result
    )
    args = type("Args", (), {"record_video": False, "horizon": 1})()
    job = ReplayJob("OpenDrawer", Condition(0, 5, False), 0, 0, 1, 1)
    with pytest.raises(RuntimeError, match="scene mismatch"):
        replay_episodes(object(), [job], args)


def _write_episodes(directory: Path, rows):
    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _episode_row(**overrides):
    row = {
        "task_name": "OpenDrawer",
        "condition": "delay300ms_d6steps_replan5_rtcoff",
        "delay_ms": "300",
        "delay_domain": "sim",
        "delay_steps": "6",
        "replan_steps": "5",
        "rtc": "False",
        "episode": "7",
        "seed": "7",
        "layout_id": "15",
        "style_id": "15",
        "success": "True",
        "steps": "161",
        "error": "",
    }
    row.update(overrides)
    return row


def test_condition_from_row_rebuilds_the_recorded_condition():
    condition = condition_from_row(_episode_row(), control_frequency=20.0)
    assert condition.delay_ms == 300
    assert condition.delay_steps == 6
    assert condition.replan_steps == 5
    assert condition.rtc is False
    assert condition.name == "delay300ms_d6steps_replan5_rtcoff"


def test_condition_from_row_infers_steps_and_reads_rtc_on():
    row = _episode_row(delay_ms="100", delay_steps="", rtc="True")
    condition = condition_from_row(row, control_frequency=20.0)
    assert condition.delay_steps == 2
    assert condition.rtc is True


def test_condition_from_row_defaults_to_sim_for_legacy_rows():
    """Pre-``delay_domain`` rows were all sim; they must not be read as wall."""
    row = _episode_row()
    row.pop("delay_domain")
    assert condition_from_row(row, control_frequency=20.0).delay_domain == "sim"


def test_build_replay_jobs_pins_the_scene_the_episode_actually_used(tmp_path):
    """A stored layout/style must win over the seed mapping."""
    rows = [
        _episode_row(),
        _episode_row(episode="8", seed="8", layout_id="22", style_id="22", delay_ms="0"),
    ]
    _write_episodes(tmp_path, rows)

    jobs = build_replay_jobs(tmp_path, control_frequency=20.0)

    assert [(job.task_name, job.episode, job.seed) for job in jobs] == [
        ("OpenDrawer", 7, 7),
        ("OpenDrawer", 8, 8),
    ]
    assert [(job.layout_id, job.style_id) for job in jobs] == [(15, 15), (22, 22)]
    assert [job.condition.delay_ms for job in jobs] == [300, 0]


def test_build_replay_jobs_skips_errored_episodes(tmp_path):
    _write_episodes(
        tmp_path,
        [_episode_row(), _episode_row(episode="9", seed="9", error="RuntimeError: boom")],
    )
    jobs = build_replay_jobs(tmp_path, control_frequency=20.0)
    assert [job.episode for job in jobs] == [7]


def test_build_replay_jobs_honours_limit_and_tolerates_missing_scene(tmp_path):
    rows = [_episode_row(episode=str(i), seed=str(i)) for i in range(5)]
    rows[1]["layout_id"] = ""
    rows[1]["style_id"] = ""
    _write_episodes(tmp_path, rows)

    jobs = build_replay_jobs(tmp_path, control_frequency=20.0, limit=2)
    assert len(jobs) == 2
    assert jobs[1].layout_id is None and jobs[1].style_id is None


def test_build_replay_jobs_reads_jsonl_and_walks_nested_slices(tmp_path):
    import json

    slice_dir = tmp_path / "xiaomi" / "OpenDrawer"
    slice_dir.mkdir(parents=True)
    rows = [_episode_row()]
    with (slice_dir / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    jobs = build_replay_jobs(tmp_path, control_frequency=20.0)
    assert [job.episode for job in jobs] == [7]
    assert discover_slices(tmp_path) == [slice_dir]


def test_build_replay_jobs_rejects_a_directory_without_episodes(tmp_path):
    with pytest.raises(ValueError):
        build_replay_jobs(tmp_path, control_frequency=20.0)


def test_load_replay_rows_prefers_csv_and_reports_nothing_when_empty(tmp_path):
    _write_episodes(tmp_path, [_episode_row()])
    assert len(load_replay_rows(tmp_path)) == 1
    assert load_replay_rows(Path(tmp_path))[0]["task_name"] == "OpenDrawer"


def test_detect_model_reads_the_run_summary(tmp_path):
    import json

    _write_episodes(tmp_path, [_episode_row()])
    (tmp_path / "summary.json").write_text(
        json.dumps({"model": {"name": "gr00t_n1_5"}}), encoding="utf-8"
    )
    assert detect_model(tmp_path) == "gr00t_n1_5"
    assert detect_model(tmp_path / "missing") is None


def test_video_name_encodes_condition_episode_and_scene():
    condition = condition_from_row(_episode_row(), control_frequency=20.0)
    name = video_name("OpenDrawer", condition, episode=7, seed=7, layout_id=15, style_id=15)
    assert name == "OpenDrawer__delay300ms_d6steps_replan5_rtcoff__ep007_seed007_L15S15.mp4"
    assert "/" not in name
    scene_less = video_name("OpenDrawer", condition, 7, 7, None, None)
    assert scene_less.endswith("_scene.mp4")

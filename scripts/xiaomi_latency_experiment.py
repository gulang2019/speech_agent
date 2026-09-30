"""Evaluate action-chunk policies under injected inference-delay and RTC conditions.

The experiment uses a fixed set of episode seeds for every condition.  Each
model request returns a native action chunk.  ``replan_steps`` controls when the
next request is launched, while the configured delay controls when a response
becomes *usable* by the rollout.

Two delay domains are supported (``--delay-domain``):

* ``sim`` (default): the injected delay is measured in simulation steps, so a
  response can be installed exactly
  ``delay_steps = round(delay_ms * control_frequency / 1000)`` steps after it was
  requested.  Real model latency is recorded but does not influence staleness,
  which makes the injected delay the only controlled variable and keeps the
  stale horizon identical across models with different native inference speeds.
  If the model is slower than the injected delay the main loop blocks until the
  response arrives; the blocking time is reported as ``mean_block_wait_ms``.
* ``wall``: legacy behavior.  The worker sleeps for ``delay_ms`` on top of the
  real model latency, so staleness depends on ``model latency + injected delay``.

RTC semantics used here are deliberately explicit and reproducible:

* RTC off: when a response arrives, replace the queued actions with the new
  chunk starting at action zero.
* RTC on: a completed chunk is aligned to the simulation step at which it
  becomes available.  Actions already made stale by the delay are discarded,
  and the aligned chunk is linearly blended with the currently queued actions.

The loop advances the simulator at ``control_frequency`` by default.  Set
``--control-frequency 0`` for a fast, non-real-time smoke run; in that mode
``sim`` delays still resolve in steps, while ``wall`` delays are only visible
through staleness.

The recorded task, condition, seed, and scene can be re-run after the fact:
``--record-video`` writes an mp4 per episode, and ``--replay-from`` reads the
``episodes.csv`` of a finished run and restores its (task, delay, replan, rtc,
episode, scene) tuples. Model inference can still vary across runtime versions
or nondeterministic backends, so the replay records its own outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from scripts.latency_benchmark_config import (
        DEFAULT_MANIFEST,
        get_scene_set,
        get_task_set,
        load_manifest,
    )
    from scripts.latency_model_adapters import (
        ActionChunkPolicy,
        PolicyObservation,
        create_policy,
        get_model_spec,
        require_implemented_model,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from latency_benchmark_config import DEFAULT_MANIFEST, get_scene_set, get_task_set, load_manifest
    from latency_model_adapters import (
        ActionChunkPolicy,
        PolicyObservation,
        create_policy,
        get_model_spec,
        require_implemented_model,
    )


ROOT_DIR = Path(__file__).resolve().parents[1]

CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
ROBOT_TYPE = "robocasa_mg"
STATE_DIM = 60
ACTION_DIM = 7
MODEL_ACTION_CHUNK = 10
GRIPPER_DIM = 6
CONTROL_MODE_DIM = 11
LEGACY_EVAL_SCENES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))

# Video defaults.  512x768 matches RoboCasa's own ``run_random_rollouts``
# preview, and the agentview camera shows the robot and the fixtures it is
# interacting with, which the three policy cameras do not.
VIDEO_CAMERA = "robot0_agentview_center"
VIDEO_WIDTH = 768
VIDEO_HEIGHT = 512

# Task aliases for names that the official v0.2 evaluator used but that are not
# separate registrations in this repository.  The value is expanded into env
# name, extra constructor kwargs, and the horizon used for the episode cap.
TASK_ALIASES: dict[str, dict[str, Any]] = {
    # v0.2 "OpenDoubleDoor" == open a hinge (double-door) cabinet.
    "OpenDoubleDoor": {
        "env_name": "OpenDoor",
        "env_kwargs": {"fixture_id": 21},  # FixtureType.CABINET_DOUBLE_DOOR
        "horizon": 1050,
    },
}


@dataclass(frozen=True)
class Condition:
    delay_ms: int
    replan_steps: int
    rtc: bool
    delay_steps: int = 0
    delay_domain: str = "sim"

    @property
    def name(self) -> str:
        rtc_name = "on" if self.rtc else "off"
        if self.delay_domain == "wall":
            return f"delay{self.delay_ms}ms_wall_replan{self.replan_steps}_rtc{rtc_name}"
        return (
            f"delay{self.delay_ms}ms_d{self.delay_steps}steps"
            f"_replan{self.replan_steps}_rtc{rtc_name}"
        )


@dataclass
class PendingInference:
    future: Future[tuple[np.ndarray, float, float]]
    requested_step: int
    available_step: int


@dataclass
class EpisodeResult:
    task_name: str
    condition: str
    delay_ms: int
    delay_domain: str
    delay_steps: int
    replan_steps: int
    rtc: bool
    episode: int
    seed: int
    layout_id: int | None
    style_id: int | None
    success: bool
    steps: int
    inference_requests: int
    mean_arrival_age_steps: float
    max_arrival_age_steps: int
    stale_actions_discarded: int
    queued_actions_replaced: int
    fallback_steps: int
    mean_model_latency_ms: float
    mean_total_inference_latency_ms: float
    mean_block_wait_ms: float
    wall_time_s: float
    error: str | None = None


def delay_in_steps(delay_ms: int, control_frequency: float) -> int:
    """Convert an injected delay in milliseconds to simulation control steps."""
    if delay_ms <= 0 or control_frequency <= 0:
        return 0
    return int(round(delay_ms * control_frequency / 1000.0))


class PolicyClient:
    """Client-side input formatting for the local ``robocasa_mg`` checkpoint."""

    action_chunk_length = MODEL_ACTION_CHUNK

    def __init__(
        self,
        model_path: str,
        model_seed: int,
        diffusion_steps: int,
        attn_implementation: str,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for the Xiaomi latency experiment")

        self.device = torch.device("cuda")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.model_seed = model_seed
        self.diffusion_steps = diffusion_steps
        robot_types = self.processor.list_robot_types()
        if ROBOT_TYPE not in robot_types:
            self.close()
            raise ValueError(
                f"Checkpoint does not provide {ROBOT_TYPE!r}; available: {robot_types}"
            )

    def infer(self, observation: PolicyObservation) -> np.ndarray:
        state = observation.xiaomi_state
        images = observation.images
        instruction = observation.instruction
        state_padded = np.zeros(STATE_DIM, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size > STATE_DIM:
            raise ValueError(f"State has {state.size} values, expected <= {STATE_DIM}")
        state_padded[: state.size] = state

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Base View\n"},
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        data = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state_padded.reshape(1, 1, STATE_DIM),
            robot_type=ROBOT_TYPE,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in data.items()
        }
        with self.torch.inference_mode():
            output = self.model(
                **inputs,
                num_steps=self.diffusion_steps,
                seed=self.model_seed,
            )
        actions = self.processor.decode_action(
            output.actions, robot_type=ROBOT_TYPE
        )[0, :, :ACTION_DIM].float().cpu().numpy()
        if actions.shape[0] < MODEL_ACTION_CHUNK:
            raise RuntimeError(f"Model returned {actions.shape[0]} actions, expected at least {MODEL_ACTION_CHUNK}")
        return np.asarray(actions[:MODEL_ACTION_CHUNK], dtype=np.float32)

    def close(self) -> None:
        self.model = None


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def render_observation(
    env: Any, image_size: int, crop_ratio: float, instruction: str
) -> PolicyObservation:
    rgbs = []
    for camera_name in CAMERA_NAMES:
        image = env.sim.render(
            height=image_size,
            width=image_size,
            camera_name=camera_name,
            depth=False,
            segmentation=False,
        )
        rgbs.append(np.ascontiguousarray(image[::-1]))

    robot = env.robots[0]
    proprioception = np.concatenate(
        [
            robot.get_robot_joint_positions(),
            robot.get_gripper_joint_positions("right")[0:1],
        ]
    ).astype(np.float32)
    images = [center_crop(image, crop_ratio) for image in rgbs]
    raw = env._get_observations(force_update=True)
    robot_state = {
        "state.end_effector_position_relative": np.asarray(
            raw["robot0_base_to_eef_pos"], dtype=np.float32
        ),
        "state.end_effector_rotation_relative": np.asarray(
            raw["robot0_base_to_eef_quat"], dtype=np.float32
        ),
        "state.gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        "state.base_position": np.asarray(raw["robot0_base_pos"], dtype=np.float32),
        "state.base_rotation": np.asarray(raw["robot0_base_quat"], dtype=np.float32),
    }
    return PolicyObservation(
        images=images,
        instruction=instruction,
        xiaomi_state=proprioception,
        robot_state=robot_state,
    )


def clone_observation(observation: PolicyObservation) -> PolicyObservation:
    """Detach an asynchronous request from the simulator's mutable frame."""
    return PolicyObservation(
        images=[image.copy() for image in observation.images],
        instruction=observation.instruction,
        xiaomi_state=observation.xiaomi_state.copy(),
        robot_state={key: value.copy() for key, value in observation.robot_state.items()},
    )


def make_action(env: Any, policy_action: np.ndarray) -> np.ndarray:
    """Map a policy action vector onto the environment's dense action.

    RoboCasa's ``PandaOmron`` robot exposes a 12D action:
    ``[0:6]`` end-effector pose, ``[6:7]`` gripper, ``[7:11]`` base motion and
    ``[11:12]`` control mode (which selects base-vs-arm tracking).  Models that
    predict only the first 7 dims therefore lose base control and the control
    mode entirely, so whatever the model returns is copied verbatim and only the
    remainder is left at zero.

    The two binary channels follow RoboCasa's own convention: the gym wrapper
    (``PandaOmronKeyConverter.unmap_action``) thresholds ``gripper_close`` and
    ``control_mode`` at 0.5 and emits ``-1``/``+1``.  This matters because the
    gripper controller uses ``np.sign(action)`` -- a raw ``0`` freezes the
    fingers -- and checkpoints that emit {0, 1} (GR00T) would otherwise never
    close the gripper.
    """
    action_spec = env.action_spec
    action = np.zeros_like(action_spec[0], dtype=np.float32)
    policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    width = min(policy_action.size, action.size)
    action[:width] = policy_action[:width]
    if width > GRIPPER_DIM:
        action[GRIPPER_DIM] = -1.0 if action[GRIPPER_DIM] < 0.5 else 1.0
    if width > CONTROL_MODE_DIM:
        action[CONTROL_MODE_DIM] = -1.0 if action[CONTROL_MODE_DIM] < 0.5 else 1.0
    return action


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return values


def parse_rtc_list(value: str) -> list[bool]:
    values = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized in {"on", "true", "1"}:
            values.append(True)
        elif normalized in {"off", "false", "0"}:
            values.append(False)
        elif normalized:
            raise argparse.ArgumentTypeError("RTC values must be on/off")
    if not values:
        raise argparse.ArgumentTypeError("--rtc cannot be empty")
    return values


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == total else min(1.0, center + margin)
    return low, high


def blend_plans(old_plan: deque[np.ndarray], new_plan: np.ndarray, blend_steps: int) -> deque[np.ndarray]:
    """Blend the beginning of a newly aligned chunk with queued actions."""
    result = deque(old_plan)
    overlap = min(len(result), len(new_plan), blend_steps)
    for index in range(overlap):
        alpha = float(index + 1) / float(overlap)
        result[index] = (1.0 - alpha) * result[index] + alpha * new_plan[index]
    result.extend(new_plan[overlap:])
    return result


def install_completed_inference(
    pending: PendingInference,
    step: int,
    plan: deque[np.ndarray],
    rtc: bool,
    replan_steps: int,
    blend_steps: int,
) -> tuple[deque[np.ndarray], int, int, int, float, float]:
    """Install a response and return plan, age, stale, replaced, and latencies."""
    chunk, model_latency_ms, total_latency_ms = pending.future.result()
    age = max(0, step - pending.requested_step)
    if rtc:
        stale = min(age, len(chunk))
        aligned = chunk[age:]
        if len(aligned):
            plan = blend_plans(plan, aligned[:replan_steps], blend_steps)
        replaced = 0
    else:
        stale = 0
        replaced = len(plan)
        plan = deque(chunk[:replan_steps])
    return plan, age, stale, replaced, model_latency_ms, total_latency_ms


def inference_job(
    policy: ActionChunkPolicy,
    observation: PolicyObservation,
    delay_ms: int,
    delay_domain: str = "sim",
) -> tuple[np.ndarray, float, float]:
    """Run one inference request.

    In ``sim`` mode the injected delay is resolved by the main loop, so the
    worker returns as soon as the model finishes.  In ``wall`` mode the worker
    sleeps for ``delay_ms`` on top of the real model latency.

    ``total_latency_ms`` is the nominal total response time (model latency plus
    the injected delay) so that it stays comparable across models in both
    domains.  In ``sim`` mode the realized instant also includes any blocking
    wait, which is reported separately as ``mean_block_wait_ms``.
    """
    started = time.perf_counter()
    chunk = policy.infer(observation)
    model_latency_ms = (time.perf_counter() - started) * 1000.0
    if delay_ms and delay_domain == "wall":
        time.sleep(delay_ms / 1000.0)
        total_latency_ms = (time.perf_counter() - started) * 1000.0
    elif delay_ms:
        total_latency_ms = model_latency_ms + delay_ms
    else:
        total_latency_ms = (time.perf_counter() - started) * 1000.0
    return chunk, model_latency_ms, total_latency_ms


def resolve_task(task_name: str) -> tuple[str, dict[str, Any], int | None]:
    """Map a task name (possibly an alias) to env name, env kwargs, and horizon."""
    alias = TASK_ALIASES.get(task_name)
    if alias is None:
        return task_name, {}, None
    return alias["env_name"], dict(alias["env_kwargs"]), alias["horizon"]


def _video_font(size: int = 15):
    """Use matplotlib's bundled DejaVu so annotation does not depend on the host."""
    from PIL import ImageFont

    candidates = [
        Path(matplotlib_fonts_dir()) / "DejaVuSans.ttf",
        Path(matplotlib_fonts_dir()) / "DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def matplotlib_fonts_dir() -> str:
    import matplotlib

    return str(Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf")


class VideoRecorder:
    """Write one mp4 per episode, annotated with the condition being replayed.

    The overlay is deliberately explicit about *why* a frame looks the way it
    does: the injected delay, the replan period and whether the executed action
    came from the model plan or from the fallback replay are burned in, so a
    viewer can correlate the visible behaviour with the mechanism rather than
    guessing from the numbers alone.
    """

    def __init__(self, path: Path, camera: str, width: int, height: int, fps: float, label: str):
        import imageio

        self.path = path
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.label = label
        self.frames = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=self.fps, macro_block_size=8)
        self.font = _video_font()

    def append(self, env: Any, step: int, status: str, success: bool = False) -> None:
        from PIL import Image, ImageDraw

        frame = np.asarray(
            env.sim.render(height=self.height, width=self.width, camera_name=self.camera)[::-1],
            dtype=np.uint8,
        ).copy()
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        header = f"{self.label}  |  step {step}"
        footer = f"action: {status}" + ("  |  SUCCESS" if success else "")
        for text, y, colour in ((header, 8, (255, 255, 255)), (footer, self.height - 26, None)):
            if colour is None:
                colour = (120, 255, 120) if success else (
                    (255, 210, 120) if status.startswith("fallback") else (255, 255, 255)
                )
            box = draw.textbbox((10, y), text, font=self.font)
            draw.rectangle(
                (box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3), fill=(0, 0, 0)
            )
            draw.text((10, y), text, font=self.font, fill=colour)
        self.writer.append_data(np.asarray(image, dtype=np.uint8))
        self.frames += 1

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            self.writer = None


def load_replay_rows(path: Path) -> list[dict[str, Any]]:
    """Read the recorded tuples needed to re-run a finished experiment."""
    candidates = [path / "episodes.csv", path / "episodes.jsonl"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".csv":
            with candidate.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        else:
            rows = [
                json.loads(line)
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if rows:
            source = candidate
            break
    else:
        raise FileNotFoundError(f"no episodes.csv/jsonl under {path}")
    usable = [row for row in rows if not row.get("error")]
    skipped = len(rows) - len(usable)
    if skipped:
        logging.info("replay: skipping %d errored episode(s) in %s", skipped, source)
    return usable


def condition_from_row(row: dict[str, Any], control_frequency: float) -> Condition:
    """Rebuild the exact condition an episode was run under."""
    delay_ms = int(row["delay_ms"])
    return Condition(
        delay_ms=delay_ms,
        replan_steps=int(row["replan_steps"]),
        rtc=str(row["rtc"]) in {"True", "true"},
        delay_steps=int(row.get("delay_steps") or delay_in_steps(delay_ms, control_frequency)),
        delay_domain=row.get("delay_domain") or "sim",
    )


def video_name(task: str, condition: Condition, episode: int, seed: int, layout_id, style_id) -> str:
    scene = f"L{layout_id}S{style_id}" if layout_id is not None else "scene"
    return (
        f"{task}__{condition.name}__ep{episode:03d}_seed{seed:03d}_{scene}.mp4"
    )


def pin_scene(
    seed: int,
    layout_and_style_ids: tuple[tuple[int, int], ...] | None,
    episode_index: int = 0,
) -> dict[str, Any]:
    """Choose the (layout, style) for an episode deterministically.

    ``Kitchen._setup_model`` samples ``self.rng.choice(self.layout_and_style_ids)``
    every time it runs, and ``_load_model`` re-runs it whenever a fixture
    placement fails and the model is rebuilt.  Because each retry advances
    ``self.rng``, the *final* scene depends on how many rebuilds happened, which
    in turn depends on interpreter state -- so the same seed landed on different
    layouts in different runs of the same sweep.

    Pinning the pair up front (via ``layout_ids``/``style_ids``) makes the scene
    a pure function of (seed, episode) and removes the rebuild dependency.
    """
    if not layout_and_style_ids:
        return {}
    choices = list(layout_and_style_ids)
    index = (seed + episode_index) % len(choices)
    layout_id, style_id = choices[index]
    return {"layout_ids": layout_id, "style_ids": style_id}


def resolve_scene_parameters(args: argparse.Namespace) -> tuple[str | None, str, tuple[tuple[int, int], ...] | None]:
    """Return split, object split, and optional fixed layout/style ids."""
    if args.benchmark_manifest:
        manifest = load_manifest(args.benchmark_manifest)
        if args.scene_set in manifest["scene_sets"]:
            scene_set = get_scene_set(manifest, args.scene_set)
            return (
                scene_set.get("split"),
                scene_set["object_split"],
                scene_set["layout_and_style_ids"],
            )

    if args.scene_set == "legacy5":
        return None, args.object_split, LEGACY_EVAL_SCENES
    return args.scene_set, args.object_split, None


def resolve_replay_scene_mode(
    seed: int,
    replay_scene: tuple[int, int] | None,
    layout_and_style_ids: tuple[tuple[int, int], ...] | None,
    mode: str,
) -> str:
    """Choose whether a replay needs the recorded sampling RNG path.

    Older runs sampled from the whole scene list. Pinning only their final
    layout/style pair changes the RNG state before object placement, which can
    produce a different episode despite the same visible kitchen. Newer runs
    use ``pin_scene`` directly. ``auto`` recognizes the latter when the stored
    pair matches the deterministic pin mapping.
    """
    if mode != "auto" or replay_scene is None or not layout_and_style_ids:
        return mode
    pinned = pin_scene(seed, layout_and_style_ids)
    pinned_scene = (pinned["layout_ids"], pinned["style_ids"])
    return "pinned" if pinned_scene == replay_scene else "sampled"


def run_episode(
    policy: ActionChunkPolicy,
    task_name: str,
    condition: Condition,
    episode: int,
    seed: int,
    args: argparse.Namespace,
    horizon: int,
    replay_scene: tuple[int, int] | None = None,
) -> EpisodeResult:
    """Run one episode.

    ``replay_scene`` verifies the kitchen against the saved ``(layout, style)``
    pair. Legacy runs can replay through the original scene-sampling path so
    their placement RNG state is preserved.
    """
    started = time.perf_counter()
    env = None
    recorder: VideoRecorder | None = None
    executor: ThreadPoolExecutor | None = None
    pending: PendingInference | None = None
    inference_requests = 0
    arrival_ages: list[int] = []
    stale_actions = 0
    replaced_actions = 0
    fallback_steps = 0
    layout_id = None
    style_id = None
    model_latencies: list[float] = []
    total_latencies: list[float] = []
    block_waits_ms: list[float] = []
    try:
        from robocasa.utils.env_utils import create_env

        env_name, alias_kwargs, _alias_horizon = resolve_task(task_name)
        env_kwargs = dict(
            env_name=env_name,
            robots="PandaOmron",
            camera_names=list(CAMERA_NAMES),
            camera_widths=args.image_size,
            camera_heights=args.image_size,
            seed=seed,
            render_onscreen=False,
            randomize_cameras=False,
        )
        scene_split, object_split, layout_and_style_ids = resolve_scene_parameters(args)
        replay_scene_mode = resolve_replay_scene_mode(
            seed,
            replay_scene,
            layout_and_style_ids,
            getattr(args, "replay_scene_mode", "auto"),
        )
        if layout_and_style_ids is not None:
            env_kwargs.update(
                split=scene_split,
                obj_instance_split=object_split,
            )
            if replay_scene is not None and replay_scene_mode == "sampled":
                env_kwargs["layout_and_style_ids"] = list(layout_and_style_ids)
            elif replay_scene is not None:
                env_kwargs.update(pin_scene(seed, (tuple(replay_scene),)))
            else:
                # Pin normal benchmark runs so fixture-placement retries cannot
                # move the episode to another kitchen.
                env_kwargs.update(pin_scene(seed, layout_and_style_ids))
        elif getattr(args, "scene_selection", "pinned") == "sampled":
            # Historical wall-domain experiments delegated scene selection to
            # RoboCasa's seeded RNG. Retain that mode solely for extensions of
            # those archived sweeps; new benchmarks should keep the default
            # pinned mapping for exact pairing across conditions.
            env_kwargs["layout_and_style_ids"] = list(layout_and_style_ids)
        else:
            env_kwargs["split"] = scene_split
        env_kwargs.update(alias_kwargs)
        env = create_env(**env_kwargs)
        env.reset()
        layout_id = int(env.layout_id)
        style_id = int(env.style_id)
        if replay_scene is not None and (layout_id, style_id) != replay_scene:
            raise RuntimeError(
                "replay scene mismatch: expected "
                f"{replay_scene}, got {(layout_id, style_id)} "
                f"using {replay_scene_mode} mode"
            )
        instruction = env.get_ep_meta()["lang"]
        observation = render_observation(
            env, args.image_size, args.crop_ratio, instruction
        )

        executor = ThreadPoolExecutor(max_workers=1)
        initial_future = executor.submit(
            inference_job,
            policy,
            clone_observation(observation),
            condition.delay_ms,
            condition.delay_domain,
        )
        chunk, model_ms, total_ms = initial_future.result()
        model_latencies.append(model_ms)
        total_latencies.append(total_ms)

        if getattr(args, "record_video", False):
            video_dir = Path(args.video_dir) if args.video_dir else Path(args.output_dir) / "videos"
            recorder = VideoRecorder(
                video_dir / video_name(task_name, condition, episode, seed, layout_id, style_id),
                camera=args.video_camera,
                width=args.video_width,
                height=args.video_height,
                fps=args.control_frequency or 20.0,
                label=(
                    f"{args.model_name} | {task_name} | delay {condition.delay_ms}ms "
                    f"({condition.delay_steps} steps) | replan {condition.replan_steps} | "
                    f"rtc {condition.rtc} | ep {episode} seed {seed} | L{layout_id}S{style_id}"
                ),
            )

        plan: deque[np.ndarray] = deque(chunk)
        # Keep the fallback action at the model's native width so a slow model
        # cannot silently drop its base/control channels on a starving step.
        last_action = np.zeros(np.asarray(chunk).shape[-1], dtype=np.float32)
        next_request_step = condition.replan_steps
        success = False
        steps = 0
        next_tick = time.perf_counter()

        def install_if_ready() -> bool:
            """Install a response once it is both due and computed.

            ``sim`` mode gates purely on the injected step count, so a fast model
            is held back until ``available_step`` and a slow model blocks the
            loop until its response arrives.  ``wall`` mode keeps the legacy
            worker-drives-availability behavior.
            """
            nonlocal pending, plan, stale_actions, replaced_actions
            if pending is None:
                return False
            if condition.delay_domain == "sim":
                if steps < pending.available_step:
                    return False
            elif not pending.future.done():
                return False

            if not pending.future.done():
                wait_started = time.perf_counter()
                pending.future.result()
                block_waits_ms.append((time.perf_counter() - wait_started) * 1000.0)
            (
                plan,
                age,
                stale,
                replaced,
                model_ms,
                total_ms,
            ) = install_completed_inference(
                pending,
                steps,
                plan,
                condition.rtc,
                condition.replan_steps,
                args.rtc_blend_steps,
            )
            arrival_ages.append(age)
            stale_actions += stale
            replaced_actions += replaced
            model_latencies.append(model_ms)
            total_latencies.append(total_ms)
            pending = None
            return True

        while steps < horizon:
            install_if_ready()

            if pending is None and steps >= next_request_step:
                request_observation = clone_observation(observation)
                available_step = steps + condition.delay_steps
                pending = PendingInference(
                    future=executor.submit(
                        inference_job,
                        policy,
                        request_observation,
                        condition.delay_ms,
                        condition.delay_domain,
                    ),
                    requested_step=steps,
                    available_step=available_step,
                )
                inference_requests += 1
                next_request_step = steps + condition.replan_steps
                # With a zero-step injected delay the response is due at this very
                # step, so install it here instead of one step later.
                install_if_ready()

            if plan:
                action = np.asarray(plan.popleft(), dtype=np.float32)
                action_source = f"plan[{len(plan)} left]"
            else:
                action = last_action.copy()
                fallback_steps += 1
                action_source = "fallback (replaying last action)"

            env.step(make_action(env, action))
            last_action = action
            steps += 1
            if recorder is not None:
                recorder.append(env, steps, action_source)
            observation = render_observation(
                env, args.image_size, args.crop_ratio, instruction
            )
            success = bool(env._check_success())
            if success:
                if recorder is not None:
                    recorder.append(env, steps, f"done - success in {steps} steps", success=True)
                break

            if args.control_frequency > 0:
                next_tick += 1.0 / args.control_frequency
                time.sleep(max(0.0, next_tick - time.perf_counter()))

        # Wait for a pending request so latency statistics and exceptions are not
        # silently lost when an episode succeeds early.
        if pending is not None:
            _chunk, model_ms, total_ms = pending.future.result()
            model_latencies.append(model_ms)
            total_latencies.append(total_ms)
        return EpisodeResult(
            task_name=task_name,
            condition=condition.name,
            delay_ms=condition.delay_ms,
            delay_domain=condition.delay_domain,
            delay_steps=condition.delay_steps,
            replan_steps=condition.replan_steps,
            rtc=condition.rtc,
            episode=episode,
            seed=seed,
            layout_id=layout_id,
            style_id=style_id,
            success=success,
            steps=steps,
            inference_requests=inference_requests + 1,
            mean_arrival_age_steps=float(np.mean(arrival_ages)) if arrival_ages else 0.0,
            max_arrival_age_steps=int(max(arrival_ages)) if arrival_ages else 0,
            stale_actions_discarded=stale_actions,
            queued_actions_replaced=replaced_actions,
            fallback_steps=fallback_steps,
            mean_model_latency_ms=float(np.mean(model_latencies)),
            mean_total_inference_latency_ms=float(np.mean(total_latencies)),
            mean_block_wait_ms=float(np.mean(block_waits_ms)) if block_waits_ms else 0.0,
            wall_time_s=time.perf_counter() - started,
        )
    except Exception as error:
        logging.exception("Episode failed: task=%s condition=%s episode=%d", task_name, condition.name, episode)
        return EpisodeResult(
            task_name=task_name,
            condition=condition.name,
            delay_ms=condition.delay_ms,
            delay_domain=condition.delay_domain,
            delay_steps=condition.delay_steps,
            replan_steps=condition.replan_steps,
            rtc=condition.rtc,
            episode=episode,
            seed=seed,
            layout_id=layout_id,
            style_id=style_id,
            success=False,
            steps=0,
            inference_requests=inference_requests,
            mean_arrival_age_steps=float(np.mean(arrival_ages)) if arrival_ages else 0.0,
            max_arrival_age_steps=int(max(arrival_ages)) if arrival_ages else 0,
            stale_actions_discarded=stale_actions,
            queued_actions_replaced=replaced_actions,
            fallback_steps=fallback_steps,
            mean_model_latency_ms=float(np.mean(model_latencies)) if model_latencies else 0.0,
            mean_total_inference_latency_ms=float(np.mean(total_latencies)) if total_latencies else 0.0,
            mean_block_wait_ms=float(np.mean(block_waits_ms)) if block_waits_ms else 0.0,
            wall_time_s=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if recorder is not None:
            recorder.close()
        if executor is not None:
            executor.shutdown(wait=True)
        if env is not None:
            env.close()


def write_outputs(output_dir: Path, results: list[EpisodeResult], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    fields = list(asdict(results[0]).keys()) if results else list(EpisodeResult.__dataclass_fields__.keys())
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = grouped.setdefault(
            result.condition,
            {
                "condition": result.condition,
                "delay_ms": result.delay_ms,
                "delay_domain": result.delay_domain,
                "delay_steps": result.delay_steps,
                "replan_steps": result.replan_steps,
                "rtc": result.rtc,
                "num_episodes": 0,
                "successes": 0,
                "errors": 0,
                "success_rate": 0.0,
                "success_rate_ci95_low": 0.0,
                "success_rate_ci95_high": 0.0,
                "mean_steps": 0.0,
                "mean_arrival_age_steps": 0.0,
                "max_arrival_age_steps": 0.0,
                "mean_stale_actions_discarded": 0.0,
                "mean_queued_actions_replaced": 0.0,
                "mean_model_latency_ms": 0.0,
                "mean_total_inference_latency_ms": 0.0,
                "mean_block_wait_ms": 0.0,
                "mean_fallback_steps": 0.0,
            },
        )
        bucket["num_episodes"] += 1
        bucket["successes"] += int(result.success)
        bucket["errors"] += int(result.error is not None)
        bucket["mean_steps"] += result.steps
        bucket["mean_arrival_age_steps"] += result.mean_arrival_age_steps
        bucket["max_arrival_age_steps"] = max(
            bucket["max_arrival_age_steps"], result.max_arrival_age_steps
        )
        bucket["mean_stale_actions_discarded"] += result.stale_actions_discarded
        bucket["mean_queued_actions_replaced"] += result.queued_actions_replaced
        bucket["mean_model_latency_ms"] += result.mean_model_latency_ms
        bucket["mean_total_inference_latency_ms"] += result.mean_total_inference_latency_ms
        bucket["mean_block_wait_ms"] += result.mean_block_wait_ms
        bucket["mean_fallback_steps"] += result.fallback_steps

    for bucket in grouped.values():
        n = bucket["num_episodes"]
        bucket["success_rate"] = bucket["successes"] / n if n else 0.0
        ci_low, ci_high = wilson_interval(bucket["successes"], n)
        bucket["success_rate_ci95_low"] = ci_low
        bucket["success_rate_ci95_high"] = ci_high
        for key in (
            "mean_steps",
            "mean_arrival_age_steps",
            "mean_stale_actions_discarded",
            "mean_queued_actions_replaced",
            "mean_model_latency_ms",
            "mean_total_inference_latency_ms",
            "mean_block_wait_ms",
            "mean_fallback_steps",
        ):
            bucket[key] /= n

    summary = {
        "experiment": "latency_benchmark",
        "model": asdict(get_model_spec(args.model_name)),
        "semantics": {
            "action_chunk_length": args.action_chunk_length,
            "legacy_eval_scenes": [list(scene) for scene in LEGACY_EVAL_SCENES],
            "delay_domain": args.delay_domain,
            "delay_domain_sim": (
                "response becomes usable exactly delay_steps control steps after "
                "the request; model latency is recorded but does not affect staleness"
            ),
            "delay_domain_wall": (
                "legacy: worker sleeps for delay_ms on top of real model latency, "
                "so staleness depends on model latency + injected delay"
            ),
            "rtc_off": "replace queued plan with response chunk at index zero",
            "rtc_on": "align response to arrival step and linearly blend into queued plan",
        },
        "args": vars(args),
        "conditions": list(grouped.values()),
        "num_results": len(results),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    condition_rows = list(grouped.values())
    if condition_rows:
        with (output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(condition_rows[0]))
            writer.writeheader()
            writer.writerows(condition_rows)


@dataclass
class ReplayJob:
    task_name: str
    condition: Condition
    episode: int
    seed: int
    layout_id: int | None
    style_id: int | None


def discover_slices(path: Path) -> list[Path]:
    """Find every directory that holds an ``episodes.csv``/``episodes.jsonl``.

    Accepts a single slice directory (``<root>/<model>/<task>``), a whole model
    directory, or the entire sweep root, so one flag re-renders any level.
    """
    if (path / "episodes.csv").exists() or (path / "episodes.jsonl").exists():
        return [path]
    found = {entry.parent for pattern in ("episodes.csv", "episodes.jsonl")
             for entry in path.rglob(pattern)}
    # ``replay_run_video.py`` stages its filtered selection in a ``_``-prefixed
    # directory; skipping those keeps a second pass from re-reading the copy of
    # a run it just wrote.
    return sorted(entry for entry in found if not entry.name.startswith("_"))


def detect_model(path: Path) -> str | None:
    """Read the model a run was produced with from its own summary.json."""
    for summary in sorted(path.rglob("summary.json")):
        try:
            model = json.loads(summary.read_text(encoding="utf-8")).get("model", {})
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(model, dict) and model.get("name"):
            return model["name"]
    return None


def build_replay_jobs(path: Path, control_frequency: float, limit: int | None = None) -> list[ReplayJob]:
    """Recreate the exact (task, condition, seed, scene) tuples of a past run.

    The stored ``layout_id``/``style_id`` is the scene the original episode
    actually used, so pinning it keeps a replay on that same kitchen instead of
    relying on the seed mapping, which is what the original sweep did not do.
    """
    slices = discover_slices(path)
    if not slices:
        raise ValueError(f"no episodes.csv/jsonl found under {path}")
    jobs: list[ReplayJob] = []
    for slice_dir in slices:
        for row in load_replay_rows(slice_dir):
            if not row.get("task_name"):
                continue
            scene = None
            if row.get("layout_id") not in (None, "", "None") and row.get("style_id") not in (None, "", "None"):
                scene = (int(row["layout_id"]), int(row["style_id"]))
            jobs.append(
                ReplayJob(
                    task_name=row["task_name"],
                    condition=condition_from_row(row, control_frequency),
                    episode=int(row.get("episode") or 0),
                    seed=int(row.get("seed") or 0),
                    layout_id=scene[0] if scene else None,
                    style_id=scene[1] if scene else None,
                )
            )
    if limit is not None:
        jobs = jobs[:limit]
    if not jobs:
        raise ValueError(f"no replayable episodes found under {path}")
    logging.info("replay: %d slice(s), %d episode(s)", len(slices), len(jobs))
    return jobs


def replay_episodes(policy: ActionChunkPolicy, jobs: list[ReplayJob], args: argparse.Namespace) -> list[EpisodeResult]:
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if not args.record_video:
        logging.warning("--replay-from without --record-video only re-runs the episodes")
    results: list[EpisodeResult] = []
    for index, job in enumerate(jobs, start=1):
        _env_name, _kwargs, alias_horizon = resolve_task(job.task_name)
        horizon = args.horizon or alias_horizon or get_task_horizon(job.task_name)
        scene = None if job.layout_id is None else (job.layout_id, job.style_id)
        logging.info(
            "Replay %d/%d task=%s episode=%d seed=%d %s scene=%s",
            index, len(jobs), job.task_name, job.episode, job.seed, job.condition.name, scene,
        )
        results.append(
            run_episode(
                policy, job.task_name, job.condition, job.episode, job.seed, args,
                horizon, replay_scene=scene,
            )
        )
        if results[-1].error is not None:
            raise RuntimeError(
                "replay failed for "
                f"{job.task_name} episode={job.episode} seed={job.seed}: "
                f"{results[-1].error}"
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        choices=("xiaomi", "gr00t_n1_5", "pi0_5", "pi0", "diffusion_policy"),
        default=None,
        help="Defaults to 'xiaomi'; with --replay-from it is auto-detected from the run.",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-env", default=None, help="Override the isolated worker environment.")
    parser.add_argument("--task-name", action="append", default=None)
    parser.add_argument("--task-set", default=None, help="Task set name from the benchmark manifest.")
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_MANIFEST),
        help="JSON manifest containing task sets and fixed scene sets.",
    )
    parser.add_argument(
        "--scene-set",
        default="legacy5",
        help=(
            "Scene distribution. Names from --benchmark-manifest select fixed "
            "scene sets; legacy5 matches the local Xiaomi evaluator."
        ),
    )
    parser.add_argument(
        "--scene-selection",
        choices=("pinned", "sampled"),
        default="pinned",
        help=(
            "Use a deterministic pinned scene per seed (default), or RoboCasa's "
            "legacy seeded sampling. The latter is only for extending archived "
            "wall-domain sweeps."
        ),
    )
    parser.add_argument(
        "--object-split",
        choices=("pretrain", "target"),
        default="pretrain",
        help="Object split used with --scene-set legacy5.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument(
        "--delays-ms",
        "--inference-delays-ms",
        dest="delays_ms",
        default="0,100,300,500",
    )
    parser.add_argument(
        "--delay-domain",
        choices=("sim", "wall"),
        default="sim",
        help=(
            "How the configured delay is realized.  'sim' (default) converts it "
            "to control steps so the injected delay is the only source of "
            "staleness; 'wall' keeps the legacy sleep-based semantics."
        ),
    )
    parser.add_argument(
        "--replan-steps-list",
        "--replan-steps",
        dest="replan_steps_list",
        default="1,5,10",
    )
    parser.add_argument("--rtc", default="off,on")
    parser.add_argument("--rtc-blend-steps", type=int, default=4)
    parser.add_argument("--control-frequency", type=float, default=20.0)
    parser.add_argument(
        "--delay-step-frequency",
        type=float,
        default=None,
        help=(
            "Frequency used only to convert sim-domain milliseconds into delay "
            "steps. Defaults to --control-frequency, so a fast replay can use "
            "--control-frequency 0 --delay-step-frequency 20."
        ),
    )
    parser.add_argument("--output-dir", default="eval_results/xiaomi_latency")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Write one annotated mp4 per episode into --video-dir.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Where mp4s go.  Defaults to <output-dir>/videos.",
    )
    parser.add_argument("--video-camera", default=VIDEO_CAMERA)
    parser.add_argument("--video-width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument(
        "--replay-from",
        default=None,
        help=(
            "Directory of a finished run (containing episodes.csv/jsonl). The "
            "recorded tasks, conditions, seeds, and scenes are restored; combine "
            "with --record-video to render the rerun."
        ),
    )
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=None,
        help="Cap the number of replayed episodes (useful for a quick look).",
    )
    parser.add_argument(
        "--replay-scene-mode",
        choices=("auto", "pinned", "sampled"),
        default="auto",
        help=(
            "How to reconstruct the recorded scene: auto selects pinned when "
            "the stored pair matches pin_scene; sampled restores legacy RNG."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(args.benchmark_manifest)
    if args.task_set and args.task_name:
        raise ValueError("use either --task-set or --task-name, not both")
    if args.task_set:
        args.task_name = get_task_set(manifest, args.task_set)
    else:
        args.task_name = args.task_name or ["PickPlaceCounterToCabinet"]
    if args.replay_from and args.model_name is None:
        detected = detect_model(Path(args.replay_from))
        if detected:
            logging.info("replay: using model %r recorded in %s", detected, args.replay_from)
            args.model_name = detected
    args.model_name = args.model_name or "xiaomi"
    model_spec = get_model_spec(args.model_name)
    if args.model_env:
        model_spec = replace(model_spec, runtime_env=args.model_env)
    if not args.dry_run:
        require_implemented_model(args.model_name)
    if args.model_path is None:
        args.model_path = model_spec.checkpoint
    if args.scene_set not in manifest["scene_sets"] and args.scene_set in {"pretrain20", "target10"}:
        raise ValueError(f"scene set {args.scene_set!r} is not defined in {args.benchmark_manifest}")
    delays = parse_int_list(args.delays_ms, "--delays-ms")
    replans = parse_int_list(args.replan_steps_list, "--replan-steps-list")
    rtcs = parse_rtc_list(args.rtc)
    if args.episodes < 1 or args.image_size < 1 or args.rtc_blend_steps < 1:
        raise ValueError("episodes, image-size, and rtc-blend-steps must be positive")
    if args.diffusion_steps < 1:
        raise ValueError("diffusion-steps must be positive")
    if args.horizon is not None and args.horizon < 1:
        raise ValueError("horizon must be positive")
    if any(delay < 0 for delay in delays):
        raise ValueError("injected delays must be non-negative")
    delay_step_frequency = args.delay_step_frequency
    if delay_step_frequency is None:
        delay_step_frequency = args.control_frequency
    if delay_step_frequency < 0:
        raise ValueError("delay-step-frequency must be non-negative")
    if args.delay_domain == "sim" and 0.0 < delay_step_frequency < 1.0:
        raise ValueError("delay-step-frequency must be >= 1 Hz for sim-domain delays")
    if args.delay_domain == "sim" and 0.0 < delay_step_frequency <= 1000.0:
        step_ms = 1000.0 / delay_step_frequency
        off_grid = [delay for delay in delays if delay % step_ms > 1e-9 and (step_ms - delay % step_ms) % step_ms > 1e-9]
        if off_grid:
            logging.warning(
                "Delays %s are not multiples of the %.3f ms control period; "
                "each is rounded to the nearest control step.",
                off_grid,
                step_ms,
            )
    args.action_chunk_length = model_spec.action_chunk_length
    if any(replan < 1 for replan in replans):
        raise ValueError(f"replan steps must be positive; got {replans}")

    conditions = [
        Condition(
            delay_ms=delay,
            replan_steps=replan,
            rtc=rtc,
            delay_steps=delay_in_steps(delay, delay_step_frequency),
            delay_domain=args.delay_domain,
        )
        for delay in delays
        for replan in replans
        for rtc in rtcs
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Conditions: %s", ", ".join(condition.name for condition in conditions))
    if args.dry_run:
        for condition in conditions:
            print(condition.name)
        return

    from robocasa.utils.dataset_registry_utils import get_task_horizon

    policy = create_policy(
        model_spec,
        model_path=args.model_path,
        model_seed=args.model_seed,
        diffusion_steps=args.diffusion_steps,
        attn_implementation=args.attn_implementation,
    )
    output_dir = Path(args.output_dir)
    if args.replay_from:
        try:
            jobs = build_replay_jobs(
                Path(args.replay_from), args.control_frequency, limit=args.replay_limit
            )
            logging.info(
                "Replaying %d episode(s) from %s%s",
                len(jobs), args.replay_from,
                " with video" if args.record_video else "",
            )
            replayed = replay_episodes(policy, jobs, args)
            write_outputs(output_dir, replayed, args)
            for result in replayed:
                logging.info(
                    "Replayed %s ep=%d delay=%dms success=%s steps=%d",
                    result.task_name, result.episode, result.delay_ms, result.success, result.steps,
                )
            logging.info("Replay wrote %d video(s) to %s", len(replayed),
                         Path(args.video_dir) if args.video_dir else f"{args.output_dir}/videos")
        finally:
            policy.close()
        return

    results: list[EpisodeResult] = []
    try:
        for condition in conditions:
            for task_name in args.task_name:
                _env_name, _kwargs, alias_horizon = resolve_task(task_name)
                horizon = args.horizon or alias_horizon or get_task_horizon(task_name)
                for episode in range(args.episodes):
                    seed = args.seed + episode
                    logging.info("Running %s task=%s episode=%d/%d", condition.name, task_name, episode + 1, args.episodes)
                    result = run_episode(
                        policy, task_name, condition, episode, seed, args, horizon
                    )
                    results.append(result)
                    write_outputs(output_dir, results, args)
                    logging.info("Result: success=%s steps=%d wall=%.2fs", result.success, result.steps, result.wall_time_s)
    finally:
        policy.close()
    write_outputs(output_dir, results, args)
    logging.info("Wrote %d episode results to %s", len(results), args.output_dir)


if __name__ == "__main__":
    main()

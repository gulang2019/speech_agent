"""RoboDojo <-> Xiaomi compatibility boundary.

RoboDojo and the local RoboCasa runner intentionally use different contracts.
This module keeps that difference explicit so a RoboDojo observation can be
fed through the existing Xiaomi ``PolicyClient`` without changing the
RoboCasa evaluator or the model checkpoint.

The adapter is deliberately conservative:

* only the documented v1 observation shape is accepted;
* camera aliases are additive, never positional;
* the active arm and action semantics are configurable;
* unknown future data-format versions fail closed instead of silently
  changing the robot command.

The local RoboCasa checkpoint is a single-arm ``robocasa_mg`` model. The
default bridge therefore controls the RoboDojo right arm and explicitly holds
the other arm at its observed target. This is a compatibility/evaluation bridge, not a claim that a
RoboCasa-trained checkpoint is equivalent to a RoboDojo-trained checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    from scripts.latency_model_adapters import PolicyObservation
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from latency_model_adapters import PolicyObservation


ROBODOJO_V1 = "v1.0"
ROBODOJO_TASK = "pour_liquid_into_cup"
ROBODOJO_INSTRUCTION = "Pour the liquid from the bottle into the cup."


@dataclass(frozen=True)
class RoboDojoBridgeConfig:
    """Stable bridge settings for a single-arm Xiaomi checkpoint."""

    active_arm: str = "right"
    action_mode: str = "delta_ee"
    delta_frame: str = "base"
    action_units: str = "robocasa_normalized"
    gripper_mode: str = "auto"
    position_scale: float = 0.05
    rotation_scale: float = 0.5
    base_rotation_quat_wxyz: tuple[float, float, float, float] = (0.707, 0.0, 0.0, 0.707)
    image_size: int | None = None
    crop_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.active_arm not in {"left", "right"}:
            raise ValueError("active_arm must be 'left' or 'right'")
        if self.action_mode not in {"delta_ee", "absolute_ee"}:
            raise ValueError("action_mode must be 'delta_ee' or 'absolute_ee'")
        if self.delta_frame not in {"ee", "base", "world"}:
            raise ValueError("delta_frame must be 'ee', 'base', or 'world'")
        if self.action_units not in {"robocasa_normalized", "physical"}:
            raise ValueError("action_units must be 'robocasa_normalized' or 'physical'")
        if self.gripper_mode not in {"auto", "zero_one", "minus_one_one"}:
            raise ValueError(
                "gripper_mode must be 'auto', 'zero_one', or 'minus_one_one'"
            )
        if self.image_size is not None and self.image_size < 1:
            raise ValueError("image_size must be positive")
        if not 0 < self.crop_ratio <= 1:
            raise ValueError("crop_ratio must be in (0, 1]")
        if self.position_scale <= 0 or self.rotation_scale <= 0:
            raise ValueError("position_scale and rotation_scale must be positive")
        _as_float_array(self.base_rotation_quat_wxyz, name="base_rotation_quat_wxyz", size=4)


def _as_float_array(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} must have {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _image_from_camera(vision: Mapping[str, Any], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        camera = vision.get(name)
        if camera is None:
            continue
        if isinstance(camera, Mapping):
            for key in ("color", "colors", "rgb"):
                if key in camera:
                    image = camera[key]
                    break
            else:
                continue
        else:
            image = camera
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"camera {name!r} must be HWC or CHW, got {image.shape}")
        if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"camera {name!r} is not an RGB image: {image.shape}")
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = np.asarray(image, dtype=np.uint8)
        return np.ascontiguousarray(image)
    raise KeyError(f"none of the camera aliases are present: {names}")


def _prepare_image(image: np.ndarray, config: RoboDojoBridgeConfig) -> Image.Image:
    pil = Image.fromarray(image)
    if config.crop_ratio < 1.0:
        width, height = pil.size
        crop_width = max(1, int(width * config.crop_ratio))
        crop_height = max(1, int(height * config.crop_ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        pil = pil.crop((left, top, left + crop_width, top + crop_height))
    if config.image_size is not None:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        pil = pil.resize((config.image_size, config.image_size), resampling)
    return pil


def _arm_state(obs: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = obs.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("RoboDojo observation is missing a state mapping")
    joints = _as_float_array(state.get(f"{arm}_arm_joint_state"), name=f"{arm}_arm_joint_state")
    if joints.size not in {6, 7}:
        raise ValueError(f"{arm}_arm_joint_state must contain 6 or 7 values, got {joints.size}")
    gripper = _as_float_array(state.get(f"{arm}_ee_joint_state"), name=f"{arm}_ee_joint_state")
    pose = _as_float_array(state.get(f"{arm}_ee_pose"), name=f"{arm}_ee_pose", size=7)
    if joints.size == 7:
        joints = joints[:6]
    if gripper.size < 1:
        raise ValueError(f"{arm}_ee_joint_state must contain at least one value")
    return joints, gripper[:1], pose


def validate_robodojo_observation(obs: Mapping[str, Any]) -> None:
    """Validate the stable subset consumed by this bridge."""

    version = obs.get("data_format_version", ROBODOJO_V1)
    if version != ROBODOJO_V1:
        raise ValueError(
            f"unsupported RoboDojo observation version {version!r}; "
            f"supported versions: {ROBODOJO_V1!r}"
        )
    vision = obs.get("vision")
    if not isinstance(vision, Mapping):
        raise ValueError("RoboDojo observation is missing vision")
    for aliases in (
        ("cam_head", "cam_high", "head_camera"),
        ("cam_left_wrist", "left_camera", "wrist_left"),
        ("cam_right_wrist", "right_camera", "wrist_right"),
    ):
        _image_from_camera(vision, aliases)
    _arm_state(obs, "left")
    _arm_state(obs, "right")


def robodojo_to_policy_observation(
    obs: Mapping[str, Any], config: RoboDojoBridgeConfig | None = None
) -> PolicyObservation:
    """Convert one RoboDojo v1 observation to the local policy contract."""

    config = config or RoboDojoBridgeConfig()
    validate_robodojo_observation(obs)
    arm = config.active_arm
    joints, gripper, pose = _arm_state(obs, arm)
    vision = obs["vision"]
    head = _image_from_camera(vision, ("cam_head", "cam_high", "head_camera"))
    wrist = _image_from_camera(
        vision,
        ("cam_right_wrist", "right_camera", "wrist_right")
        if arm == "right"
        else ("cam_left_wrist", "left_camera", "wrist_left"),
    )
    images = [
        _prepare_image(head, config),
        _prepare_image(head, config),
        _prepare_image(wrist, config),
    ]
    instruction = obs.get("instruction", ROBODOJO_INSTRUCTION)
    if isinstance(instruction, list):
        instruction = instruction[0] if instruction else ROBODOJO_INSTRUCTION
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = ROBODOJO_INSTRUCTION

    # Keep the canonical state small and let the existing Xiaomi adapter pad it
    # to 60 dimensions. The full RoboDojo state remains available for action
    # conversion in robot_state.
    robot_state = {
        "robodojo.active_arm": np.asarray([0 if arm == "left" else 1], dtype=np.float32),
        "robodojo.active_ee_pose": pose.astype(np.float32),
        "robodojo.left_ee_pose": _arm_state(obs, "left")[2].astype(np.float32),
        "robodojo.right_ee_pose": _arm_state(obs, "right")[2].astype(np.float32),
        "robodojo.left_gripper": _arm_state(obs, "left")[1].astype(np.float32),
        "robodojo.right_gripper": _arm_state(obs, "right")[1].astype(np.float32),
    }
    return PolicyObservation(
        images=images,
        instruction=instruction.strip(),
        xiaomi_state=np.concatenate([joints, gripper]).astype(np.float32),
        robot_state=robot_state,
    )


def _gripper_to_zero_one(value: float, mode: str) -> float:
    if mode == "auto":
        mode = "minus_one_one" if value < -0.05 else "zero_one"
    if mode == "minus_one_one":
        value = (value + 1.0) / 2.0
    return float(np.clip(value, 0.0, 1.0))


def _pose_from_xiaomi_action(
    action: np.ndarray,
    current_pose: np.ndarray,
    config: RoboDojoBridgeConfig,
) -> np.ndarray:
    action = _as_float_array(action, name="Xiaomi action", size=7)
    current_pose = _as_float_array(current_pose, name="current ee pose", size=7)
    delta = action[:6].astype(np.float64)
    if config.action_units == "robocasa_normalized":
        delta[:3] *= config.position_scale
        delta[3:6] *= config.rotation_scale
    current_pos = current_pose[:3].astype(np.float64)
    current_rot = Rotation.from_quat(current_pose[4:7].tolist() + [float(current_pose[3])])
    if config.action_mode == "absolute_ee":
        target_pos = action[:3].astype(np.float64)
        target_rot = Rotation.from_rotvec(action[3:6].astype(np.float64))
    elif config.delta_frame == "ee":
        target_pos = current_pos + current_rot.apply(delta[:3])
        target_rot = current_rot * Rotation.from_rotvec(delta[3:6])
    elif config.delta_frame == "base":
        base_quat = _as_float_array(
            config.base_rotation_quat_wxyz,
            name="base_rotation_quat_wxyz",
            size=4,
        )
        base_rot = Rotation.from_quat(base_quat[1:4].tolist() + [float(base_quat[0])])
        target_pos = current_pos + base_rot.apply(delta[:3])
        target_rot = Rotation.from_rotvec(base_rot.apply(delta[3:6])) * current_rot
    else:
        target_pos = current_pos + delta[:3]
        target_rot = Rotation.from_rotvec(delta[3:6]) * current_rot
    quat_xyzw = target_rot.as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    if quat_wxyz[0] < 0:
        quat_wxyz *= -1.0
    return np.concatenate([target_pos, quat_wxyz]).astype(np.float32)


def xiaomi_action_to_robodojo_action(
    policy_action: np.ndarray,
    obs: Mapping[str, Any],
    config: RoboDojoBridgeConfig | None = None,
) -> dict[str, np.ndarray]:
    """Convert one local 7D Xiaomi action to a complete dual-arm action dict.

    RoboDojo's ``arx_x5`` EE controller consumes both arms on every step. The
    inactive arm is therefore returned at its current pose/gripper target so
    it is explicitly held instead of being omitted from the action contract.
    """

    config = config or RoboDojoBridgeConfig()
    validate_robodojo_observation(obs)
    arm = config.active_arm
    current_pose = _arm_state(obs, arm)[2]
    action = _as_float_array(policy_action, name="policy action")
    if action.size < 7:
        raise ValueError(f"Xiaomi action must have at least 7 values, got {action.size}")
    pose = _pose_from_xiaomi_action(action[:7], current_pose, config)
    gripper = np.asarray(
        [_gripper_to_zero_one(float(action[6]), config.gripper_mode)], dtype=np.float32
    )
    action_dict: dict[str, np.ndarray] = {}
    for candidate in ("left", "right"):
        candidate_pose = _arm_state(obs, candidate)[2]
        candidate_gripper = _arm_state(obs, candidate)[1]
        if candidate == arm:
            action_dict[f"{candidate}_ee_pose"] = pose
            action_dict[f"{candidate}_ee_joint_state"] = gripper
        else:
            action_dict[f"{candidate}_ee_pose"] = candidate_pose.astype(np.float32).copy()
            action_dict[f"{candidate}_ee_joint_state"] = np.asarray(
                [_gripper_to_zero_one(float(candidate_gripper[0]), config.gripper_mode)],
                dtype=np.float32,
            )
    return action_dict


def task_metadata(task_name: str = ROBODOJO_TASK) -> dict[str, Any]:
    """Return versioned metadata needed by an external RoboDojo launcher."""

    if task_name != ROBODOJO_TASK:
        raise ValueError(f"unsupported RoboDojo task: {task_name!r}")
    return {
        "task_name": task_name,
        "task_config": task_name,
        "instruction": ROBODOJO_INSTRUCTION,
        "max_steps": 400,
        "env_cfg_type": "arx_x5",
        "action_type": "ee",
        "observation_data_format_version": ROBODOJO_V1,
        "bridge_status": "compatibility_only",
    }

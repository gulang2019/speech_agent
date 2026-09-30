"""Inspect GR00T's raw vs unnormalized action to find the gripper pathology."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from model_worker import GrootAdapter, _canonicalize_observation


def main() -> None:
    from PIL import Image

    ckpt = "models/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000"
    adapter = GrootAdapter(ckpt, "cuda")
    images = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(3)]
    obs = {
        "images": [
            {"data": np.ascontiguousarray(im).tobytes(), "shape": im.shape} for im in images
        ],
        "instruction": "pick up the object and place it in the cabinet",
        "xiaomi_state": np.zeros(14, dtype=np.float32),
        "robot_state": {
            "state.end_effector_position_relative": np.array([0.0, 0.0, 0.0], np.float32),
            "state.end_effector_rotation_relative": np.array([1, 0, 0, 0], np.float32),
            "state.gripper_qpos": np.zeros(2, np.float32),
            "state.base_position": np.array([2.65, -2.31, 0.70], np.float32),
            "state.base_rotation": np.array([1, 0, 0, 0], np.float32),
        },
    }
    canonical = _canonicalize_observation(obs)

    # Reproduce infer() but intercept the intermediate stages.
    state = canonical["robot_state"]
    imgs = canonical["images"]
    groot_obs = {
        "video.robot0_agentview_left": imgs[0][None],
        "video.robot0_agentview_right": imgs[1][None],
        "video.robot0_eye_in_hand": imgs[2][None],
        "state.end_effector_position_relative": state["state.end_effector_position_relative"][None],
        "state.end_effector_rotation_relative": state["state.end_effector_rotation_relative"][None],
        "state.gripper_qpos": state["state.gripper_qpos"][None],
        "state.base_position": state["state.base_position"][None],
        "state.base_rotation": state["state.base_rotation"][None],
        "annotation.human.task_description": np.asarray([canonical["instruction"]]),
    }
    policy = adapter.policy
    from gr00t.model.policy import unsqueeze_dict_values

    obs_copy = groot_obs.copy()
    for k, v in obs_copy.items():
        if not isinstance(v, np.ndarray):
            obs_copy[k] = np.array(v)
    batched = unsqueeze_dict_values(obs_copy)
    normalized_input = policy.apply_transforms(batched)
    normalized_action = policy._get_action_from_normalized_input(normalized_input)
    print("normalized_action shape:", tuple(normalized_action.shape))
    print("normalized action[0] (first 12):", np.round(normalized_action[0, 0, :12].cpu().numpy(), 4))
    print("normalized gripper dim6 over chunk:", np.round(normalized_action[0, :, 6].cpu().numpy(), 4))
    unnorm = policy._get_unnormalized_action(normalized_action)
    import torch

    def to_np(v):
        return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

    print("unnorm keys:", list(unnorm.keys()))
    for key, val in unnorm.items():
        arr = to_np(val)
        print(f"  {key}: shape={arr.shape} first={np.round(arr.reshape(-1)[:4], 4)}")
    joined = np.concatenate(
        [to_np(unnorm[k]).reshape(to_np(unnorm[k]).shape[1], -1)
         for k in ("action.end_effector_position", "action.end_effector_rotation",
                   "action.gripper_close", "action.base_motion", "action.control_mode")],
        axis=-1,
    )
    print("joined shape:", joined.shape)
    print("joined action[0]:", np.round(joined[0, :12], 4))
    print("gripper column over chunk:", np.round(joined[:, 6], 4))


if __name__ == "__main__":
    main()

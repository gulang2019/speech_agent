from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robodojo_compat import (
    ROBODOJO_INSTRUCTION,
    RoboDojoBridgeConfig,
    robodojo_to_policy_observation,
    task_metadata,
    validate_robodojo_observation,
    xiaomi_action_to_robodojo_action,
)
from scripts.robodojo_xiaomi_server import RoboDojoXiaomiModel


def _obs():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    return {
        "data_format_version": "v1.0",
        "instruction": ROBODOJO_INSTRUCTION,
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image + 1},
            "cam_right_wrist": {"color": image + 2},
        },
        "state": {
            "left_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": np.array([0.2]),
            "left_ee_pose": np.array([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]),
            "right_arm_joint_state": np.arange(6, dtype=np.float32),
            "right_ee_joint_state": np.array([0.3]),
            "right_ee_pose": np.array([0.1, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]),
        },
    }


def test_converts_v1_observation_to_xiaomi_contract():
    observation = robodojo_to_policy_observation(_obs())
    assert len(observation.images) == 3
    assert observation.images[0].size == (10, 8)
    np.testing.assert_allclose(observation.xiaomi_state, np.r_[np.arange(6), 0.3])
    assert observation.instruction == ROBODOJO_INSTRUCTION


def test_action_delta_is_absolute_pose_and_normalized_gripper():
    action = xiaomi_action_to_robodojo_action(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), _obs()
    )
    # RoboCasa emits normalized base-frame deltas. ARX X5's default base is
    # rotated +90 degrees around z, so +x in base becomes +y in world.
    np.testing.assert_allclose(action["right_ee_pose"][:3], [0.1, 0.25, 0.4])
    assert action["right_ee_joint_state"][0] == 1.0


def test_inactive_arm_is_explicitly_held():
    action = xiaomi_action_to_robodojo_action(
        np.zeros(7), _obs(), RoboDojoBridgeConfig(active_arm="left")
    )
    assert set(action) == {
        "left_ee_pose",
        "left_ee_joint_state",
        "right_ee_pose",
        "right_ee_joint_state",
    }
    np.testing.assert_allclose(action["right_ee_pose"], _obs()["state"]["right_ee_pose"])
    np.testing.assert_allclose(action["right_ee_joint_state"], [0.3])


class _FakePolicy:
    def infer(self, observation):
        assert len(observation.images) == 3
        assert observation.xiaomi_state.shape == (7,)
        return np.zeros((2, 7), dtype=np.float32)


def test_model_surface_returns_a_dual_arm_action_chunk():
    model = RoboDojoXiaomiModel(model_path="unused", policy=_FakePolicy())
    model.update_obs(_obs())
    actions = model.get_action()
    assert len(actions) == 2
    assert set(actions[0]) == {
        "left_ee_pose",
        "left_ee_joint_state",
        "right_ee_pose",
        "right_ee_joint_state",
    }
    np.testing.assert_allclose(actions[0]["right_ee_pose"][:3], [0.1, 0.2, 0.4])


def test_unknown_version_fails_closed():
    observation = _obs()
    observation["data_format_version"] = "v2.0"
    with pytest.raises(ValueError, match="unsupported RoboDojo observation version"):
        validate_robodojo_observation(observation)


def test_missing_camera_does_not_fall_back_to_positional_data():
    observation = _obs()
    del observation["vision"]["cam_right_wrist"]
    with pytest.raises(KeyError):
        validate_robodojo_observation(observation)


def test_task_metadata_is_explicitly_versioned():
    metadata = task_metadata()
    assert metadata["task_config"] == "pour_liquid_into_cup"
    assert metadata["observation_data_format_version"] == "v1.0"
    assert metadata["bridge_status"] == "compatibility_only"

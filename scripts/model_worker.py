"""Isolated model worker for the cross-model latency benchmark.

The worker keeps model imports and GPU memory out of the RoboCasa process.
Stdout is reserved for an 8-byte length-prefixed pickle protocol; all library
logs are redirected to stderr.
"""

from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


# Make the local compatibility modules available before any model package is
# imported.  GR00T's optional Eagle/RADIO modules import ``flash_attn`` during
# package initialization even for checkpoints that do not use them.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def send_frame(stream, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def read_frame(stream) -> Any:
    header = stream.read(8)
    if len(header) != 8:
        return None
    size = struct.unpack("!Q", header)[0]
    if size > 1 << 32:
        raise RuntimeError(f"invalid request frame size: {size}")
    payload = stream.read(size)
    if len(payload) != size:
        raise RuntimeError("truncated request frame")
    return pickle.loads(payload)


def _image_list(observation: dict[str, Any]) -> list[np.ndarray]:
    images = []
    for image in observation["images"]:
        if isinstance(image, dict) and "data" in image and "shape" in image:
            image_array = np.frombuffer(image["data"], dtype=np.uint8).reshape(tuple(image["shape"]))
        else:
            # Keep compatibility with hand-written worker requests.
            image_array = np.asarray(image, dtype=np.uint8)
        images.append(image_array)
    if len(images) != 3:
        raise ValueError(f"expected three camera images, got {len(images)}")
    return images


def _canonicalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Rebuild array values after the environment-independent wire format."""
    result = dict(observation)
    result["images"] = _image_list(observation)
    result["xiaomi_state"] = np.asarray(observation["xiaomi_state"], dtype=np.float32)
    result["robot_state"] = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in observation["robot_state"].items()
    }
    return result


class GrootAdapter:
    action_chunk_length = 16

    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        self.torch = torch
        data_config = DATA_CONFIG_MAP["panda_omron"]
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=4,
            device=device,
        )

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        images = _image_list(observation)
        state = observation["robot_state"]
        groot_obs: dict[str, Any] = {
            # GR00T expects a time axis even for a single-frame observation;
            # its policy wrapper adds the batch axis itself.
            "video.robot0_agentview_left": images[0][None],
            "video.robot0_agentview_right": images[1][None],
            "video.robot0_eye_in_hand": images[2][None],
            "state.end_effector_position_relative": state["state.end_effector_position_relative"][None],
            "state.end_effector_rotation_relative": state["state.end_effector_rotation_relative"][None],
            "state.gripper_qpos": state["state.gripper_qpos"][None],
            "state.base_position": state["state.base_position"][None],
            "state.base_rotation": state["state.base_rotation"][None],
            "annotation.human.task_description": np.asarray([observation["instruction"]]),
        }
        result = self.policy.get_action(groot_obs)
        action_keys = (
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        )
        chunks = [np.asarray(result[key], dtype=np.float32) for key in action_keys]
        action = np.concatenate(chunks, axis=-1)
        self.last_raw_dim = action.shape[-1]
        return action


class OpenPIAdapter:
    def __init__(self, model_path: str, device: str) -> None:
        import dataclasses

        from openpi.policies import policy_config
        from openpi.training import config
        from openpi.shared import normalize

        self.config = config.get_config("pi05_pretrain_human300")
        # The public RoboCasa config points at the original training dataset
        # for its assets.  It is not needed at inference time and is not
        # present on the benchmark node; the checkpoint contains the exact
        # statistics used during training.
        norm_stats = normalize.load(Path(model_path) / "assets")
        base_config = config.DataConfig(norm_stats=norm_stats)
        data_config = dataclasses.replace(self.config.data, base_config=base_config)
        self.config = dataclasses.replace(self.config, data=data_config)
        self.policy = policy_config.create_trained_policy(
            self.config, model_path, norm_stats=norm_stats, pytorch_device=device
        )
        self.action_chunk_length = int(self.config.model.action_horizon)
        if self.action_chunk_length <= 0:
            raise ValueError("OpenPI config returned a non-positive action horizon")

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        images = _image_list(observation)
        state = observation["robot_state"]
        state_16d = np.concatenate(
            (
                state["state.end_effector_position_relative"],
                state["state.end_effector_rotation_relative"],
                state["state.base_position"],
                state["state.base_rotation"],
                state["state.gripper_qpos"],
            ),
            axis=0,
        ).astype(np.float32)
        result = self.policy.infer(
            {
                "observation/image": images[0],
                "observation/wrist_image": images[2],
                "observation/right_image": images[1],
                "observation/state": state_16d,
                "prompt": observation["instruction"],
            }
        )
        action = np.asarray(result["actions"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(f"OpenPI returned unexpected action shape {action.shape}")
        self.last_raw_dim = action.shape[-1]
        return action


class DiffusionPolicyAdapter:
    def __init__(self, model_path: str, device: str) -> None:
        import copy
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        payload = torch.load(model_path, pickle_module=dill, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(payload["cfg"])))
        workspace_cls = hydra.utils.get_class(cfg._target_)
        self.workspace = workspace_cls(cfg, output_dir=str(Path(model_path).parent.parent / "worker"))
        self.workspace.load_payload(payload)
        self.policy = self.workspace.ema_model if cfg.training.use_ema else self.workspace.model
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self.torch = torch
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.action_chunk_length = int(cfg.n_action_steps)
        self.previous: dict[str, np.ndarray] | None = None

        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        clip_path = os.environ.get(
            "ROBOCASA_CLIP_PATH",
            str(Path(__file__).resolve().parents[1] / "models/clip-vit-large-patch14"),
        )
        self.lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            clip_path, local_files_only=True
        ).to(self.device).eval()
        self.lang_tokenizer = AutoTokenizer.from_pretrained(
            clip_path, local_files_only=True
        )

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        import torch

        images = _image_list(observation)
        state = observation["robot_state"]
        current = {
            "robot0_agentview_right_image": np.moveaxis(images[1], -1, 0).astype(np.float32) / 255.0,
            "robot0_agentview_left_image": np.moveaxis(images[0], -1, 0).astype(np.float32) / 255.0,
            "robot0_eye_in_hand_image": np.moveaxis(images[2], -1, 0).astype(np.float32) / 255.0,
            "robot0_base_to_eef_pos": np.asarray(state["state.end_effector_position_relative"], dtype=np.float32),
            "robot0_base_to_eef_quat": np.asarray(state["state.end_effector_rotation_relative"], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray(state["state.gripper_qpos"], dtype=np.float32),
        }
        previous = self.previous or current
        self.previous = {key: value.copy() for key, value in current.items()}
        obs_dict: dict[str, torch.Tensor] = {}
        for key, value in current.items():
            stacked = np.stack([previous[key], value], axis=0)
            obs_dict[key] = torch.from_numpy(stacked[None]).to(self.device)
        tokens = self.lang_tokenizer(
            text=observation["instruction"],
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            lang = self.lang_emb_model(**tokens)["text_embeds"][0]
        lang_np = lang.detach().cpu().numpy().astype(np.float32)
        obs_dict["lang_emb"] = torch.from_numpy(np.tile(lang_np, (1, self.n_obs_steps, 1))).to(self.device)
        with torch.inference_mode():
            result = self.policy.predict_action(obs_dict)
        action = result["action"].detach().cpu().numpy()
        self.last_raw_dim = action.shape[-1]
        return np.asarray(action[0, : self.action_chunk_length], dtype=np.float32)


def build_adapter(args: argparse.Namespace):
    if args.backend == "diffusion_policy":
        # The upstream repository is intentionally a source checkout without
        # a top-level __init__.py, so expose it as a namespace package.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps/diffusion_policy"))
    if args.backend == "groot":
        return GrootAdapter(args.model_path, args.device)
    if args.backend == "openpi":
        return OpenPIAdapter(args.model_path, args.device)
    if args.backend == "diffusion_policy":
        return DiffusionPolicyAdapter(args.model_path, args.device)
    raise ValueError(f"unsupported worker backend: {args.backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Duplicate the pipe before redirecting fd 1.  Third-party libraries are
    # allowed to print freely, while protocol frames remain binary and clean.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        adapter = build_adapter(args)
        send_frame(protocol_out, {"ok": True, "action_chunk_length": adapter.action_chunk_length})
    except Exception as error:
        traceback.print_exc()
        send_frame(protocol_out, {"ok": False, "error": f"{type(error).__name__}: {error}"})
        return

    while True:
        request = read_frame(sys.stdin.buffer)
        if request is None:
            break
        try:
            if request.get("op") == "close":
                send_frame(protocol_out, {"ok": True})
                break
            if request.get("op") != "infer":
                raise ValueError(f"unknown worker operation: {request.get('op')!r}")
            action = adapter.infer(_canonicalize_observation(request["observation"]))
            if action.ndim != 2 or action.shape[1] < 7 or not np.all(np.isfinite(action)):
                raise ValueError(f"invalid action returned by adapter: {action.shape}")
            # Keep the response NumPy-version agnostic for the same reason as
            # the request payload above.
            send_frame(protocol_out, {"ok": True, "action": action.tolist()})
        except Exception as error:
            traceback.print_exc()
            send_frame(protocol_out, {"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()

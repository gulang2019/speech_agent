"""Model registry and common action-chunk interface.

Model stacks with conflicting dependencies run in ``model_worker.py`` under
their own virtual environment.  The simulator process only sees a canonical
single-frame observation and the model's native direct-control action chunk.

The chunk width is model dependent: some checkpoints emit the 7D subset
(arm pose + gripper), while the RoboCasa365-era checkpoints emit the full 12D
vector that also carries ``base_motion`` and ``control_mode``.  Everything the
model returns is forwarded; ``make_action`` in the experiment script is the only
place that knows how the dense environment action is laid out.
"""

from __future__ import annotations

import pickle
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


# The narrowest legitimate chunk is the 7D arm+gripper subset used by the
# legacy ``robocasa_mg`` checkpoints.  Wider chunks (the 12D RoboCasa365 layout)
# are forwarded verbatim so ``base_motion``/``control_mode`` survive.
MODEL_ACTION_FLOOR_DIM = 7


@dataclass
class PolicyObservation:
    """Canonical observation shared by all benchmark adapters."""

    images: list[Image.Image]
    instruction: str
    xiaomi_state: np.ndarray
    robot_state: dict[str, np.ndarray]

    def payload(self) -> dict[str, Any]:
        return {
            # Do not pickle NumPy arrays across the isolated environments:
            # NumPy 1.x and 2.x use different internal module paths.  A small
            # self-describing bytes record is stable across both runtimes and
            # avoids converting camera frames to huge Python integer lists.
            "images": [
                {
                    "data": np.ascontiguousarray(np.asarray(image, dtype=np.uint8)).tobytes(),
                    "shape": tuple(np.asarray(image).shape),
                }
                for image in self.images
            ],
            "instruction": self.instruction,
            "xiaomi_state": np.asarray(self.xiaomi_state, dtype=np.float32).tolist(),
            "robot_state": {
                key: np.asarray(value, dtype=np.float32).tolist()
                for key, value in self.robot_state.items()
            },
        }


class ActionChunkPolicy(Protocol):
    action_chunk_length: int

    def infer(self, observation: PolicyObservation) -> np.ndarray:
        """Return a finite ``[time, >=7]`` direct-control action chunk."""

    def close(self) -> None:
        """Release model resources."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    backend: str
    checkpoint: str
    status: str
    notes: str
    supported_splits: tuple[str, ...]
    action_chunk_length: int
    runtime_env: str | None = None


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_SPECS: dict[str, ModelSpec] = {
    "xiaomi": ModelSpec(
        "xiaomi", "Xiaomi-Robotics-1", "local_transformers",
        str(ROOT_DIR / "models/xiaomi-robotics-1-robocasa"), "ready",
        "Local RoboCasa checkpoint; 10-step action chunks.",
        ("legacy5", "pretrain", "target"), 10,
    ),
    "gr00t_n1_5": ModelSpec(
        "gr00t_n1_5", "GR00T N1.5", "groot",
        str(ROOT_DIR / "models/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000"),
        "ready", "Official RoboCasa multitask checkpoint; 16-step native chunks.",
        ("pretrain",), 16, str(ROOT_DIR / ".runtime/envs/groot"),
    ),
    "pi0_5": ModelSpec(
        "pi0_5", "pi0.5", "openpi",
        str(ROOT_DIR / "models/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"),
        "ready", "Official RoboCasa OpenPI checkpoint; native horizon is validated by the worker.",
        ("pretrain",), 50, str(ROOT_DIR / ".runtime/envs/openpi"),
    ),
    "pi0": ModelSpec(
        "pi0", "pi0", "openpi",
        str(ROOT_DIR / "models/robocasa365_checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"),
        "planned", "Optional same-family ablation; checkpoint not downloaded in this phase.",
        ("pretrain",), 50, str(ROOT_DIR / ".runtime/envs/openpi"),
    ),
    "diffusion_policy": ModelSpec(
        "diffusion_policy", "Diffusion Policy", "diffusion_policy",
        str(ROOT_DIR / "models/robocasa365_checkpoints/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=0500-test_mean_score=-1.000.ckpt"),
        "ready", "Official RoboCasa hybrid Transformer baseline; 8-step native chunks.",
        ("pretrain",), 8, str(ROOT_DIR / ".runtime/envs/diffusion_policy"),
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"unknown model {name!r}; available: {available}") from error


def require_implemented_model(name: str) -> ModelSpec:
    spec = get_model_spec(name)
    if spec.status != "ready":
        raise NotImplementedError(f"model {name!r} is registered but not runnable: {spec.checkpoint}")
    if not Path(spec.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint for {name!r} does not exist: {spec.checkpoint}")
    if spec.runtime_env and not (Path(spec.runtime_env) / "bin/python").exists():
        raise FileNotFoundError(f"runtime environment for {name!r} is missing: {spec.runtime_env}")
    return spec


def _read_frame(stream) -> Any:
    header = stream.read(8)
    if len(header) != 8:
        raise RuntimeError("model worker closed its output stream")
    size = struct.unpack("!Q", header)[0]
    if size > 1 << 32:
        raise RuntimeError(f"invalid model worker frame size: {size}")
    payload = stream.read(size)
    if len(payload) != size:
        raise RuntimeError("truncated model worker response")
    return pickle.loads(payload)


class SubprocessActionChunkPolicy:
    """Synchronous request/response client for an isolated model worker."""

    def __init__(self, spec: ModelSpec, model_path: str | None = None, device: str = "cuda"):
        require_implemented_model(spec.name)
        runtime = Path(spec.runtime_env or "")
        command = [str(runtime / "bin/python"), str(ROOT_DIR / "scripts/model_worker.py"),
                   "--backend", spec.backend, "--model-path", model_path or spec.checkpoint,
                   "--device", device]
        self.action_chunk_length = spec.action_chunk_length
        self.process = subprocess.Popen(command, cwd=ROOT_DIR, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None, bufsize=0)
        try:
            response = _read_frame(self.process.stdout)
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "model worker failed to initialize"))
            reported = int(response.get("action_chunk_length", self.action_chunk_length))
            if reported != self.action_chunk_length:
                raise RuntimeError(f"worker reports chunk length {reported}, registry says {self.action_chunk_length}")
        except Exception:
            self.close()
            raise

    def infer(self, observation: PolicyObservation) -> np.ndarray:
        if self.process.poll() is not None:
            raise RuntimeError(f"model worker exited with code {self.process.returncode}")
        payload = pickle.dumps({"op": "infer", "observation": observation.payload()}, protocol=5)
        self.process.stdin.write(struct.pack("!Q", len(payload)))
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        response = _read_frame(self.process.stdout)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "model worker inference failed"))
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < MODEL_ACTION_FLOOR_DIM:
            raise RuntimeError(
                f"worker returned invalid action shape {action.shape}, "
                f"expected [T, >={MODEL_ACTION_FLOOR_DIM}]"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("worker returned NaN or Inf actions")
        return action

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                payload = pickle.dumps({"op": "close"}, protocol=5)
                process.stdin.write(struct.pack("!Q", len(payload)))
                process.stdin.write(payload)
                process.stdin.flush()
                _read_frame(process.stdout)
            except Exception:
                process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def create_policy(spec: ModelSpec, model_path: str | None = None, **kwargs: Any) -> ActionChunkPolicy:
    if spec.backend == "local_transformers":
        try:
            from scripts.xiaomi_latency_experiment import PolicyClient
        except ModuleNotFoundError:  # direct execution: python scripts/...
            from xiaomi_latency_experiment import PolicyClient
        return PolicyClient(model_path or spec.checkpoint, kwargs.get("model_seed", 42),
                            kwargs.get("diffusion_steps", 5), kwargs.get("attn_implementation", "eager"))
    return SubprocessActionChunkPolicy(spec, model_path=model_path)

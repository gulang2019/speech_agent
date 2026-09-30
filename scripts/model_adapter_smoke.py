"""Synthetic forward smoke test for an isolated model adapter.

This deliberately does not create a simulator.  It verifies that the pinned
checkpoint can load on the allocated GPU, accepts the canonical observation,
and returns the registered native chunk length mapped to seven direct-control
dimensions.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image

try:
    from scripts.latency_model_adapters import PolicyObservation, create_policy, get_model_spec
except ModuleNotFoundError:  # direct execution from the repository's scripts directory
    from latency_model_adapters import PolicyObservation, create_policy, get_model_spec


def synthetic_observation(size: int) -> PolicyObservation:
    image = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    return PolicyObservation(
        images=[image.copy(), image.copy(), image.copy()],
        instruction="pick up the object and place it in the cabinet",
        xiaomi_state=np.zeros(8, dtype=np.float32),
        robot_state={
            "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
            "state.end_effector_rotation_relative": np.array([1, 0, 0, 0], dtype=np.float32),
            "state.gripper_qpos": np.zeros(2, dtype=np.float32),
            "state.base_position": np.zeros(3, dtype=np.float32),
            "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=("gr00t_n1_5", "pi0_5", "diffusion_policy"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-env", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    spec = get_model_spec(args.model_name)
    if args.model_env:
        from dataclasses import replace
        spec = replace(spec, runtime_env=args.model_env)
    policy = None
    try:
        started = time.perf_counter()
        policy = create_policy(spec, model_path=args.model_path)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        action = policy.infer(synthetic_observation(args.image_size))
        infer_ms = (time.perf_counter() - started) * 1000.0
        expected = spec.action_chunk_length
        if action.shape != (expected, 7):
            raise RuntimeError(f"unexpected action shape {action.shape}; expected {(expected, 7)}")
        if not np.all(np.isfinite(action)):
            raise RuntimeError("adapter returned NaN or Inf")
        print(json.dumps({
            "model": spec.name,
            "backend": spec.backend,
            "action_shape": list(action.shape),
            "load_seconds": load_seconds,
            "infer_ms": infer_ms,
            "finite": True,
        }, sort_keys=True))
    finally:
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    main()

"""Serve the local Xiaomi RoboCasa checkpoint to a RoboDojo client.

RoboDojo's simulator owns the episode loop and sends XPolicyLab-compatible
observations over WebSocket. This process owns only the model and the
RoboDojo/Xiaomi compatibility boundary.

The XPolicyLab repository must be on ``PYTHONPATH`` because its WebSocket
server is the public transport implementation used by RoboDojo. Example:

    PYTHONPATH=/path/to/XPolicyLab:$PYTHONPATH \
      python scripts/robodojo_xiaomi_server.py \
      --model-path models/xiaomi-robotics-1-robocasa

This server is intentionally single-environment. Set RoboDojo's
``eval_batch=false`` for this compatibility policy; batching a single-arm
RoboCasa checkpoint across two RoboDojo scenes would otherwise make its
mutable observation state ambiguous.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import numpy as np

try:
    from scripts.robodojo_compat import (
        RoboDojoBridgeConfig,
        robodojo_to_policy_observation,
        xiaomi_action_to_robodojo_action,
    )
    from scripts.xiaomi_latency_experiment import PolicyClient
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from robodojo_compat import (
        RoboDojoBridgeConfig,
        robodojo_to_policy_observation,
        xiaomi_action_to_robodojo_action,
    )
    from xiaomi_latency_experiment import PolicyClient


class RoboDojoXiaomiModel:
    """XPolicyLab model-method surface backed by the local Xiaomi client."""

    def __init__(
        self,
        model_path: str,
        model_seed: int = 42,
        diffusion_steps: int = 5,
        attn_implementation: str = "eager",
        bridge_config: RoboDojoBridgeConfig | None = None,
        policy: Any | None = None,
    ) -> None:
        self.bridge_config = bridge_config or RoboDojoBridgeConfig()
        self.policy = policy if policy is not None else PolicyClient(
            model_path, model_seed, diffusion_steps, attn_implementation
        )
        self._raw_obs: dict[str, Any] | None = None

    def update_obs(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            raise TypeError(f"update_obs expects a dict, got {type(obs).__name__}")
        # Validate and normalize before starting GPU inference. This makes
        # protocol/schema errors local and actionable rather than surfacing as
        # a model tensor-shape failure several calls later.
        robodojo_to_policy_observation(obs, self.bridge_config)
        self._raw_obs = obs

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._raw_obs is None:
            raise RuntimeError("update_obs must be called before get_action")
        observation = robodojo_to_policy_observation(self._raw_obs, self.bridge_config)
        chunk = np.asarray(self.policy.infer(observation), dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"Xiaomi policy returned unexpected action shape {chunk.shape}")
        return [
            xiaomi_action_to_robodojo_action(action[:7], self._raw_obs, self.bridge_config)
            for action in chunk
        ]

    def reset(self) -> None:
        self._raw_obs = None

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/xiaomi-robotics-1-robocasa")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--active-arm", choices=("left", "right"), default="right")
    parser.add_argument("--action-mode", choices=("delta_ee", "absolute_ee"), default="delta_ee")
    parser.add_argument("--delta-frame", choices=("ee", "base", "world"), default="base")
    parser.add_argument(
        "--action-units",
        choices=("robocasa_normalized", "physical"),
        default="robocasa_normalized",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("auto", "zero_one", "minus_one_one"),
        default="auto",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--crop-ratio", type=float, default=1.0)
    return parser


async def serve_model(args: argparse.Namespace) -> None:
    try:
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "RoboDojo/XPolicyLab WebSocket dependencies are missing. "
            "Put the XPolicyLab checkout on PYTHONPATH and install its runtime dependencies."
        ) from error

    bridge_config = RoboDojoBridgeConfig(
        active_arm=args.active_arm,
        action_mode=args.action_mode,
        delta_frame=args.delta_frame,
        action_units=args.action_units,
        gripper_mode=args.gripper_mode,
        image_size=args.image_size,
        crop_ratio=args.crop_ratio,
    )
    model = RoboDojoXiaomiModel(
        model_path=args.model_path,
        model_seed=args.model_seed,
        diffusion_steps=args.diffusion_steps,
        attn_implementation=args.attn_implementation,
        bridge_config=bridge_config,
    )
    server = PolicyServer(
        model,
        PolicyServerConfig(host=args.host, port=args.port),
    )
    try:
        await server.serve_forever()
    finally:
        model.close()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(serve_model(args))


if __name__ == "__main__":
    main()

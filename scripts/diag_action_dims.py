"""Diagnose action-vector fidelity for the non-Xiaomi adapters.

The RoboCasa gym env expects a 12D action vector:
    [0:3]   end_effector_position
    [3:6]   end_effector_rotation
    [6:7]   gripper_close
    [7:11]  base_motion  (4D)
    [11:12] control_mode (1D)

The benchmark truncates every model to 7D and builds a dense action by
zero-padding to ``env.action_spec``.  This script records what each adapter
actually returns and whether the dropped dimensions carry signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.latency_model_adapters import PolicyObservation, create_policy, get_model_spec
from scripts.xiaomi_latency_experiment import (
    CAMERA_NAMES,
    clone_observation,
    make_action,
    render_observation,
    resolve_scene_parameters,
    resolve_task,
)


def build_env(args):
    import argparse as _argparse

    import robocasa  # noqa: F401  (registers gym tasks)
    from robocasa.utils.env_utils import create_env

    argv = _argparse.Namespace(
        scene_set=args.scene_set,
        object_split=args.object_split,
        benchmark_manifest=args.benchmark_manifest,
    )
    env_name, _kwargs, _horizon = resolve_task(args.task_name)
    env_kwargs = dict(
        env_name=env_name,
        robots="PandaOmron",
        camera_names=list(CAMERA_NAMES),
        camera_widths=args.image_size,
        camera_heights=args.image_size,
        seed=args.seed,
        render_onscreen=False,
        randomize_cameras=False,
    )
    split, object_split, layout_and_style_ids = resolve_scene_parameters(argv)
    if layout_and_style_ids is not None:
        env_kwargs.update(
            split=split,
            obj_instance_split=object_split,
            layout_and_style_ids=layout_and_style_ids,
        )
    else:
        env_kwargs["split"] = split
    return create_env(**env_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--task-name", default="PickPlaceCounterToCabinet")
    parser.add_argument("--scene-set", default="legacy5")
    parser.add_argument("--object-split", default="pretrain")
    parser.add_argument("--benchmark-manifest", default=str(Path("configs/latency_benchmark_v1.json")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=40, help="control steps to roll out")
    parser.add_argument("--chunk-hold", type=int, default=8, help="replan interval")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    spec = get_model_spec(args.model_name)
    policy = create_policy(
        spec,
        model_path=args.model_path or spec.checkpoint,
        model_seed=args.model_seed,
        diffusion_steps=args.diffusion_steps,
        attn_implementation=args.attn_implementation,
    )
    report: dict[str, object] = {
        "model": args.model_name,
        "declared_chunk_length": policy.action_chunk_length,
        "observations": [],
    }
    env = build_env(args)
    try:
        env.reset()
        instruction = env.get_ep_meta()["lang"]
        observation = render_observation(env, args.image_size, args.crop_ratio, instruction)
        print("env action_spec shape:", np.asarray(env.action_spec).shape)
        print("action dim expected by env:", int(np.asarray(env.action_spec).shape[-1]))
        chunk = None
        for step in range(args.steps):
            if step % args.chunk_hold == 0:
                chunk = np.asarray(
                    policy.infer(clone_observation(observation)), dtype=np.float32
                )
                record = {
                    "step": step,
                    "chunk_shape": list(chunk.shape),
                    "per_dim_std": chunk.std(axis=0).tolist(),
                    "per_dim_min": chunk.min(axis=0).tolist(),
                    "per_dim_max": chunk.max(axis=0).tolist(),
                    "first_action": chunk[0].tolist(),
                }
                report["observations"].append(record)
                print(f"step {step}: chunk {chunk.shape}")
                if chunk.shape[1] > 7:
                    tail = chunk[:, 7:]
                    print(
                        "  dropped dims 7+: std="
                        f"{np.round(tail.std(axis=0), 4).tolist()} "
                        f"range=[{tail.min():.4f}, {tail.max():.4f}]"
                    )
            # Use the production mapping so the diagnostic exercises the same
            # binary-channel thresholding the benchmark relies on.
            policy_action = chunk[step % args.chunk_hold]
            dense = make_action(env, policy_action)
            if step == 0:
                print("  dense action (post make_action):", np.round(dense, 4).tolist())
            env.step(dense)
            observation = render_observation(env, args.image_size, args.crop_ratio, instruction)
    finally:
        policy.close()
        env.close()

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print("wrote", args.output)


if __name__ == "__main__":
    main()

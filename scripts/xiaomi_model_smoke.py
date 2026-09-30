"""Load Xiaomi-Robotics-1 and run one local RoboCasa-style action forward.

This test deliberately uses synthetic camera images and zero proprioception.
It validates the checkpoint, custom Transformers code, processor contract, GPU
placement, and action-head output shape. It is not a task-success evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/xiaomi-robotics-1-robocasa",
        help="Local checkpoint directory",
    )
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm GPU is required for the Xiaomi model smoke test")

    model_dir = Path(args.model).resolve()
    device = torch.device("cuda")
    print("model_dir:", model_dir)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    print("processor:", type(processor).__name__)
    print("robot_types:", processor.list_robot_types())

    print("loading model ...", flush=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    print("model:", type(model).__name__)
    print("load_seconds:", round(time.perf_counter() - started, 2))

    # The checkpoint's custom processor expects the RoboCasa multimodal prompt
    # and a 60-D state. The model's action head expects state=(B, 1, 60).
    size = args.image_size
    image = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    prompt = "<|vision_start|><|image_pad|><|vision_end|>\nPick up the object."
    inputs = processor(
        images=[image],
        text=[prompt],
        state=np.zeros((1, 1, 60), dtype=np.float32),
        robot_type="robocasa_mg",
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    print(
        "inputs:",
        json.dumps(
            {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            },
            sort_keys=True,
        ),
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, num_steps=args.num_steps, seed=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    actions = output.actions
    expected = (1, 10, 60)
    if tuple(actions.shape) != expected:
        raise RuntimeError(f"unexpected action shape: {tuple(actions.shape)} != {expected}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("model returned non-finite actions")
    print("actions:", tuple(actions.shape), actions.dtype)
    print("action_range:", float(actions.min()), float(actions.max()))
    print("forward_seconds:", round(elapsed, 3))
    print("XIAOMI_MODEL_SMOKE_OK")


if __name__ == "__main__":
    main()

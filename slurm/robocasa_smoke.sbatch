#!/usr/bin/env bash
#SBATCH --job-name=robocasa-smoke
#SBATCH --output=logs/robocasa-%j.out
#SBATCH --error=logs/robocasa-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

# Slurm copies the script into its spool directory. The submit directory is
# the stable path back to this checkout.
ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

# Use EGL for headless MuJoCo rendering on compute nodes.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Missing environment: ${ENV_DIR}" >&2
    echo "Create it on a login node first (see INSTALL.md)." >&2
    exit 2
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

"${ENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as md
import os

import gymnasium as gym
import mujoco
import robocasa
import robosuite
import torch

print("robocasa:", md.version("robocasa"))
print("robosuite:", md.version("robosuite"))
print("mujoco:", mujoco.__version__)
print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("torch CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated no CUDA device to PyTorch")
print("GPU:", torch.cuda.get_device_name(0))
print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))

# Verify headless EGL rendering with the Slurm-allocated GPU.
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 2"/><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    image = renderer.render()
assert image.shape == (64, 64, 3), image.shape
print("MuJoCo EGL render:", image.shape, "OK")
print("RoboCasa import: OK")

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain",
    seed=0,
    camera_widths=64,
    camera_heights=64,
    disable_env_checker=True,
)
obs, info = env.reset()
action = env.action_space.sample()
if isinstance(action, dict) and "action.base_motion" in action:
    action["action.base_motion"][:] = 0.0
obs, reward, terminated, truncated, info = env.step(action)
camera_shapes = {
    key: value.shape
    for key, value in obs.items()
    if key.startswith("video.") and hasattr(value, "shape")
}
print("Task reset/step: OK")
print("Observation keys:", len(obs))
print("Camera observations:", camera_shapes)
env.close()
PY

echo "Smoke test completed, including a real RoboCasa task reset and step."

#!/usr/bin/env bash
#SBATCH --job-name=model-adapter-smoke
#SBATCH --output=logs/model-adapter-smoke-%j.out
#SBATCH --error=logs/model-adapter-smoke-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
cd "$ROOT_DIR"
mkdir -p logs
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to gr00t_n1_5, pi0_5, or diffusion_policy}"
MODEL_ARGS=(--model-name "$MODEL_NAME")
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ -n "${MODEL_ENV:-}" ]]; then
    MODEL_ARGS+=(--model-env "$MODEL_ENV")
fi

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
.conda-env/bin/python scripts/model_adapter_smoke.py "${MODEL_ARGS[@]}"

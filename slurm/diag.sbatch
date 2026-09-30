#!/usr/bin/env bash
#SBATCH --job-name=diag
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
cd "$ROOT_DIR"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.conda-env/bin/python scripts/diag_door.py "$1" "${2:-2}"

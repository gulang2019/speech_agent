#!/usr/bin/env bash
#SBATCH --job-name=xiaomi-model-smoke
#SBATCH --output=logs/xiaomi-model-%j.out
#SBATCH --error=logs/xiaomi-model-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
cd "$ROOT_DIR"
mkdir -p logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Missing environment: ${ENV_DIR}" >&2
    exit 2
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

"${ENV_DIR}/bin/python" scripts/xiaomi_model_smoke.py \
    --model "${ROOT_DIR}/models/xiaomi-robotics-1-robocasa" \
    --num-steps "${NUM_STEPS:-1}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-eager}"

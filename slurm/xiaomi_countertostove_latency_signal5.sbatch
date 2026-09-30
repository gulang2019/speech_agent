#!/usr/bin/env bash
#SBATCH --job-name=xiaomi-countertostove-signal5
#SBATCH --output=logs/xiaomi-countertostove-signal5-%j.out
#SBATCH --error=logs/xiaomi-countertostove-signal5-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
MANIFEST="${ROOT_DIR}/configs/latency_benchmark_v1.json"

cd "$ROOT_DIR"
mkdir -p logs
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

"${ENV_DIR}/bin/python" scripts/xiaomi_latency_experiment.py \
    --model-name xiaomi \
    --benchmark-manifest "$MANIFEST" \
    --task-name PickPlaceCounterToStove \
    --scene-set countertostove_latency_signal5 \
    --object-split pretrain \
    --episodes "${EPISODES:-150}" \
    --seed "${SEED:-0}" \
    --delays-ms 0,100 \
    --delay-domain sim \
    --replan-steps-list 5 \
    --rtc off \
    --control-frequency 0 \
    --delay-step-frequency 20 \
    --output-dir "${OUTPUT_DIR:?OUTPUT_DIR is required}"

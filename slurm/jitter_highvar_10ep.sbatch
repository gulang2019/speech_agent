#!/usr/bin/env bash
# Higher-variance jitter extension using lower-bounded two-point samples.
# Submit with: sbatch --partition=gpu-scavenger --array=0-11%8 slurm/jitter_highvar_10ep.sbatch

#SBATCH --job-name=jitter-highvar
#SBATCH --output=logs/jitter-highvar-%A_%a.out
#SBATCH --error=logs/jitter-highvar-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
ROOT_OUT="${ROOT_OUT:-eval_results/jitter_highvar_10ep_20261007}"
EPISODES="${EPISODES:-10}"
TASK_NAME="PickPlaceCounterToCabinet"

MODELS=(xiaomi gr00t_n1_5)
AXES=(delay replan)
DELAY_VARIANCES=(30000 50000 90000)
REPLAN_VARIANCES=(36 90 180)

index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
model="${MODELS[$((index % 2))]}"
axis="${AXES[$(((index / 2) % 2))]}"
if [[ "$axis" == delay ]]; then
    variance="${DELAY_VARIANCES[$((index / 4))]}"
    extra=(--delay-jitter-variance-ms2 "$variance" --delay-jitter-distribution lower_bound_twopoint)
else
    variance="${REPLAN_VARIANCES[$((index / 4))]}"
    extra=(--replan-jitter-variance-steps2 "$variance" --replan-jitter-distribution lower_bound_twopoint)
fi

OUTPUT_DIR="${ROOT_OUT}/${axis}/${model}/var${variance}"
cd "$ROOT_DIR"
mkdir -p logs "$OUTPUT_DIR"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Missing environment: ${ENV_DIR}" >&2
    exit 2
fi

echo "model=${model} axis=${axis} variance=${variance} distribution=lower_bound_twopoint output=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

"${ENV_DIR}/bin/python" scripts/xiaomi_latency_experiment.py \
    --model-name "$model" \
    --benchmark-manifest "${ROOT_DIR}/configs/latency_benchmark_v1.json" \
    --task-name "$TASK_NAME" \
    --scene-set legacy5 \
    --scene-selection pinned \
    --object-split pretrain \
    --episodes "$EPISODES" \
    --delays-ms 100 \
    --delay-domain sim \
    --replan-steps-list 10 \
    --rtc off \
    --control-frequency 0 \
    --delay-step-frequency 20 \
    --output-dir "$OUTPUT_DIR" \
    "${extra[@]}"

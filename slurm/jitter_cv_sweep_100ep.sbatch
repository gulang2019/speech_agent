#!/usr/bin/env bash
# CV^2 jitter sweeps, 100 episodes per point.
# Submit with:
#   sbatch --partition=gpu-scavenger --array=0-33%8 slurm/jitter_cv_sweep_100ep.sbatch

#SBATCH --job-name=jitter-cv-100
#SBATCH --output=logs/jitter-cv-100-%A_%a.out
#SBATCH --error=logs/jitter-cv-100-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
ROOT_OUT="${ROOT_OUT:-eval_results/jitter_cv_sweep_100ep_20261008}"
EPISODES="${EPISODES:-100}"
TASK_NAME="PickPlaceCounterToCabinet"

index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( index < 9 )); then
    MODEL_NAME="xiaomi"
    AXIS="delay"
    DELAY_CV2=(1 2 3 4 5 6 7 8 9)
    CV2="${DELAY_CV2[$index]}"
    DISTRIBUTION="lognormal"
elif (( index < 16 )); then
    MODEL_NAME="gr00t_n1_5"
    AXIS="delay"
    DELAY_CV2=(9 11 13 15 17 19 20)
    CV2="${DELAY_CV2[$((index - 9))]}"
    DISTRIBUTION="lognormal"
elif (( index < 25 )); then
    MODEL_NAME="xiaomi"
    AXIS="replan"
    REPLAN_CV2=(0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9)
    CV2="${REPLAN_CV2[$((index - 16))]}"
    DISTRIBUTION="shifted_negative_binomial"
else
    MODEL_NAME="gr00t_n1_5"
    AXIS="replan"
    REPLAN_CV2=(0.3 0.8 1.3 1.8 2.3 2.8 3.3 3.8 4.3)
    CV2="${REPLAN_CV2[$((index - 25))]}"
    DISTRIBUTION="shifted_negative_binomial"
fi

CV_TAG="${CV2//./p}"
OUTPUT_DIR="${ROOT_OUT}/${AXIS}/${MODEL_NAME}/cv${CV_TAG}"
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

echo "index=${index} model=${MODEL_NAME} axis=${AXIS} cv2=${CV2} distribution=${DISTRIBUTION} episodes=${EPISODES}"
echo "output=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

COMMON_ARGS=(
    --model-name "$MODEL_NAME"
    --benchmark-manifest "${ROOT_DIR}/configs/latency_benchmark_v1.json"
    --task-name "$TASK_NAME"
    --scene-set legacy5
    --scene-selection pinned
    --object-split pretrain
    --episodes "$EPISODES"
    --delays-ms 100
    --delay-domain sim
    --replan-steps-list 10
    --rtc off
    --control-frequency 0
    --delay-step-frequency 20
    --output-dir "$OUTPUT_DIR"
)

if [[ "$AXIS" == delay ]]; then
    COMMON_ARGS+=(
        --delay-jitter-cv2 "$CV2"
        --delay-jitter-distribution "$DISTRIBUTION"
    )
else
    COMMON_ARGS+=(
        --replan-jitter-cv2 "$CV2"
        --replan-jitter-distribution "$DISTRIBUTION"
    )
fi

"${ENV_DIR}/bin/python" scripts/xiaomi_latency_experiment.py "${COMMON_ARGS[@]}"

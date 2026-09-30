#!/usr/bin/env bash
#SBATCH --job-name=xiaomi-latency
#SBATCH --output=logs/xiaomi-latency-%j.out
#SBATCH --error=logs/xiaomi-latency-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
MODEL_DIR="${ROOT_DIR}/models/xiaomi-robotics-1-robocasa"
MANIFEST="${ROOT_DIR}/configs/latency_benchmark_v1.json"

cd "$ROOT_DIR"
mkdir -p logs
export MUJOCO_GL="${MUJOCO_GL:-egl}"
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

HORIZON_ARGS=()
if [[ -n "${HORIZON:-}" ]]; then
    HORIZON_ARGS=(--horizon "$HORIZON")
fi

REPLAN_STEPS_ARGS=()
if [[ -n "${REPLAN_VALUES:-}" ]]; then
    REPLAN_STEPS_ARGS=(--replan-steps-list "${REPLAN_VALUES// /,}")
elif [[ -n "${REPLAN_START:-}" || -n "${REPLAN_END:-}" ]]; then
    if [[ -z "${REPLAN_START:-}" || -z "${REPLAN_END:-}" ]]; then
        echo "REPLAN_START and REPLAN_END must be provided together" >&2
        exit 2
    fi
    REPLAN_STEPS_ARGS=(--replan-steps-list "$(seq -s, "$REPLAN_START" "$REPLAN_END")")
else
    REPLAN_STEPS_ARGS=(--replan-steps-list "${REPLAN_STEPS_LIST:-1,5,10}")
fi

DELAY_STEP_FREQUENCY_ARGS=()
if [[ -n "${DELAY_STEP_FREQUENCY:-}" ]]; then
    DELAY_STEP_FREQUENCY_ARGS=(--delay-step-frequency "$DELAY_STEP_FREQUENCY")
fi

MODEL_PATH_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_PATH_ARGS=(--model-path "$MODEL_PATH")
fi

MODEL_ENV_ARGS=()
if [[ -n "${MODEL_ENV:-}" ]]; then
    MODEL_ENV_ARGS=(--model-env "$MODEL_ENV")
fi

TASK_SET_ARGS=()
if [[ -n "${TASK_SET:-}" ]]; then
    TASK_SET_ARGS=(--task-set "$TASK_SET")
fi

TASK_NAME_ARGS=()
if [[ -n "${TASK_NAME:-}" ]]; then
    TASK_NAME_ARGS=(--task-name "$TASK_NAME")
fi

REPLAY_ARGS=()
if [[ -n "${REPLAY_FROM:-}" ]]; then
    REPLAY_ARGS=(--replay-from "$REPLAY_FROM" --replay-scene-mode "${REPLAY_SCENE_MODE:-auto}")
    if [[ -n "${REPLAY_LIMIT:-}" ]]; then
        REPLAY_ARGS+=(--replay-limit "$REPLAY_LIMIT")
    fi
fi

"${ENV_DIR}/bin/python" scripts/xiaomi_latency_experiment.py \
    --model-name "${MODEL_NAME:-xiaomi}" \
    --benchmark-manifest "${BENCHMARK_MANIFEST:-$MANIFEST}" \
    "${MODEL_PATH_ARGS[@]}" \
    "${MODEL_ENV_ARGS[@]}" \
    --diffusion-steps "${DIFFUSION_STEPS:-5}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
    "${TASK_SET_ARGS[@]}" \
    "${TASK_NAME_ARGS[@]}" \
    "${REPLAY_ARGS[@]}" \
    --scene-set "${SCENE_SET:-legacy5}" \
    --scene-selection "${SCENE_SELECTION:-pinned}" \
    --object-split "${OBJECT_SPLIT:-pretrain}" \
    --episodes "${EPISODES:-5}" \
    --delays-ms "${DELAY_MS:-${DELAYS_MS:-0,100,300,500}}" \
    --delay-domain "${DELAY_DOMAIN:-sim}" \
    "${DELAY_STEP_FREQUENCY_ARGS[@]}" \
    "${REPLAN_STEPS_ARGS[@]}" \
    --rtc "${RTC:-off,on}" \
    --control-frequency "${CONTROL_FREQUENCY:-20}" \
    --output-dir "${OUTPUT_DIR:-eval_results/xiaomi_latency/${SLURM_JOB_ID:-local}}" \
    "${HORIZON_ARGS[@]}"

#!/usr/bin/env bash
# Xiaomi PickPlaceCounterToCabinet: sim-domain delay x replan sweep.
#
# One array element is one (delay, replan) cell with 100 episodes. The %2
# throttle intentionally limits this experiment to two GPUs at a time.
#
# Submit:
#   sbatch slurm/submit_xiaomi_counter_to_cabinet_sim_replan.sbatch
#
# Override the output root or concurrency at submission time if needed:
#   ROOT_OUT=eval_results/... sbatch --array=0-39%2 slurm/...

#SBATCH --job-name=xiaomi-cabinet-sim
#SBATCH --output=logs/xiaomi-cabinet-sim-%A_%a.out
#SBATCH --error=logs/xiaomi-cabinet-sim-%A_%a.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-39%2

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"
ROOT_OUT="${ROOT_OUT:-eval_results/xiaomi_counter_to_cabinet_sim_replan_100ep}"
TASK_NAME="PickPlaceCounterToCabinet"
EPISODES="${EPISODES:-100}"

DELAYS_MS=(0 100 300 500)
REPLAN_STEPS=(1 3 5 7 9 11 13 15 17 19)

index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
delay_index=$((index / ${#REPLAN_STEPS[@]}))
replan_index=$((index % ${#REPLAN_STEPS[@]}))
delay_ms="${DELAYS_MS[$delay_index]}"
replan_steps="${REPLAN_STEPS[$replan_index]}"

OUTPUT_DIR="${ROOT_OUT}/${TASK_NAME}/delay${delay_ms}ms/replan${replan_steps}"

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

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    expected="$(${ENV_DIR}/bin/python - "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("num_results") == 100 and len(data.get("conditions", [])) == 1)
PY
)"
    if [[ "$expected" == true ]]; then
        echo "Already complete: ${OUTPUT_DIR}"
        exit 0
    fi
fi

echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Condition: delay=${delay_ms}ms replan=${replan_steps}"
echo "Delay semantics: sim-domain at 20 Hz; no wall-domain delay"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

"${ENV_DIR}/bin/python" scripts/xiaomi_latency_experiment.py \
    --model-name xiaomi \
    --benchmark-manifest "${ROOT_DIR}/configs/latency_benchmark_v1.json" \
    --task-name "$TASK_NAME" \
    --scene-set pretrain20 \
    --scene-selection pinned \
    --object-split pretrain \
    --episodes "$EPISODES" \
    --delays-ms "$delay_ms" \
    --delay-domain sim \
    --replan-steps-list "$replan_steps" \
    --rtc off \
    --control-frequency 20 \
    --output-dir "$OUTPUT_DIR"

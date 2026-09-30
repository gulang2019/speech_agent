#!/usr/bin/env bash
# Submit one fixed-condition job per replan value, then merge the slices.
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-pi0_5}"
TASK_NAME="${TASK_NAME:-PickPlaceCounterToCabinet}"
SCENE_SET="${SCENE_SET:-single_scene_L7S10}"
DELAY_MS="${DELAY_MS:-100}"
EPISODES="${EPISODES:-20}"
ROOT_OUT="${ROOT_OUT:-eval_results/replan_sweep_pi05_counter_to_cabinet_L7S10_d100ms_20ep_parts}"

for replan in $(seq 1 50); do
    out_dir="${ROOT_OUT}/replan${replan}"
    if [[ -f "${out_dir}/summary.json" ]]; then
        echo "skip (done): ${out_dir}"
        continue
    fi
    MODEL_NAME="$MODEL_NAME" TASK_NAME="$TASK_NAME" SCENE_SET="$SCENE_SET" \
    DELAYS_MS="$DELAY_MS" DELAY_DOMAIN=sim DELAY_STEP_FREQUENCY=20 \
    REPLAN_STEPS_LIST="$replan" RTC=off EPISODES="$EPISODES" CONTROL_FREQUENCY=0 \
    OUTPUT_DIR="$out_dir" \
        sbatch --job-name="replan-${replan}" slurm/xiaomi_latency_experiment.sbatch
done

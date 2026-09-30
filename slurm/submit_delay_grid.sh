#!/usr/bin/env bash
# Delay sweep 0-100ms (10ms step) at fixed replan=5, rtc=off.
#
# Tasks: PickPlaceCounterToCabinet (reference, already complete at its
# registered 750-step horizon), OpenCabinet (single-step door), OpenDoubleDoor,
# and TurnOffMicrowave.
#
# Door/microwave tasks need horizons above the registry default: the policy
# does reach the goal but only after ~320-1040 steps, so the registered caps
# truncate most episodes.  Horizons below are calibrated from measured
# successful-episode lengths (see eval_results/calib).
#
# Usage: bash slurm/submit_delay_grid.sh [task ...]
set -euo pipefail

ROOT_DIR="/fact_home/xunyuanliu/dev/robo"
SBATCH_SCRIPT="${ROOT_DIR}/slurm/xiaomi_latency_experiment.sbatch"
OUT_ROOT="eval_results/xiaomi_delay_grid_r5_rtcoff"
EPISODES="${EPISODES:-30}"

CHUNKS=("0,10,20,30" "40,50,60,70" "80,90,100")

declare -A HORIZONS=(
    [PickPlaceCounterToCabinet]=750
    [OpenCabinet]=1800
    [OpenDoubleDoor]=2400
    [TurnOffMicrowave]=900
)

if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=(OpenCabinet OpenDoubleDoor TurnOffMicrowave)
fi

cd "$ROOT_DIR"
mkdir -p logs

for task in "${TASKS[@]}"; do
    horizon="${HORIZONS[$task]:?no calibrated horizon for $task}"
    short="${task:0:11}"
    for index in "${!CHUNKS[@]}"; do
        part=$((index + 1))
        TASK_NAME="$task" \
        EPISODES="$EPISODES" \
        DELAYS_MS="${CHUNKS[$index]}" \
        REPLAN_STEPS_LIST=5 \
        RTC=off \
        HORIZON="$horizon" \
        OUTPUT_DIR="${OUT_ROOT}/${task}/part${part}" \
        sbatch --job-name="dg-p${part}-${short}" "$SBATCH_SCRIPT"
    done
done

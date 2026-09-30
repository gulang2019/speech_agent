#!/usr/bin/env bash
# Submit the Xiaomi sim-domain replan sweep for PickPlaceCounterToCabinet.
# One job evaluates both delays (0 and 100 ms) at one replan value.
# Jobs are distributed across dependency lanes to cap GPU concurrency.
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

MODEL_NAME="xiaomi"
TASK_NAME="PickPlaceCounterToCabinet"
SCENE_SET="legacy5"
SCENE_SELECTION="sampled"
OBJECT_SPLIT="pretrain"
EPISODES="${EPISODES:-25}"
DELAYS_MS="0,100"
DELAY_DOMAIN="sim"
DELAY_STEP_FREQUENCY="20"
RTC="off"
CONTROL_FREQUENCY="20"
HORIZON="750"
ROOT_OUT="eval_results/xiaomi_replan_sim_0_100ms_1_50_50ep"
LANES="${LANES:-4}"
REPLAN_POINTS="${REPLAN_POINTS:-1 6 11 16 21 26 31 36 41 46}"

mkdir -p "$ROOT_OUT"

declare -a lane_tail
for ((lane = 0; lane < LANES; lane++)); do
    lane_tail[$lane]=""
done

for replan in $REPLAN_POINTS; do
    out_dir="${ROOT_OUT}/replan${replan}"
    if [[ -f "${out_dir}/summary.json" ]]; then
        echo "skip (summary exists): ${out_dir}"
        continue
    fi

    lane=$(( (replan - 1) % LANES ))
    dependency=()
    if [[ -n "${lane_tail[$lane]}" ]]; then
        dependency=(--dependency="afterany:${lane_tail[$lane]}")
    fi

    job_id=$( \
        MODEL_NAME="$MODEL_NAME" \
        TASK_NAME="$TASK_NAME" \
        SCENE_SET="$SCENE_SET" \
        SCENE_SELECTION="$SCENE_SELECTION" \
        OBJECT_SPLIT="$OBJECT_SPLIT" \
        EPISODES="$EPISODES" \
        DELAYS_MS="$DELAYS_MS" \
        DELAY_DOMAIN="$DELAY_DOMAIN" \
        DELAY_STEP_FREQUENCY="$DELAY_STEP_FREQUENCY" \
        REPLAN_VALUES="$replan" \
        RTC="$RTC" \
        CONTROL_FREQUENCY="$CONTROL_FREQUENCY" \
        HORIZON="$HORIZON" \
        OUTPUT_DIR="$out_dir" \
        sbatch --parsable --partition=ai "${dependency[@]}" \
            --job-name="xiaomi-sim-r${replan}" \
            slurm/xiaomi_latency_experiment.sbatch
    )
    lane_tail[$lane]="$job_id"
    echo "submitted replan=${replan} job=${job_id} lane=${lane} dependency=${lane_tail[$lane]}"
done

echo "submitted sweep; at most ${LANES} jobs can run concurrently"

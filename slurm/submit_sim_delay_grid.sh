#!/usr/bin/env bash
# Submit the post-fix sim-domain cross-model delay grid, one job per
# (model, task) so a failure only costs that slice and jobs can spread across
# the cluster.  Results land in eval_results/simdelay_grid/<model>/<task>/.
#
# Usage:
#   bash slurm/submit_sim_delay_grid.sh                       # all 4 models, 8 tasks
#   MODELS="gr00t_n1_5 pi0_5" TASKS="PickPlaceCounterToCabinet" \
#     bash slurm/submit_sim_delay_grid.sh
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

MODELS="${MODELS:-xiaomi gr00t_n1_5 pi0_5 diffusion_policy}"
TASKS="${TASKS:-PickPlaceCounterToCabinet PickPlaceCounterToSink PickPlaceCabinetToCounter PickPlaceCounterToMicrowave PickPlaceCounterToStove OpenDrawer TurnOffStove CoffeeSetupMug}"
DELAYS_MS="${DELAYS_MS:-0,100,300,500}"
REPLAN_STEPS_LIST="${REPLAN_STEPS_LIST:-5}"
RTC="${RTC:-off}"
EPISODES="${EPISODES:-30}"
SCENE_SET="${SCENE_SET:-pretrain20}"
ROOT_OUT="${ROOT_OUT:-eval_results/simdelay_grid}"
DELAY_DOMAIN="${DELAY_DOMAIN:-sim}"

for model in $MODELS; do
    for task in $TASKS; do
        out_dir="${ROOT_OUT}/${model}/${task}"
        if [[ -f "${out_dir}/summary.json" ]]; then
            echo "skip (done): ${out_dir}"
            continue
        fi
        MODEL_NAME="$model" TASK_NAME="$task" \
        SCENE_SET="$SCENE_SET" DELAYS_MS="$DELAYS_MS" \
        REPLAN_STEPS_LIST="$REPLAN_STEPS_LIST" RTC="$RTC" EPISODES="$EPISODES" \
        DELAY_DOMAIN="$DELAY_DOMAIN" OUTPUT_DIR="$out_dir" \
            sbatch --job-name="simdelay-${model}-${task}" \
                   slurm/xiaomi_latency_experiment.sbatch
    done
done

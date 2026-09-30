#!/usr/bin/env bash
#SBATCH --job-name=robocasa-video
#SBATCH --output=logs/robocasa-video-%j.out
#SBATCH --error=logs/robocasa-video-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Re-render selected episodes from a finished latency benchmark. The replay
# restores each episode's recorded task, condition, seed, and scene; model
# inference is executed again on the allocated GPU.
#
# Example:
#   RUN=eval_results/simdelay_grid/gr00t_n1_5/OpenDrawer \
#   SEEDS=0 DELAYS=0,500 OUT=eval_results/simdelay_grid_analysis/videos/gr00t \
#   sbatch slurm/replay_run_video.sbatch

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ENV_DIR="${ROOT_DIR}/.conda-env"

cd "$ROOT_DIR"
mkdir -p logs
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Missing environment: $ENV_DIR" >&2
    exit 2
fi
if [[ -z "${RUN:-}" ]]; then
    echo "Set RUN to a result slice, model directory, or sweep root." >&2
    exit 2
fi

ARGS=(--run "$RUN" --control-frequency "${CONTROL_FREQUENCY:-0}")
if [[ -n "${DELAY_STEP_FREQUENCY:-}" ]]; then
    ARGS+=(--extra --delay-step-frequency "$DELAY_STEP_FREQUENCY")
fi
if [[ -n "${SCENE_MODE:-}" ]]; then
    ARGS+=(--scene-mode "$SCENE_MODE")
fi
if [[ -n "${OUT:-}" ]]; then
    ARGS+=(--out "$OUT")
fi
if [[ -n "${SEEDS:-}" ]]; then
    ARGS+=(--seeds "$SEEDS")
fi
if [[ -n "${DELAYS:-}" ]]; then
    ARGS+=(--delays "$DELAYS")
fi
if [[ -n "${TASKS:-}" ]]; then
    ARGS+=(--tasks "$TASKS")
fi
if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${HORIZON:-}" ]]; then
    ARGS+=(--horizon "$HORIZON")
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Replay: $RUN"
"$ENV_DIR/bin/python" scripts/replay_run_video.py "${ARGS[@]}"

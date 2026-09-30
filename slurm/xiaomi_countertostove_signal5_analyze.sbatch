#!/usr/bin/env bash
#SBATCH --job-name=xiaomi-countertostove-signal5-analyze
#SBATCH --output=logs/xiaomi-countertostove-signal5-analyze-%j.out
#SBATCH --error=logs/xiaomi-countertostove-signal5-analyze-%j.err
#SBATCH --partition=dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
cd "$ROOT_DIR"
"${ROOT_DIR}/.conda-env/bin/python" scripts/analyze_xiaomi_countertostove_signal5.py \
  --run eval_results/simdelay_grid_analysis/xiaomi_counter_to_stove_signal5/replicate_a_seed000_149 \
  --run eval_results/simdelay_grid_analysis/xiaomi_counter_to_stove_signal5/replicate_b_seed150_299 \
  --output-dir eval_results/simdelay_grid_analysis/xiaomi_counter_to_stove_signal5/analysis

#!/usr/bin/env bash
# Native host Vulkan probe. No Singularity, Podman, or Isaac Python involved.

#SBATCH --job-name=rdj-native-vk
#SBATCH --output=logs/rdj-native-vk-%A.out
#SBATCH --error=logs/rdj-native-vk-%A.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
OUT_DIR="${ROOT_DIR}/logs/rdj-native-vk-${SLURM_JOB_ID:-local}"
mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"
export VULKAN_ICD_FILENAMES="${VK_ICD_FILENAMES}"
export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
export VULKAN_PROBE_ICD="/usr/lib64/libGLX_nvidia.so.0"
export VK_LOADER_DEBUG="all"

{
    echo "host=$(hostname)"
    echo "job=${SLURM_JOB_ID:-local}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader
    echo "vulkan_loader=$(readlink -f /usr/lib64/libvulkan.so.1)"
    echo "nvidia_icd=$(cat "${VK_ICD_FILENAMES}")"
    ldd /usr/lib64/libGLX_nvidia.so.0
} | tee "${OUT_DIR}/host.txt"

python3 -u container/vulkan_probe.py 2>&1 | tee "${OUT_DIR}/probe.txt"
printf 'complete\n' > "${OUT_DIR}/COMPLETE"

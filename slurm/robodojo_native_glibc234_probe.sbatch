#!/usr/bin/env bash
# Run Isaac Sim 5.1 against the host glibc instead of a container libc.
# The prebuilt runtime is only bind-mounted at its original installation path.

#SBATCH --job-name=rdj-native-glibc234
#SBATCH --output=logs/rdj-native-glibc234-%j.out
#SBATCH --error=logs/rdj-native-glibc234-%j.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
RUNTIME_DIR="${ROBODOJO_RUNTIME_DIR:-${ROOT_DIR}/.runtime/robodojo-singularity}"
PREFIX="/opt/robo-state/miniconda3/envs/RoboDojo"
OUT_DIR="${ROOT_DIR}/logs/rdj-native-glibc234-${SLURM_JOB_ID:-local}"
HOST_ICD="${HOST_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

for required in \
    "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" \
    "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/lib/libstdc++.so.6" \
    "${HOST_ICD}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-local}"
echo "host_glibc=$(ldd --version | head -n 1)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader

# Do not alter CUDA_VISIBLE_DEVICES: Slurm owns GPU isolation. Bubblewrap only
# restores the absolute install prefix expected by the existing Isaac runtime.
bwrap \
    --ro-bind / / \
    --tmpfs /tmp \
    --tmpfs /var/tmp \
    --tmpfs /opt \
    --dir /opt/robo-state \
    --bind "${RUNTIME_DIR}" /opt/robo-state \
    --proc /proc \
    --dev-bind /dev /dev \
    --setenv OMNI_KIT_ACCEPT_EULA YES \
    --setenv ACCEPT_EULA Y \
    --setenv PRIVACY_CONSENT Y \
    --setenv PYTHONUNBUFFERED 1 \
    --setenv TMPDIR /tmp \
    --setenv LD_LIBRARY_PATH "${PREFIX}/lib:/usr/lib64" \
    --setenv VK_ICD_FILENAMES "${HOST_ICD}" \
    --setenv VK_DRIVER_FILES "${HOST_ICD}" \
    "${PREFIX}/bin/python" - <<'PY'
import platform
import sys

print("runtime_glibc=", platform.libc_ver(), flush=True)
print("runtime_python=", sys.executable, flush=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "enable_cameras": False})
print("simulation_app=ready", flush=True)
app.close()
print("simulation_app=closed", flush=True)
PY

printf 'complete\n' > "${OUT_DIR}/COMPLETE"

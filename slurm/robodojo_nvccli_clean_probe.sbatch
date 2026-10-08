#!/usr/bin/env bash
# Clean user-local nvccli probe. This intentionally avoids hand-managed
# Vulkan/GLX binds so nvidia-container-cli owns the driver injection.

#SBATCH --job-name=rdj-nvccli-clean
#SBATCH --output=logs/rdj-nvccli-clean-%A.out
#SBATCH --error=logs/rdj-nvccli-clean-%A.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
BASE_IMAGE="${ROBODOJO_NVCCLI_BASE_IMAGE:-${ROOT_DIR}/.runtime/ubuntu-24.04.nvccli-base}"
RUNTIME_DIR="${ROBODOJO_RUNTIME_DIR:-${ROOT_DIR}/.runtime/robodojo-singularity}"
LOCAL_SINGULARITY="${ROBODOJO_LOCAL_SINGULARITY:-${ROOT_DIR}/.runtime/singularity-ce-4.5.1/bin/singularity}"
LOCAL_CONFIG="${ROBODOJO_LOCAL_SINGULARITY_CONFIG:-${ROOT_DIR}/.runtime/singularity-ce-4.5.1/user-config/singularity.conf}"
OUT_DIR="${ROOT_DIR}/logs/rdj-nvccli-clean-${SLURM_JOB_ID:-local}"
IMAGE="${OUT_DIR}/rootfs-$(hostname)-$(date +%s%N)"
HOST_VULKAN_LIB="${HOST_VULKAN_LIB:-/usr/lib64/libvulkan.so.1}"
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-/usr/lib64}"
HOST_GRAPHICS_LIBS=(
    libGL.so.1 libGLX.so.0 libGLX_nvidia.so.0 libGLdispatch.so.0
    libX11.so.6 libXext.so.6 libxcb.so.1 libXau.so.6 libXt.so.6
    libXrandr.so.2 libXfixes.so.3 libXrender.so.1 libXi.so.6
    libSM.so.6 libICE.so.6 libuuid.so.1
)
GRAPHICS_BINDS=()
for graphics_lib in "${HOST_GRAPHICS_LIBS[@]}"; do
    graphics_src="${NVIDIA_LIB_DIR}/${graphics_lib}"
    [[ -e "${graphics_src}" ]] || continue
    graphics_dst="/usr/lib/x86_64-linux-gnu/${graphics_lib}"
    GRAPHICS_BINDS+=( -B "$(readlink -f "${graphics_src}"):${graphics_dst}:ro" )
done

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

for required in "${BASE_IMAGE}" "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" \
                "${LOCAL_SINGULARITY}" "${LOCAL_CONFIG}" "${HOST_VULKAN_LIB}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

# Slurm exposes only the allocated GPU to nvidia-smi in this job. Passing that
# UUID avoids selecting a physical index while preventing nvccli from falling
# back to its legacy NVIDIA_VISIBLE_DEVICES=all behavior.
ALLOCATED_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
[[ -n "${ALLOCATED_GPU_UUID}" ]] || { echo "Could not determine allocated GPU UUID" >&2; exit 2; }

cp -a "${BASE_IMAGE}" "${IMAGE}"
mkdir -p "${IMAGE}/lib/x86_64-linux-gnu" "${IMAGE}/usr/lib/x86_64-linux-gnu"
touch "${IMAGE}/lib/x86_64-linux-gnu/libvulkan.so.1"
touch "${IMAGE}/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
for graphics_lib in "${HOST_GRAPHICS_LIBS[@]}"; do
    [[ -e "${NVIDIA_LIB_DIR}/${graphics_lib}" ]] || continue
    touch "${IMAGE}/usr/lib/x86_64-linux-gnu/${graphics_lib}"
done

export NVIDIA_VISIBLE_DEVICES="${ALLOCATED_GPU_UUID}"
export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="${ALLOCATED_GPU_UUID}"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video"
export SINGULARITYENV_NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}"
export SINGULARITYENV_OMNI_KIT_ACCEPT_EULA="YES"
export SINGULARITYENV_PYTHONUNBUFFERED="1"

# Ensure no settings from a parent shell force a hand-picked Vulkan loader or
# ICD. nvccli should inject the complete host driver stack for this job.
unset VK_ICD_FILENAMES VULKAN_ICD_FILENAMES VK_DRIVER_FILES LD_PRELOAD
unset SINGULARITYENV_VK_ICD_FILENAMES SINGULARITYENV_VULKAN_ICD_FILENAMES
unset SINGULARITYENV_VK_DRIVER_FILES SINGULARITYENV_LD_PRELOAD

{
    echo "host=$(hostname)"
    echo "job=${SLURM_JOB_ID:-local}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    echo "allocated_gpu_uuid=${ALLOCATED_GPU_UUID}"
    echo "local_singularity=$(${LOCAL_SINGULARITY} --version)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader
} | tee "${OUT_DIR}/host.txt"

"${LOCAL_SINGULARITY}" --config "${LOCAL_CONFIG}" exec --nvccli --writable --no-home \
    -B "${ROOT_DIR}:/mnt" \
    -B "${RUNTIME_DIR}:/opt/robo-state" \
    -B "${HOST_VULKAN_LIB}:/lib/x86_64-linux-gnu/libvulkan.so.1:ro" \
    -B "${HOST_VULKAN_LIB}:/usr/lib/x86_64-linux-gnu/libvulkan.so.1:ro" \
    "${GRAPHICS_BINDS[@]}" \
    "${IMAGE}" bash -lc '
        set -euo pipefail
        source /opt/robo-state/miniconda3/etc/profile.d/conda.sh
        conda activate RoboDojo
        echo "container_cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
        echo "container_nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES-<unset>}"
        echo "container_nvidia_driver_capabilities=${NVIDIA_DRIVER_CAPABILITIES-<unset>}"
        export VK_ICD_FILENAMES=/mnt/container/nvidia_icd_ubuntu_bind.json
        export VULKAN_ICD_FILENAMES="$VK_ICD_FILENAMES"
        export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
        python - <<"PY"
import torch
print("torch_version=", torch.__version__)
print("torch_cuda_version=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0=", torch.cuda.get_device_name(0))
PY
        python -u /mnt/container/vulkan_probe.py
        python - <<"PY"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": False})
print("isaac_simulation_app=ready")
app.close()
print("isaac_simulation_app=closed")
PY
    ' 2>&1 | tee "${OUT_DIR}/container.txt"

printf 'complete\n' > "${OUT_DIR}/COMPLETE"

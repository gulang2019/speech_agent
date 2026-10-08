#!/usr/bin/env bash
# User-local SingularityCE + nvidia-container-cli probe.
# This job is intentionally independent of the experiment sweep.

#SBATCH --job-name=rdj-nvccli-probe
#SBATCH --output=logs/rdj-nvccli-probe-%A.out
#SBATCH --error=logs/rdj-nvccli-probe-%A.err
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
OUT_DIR="${ROOT_DIR}/logs/rdj-nvccli-probe-${SLURM_JOB_ID:-local}"
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-/usr/lib64}"
HOST_VULKAN_LIB="${HOST_VULKAN_LIB:-${NVIDIA_LIB_DIR}/libvulkan.so.1}"
HOST_GRAPHICS_LIBS=(
    libGL.so.1 libGLX.so.0 libGLdispatch.so.0
    libX11.so.6 libXext.so.6 libxcb.so.1 libXau.so.6
    libXt.so.6 libXrandr.so.2 libXfixes.so.3 libXrender.so.1
    libXi.so.6 libSM.so.6 libICE.so.6 libuuid.so.1
)

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

for required in "${BASE_IMAGE}" "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" \
                "${LOCAL_SINGULARITY}" "${LOCAL_CONFIG}" \
                "${HOST_VULKAN_LIB}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

# nvccli writes driver-specific links and an ld cache. Keep those writes in a
# one-attempt rootfs so retries on nodes with a different driver stay isolated.
IMAGE="${OUT_DIR}/nvccli-rootfs-$(hostname)-$(date +%s%N)"
cp -a "${BASE_IMAGE}" "${IMAGE}"
mkdir -p "${IMAGE}/usr/local/lib"

GRAPHICS_BINDS=()
for graphics_lib in "${HOST_GRAPHICS_LIBS[@]}"; do
    graphics_src="$(readlink -f "${NVIDIA_LIB_DIR}/${graphics_lib}")"
    graphics_dst="/usr/lib/x86_64-linux-gnu/${graphics_lib}"
    [[ -f "${graphics_src}" ]] || { echo "Missing graphics library: ${graphics_src}" >&2; exit 2; }
    GRAPHICS_BINDS+=( -B "${graphics_src}:${graphics_dst}:ro" )
done

NVIDIA_ICD_BINDS=()
NVIDIA_GLX_SRC="$(readlink -f "${NVIDIA_LIB_DIR}/libGLX_nvidia.so.0")"
touch "${IMAGE}/usr/local/lib/libGLX_nvidia.so.0"
NVIDIA_ICD_BINDS+=( -B "${NVIDIA_GLX_SRC}:/usr/local/lib/libGLX_nvidia.so.0:ro" )
while IFS= read -r driver_src; do
    driver_base="$(basename "${driver_src}")"
    driver_dst="/usr/local/lib/${driver_base}"
    [[ -e "${IMAGE}${driver_dst}" ]] || touch "${IMAGE}${driver_dst}"
    NVIDIA_ICD_BINDS+=( -B "${driver_src}:${driver_dst}:ro" )
done < <(
    ldd "${HOST_NVIDIA_GLX_LIB:-${NVIDIA_LIB_DIR}/libGLX_nvidia.so.0}" 2>/dev/null \
        | awk '$3 ~ /^\// && $3 ~ /libnvidia-/ {print $3}' \
        | sort -u
)
for driver_pattern in libnvidia-glvkspirv.so.* libnvidia-rtcore.so.*; do
    while IFS= read -r driver_src; do
        driver_base="$(basename "${driver_src}")"
        driver_dst="/usr/local/lib/${driver_base}"
        [[ -e "${IMAGE}${driver_dst}" ]] || touch "${IMAGE}${driver_dst}"
        NVIDIA_ICD_BINDS+=( -B "${driver_src}:${driver_dst}:ro" )
    done < <(find "${NVIDIA_LIB_DIR}" -maxdepth 1 -type f -name "${driver_pattern}" -print 2>/dev/null | sort -u)
done

BUNDLED_VULKAN_LOADER="$(find "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/lib/python3.11/site-packages/isaacsim/extscache" \
    -path '*/omni.gpu_foundation-*/bin/deps/libvulkan.so.1.*' -type f -print -quit 2>/dev/null || true)"
[[ -n "${BUNDLED_VULKAN_LOADER}" ]] || { echo "Isaac bundled Vulkan loader not found" >&2; exit 2; }
BUNDLED_VULKAN_DIR="$(dirname "${BUNDLED_VULKAN_LOADER}")"
BUNDLED_VULKAN_CONTAINER_DIR="/opt/robo-state${BUNDLED_VULKAN_DIR#${RUNTIME_DIR}}"
BUNDLED_VULKAN_DEST="/opt/robo-state${BUNDLED_VULKAN_LOADER#${RUNTIME_DIR}}"

export PATH="/usr/libexec/singularity/bin:${PATH}"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video"
export SINGULARITYENV_NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}"
export SINGULARITYENV_OMNI_KIT_ACCEPT_EULA="YES"
export SINGULARITYENV_PYTHONUNBUFFERED="1"

{
    echo "host=$(hostname)"
    echo "job=${SLURM_JOB_ID:-local}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    echo "local_singularity=$(${LOCAL_SINGULARITY} --version)"
    echo "image=${IMAGE}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "${OUT_DIR}/host.txt"

"${LOCAL_SINGULARITY}" --config "${LOCAL_CONFIG}" exec --nvccli --writable --no-home \
    -B "${ROOT_DIR}:/mnt" \
    -B "${RUNTIME_DIR}:/opt/robo-state" \
    -B "${HOST_VULKAN_LIB}:/usr/lib/x86_64-linux-gnu/libvulkan.so.1:ro" \
    -B "${HOST_VULKAN_LIB}:${BUNDLED_VULKAN_DEST}:ro" \
    -B "${HOST_VULKAN_LIB}:${BUNDLED_VULKAN_CONTAINER_DIR}/libvulkan.so.1:ro" \
    -B "${HOST_VULKAN_LIB}:${BUNDLED_VULKAN_CONTAINER_DIR}/libvulkan.so:ro" \
    "${GRAPHICS_BINDS[@]}" \
    "${NVIDIA_ICD_BINDS[@]}" \
    "${IMAGE}" bash -lc '
        set -euo pipefail
        source /opt/robo-state/miniconda3/etc/profile.d/conda.sh
        conda activate RoboDojo
        echo "container_cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
        echo "nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES-<unset>}"
        echo "nvidia_driver_capabilities=${NVIDIA_DRIVER_CAPABILITIES-<unset>}"
        python - <<"PY"
import torch
print("torch_version=", torch.__version__)
print("torch_cuda_version=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0=", torch.cuda.get_device_name(0))
PY
        export VK_ICD_FILENAMES=/mnt/container/nvidia_icd_nvccli.json
        export VULKAN_ICD_FILENAMES="$VK_ICD_FILENAMES"
        export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
        export VK_LOADER_DEBUG="all"
        export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libvulkan.so.1
        echo "vulkan_icd_file=$(cat "$VK_ICD_FILENAMES")"
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

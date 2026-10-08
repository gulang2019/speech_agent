#!/usr/bin/env bash
# Rootless Podman + NVIDIA CDI probe. This is independent of Singularity.

#SBATCH --job-name=rdj-podman-cdi
#SBATCH --output=logs/rdj-podman-cdi-%A.out
#SBATCH --error=logs/rdj-podman-cdi-%A.err
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
OUT_DIR="${ROOT_DIR}/logs/rdj-podman-cdi-${SLURM_JOB_ID:-local}"
ROOTFS="${OUT_DIR}/rootfs-$(hostname)-$(date +%s%N)"
XDG_RUNTIME_DIR="${OUT_DIR}/xdg-runtime"
PODMAN_RUNROOT="${OUT_DIR}/podman-runroot"
PODMAN_GRAPHROOT="${OUT_DIR}/podman-graphroot"
PODMAN_CONF="${OUT_DIR}/storage.conf"
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-/usr/lib64}"
HOST_VULKAN_LIB="${HOST_VULKAN_LIB:-${NVIDIA_LIB_DIR}/libvulkan.so.1}"
HOST_GRAPHICS_LIBS=(
    libGL.so.1 libGLX.so.0 libGLdispatch.so.0
    libX11.so.6 libXext.so.6 libxcb.so.1 libXau.so.6 libXt.so.6
    libXrandr.so.2 libXfixes.so.3 libXrender.so.1 libXi.so.6
    libSM.so.6 libICE.so.6 libuuid.so.1
)
GRAPHICS_BINDS=()

mkdir -p "${OUT_DIR}" "${XDG_RUNTIME_DIR}" "${PODMAN_RUNROOT}" "${PODMAN_GRAPHROOT}"
chmod 700 "${XDG_RUNTIME_DIR}"
cd "${ROOT_DIR}"

for required in "${BASE_IMAGE}" "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" \
                /usr/bin/podman /etc/cdi/nvidia.yaml /usr/bin/fuse-overlayfs \
                "${HOST_VULKAN_LIB}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

ALLOCATED_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
[[ -n "${ALLOCATED_GPU_UUID}" ]] || { echo "Could not determine allocated GPU UUID" >&2; exit 2; }
CDI_DEVICE="nvidia.com/gpu=${ALLOCATED_GPU_UUID}"

cp -a "${BASE_IMAGE}" "${ROOTFS}"
mkdir -p "${ROOTFS}/lib/x86_64-linux-gnu" "${ROOTFS}/usr/lib/x86_64-linux-gnu"
mkdir -p "${ROOTFS}/usr/share/vulkan/icd.d" "${ROOTFS}/etc/vulkan/icd.d"
cp /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
    "${ROOTFS}/etc/vulkan/icd.d/nvidia_icd.x86_64.json"
cp /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
    "${ROOTFS}/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"
touch "${ROOTFS}/lib/x86_64-linux-gnu/libvulkan.so.1"
touch "${ROOTFS}/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
for graphics_lib in "${HOST_GRAPHICS_LIBS[@]}"; do
    graphics_src="${NVIDIA_LIB_DIR}/${graphics_lib}"
    [[ -e "${graphics_src}" ]] || continue
    graphics_dst="/usr/lib/x86_64-linux-gnu/${graphics_lib}"
    mkdir -p "${ROOTFS}$(dirname "${graphics_dst}")"
    touch "${ROOTFS}${graphics_dst}"
    GRAPHICS_BINDS+=( -v "$(readlink -f "${graphics_src}"):${graphics_dst}:ro" )
done

# Keep all rootless Podman state inside this job directory. The system cache
# has a stale boot-id marker on this host and is not modified here.
cat > "${PODMAN_CONF}" <<EOF
[storage]
driver = "overlay"
runroot = "${PODMAN_RUNROOT}"
graphroot = "${PODMAN_GRAPHROOT}"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
EOF

export XDG_RUNTIME_DIR
export TMPDIR="${OUT_DIR}/tmp"
export PODMAN_TMPDIR="${OUT_DIR}/tmp"
export CONTAINERS_STORAGE_CONF="${PODMAN_CONF}"
export NVIDIA_VISIBLE_DEVICES="${ALLOCATED_GPU_UUID}"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video"
export OMNI_KIT_ACCEPT_EULA="YES"
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}"

{
    echo "host=$(hostname)"
    echo "job=${SLURM_JOB_ID:-local}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    echo "allocated_gpu_uuid=${ALLOCATED_GPU_UUID}"
    echo "cdi_device=${CDI_DEVICE}"
    podman --log-level=warn info --format '{{.Version.Version}}'
    nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader
} | tee "${OUT_DIR}/host.txt"

podman --log-level=info run --rm \
    --device "${CDI_DEVICE}" \
    --security-opt=label=disable \
    --network=host \
    --shm-size=2g \
    -e NVIDIA_VISIBLE_DEVICES="${ALLOCATED_GPU_UUID}" \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}" \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.x86_64.json \
    -e VULKAN_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.x86_64.json \
    -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.x86_64.json \
    -v "${ROOT_DIR}:/mnt:ro" \
    -v "${RUNTIME_DIR}:/opt/robo-state:ro" \
    -v "${HOST_VULKAN_LIB}:/lib/x86_64-linux-gnu/libvulkan.so.1:ro" \
    -v "${HOST_VULKAN_LIB}:/usr/lib/x86_64-linux-gnu/libvulkan.so.1:ro" \
    "${GRAPHICS_BINDS[@]}" \
    --rootfs "${ROOTFS}" \
    bash -lc '
        set -euo pipefail
        echo "container_cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
        echo "container_nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES-<unset>}"
        echo "container_nvidia_driver_capabilities=${NVIDIA_DRIVER_CAPABILITIES-<unset>}"
        export LD_LIBRARY_PATH="/usr/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
        export VULKAN_PROBE_ICD=/usr/lib64/libGLX_nvidia.so.0
        ls -l /dev/nvidia* /dev/dri/* 2>/dev/null || true
        source /opt/robo-state/miniconda3/etc/profile.d/conda.sh
        conda activate RoboDojo
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

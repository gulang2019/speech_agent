#!/usr/bin/env bash
# RoboDojo pour_liquid_into_cup x Xiaomi compatibility bridge smoke sweep.
# One array element is one (sim-delay, replan) cell; 4 episodes/cell.

#SBATCH --job-name=rdj-xiaomi-smoke
#SBATCH --output=logs/rdj-xiaomi-smoke-%A_%a.out
#SBATCH --error=logs/rdj-xiaomi-smoke-%A_%a.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-8%2

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ROBO_DIR="${ROBODOJO_DIR:-${ROOT_DIR}/.deps/RoboDojo}"
XPOLICY_DIR="${XPOLICYLAB_DIR:-${ROOT_DIR}/.deps/XPolicyLab}"
POLICY_PYTHON="${POLICY_PYTHON:-${ROOT_DIR}/.conda-env/bin/python}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/xiaomi-robotics-1-robocasa}"
ROOT_OUT="${ROOT_OUT:-${ROOT_DIR}/eval_results/robodojo_xiaomi_delay_replan_smoke}"
EPISODES="${EPISODES:-4}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
SIM_IMAGE="${ROBODOJO_BASE_IMAGE:-${ROOT_DIR}/.runtime/ubuntu-24.04.sif}"
SIM_RUNTIME_DIR="${ROBODOJO_RUNTIME_DIR:-${ROOT_DIR}/.runtime/robodojo-singularity}"
SIM_ROOT="${ROBODOJO_CONTAINER_ROOT:-/workspace/robo}"
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-/usr/lib64}"
HOST_VULKAN_LIB="${HOST_VULKAN_LIB:-${NVIDIA_LIB_DIR}/libvulkan.so.1}"
HOST_NVIDIA_GLX_LIB="${HOST_NVIDIA_GLX_LIB:-${NVIDIA_LIB_DIR}/libGLX_nvidia.so.0}"
HOST_LVP_LIB="${HOST_LVP_LIB:-${NVIDIA_LIB_DIR}/libvulkan_lvp.so}"
VULKAN_ICD_MODE="${ROBODOJO_VULKAN_ICD:-nvidia}"
NVIDIA_DRIVER_LIB_BINDS=()
while IFS= read -r driver_lib; do
    [[ -z "${driver_lib}" ]] && continue
    NVIDIA_DRIVER_LIB_BINDS+=( -B "${driver_lib}:${driver_lib}:ro" )
done < <(
    ldd "${HOST_NVIDIA_GLX_LIB}" 2>/dev/null \
        | awk '{print $3}' \
        | rg '^/(usr/)?lib64/(libGLX_nvidia|libnvidia-(glsi|tls|glcore|gpucomp)|libGLX|libGLdispatch|libX11|libXext|libxcb|libXau)' \
        | sort -u
)
VULKAN_DRIVER_LIB_BINDS=()
VULKAN_ICD_JSON="${SIM_ROOT}/container/nvidia_icd_host_bind.json"
VULKAN_DRIVER_LD_PATH="/usr/lib64"
if [[ "${VULKAN_ICD_MODE}" == "lvp" ]]; then
    VULKAN_ICD_JSON="${SIM_ROOT}/container/lvp_icd_host_bind.json"
    VULKAN_DRIVER_LD_PATH="/host-lvp:/usr/lib64"
    while IFS= read -r driver_lib; do
        [[ -z "${driver_lib}" ]] && continue
        driver_base="$(basename "${driver_lib}")"
        case "${driver_base}" in
            libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libgcc_s.so*|libstdc++.so*)
                continue
                ;;
        esac
        VULKAN_DRIVER_LIB_BINDS+=( -B "${driver_lib}:/host-lvp/${driver_base}:ro" )
    done < <(
        ldd "${HOST_LVP_LIB}" 2>/dev/null \
            | awk '{print $3}' \
            | rg '^/' \
            | sort -u
    )
    VULKAN_DRIVER_LIB_BINDS+=( -B "${HOST_LVP_LIB}:/host-lvp/libvulkan_lvp.so:ro" )
fi
BUNDLED_VULKAN_LOADER="$(find "${SIM_RUNTIME_DIR}/miniconda3/envs/RoboDojo/lib/python3.11/site-packages/isaacsim/extscache" -path '*/omni.gpu_foundation-*/bin/deps/libvulkan.so.1.*' -type f -print -quit 2>/dev/null || true)"
BUNDLED_VULKAN_DEST=""
BUNDLED_VULKAN_DIR=""
if [[ -n "${BUNDLED_VULKAN_LOADER}" ]]; then
    BUNDLED_VULKAN_DEST="/opt/robo-state${BUNDLED_VULKAN_LOADER#${SIM_RUNTIME_DIR}}"
    BUNDLED_VULKAN_DIR="${BUNDLED_VULKAN_DEST%/*}"
fi

DELAYS_MS=(0 100 300)
REPLAN_STEPS=(1 5 10)
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DELAY_MS="${DELAYS_MS[$((INDEX / ${#REPLAN_STEPS[@]}))]}"
REPLAN_STEPS_VALUE="${REPLAN_STEPS[$((INDEX % ${#REPLAN_STEPS[@]}))]}"
CELL="delay${DELAY_MS}ms/replan${REPLAN_STEPS_VALUE}"
OUTPUT_DIR="${ROOT_OUT}/${CELL}"
METRICS_FILE="${OUTPUT_DIR}/timing.jsonl"
POLICY_NAME="RoboDojo_Xiaomi_Compat"
TASK_NAME="pour_liquid_into_cup"
PORT=$((20000 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-1}} % 20000) + INDEX))

cd "${ROOT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/tmp"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ROBODOJO_SIM_DEVICE="${ROBODOJO_SIM_DEVICE:-cpu}"
export ROBODOJO_DELAY_MS="${DELAY_MS}"
export ROBODOJO_REPLAN_STEPS="${REPLAN_STEPS_VALUE}"
export ROBODOJO_DELAY_STEP_FREQUENCY="25"
export ROBODOJO_SWEEP_METRICS="${METRICS_FILE}"
export ROBODOJO_RUN_ID="rdj-xiaomi-${SLURM_ARRAY_JOB_ID:-local}-${INDEX}"

for required in "${ROBO_DIR}/scripts/eval_policy.sh" \
                "${XPOLICY_DIR}/client_server/ws/model_server.py" \
                "${MODEL_PATH}" "${SIM_IMAGE}" "${SIM_RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" "${POLICY_PYTHON}" \
                "${HOST_VULKAN_LIB}" "${HOST_NVIDIA_GLX_LIB}" "${BUNDLED_VULKAN_LOADER}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required path: ${required}" >&2
        echo "Set ROBODOJO_DIR, XPOLICYLAB_DIR, ROBODOJO_SINGULARITY_IMAGE, POLICY_PYTHON, or MODEL_PATH." >&2
        exit 2
    fi
done

ENV_MARKER="${ROBODOJO_ENV_MARKER:-${ROOT_DIR}/.runtime/RoboDojo.ready}"
if [[ "${ENV_MARKER}" == "${ROOT_DIR}/.runtime/RoboDojo.ready" && -f "${ROOT_DIR}/.runtime/RoboDojo-singularity.ready" ]]; then
    ENV_MARKER="${ROOT_DIR}/.runtime/RoboDojo-singularity.ready"
fi
if [[ ! -f "${ENV_MARKER}" ]]; then
    echo "RoboDojo environment marker missing: ${ENV_MARKER}" >&2
    echo "Submit slurm/setup_robodojo_singularity_gpu_scavenger.sbatch first." >&2
    exit 2
fi

# Install the experiment-only deploy entry point into the XPolicyLab checkout.
POLICY_PACKAGE="${XPOLICY_DIR}/policy/${POLICY_NAME}"
mkdir -p "${POLICY_PACKAGE}"
cp "${ROOT_DIR}/scripts/robodojo_delay_replan_deploy.py" "${POLICY_PACKAGE}/deploy.py"
printf '__all__ = []\n' > "${POLICY_PACKAGE}/__init__.py"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Host: $(hostname)"
echo "Array: ${SLURM_ARRAY_JOB_ID:-local}_${INDEX}"
echo "Cell: ${CELL}; episodes=${EPISODES}"
echo "Delay semantics: sim-domain at 25 Hz; replan=${REPLAN_STEPS_VALUE}"
echo "RoboDojo: ${ROBO_DIR}"
echo "XPolicyLab: ${XPOLICY_DIR}"
echo "Simulator image: ${SIM_IMAGE}"
echo "Policy Python: ${POLICY_PYTHON}"
echo "Vulkan ICD: ${VULKAN_ICD_MODE} (${VULKAN_ICD_JSON})"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

export PYTHONPATH="${ROOT_DIR}:${ROBO_DIR}:${XPOLICY_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
echo "Host CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${POLICY_PYTHON}" "${ROOT_DIR}/scripts/robodojo_xiaomi_server.py" \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    > "${OUTPUT_DIR}/policy_server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Policy server exited before becoming ready" >&2
        cat "${OUTPUT_DIR}/policy_server.log" >&2 || true
        exit 1
    fi
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
        break
    fi
    sleep 5
done
if ! timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    echo "Policy server did not listen on ${PORT}" >&2
    cat "${OUTPUT_DIR}/policy_server.log" >&2 || true
    exit 1
fi

export PYTHONPATH="${ROBO_DIR}:${XPOLICY_DIR}:${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH}"
export SINGULARITYENV_LD_LIBRARY_PATH="${VULKAN_DRIVER_LD_PATH}:/usr/lib/x86_64-linux-gnu:/opt/robo-state/miniconda3/envs/RoboDojo/lib:/.singularity.d/libs"
export SINGULARITYENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
export SINGULARITYENV_VK_ICD_FILENAMES="${VULKAN_ICD_JSON}"
export SINGULARITYENV_VULKAN_ICD_FILENAMES="${VULKAN_ICD_JSON}"
export SINGULARITYENV_VK_DRIVER_FILES="${SIM_ROOT}/container/nvidia_icd_host_bind.json"
export SINGULARITYENV_NVIDIA_DRIVER_CAPABILITIES="all"

"${SINGULARITY_BIN}" exec --nv \
    -B "${ROOT_DIR}:${SIM_ROOT}" \
    -B "${SIM_RUNTIME_DIR}:/opt/robo-state" \
    "${NVIDIA_DRIVER_LIB_BINDS[@]}" \
    "${VULKAN_DRIVER_LIB_BINDS[@]}" \
    -B "${HOST_VULKAN_LIB}:/usr/lib/x86_64-linux-gnu/libvulkan.so.1:ro" \
    -B "${HOST_NVIDIA_GLX_LIB}:/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:ro" \
    -B "${HOST_VULKAN_LIB}:${BUNDLED_VULKAN_DIR}/libvulkan.so:ro" \
    -B "${HOST_VULKAN_LIB}:${BUNDLED_VULKAN_DIR}/libvulkan.so.1:ro" \
    -B "${HOST_VULKAN_LIB}:${BUNDLED_VULKAN_DEST}:ro" \
    -B /usr/share/vulkan:/usr/share/vulkan:ro \
    -B /etc/vulkan:/etc/vulkan:ro \
    "${SIM_IMAGE}" bash -lc \
    "source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && unset CUDA_VISIBLE_DEVICES && export ROBODOJO_PRESERVE_CUDA_VISIBLE_DEVICES=1 && export LD_LIBRARY_PATH='${VULKAN_DRIVER_LD_PATH}:/usr/lib/x86_64-linux-gnu':\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 && export TMPDIR='${SIM_ROOT}/${OUTPUT_DIR#${ROOT_DIR}}/tmp' && export TMP='\$TMPDIR' && export TEMP='\$TMPDIR' && export ROBODOJO_SIM_DEVICE='${ROBODOJO_SIM_DEVICE}' && export VK_ICD_FILENAMES='${VULKAN_ICD_JSON}' && export VULKAN_ICD_FILENAMES=\$VK_ICD_FILENAMES && export VK_DRIVER_FILES=\$VK_ICD_FILENAMES && cd '${SIM_ROOT}' && export PYTHONPATH='${SIM_ROOT}/.deps/RoboDojo:${SIM_ROOT}/.deps/RoboDojo/XPolicyLab:${SIM_ROOT}:${SIM_ROOT}/scripts' && export ROBODOJO_DELAY_MS='${DELAY_MS}' && export ROBODOJO_REPLAN_STEPS='${REPLAN_STEPS_VALUE}' && export ROBODOJO_DELAY_STEP_FREQUENCY=25 && export ROBODOJO_SWEEP_METRICS='${SIM_ROOT}/${METRICS_FILE#${ROOT_DIR}/}' && export ROBODOJO_RUN_ID='rdj-xiaomi-${SLURM_ARRAY_JOB_ID:-local}-${INDEX}' && EVAL_NUM='${EPISODES}' bash '${SIM_ROOT}/.deps/RoboDojo/scripts/eval_policy.sh' --root_dir '${SIM_ROOT}/.deps/RoboDojo' --task_name '${TASK_NAME}' --env_cfg_type arx_x5 --device_id 0 --policy_name '${POLICY_NAME}' --port '${PORT}' --eval_batch false --additional_info 'xiaomi_compat_${CELL}' --seed '${SEED:-0}' --host 127.0.0.1 --protocol ws --policy_server_url 'ws://127.0.0.1:${PORT}'"

if [[ ! -s "${METRICS_FILE}" ]]; then
    echo "Simulator exited without recording episode metrics: ${METRICS_FILE}" >&2
    exit 1
fi

printf 'complete\n' > "${OUTPUT_DIR}/COMPLETE"
echo "Completed: ${OUTPUT_DIR}"

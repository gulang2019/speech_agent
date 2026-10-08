#!/usr/bin/env bash
# Deploy/check/run RoboDojo pour_liquid_into_cup with the Xiaomi bridge.
# Outside Slurm this script submits itself to gpu-scavenger.

#SBATCH --job-name=rdj-xiaomi-oneclick
#SBATCH --output=logs/rdj-xiaomi-oneclick-%j.out
#SBATCH --error=logs/rdj-xiaomi-oneclick-%j.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="${ROBODOJO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBO_DIR="${ROBODOJO_DIR:-${ROOT_DIR}/.deps/RoboDojo}"
XPOLICY_DIR="${XPOLICYLAB_DIR:-${ROOT_DIR}/.deps/XPolicyLab}"
POLICY_PYTHON="${POLICY_PYTHON:-${ROOT_DIR}/.conda-env/bin/python}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/xiaomi-robotics-1-robocasa}"
RUNTIME_DIR="${ROBODOJO_RUNTIME_DIR:-${ROOT_DIR}/.runtime/robodojo-singularity}"
BASE_IMAGE="${ROBODOJO_BASE_IMAGE:-${ROOT_DIR}/.runtime/ubuntu-24.04.sif}"
ROOTFS_PREFIX="${ROBODOJO_ROOTFS_PREFIX:-${ROOT_DIR}/.runtime/ubuntu-24.04-rootfs}"
OUTPUT_ROOT="${ROBODOJO_OUTPUT_DIR:-${ROOT_DIR}/eval_results/robodojo_xiaomi_delay_replan_oneclick}"
EPISODES="${EPISODES:-1}"
DELAYS_MS="${DELAYS_MS:-0,100,300}"
REPLAN_STEPS_LIST="${REPLAN_STEPS_LIST:-1,5,10}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
TASK_NAME="pour_liquid_into_cup"
POLICY_NAME="RoboDojo_Xiaomi_Compat"
PORT="${ROBODOJO_POLICY_PORT:-$((20000 + (${SLURM_JOB_ID:-1} % 20000)))}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "${ROOT_DIR}"
    mkdir -p logs
    job_id="$(sbatch --parsable "$0")"
    echo "Submitted RoboDojo one-click job: ${job_id}"
    echo "Logs: ${ROOT_DIR}/logs/rdj-xiaomi-oneclick-${job_id}.out"
    exit 0
fi

cd "${ROOT_DIR}"
mkdir -p logs "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}" PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

log() { printf '[robodojo-oneclick] %s\n' "$*"; }
die() { printf '[robodojo-oneclick] ERROR: %s\n' "$*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

for command_name in bwrap singularity unsquashfs git nvidia-smi sha256sum timeout; do
    need "${command_name}"
done
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is empty; use Slurm"

clone_if_missing() {
    local path="$1" url="$2"
    [[ -d "${path}/.git" ]] && return
    mkdir -p "$(dirname "${path}")"
    log "Cloning ${url}"
    git clone --filter=blob:none "${url}" "${path}"
}

ensure_model() {
    [[ -f "${MODEL_PATH}/config.json" ]] && return
    [[ "${DOWNLOAD_MODEL}" == 1 ]] || die "missing model: ${MODEL_PATH}"
    [[ -x "${POLICY_PYTHON}" ]] || die "missing policy Python: ${POLICY_PYTHON}"
    log "Downloading Xiaomi-Robotics-1-RoboCasa"
    mkdir -p "${MODEL_PATH}"
    if command -v hf >/dev/null 2>&1; then
        hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --local-dir "${MODEL_PATH}"
    else
        "${POLICY_PYTHON}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa", local_dir="${MODEL_PATH}")
PY
    fi
}

ensure_runtime() {
    [[ -x "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" ]] && return
    log "Installing RoboDojo runtime in user storage"
    ROBODOJO_SKIP_SIM_SMOKE=1 bash "${ROOT_DIR}/slurm/setup_robodojo_singularity_gpu_scavenger.sbatch"
}

ensure_assets() {
    local path missing=0
    for path in "${ROBO_DIR}/Assets/Robots" \
                "${ROBO_DIR}/Assets/Object/RoboDojo" \
                "${ROBO_DIR}/Assets/Eval_Layout/RoboDojo" \
                "${ROBO_DIR}/Assets/Material"; do
        [[ -e "${path}" ]] || missing=1
    done
    [[ "${missing}" == 0 ]] && return
    log "Initializing RoboDojo assets"
    singularity exec -B "${ROOT_DIR}:/workspace/robo" \
        -B "${RUNTIME_DIR}:/opt/robo-state" "${BASE_IMAGE}" bash -lc \
        'source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && cd /workspace/robo/.deps/RoboDojo && bash scripts/init_assets.sh'
}

ensure_rootfs() {
    if [[ ! -f "${BASE_IMAGE}" ]]; then
        local pull_dir
        pull_dir="$(mktemp -d "${ROOT_DIR}/.runtime/ubuntu-pull.XXXXXX")"
        log "Downloading Ubuntu 24.04 base image"
        singularity pull "${pull_dir}/ubuntu-24.04.sif" docker://ubuntu:24.04
        mv "${pull_dir}/ubuntu-24.04.sif" "${BASE_IMAGE}"
        rmdir "${pull_dir}" 2>/dev/null || true
    fi
    local tag rootfs build_dir
    tag="$(sha256sum "${BASE_IMAGE}" | awk '{print substr($1,1,12)}')"
    rootfs="${ROOTFS_PREFIX}-${tag}"
    if [[ ! -x "${rootfs}/lib64/ld-linux-x86-64.so.2" ]]; then
        build_dir="$(mktemp -d "${ROOT_DIR}/.runtime/rootfs-build.XXXXXX")"
        log "Extracting Ubuntu rootfs ${tag}"
        singularity sif dump 4 "${BASE_IMAGE}" > "${build_dir}/ubuntu.squashfs"
        unsquashfs -q -d "${build_dir}/rootfs" "${build_dir}/ubuntu.squashfs"
        mkdir -p "${build_dir}/rootfs/workspace" "${build_dir}/rootfs/opt/host-vulkan"
        printf 'source_sha256=%s\n' "$(sha256sum "${BASE_IMAGE}" | awk '{print $1}')" \
            > "${build_dir}/rootfs/.robodojo-rootfs"
        mv "${build_dir}/rootfs" "${rootfs}"
        rm -rf "${build_dir}"
    fi
    mkdir -p "${rootfs}/workspace" "${rootfs}/opt/host-vulkan"
    ROOTFS="${rootfs}"
}

install_policy_entrypoint() {
    local policy_root
    for policy_root in "${XPOLICY_DIR}" "${ROBO_DIR}/XPolicyLab"; do
        mkdir -p "${policy_root}/policy/${POLICY_NAME}"
        cp "${ROOT_DIR}/scripts/robodojo_delay_replan_deploy.py" \
            "${policy_root}/policy/${POLICY_NAME}/deploy.py"
        printf '__all__ = []\n' > "${policy_root}/policy/${POLICY_NAME}/__init__.py"
    done
}

clone_if_missing "${ROBO_DIR}" "${ROBODOJO_REPO_URL:-https://github.com/RoboDojo-Benchmark/RoboDojo.git}"
clone_if_missing "${XPOLICY_DIR}" "${XPOLICYLAB_REPO_URL:-https://github.com/XPolicyLab/XPolicyLab.git}"
ensure_model
ensure_runtime
ensure_rootfs
ensure_assets
install_policy_entrypoint

[[ -f "${ROBO_DIR}/scripts/eval_policy.sh" ]] || die "RoboDojo eval launcher missing"
[[ -f "${ROBO_DIR}/task/RoboDojo/config/${TASK_NAME}.yml" ]] || die "task config missing"
[[ -x "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" ]] || die "simulator Python missing"
[[ -f "${MODEL_PATH}/config.json" ]] || die "incomplete model: ${MODEL_PATH}"

PREFIX="/opt/robo-state/miniconda3/envs/RoboDojo"
HOST_ICD="/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"
HOST_VULKAN="/usr/lib64/libvulkan.so.1"
[[ -f "${HOST_ICD}" && -e "${HOST_VULKAN}" ]] || die "host Vulkan files unavailable"
mkdir -p "${OUTPUT_ROOT}/preflight"
log "job=${SLURM_JOB_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader \
    | tee "${OUTPUT_ROOT}/preflight/gpu.txt"

# Ubuntu supplies glibc >= 2.35; host driver libraries stay private. GPU
# allocation and visibility remain controlled by Slurm.
HOST_DRIVER_BINDS=()
declare -A seen_driver_basenames=()
for driver_file in \
    /usr/lib64/libGLX_nvidia.so* /usr/lib64/libEGL_nvidia.so* \
    /usr/lib64/libnvidia-*.so* /usr/lib64/libGLdispatch.so* \
    /usr/lib64/libX11.so* /usr/lib64/libXext.so* /usr/lib64/libxcb.so* \
    /usr/lib64/libXau.so* /usr/lib64/libdrm.so* /usr/lib64/libwayland*.so*; do
    [[ -e "${driver_file}" ]] || continue
    driver_base="$(basename "${driver_file}")"
    [[ -n "${seen_driver_basenames[${driver_base}]:-}" ]] && continue
    seen_driver_basenames["${driver_base}"]=1
    driver_real="$(readlink -f "${driver_file}")"
    [[ -f "${driver_real}" ]] || continue
    HOST_DRIVER_BINDS+=(--ro-bind "${driver_real}" "/opt/host-driver/${driver_base}")
done
[[ "${#HOST_DRIVER_BINDS[@]}" -gt 0 ]] || die "no NVIDIA user-space driver libraries found"
BWRAP_ARGS=(
    --ro-bind "${ROOTFS}" /
    --tmpfs /workspace
    --bind "${ROOT_DIR}" /workspace/robo
    --tmpfs /opt
    --bind "${RUNTIME_DIR}" /opt/robo-state
    --dir /opt/host-driver
    "${HOST_DRIVER_BINDS[@]}"
    --dir /opt/host-vulkan
    --ro-bind "${HOST_VULKAN}" /opt/host-vulkan/libvulkan.so.1
    --proc /proc --dev-bind /dev /dev --ro-bind /sys /sys
    --ro-bind /proc/driver /proc/driver
    --tmpfs /tmp --tmpfs /var/tmp
    --setenv HOME /workspace/robo
    --setenv PATH "${PREFIX}:${PREFIX}/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    --setenv LD_LIBRARY_PATH "${PREFIX}/lib:/opt/host-vulkan:/opt/host-driver"
    --setenv VK_ICD_FILENAMES /workspace/robo/container/nvidia_icd_bwrap.json
    --setenv VULKAN_ICD_FILENAMES /workspace/robo/container/nvidia_icd_bwrap.json
    --setenv VK_DRIVER_FILES /workspace/robo/container/nvidia_icd_bwrap.json
    --setenv TMPDIR /tmp --setenv PYTHONUNBUFFERED 1
    --chdir /workspace/robo
)
run_sim() { bwrap "${BWRAP_ARGS[@]}" "$@"; }

container_path() {
    local path="$1"
    [[ "${path}" == "${ROOT_DIR}"/* ]] || die "path must be under repo root: ${path}"
    printf '/workspace/robo/%s\n' "${path#${ROOT_DIR}/}"
}

log "Checking glibc, bridge, and task metadata in the user-local rootfs"
run_sim "${PREFIX}/bin/python" - <<'PY' | tee "${OUTPUT_ROOT}/preflight/runtime.txt"
import platform
import sys
from scripts.robodojo_compat import task_metadata
print("runtime_python=", sys.executable)
print("runtime_glibc=", platform.libc_ver())
print("task=", task_metadata()["task_config"])
PY

log "Starting headless Isaac SimulationApp preflight"
set +e
run_sim "${PREFIX}/bin/python" - <<'PY' \
    > "${OUTPUT_ROOT}/preflight/isaacsim.out" \
    2> "${OUTPUT_ROOT}/preflight/isaacsim.err"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": False})
print("simulation_app=ready", flush=True)
app.close()
print("simulation_app=closed", flush=True)
PY
isaac_rc=$?
set -e
cat "${OUTPUT_ROOT}/preflight/isaacsim.out"
if [[ "${isaac_rc}" != 0 ]] || ! grep -q 'simulation_app=ready' "${OUTPUT_ROOT}/preflight/isaacsim.out"; then
    cat "${OUTPUT_ROOT}/preflight/isaacsim.err" >&2 || true
    die "Isaac SimulationApp preflight failed; evaluation was not started"
fi

export PYTHONPATH="${ROOT_DIR}:${ROBO_DIR}:${XPOLICY_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
POLICY_LOG="${OUTPUT_ROOT}/policy_server.log"
log "Starting Xiaomi policy server on 127.0.0.1:${PORT}"
"${POLICY_PYTHON}" "${ROOT_DIR}/scripts/robodojo_xiaomi_server.py" \
    --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${PORT}" \
    > "${POLICY_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "${SERVER_PID:-}" 2>/dev/null || true
    wait "${SERVER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        cat "${POLICY_LOG}" >&2 || true
        die "Xiaomi policy server exited early"
    fi
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then break; fi
    sleep 2
done
timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null \
    || { cat "${POLICY_LOG}" >&2 || true; die "Xiaomi policy server did not listen"; }

export ROBODOJO_PRESERVE_CUDA_VISIBLE_DEVICES=1
export ROBODOJO_SIM_DEVICE=cpu
export EVAL_NUM="${EPISODES}"
IFS=',' read -r -a delay_values <<< "${DELAYS_MS}"
IFS=',' read -r -a replan_values <<< "${REPLAN_STEPS_LIST}"

for delay_ms in "${delay_values[@]}"; do
    for replan_steps in "${replan_values[@]}"; do
        cell="delay${delay_ms}ms/replan${replan_steps}"
        cell_dir="${OUTPUT_ROOT}/${cell}"
        metrics="${cell_dir}/timing.jsonl"
        mkdir -p "${cell_dir}/tmp"
        log "Evaluating ${cell} for ${EPISODES} episode(s)"
        export ROBODOJO_DELAY_MS="${delay_ms}"
        export ROBODOJO_REPLAN_STEPS="${replan_steps}"
        export ROBODOJO_DELAY_STEP_FREQUENCY=25
        export ROBODOJO_SWEEP_METRICS="$(container_path "${metrics}")"
        export ROBODOJO_RUN_ID="rdj-xiaomi-${SLURM_JOB_ID}-${delay_ms}-${replan_steps}"
        run_sim bash "$(container_path "${ROBO_DIR}/scripts/eval_policy.sh")" \
            --root_dir "$(container_path "${ROBO_DIR}")" \
            --task_name "${TASK_NAME}" --env_cfg_type arx_x5 --device_id 0 \
            --policy_name "${POLICY_NAME}" --port "${PORT}" --eval_batch false \
            --additional_info "xiaomi_compat_${cell}" --seed "${SEED:-0}" \
            --host 127.0.0.1 --protocol ws \
            --policy_server_url "ws://127.0.0.1:${PORT}" \
            2>&1 | tee "${cell_dir}/eval.log"
        [[ -s "${metrics}" ]] || die "evaluation produced no metrics: ${metrics}"
        printf 'complete\n' > "${cell_dir}/COMPLETE"
    done
done

python3 - "${OUTPUT_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("delay*ms/replan*/timing.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
print(f"summary_rows={len(rows)}")
PY
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
log "Completed: ${OUTPUT_ROOT}"

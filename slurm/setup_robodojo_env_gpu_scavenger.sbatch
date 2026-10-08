#!/usr/bin/env bash
# Prepare the RoboDojo Isaac Sim environment for the Xiaomi smoke sweep.

#SBATCH --job-name=rdj-env-setup
#SBATCH --output=logs/rdj-env-setup-%j.out
#SBATCH --error=logs/rdj-env-setup-%j.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ROBO_DIR="${ROBODOJO_DIR:-${ROOT_DIR}/.deps/RoboDojo}"
ENV_NAME="${ROBODOJO_CONDA_ENV:-RoboDojo}"
MARKER="${ROBODOJO_ENV_MARKER:-${ROOT_DIR}/.runtime/${ENV_NAME}.ready}"

cd "${ROOT_DIR}"
mkdir -p logs "$(dirname "${MARKER}")"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export TERM="${TERM:-xterm-256color}"

if [[ ! -f "${ROBO_DIR}/scripts/install.sh" ]]; then
    echo "Missing RoboDojo checkout: ${ROBO_DIR}" >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required on the setup node" >&2
    exit 2
fi

# Isaac Sim 5.1 publishes the Linux wheel with a manylinux_2_35 tag. The
# scavenger host currently has glibc 2.34, so pip silently falls back to the
# PyPI source placeholder and fails later with "Didn't find wheel". Fail
# before spending time on base dependency installation; use the Singularity
# runner/container path on hosts below glibc 2.35.
if command -v getconf >/dev/null 2>&1; then
    glibc_version="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
    if [[ -n "${glibc_version}" ]]; then
        glibc_major="${glibc_version%%.*}"
        glibc_minor="${glibc_version#*.}"
        glibc_minor="${glibc_minor%%.*}"
        if (( glibc_major < 2 || (glibc_major == 2 && glibc_minor < 35) )); then
            cat >&2 <<EOF
Isaac Sim 5.1 requires glibc >= 2.35 (manylinux_2_35_x86_64), but this host has glibc ${glibc_version}.
The native setup is unsupported here; pip would select the 1.1 KB PyPI source stub and fail with
"Didn't find wheel for isaacsim 5.1.0.0". Run the simulator in an Ubuntu 24.04+/NGC container via
Singularity/Apptainer, or schedule on a node with glibc >= 2.35. Do not substitute Isaac Sim 4.5
without validating the RoboDojo/IsaacLab version contract.
EOF
            exit 3
        fi
    fi
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"

# The official installer is resumable and installs the pinned Isaac Sim,
# IsaacLab, CuRobo, and runtime dependency set for the current RoboDojo tree.
bash "${ROBO_DIR}/scripts/install.sh" --from base_deps

python - <<'PY'
import importlib.util

required = ("isaacsim", "isaaclab", "omni", "pxr", "yaml", "torch")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"RoboDojo environment missing imports: {missing}")
print("RoboDojo imports: OK")
PY

tmp="${MARKER}.tmp.${SLURM_JOB_ID:-local}"
printf 'ready\n' > "${tmp}"
mv -f "${tmp}" "${MARKER}"
echo "Environment ready: ${ENV_NAME}"

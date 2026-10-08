#!/usr/bin/env bash
# Prepare a persistent RoboDojo runtime inside an Ubuntu 24.04 Singularity
# container. The SIF stays immutable; the conda environment and caches live in
# .runtime so the same runtime is usable by every sweep array element.

#SBATCH --job-name=rdj-container-setup
#SBATCH --output=logs/rdj-container-setup-%j.out
#SBATCH --error=logs/rdj-container-setup-%j.err
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/fact_home/xunyuanliu/dev/robo}"
ROBO_DIR="${ROBODOJO_DIR:-${ROOT_DIR}/.deps/RoboDojo}"
RUNTIME_DIR="${ROBODOJO_RUNTIME_DIR:-${ROOT_DIR}/.runtime/robodojo-singularity}"
BASE_IMAGE="${ROBODOJO_BASE_IMAGE:-${ROOT_DIR}/.runtime/ubuntu-24.04.sif}"
MARKER="${ROBODOJO_CONTAINER_MARKER:-${ROOT_DIR}/.runtime/RoboDojo-singularity.ready}"
MINICONDA_INSTALLER="${ROOT_DIR}/.runtime/Miniconda3-latest-Linux-x86_64.sh"
ISAACSIM_WHEEL_URL="${ISAACSIM_WHEEL_URL:-https://pypi.nvidia.com/isaacsim/isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl#sha256=ad2c027831ed5d4a62552735bb799dea4e4604530d2ab9b526ddb6cd19a98c11}"

cd "${ROOT_DIR}"
mkdir -p logs "${RUNTIME_DIR}" "$(dirname "${BASE_IMAGE}")" "$(dirname "${MARKER}")"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${ROOT_DIR}/.runtime/apptainer-cache}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES

command -v singularity >/dev/null || { echo "singularity is required" >&2; exit 2; }

if [[ ! -f "${BASE_IMAGE}" ]]; then
    echo "Pulling ${BASE_IMAGE}"
    singularity pull "${BASE_IMAGE}" docker://ubuntu:24.04
fi

bind_args=(-B "${ROOT_DIR}:/workspace/robo" -B "${RUNTIME_DIR}:/opt/robo-state")
container_bash() {
    singularity exec --nv "${bind_args[@]}" "${BASE_IMAGE}" bash -lc \
        "export HOME='${HOME}'; export PATH=/opt/robo-state/miniconda3/bin:\$PATH; export CONDA_EXE=/opt/robo-state/miniconda3/bin/conda; export PYTHONNOUSERSITE=1; export PIP_USER=0; $1"
}

if [[ ! -x "${RUNTIME_DIR}/miniconda3/bin/python" ]]; then
    echo "Installing Miniconda into ${RUNTIME_DIR}"
    if [[ ! -s "${MINICONDA_INSTALLER}" ]]; then
        curl -fL --retry 3 https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${MINICONDA_INSTALLER}"
    fi
    container_bash 'bash /workspace/robo/.runtime/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/robo-state/miniconda3'
fi

if [[ ! -x "${RUNTIME_DIR}/miniconda3/envs/RoboDojo/bin/python" ]]; then
    echo "Creating RoboDojo conda environment"
    container_bash 'source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; conda create -n RoboDojo python=3.11 -y && conda install -n base -c conda-forge git git-lfs -y'
fi

echo "Installing RoboDojo Python dependencies in the container"
container_bash 'source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda install -n RoboDojo -c conda-forge gcc_linux-64 gxx_linux-64 libxml2 libglu libglvnd xorg-libx11 xorg-libxext xorg-libxt xorg-libxfixes xorg-libxi xorg-libxrender xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libsm xorg-libice -y'
    container_bash "source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && cd /workspace/robo/${ROBO_DIR#${ROOT_DIR}/} && python -m pip install --upgrade pip && python -m pip install -r scripts/requirements.txt && python -m pip install opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0 && python -m pip install --no-deps 'https://files.pythonhosted.org/packages/1c/2c/8a0b02d60a1dbbae7faa5af30484b016aa3023f9833dfc0d19b0b770dd6a/botocore-1.39.11-py3-none-any.whl#sha256=1545352931a8a186f3e977b1e1a4542d7d434796e274c3c62efd0210b5ea76dc' && python -m pip install --no-deps 'https://files.pythonhosted.org/packages/72/66/88566a6484e746c0b075f7c9bb248e8548eda0a486de4460d150a41e2d57/boto3-1.39.11-py3-none-any.whl#sha256=af8f1dad35eceff7658fab43b39b0f55892b6e3dd12308733521cc24dd2c9a02' && ROBODOJO_SKIP_SYSTEM_DEPS=1 ISAACSIM_WHEEL_URL='${ISAACSIM_WHEEL_URL}' bash scripts/install.sh --from isaacsim"

if [[ "${ROBODOJO_SKIP_SIM_SMOKE:-0}" != "1" ]]; then
    container_bash "source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && python -c 'from isaacsim import SimulationApp; app=SimulationApp({\"headless\": True, \"enable_cameras\": False}); from pxr import Usd; import isaaclab, curobo, torch; print(\"RoboDojo SimulationApp smoke: OK\", Usd.GetVersion(), torch.__version__); app.close()'"
else
    echo "Skipping Singularity SimulationApp smoke; the one-click runner validates Isaac through its user-local glibc rootfs."
fi

if [[ ! -d "${ROBO_DIR}/Assets/Robots" || ! -d "${ROBO_DIR}/Assets/Object/RoboDojo" ]]; then
    echo "Downloading RoboDojo assets"
    container_bash "source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && cd /workspace/robo/${ROBO_DIR#${ROOT_DIR}/} && bash scripts/init_assets.sh"
fi

container_bash "source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && cd /workspace/robo/${ROBO_DIR#${ROOT_DIR}/} && python utils/update_embodiment_config_path.py"

container_bash "source /opt/robo-state/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo && cd /workspace/robo/${ROBO_DIR#${ROOT_DIR}/} && bash scripts/internal/verify_install.sh --skip-isaac"

tmp="${MARKER}.tmp.${SLURM_JOB_ID:-local}"
printf 'base_image=%s\nruntime_dir=%s\nready\n' "${BASE_IMAGE}" "${RUNTIME_DIR}" > "${tmp}"
mv -f "${tmp}" "${MARKER}"
echo "Container runtime ready: ${MARKER}"

#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash mini_vllm/profile/install_and_profile.sh [script options] -- [mini_vllm.profile.repro args]

Script options:
  --venv-dir PATH   Virtualenv location. Default: <repo>/.venvs/mini-vllm-profile
  --python BIN      Python executable used to create the venv. Default: python3
  --skip-install    Reuse the existing virtualenv and skip pip installation.
  --help            Show this message.

Everything after `--` is forwarded to `python -m mini_vllm.profile.repro`.

Example:
  bash mini_vllm/profile/install_and_profile.sh -- \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir runs/rtx5090 \
    --compilation_mode stock_torch_compile \
    --cudagraph_mode none
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VENV_DIR="${REPO_ROOT}/.venvs/mini-vllm-profile"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      [[ $# -ge 2 ]] || { echo "--venv-dir requires a value" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown script option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPRO_ARGS=("$@")

cd "${REPO_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found on PATH. The current profiler requires NVIDIA GPUs for power sampling." >&2
  exit 1
fi

git submodule update --init --recursive 3rdparty/vllm

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "${REPO_ROOT}/3rdparty/vllm"
  python -m pip install numpy matplotlib tqdm pynvml pyyaml
else
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Virtualenv not found at ${VENV_DIR}. Drop --skip-install or set --venv-dir." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

VLLM_COMMIT=$(git -C "${REPO_ROOT}/3rdparty/vllm" rev-parse HEAD)
VLLM_DESCRIBE=$(git -C "${REPO_ROOT}/3rdparty/vllm" describe --tags --always --dirty)

echo "Using repo root: ${REPO_ROOT}"
echo "Using virtualenv: ${VENV_DIR}"
echo "Using pinned vLLM submodule: ${VLLM_COMMIT} (${VLLM_DESCRIBE})"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m mini_vllm.profile.repro "${REPRO_ARGS[@]}"

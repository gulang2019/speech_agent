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

load_vllm_source_spec() {
  local -a spec_lines=()
  mapfile -t spec_lines < <(
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<'PY'
from mini_vllm.profile.vllm_source import VLLM_SOURCE

print(VLLM_SOURCE.repo_url)
print(VLLM_SOURCE.commit)
print(VLLM_SOURCE.managed_checkout_dirname)
print(VLLM_SOURCE.patch_path)
print("1" if VLLM_SOURCE.patch_is_nonempty else "0")
print(VLLM_SOURCE.patch_sha256 or "")
PY
  )

  if [[ "${#spec_lines[@]}" -lt 6 ]]; then
    echo "Failed to load the pinned vLLM source spec." >&2
    exit 1
  fi

  VLLM_REPO_URL="${spec_lines[0]}"
  VLLM_COMMIT="${spec_lines[1]}"
  VLLM_CHECKOUT_DIRNAME="${spec_lines[2]}"
  VLLM_PATCH_FILE="${spec_lines[3]}"
  VLLM_PATCH_IS_NONEMPTY="${spec_lines[4]}"
  VLLM_PATCH_SHA256="${spec_lines[5]}"
  VLLM_SRC_DIR="${DEPS_DIR}/${VLLM_CHECKOUT_DIRNAME}"
}

ensure_vllm_checkout() {
  mkdir -p "${DEPS_DIR}"

  if [[ -d "${VLLM_SRC_DIR}/.git" ]]; then
    return
  fi

  if [[ -e "${VLLM_SRC_DIR}" ]]; then
    echo "Managed vLLM checkout path exists but is not a git repo: ${VLLM_SRC_DIR}" >&2
    exit 1
  fi

  git clone --filter=blob:none "${VLLM_REPO_URL}" "${VLLM_SRC_DIR}"
  git -C "${VLLM_SRC_DIR}" checkout --detach "${VLLM_COMMIT}"

  if [[ "${VLLM_PATCH_IS_NONEMPTY}" == "1" ]]; then
    git -C "${VLLM_SRC_DIR}" apply --check "${VLLM_PATCH_FILE}"
    git -C "${VLLM_SRC_DIR}" apply "${VLLM_PATCH_FILE}"
  fi
}

verify_vllm_install() {
  python - "${VLLM_SRC_DIR}" <<'PY'
import json
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

expected_source = Path(sys.argv[1]).resolve()
expected_url = expected_source.as_uri()

try:
    distribution = importlib_metadata.distribution("vllm")
except importlib_metadata.PackageNotFoundError as exc:
    raise SystemExit("vllm is not installed in the selected virtualenv.") from exc

installed_url = None
direct_url_text = distribution.read_text("direct_url.json")
if direct_url_text:
    try:
        installed_url = json.loads(direct_url_text).get("url")
    except Exception:
        installed_url = None

if installed_url != expected_url:
    raise SystemExit(
        "Installed vllm source does not match the pinned profiling checkout.\n"
        f"expected: {expected_url}\n"
        f"installed: {installed_url or 'unknown'}\n"
        "Drop --skip-install to reinstall the pinned source."
    )
PY
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
DEPS_DIR="${REPO_ROOT}/.deps"
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
load_vllm_source_spec
ensure_vllm_checkout

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found on PATH. The current profiler requires NVIDIA GPUs for power sampling." >&2
  exit 1
fi

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "${VLLM_SRC_DIR}"
  python -m pip install numpy matplotlib tqdm pynvml pyyaml
else
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Virtualenv not found at ${VENV_DIR}. Drop --skip-install or set --venv-dir." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

verify_vllm_install

VLLM_DESCRIBE=$(git -C "${VLLM_SRC_DIR}" describe --tags --always --dirty 2>/dev/null || true)

echo "Using repo root: ${REPO_ROOT}"
echo "Using virtualenv: ${VENV_DIR}"
echo "Using managed vLLM checkout: ${VLLM_SRC_DIR}"
echo "Using pinned vLLM source: ${VLLM_COMMIT}${VLLM_DESCRIBE:+ (${VLLM_DESCRIBE})}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MINI_VLLM_PROFILE_VLLM_SRC="${VLLM_SRC_DIR}"
export MINI_VLLM_PROFILE_VLLM_REPO_URL="${VLLM_REPO_URL}"
export MINI_VLLM_PROFILE_VLLM_COMMIT="${VLLM_COMMIT}"
export MINI_VLLM_PROFILE_VLLM_PATCH_FILE="${VLLM_PATCH_FILE}"
export MINI_VLLM_PROFILE_VLLM_PATCH_APPLIED="${VLLM_PATCH_IS_NONEMPTY}"
export MINI_VLLM_PROFILE_VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256}"

python -m mini_vllm.profile.repro "${REPRO_ARGS[@]}"

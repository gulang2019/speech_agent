import importlib.util
import os
import sys
from pathlib import Path


_DISABLE_SKLEARN_ENV = "MINI_VLLM_DISABLE_SKLEARN_IMPORT"
_PATCHED_ATTR = "_mini_vllm_patched"


def _prepend_path_var(path_var: str, path: str) -> str:
    entries = [entry for entry in path_var.split(os.pathsep) if entry]
    if path not in entries:
        entries.insert(0, path)
    return os.pathsep.join(entries)


def _patch_optional_import_detection() -> None:
    find_spec = importlib.util.find_spec
    if getattr(find_spec, _PATCHED_ATTR, False):
        return

    def patched_find_spec(name: str, *args, **kwargs):
        if os.environ.get(_DISABLE_SKLEARN_ENV) == "1" and name == "sklearn":
            return None
        return find_spec(name, *args, **kwargs)

    setattr(patched_find_spec, _PATCHED_ATTR, True)
    importlib.util.find_spec = patched_find_spec


def bootstrap_vllm_import_env() -> None:
    repo_root = str(Path(__file__).resolve().parent.parent)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault(_DISABLE_SKLEARN_ENV, "1")
    os.environ["PYTHONPATH"] = _prepend_path_var(
        os.environ.get("PYTHONPATH", ""),
        repo_root,
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    _patch_optional_import_detection()

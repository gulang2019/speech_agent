from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Optional

@dataclass(frozen=True)
class ProfilePreset:
    name: str
    description: str
    batch_config_path: Path
    output_filename: str
    plot_prefix: str


_THIS_DIR = Path(__file__).resolve().parent

PRESET_ORDER = (
    "prefill_single_decode_sweep",
    "prefill_decode_only",
)

PRESETS = {
    "prefill_single_decode_sweep": ProfilePreset(
        name="prefill_single_decode_sweep",
        description=(
            "Single-request prefill token sweep plus decode concurrency sweep. "
            "Generates prefill_single_decode_sweep_energy_per_token_vs_batch_tokens.png."
        ),
        batch_config_path=_THIS_DIR / "presets" / "prefill_single_decode_sweep.json",
        output_filename="prefill_single_decode_sweep.jsonl",
        plot_prefix="prefill_single_decode_sweep",
    ),
    "prefill_decode_only": ProfilePreset(
        name="prefill_decode_only",
        description=(
            "Fixed-size prefill concurrency sweep plus decode concurrency sweep. "
            "Generates profile_plot_power_vs_concurrency.png."
        ),
        batch_config_path=_THIS_DIR / "presets" / "prefill_decode_only.json",
        output_filename="profile.jsonl",
        plot_prefix="profile_plot",
    ),
}

_MANAGED_FORWARD_FLAGS = (
    "--batch_config",
    "--input",
    "--output",
    "--plot_only",
    "--plot_prefix",
)

_REPO_ROOT = _THIS_DIR.parent.parent


def _parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run packaged mini_vllm profiling presets and capture the config and "
            "hardware metadata needed to reproduce the plots on another GPU."
        )
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=PRESET_ORDER,
        default=None,
        help="Run only the selected preset(s). Defaults to all packaged presets.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("profile_repro"),
        help="Directory where per-preset artifacts are written.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help=(
            "Reuse the preset JSONL already present under --output_dir and regenerate "
            "plots and metadata without rerunning batches."
        ),
    )
    parser.add_argument(
        "--list_presets",
        action="store_true",
        help="List the packaged presets and exit.",
    )
    return parser.parse_known_args(argv)


def _selected_presets(selected: Optional[list[str]]) -> list[ProfilePreset]:
    names = selected or list(PRESET_ORDER)
    return [PRESETS[name] for name in names]


def _forbid_managed_flags(forwarded_args: list[str]) -> None:
    for arg in forwarded_args:
        for flag in _MANAGED_FORWARD_FLAGS:
            if arg == flag or arg.startswith(flag + "="):
                raise ValueError(
                    f"{flag} is managed by mini_vllm.profile.repro; remove it from the command."
                )


def _extract_forwarded_value(forwarded_args: list[str], flag: str) -> Optional[str]:
    for index, arg in enumerate(forwarded_args):
        if arg == flag:
            if index + 1 >= len(forwarded_args):
                return None
            return forwarded_args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _safe_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _copy_batch_config(preset: ProfilePreset, destination: Path) -> None:
    destination.write_text(preset.batch_config_path.read_text(encoding="utf-8"), encoding="utf-8")


def _run_command(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = completed.stdout.strip()
    return output or None


def _collect_submodule_git_metadata(submodule_path: Path) -> dict[str, object]:
    if not submodule_path.exists():
        return {
            "path": str(submodule_path),
            "present": False,
        }

    commit = _run_command(["git", "-C", str(submodule_path), "rev-parse", "HEAD"])
    describe = _run_command(
        ["git", "-C", str(submodule_path), "describe", "--tags", "--always", "--dirty"]
    )
    dirty = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(submodule_path), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(completed.stdout.strip())
    except Exception:
        dirty = None

    return {
        "path": str(submodule_path.resolve()),
        "present": True,
        "commit": commit,
        "describe": describe,
        "dirty": dirty,
    }


def _collect_installed_distribution_metadata(distribution_name: str) -> dict[str, object]:
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return {
            "name": distribution_name,
            "installed": False,
        }
    except Exception as exc:
        return {
            "name": distribution_name,
            "installed": False,
            "query_error": str(exc),
        }

    editable_root = None
    try:
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            editable_root = direct_url.get("url")
    except Exception:
        editable_root = None

    return {
        "name": distribution_name,
        "installed": True,
        "version": distribution.version,
        "location": str(distribution.locate_file("")),
        "editable_url": editable_root,
    }


def _collect_git_metadata() -> dict[str, object]:
    commit = _run_command(["git", "rev-parse", "HEAD"])
    dirty = None
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(completed.stdout.strip())
    except Exception:
        dirty = None
    return {
        "commit": commit,
        "dirty_tracked_files": dirty,
    }


def _collect_software_metadata() -> dict[str, object]:
    vllm_submodule_path = _REPO_ROOT / "3rdparty" / "vllm"
    return {
        "repo_root": str(_REPO_ROOT.resolve()),
        "vllm_submodule": _collect_submodule_git_metadata(vllm_submodule_path),
        "installed_python_packages": {
            "vllm": _collect_installed_distribution_metadata("vllm"),
            "numpy": _collect_installed_distribution_metadata("numpy"),
            "matplotlib": _collect_installed_distribution_metadata("matplotlib"),
            "torch": _collect_installed_distribution_metadata("torch"),
        },
    }


def _collect_gpu_metadata(device_index: Optional[int]) -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "backend": "unknown",
            "nvidia_smi_available": False,
            "target_device_index": device_index,
        }

    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
        "clocks.max.graphics",
        "pci.bus_id",
    ]
    command = [
        nvidia_smi,
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.extend(["-i", str(device_index)])

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {
            "backend": "nvidia",
            "nvidia_smi_available": True,
            "target_device_index": device_index,
            "query_error": str(exc),
        }

    devices = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        values = next(csv.reader([line], skipinitialspace=True))
        devices.append({key: value for key, value in zip(fields, values)})

    return {
        "backend": "nvidia",
        "nvidia_smi_available": True,
        "target_device_index": device_index,
        "devices": devices,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _artifact_summary(run_dir: Path, preset: ProfilePreset) -> dict[str, str]:
    plot_prefix = run_dir / preset.plot_prefix
    return {
        "run_dir": str(run_dir.resolve()),
        "batch_config": str((run_dir / "batch_config.json").resolve()),
        "profile_jsonl": str((run_dir / preset.output_filename).resolve()),
        "metadata_json": str((run_dir / "run_metadata.json").resolve()),
        "power_vs_concurrency_png": str(
            (run_dir / f"{preset.plot_prefix}_power_vs_concurrency.png").resolve()
        ),
        "energy_per_token_vs_batch_tokens_png": str(
            (run_dir / f"{preset.plot_prefix}_energy_per_token_vs_batch_tokens.png").resolve()
        ),
        "plot_prefix": str(plot_prefix.resolve()),
    }


def _run_preset(
    preset: ProfilePreset,
    output_dir: Path,
    plot_only: bool,
    forwarded_args: list[str],
) -> dict[str, object]:
    run_dir = output_dir / preset.name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = run_dir / "batch_config.json"
    output_path = run_dir / preset.output_filename
    plot_prefix = run_dir / preset.plot_prefix
    metadata_path = run_dir / "run_metadata.json"

    _copy_batch_config(preset, config_copy_path)

    profile_args = list(forwarded_args)
    if plot_only:
        if not output_path.exists():
            raise FileNotFoundError(
                f"{output_path} does not exist. Run the preset once before using --plot_only."
            )
        profile_args.extend(
            [
                "--plot_only",
                "--input",
                str(output_path),
                "--plot_prefix",
                str(plot_prefix),
            ]
        )
    else:
        profile_args.extend(
            [
                "--batch_config",
                str(config_copy_path),
                "--output",
                str(output_path),
                "--plot_prefix",
                str(plot_prefix),
            ]
        )

    device_index_str = _extract_forwarded_value(forwarded_args, "--device_index")
    model_name = _extract_forwarded_value(forwarded_args, "--model_name")
    started_at = datetime.now(timezone.utc)

    run_state = "success"
    error_message = None
    try:
        from mini_vllm.profile import cli as profile_cli

        exit_code = profile_cli.main(profile_args)
        if exit_code:
            run_state = "failed"
            error_message = f"mini_vllm.profile.cli exited with code {exit_code}"
            raise RuntimeError(error_message)
    except Exception as exc:
        run_state = "failed"
        error_message = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "preset": {
                "name": preset.name,
                "description": preset.description,
                "batch_config_source": str(preset.batch_config_path.resolve()),
                "batch_config_snapshot": str(config_copy_path.resolve()),
            },
            "run": {
                "mode": "plot_only" if plot_only else "collect_and_plot",
                "state": run_state,
                "error_message": error_message,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
            },
            "artifacts": _artifact_summary(run_dir, preset),
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "cwd": str(Path.cwd().resolve()),
            },
            "git": _collect_git_metadata(),
            "software": _collect_software_metadata(),
            "gpu": _collect_gpu_metadata(_safe_int(device_index_str, 0)),
            "command": {
                "wrapper_argv": sys.argv[1:],
                "forwarded_cli_args": forwarded_args,
                "resolved_profile_cli_args": profile_args,
                "resolved_profile_cli_shell": shlex.join(profile_args),
                "model_name": model_name,
            },
        }
        _write_json(metadata_path, metadata)

    summary = _artifact_summary(run_dir, preset)
    print(f"[{preset.name}] profile: {summary['profile_jsonl']}")
    print(f"[{preset.name}] power plot: {summary['power_vs_concurrency_png']}")
    print(
        f"[{preset.name}] energy/token plot: "
        f"{summary['energy_per_token_vs_batch_tokens_png']}"
    )
    return {
        "preset": preset.name,
        "description": preset.description,
        **summary,
    }


def _print_presets() -> None:
    for preset in _selected_presets(None):
        print(f"{preset.name}: {preset.description}")
        print(f"  config: {preset.batch_config_path}")
        print(f"  output: {preset.output_filename}")
        print(f"  plot prefix: {preset.plot_prefix}")


def main(argv: Optional[list[str]] = None) -> int:
    args, forwarded_args = _parse_args(argv)
    if args.list_presets:
        _print_presets()
        return 0

    _forbid_managed_flags(forwarded_args)
    selected_presets = _selected_presets(args.preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for preset in selected_presets:
        print(f"Running preset {preset.name}...")
        manifest_entries.append(
            _run_preset(
                preset=preset,
                output_dir=args.output_dir,
                plot_only=args.plot_only,
                forwarded_args=forwarded_args,
            )
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir.resolve()),
        "presets": manifest_entries,
    }
    _write_json(args.output_dir / "repro_manifest.json", manifest)
    print(f"Manifest: {(args.output_dir / 'repro_manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

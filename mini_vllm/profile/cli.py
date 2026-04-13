from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile mini_vllm model runner.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_config", type=str, default=None)
    parser.add_argument(
        "--batch_name",
        action="append",
        default=None,
        help="Only run matching batch spec name(s). Can be passed multiple times.",
    )
    parser.add_argument("--max_memory_utilization", type=float, default=0.8)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable vLLM torch.compile/CUDA graph execution and run eagerly.",
    )
    parser.add_argument(
        "--compilation_mode",
        type=str,
        choices=("none", "stock_torch_compile", "dynamo_trace_once", "vllm_compile"),
        default=None,
        help="Override vLLM compilation mode.",
    )
    parser.add_argument(
        "--compilation_backend",
        type=str,
        default="",
        help="Override vLLM compilation backend, e.g. inductor or eager.",
    )
    parser.add_argument(
        "--cudagraph_mode",
        type=str,
        choices=("none", "piecewise", "full", "full_decode_only", "full_and_piecewise"),
        default=None,
        help="Override vLLM cudagraph mode.",
    )

    parser.add_argument("--output", type=str, default="profile.jsonl")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup_s", type=float, default=10.0)
    parser.add_argument(
        "--warmup_scope",
        type=str,
        choices=("global", "batch"),
        default="global",
    )
    parser.add_argument("--idle_s", type=float, default=2.0)
    parser.add_argument("--sample_interval_s", type=float, default=0.01)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--no_sync_cuda", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")

    parser.add_argument("--graphics_clock", type=str, default=None)
    parser.add_argument("--power_limit_w", type=int, default=None)

    parser.add_argument("--model_out", type=str, default=None)
    parser.add_argument("--plot_prefix", type=str, default=None)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=("linear_1d", "max_affine"),
        default="linear_1d",
    )
    parser.add_argument("--num_components", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def _run_plot_only(args: argparse.Namespace) -> int:
    from mini_vllm.profile.modeler import ProfileModeler

    input_path = args.input or args.output
    if not input_path:
        raise ValueError("--plot_only requires --input or --output")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if not args.plot_prefix and not args.model_out:
        raise ValueError("--plot_only requires --plot_prefix and/or --model_out")

    modeler = ProfileModeler(
        model_type=args.model_type,
        num_components=args.num_components,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    records = modeler.load(input_path)
    if args.model_out or args.plot_prefix:
        models = modeler.fit_models(records)
        if args.model_out:
            modeler.save_models(args.model_out, models)
        if args.plot_prefix:
            modeler.plot(records, args.plot_prefix)
    return 0


def _estimate_max_query_tokens(batch_config_path: str) -> Optional[int]:
    try:
        if batch_config_path.endswith(".json"):
            with open(batch_config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            import yaml  # type: ignore

            with open(batch_config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(raw, list):
        return None
    max_tokens = 0
    for batch in raw:
        total_query = 0
        requests = batch.get("requests")
        if requests is not None:
            for req in requests:
                try:
                    total_query += int(req.get("query_len", 0))
                except Exception:
                    pass
        else:
            request_template = batch.get("request_template")
            try:
                num_reqs = int(batch.get("num_reqs", 0))
            except Exception:
                num_reqs = 0
            if request_template is not None and num_reqs > 0:
                try:
                    total_query = int(request_template.get("query_len", 0)) * num_reqs
                except Exception:
                    pass
        if total_query > max_tokens:
            max_tokens = total_query
    return max_tokens if max_tokens > 0 else None


def _configure_cuda_device(device_index: int) -> None:
    # Keep GPU selection consistent between torch/vLLM and nvidia-smi sampling.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.plot_only:
        return _run_plot_only(args)
    if not args.model_name:
        raise ValueError("--model_name is required unless --plot_only is set")
    if not args.batch_config:
        raise ValueError("--batch_config is required unless --plot_only is set")

    _configure_cuda_device(args.device_index)

    from mini_vllm.profile.batch_sampler import BatchSampler
    from mini_vllm.profile.energy_meter import EnergyMeter, GpuFrequencyController
    from mini_vllm.profile.modeler import ProfileModeler
    from mini_vllm.profile.profiler import BatchProfiler
    from mini_vllm.struct import Config
    from mini_vllm.vllm_utils import get_vllm_config
    from mini_vllm.model_runner import ModelRunner

    max_from_config = _estimate_max_query_tokens(args.batch_config)
    max_num_batched_tokens = (
        args.max_num_batched_tokens
        if args.max_num_batched_tokens is not None
        else max_from_config
    )

    config = Config(
        model_name=args.model_name,
        max_memory_utilization=args.max_memory_utilization,
        block_size=args.block_size,
        max_num_batched_tokens=max(max_num_batched_tokens,10),
        enforce_eager=args.enforce_eager,
        compilation_mode=args.compilation_mode,
        compilation_backend=args.compilation_backend,
        cudagraph_mode=args.cudagraph_mode,
    )
    vllm_config = get_vllm_config(config)
    model_runner = ModelRunner(vllm_config)

    sampler = BatchSampler(block_size=model_runner.block_size, num_blocks=model_runner.num_blocks)
    specs = sampler.load_specs(args.batch_config)
    if args.batch_name:
        selected_names = set(args.batch_name)
        specs = [spec for spec in specs if spec.name in selected_names]
        if not specs:
            raise ValueError(
                "No batch specs matched --batch_name: "
                + ", ".join(sorted(selected_names))
            )

    energy_meter = EnergyMeter(
        device_index=args.device_index,
        sample_interval_s=args.sample_interval_s,
    )

    profiler = BatchProfiler(
        model_runner=model_runner,
        energy_meter=energy_meter,
        warmup=args.warmup,
        warmup_s=args.warmup_s,
        repeats=args.repeats,
        warmup_scope=args.warmup_scope,
        sync_cuda=not args.no_sync_cuda,
        show_progress=not args.no_progress,
    )

    freq_ctrl = None
    try:
        if args.graphics_clock or args.power_limit_w is not None:
            freq_ctrl = GpuFrequencyController(device_index=args.device_index)
            if args.power_limit_w is not None:
                freq_ctrl.set_power_limit(args.power_limit_w)
            if args.graphics_clock:
                parts = args.graphics_clock.split(",")
                if len(parts) != 2:
                    raise ValueError("--graphics_clock expects MIN,MAX in MHz")
                min_mhz = int(parts[0])
                max_mhz = int(parts[1])
                freq_ctrl.set_graphics_clock(min_mhz, max_mhz)

        results = profiler.profile(
            specs=specs,
            batch_builder=sampler.build_batch,
            output_path=args.output,
            idle_s=args.idle_s,
            resume=args.resume,
            retry_failed=args.retry_failed,
            continue_on_error=args.continue_on_error,
        )
    finally:
        if freq_ctrl is not None:
            try:
                freq_ctrl.reset_graphics_clock()
            except Exception:
                pass

    if args.model_out or args.plot_prefix:
        modeler = ProfileModeler(
            model_type=args.model_type,
            num_components=args.num_components,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        models = modeler.fit_models(results)
        if args.model_out:
            modeler.save_models(args.model_out, models)
        if args.plot_prefix:
            modeler.plot(results, args.plot_prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

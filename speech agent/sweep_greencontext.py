#!/usr/bin/env python3
"""
GreenContext SM Allocation Sweep Script

流程：
1. Baseline: 等分 SM (33,33,34)，找出瓶颈阶段
2. Targeted search: 根据瓶颈阶段分配更多 SM
3. 输出最优分配方案
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger('sweep')


def parse_args():
    parser = argparse.ArgumentParser(description="GreenContext SM allocation sweep")
    parser.add_argument("--lambda-rates", nargs="+", type=float, default=[100], metavar="R")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8], metavar="N")
    parser.add_argument("--experiment-duration", type=float, default=60.0)
    parser.add_argument("--slo-ttff", type=float, default=5.0)
    parser.add_argument("--slo-tbf", type=float, default=0.02)
    parser.add_argument("--wq-config", type=str, default="8,8,8",
                        help="WQ concurrency limits per model (default: 8,8,8)")
    parser.add_argument("--output", type=str, default="sweep_results.json")
    parser.add_argument("--dataset-path", type=str, default="/data2/liuxunyuan/datasets")
    return parser.parse_args()


async def run_single(
    convlist,
    ec_config_str: str,
    wq_config_str: str,
    lambda_rate: float,
    batch_size: int,
    experiment_duration: float,
    slo: dict,
) -> dict:
    """Run one experiment configuration, returns parsed result dict."""
    from request_utils import GreenContextConfig, SMResource, WQConfig

    # Dynamically load dataloader module to access its single_task (3-model pipeline)
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataloader_module", "dataloader.py")
    dm = importlib.util.module_from_spec(spec)
    sys.modules["dataloader_module"] = dm
    spec.loader.exec_module(dm)
    single_task = dm.single_task

    # Parse SM ratios
    sm_ratios = [float(x) / 100.0 for x in ec_config_str.split(',')]
    wq_limits = [int(x) for x in wq_config_str.split(',')]

    ec_configs = []
    for i, (sm_ratio, wq_limit) in enumerate(zip(sm_ratios, wq_limits)):
        ec_configs.append(GreenContextConfig(
            model_index=i,
            sm_resource=SMResource(),
            wq_config=WQConfig(wq_concurrency_limit=wq_limit),
            sm_allocation_ratio=sm_ratio,
        ))

    logger.info(f"Running: ec_config={ec_config_str}, wq_config={wq_config_str}, "
                f"lambda={lambda_rate}, batch_size={batch_size}")

    start = time.perf_counter()

    # single_task returns final_result dict (3-model pipeline: STT→LM→TTS)
    result_dict = await single_task(
        convlist,
        lambda_rate=lambda_rate,
        batch_size=batch_size,
        experiment_duration=experiment_duration,
        detailed_log=False,
        slo=slo,
        plotting=False,
        ec_configs=ec_configs,
    )

    elapsed = time.perf_counter() - start
    logger.info(f"  Completed in {elapsed:.1f}s")

    key = f"batch_size_{batch_size}_lambda_{lambda_rate}"
    if key not in result_dict:
        raise RuntimeError(f"Result key '{key}' not found in returned result. "
                           f"Available keys: {list(result_dict.keys())}")

    return result_dict[key]


def find_bottleneck(result: dict) -> str:
    """Find bottleneck model based on per-model processing time."""
    tp = result.get("throughput", {})
    if not tp:
        return "LM"

    # 计算各模型每帧处理时间
    frame_times = {}
    for model_type, stats in tp.items():
        frames = max(stats.get("total_frames", 1), 1)
        runtime = stats.get("runtime_s", 1)
        frame_times[model_type] = runtime / frames

    bottleneck = max(frame_times, key=frame_times.get)
    logger.info(f"Bottleneck detection: frame_times={ {k: f'{v:.4f}' for k,v in frame_times.items()} } "
                f"→ bottleneck={bottleneck}")
    return bottleneck


def generate_candidates(baseline_ec_config: str, bottleneck: str, n: int = 5) -> list[str]:
    """Generate SM allocation candidates based on bottleneck (max n-1 candidates)."""
    candidates = set()

    if bottleneck == "STT":
        candidates.update(["50,25,25", "60,20,20", "45,30,25"])
    elif bottleneck == "LM":
        candidates.update(["25,50,25", "20,60,20", "30,45,25"])
    elif bottleneck == "TTS":
        candidates.update(["25,25,50", "20,20,60", "25,30,45"])
    else:
        candidates.update(["40,30,30", "30,40,30"])

    candidates.update(["35,35,30", "30,35,35"])

    result = list(candidates)[:n]
    logger.info(f"Generated candidates: {result}")
    return result


async def run_sweep(
    convlist,
    lambda_rate: float,
    batch_size: int,
    experiment_duration: float,
    slo: dict,
    wq_config: str,
    sweep_duration: float = 20.0,
):
    """执行完整 sweep 流程。"""
    results = []

    # Step 1: Baseline
    ec_config = "33,33,34"
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 1: Baseline run (ec_config={ec_config})")
    logger.info(f"{'='*60}")

    result = await run_single(
        convlist, ec_config, wq_config,
        lambda_rate, batch_size, sweep_duration, slo
    )
    results.append((ec_config, result))

    # Step 2: 瓶颈检测
    bottleneck = find_bottleneck(result)
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 2: Bottleneck = {bottleneck}")
    logger.info(f"{'='*60}")

    # Step 3: Targeted search
    candidates = generate_candidates(ec_config, bottleneck)

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 3: Targeted search ({len(candidates)} candidates)")
    logger.info(f"{'='*60}")

    for ec_cfg in candidates:
        if ec_cfg == ec_config:
            continue
        try:
            result = await run_single(
                convlist, ec_cfg, wq_config,
                lambda_rate, batch_size, sweep_duration, slo
            )
            results.append((ec_cfg, result))
        except Exception as e:
            logger.error(f"  Failed for {ec_cfg}: {e}")

    return results, bottleneck


def analyze_and_report(results: list, bottleneck: str, output_path: str):
    """分析并输出报告。"""
    if not results:
        print("No results."); return

    # 按 TTFF 排序
    sorted_by_ttff = sorted(results, key=lambda x: x[1].get("average_TTFF", float('inf')))
    sorted_by_tbf = sorted(results, key=lambda x: x[1].get("average_TBF", float('inf')))
    sorted_by_viol = sorted(results, key=lambda x: x[1].get("TTFF_violation_rate", 1.0))

    best_ttff = sorted_by_ttff[0]
    best_tbf = sorted_by_tbf[0]
    best_viol = sorted_by_viol[0]

    report_lines = [
        "",
        "=" * 70,
        "GREENCONTEXT SM ALLOCATION SWEEP REPORT",
        "=" * 70,
        f"Bottleneck identified: {bottleneck}",
        f"Configurations tested: {len(results)}",
        "",
        "--- Best by Average TTFF ---",
        f"  Config: {best_ttff[0]}  TTFF: {best_ttff[1].get('average_TTFF',0)*1000:.2f}ms  "
        f"Viol: {best_ttff[1].get('TTFF_violation_rate',0):.2%}",
        "",
        "--- Best by Average TBF ---",
        f"  Config: {best_tbf[0]}  TBF: {best_tbf[1].get('average_TBF',0)*1000:.4f}ms  "
        f"Viol: {best_tbf[1].get('TBF_violation_rate',0):.2%}",
        "",
        "--- Best by TTFF Violation Rate ---",
        f"  Config: {best_viol[0]}  Viol: {best_viol[1].get('TTFF_violation_rate',0):.2%}  "
        f"TTFF: {best_viol[1].get('average_TTFF',0)*1000:.2f}ms",
        "",
        "--- All Results (sorted by TTFF) ---",
        f"{'Config':<15} {'TTFF(ms)':<12} {'TBF(ms)':<12} "
        f"{'TTFF viol%':<12} {'TBF viol%':<12} {'TTFF slope':<12}",
        "-" * 75,
    ]

    for ec_cfg, r in sorted_by_ttff:
        report_lines.append(
            f"{ec_cfg:<15} {r.get('average_TTFF',0)*1000:<12.2f} {r.get('average_TBF',0)*1000:<12.4f} "
            f"{r.get('TTFF_violation_rate',0):<12.2%} {r.get('TBF_violation_rate',0):<12.2%} "
            f"{r.get('TTFF_slope_s_per_s',0):<12.6f}"
        )

    report_lines.append("")
    report_text = "\n".join(report_lines)
    print(report_text)

    # 保存 JSON
    serializable = [{"ec_config": ec, **r} for ec, r in results]
    output_data = {
        "bottleneck": bottleneck,
        "best_by_ttff": best_ttff[0],
        "best_by_tbf": best_tbf[0],
        "best_by_violation": best_viol[0],
        "results": serializable,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")
    return best_ttff[0]


async def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    slo = {"TTFF": args.slo_ttff, "TBF": args.slo_tbf}

    print("Loading dataset...")
    from datasets import load_dataset
    MultiD = load_dataset(args.dataset_path)["validation"]

    # 复用 dataloader 的 prepare_data
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataloader", "dataloader.py")
    dm = importlib.util.module_from_spec(spec)
    sys.modules["dataloader"] = dm
    spec.loader.exec_module(dm)
    convlist = dm.prepare_data(MultiD, args.dataset_path)
    print(f"Loaded {len(convlist)} conversations")

    lambda_rate = args.lambda_rates[0]
    batch_size = args.batch_sizes[0]

    results, bottleneck = await run_sweep(
        convlist,
        lambda_rate=lambda_rate,
        batch_size=batch_size,
        experiment_duration=args.experiment_duration,
        slo=slo,
        wq_config=args.wq_config,
    )

    best = analyze_and_report(results, bottleneck, args.output)
    print(f"\nRecommended SM allocation: {best}")


if __name__ == "__main__":
    asyncio.run(main())

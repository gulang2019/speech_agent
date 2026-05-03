#!/usr/bin/env python3
"""
Replay a recorded model trace at 100% SM allocation and measure throughput.

Usage:
    # First, record a trace:
    python dataloader.py --trace-model LM --ec-config 10,60,30 ...

    # Then, replay at 100% SM:
    python replay_trace.py --trace-file LM_trace.json

    # Or compare multiple SM allocations:
    python replay_trace.py --trace-file LM_trace.json --sm-allocations 1.0 0.5 0.33 --original-sm 0.6
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from request_utils import replay_model_throughput


def load_trace(trace_file: str) -> dict:
    with open(trace_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Replay model trace and measure throughput")
    parser.add_argument("--trace-file", type=str, required=True,
                        help="Path to trace JSON file (e.g. LM_trace.json)")
    parser.add_argument("--sm-allocations", nargs="+", type=float, default=[1.0],
                        help="SM allocation ratios to test (default: 1.0)")
    parser.add_argument("--original-sm", type=float, default=None,
                        help="Original SM allocation during trace recording (for comparison)")
    args = parser.parse_args()

    trace_data = load_trace(args.trace_file)
    model_type = trace_data['model_type']
    model_name = trace_data['model_name']
    trace_batches = trace_data['trace_batches']

    total_in = sum(b['frames_in'] for b in trace_batches)
    total_out = sum(b['frames_out'] for b in trace_batches)
    batch_sizes = [b['batch_size'] for b in trace_batches]
    prefills = sum(1 for b in trace_batches if b['status'] == 'Prefill')
    decodes = sum(1 for b in trace_batches if b['status'] == 'Decode')

    print(f"\n{'='*60}")
    print(f"Trace Replay: {model_name} ({model_type})")
    print(f"Batches: {len(trace_batches)} | Prefill: {prefills} | Decode: {decodes}")
    print(f"Total input frames: {total_in:,} | Total output frames: {total_out:,}")
    print(f"Avg batch size: {sum(batch_sizes)/len(batch_sizes):.2f}")
    print(f"{'='*60}")

    print(f"\n{'SM%':<6} {'Frames In/s':<16} {'Frames Out/s':<16} {'Rounds':<10} {'Round/s':<10} {'Sim Time (s)':<14}")
    print("-" * 70)

    results = {}
    for sm_ratio in args.sm_allocations:
        r = replay_model_throughput(
            model_name=model_name,
            model_type=model_type,
            trace_batches=trace_batches,
            sm_allocation_ratio=sm_ratio,
        )
        sm_pct = int(sm_ratio * 100) if sm_ratio == int(sm_ratio * 100) / 100 else f"{sm_ratio*100:.1f}"
        print(f"{sm_pct}%{'':<4} {r['throughput_in']:<16.2f} {r['throughput_out']:<16.2f} {r['total_rounds']:<10} {r['round_throughput']:<10.2f} {r['total_time_s']:<14.4f}")
        results[sm_ratio] = r

    if args.original_sm and args.original_sm in results:
        print(f"\nComparison (at {int(args.original_sm*100)}% SM original allocation):")
        for sm_ratio, r in results.items():
            speedup = r['throughput_out'] / results[args.original_sm]['throughput_out']
            expected = sm_ratio / args.original_sm
            print(f"  {int(sm_ratio*100)}% SM: {r['throughput_out']:.2f} frames/s "
                  f"(vs original: {speedup:.2f}x, expected linear: {expected:.2f}x)")

    print()


if __name__ == "__main__":
    main()

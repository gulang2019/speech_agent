from typing import List
import numpy as np
from .executor import ExecutionResult


def compute_p99(results: List[ExecutionResult]) -> float:
    latencies = sorted([r.latency for r in results if r.success])
    if not latencies:
        return float('inf')
    idx = int(len(latencies) * 0.99) - 1
    return latencies[max(0, idx)]


def compute_mean(results: List[ExecutionResult]) -> float:
    latencies = [r.latency for r in results if r.success]
    return np.mean(latencies) if latencies else float('inf')


def compute_throughput(results: List[ExecutionResult]) -> float:
    if not results:
        return 0.0
    total_time = max(r.latency for r in results) if results else 0.0
    return len(results) / total_time if total_time > 0 else 0.0


def compute_percentile(results: List[ExecutionResult], percentile: float) -> float:
    latencies = sorted([r.latency for r in results if r.success])
    if not latencies:
        return float('inf')
    idx = int(len(latencies) * percentile / 100.0)
    return latencies[min(idx, len(latencies) - 1)]
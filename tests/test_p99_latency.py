"""
P99 Latency Benchmark for vLLM models.

测试不同模型在不同任务到达率下的 P99 延迟。
参考 framework_summary.md 的 Erlang-k 到达过程。
"""

import os
import sys
import time
import numpy as np
import threading
from dataclasses import dataclass
from datetime import datetime
import csv

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RequestRecord:
    req_id: str
    submission_time: float
    completion_time: float = 0.0

    @property
    def latency_ms(self) -> float:
        return (self.completion_time - self.submission_time) * 1000


class ErlangWorkloadGenerator:
    """Erlang-k 到达过程工作负载生成器。"""

    def __init__(self, lambda_rate: float, erlang_k: int = 50):
        self.lambda_rate = lambda_rate
        self.erlang_k = erlang_k

    def get_next_interval(self) -> float:
        if self.lambda_rate <= 0:
            return float('inf')
        return np.random.gamma(self.erlang_k, 1.0 / (self.erlang_k * self.lambda_rate))


class LatencyCollector:
    """收集请求延迟数据。"""

    def __init__(self):
        self.records: list[RequestRecord] = []
        self._lock = threading.Lock()

    def add_request(self, req_id: str, submission_time: float, completion_time: float):
        with self._lock:
            self.records.append(RequestRecord(req_id, submission_time, completion_time))

    def get_latencies(self) -> list[float]:
        with self._lock:
            return [r.latency_ms for r in self.records]

    def get_percentile(self, percentile: float) -> float:
        latencies = self.get_latencies()
        if not latencies:
            return 0.0
        return float(np.percentile(latencies, percentile))

    def get_p99(self) -> float:
        return self.get_percentile(99)

    def get_mean(self) -> float:
        latencies = self.get_latencies()
        return float(np.mean(latencies)) if latencies else 0.0

    def get_throughput(self, duration: float) -> float:
        return len(self.get_latencies()) / duration if duration > 0 else 0.0

    def reset(self):
        with self._lock:
            self.records.clear()


def _run_single_trial(llm, sampling_params, prompts, batch_size, lambda_rate, duration_seconds):
    """单次测试运行，返回 collector 和 actual_duration。"""
    collector = LatencyCollector()
    workload_gen = ErlangWorkloadGenerator(lambda_rate, erlang_k=50)

    start_time = time.time()
    pending_completions: list[tuple[str, float, float]] = []
    next_arrival = start_time

    while time.time() - start_time < duration_seconds:
        current_time = time.time()

        while next_arrival <= current_time:
            req_id = f"req_{len(pending_completions)}_{int(current_time * 1000)}"
            pending_completions.append((req_id, next_arrival, 0.0))
            next_arrival += workload_gen.get_next_interval()

        if len(pending_completions) >= batch_size:
            batch_reqs = pending_completions[:batch_size]
            pending_completions = pending_completions[batch_size:]

            llm.generate(prompts[:batch_size], sampling_params)
            completion_time = time.time()

            for req_id, sub_time, _ in batch_reqs:
                collector.add_request(req_id, sub_time, completion_time)
        else:
            time.sleep(0.01)

    # 处理剩余请求
    while pending_completions:
        batch_reqs = pending_completions[:batch_size]
        pending_completions = pending_completions[batch_size:]

        llm.generate(prompts[:len(batch_reqs)], sampling_params)
        completion_time = time.time()

        for req_id, sub_time, _ in batch_reqs:
            collector.add_request(req_id, sub_time, completion_time)

    actual_duration = time.time() - start_time
    return collector, actual_duration


def run_benchmark(
    model_name: str,
    lambda_rates: list[float],
    output_csv: str = None,
    duration_seconds: float = 30.0,
    batch_size: int = 8,
    max_tokens: int = 256,
    num_trials: int = 3,
):
    """运行 P99 基准测试，每个 lambda 重复 num_trials 次取平均。"""

    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"Trials per lambda: {num_trials}, Duration per trial: {duration_seconds}s")
    print(f"{'='*60}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
    )

    prompts = [f"Generate a response for request {i}" for i in range(batch_size)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    # 预热
    for _ in range(3):
        llm.generate(prompts[:batch_size], sampling_params)
        time.sleep(0.1)

    results = []
    all_latency_records: list[dict] = []

    for lambda_rate in lambda_rates:
        trial_p99s, trial_p50s, trial_means, trial_thrs = [], [], [], []

        for trial in range(1, num_trials + 1):
            print(f"  λ={lambda_rate:>3} trial {trial}/{num_trials}...", end=" ", flush=True)

            collector, actual_duration = _run_single_trial(
                llm, sampling_params, prompts, batch_size, lambda_rate, duration_seconds
            )
            latencies = collector.get_latencies()

            # 记录逐请求延迟（含 trial 编号）
            for lat in latencies:
                all_latency_records.append({
                    "lambda": lambda_rate,
                    "trial": trial,
                    "latency_ms": lat,
                })

            trial_p99s.append(collector.get_p99())
            trial_p50s.append(collector.get_percentile(50))
            trial_means.append(collector.get_mean())
            trial_thrs.append(collector.get_throughput(actual_duration))

            print(f"P99={trial_p99s[-1]:.0f}ms, Mean={trial_means[-1]:.0f}ms, "
                  f"Thr={trial_thrs[-1]:.2f} r/s, N={len(latencies)}")

        result = {
            "lambda": lambda_rate,
            "p99_ms": float(np.mean(trial_p99s)),
            "p99_std": float(np.std(trial_p99s)),
            "p50_ms": float(np.mean(trial_p50s)),
            "p50_std": float(np.std(trial_p50s)),
            "mean_ms": float(np.mean(trial_means)),
            "mean_std": float(np.std(trial_means)),
            "throughput": float(np.mean(trial_thrs)),
            "throughput_std": float(np.std(trial_thrs)),
        }
        results.append(result)

        print(f"  λ={lambda_rate:>3} avg → P99={result['p99_ms']:.0f}±{result['p99_std']:.0f}ms, "
              f"Mean={result['mean_ms']:.0f}±{result['mean_std']:.0f}ms, "
              f"Thr={result['throughput']:.2f}±{result['throughput_std']:.2f} r/s\n")

    # 保存汇总 CSV
    if output_csv:
        fields = ['lambda', 'p99_ms', 'p99_std', 'p50_ms', 'p50_std',
                  'mean_ms', 'mean_std', 'throughput', 'throughput_std']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"Summary saved to {output_csv}")

    # 保存逐请求延迟 CSV
    if output_csv:
        per_request_csv = output_csv.replace('.csv', '_per_request.csv')
        with open(per_request_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['lambda', 'trial', 'latency_ms'])
            writer.writeheader()
            writer.writerows(all_latency_records)
        print(f"Per-request latencies saved to {per_request_csv}")

    return results, all_latency_records


def main():
    models = [
        "openai-community/gpt2",
        "HuggingFaceTB/SmolLM-135M",
        "state-spaces/mamba-130m",
        "HuggingFaceTB/SmolLM-360M",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ]

    lambda_rates = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    duration_seconds = 120.0
    batch_size = 8
    num_trials = 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/home/liuxunyuan/speech/profile/speech_agent/tests/output"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model_name in models:
        model_short = model_name.split("/")[-1]
        output_csv = os.path.join(output_dir, f"p99_{model_short}_{timestamp}.csv")

        try:
            results, _ = run_benchmark(
                model_name=model_name,
                lambda_rates=lambda_rates,
                output_csv=output_csv,
                duration_seconds=duration_seconds,
                batch_size=batch_size,
                num_trials=num_trials,
            )
            all_results[model_short] = results
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 打印汇总
    print(f"\n{'='*90}")
    print("SUMMARY: P99 Latency (ms) with ±1σ across trials")
    print(f"{'='*90}")
    header = f"{'Lambda':<8}"
    for model_name in all_results:
        header += f"{model_name + ' P99':<22}{model_name + ' Mean':<22}"
    print(header)
    print("-" * 90)

    for i, lambda_rate in enumerate(lambda_rates):
        line = f"{lambda_rate:<8}"
        for model_name in all_results:
            r = all_results[model_name][i]
            line += f"{r['p99_ms']:.0f}±{r['p99_std']:.0f}ms".ljust(22)
            line += f"{r['mean_ms']:.0f}±{r['mean_std']:.0f}ms".ljust(22)
        print(line)


if __name__ == "__main__":
    main()
import argparse
import yaml
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import get_loader
from src.workload import PoissonWorkloadGenerator, Job
from src.executor import BatchExecutor
from src.metrics import compute_p99, compute_throughput
from src.analyzer import BenchmarkAnalyzer


def run_benchmark_for_model(model_config: dict, lambdas: list, duration: float = 30.0):
    loader = get_loader(model_config['type'])
    model = loader.load(
        model_config['name'],
        model_config['path'],
        batch_size=model_config.get('batch_size', 8),
        max_tokens=model_config.get('max_tokens', 256),
        audio_length_ms=model_config.get('audio_length_ms', 30000)
    )

    print(f"Warming up {model_config['name']}...")
    model.warmup()

    results = []
    for lambda_val in lambdas:
        print(f"  Testing lambda={lambda_val}...")
        job_type = 'audio' if model_config['type'] == 'asr' else 'tokens'
        job_size = model_config.get('audio_length_ms', 128) if job_type == 'audio' else model_config.get('max_tokens', 128)

        generator = PoissonWorkloadGenerator(lambda_val, job_size, job_type)
        executor = BatchExecutor(model, model_config.get('batch_size', 8))

        generator.start()
        executor.start()

        time.sleep(duration)

        generator.stop()
        time.sleep(0.5)
        executor.stop()

        exec_results = executor.get_results()
        p99 = compute_p99(exec_results)
        throughput = compute_throughput(exec_results)

        results.append({
            'lambda': lambda_val,
            'p99_latency': p99,
            'throughput': throughput,
            'num_requests': len(exec_results)
        })
        print(f"    P99 latency: {p99:.4f}s, throughput: {throughput:.2f} jobs/s")

    return results


def main():
    parser = argparse.ArgumentParser(description='LM Performance Test Framework')
    parser.add_argument('--model', type=str, help='Specific model to test')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1, 5, 10, 50],
                        help='Arrival rates to test')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Duration for each lambda test in seconds')
    parser.add_argument('--config', type=str, default='config/models.yaml',
                        help='Path to models config')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    analyzer = BenchmarkAnalyzer(args.output_dir)
    all_results = {}

    models_to_test = [m for m in config['models'] if args.model is None or m['name'] == args.model]

    for model_config in models_to_test:
        print(f"\n=== Testing {model_config['name']} ===")
        try:
            results = run_benchmark_for_model(model_config, args.lambdas, args.duration)
            all_results[model_config['name']] = results

            csv_path = analyzer.save_results(model_config['name'], results)
            print(f"Results saved to {csv_path}")
        except Exception as e:
            print(f"Error testing {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        plot_path = analyzer.plot_all_models(all_results)
        print(f"\nComparison plot saved to {plot_path}")

    print("\n=== Benchmark Complete ===")


if __name__ == '__main__':
    main()
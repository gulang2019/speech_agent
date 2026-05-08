import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


class BenchmarkAnalyzer:
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, model_name: str, results: List[Dict]):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, f'{model_name}_results.csv')
        df.to_csv(csv_path, index=False)
        return csv_path

    def plot_p99_vs_lambda(self, results: Dict[str, List[Dict]], output_path: str = None):
        plt.figure(figsize=(10, 6))
        for model_name, data in results.items():
            lambdas = [d['lambda'] for d in data]
            p99s = [d['p99_latency'] for d in data]
            plt.plot(lambdas, p99s, marker='o', label=model_name)

        plt.xlabel('Arrival Rate (lambda) [jobs/sec]')
        plt.ylabel('P99 Latency [sec]')
        plt.title('P99 Latency vs Arrival Rate')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')

        if output_path:
            plt.savefig(output_path)
        else:
            plt.savefig(os.path.join(self.output_dir, 'p99_vs_lambda.png'))
        plt.close()

    def plot_all_models(self, results: Dict[str, List[Dict]]):
        output_path = os.path.join(self.output_dir, 'all_models_comparison.png')
        self.plot_p99_vs_lambda(results, output_path)
        return output_path
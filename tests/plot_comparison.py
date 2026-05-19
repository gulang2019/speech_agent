import csv
import matplotlib.pyplot as plt
import numpy as np

opt_file = "tests/output/p99_opt-350m_20260519_145821.csv"
qwen_file = "tests/output/p99_Qwen2-0.5B-Instruct_20260519_181615.csv"

def load(path):
    ls, p99s, p99_std, means, mean_std, thr = [], [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ls.append(float(row["lambda"]))
            p99s.append(float(row["p99_ms"]) / 1000)
            p99_std.append(float(row["p99_std"]) / 1000)
            means.append(float(row["mean_ms"]) / 1000)
            mean_std.append(float(row["mean_std"]) / 1000)
            thr.append(float(row["throughput"]))
    return ls, p99s, p99_std, means, mean_std, thr

opt = load(opt_file)
qwen = load(qwen_file)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图: P99 对比
ax1.errorbar(opt[0], opt[1], yerr=opt[2], fmt="o-", color="coral", capsize=3,
             linewidth=2, markersize=8, label="OPT-350m P99")
ax1.errorbar(qwen[0], qwen[1], yerr=qwen[2], fmt="s-", color="steelblue", capsize=3,
             linewidth=2, markersize=8, label="Qwen2-0.5B P99")
ax1.set_xlabel("Request Arrival Rate λ (jobs/s)", fontsize=12)
ax1.set_ylabel("P99 Latency (s)", fontsize=12)
ax1.set_title("P99 Latency: OPT-350m vs Qwen2-0.5B", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(opt[0])

# 右图: 吞吐对比
ax2.plot(opt[0], opt[5], "o-", color="coral", linewidth=2, markersize=8, label="OPT-350m")
ax2.plot(qwen[0], qwen[5], "s-", color="steelblue", linewidth=2, markersize=8, label="Qwen2-0.5B")
ax2.axhline(y=14.0, color="steelblue", linestyle=":", alpha=0.4)
ax2.axhline(y=10.0, color="coral", linestyle=":", alpha=0.4)
ax2.set_xlabel("Request Arrival Rate λ (jobs/s)", fontsize=12)
ax2.set_ylabel("Throughput (req/s)", fontsize=12)
ax2.set_title("Throughput: OPT-350m vs Qwen2-0.5B", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(opt[0])

plt.tight_layout()
out = "tests/output/p99_comparison_opt_vs_qwen.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.close()

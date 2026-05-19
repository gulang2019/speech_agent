import csv
import matplotlib.pyplot as plt
import numpy as np

data_file = "/home/liuxunyuan/speech/profile/speech_agent/tests/output/p99_opt-350m_20260519_141126.csv"
per_request_file = "/home/liuxunyuan/speech/profile/speech_agent/tests/output/p99_opt-350m_20260519_141126_per_request.csv"

# 读汇总数据
lambdas, p99s, p50s, means = [], [], [], []
with open(data_file) as f:
    for row in csv.DictReader(f):
        lambdas.append(float(row["lambda"]))
        p99s.append(float(row["p99_ms"]))
        p50s.append(float(row["p50_ms"]))
        means.append(float(row["mean_ms"]))

# 读逐请求数据
req_by_lambda = {}
with open(per_request_file) as f:
    for row in csv.DictReader(f):
        l = float(row["lambda"])
        lat = float(row["latency_ms"])
        req_by_lambda.setdefault(l, []).append(lat)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图: lambda-P99 曲线
ax1.plot(lambdas, p99s, "o-", color="coral", linewidth=2, markersize=8, label="P99")
ax1.plot(lambdas, means, "s--", color="steelblue", linewidth=2, markersize=8, label="Mean")
ax1.set_xlabel("Request Arrival Rate λ (jobs/s)", fontsize=12)
ax1.set_ylabel("Latency (ms)", fontsize=12)
ax1.set_title("OPT-350m: λ vs Latency", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale("log")
ax1.set_yscale("log")

# 标注每个点的值
for l, p, m in zip(lambdas, p99s, means):
    ax1.annotate(f"{p:.0f}ms", (l, p), textcoords="offset points", xytext=(0, 12),
                 fontsize=8, ha="center", color="coral")

# 右图: 每个 lambda 下的延迟分布 (boxplot)
box_data = [req_by_lambda[l] for l in lambdas]
bp = ax2.boxplot(box_data, labels=[str(int(l)) for l in lambdas], patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
ax2.set_xlabel("Request Arrival Rate λ (jobs/s)", fontsize=12)
ax2.set_ylabel("Latency (ms)", fontsize=12)
ax2.set_title("Per-Request Latency Distribution by λ", fontsize=14)
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "/home/liuxunyuan/speech/profile/speech_agent/tests/output/p99_latency_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out_path}")
plt.close()

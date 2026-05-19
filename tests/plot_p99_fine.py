import csv
import matplotlib.pyplot as plt
import numpy as np

data_file = "tests/output/p99_opt-350m_fine_20260519_142253.csv"
per_request_file = "tests/output/p99_opt-350m_fine_20260519_142253_per_request.csv"

lambdas, p99s, p50s, means, thr = [], [], [], [], []
with open(data_file) as f:
    for row in csv.DictReader(f):
        lambdas.append(float(row["lambda"]))
        p99s.append(float(row["p99_ms"]) / 1000)
        p50s.append(float(row["p50_ms"]) / 1000)
        means.append(float(row["mean_ms"]) / 1000)
        thr.append(float(row["throughput"]))

req_by_lambda = {}
with open(per_request_file) as f:
    for row in csv.DictReader(f):
        l = float(row["lambda"])
        lat = float(row["latency_ms"]) / 1000
        req_by_lambda.setdefault(l, []).append(lat)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

# 左图: lambda-P99
ax1.plot(lambdas, p99s, "o-", color="coral", linewidth=2, markersize=8, label="P99")
ax1.plot(lambdas, p50s, "s-", color="seagreen", linewidth=1.5, markersize=7, label="P50")
ax1.plot(lambdas, means, "D--", color="steelblue", linewidth=1.5, markersize=7, label="Mean")
ax1.axvline(x=6, color="gray", linestyle=":", alpha=0.5)
ax1.axvline(x=12, color="gray", linestyle=":", alpha=0.5)
ax1.set_xlabel("Request Arrival Rate λ (jobs/s)", fontsize=12)
ax1.set_ylabel("Latency (s)", fontsize=12)
ax1.set_title("OPT-350m: λ vs Latency (fine sweep, step=2)", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 标注
for l, p in zip(lambdas, p99s):
    ax1.annotate(f"{p:.1f}s", (l, p), textcoords="offset points", xytext=(0, 10),
                 fontsize=7.5, ha="center", color="coral")

# 右图: boxplot
box_data = [req_by_lambda[l] for l in lambdas]
bp = ax2.boxplot(box_data, tick_labels=[str(int(l)) for l in lambdas], patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
ax2.set_xlabel("Request Arrival Rate λ (jobs/s)", fontsize=12)
ax2.set_ylabel("Latency (s)", fontsize=12)
ax2.set_title("Per-Request Latency Distribution by λ", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "tests/output/p99_latency_fine_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()

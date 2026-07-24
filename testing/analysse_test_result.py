import pandas as pd
import numpy as np

# Path to your CSV
csv_file = "quality_report.csv"

# Read CSV
df = pd.read_csv(csv_file, encoding="utf-8")

print("=" * 70)
print("HKL-VITS Quality Evaluation Summary")
print("=" * 70)

# Columns to summarize
metrics = [
    "latency_seconds",
    "duration",
    "pesq",
    "stoi",
    "mcd",
    "wer",
    "cer",
    "words_per_second",
    "chars_per_second",
    "silence_ratio",
    "dynamic_range_db",
    "clipping_percent"
]

summary = []

for metric in metrics:

    if metric not in df.columns:
        continue

    values = pd.to_numeric(df[metric], errors="coerce").dropna()

    if len(values) == 0:
        continue

    summary.append({
        "Metric": metric,
        "Count": len(values),
        "Average": values.mean(),
        "Minimum": values.min(),
        "Maximum": values.max(),
        "Std Dev": values.std(),
        "Median": values.median(),
        "95th Percentile": values.quantile(0.95)
    })

summary_df = pd.DataFrame(summary)

pd.set_option("display.float_format", "{:.6f}".format)

print(summary_df)

# Save summary
summary_file = "quality_summary.csv"
summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

print(f"\nSummary saved to:\n{summary_file}")
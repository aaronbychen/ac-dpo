import json
import pandas as pd

# Load evaluation results
with open("results/evaluation/evaluation_summary.json") as f:
    data = json.load(f)

# Parse results
results = []
for item in data["summaries"]:
    results.append({
        "Model": item["model"],
        "Split": item["split"],
        "Sum Accuracy": item["sum_accuracy"],
        "Token Accuracy": item["avg_token_accuracy"],
        "Sum Margin": item["avg_sum_margin"],
        "Token Margin": item["avg_token_margin"]
    })

df = pd.DataFrame(results)

# Pivot for comparison
comparison = df.pivot_table(
    index="Split",
    columns="Model",
    values=["Sum Accuracy", "Token Accuracy", "Sum Margin", "Token Margin"]
)

print("=" * 80)
print("AC-DPO vs Baseline (r=64) Comparison")
print("=" * 80)
print(comparison.to_string())

# Calculate differences
print("\n" + "=" * 80)
print("Improvement (AC-DPO - Baseline)")
print("=" * 80)
for split in ["eval_easy", "eval_hard", "eval_all"]:
    acdpo = df[(df["Model"] == "acdpo") & (df["Split"] == split)].iloc[0]
    baseline = df[(df["Model"] == "baseline_r64") & (df["Split"] == split)].iloc[0]
    
    print(f"\n{split}:")
    print(f"  Sum Accuracy:   {acdpo['Sum Accuracy'] - baseline['Sum Accuracy']:+.4f}")
    print(f"  Token Accuracy: {acdpo['Token Accuracy'] - baseline['Token Accuracy']:+.4f}")
    print(f"  Sum Margin:     {acdpo['Sum Margin'] - baseline['Sum Margin']:+.4f}")
    print(f"  Token Margin:   {acdpo['Token Margin'] - baseline['Token Margin']:+.4f}")

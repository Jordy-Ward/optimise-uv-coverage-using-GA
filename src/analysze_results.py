import numpy as np
from scipy.stats import mannwhitneyu

# 1. Load the history files (assuming shape is 30 runs x N iterations/generations)
ga_base = np.load("../out/ga_all_histories.npy")
ga_v1 = np.load("../out/ga_v1_all_histories.npy")
ga_v2 = np.load("../out/ga_v2_all_histories.npy")
ga_v3 = np.load("../out/ga_v3_all_histories.npy")
sa_base = np.load("../out/sa_all_histories.npy")

# 2. Extract the FINAL fitness score for each of the 30 runs
# (The last column [-1] contains the final score achieved in that run)
final_scores = {
    "GA Baseline": ga_base[:, -1],
    "Variant 1 (Reduced Selection)": ga_v1[:, -1],
    "Variant 2 (Enhanced Diversity)": ga_v2[:, -1],
    "Variant 3 (High Mutation)": ga_v3[:, -1],
    "Simulated Annealing": sa_base[:, -1]
}

print("--- SUMMARY STATISTICS (30 RUNS) ---")
for name, scores in final_scores.items():
    print(f"{name}:")
    print(f"  Max (Best) : {np.max(scores):.2f}")
    print(f"  Min (Worst): {np.min(scores):.2f}")
    print(f"  Mean       : {np.mean(scores):.2f}")
    print(f"  Std Dev    : {np.std(scores):.2f}\n")

# 3. Perform Mann-Whitney U Tests
print("--- STATISTICAL SIGNIFICANCE (Mann-Whitney U Test) ---")
# Compare GA Baseline to Simulated Annealing
stat, p_val_sa = mannwhitneyu(final_scores["GA Baseline"], final_scores["Simulated Annealing"], alternative='two-sided')
print(f"GA Baseline vs. Simulated Annealing -> p-value: {p_val_sa:.5e}")

# Compare the Best Variant (assuming V2) to GA Baseline
stat, p_val_v2 = mannwhitneyu(final_scores["Variant 2 (Enhanced Diversity)"], final_scores["GA Baseline"], alternative='greater')
print(f"Variant 2 vs. GA Baseline -> p-value: {p_val_v2:.5e}")
import numpy as np
import matplotlib.pyplot as plt

# 1. Load all saved 2D matrices
try:
    ga_base = np.load("../out/ga_all_histories.npy")
    ga_v1 = np.load("../out/ga_v1_all_histories.npy")
    ga_v2 = np.load("../out/ga_v2_all_histories.npy")
    ga_v3 = np.load("../out/ga_v3_all_histories.npy")
    sa_base = np.load("../out/sa_all_histories.npy")
except FileNotFoundError:
    print("Error: Ensure all .npy history files are generated and exist in the ../out/ folder.")
    exit()

# Parameters matching the experiment
POP_SIZE = 500
GENERATIONS = 1000
TOTAL_EVALUATIONS = POP_SIZE * GENERATIONS # 500,000

# 2. X-axis evaluation points
# GA evaluates a whole population per generation
ga_eval_points = np.linspace(POP_SIZE, TOTAL_EVALUATIONS, GENERATIONS)
# SA evaluates one at a time
sa_eval_points = np.arange(1, TOTAL_EVALUATIONS + 1)

# Dictionary for clean plotting loops
ga_variants = {
    "GA: Standard Baseline": (ga_base, 'blue'),
    "GA: Variant One (Selection)": (ga_v1, 'green'),
    "GA: Variant Two (Diversity)": (ga_v2, 'red'),
    "GA: Variant Three (Mutation)": (ga_v3, 'purple')
}

plt.figure(figsize=(14, 8))

# 3. Plot all GA Variants
for label, (data, color) in ga_variants.items():
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.plot(ga_eval_points, mean, label=label, color=color, linewidth=2)
    plt.fill_between(ga_eval_points, mean - std, mean + std, color=color, alpha=0.1)

# 4. Plot Simulated Annealing
sa_mean = np.mean(sa_base, axis=0)
sa_std = np.std(sa_base, axis=0)

downsample = 500 # Downsample SA for clean/fast rendering of 500k points
plt.plot(sa_eval_points[::downsample], sa_mean[::downsample], 
         label='Simulated Annealing', color='orange', linewidth=2.5)
plt.fill_between(sa_eval_points[::downsample], 
                 (sa_mean - sa_std)[::downsample], 
                 (sa_mean + sa_std)[::downsample], 
                 color='orange', alpha=0.15)

# 5. Formatting
# plt.title('Convergence Trajectories: Simulated Annealing vs. GA Variants (30 Runs)', fontsize=14)

# Drastically increased from 14 to 22
plt.xlabel('Total Objective Function Evaluations', fontsize=30) 
plt.ylabel('Weighted $uv$-Coverage Density Score', fontsize=30) 

# Drastically increased from 12 to 18
plt.tick_params(axis='both', which='major', labelsize=25) 

# Increased legend text from 11 to 16
plt.legend(loc='lower right', fontsize=16) 

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.ylim(14200,15000)

# Save matching the filename used in our LaTeX document!
plt.savefig("../out/zoomed_convergence_comparison_30runs.png", dpi=300)
print("Successfully saved plot to ../out/zoomed_convergence_comparison_30runs.png")
plt.show()
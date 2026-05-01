import numpy as np
import matplotlib.pyplot as plt

# Load the saved 2D matrices
try:
    ga_histories = np.load("../out/ga_all_histories.npy")
    sa_histories = np.load("../out/sa_all_histories.npy")
except FileNotFoundError:
    print("Error: Make sure both run_ga.py and run_sa.py have finished and saved their data!")
    exit()

# UPDATE: Match your Deep Search parameters
POP_SIZE = 500
GENERATIONS = 1000
TOTAL_EVALUATIONS = POP_SIZE * GENERATIONS # 500,000

ga_mean = np.mean(ga_histories, axis=0)
ga_std = np.std(ga_histories, axis=0)
sa_mean = np.mean(sa_histories, axis=0)
sa_std = np.std(sa_histories, axis=0)

ga_eval_points = np.linspace(POP_SIZE, TOTAL_EVALUATIONS, GENERATIONS)
sa_eval_points = np.arange(1, TOTAL_EVALUATIONS + 1)

plt.figure(figsize=(12, 7))
plt.plot(ga_eval_points, ga_mean, label='GA (Mean)', color='blue', linewidth=2)
plt.fill_between(ga_eval_points, ga_mean - ga_std, ga_mean + ga_std, color='blue', alpha=0.2)

# Downsample SA for clean rendering of 500k points
downsample = 500 
plt.plot(sa_eval_points[::downsample], sa_mean[::downsample], label='SA (Mean)', color='orange', linewidth=2)
plt.fill_between(sa_eval_points[::downsample], 
                 (sa_mean - sa_std)[::downsample], 
                 (sa_mean + sa_std)[::downsample], 
                 color='orange', alpha=0.2)

plt.title('Convergence: GA vs SA (30 Runs, 500k Evaluations, Gaussian Weighting)')
plt.xlabel('Total Fitness Evaluations')
plt.ylabel('Weighted UV Coverage Density')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("../out/convergence_final_500k.png", dpi=300)
plt.show()
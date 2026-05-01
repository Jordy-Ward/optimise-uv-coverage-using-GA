import numpy as np
import physics 
from alg_solvers import GeneticAlgorithm
import time

meerkat_64_coords = np.genfromtxt("../data/meerkat64.csv", delimiter=",", skip_header=1)


def create_gaussian_weights(grid_size=512, sigma=128):
    """
    Creates a 2D Gaussian weight matrix centered at (grid_size/2).
    Sigma controls how much you favor the center vs the edges.
    """
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    x, y = np.meshgrid(x, y)
    
    center = (grid_size - 1) / 2
    # Standard 2D Gaussian formula
    weights = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    return weights

# Pre-calculate once in your main runner
WEIGHT_MATRIX = create_gaussian_weights(grid_size=512, sigma=128)

def evaluate_layout(binary_chromosome):
    active_indices = np.where(binary_chromosome == 1)[0]
    active_antennas = meerkat_64_coords[active_indices]
    
    uv_tracks = physics.generate_uv_coverage_vectorized(
        antennas=active_antennas, 
        H0_hours=-4, H1_hours=4,
        dec_degree=-74.66, latitude_deg=-30.72,
        time_steps=100
    )
    
    uv_mask = physics.create_uv_mask(uv_tracks, grid_size=512)
    
    # Apply weights: pixels in the center are worth more than pixels at the edge
    weighted_score = np.sum(uv_mask * WEIGHT_MATRIX)
    
    return float(weighted_score)

if __name__ == "__main__":
    NUM_RUNS = 30
    POP_SIZE = 500
    GENERATIONS = 1000
    
    all_histories = []
    best_overall_score = -float('inf')
    best_overall_layout = None

    print(f"--- Starting Genetic Algorithm: {NUM_RUNS} Runs ---")
    start_time = time.time()

    for run in range(NUM_RUNS):
        print(f"\nStarting Run {run + 1}/{NUM_RUNS}...")
        ga = GeneticAlgorithm(num_antennas=64, k=16, pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=0.15)
        best_layout, best_score, history = ga.run(fitness_function=evaluate_layout)
        
        all_histories.append(history)
        
        # Track the absolute best across all 30 runs
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_layout = best_layout
            
        print(f"Run {run + 1} Finished! Best Score for this run: {best_score}")

    total_time = time.time() - start_time
    print(f"\nAll {NUM_RUNS} GA Runs Completed in {total_time/60:.2f} minutes.")
    print(f"Absolute Best Score Found: {best_overall_score}")
    
    # Save the 2D array of histories (30 runs, 500 generations)
    np.save("../out/ga_all_histories.npy", np.array(all_histories))
    np.save("../out/ga_absolute_best_layout.npy", best_overall_layout)
    print("Data saved to out/ga_all_histories.npy and out/ga_absolute_best_layout.npy")
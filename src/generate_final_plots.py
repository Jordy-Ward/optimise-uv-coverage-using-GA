import numpy as np
import matplotlib.pyplot as plt
import physics # Ensure this matches your latest physics file name

# 1. Load the coordinates and the winning layouts
meerkat_64_coords = np.genfromtxt("../data/meerkat64.csv", delimiter=",", skip_header=1)
ga_best_binary = np.load("../out/ga_absolute_best_layout.npy")
sa_best_binary = np.load("../out/sa_absolute_best_layout.npy")

def get_high_fi_data(binary_chromosome, title):
    """Runs a high-resolution simulation for a given layout."""
    active_indices = np.where(binary_chromosome == 1)[0]
    active_antennas = meerkat_64_coords[active_indices]
    
    # Crank up the resolution!
    uv_tracks = physics.generate_uv_coverage_vectorized(
        antennas=active_antennas, 
        H0_hours=-4, H1_hours=4,
        dec_degree=-74.66, latitude_deg=-30.72,
        time_steps=500  # High fidelity time resolution
    )
    
    # Larger grid for better visualization
    uv_mask = physics.create_uv_mask(uv_tracks, grid_size=1024)
    return active_antennas, uv_tracks, uv_mask

# --- Process GA and SA Results ---
ga_antennas, ga_tracks, ga_mask = get_high_fi_data(ga_best_binary, "GA Best")
sa_antennas, sa_tracks, sa_mask = get_high_fi_data(sa_best_binary, "SA Best")

# --- Plotting 2x2 Grid (Physical Layout vs UV Coverage) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Row 1: Genetic Algorithm
axes[0, 0].scatter(ga_antennas[:, 0], ga_antennas[:, 1], c='blue', s=40)
axes[0, 0].set_title("GA Best Physical Layout (ENU)")
axes[0, 0].set_aspect('equal')

axes[0, 1].imshow(ga_mask, cmap='hot', origin='upper')
axes[0, 1].set_title(f"GA UV Coverage (Density: {np.sum(ga_mask)})")

# Row 2: Simulated Annealing
axes[1, 0].scatter(sa_antennas[:, 0], sa_antennas[:, 1], c='orange', s=40)
axes[1, 0].set_title("SA Best Physical Layout (ENU)")
axes[1, 0].set_aspect('equal')

axes[1, 1].imshow(sa_mask, cmap='hot', origin='upper')
axes[1, 1].set_title(f"SA UV Coverage (Density: {np.sum(sa_mask)})")

plt.tight_layout()
plt.savefig("../out/high_fi_comparison.png", dpi=300)
plt.show()
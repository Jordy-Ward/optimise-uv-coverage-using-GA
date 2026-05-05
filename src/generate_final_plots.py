import numpy as np
import matplotlib.pyplot as plt
import physics 

# Force all axis numericals (ticks) to be larger for academic legibility
plt.rcParams.update({
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# 1. Load the coordinates and the winning layouts
meerkat_64_coords = np.genfromtxt("../data/meerkat64.csv", delimiter=",", skip_header=1)
ga_best_binary = np.load("../out/ga_absolute_best_layout.npy")
sa_best_binary = np.load("../out/sa_absolute_best_layout.npy")

def get_high_fi_data(binary_chromosome):
    """Runs a high-resolution simulation for a given layout."""
    active_indices = np.where(binary_chromosome == 1)[0]
    active_antennas = meerkat_64_coords[active_indices]
    
    uv_tracks = physics.generate_uv_coverage_vectorized(
        antennas=active_antennas, 
        H0_hours=-4, H1_hours=4,
        dec_degree=-74.66, latitude_deg=-30.72,
        time_steps=500  # High fidelity
    )
    uv_mask = physics.create_uv_mask(uv_tracks, grid_size=1024)
    return active_antennas, uv_mask

ga_antennas, ga_mask = get_high_fi_data(ga_best_binary)
sa_antennas, sa_mask = get_high_fi_data(sa_best_binary)

# --- Plot 1: GA Physical Layout ---
plt.figure(figsize=(7, 7))
plt.scatter(ga_antennas[:, 0], ga_antennas[:, 1], c='blue', s=80)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("../out/ga_physical.png", dpi=300)
plt.close()

# --- Plot 2: GA UV Coverage ---
plt.figure(figsize=(7, 7))
plt.imshow(ga_mask, cmap='hot', origin='upper')
plt.tight_layout()
plt.savefig("../out/ga_uv.png", dpi=300)
plt.close()

# --- Plot 3: SA Physical Layout ---
plt.figure(figsize=(7, 7))
plt.scatter(sa_antennas[:, 0], sa_antennas[:, 1], c='orange', s=80)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("../out/sa_physical.png", dpi=300)
plt.close()

# --- Plot 4: SA UV Coverage ---
plt.figure(figsize=(7, 7))
plt.imshow(sa_mask, cmap='hot', origin='upper')
plt.tight_layout()
plt.savefig("../out/sa_uv.png", dpi=300)
plt.close()

print("Successfully saved 4 title-free, large-axis images to ../out/")
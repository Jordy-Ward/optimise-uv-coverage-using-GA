import numpy

def generate_uv_coverage_vectorized(antennas, H0_hours, H1_hours, dec_degree, latitude_deg, freq=1.4e9, time_steps=100):
    """
    Fully vectorized UV coverage generator. Zero Python loops.
    """
    # 1. Vectorized Baseline Generation
    # Gets all unique pairs of antennas without loops
    antennas = numpy.array(antennas)
    idx1, idx2 = numpy.triu_indices(len(antennas), k=1)
    baselines = antennas[idx2] - antennas[idx1]  # Shape: (B, 3) where B is number of baselines

    # 2. Vectorized XYZ Translation
    L = numpy.radians(latitude_deg)
    E_val, N_val, U_val = baselines[:, 0], baselines[:, 1], baselines[:, 2]

    distance = numpy.sqrt(E_val**2 + N_val**2 + U_val**2)
    azimuth = numpy.arctan2(E_val, N_val)
    elevation = numpy.arcsin(U_val / distance)

    # Calculate XYZ coordinates for all baselines simultaneously
    XYZ = distance[:, numpy.newaxis] * numpy.column_stack((
        numpy.cos(L) * numpy.sin(elevation) - numpy.sin(L) * numpy.cos(elevation) * numpy.cos(azimuth),
        numpy.cos(elevation) * numpy.sin(azimuth),
        numpy.sin(L) * numpy.sin(elevation) + numpy.cos(L) * numpy.cos(elevation) * numpy.cos(azimuth)
    ))  # Shape: (B, 3)

    # 3. Vectorized UV Projection over Time
    H = numpy.radians(numpy.linspace(H0_hours, H1_hours, time_steps) * 15)
    dec = numpy.radians(dec_degree)
    lamda_inverse = freq / 3e8

    # Pre-calculate trig functions for all time steps
    sin_H = numpy.sin(H)
    cos_H = numpy.cos(H)
    sin_dec = numpy.sin(dec)
    cos_dec = numpy.cos(dec)

    # Build a 3D tensor of projection matrices for all time steps. Shape: (T, 2, 3)
    M = numpy.zeros((time_steps, 2, 3))
    M[:, 0, 0] = sin_H
    M[:, 0, 1] = cos_H
    M[:, 1, 0] = -sin_dec * cos_H
    M[:, 1, 1] = sin_dec * sin_H
    M[:, 1, 2] = cos_dec
    M *= lamda_inverse

    # Matrix multiplication: M (T, 2, 3) times XYZ transposed (3, B) -> Output (T, 2, B)
    uv_coords_tensor = numpy.matmul(M, XYZ.T)

    # 4. Flatten and add Conjugates
    # Swap axes to get shape (T, B, 2) and flatten to a 2D array of (u, v) pairs
    uv_tracks = numpy.transpose(uv_coords_tensor, (0, 2, 1)).reshape(-1, 2)

    # Add the conjugate points (-u, -v) by concatenating the inverted array
    uv_tracks_full = numpy.concatenate([uv_tracks, -uv_tracks], axis=0)

    return uv_tracks_full
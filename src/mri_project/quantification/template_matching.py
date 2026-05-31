"""Template matching for MRF quantitative map estimation."""

from __future__ import annotations

import numpy as np


def generate_mock_coeff_maps(
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    dict_compressed: np.ndarray,
    nx: int = 100,
    ny: int = 100,
    noise_level: float = 0.03,
    seed: int = 99,
) -> np.ndarray:
    """Generate mock subspace coefficient maps for template-matching checks."""
    t1_grid = np.asarray(t1_grid)
    t2_grid = np.asarray(t2_grid)
    dict_compressed = np.asarray(dict_compressed)

    if dict_compressed.ndim != 3:
        raise ValueError("dict_compressed must have shape (n_t1, n_t2, n_bases).")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")

    num_bases = dict_compressed.shape[2]
    idx_t1_gm = int(np.argmin(np.abs(t1_grid - 1350)))
    idx_t2_gm = int(np.argmin(np.abs(t2_grid - 80)))
    idx_t1_wm = int(np.argmin(np.abs(t1_grid - 850)))
    idx_t2_wm = int(np.argmin(np.abs(t2_grid - 60)))

    signal_gm = dict_compressed[idx_t1_gm, idx_t2_gm, :]
    signal_wm = dict_compressed[idx_t1_wm, idx_t2_wm, :]

    coeff_maps = np.empty((nx, ny, num_bases), dtype=np.complex128)
    coeff_maps[:, : ny // 2, :] = signal_gm
    coeff_maps[:, ny // 2 :, :] = signal_wm

    rng = np.random.default_rng(seed)
    noise = noise_level * (
        rng.standard_normal(coeff_maps.shape) + 1j * rng.standard_normal(coeff_maps.shape)
    )
    return coeff_maps + noise


def execute_template_matching(
    coeff_maps: np.ndarray,
    dict_compressed: np.ndarray,
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match subspace coefficient maps to a compressed MRF dictionary.

    The match is based on maximum normalized complex inner product. Returns
    quantitative T1, T2, and proton-density maps.
    """
    coeff_maps = np.asarray(coeff_maps)
    dict_compressed = np.asarray(dict_compressed)
    t1_grid = np.asarray(t1_grid)
    t2_grid = np.asarray(t2_grid)

    if coeff_maps.ndim != 3:
        raise ValueError("coeff_maps must have shape (nx, ny, n_bases).")
    if dict_compressed.ndim != 3:
        raise ValueError("dict_compressed must have shape (n_t1, n_t2, n_bases).")
    if coeff_maps.shape[2] != dict_compressed.shape[2]:
        raise ValueError("coeff_maps and dict_compressed must have the same basis dimension.")
    if t1_grid.shape[0] != dict_compressed.shape[0]:
        raise ValueError("t1_grid length must match dict_compressed.shape[0].")
    if t2_grid.shape[0] != dict_compressed.shape[1]:
        raise ValueError("t2_grid length must match dict_compressed.shape[1].")

    nx, ny, num_bases = coeff_maps.shape
    num_t1, num_t2, _ = dict_compressed.shape

    norm_dict = np.linalg.norm(dict_compressed, axis=-1, keepdims=True)
    norm_dict[norm_dict == 0] = np.inf
    dict_flat = (dict_compressed / norm_dict).reshape(-1, num_bases)

    norm_maps = np.linalg.norm(coeff_maps, axis=-1, keepdims=True)
    pd_map = norm_maps.reshape(nx, ny)
    safe_norm_maps = norm_maps.copy()
    safe_norm_maps[safe_norm_maps == 0] = np.inf
    maps_flat = (coeff_maps / safe_norm_maps).reshape(-1, num_bases)

    similarity = np.abs(maps_flat @ dict_flat.conj().T)
    best_match_indices = np.argmax(similarity, axis=1)
    idx_t1, idx_t2 = np.unravel_index(best_match_indices, (num_t1, num_t2))

    t1_map = t1_grid[idx_t1].reshape(nx, ny)
    t2_map = t2_grid[idx_t2].reshape(nx, ny)
    return t1_map, t2_map, pd_map

"""MRF forward k-space simulation."""

from __future__ import annotations

import numpy as np

from mri_project.recon.nufft_ops import nufft_forward


def _nearest_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    grid = np.asarray(grid)
    assert grid.ndim == 1 and grid.size > 0, "grid must be a non-empty 1D array"
    return np.abs(values[..., np.newaxis] - grid).argmin(axis=-1)


def _default_sens_maps(n_coils: int, shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    sens_maps = np.empty((n_coils, *shape), dtype=np.complex64)
    for coil in range(n_coils):
        angle = 2.0 * np.pi * coil / max(n_coils, 1)
        center_y = 0.45 * np.sin(angle)
        center_x = 0.45 * np.cos(angle)
        magnitude = np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / 1.4)
        phase = np.exp(1j * (xx * np.cos(angle) + yy * np.sin(angle)))
        sens_maps[coil] = magnitude * phase
    return sens_maps


def simulate_kspace(
    param_map: np.ndarray,
    signal_dict: np.ndarray,
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    traj: np.ndarray,
    n_coils: int = 12,
    noise_level: float = 0.25,
    sens_maps: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate multi-coil non-Cartesian k-space from a parameter map.

    Args:
        param_map: ``(H, W, 3)`` map containing T1, T2, and PD.
        signal_dict: ``(n_t1, n_t2, n_tr)`` complex MRF dictionary.
        t1_grid: T1 grid values matching ``signal_dict`` axis 0.
        t2_grid: T2 grid values matching ``signal_dict`` axis 1.
        traj: ``(n_tr, n_samples, 2)`` coordinates in SigPy pixel units.

    Returns:
        Complex array with shape ``(n_coils, n_tr, n_samples)``.
    """

    param_map = np.asarray(param_map)
    signal_dict = np.asarray(signal_dict)
    traj = np.asarray(traj)

    assert param_map.ndim == 3 and param_map.shape[-1] == 3, "param_map must have shape (H, W, 3)"
    assert signal_dict.ndim == 3, "signal_dict must have shape (n_t1, n_t2, n_tr)"
    assert traj.ndim == 3 and traj.shape[-1] == 2, "traj must have shape (n_tr, n_samples, 2)"
    assert signal_dict.shape[:2] == (len(t1_grid), len(t2_grid)), "dictionary grid dimensions do not match"
    assert signal_dict.shape[2] == traj.shape[0], "dictionary n_tr must match trajectory n_tr"
    assert n_coils > 0, "n_coils must be positive"
    assert noise_level >= 0, "noise_level must be non-negative"

    shape = param_map.shape[:2]
    t1_idx = _nearest_indices(param_map[..., 0], np.asarray(t1_grid))
    t2_idx = _nearest_indices(param_map[..., 1], np.asarray(t2_grid))
    pd = param_map[..., 2].astype(np.float32)

    img_series = signal_dict[t1_idx, t2_idx, :] * pd[..., np.newaxis]
    img_series = np.moveaxis(img_series, -1, 0).astype(np.complex64)

    if sens_maps is None:
        sens_maps = _default_sens_maps(n_coils, shape)
    else:
        sens_maps = np.asarray(sens_maps)
        assert sens_maps.shape == (n_coils, *shape), "sens_maps must have shape (n_coils, H, W)"

    kspace = np.empty((n_coils, traj.shape[0], traj.shape[1]), dtype=np.complex64)
    for coil in range(n_coils):
        for tr_index in range(traj.shape[0]):
            coil_image = img_series[tr_index] * sens_maps[coil]
            kspace[coil, tr_index] = nufft_forward(coil_image, traj[tr_index])

    if noise_level > 0:
        rng = np.random.default_rng(seed)
        sigma = np.mean(np.abs(kspace)) * noise_level
        noise = rng.normal(0.0, sigma, kspace.shape) + 1j * rng.normal(0.0, sigma, kspace.shape)
        kspace = kspace + noise.astype(np.complex64)

    return kspace

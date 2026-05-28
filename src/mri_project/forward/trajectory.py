"""2D spiral trajectory generation."""

from __future__ import annotations

import numpy as np


def generate_spiral_trajectory(
    fov: float = 220e-3,
    res: float = 1e-3,
    readout_time: float = 7e-3,
    undersample_center: int = 12,
    undersample_edge: int = 24,
    adc_rate: float = 1e6,
) -> np.ndarray:
    """Generate one variable-density Archimedean spiral in SigPy pixel units.

    SigPy's NUFFT expects coordinates in image-pixel frequency units, roughly
    ``[-shape / 2, shape / 2]``. The returned coordinate array has shape
    ``(n_samples, 2)`` ordered as ``(ky, kx)``.
    """

    assert fov > 0, "fov must be positive"
    assert res > 0, "res must be positive"
    assert readout_time > 0, "readout_time must be positive"
    assert undersample_center > 0, "undersample_center must be positive"
    assert undersample_edge > 0, "undersample_edge must be positive"
    assert adc_rate > 0, "adc_rate must be positive"

    matrix_size = int(round(fov / res))
    assert matrix_size > 0, "fov / res must produce a positive matrix size"

    n_points = max(2, int(round(readout_time * adc_rate)))
    k_max = matrix_size / 2.0
    radius = np.linspace(0.0, k_max, n_points, dtype=np.float32)

    alpha = np.log(undersample_edge / undersample_center) / max(k_max, 1.0)
    density = undersample_center * np.exp(alpha * radius)
    theta = 2.0 * np.pi * np.cumsum(density) / matrix_size

    ky = radius * np.sin(theta)
    kx = radius * np.cos(theta)
    return np.stack([ky, kx], axis=1).astype(np.float32)


def generate_tga_trajectory(
    n_tr: int = 500,
    tiny_golden_angle: float = 23.63,
    **spiral_kwargs: float,
) -> np.ndarray:
    """Generate a tiny-golden-angle rotated spiral stack.

    Returns:
        Array with shape ``(n_tr, n_samples, 2)``.
    """

    assert n_tr > 0, "n_tr must be positive"
    traj_single = generate_spiral_trajectory(**spiral_kwargs)
    angles = np.deg2rad(np.arange(n_tr, dtype=np.float32) * tiny_golden_angle)

    traj_full = np.empty((n_tr, traj_single.shape[0], 2), dtype=np.float32)
    for index, theta in enumerate(angles):
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )
        traj_full[index] = traj_single @ rotation.T

    return traj_full

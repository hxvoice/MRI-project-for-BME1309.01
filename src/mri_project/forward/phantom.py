"""Simple 2D brain phantom preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import plot_phantom, save_npy


def prepare_phantom(
    shape: tuple[int, int] = (220, 220),
    save_path: str | Path | None = None,
    figure_path: str | Path | None = None,
) -> np.ndarray:
    """Create a three-layer 2D brain parameter map.

    Returns:
        Array with shape ``(height, width, 3)`` storing ``T1``, ``T2``, and
        proton-density values.
    """

    assert len(shape) == 2, "shape must be a 2D tuple"
    assert shape[0] > 0 and shape[1] > 0, "shape must be positive"

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )

    brain_mask = (xx**2 + yy**2) < 0.8**2
    wm_mask = (xx**2 + yy**2) < 0.5**2
    gm_mask = brain_mask & ~wm_mask
    csf_mask = ((xx**2 + yy**2) < 0.85**2) & ~brain_mask

    t1 = np.zeros(shape, dtype=np.float32)
    t2 = np.zeros(shape, dtype=np.float32)
    pd = np.zeros(shape, dtype=np.float32)

    t1[wm_mask] = 850.0
    t2[wm_mask] = 60.0
    pd[wm_mask] = 0.8

    t1[gm_mask] = 1350.0
    t2[gm_mask] = 80.0
    pd[gm_mask] = 0.9

    t1[csf_mask] = 4000.0
    t2[csf_mask] = 2000.0
    pd[csf_mask] = 1.0

    if pd.max() > 0:
        pd = pd / pd.max()

    param_map = np.stack([t1, t2, pd], axis=-1)

    if save_path is not None:
        save_npy(param_map, save_path)
    if figure_path is not None:
        plot_phantom(t1, t2, pd, figure_path)

    return param_map

"""Simple 2D brain phantom preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import plot_phantom, save_npy


def prepare_phantom(
    shape: tuple[int, int] = (96, 96),
    brain_radius: float = 0.8,
    wm_radius: float = 0.5,
    csf_radius: float = 0.85,
    tissue_values: dict[str, dict[str, float]] | None = None,
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
    assert brain_radius > 0, "brain_radius must be positive"
    assert wm_radius > 0, "wm_radius must be positive"
    assert csf_radius > 0, "csf_radius must be positive"
    tissue_values = {
        "wm": {"t1": 850.0, "t2": 60.0, "pd": 0.8},
        "gm": {"t1": 1350.0, "t2": 80.0, "pd": 0.9},
        "csf": {"t1": 4000.0, "t2": 2000.0, "pd": 1.0},
    } if tissue_values is None else tissue_values
    for tissue_name in ("wm", "gm", "csf"):
        assert tissue_name in tissue_values, f"missing tissue values for {tissue_name}"
        for field_name in ("t1", "t2", "pd"):
            assert field_name in tissue_values[tissue_name], f"missing {field_name} for {tissue_name}"

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )

    brain_mask = (xx**2 + yy**2) < brain_radius**2
    wm_mask = (xx**2 + yy**2) < wm_radius**2
    gm_mask = brain_mask & ~wm_mask
    csf_mask = ((xx**2 + yy**2) < csf_radius**2) & ~brain_mask

    t1 = np.zeros(shape, dtype=np.float32)
    t2 = np.zeros(shape, dtype=np.float32)
    pd = np.zeros(shape, dtype=np.float32)

    t1[wm_mask] = tissue_values["wm"]["t1"]
    t2[wm_mask] = tissue_values["wm"]["t2"]
    pd[wm_mask] = tissue_values["wm"]["pd"]

    t1[gm_mask] = tissue_values["gm"]["t1"]
    t2[gm_mask] = tissue_values["gm"]["t2"]
    pd[gm_mask] = tissue_values["gm"]["pd"]

    t1[csf_mask] = tissue_values["csf"]["t1"]
    t2[csf_mask] = tissue_values["csf"]["t2"]
    pd[csf_mask] = tissue_values["csf"]["pd"]

    if pd.max() > 0:
        pd = pd / pd.max()

    param_map = np.stack([t1, t2, pd], axis=-1)

    if save_path is not None:
        save_npy(param_map, save_path)
    if figure_path is not None:
        plot_phantom(t1, t2, pd, figure_path)

    return param_map

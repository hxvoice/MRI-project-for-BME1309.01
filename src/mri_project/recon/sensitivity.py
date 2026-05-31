"""Coil sensitivity estimation for multi-coil non-Cartesian MRF data."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import tempfile

import numpy as np


def _ensure_numba_cache_dir() -> None:
    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    cache_dir = Path(tempfile.gettempdir()) / "mri_project_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


def _validate_multicoil_time_kspace(kspace: np.ndarray) -> np.ndarray:
    kspace = np.asarray(kspace)
    assert kspace.ndim == 3, f"kspace must have shape (n_coils, n_tr, n_samples), got {kspace.shape}"
    assert kspace.shape[0] > 0, "kspace must contain at least one coil"
    assert kspace.shape[1] > 0, "kspace must contain at least one TR"
    assert kspace.shape[2] > 0, "kspace must contain at least one sample"
    assert np.iscomplexobj(kspace), "kspace must be complex-valued"
    assert np.all(np.isfinite(kspace)), "kspace contains non-finite values"
    return kspace


def _validate_coord(coord: np.ndarray, n_tr: int, n_samples: int) -> np.ndarray:
    coord = np.asarray(coord)
    expected_shape = (n_tr, n_samples, 2)
    assert coord.shape == expected_shape, f"coord must have shape {expected_shape}, got {coord.shape}"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert np.all(np.isfinite(coord)), "coord contains non-finite values"
    return coord


def _validate_img_shape(img_shape: Sequence[int]) -> tuple[int, int]:
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"
    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    return shape


def time_average_kspace(kspace: np.ndarray) -> np.ndarray:
    """Average multi-coil k-space over TR/time dimension."""

    kspace = _validate_multicoil_time_kspace(kspace)
    averaged = np.mean(kspace, axis=1)
    expected_shape = (kspace.shape[0], kspace.shape[2])
    assert averaged.shape == expected_shape, f"averaged kspace shape {averaged.shape} must match {expected_shape}"
    assert np.iscomplexobj(averaged), "averaged kspace must be complex-valued"
    assert np.all(np.isfinite(averaged)), "averaged kspace contains non-finite values"
    return np.asarray(averaged)


def grid_center_calibration_kspace(
    averaged_kspace: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    center_width: int = 24,
) -> np.ndarray:
    """Nearest-neighbor grid central non-Cartesian samples for ESPIRiT calibration."""

    averaged_kspace = np.asarray(averaged_kspace)
    assert averaged_kspace.ndim == 2, (
        f"averaged_kspace must have shape (n_coils, n_samples), got {averaged_kspace.shape}"
    )
    assert averaged_kspace.shape[0] > 0 and averaged_kspace.shape[1] > 0, "averaged_kspace dimensions must be positive"
    assert np.iscomplexobj(averaged_kspace), "averaged_kspace must be complex-valued"
    assert np.all(np.isfinite(averaged_kspace)), "averaged_kspace contains non-finite values"

    coord = np.asarray(coord)
    assert coord.ndim == 3 and coord.shape[-1] == 2, f"coord must have shape (n_tr, n_samples, 2), got {coord.shape}"
    assert coord.shape[1] == averaged_kspace.shape[1], (
        f"coord n_samples ({coord.shape[1]}) must match averaged_kspace samples ({averaged_kspace.shape[1]})"
    )
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert np.all(np.isfinite(coord)), "coord contains non-finite values"

    shape = _validate_img_shape(img_shape)
    center_width = int(center_width)
    assert center_width > 0, "center_width must be positive"

    representative_coord = coord[0]
    radius = np.linalg.norm(representative_coord, axis=1)
    center_mask = radius <= center_width / 2.0
    assert np.any(center_mask), "no samples found inside requested center_width"

    height, width = shape
    selected_coord = representative_coord[center_mask]
    selected_kspace = averaged_kspace[:, center_mask]

    rows = np.rint(selected_coord[:, 0] + height / 2.0).astype(np.int64)
    cols = np.rint(selected_coord[:, 1] + width / 2.0).astype(np.int64)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    assert np.any(valid), "center samples do not map inside img_shape"

    rows = rows[valid]
    cols = cols[valid]
    selected_kspace = selected_kspace[:, valid]

    calib_kspace = np.zeros((averaged_kspace.shape[0], height, width), dtype=averaged_kspace.dtype)
    sample_count = np.zeros((height, width), dtype=np.float32)

    for sample_index, (row, col) in enumerate(zip(rows, cols, strict=True)):
        calib_kspace[:, row, col] += selected_kspace[:, sample_index]
        sample_count[row, col] += 1.0

    sampled = sample_count > 0
    calib_kspace[:, sampled] /= sample_count[sampled]
    assert np.all(np.isfinite(calib_kspace)), "calibration kspace contains non-finite values"
    return calib_kspace


def estimate_sens_maps_espirit(
    kspace: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    center_width: int = 24,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop: float = 0.95,
    max_iter: int = 100,
    show_pbar: bool = False,
) -> np.ndarray:
    """Estimate coil sensitivity maps from multi-TR non-Cartesian k-space using ESPIRiT.

    The input k-space is first averaged across TRs. Central low-frequency
    samples are then nearest-neighbor gridded onto a Cartesian calibration
    array before calling ``sigpy.mri.app.EspiritCalib``.
    """

    kspace = _validate_multicoil_time_kspace(kspace)
    coord = _validate_coord(coord, kspace.shape[1], kspace.shape[2])
    shape = _validate_img_shape(img_shape)

    averaged_kspace = time_average_kspace(kspace)
    calib_kspace = grid_center_calibration_kspace(
        averaged_kspace,
        coord,
        shape,
        center_width=center_width,
    )

    _ensure_numba_cache_dir()
    import sigpy.mri.app as sigpy_mri_app

    sens_maps = sigpy_mri_app.EspiritCalib(
        calib_kspace,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        crop=crop,
        max_iter=max_iter,
        show_pbar=show_pbar,
    ).run()
    sens_maps = np.asarray(sens_maps)

    expected_shape = (kspace.shape[0], *shape)
    assert sens_maps.shape == expected_shape, f"sens_maps shape {sens_maps.shape} must match {expected_shape}"
    assert np.iscomplexobj(sens_maps), "sens_maps must be complex-valued"
    assert np.all(np.isfinite(sens_maps)), "sens_maps contains non-finite values"
    return sens_maps

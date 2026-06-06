"""Local low-rank regularization utilities for subspace MRF."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mri_project.array_backend import ArrayBackend, get_array_backend


def _validate_patch_shape(patch_shape: Sequence[int]) -> tuple[int, int]:
    assert len(patch_shape) == 2, f"patch_shape must have length 2, got {patch_shape}"
    shape = tuple(int(size) for size in patch_shape)
    assert all(size > 0 for size in shape), f"patch_shape must be positive, got {patch_shape}"
    return shape


def _validate_coeff_maps(coeff_maps: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    coeff_maps = backend.to_device(coeff_maps)
    assert coeff_maps.ndim == 3, f"coeff_maps must have shape (rank, H, W), got {coeff_maps.shape}"
    assert coeff_maps.shape[0] > 0, "coeff_maps rank must be positive"
    assert coeff_maps.shape[1] > 0 and coeff_maps.shape[2] > 0, (
        f"coeff_maps spatial shape must be positive, got {coeff_maps.shape[1:]}"
    )
    assert coeff_maps.dtype.kind == "c", "coeff_maps must be complex-valued"
    assert backend.all_finite(coeff_maps), "coeff_maps contains non-finite values"
    return coeff_maps


def _iter_patch_slices(height: int, width: int, patch_shape: tuple[int, int]):
    patch_h, patch_w = patch_shape
    for y0 in range(0, height, patch_h):
        y1 = min(y0 + patch_h, height)
        for x0 in range(0, width, patch_w):
            x1 = min(x0 + patch_w, width)
            yield slice(y0, y1), slice(x0, x1)


def llr_soft_threshold(
    coeff_maps: np.ndarray,
    patch_shape: Sequence[int] = (8, 8),
    threshold: float = 0.0,
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply non-overlapping local low-rank singular-value soft-thresholding."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    coeff_maps = _validate_coeff_maps(coeff_maps, backend)
    patch_shape = _validate_patch_shape(patch_shape)
    threshold = float(threshold)
    assert threshold >= 0.0, "threshold must be non-negative"
    assert np.isfinite(threshold), "threshold must be finite"

    if threshold == 0.0:
        return coeff_maps.copy()

    rank, height, width = coeff_maps.shape
    denoised = np.empty_like(coeff_maps)
    for y_slice, x_slice in _iter_patch_slices(height, width, patch_shape):
        patch = coeff_maps[:, y_slice, x_slice]
        matrix = patch.reshape(rank, -1)
        u, singular_values, vh = xp.linalg.svd(matrix, full_matrices=False)
        singular_values = xp.maximum(singular_values - threshold, 0.0)
        denoised[:, y_slice, x_slice] = ((u * singular_values) @ vh).reshape(patch.shape)

    assert denoised.shape == coeff_maps.shape, (
        f"llr_soft_threshold returned shape {denoised.shape}, expected {coeff_maps.shape}"
    )
    assert denoised.dtype.kind == "c", "llr_soft_threshold returned a non-complex array"
    assert backend.all_finite(denoised), "llr_soft_threshold returned non-finite values"
    return denoised


def llr_nuclear_norm(
    coeff_maps: np.ndarray,
    patch_shape: Sequence[int] = (8, 8),
    device: str = "cpu",
    device_id: int = 0,
) -> float:
    """Return the summed nuclear norm over non-overlapping local patches."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    coeff_maps = _validate_coeff_maps(coeff_maps, backend)
    patch_shape = _validate_patch_shape(patch_shape)

    rank, height, width = coeff_maps.shape
    norm = 0.0
    for y_slice, x_slice in _iter_patch_slices(height, width, patch_shape):
        patch = coeff_maps[:, y_slice, x_slice]
        singular_values = xp.linalg.svd(patch.reshape(rank, -1), compute_uv=False)
        norm += float(backend.scalar_to_python(xp.sum(singular_values)))

    assert np.isfinite(norm), "LLR nuclear norm is non-finite"
    return norm

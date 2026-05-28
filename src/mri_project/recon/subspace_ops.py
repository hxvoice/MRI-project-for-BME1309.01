"""MRF subspace operators composed with 2D NUFFT wrappers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .nufft_ops import nufft_adjoint, nufft_forward


def _validate_basis(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis)
    assert basis.ndim == 2, f"basis must have shape (n_tr, rank), got {basis.shape}"
    assert basis.shape[0] > 0, "basis must contain at least one TR"
    assert basis.shape[1] > 0, "basis rank must be positive"
    assert np.issubdtype(basis.dtype, np.number), "basis must be numeric"
    assert np.all(np.isfinite(basis)), "basis contains non-finite values"
    return basis


def _validate_coeff_maps(coeff_maps: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coeff_maps = np.asarray(coeff_maps)
    assert coeff_maps.ndim == 3, f"coeff_maps must have shape (rank, H, W), got {coeff_maps.shape}"
    assert coeff_maps.shape[0] == basis.shape[1], (
        f"coeff_maps rank ({coeff_maps.shape[0]}) must match basis rank ({basis.shape[1]})"
    )
    assert coeff_maps.shape[1] > 0 and coeff_maps.shape[2] > 0, (
        f"coeff_maps spatial shape must be positive, got {coeff_maps.shape[1:]}"
    )
    assert np.iscomplexobj(coeff_maps), "coeff_maps must be complex-valued"
    assert np.all(np.isfinite(coeff_maps)), "coeff_maps contains non-finite values"
    return coeff_maps


def _validate_time_images(time_images: np.ndarray, basis: np.ndarray) -> np.ndarray:
    time_images = np.asarray(time_images)
    assert time_images.ndim == 3, f"time_images must have shape (n_tr, H, W), got {time_images.shape}"
    assert time_images.shape[0] == basis.shape[0], (
        f"time_images n_tr ({time_images.shape[0]}) must match basis n_tr ({basis.shape[0]})"
    )
    assert time_images.shape[1] > 0 and time_images.shape[2] > 0, (
        f"time_images spatial shape must be positive, got {time_images.shape[1:]}"
    )
    assert np.iscomplexobj(time_images), "time_images must be complex-valued"
    assert np.all(np.isfinite(time_images)), "time_images contains non-finite values"
    return time_images


def _validate_subspace_coord(coord: np.ndarray, n_tr: int) -> np.ndarray:
    coord = np.asarray(coord)
    assert coord.ndim == 3, f"coord must have shape (n_tr, n_samples, 2), got {coord.shape}"
    assert coord.shape[0] == n_tr, f"coord n_tr ({coord.shape[0]}) must match basis n_tr ({n_tr})"
    assert coord.shape[1] > 0, "coord must contain at least one k-space sample per TR"
    assert coord.shape[2] == 2, f"coord last dimension must be 2, got {coord.shape}"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert np.all(np.isfinite(coord)), "coord contains non-finite values"
    return coord


def _validate_subspace_kspace(kspace: np.ndarray, coord: np.ndarray) -> np.ndarray:
    kspace = np.asarray(kspace)
    expected_shape = coord.shape[:2]
    assert kspace.ndim == 2, f"kspace must have shape (n_tr, n_samples), got {kspace.shape}"
    assert kspace.shape == expected_shape, (
        f"kspace shape {kspace.shape} must match coord first two dimensions {expected_shape}"
    )
    assert np.iscomplexobj(kspace), "kspace must be complex-valued"
    assert np.all(np.isfinite(kspace)), "kspace contains non-finite values"
    return kspace


def _validate_img_shape(img_shape: Sequence[int]) -> tuple[int, int]:
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"
    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    return shape


def subspace_expand(coeff_maps: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Expand coefficient maps into a time-series image stack."""

    basis = _validate_basis(basis)
    coeff_maps = _validate_coeff_maps(coeff_maps, basis)

    time_images = np.einsum("tr,rhw->thw", basis, coeff_maps)
    expected_shape = (basis.shape[0], coeff_maps.shape[1], coeff_maps.shape[2])
    assert time_images.shape == expected_shape, (
        f"subspace_expand returned shape {time_images.shape}, expected {expected_shape}"
    )
    assert np.iscomplexobj(time_images), "subspace_expand returned a non-complex array"
    return np.asarray(time_images)


def subspace_project(time_images: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Apply the adjoint subspace projection to a time-series image stack."""

    basis = _validate_basis(basis)
    time_images = _validate_time_images(time_images, basis)

    coeff_maps = np.einsum("tr,thw->rhw", basis.conj(), time_images)
    expected_shape = (basis.shape[1], time_images.shape[1], time_images.shape[2])
    assert coeff_maps.shape == expected_shape, (
        f"subspace_project returned shape {coeff_maps.shape}, expected {expected_shape}"
    )
    assert np.iscomplexobj(coeff_maps), "subspace_project returned a non-complex array"
    return np.asarray(coeff_maps)


def subspace_nufft_forward(coeff_maps: np.ndarray, basis: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Apply subspace expansion followed by per-TR NUFFT forward operators."""

    basis = _validate_basis(basis)
    coeff_maps = _validate_coeff_maps(coeff_maps, basis)
    coord = _validate_subspace_coord(coord, basis.shape[0])

    time_images = subspace_expand(coeff_maps, basis)
    kspace = np.empty(coord.shape[:2], dtype=np.result_type(time_images.dtype, np.complex64))
    for tr_index in range(basis.shape[0]):
        kspace[tr_index] = nufft_forward(time_images[tr_index], coord[tr_index])

    expected_shape = coord.shape[:2]
    assert kspace.shape == expected_shape, (
        f"subspace_nufft_forward returned shape {kspace.shape}, expected {expected_shape}"
    )
    return kspace


def subspace_nufft_adjoint(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
) -> np.ndarray:
    """Apply per-TR NUFFT adjoints followed by adjoint subspace projection."""

    basis = _validate_basis(basis)
    coord = _validate_subspace_coord(coord, basis.shape[0])
    kspace = _validate_subspace_kspace(kspace, coord)
    shape = _validate_img_shape(img_shape)

    time_images = np.empty((basis.shape[0], *shape), dtype=np.result_type(kspace.dtype, np.complex64))
    for tr_index in range(basis.shape[0]):
        time_images[tr_index] = nufft_adjoint(kspace[tr_index], coord[tr_index], shape)

    coeff_maps = subspace_project(time_images, basis)
    expected_shape = (basis.shape[1], *shape)
    assert coeff_maps.shape == expected_shape, (
        f"subspace_nufft_adjoint returned shape {coeff_maps.shape}, expected {expected_shape}"
    )
    return coeff_maps

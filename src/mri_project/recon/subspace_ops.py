"""MRF subspace operators composed with 2D NUFFT wrappers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mri_project.array_backend import ArrayBackend, get_array_backend

from .nufft_ops import nufft_adjoint, nufft_forward


def _validate_basis(basis: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    basis = backend.to_device(basis)
    assert basis.ndim == 2, f"basis must have shape (n_tr, rank), got {basis.shape}"
    assert basis.shape[0] > 0, "basis must contain at least one TR"
    assert basis.shape[1] > 0, "basis rank must be positive"
    assert np.issubdtype(basis.dtype, np.number), "basis must be numeric"
    assert backend.all_finite(basis), "basis contains non-finite values"
    return basis


def _validate_coeff_maps(coeff_maps: np.ndarray, basis: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    coeff_maps = backend.to_device(coeff_maps)
    assert coeff_maps.ndim == 3, f"coeff_maps must have shape (rank, H, W), got {coeff_maps.shape}"
    assert coeff_maps.shape[0] == basis.shape[1], (
        f"coeff_maps rank ({coeff_maps.shape[0]}) must match basis rank ({basis.shape[1]})"
    )
    assert coeff_maps.shape[1] > 0 and coeff_maps.shape[2] > 0, (
        f"coeff_maps spatial shape must be positive, got {coeff_maps.shape[1:]}"
    )
    assert coeff_maps.dtype.kind == "c", "coeff_maps must be complex-valued"
    assert backend.all_finite(coeff_maps), "coeff_maps contains non-finite values"
    return coeff_maps


def _validate_time_images(time_images: np.ndarray, basis: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    time_images = backend.to_device(time_images)
    assert time_images.ndim == 3, f"time_images must have shape (n_tr, H, W), got {time_images.shape}"
    assert time_images.shape[0] == basis.shape[0], (
        f"time_images n_tr ({time_images.shape[0]}) must match basis n_tr ({basis.shape[0]})"
    )
    assert time_images.shape[1] > 0 and time_images.shape[2] > 0, (
        f"time_images spatial shape must be positive, got {time_images.shape[1:]}"
    )
    assert time_images.dtype.kind == "c", "time_images must be complex-valued"
    assert backend.all_finite(time_images), "time_images contains non-finite values"
    return time_images


def _validate_subspace_coord(coord: np.ndarray, n_tr: int, backend: ArrayBackend) -> np.ndarray:
    coord = backend.to_device(coord)
    assert coord.ndim == 3, f"coord must have shape (n_tr, n_samples, 2), got {coord.shape}"
    assert coord.shape[0] == n_tr, f"coord n_tr ({coord.shape[0]}) must match basis n_tr ({n_tr})"
    assert coord.shape[1] > 0, "coord must contain at least one k-space sample per TR"
    assert coord.shape[2] == 2, f"coord last dimension must be 2, got {coord.shape}"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert backend.all_finite(coord), "coord contains non-finite values"
    return coord


def _validate_subspace_kspace(kspace: np.ndarray, coord: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    kspace = backend.to_device(kspace)
    expected_shape = coord.shape[:2]
    assert kspace.ndim == 2, f"kspace must have shape (n_tr, n_samples), got {kspace.shape}"
    assert kspace.shape == expected_shape, (
        f"kspace shape {kspace.shape} must match coord first two dimensions {expected_shape}"
    )
    assert kspace.dtype.kind == "c", "kspace must be complex-valued"
    assert backend.all_finite(kspace), "kspace contains non-finite values"
    return kspace


def _validate_multicoil_kspace(kspace: np.ndarray, coord: np.ndarray, n_coils: int, backend: ArrayBackend) -> np.ndarray:
    kspace = backend.to_device(kspace)
    expected_shape = (n_coils, *coord.shape[:2])
    assert kspace.ndim == 3, f"kspace must have shape (n_coils, n_tr, n_samples), got {kspace.shape}"
    assert kspace.shape == expected_shape, f"kspace shape {kspace.shape} must match expected shape {expected_shape}"
    assert kspace.dtype.kind == "c", "kspace must be complex-valued"
    assert backend.all_finite(kspace), "kspace contains non-finite values"
    return kspace


def _validate_img_shape(img_shape: Sequence[int]) -> tuple[int, int]:
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"
    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    return shape


def _validate_sens_maps(
    sens_maps: np.ndarray,
    backend: ArrayBackend,
    img_shape: Sequence[int] | None = None,
) -> np.ndarray:
    sens_maps = backend.to_device(sens_maps)
    assert sens_maps.ndim == 3, f"sens_maps must have shape (n_coils, H, W), got {sens_maps.shape}"
    assert sens_maps.shape[0] > 0, "sens_maps must contain at least one coil"
    assert sens_maps.shape[1] > 0 and sens_maps.shape[2] > 0, (
        f"sens_maps spatial shape must be positive, got {sens_maps.shape[1:]}"
    )
    if img_shape is not None:
        shape = _validate_img_shape(img_shape)
        assert sens_maps.shape[1:] == shape, f"sens_maps spatial shape {sens_maps.shape[1:]} must match {shape}"
    assert sens_maps.dtype.kind == "c", "sens_maps must be complex-valued"
    assert backend.all_finite(sens_maps), "sens_maps contains non-finite values"
    return sens_maps


def subspace_expand(
    coeff_maps: np.ndarray,
    basis: np.ndarray,
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Expand coefficient maps into a time-series image stack."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    basis = _validate_basis(basis, backend)
    coeff_maps = _validate_coeff_maps(coeff_maps, basis, backend)

    time_images = xp.einsum("tr,rhw->thw", basis, coeff_maps)
    expected_shape = (basis.shape[0], coeff_maps.shape[1], coeff_maps.shape[2])
    assert time_images.shape == expected_shape, (
        f"subspace_expand returned shape {time_images.shape}, expected {expected_shape}"
    )
    assert time_images.dtype.kind == "c", "subspace_expand returned a non-complex array"
    return time_images


def subspace_project(
    time_images: np.ndarray,
    basis: np.ndarray,
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply the adjoint subspace projection to a time-series image stack."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    basis = _validate_basis(basis, backend)
    time_images = _validate_time_images(time_images, basis, backend)

    coeff_maps = xp.einsum("tr,thw->rhw", basis.conj(), time_images)
    expected_shape = (basis.shape[1], time_images.shape[1], time_images.shape[2])
    assert coeff_maps.shape == expected_shape, (
        f"subspace_project returned shape {coeff_maps.shape}, expected {expected_shape}"
    )
    assert coeff_maps.dtype.kind == "c", "subspace_project returned a non-complex array"
    return coeff_maps


def subspace_nufft_forward(
    coeff_maps: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply subspace expansion followed by per-TR NUFFT forward operators."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    basis = _validate_basis(basis, backend)
    coeff_maps = _validate_coeff_maps(coeff_maps, basis, backend)
    coord = _validate_subspace_coord(coord, basis.shape[0], backend)

    time_images = subspace_expand(coeff_maps, basis, device=device, device_id=device_id)
    kspace = xp.empty(coord.shape[:2], dtype=xp.result_type(time_images.dtype, xp.complex64))
    for tr_index in range(basis.shape[0]):
        kspace[tr_index] = nufft_forward(time_images[tr_index], coord[tr_index], device=device, device_id=device_id)

    expected_shape = coord.shape[:2]
    assert kspace.shape == expected_shape, (
        f"subspace_nufft_forward returned shape {kspace.shape}, expected {expected_shape}"
    )
    return kspace


def multicoil_subspace_nufft_forward(
    coeff_maps: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    sens_maps: np.ndarray,
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply the multi-coil ``P F S Phi`` subspace forward operator."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    basis = _validate_basis(basis, backend)
    coeff_maps = _validate_coeff_maps(coeff_maps, basis, backend)
    coord = _validate_subspace_coord(coord, basis.shape[0], backend)
    sens_maps = _validate_sens_maps(sens_maps, backend, coeff_maps.shape[1:])

    time_images = subspace_expand(coeff_maps, basis, device=device, device_id=device_id)
    n_coils = sens_maps.shape[0]
    kspace = xp.empty((n_coils, *coord.shape[:2]), dtype=xp.result_type(time_images.dtype, sens_maps.dtype, xp.complex64))

    for coil in range(n_coils):
        for tr_index in range(basis.shape[0]):
            coil_image = time_images[tr_index] * sens_maps[coil]
            kspace[coil, tr_index] = nufft_forward(coil_image, coord[tr_index], device=device, device_id=device_id)

    expected_shape = (n_coils, *coord.shape[:2])
    assert kspace.shape == expected_shape, (
        f"multicoil_subspace_nufft_forward returned shape {kspace.shape}, expected {expected_shape}"
    )
    return kspace


def subspace_nufft_adjoint(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply per-TR NUFFT adjoints followed by adjoint subspace projection."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    basis = _validate_basis(basis, backend)
    coord = _validate_subspace_coord(coord, basis.shape[0], backend)
    kspace = _validate_subspace_kspace(kspace, coord, backend)
    shape = _validate_img_shape(img_shape)

    time_images = xp.empty((basis.shape[0], *shape), dtype=xp.result_type(kspace.dtype, xp.complex64))
    for tr_index in range(basis.shape[0]):
        time_images[tr_index] = nufft_adjoint(kspace[tr_index], coord[tr_index], shape, device=device, device_id=device_id)

    coeff_maps = subspace_project(time_images, basis, device=device, device_id=device_id)
    expected_shape = (basis.shape[1], *shape)
    assert coeff_maps.shape == expected_shape, (
        f"subspace_nufft_adjoint returned shape {coeff_maps.shape}, expected {expected_shape}"
    )
    return coeff_maps


def multicoil_subspace_nufft_adjoint(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    sens_maps: np.ndarray,
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply the adjoint of the multi-coil ``P F S Phi`` subspace operator."""

    backend = get_array_backend(device, device_id)
    xp = backend.xp
    basis = _validate_basis(basis, backend)
    coord = _validate_subspace_coord(coord, basis.shape[0], backend)
    shape = _validate_img_shape(img_shape)
    sens_maps = _validate_sens_maps(sens_maps, backend, shape)
    kspace = _validate_multicoil_kspace(kspace, coord, sens_maps.shape[0], backend)

    time_images = xp.zeros((basis.shape[0], *shape), dtype=xp.result_type(kspace.dtype, sens_maps.dtype, xp.complex64))
    for coil in range(sens_maps.shape[0]):
        for tr_index in range(basis.shape[0]):
            coil_adjoint = nufft_adjoint(
                kspace[coil, tr_index],
                coord[tr_index],
                shape,
                device=device,
                device_id=device_id,
            )
            time_images[tr_index] += xp.conj(sens_maps[coil]) * coil_adjoint

    coeff_maps = subspace_project(time_images, basis, device=device, device_id=device_id)
    expected_shape = (basis.shape[1], *shape)
    assert coeff_maps.shape == expected_shape, (
        f"multicoil_subspace_nufft_adjoint returned shape {coeff_maps.shape}, expected {expected_shape}"
    )
    return coeff_maps

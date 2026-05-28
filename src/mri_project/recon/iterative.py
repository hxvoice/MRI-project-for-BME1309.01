"""Simple iterative reconstruction routines for subspace MRF."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .subspace_ops import (
    multicoil_subspace_nufft_adjoint,
    multicoil_subspace_nufft_forward,
    subspace_nufft_adjoint,
    subspace_nufft_forward,
)


def _validate_inputs(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    n_iter: int,
    step_size: float,
    sens_maps: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], np.ndarray | None]:
    kspace = np.asarray(kspace)
    basis = np.asarray(basis)
    coord = np.asarray(coord)
    sens_maps = None if sens_maps is None else np.asarray(sens_maps)

    assert basis.ndim == 2, f"basis must have shape (n_tr, rank), got {basis.shape}"
    assert basis.shape[0] > 0 and basis.shape[1] > 0, "basis dimensions must be positive"
    assert coord.ndim == 3 and coord.shape[-1] == 2, f"coord must have shape (n_tr, n_samples, 2), got {coord.shape}"
    assert coord.shape[0] == basis.shape[0], "coord n_tr must match basis n_tr"
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"

    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    if sens_maps is None:
        assert kspace.ndim == 2, f"kspace must have shape (n_tr, n_samples), got {kspace.shape}"
        assert kspace.shape == coord.shape[:2], "kspace shape must match coord first two dimensions"
    else:
        assert sens_maps.ndim == 3, f"sens_maps must have shape (n_coils, H, W), got {sens_maps.shape}"
        assert sens_maps.shape[0] > 0, "sens_maps must contain at least one coil"
        assert sens_maps.shape[1:] == shape, f"sens_maps spatial shape {sens_maps.shape[1:]} must match {shape}"
        expected_shape = (sens_maps.shape[0], *coord.shape[:2])
        assert kspace.ndim == 3, f"kspace must have shape (n_coils, n_tr, n_samples), got {kspace.shape}"
        assert kspace.shape == expected_shape, f"kspace shape {kspace.shape} must match expected shape {expected_shape}"
        assert np.iscomplexobj(sens_maps), "sens_maps must be complex-valued"
        assert np.all(np.isfinite(sens_maps)), "sens_maps contains non-finite values"
    assert n_iter > 0, "n_iter must be positive"
    assert step_size > 0, "step_size must be positive"
    assert np.iscomplexobj(kspace), "kspace must be complex-valued"
    assert np.issubdtype(basis.dtype, np.number), "basis must be numeric"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert np.all(np.isfinite(kspace)), "kspace contains non-finite values"
    assert np.all(np.isfinite(basis)), "basis contains non-finite values"
    assert np.all(np.isfinite(coord)), "coord contains non-finite values"

    result_dtype = np.result_type(kspace.dtype, basis.dtype, np.complex64)
    if sens_maps is not None:
        result_dtype = np.result_type(result_dtype, sens_maps.dtype)
    return (
        kspace.astype(result_dtype, copy=False),
        basis.astype(result_dtype, copy=False),
        coord.astype(np.float32, copy=False),
        shape,
        None if sens_maps is None else sens_maps.astype(result_dtype, copy=False),
    )


def reconstruct_subspace_gd(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    n_iter: int = 30,
    step_size: float = 1e-3,
    sens_maps: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Reconstruct subspace coefficient maps with plain gradient descent.

    This solves only the data-consistency term and intentionally does not apply
    LLR or any other regularization.
    """

    kspace, basis, coord, shape, sens_maps = _validate_inputs(
        kspace,
        basis,
        coord,
        img_shape,
        n_iter,
        step_size,
        sens_maps,
    )

    if sens_maps is None:
        adjoint = subspace_nufft_adjoint
        forward = subspace_nufft_forward
        coeff_maps = adjoint(kspace, basis, coord, shape)
    else:
        coeff_maps = multicoil_subspace_nufft_adjoint(kspace, basis, coord, shape, sens_maps)

    result_dtype = np.result_type(kspace.dtype, basis.dtype, np.complex64)
    if sens_maps is not None:
        result_dtype = np.result_type(result_dtype, sens_maps.dtype)
    coeff_maps = coeff_maps.astype(result_dtype, copy=False)
    losses: list[float] = []

    for _ in range(n_iter):
        if sens_maps is None:
            pred = forward(coeff_maps, basis, coord)
        else:
            pred = multicoil_subspace_nufft_forward(coeff_maps, basis, coord, sens_maps)
        residual = pred - kspace
        loss = float(0.5 * np.vdot(residual, residual).real)
        losses.append(loss)

        if sens_maps is None:
            grad = adjoint(residual, basis, coord, shape)
        else:
            grad = multicoil_subspace_nufft_adjoint(residual, basis, coord, shape, sens_maps)
        coeff_maps = coeff_maps - step_size * grad.astype(coeff_maps.dtype, copy=False)
        assert np.all(np.isfinite(coeff_maps)), "coeff_maps contains non-finite values"

    return coeff_maps, losses

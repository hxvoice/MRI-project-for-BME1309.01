"""Simple iterative reconstruction routines for subspace MRF."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .subspace_ops import subspace_nufft_adjoint, subspace_nufft_forward


def _validate_inputs(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    n_iter: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    kspace = np.asarray(kspace)
    basis = np.asarray(basis)
    coord = np.asarray(coord)

    assert basis.ndim == 2, f"basis must have shape (n_tr, rank), got {basis.shape}"
    assert basis.shape[0] > 0 and basis.shape[1] > 0, "basis dimensions must be positive"
    assert coord.ndim == 3 and coord.shape[-1] == 2, f"coord must have shape (n_tr, n_samples, 2), got {coord.shape}"
    assert coord.shape[0] == basis.shape[0], "coord n_tr must match basis n_tr"
    assert kspace.ndim == 2, f"kspace must have shape (n_tr, n_samples), got {kspace.shape}"
    assert kspace.shape == coord.shape[:2], "kspace shape must match coord first two dimensions"
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"

    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    assert n_iter > 0, "n_iter must be positive"
    assert step_size > 0, "step_size must be positive"
    assert np.iscomplexobj(kspace), "kspace must be complex-valued"
    assert np.issubdtype(basis.dtype, np.number), "basis must be numeric"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert np.all(np.isfinite(kspace)), "kspace contains non-finite values"
    assert np.all(np.isfinite(basis)), "basis contains non-finite values"
    assert np.all(np.isfinite(coord)), "coord contains non-finite values"

    result_dtype = np.result_type(kspace.dtype, basis.dtype, np.complex64)
    return (
        kspace.astype(result_dtype, copy=False),
        basis.astype(result_dtype, copy=False),
        coord.astype(np.float32, copy=False),
        shape,
    )


def reconstruct_subspace_gd(
    kspace: np.ndarray,
    basis: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    n_iter: int = 30,
    step_size: float = 1e-3,
) -> tuple[np.ndarray, list[float]]:
    """Reconstruct subspace coefficient maps with plain gradient descent.

    This solves only the data-consistency term and intentionally does not apply
    LLR or any other regularization.
    """

    kspace, basis, coord, shape = _validate_inputs(kspace, basis, coord, img_shape, n_iter, step_size)

    coeff_maps = subspace_nufft_adjoint(kspace, basis, coord, shape).astype(
        np.result_type(kspace.dtype, basis.dtype, np.complex64),
        copy=False,
    )
    losses: list[float] = []

    for _ in range(n_iter):
        pred = subspace_nufft_forward(coeff_maps, basis, coord)
        residual = pred - kspace
        loss = float(0.5 * np.vdot(residual, residual).real)
        losses.append(loss)

        grad = subspace_nufft_adjoint(residual, basis, coord, shape)
        coeff_maps = coeff_maps - step_size * grad.astype(coeff_maps.dtype, copy=False)
        assert np.all(np.isfinite(coeff_maps)), "coeff_maps contains non-finite values"

    return coeff_maps, losses

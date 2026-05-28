"""Minimal 2D NUFFT operators built on SigPy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import sigpy as sp
except ModuleNotFoundError:  # pragma: no cover - exercised only when SigPy is absent.
    sp = None


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    assert image.ndim == 2, f"image must be 2D, got shape {image.shape}"
    assert all(size > 0 for size in image.shape), f"image shape must be positive, got {image.shape}"
    assert np.iscomplexobj(image), "image must be complex-valued"
    assert np.all(np.isfinite(image)), "image contains non-finite values"
    return image


def _validate_coord(coord: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord)
    assert coord.ndim == 2, f"coord must have shape (n_samples, 2), got {coord.shape}"
    assert coord.shape[1] == 2, f"coord last dimension must be 2, got {coord.shape}"
    assert coord.shape[0] > 0, "coord must contain at least one k-space sample"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert np.all(np.isfinite(coord)), "coord contains non-finite values"
    return coord


def _validate_kspace(kspace: np.ndarray, coord: np.ndarray) -> np.ndarray:
    kspace = np.asarray(kspace)
    assert kspace.ndim == 1, f"kspace must have shape (n_samples,), got {kspace.shape}"
    assert kspace.shape[0] == coord.shape[0], (
        f"kspace samples ({kspace.shape[0]}) must match coord samples ({coord.shape[0]})"
    )
    assert np.iscomplexobj(kspace), "kspace must be complex-valued"
    assert np.all(np.isfinite(kspace)), "kspace contains non-finite values"
    return kspace


def _validate_img_shape(img_shape: Sequence[int]) -> tuple[int, int]:
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"
    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    return shape


def nufft_forward(image: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Apply the 2D NUFFT forward operator."""

    image = _validate_image(image)
    coord = _validate_coord(coord)

    if sp is None:
        kspace = _direct_nudft_forward(image, coord)
    else:
        kspace = sp.nufft(image, coord)
    assert kspace.shape == (coord.shape[0],), (
        f"NUFFT forward returned shape {kspace.shape}, expected {(coord.shape[0],)}"
    )
    return np.asarray(kspace)


def nufft_adjoint(kspace: np.ndarray, coord: np.ndarray, img_shape: Sequence[int]) -> np.ndarray:
    """Apply the 2D NUFFT adjoint operator."""

    coord = _validate_coord(coord)
    kspace = _validate_kspace(kspace, coord)
    shape = _validate_img_shape(img_shape)

    if sp is None:
        image = _direct_nudft_adjoint(kspace, coord, shape)
    else:
        image = sp.nufft_adjoint(kspace, coord, oshape=shape)
    assert image.shape == shape, f"NUFFT adjoint returned shape {image.shape}, expected {shape}"
    return np.asarray(image)


def _direct_nudft_forward(image: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Small-array direct non-uniform DFT fallback used when SigPy is unavailable."""

    height, width = image.shape
    y = np.arange(height, dtype=np.float32) - height / 2.0
    x = np.arange(width, dtype=np.float32) - width / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    points = np.stack([yy.ravel() / height, xx.ravel() / width], axis=1)
    phase = -2j * np.pi * (coord @ points.T)
    return np.exp(phase) @ image.ravel()


def _direct_nudft_adjoint(kspace: np.ndarray, coord: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Adjoint of the direct non-uniform DFT fallback."""

    height, width = shape
    y = np.arange(height, dtype=np.float32) - height / 2.0
    x = np.arange(width, dtype=np.float32) - width / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    points = np.stack([yy.ravel() / height, xx.ravel() / width], axis=1)
    phase = 2j * np.pi * (coord @ points.T)
    return (np.exp(phase).T @ kspace).reshape(shape)

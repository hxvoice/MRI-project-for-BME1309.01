"""Minimal 2D NUFFT operators built on SigPy.

SigPy is imported lazily because its Numba-backed import path can be slow when
Numba has to probe an unsuitable cache location. Before importing SigPy, this
module sets a writable default ``NUMBA_CACHE_DIR`` when the user has not already
provided one.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import tempfile
from types import ModuleType

import numpy as np

from mri_project.array_backend import ArrayBackend, get_array_backend

_SIGPY: ModuleType | None = None
_SIGPY_IMPORT_ATTEMPTED = False


def _ensure_numba_cache_dir() -> None:
    """Give Numba a stable writable cache directory before SigPy registers JIT kernels."""

    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    cache_dir = Path(tempfile.gettempdir()) / "mri_project_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


def _get_sigpy() -> ModuleType | None:
    """Import SigPy only when a NUFFT operator is actually executed."""

    global _SIGPY, _SIGPY_IMPORT_ATTEMPTED

    if not _SIGPY_IMPORT_ATTEMPTED:
        _SIGPY_IMPORT_ATTEMPTED = True
        _ensure_numba_cache_dir()
        try:
            import sigpy as sp
        except ModuleNotFoundError:  # pragma: no cover - exercised only when SigPy is absent.
            _SIGPY = None
        else:
            _SIGPY = sp

    return _SIGPY


def _validate_image(image: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    image = backend.to_device(image)
    assert image.ndim == 2, f"image must be 2D, got shape {image.shape}"
    assert all(size > 0 for size in image.shape), f"image shape must be positive, got {image.shape}"
    assert image.dtype.kind == "c", "image must be complex-valued"
    assert backend.all_finite(image), "image contains non-finite values"
    return image


def _validate_coord(coord: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    coord = backend.to_device(coord)
    assert coord.ndim == 2, f"coord must have shape (n_samples, 2), got {coord.shape}"
    assert coord.shape[1] == 2, f"coord last dimension must be 2, got {coord.shape}"
    assert coord.shape[0] > 0, "coord must contain at least one k-space sample"
    assert np.issubdtype(coord.dtype, np.floating), "coord must be floating-point"
    assert backend.all_finite(coord), "coord contains non-finite values"
    return coord


def _validate_kspace(kspace: np.ndarray, coord: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    kspace = backend.to_device(kspace)
    assert kspace.ndim == 1, f"kspace must have shape (n_samples,), got {kspace.shape}"
    assert kspace.shape[0] == coord.shape[0], (
        f"kspace samples ({kspace.shape[0]}) must match coord samples ({coord.shape[0]})"
    )
    assert kspace.dtype.kind == "c", "kspace must be complex-valued"
    assert backend.all_finite(kspace), "kspace contains non-finite values"
    return kspace


def _validate_img_shape(img_shape: Sequence[int]) -> tuple[int, int]:
    assert len(img_shape) == 2, f"img_shape must have length 2, got {img_shape}"
    shape = tuple(int(size) for size in img_shape)
    assert all(size > 0 for size in shape), f"img_shape must be positive, got {img_shape}"
    return shape


def _run_sigpy_nufft_forward(sp: ModuleType, image: np.ndarray, coord: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    if not backend.is_cuda:
        return sp.nufft(image, coord)
    with sp.Device(backend.device_id):
        return sp.nufft(image, coord)


def _run_sigpy_nufft_adjoint(
    sp: ModuleType,
    kspace: np.ndarray,
    coord: np.ndarray,
    shape: tuple[int, int],
    backend: ArrayBackend,
) -> np.ndarray:
    if not backend.is_cuda:
        return sp.nufft_adjoint(kspace, coord, oshape=shape)
    with sp.Device(backend.device_id):
        return sp.nufft_adjoint(kspace, coord, oshape=shape)


def nufft_forward(image: np.ndarray, coord: np.ndarray, device: str = "cpu", device_id: int = 0) -> np.ndarray:
    """Apply the 2D NUFFT forward operator."""

    backend = get_array_backend(device, device_id)
    image = _validate_image(image, backend)
    coord = _validate_coord(coord, backend)
    sp = _get_sigpy()

    if sp is None:
        if backend.is_cuda:
            raise RuntimeError("CUDA NUFFT requires SigPy. Install sigpy together with CuPy to use --device cuda.")
        kspace = _direct_nudft_forward(image, coord, backend)
    else:
        kspace = _run_sigpy_nufft_forward(sp, image, coord, backend)
    assert kspace.shape == (coord.shape[0],), (
        f"NUFFT forward returned shape {kspace.shape}, expected {(coord.shape[0],)}"
    )
    return kspace if backend.is_cuda else np.asarray(kspace)


def nufft_adjoint(
    kspace: np.ndarray,
    coord: np.ndarray,
    img_shape: Sequence[int],
    device: str = "cpu",
    device_id: int = 0,
) -> np.ndarray:
    """Apply the 2D NUFFT adjoint operator."""

    backend = get_array_backend(device, device_id)
    coord = _validate_coord(coord, backend)
    kspace = _validate_kspace(kspace, coord, backend)
    shape = _validate_img_shape(img_shape)
    sp = _get_sigpy()

    if sp is None:
        if backend.is_cuda:
            raise RuntimeError("CUDA NUFFT requires SigPy. Install sigpy together with CuPy to use --device cuda.")
        image = _direct_nudft_adjoint(kspace, coord, shape, backend)
    else:
        image = _run_sigpy_nufft_adjoint(sp, kspace, coord, shape, backend)
    assert image.shape == shape, f"NUFFT adjoint returned shape {image.shape}, expected {shape}"
    return image if backend.is_cuda else np.asarray(image)


def _direct_nudft_forward(image: np.ndarray, coord: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    """Small-array direct non-uniform DFT fallback used when SigPy is unavailable."""

    xp = backend.xp
    height, width = image.shape
    y = xp.arange(height, dtype=xp.float32) - height / 2.0
    x = xp.arange(width, dtype=xp.float32) - width / 2.0
    yy, xx = xp.meshgrid(y, x, indexing="ij")
    points = xp.stack([yy.ravel() / height, xx.ravel() / width], axis=1)
    phase = -2j * xp.pi * (coord @ points.T)
    return xp.exp(phase) @ image.ravel()


def _direct_nudft_adjoint(
    kspace: np.ndarray,
    coord: np.ndarray,
    shape: tuple[int, int],
    backend: ArrayBackend,
) -> np.ndarray:
    """Adjoint of the direct non-uniform DFT fallback."""

    xp = backend.xp
    height, width = shape
    y = xp.arange(height, dtype=xp.float32) - height / 2.0
    x = xp.arange(width, dtype=xp.float32) - width / 2.0
    yy, xx = xp.meshgrid(y, x, indexing="ij")
    points = xp.stack([yy.ravel() / height, xx.ravel() / width], axis=1)
    phase = 2j * xp.pi * (coord @ points.T)
    return (xp.exp(phase).T @ kspace).reshape(shape)

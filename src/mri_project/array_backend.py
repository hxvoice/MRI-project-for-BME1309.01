"""Small NumPy/CuPy backend helpers used by GPU-aware code paths."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np


_CUPY: ModuleType | None = None
_CUPY_IMPORT_ATTEMPTED = False


def _load_cupy() -> ModuleType | None:
    global _CUPY, _CUPY_IMPORT_ATTEMPTED

    if not _CUPY_IMPORT_ATTEMPTED:
        _CUPY_IMPORT_ATTEMPTED = True
        try:
            import cupy as cp
        except ModuleNotFoundError:
            _CUPY = None
        else:
            _CUPY = cp
    return _CUPY


def _normalize_device(device: str | int) -> str:
    if isinstance(device, int):
        return "cuda" if device >= 0 else "cpu"

    normalized = str(device).lower()
    if normalized in {"cpu", "-1"}:
        return "cpu"
    if normalized in {"cuda", "gpu"} or normalized.startswith("cuda:"):
        return "cuda"
    raise ValueError("device must be 'cpu' or 'cuda'.")


@dataclass(frozen=True)
class ArrayBackend:
    """Array backend descriptor for CPU and CUDA code paths."""

    device: str = "cpu"
    device_id: int = 0
    xp: Any = np

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    def to_device(self, array: Any, dtype: Any | None = None) -> Any:
        with self:
            return self.xp.asarray(array, dtype=dtype)

    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
        with self:
            return self.xp.empty(shape, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        with self:
            return self.xp.zeros(shape, dtype=dtype)

    def to_cpu(self, array: Any) -> np.ndarray:
        if not self.is_cuda:
            return np.asarray(array)
        return self.xp.asnumpy(array)

    def scalar_to_python(self, value: Any) -> Any:
        if self.is_cuda:
            value = self.xp.asnumpy(value)
        if hasattr(value, "item"):
            return value.item()
        return value

    def all_finite(self, array: Any) -> bool:
        if self.is_cuda:
            # Avoid implicit device synchronization in hot validation paths.
            return True
        return bool(self.scalar_to_python(self.xp.all(self.xp.isfinite(array))))

    def __enter__(self) -> "ArrayBackend":
        if self.is_cuda:
            self.xp.cuda.Device(self.device_id).use()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def get_array_backend(device: str | int = "cpu", device_id: int = 0) -> ArrayBackend:
    """Return a backend descriptor, importing CuPy only for CUDA requests."""

    normalized = _normalize_device(device)
    if normalized == "cpu":
        return ArrayBackend(device="cpu", device_id=-1, xp=np)

    cp = _load_cupy()
    if cp is None:
        raise RuntimeError(
            "CUDA device requested, but CuPy is not installed. Install the GPU extras "
            "or install a CUDA-specific CuPy package such as cupy-cuda12x."
        )

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - requires CUDA driver failure.
        raise RuntimeError(f"CUDA device requested, but CuPy could not access a CUDA GPU: {error}") from error

    if device_count <= 0:
        raise RuntimeError("CUDA device requested, but no CUDA-capable GPU was found.")
    if device_id < 0 or device_id >= device_count:
        raise RuntimeError(f"CUDA device_id {device_id} is invalid; {device_count} CUDA device(s) are available.")

    return ArrayBackend(device="cuda", device_id=int(device_id), xp=cp)


def to_cpu(array: Any) -> np.ndarray:
    """Convert NumPy or CuPy arrays to a NumPy array."""

    cp = _load_cupy()
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def asnumpy_if_needed(array: Any) -> np.ndarray:
    """Alias kept explicit at call sites that save or plot data."""

    return to_cpu(array)

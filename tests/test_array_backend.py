import importlib.util

import numpy as np
import pytest

from mri_project.array_backend import get_array_backend
from mri_project.quantification import execute_template_matching
from mri_project.recon.regularization import llr_soft_threshold


def get_cuda_backend_or_skip():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    try:
        return get_array_backend("cuda", 0)
    except RuntimeError as error:
        pytest.skip(f"CUDA backend is not available: {error}")


def test_cpu_backend_is_available_without_cupy():
    backend = get_array_backend("cpu")
    array = backend.to_device([1.0, 2.0, 3.0])

    assert backend.device == "cpu"
    assert backend.xp is np
    np.testing.assert_array_equal(backend.to_cpu(array), [1.0, 2.0, 3.0])


def test_cuda_backend_reports_clear_error_when_unavailable():
    try:
        backend = get_array_backend("cuda", 0)
    except RuntimeError as error:
        message = str(error)
        assert "CUDA" in message or "CuPy" in message
    else:
        assert backend.device == "cuda"


def test_template_matching_gpu_matches_cpu_when_available():
    get_cuda_backend_or_skip()

    t1_grid = np.array([850.0, 1350.0])
    t2_grid = np.array([60.0, 80.0])
    dict_compressed = np.array(
        [
            [[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]],
            [[1.0 + 0j, 1.0 + 0j], [0.5 + 0j, 1.5 + 0j]],
        ],
        dtype=np.complex64,
    )
    coeff_maps = np.array(
        [
            [[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]],
            [[1.0 + 0j, 1.0 + 0j], [0.5 + 0j, 1.5 + 0j]],
        ],
        dtype=np.complex64,
    )

    cpu_result = execute_template_matching(coeff_maps, dict_compressed, t1_grid, t2_grid, batch_size=2)
    gpu_result = execute_template_matching(
        coeff_maps,
        dict_compressed,
        t1_grid,
        t2_grid,
        device="cuda",
        batch_size=2,
    )

    for cpu_array, gpu_array in zip(cpu_result, gpu_result, strict=True):
        np.testing.assert_allclose(gpu_array, cpu_array)


def test_llr_gpu_matches_cpu_when_available():
    get_cuda_backend_or_skip()

    rng = np.random.default_rng(123)
    coeff_maps = 0.1 * (rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4)))
    coeff_maps = coeff_maps.astype(np.complex64)

    cpu_result = llr_soft_threshold(coeff_maps, patch_shape=(2, 2), threshold=0.01)
    gpu_result = llr_soft_threshold(coeff_maps, patch_shape=(2, 2), threshold=0.01, device="cuda")

    np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-6)

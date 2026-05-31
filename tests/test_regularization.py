import numpy as np
import pytest

from mri_project.recon.regularization import llr_nuclear_norm, llr_soft_threshold


def make_coeff_maps(rank=3, shape=(9, 10), seed=50):
    rng = np.random.default_rng(seed)
    coeff_maps = rng.normal(size=(rank, *shape)) + 1j * rng.normal(size=(rank, *shape))
    return (0.1 * coeff_maps).astype(np.complex64)


def test_llr_soft_threshold_preserves_shape_dtype_and_finiteness():
    coeff_maps = make_coeff_maps()

    denoised = llr_soft_threshold(coeff_maps, patch_shape=(4, 4), threshold=0.01)

    assert denoised.shape == coeff_maps.shape
    assert denoised.dtype == coeff_maps.dtype
    assert np.iscomplexobj(denoised)
    assert np.all(np.isfinite(denoised))


def test_llr_soft_threshold_zero_threshold_returns_copy():
    coeff_maps = make_coeff_maps()

    denoised = llr_soft_threshold(coeff_maps, patch_shape=(4, 4), threshold=0.0)

    assert denoised is not coeff_maps
    np.testing.assert_allclose(denoised, coeff_maps)


def test_llr_soft_threshold_reduces_nuclear_norm():
    coeff_maps = make_coeff_maps()
    patch_shape = (4, 5)

    before = llr_nuclear_norm(coeff_maps, patch_shape=patch_shape)
    denoised = llr_soft_threshold(coeff_maps, patch_shape=patch_shape, threshold=0.03)
    after = llr_nuclear_norm(denoised, patch_shape=patch_shape)

    assert after < before


@pytest.mark.parametrize("patch_shape", [(0, 4), (4, 0), (-1, 4), (4, -1)])
def test_llr_rejects_non_positive_patch_shape(patch_shape):
    coeff_maps = make_coeff_maps()

    with pytest.raises(AssertionError):
        llr_soft_threshold(coeff_maps, patch_shape=patch_shape, threshold=0.01)


def test_llr_rejects_negative_threshold():
    coeff_maps = make_coeff_maps()

    with pytest.raises(AssertionError):
        llr_soft_threshold(coeff_maps, patch_shape=(4, 4), threshold=-0.01)

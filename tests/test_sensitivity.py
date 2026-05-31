import sys
import types

import numpy as np

from mri_project.recon.sensitivity import (
    estimate_sens_maps_espirit,
    grid_center_calibration_kspace,
    time_average_kspace,
)


def make_kspace(n_coils=3, n_tr=4, n_samples=12, seed=90):
    rng = np.random.default_rng(seed)
    kspace = rng.normal(size=(n_coils, n_tr, n_samples)) + 1j * rng.normal(size=(n_coils, n_tr, n_samples))
    return kspace.astype(np.complex64)


def make_coord(n_tr=4, n_samples=12):
    coord = np.zeros((n_tr, n_samples, 2), dtype=np.float32)
    radius = np.linspace(0.0, 5.0, n_samples, dtype=np.float32)
    for tr_index in range(n_tr):
        angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False, dtype=np.float32) + tr_index
        coord[tr_index, :, 0] = radius * np.sin(angles)
        coord[tr_index, :, 1] = radius * np.cos(angles)
    return coord


def install_fake_sigpy(monkeypatch, expected_shape):
    fake_sigpy = types.ModuleType("sigpy")
    fake_mri = types.ModuleType("sigpy.mri")
    fake_app = types.ModuleType("sigpy.mri.app")

    class FakeEspiritCalib:
        def __init__(self, ksp, **kwargs):
            self.ksp = np.asarray(ksp)
            self.kwargs = kwargs

        def run(self):
            assert self.ksp.shape == expected_shape
            n_coils, height, width = self.ksp.shape
            return np.ones((n_coils, height, width), dtype=np.complex64)

    fake_app.EspiritCalib = FakeEspiritCalib
    fake_mri.app = fake_app
    fake_sigpy.mri = fake_mri
    monkeypatch.setitem(sys.modules, "sigpy", fake_sigpy)
    monkeypatch.setitem(sys.modules, "sigpy.mri", fake_mri)
    monkeypatch.setitem(sys.modules, "sigpy.mri.app", fake_app)


def test_time_average_kspace_shape_and_values():
    kspace = make_kspace()

    averaged = time_average_kspace(kspace)

    assert averaged.shape == (kspace.shape[0], kspace.shape[2])
    assert np.iscomplexobj(averaged)
    assert np.all(np.isfinite(averaged))
    np.testing.assert_allclose(averaged, np.mean(kspace, axis=1))


def test_grid_center_calibration_kspace_shape_and_finiteness():
    kspace = make_kspace()
    coord = make_coord(n_tr=kspace.shape[1], n_samples=kspace.shape[2])
    averaged = time_average_kspace(kspace)

    calib_kspace = grid_center_calibration_kspace(averaged, coord, img_shape=(16, 16), center_width=8)

    assert calib_kspace.shape == (kspace.shape[0], 16, 16)
    assert np.iscomplexobj(calib_kspace)
    assert np.all(np.isfinite(calib_kspace))
    assert np.count_nonzero(np.abs(calib_kspace)) > 0


def test_estimate_sens_maps_espirit_calls_sigpy_and_returns_maps(monkeypatch):
    kspace = make_kspace(n_coils=2, n_tr=3, n_samples=10)
    coord = make_coord(n_tr=kspace.shape[1], n_samples=kspace.shape[2])
    install_fake_sigpy(monkeypatch, expected_shape=(2, 12, 12))

    sens_maps = estimate_sens_maps_espirit(
        kspace,
        coord,
        img_shape=(12, 12),
        center_width=8,
        calib_width=8,
        max_iter=2,
    )

    assert sens_maps.shape == (2, 12, 12)
    assert np.iscomplexobj(sens_maps)
    assert np.all(np.isfinite(sens_maps))

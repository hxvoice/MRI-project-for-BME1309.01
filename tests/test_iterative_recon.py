import numpy as np

from mri_project.recon.iterative import reconstruct_subspace_gd, reconstruct_subspace_llr
from mri_project.recon.subspace_ops import multicoil_subspace_nufft_forward, subspace_nufft_forward


def make_complex_basis(n_tr=3, rank=2, seed=20):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(n_tr, rank)) + 1j * rng.normal(size=(n_tr, rank))
    basis, _ = np.linalg.qr(basis)
    return basis.astype(np.complex64)


def make_coeff_maps(rank=2, shape=(8, 8), seed=21):
    rng = np.random.default_rng(seed)
    coeff_maps = rng.normal(size=(rank, *shape)) + 1j * rng.normal(size=(rank, *shape))
    return (0.1 * coeff_maps).astype(np.complex64)


def make_coord(n_tr=3, n_samples=16):
    coord = np.zeros((n_tr, n_samples, 2), dtype=np.float32)
    for tr_index in range(n_tr):
        angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False, dtype=np.float32)
        radius = np.linspace(0.0, 2.5, n_samples, dtype=np.float32)
        coord[tr_index, :, 0] = radius * np.sin(angles + tr_index)
        coord[tr_index, :, 1] = radius * np.cos(angles + tr_index)
    return coord


def make_sens_maps(n_coils=2, shape=(8, 8)):
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    sens_maps = np.empty((n_coils, *shape), dtype=np.complex64)
    for coil in range(n_coils):
        angle = 2.0 * np.pi * coil / n_coils
        magnitude = 0.8 + 0.2 * np.exp(-((xx - 0.4 * np.cos(angle)) ** 2 + (yy - 0.4 * np.sin(angle)) ** 2))
        phase = np.exp(1j * 0.15 * (xx * np.cos(angle) + yy * np.sin(angle)))
        sens_maps[coil] = magnitude * phase
    return sens_maps


def test_reconstruct_subspace_gd_reduces_loss():
    shape = (8, 8)
    basis = make_complex_basis()
    coeff_true = make_coeff_maps(shape=shape)
    coord = make_coord(n_tr=basis.shape[0])
    kspace = subspace_nufft_forward(coeff_true, basis, coord).astype(np.complex64)

    coeff_maps, losses = reconstruct_subspace_gd(
        kspace,
        basis,
        coord,
        shape,
        n_iter=8,
        step_size=1e-5,
    )

    assert coeff_maps.shape == (basis.shape[1], *shape)
    assert np.iscomplexobj(coeff_maps)
    assert len(losses) == 8
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_reconstruct_subspace_gd_reduces_multicoil_loss():
    shape = (8, 8)
    basis = make_complex_basis(seed=40)
    coeff_true = make_coeff_maps(shape=shape, seed=41)
    coord = make_coord(n_tr=basis.shape[0])
    sens_maps = make_sens_maps(shape=shape)
    kspace = multicoil_subspace_nufft_forward(coeff_true, basis, coord, sens_maps).astype(np.complex64)

    coeff_maps, losses = reconstruct_subspace_gd(
        kspace,
        basis,
        coord,
        shape,
        n_iter=8,
        step_size=5e-6,
        sens_maps=sens_maps,
    )

    assert coeff_maps.shape == (basis.shape[1], *shape)
    assert np.iscomplexobj(coeff_maps)
    assert len(losses) == 8
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_reconstruct_subspace_llr_reduces_loss():
    shape = (8, 8)
    basis = make_complex_basis(seed=60)
    coeff_true = make_coeff_maps(shape=shape, seed=61)
    coord = make_coord(n_tr=basis.shape[0])
    kspace = subspace_nufft_forward(coeff_true, basis, coord).astype(np.complex64)

    coeff_maps, losses = reconstruct_subspace_llr(
        kspace,
        basis,
        coord,
        shape,
        n_iter=8,
        step_size=1e-5,
        lambda_llr=1e-4,
        patch_shape=(4, 4),
    )

    assert coeff_maps.shape == (basis.shape[1], *shape)
    assert np.iscomplexobj(coeff_maps)
    assert len(losses) == 8
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_reconstruct_subspace_llr_reduces_multicoil_loss():
    shape = (8, 8)
    basis = make_complex_basis(seed=70)
    coeff_true = make_coeff_maps(shape=shape, seed=71)
    coord = make_coord(n_tr=basis.shape[0])
    sens_maps = make_sens_maps(shape=shape)
    kspace = multicoil_subspace_nufft_forward(coeff_true, basis, coord, sens_maps).astype(np.complex64)

    coeff_maps, losses = reconstruct_subspace_llr(
        kspace,
        basis,
        coord,
        shape,
        n_iter=8,
        step_size=5e-6,
        lambda_llr=1e-4,
        patch_shape=(4, 4),
        sens_maps=sens_maps,
    )

    assert coeff_maps.shape == (basis.shape[1], *shape)
    assert np.iscomplexobj(coeff_maps)
    assert len(losses) == 8
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_reconstruct_subspace_llr_runs_without_regularization():
    shape = (8, 8)
    basis = make_complex_basis(seed=80)
    coeff_true = make_coeff_maps(shape=shape, seed=81)
    coord = make_coord(n_tr=basis.shape[0])
    kspace = subspace_nufft_forward(coeff_true, basis, coord).astype(np.complex64)

    coeff_maps, losses = reconstruct_subspace_llr(
        kspace,
        basis,
        coord,
        shape,
        n_iter=4,
        step_size=1e-5,
        lambda_llr=0.0,
        patch_shape=(4, 4),
    )

    assert coeff_maps.shape == (basis.shape[1], *shape)
    assert np.iscomplexobj(coeff_maps)
    assert len(losses) == 4
    assert np.all(np.isfinite(losses))

import numpy as np

from mri_project.recon.iterative import reconstruct_subspace_gd
from mri_project.recon.subspace_ops import subspace_nufft_forward


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

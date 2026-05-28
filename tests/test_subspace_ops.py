import numpy as np

from mri_project.recon import (
    subspace_expand,
    subspace_nufft_adjoint,
    subspace_nufft_forward,
    subspace_project,
)


def make_complex_basis(n_tr=6, rank=3, seed=10):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(n_tr, rank)) + 1j * rng.normal(size=(n_tr, rank))
    basis, _ = np.linalg.qr(basis)
    return basis.astype(np.complex64)


def make_coeff_maps(rank=3, shape=(24, 24), seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    coeff_maps = np.empty((rank, *shape), dtype=np.complex64)

    for index in range(rank):
        cy, cx = rng.uniform(-0.5, 0.5, size=2)
        sy, sx = rng.uniform(0.12, 0.35, size=2)
        magnitude = np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2.0)
        phase = rng.uniform(-np.pi, np.pi, size=shape).astype(np.float32)
        coeff_maps[index] = (magnitude * np.exp(1j * 0.1 * phase)).astype(np.complex64)

    return coeff_maps


def make_subspace_coord(n_tr=6, shape=(24, 24), n_samples=80):
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    max_radius = 0.45 * min(shape)
    coord = np.empty((n_tr, n_samples, 2), dtype=np.float32)

    for tr_index in range(n_tr):
        base_angle = 2.0 * np.pi * tr_index / n_tr
        angle = base_angle + 2.0 * np.pi * 3.5 * t
        radius = max_radius * t
        coord[tr_index, :, 0] = radius * np.sin(angle)
        coord[tr_index, :, 1] = radius * np.cos(angle)

    return coord


def test_subspace_ops_shapes_and_adjointness():
    n_tr = 6
    rank = 3
    shape = (24, 24)
    n_samples = 80

    basis = make_complex_basis(n_tr=n_tr, rank=rank)
    coeff_maps = make_coeff_maps(rank=rank, shape=shape)
    coord = make_subspace_coord(n_tr=n_tr, shape=shape, n_samples=n_samples)

    time_images = subspace_expand(coeff_maps, basis)
    assert time_images.shape == (n_tr, *shape)
    assert np.iscomplexobj(time_images)
    assert np.all(np.isfinite(time_images))

    projected = subspace_project(time_images, basis)
    assert projected.shape == (rank, *shape)
    assert np.iscomplexobj(projected)
    assert np.all(np.isfinite(projected))

    kspace = subspace_nufft_forward(coeff_maps, basis, coord)
    assert kspace.shape == (n_tr, n_samples)
    assert np.iscomplexobj(kspace)
    assert np.all(np.isfinite(kspace))

    adjoint_coeff_maps = subspace_nufft_adjoint(kspace, basis, coord, shape)
    assert adjoint_coeff_maps.shape == (rank, *shape)
    assert np.iscomplexobj(adjoint_coeff_maps)
    assert np.all(np.isfinite(adjoint_coeff_maps))

    rng = np.random.default_rng(13)
    y = (rng.normal(size=(n_tr, n_samples)) + 1j * rng.normal(size=(n_tr, n_samples))).astype(np.complex64)
    lhs = np.vdot(subspace_nufft_forward(coeff_maps, basis, coord), y)
    rhs = np.vdot(coeff_maps, subspace_nufft_adjoint(y, basis, coord, shape))

    denom = max(abs(lhs), abs(rhs), 1.0)
    assert abs(lhs - rhs) / denom < 1e-3

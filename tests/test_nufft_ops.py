import numpy as np

from mri_project.recon.nufft_ops import nufft_adjoint, nufft_forward


def make_random_phantom(shape=(32, 32), n_blobs=5, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    image = np.zeros(shape, dtype=np.float32)

    for _ in range(n_blobs):
        cy, cx = rng.uniform(-0.65, 0.65, size=2)
        sy, sx = rng.uniform(0.06, 0.22, size=2)
        amp = rng.uniform(0.4, 1.0)
        image += amp * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2.0)

    image /= image.max()
    phase = rng.uniform(-np.pi, np.pi, size=shape).astype(np.float32)
    return (image * np.exp(1j * 0.05 * phase)).astype(np.complex64)


def make_spiral_like_coord(shape=(32, 32), n_interleaves=4, samples_per_interleaf=96, seed=1):
    rng = np.random.default_rng(seed)
    n_samples = n_interleaves * samples_per_interleaf
    t = np.linspace(0.0, 1.0, samples_per_interleaf, dtype=np.float32)
    max_radius = 0.45 * min(shape)

    coord = np.empty((n_samples, 2), dtype=np.float32)
    row = 0
    for arm in range(n_interleaves):
        base_angle = 2.0 * np.pi * arm / n_interleaves
        turns = 4.0 + 0.4 * rng.random()
        angle = base_angle + 2.0 * np.pi * turns * t
        radius = max_radius * t
        jitter = rng.normal(scale=0.015 * max_radius, size=(samples_per_interleaf, 2))

        coord[row : row + samples_per_interleaf, 0] = radius * np.sin(angle) + jitter[:, 0]
        coord[row : row + samples_per_interleaf, 1] = radius * np.cos(angle) + jitter[:, 1]
        row += samples_per_interleaf

    return coord


def test_nufft_forward_adjoint_shapes_and_adjointness():
    shape = (32, 32)
    image = make_random_phantom(shape=shape, seed=42)
    coord = make_spiral_like_coord(shape=shape, seed=43)

    kspace = nufft_forward(image, coord)
    assert kspace.shape == (coord.shape[0],)
    assert np.iscomplexobj(kspace)
    assert np.all(np.isfinite(kspace))

    recon = nufft_adjoint(kspace, coord, shape)
    assert recon.shape == shape
    assert np.iscomplexobj(recon)
    assert np.all(np.isfinite(recon))

    rng = np.random.default_rng(44)
    y = (rng.normal(size=coord.shape[0]) + 1j * rng.normal(size=coord.shape[0])).astype(np.complex64)
    lhs = np.vdot(nufft_forward(image, coord), y)
    rhs = np.vdot(image, nufft_adjoint(y, coord, shape))

    denom = max(abs(lhs), abs(rhs), 1.0)
    assert abs(lhs - rhs) / denom < 1e-3

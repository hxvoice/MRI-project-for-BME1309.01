"""MRF FISP EPG simulation and dictionary compression."""

from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import svd


class MRFEPGSimulator:
    """2D MRF FISP extended phase graph simulator."""

    def __init__(self, num_states: int = 200) -> None:
        assert num_states > 0, "num_states must be positive"
        self.num_states = num_states
        self.reset()

    def reset(self) -> None:
        self.f = np.zeros((2, self.num_states), dtype=np.complex128)
        self.z = np.zeros(self.num_states, dtype=np.complex128)
        self.z[0] = 1.0

    def apply_rf(self, alpha_deg: float, phi_deg: float = 0.0) -> None:
        alpha = np.deg2rad(alpha_deg)
        phi = np.deg2rad(phi_deg)

        ca2 = np.cos(alpha / 2.0) ** 2
        sa2 = np.sin(alpha / 2.0) ** 2
        sina = np.sin(alpha)

        ep = np.exp(1j * phi)
        ep2 = np.exp(1j * 2.0 * phi)
        em = np.exp(-1j * phi)
        em2 = np.exp(-1j * 2.0 * phi)

        f_plus = self.f[0, :]
        f_minus = self.f[1, :]
        z_state = self.z[:]

        self.f[0, :] = ca2 * f_plus + ep2 * sa2 * f_minus - 1j * ep * sina * z_state
        self.f[1, :] = em2 * sa2 * f_plus + ca2 * f_minus + 1j * em * sina * z_state
        self.z[:] = -0.5j * em * sina * f_plus + 0.5j * ep * sina * f_minus + np.cos(alpha) * z_state

    def apply_relaxation(self, duration: float, t1: float, t2: float) -> None:
        assert t1 > 0 and t2 > 0, "T1 and T2 must be positive"
        e1 = np.exp(-duration / t1)
        e2 = np.exp(-duration / t2)
        self.f *= e2
        self.z *= e1
        self.z[0] += 1.0 - e1

    def apply_shift(self) -> None:
        self.f[0, 1:] = self.f[0, 0:-1]
        self.f[0, 0] = np.conj(self.f[1, 0])
        self.f[1, 0:-1] = self.f[1, 1:]
        self.f[1, -1] = 0.0

    def simulate_mrf_fisp(
        self,
        t1: float | None = None,
        t2: float | None = None,
        fa_train: np.ndarray | None = None,
        tr: float = 12.0,
        te: float = 0.7,
        ti: float = 20.0,
        **kwargs: float,
    ) -> np.ndarray:
        """Simulate one tissue signal evolution for a flip-angle train."""

        t1 = kwargs.pop("T1", t1)
        t2 = kwargs.pop("T2", t2)
        tr = kwargs.pop("TR", tr)
        te = kwargs.pop("TE", te)
        ti = kwargs.pop("TI", ti)
        assert not kwargs, f"unexpected keyword arguments: {sorted(kwargs)}"
        assert t1 is not None and t2 is not None, "T1 and T2 must be provided"
        assert fa_train is not None, "fa_train must be provided"

        fa_train = np.asarray(fa_train)
        assert fa_train.ndim == 1 and fa_train.size > 0, "fa_train must be a non-empty 1D array"
        assert tr > te >= 0, "TR must be greater than TE"

        self.reset()
        signal = np.zeros(len(fa_train), dtype=np.complex128)

        self.apply_rf(180.0, 0.0)
        self.apply_relaxation(ti, t1, t2)

        for index, fa in enumerate(fa_train):
            self.apply_rf(float(fa), 0.0)
            self.apply_relaxation(te, t1, t2)
            signal[index] = self.f[0, 0]
            self.apply_relaxation(tr - te, t1, t2)
            self.apply_shift()

        return signal


def generate_mrf_fa_train_spline(
    num_trs: int = 500,
    min_fa: float = 10.0,
    max_fa: float = 90.0,
    num_anchors: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """Generate a smooth pseudo-random flip-angle train."""

    assert num_trs > 1, "num_trs must be greater than 1"
    assert num_anchors > 1, "num_anchors must be greater than 1"
    rng = np.random.default_rng(seed)

    x_anchors = np.linspace(0, num_trs - 1, num_anchors)
    y_anchors = rng.uniform(0.0, 1.0, num_anchors)
    spline = CubicSpline(x_anchors, y_anchors, bc_type="natural")
    curve = spline(np.arange(num_trs))
    curve = (curve - curve.min()) / (curve.max() - curve.min())
    return (min_fa + curve * (max_fa - min_fa)).astype(np.float32)


def default_t1_grid() -> np.ndarray:
    return np.concatenate([np.arange(20, 3001, 20), np.arange(3200, 5001, 200)])


def default_t2_grid() -> np.ndarray:
    return np.concatenate(
        [
            np.arange(10, 201, 2),
            np.arange(220, 1001, 20),
            np.arange(1050, 2001, 50),
            np.arange(2100, 4001, 100),
        ]
    )


def build_signal_dictionary(
    simulator: MRFEPGSimulator,
    fa_train: np.ndarray,
    t1_grid: np.ndarray | None = None,
    t2_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the full MRF signal dictionary over T1/T2 grids."""

    start_time = time.time()
    t1_values = default_t1_grid() if t1_grid is None else np.asarray(t1_grid)
    t2_values = default_t2_grid() if t2_grid is None else np.asarray(t2_grid)
    fa_train = np.asarray(fa_train)

    assert fa_train.ndim == 1 and fa_train.size > 0, "fa_train must be a non-empty 1D array"
    assert t1_values.ndim == 1 and t1_values.size > 0, "t1_grid must be a non-empty 1D array"
    assert t2_values.ndim == 1 and t2_values.size > 0, "t2_grid must be a non-empty 1D array"

    dictionary = np.zeros((len(t1_values), len(t2_values), len(fa_train)), dtype=np.complex128)
    valid_count = 0
    for t1_index, t1 in enumerate(t1_values):
        for t2_index, t2 in enumerate(t2_values):
            if t2 <= t1:
                dictionary[t1_index, t2_index] = simulator.simulate_mrf_fisp(t1=t1, t2=t2, fa_train=fa_train)
                valid_count += 1

    elapsed = time.time() - start_time
    print(f"Signal dictionary built: {valid_count} valid entries, elapsed={elapsed:.2f}s")

    return dictionary, t1_values, t2_values


def compress_dictionary(dictionary: np.ndarray, rank: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Project a full MRF dictionary to a low-rank temporal subspace."""

    dictionary = np.asarray(dictionary)
    assert dictionary.ndim == 3, "dictionary must have shape (n_t1, n_t2, n_tr)"
    assert rank > 0, "rank must be positive"
    assert rank <= dictionary.shape[2], "rank cannot exceed the dictionary time dimension"

    dict_2d = dictionary.reshape((dictionary.shape[0] * dictionary.shape[1], dictionary.shape[2]))
    _, _, vh = svd(dict_2d, full_matrices=False)
    bases = vh[:rank, :]
    compressed_2d = dict_2d @ bases.conj().T
    compressed = compressed_2d.reshape((dictionary.shape[0], dictionary.shape[1], rank))
    return compressed, bases


def build_and_compress_dictionary(
    simulator: MRFEPGSimulator,
    fa_train: np.ndarray,
    t1_grid: np.ndarray | None = None,
    t2_grid: np.ndarray | None = None,
    rank: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a dense MRF dictionary and project it to a temporal subspace."""

    dictionary, t1_values, t2_values = build_signal_dictionary(simulator, fa_train, t1_grid, t2_grid)
    compressed, bases = compress_dictionary(dictionary, rank=rank)
    print(f"Dictionary compressed: rank={rank}, compressed_shape={compressed.shape}")
    return compressed, bases, t1_values, t2_values

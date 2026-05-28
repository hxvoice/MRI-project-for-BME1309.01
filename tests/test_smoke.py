import numpy as np

from mri_project.dictionary import (
    MRFEPGSimulator,
    build_and_compress_dictionary,
    build_signal_dictionary,
    compress_dictionary,
    generate_mrf_fa_train_spline,
)
from mri_project.forward import generate_tga_trajectory, prepare_phantom, simulate_kspace


def test_core_imports_and_forward_shapes():
    traj = generate_tga_trajectory(n_tr=3, fov=8e-3, res=1e-3, readout_time=24e-6)
    assert traj.shape == (3, 24, 2)

    param_map = prepare_phantom(shape=(8, 8))
    assert param_map.shape == (8, 8, 3)

    signal_dict = np.ones((2, 2, 3), dtype=np.complex64)
    signal_dict[1, 1, :] = np.array([1.0, 0.8, 0.6], dtype=np.complex64)
    t1_grid = np.array([850.0, 1350.0], dtype=np.float32)
    t2_grid = np.array([60.0, 80.0], dtype=np.float32)

    kspace = simulate_kspace(
        param_map=param_map,
        signal_dict=signal_dict,
        t1_grid=t1_grid,
        t2_grid=t2_grid,
        traj=traj,
        n_coils=2,
        noise_level=0.0,
    )
    assert kspace.shape == (2, 3, 24)
    assert np.iscomplexobj(kspace)
    assert np.all(np.isfinite(kspace))


def test_dictionary_simulator_smoke():
    simulator = MRFEPGSimulator(num_states=20)
    fa_train = generate_mrf_fa_train_spline(num_trs=8, num_anchors=4)
    signal = simulator.simulate_mrf_fisp(t1=850.0, t2=60.0, fa_train=fa_train)
    legacy_signal = simulator.simulate_mrf_fisp(T1=850.0, T2=60.0, fa_train=fa_train)

    assert signal.shape == (8,)
    assert np.iscomplexobj(signal)
    assert np.all(np.isfinite(signal))
    assert legacy_signal.shape == (8,)


def test_small_dictionary_build_and_compression():
    simulator = MRFEPGSimulator(num_states=20)
    fa_train = generate_mrf_fa_train_spline(num_trs=8, num_anchors=4)
    t1_grid = np.array([850.0, 1350.0], dtype=np.float32)
    t2_grid = np.array([60.0, 80.0], dtype=np.float32)

    signal_dict, returned_t1, returned_t2 = build_signal_dictionary(simulator, fa_train, t1_grid, t2_grid)
    assert signal_dict.shape == (2, 2, 8)
    assert np.array_equal(returned_t1, t1_grid)
    assert np.array_equal(returned_t2, t2_grid)

    compressed, bases = compress_dictionary(signal_dict, rank=2)
    assert compressed.shape == (2, 2, 2)
    assert bases.shape == (2, 8)

    compressed2, bases2, returned_t1, returned_t2 = build_and_compress_dictionary(
        simulator,
        fa_train,
        t1_grid=t1_grid,
        t2_grid=t2_grid,
        rank=2,
    )
    assert compressed2.shape == (2, 2, 2)
    assert bases2.shape == (2, 8)
    assert np.array_equal(returned_t1, t1_grid)
    assert np.array_equal(returned_t2, t2_grid)

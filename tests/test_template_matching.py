import numpy as np
import pytest

from mri_project.quantification import execute_template_matching, generate_mock_coeff_maps, match_dictionary_indices


def test_execute_template_matching_recovers_exact_dictionary_entries():
    t1_grid = np.array([850.0, 1350.0])
    t2_grid = np.array([60.0, 80.0])
    dict_compressed = np.array(
        [
            [[1.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j, 0.0 + 0j]],
            [[0.0 + 0j, 0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j, 0.0 + 0j]],
        ]
    )
    coeff_maps = np.array(
        [
            [[1.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j, 0.0 + 0j]],
            [[0.0 + 0j, 0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j, 0.0 + 0j]],
        ]
    )

    t1_map, t2_map, pd_map = execute_template_matching(
        coeff_maps,
        dict_compressed,
        t1_grid,
        t2_grid,
    )

    np.testing.assert_array_equal(t1_map, [[850.0, 850.0], [1350.0, 1350.0]])
    np.testing.assert_array_equal(t2_map, [[60.0, 80.0], [60.0, 80.0]])
    np.testing.assert_allclose(pd_map, np.linalg.norm(coeff_maps, axis=-1))


def test_match_dictionary_indices_return_flat_argmax_indices():
    dict_compressed = np.array(
        [
            [[1.0 + 0j, 0.0 + 0j], [0.0 + 1j, 0.0 + 0j]],
            [[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]],
        ]
    )
    coeff_maps = np.array(
        [
            [[0.0 + 1j, 0.0 + 0j], [1.0 + 0j, 1.0 + 0j]],
            [[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]],
        ]
    )

    indices, pd_map = match_dictionary_indices(coeff_maps, dict_compressed)

    np.testing.assert_array_equal(indices.reshape(2, 2), [[1, 3], [2, 0]])
    np.testing.assert_allclose(pd_map, np.linalg.norm(coeff_maps, axis=-1))


def test_generate_mock_coeff_maps_shape_and_determinism():
    t1_grid = np.array([850.0, 1350.0])
    t2_grid = np.array([60.0, 80.0])
    dict_compressed = np.ones((2, 2, 3), dtype=np.complex128)

    coeff_maps_1 = generate_mock_coeff_maps(
        t1_grid,
        t2_grid,
        dict_compressed,
        nx=4,
        ny=5,
        seed=7,
    )
    coeff_maps_2 = generate_mock_coeff_maps(
        t1_grid,
        t2_grid,
        dict_compressed,
        nx=4,
        ny=5,
        seed=7,
    )

    assert coeff_maps_1.shape == (4, 5, 3)
    np.testing.assert_allclose(coeff_maps_1, coeff_maps_2)


def test_execute_template_matching_rejects_basis_mismatch():
    with pytest.raises(ValueError, match="basis dimension"):
        execute_template_matching(
            np.zeros((2, 2, 3)),
            np.zeros((4, 5, 2)),
            np.arange(4),
            np.arange(5),
        )


def test_execute_template_matching_zero_signal_has_zero_pd():
    t1_map, t2_map, pd_map = execute_template_matching(
        np.zeros((1, 1, 2), dtype=np.complex128),
        np.ones((2, 2, 2), dtype=np.complex128),
        np.array([850.0, 1350.0]),
        np.array([60.0, 80.0]),
    )

    assert t1_map.shape == (1, 1)
    assert t2_map.shape == (1, 1)
    np.testing.assert_array_equal(pd_map, [[0.0]])

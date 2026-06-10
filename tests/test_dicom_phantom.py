import numpy as np

from mri_project.forward.phantom import (
    TISSUE_LABELS,
    assign_tissue_parameters,
    create_brain_mask,
    robust_normalize_image,
    segment_dicom_tissues,
)


def test_dicom_helpers_create_compatible_param_map():
    image = np.zeros((16, 16), dtype=np.float32)
    image[3:13, 3:13] = np.linspace(10.0, 100.0, 100, dtype=np.float32).reshape(10, 10)

    normalized = robust_normalize_image(image)
    assert normalized.shape == image.shape
    assert normalized.dtype == np.float32
    assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0

    mask = create_brain_mask(normalized, min_size=8)
    labels = segment_dicom_tissues(normalized, mask)
    param_map = assign_tissue_parameters(
        labels,
        t1_grid=np.array([800.0, 1300.0, 4000.0], dtype=np.float32),
        t2_grid=np.array([70.0, 100.0, 2000.0], dtype=np.float32),
    )

    assert mask.shape == image.shape
    assert param_map.shape == (16, 16, 3)
    assert param_map.dtype == np.float32
    assert np.all(param_map[labels == TISSUE_LABELS["background"]] == 0)
    assert np.all(np.isfinite(param_map))
    assert np.count_nonzero(param_map[..., 2]) > 0

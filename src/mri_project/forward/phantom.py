"""Simple 2D brain phantom preparation."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from .io import plot_phantom, save_npy


TISSUE_LABELS = {
    "background": 0,
    "csf_like": 1,
    "wm_like": 2,
    "gm_like": 3,
}


def prepare_phantom(
    shape: tuple[int, int] = (96, 96),
    brain_radius: float = 0.8,
    wm_radius: float = 0.5,
    csf_radius: float = 0.85,
    tissue_values: dict[str, dict[str, float]] | None = None,
    save_path: str | Path | None = None,
    figure_path: str | Path | None = None,
) -> np.ndarray:
    """Create a three-layer 2D brain parameter map.

    Returns:
        Array with shape ``(height, width, 3)`` storing ``T1``, ``T2``, and
        proton-density values.
    """

    assert len(shape) == 2, "shape must be a 2D tuple"
    assert shape[0] > 0 and shape[1] > 0, "shape must be positive"
    assert brain_radius > 0, "brain_radius must be positive"
    assert wm_radius > 0, "wm_radius must be positive"
    assert csf_radius > 0, "csf_radius must be positive"
    tissue_values = {
        "wm": {"t1": 850.0, "t2": 60.0, "pd": 0.8},
        "gm": {"t1": 1350.0, "t2": 80.0, "pd": 0.9},
        "csf": {"t1": 4000.0, "t2": 2000.0, "pd": 1.0},
    } if tissue_values is None else tissue_values
    for tissue_name in ("wm", "gm", "csf"):
        assert tissue_name in tissue_values, f"missing tissue values for {tissue_name}"
        for field_name in ("t1", "t2", "pd"):
            assert field_name in tissue_values[tissue_name], f"missing {field_name} for {tissue_name}"

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )

    brain_mask = (xx**2 + yy**2) < brain_radius**2
    wm_mask = (xx**2 + yy**2) < wm_radius**2
    gm_mask = brain_mask & ~wm_mask
    csf_mask = ((xx**2 + yy**2) < csf_radius**2) & ~brain_mask

    t1 = np.zeros(shape, dtype=np.float32)
    t2 = np.zeros(shape, dtype=np.float32)
    pd = np.zeros(shape, dtype=np.float32)

    t1[wm_mask] = tissue_values["wm"]["t1"]
    t2[wm_mask] = tissue_values["wm"]["t2"]
    pd[wm_mask] = tissue_values["wm"]["pd"]

    t1[gm_mask] = tissue_values["gm"]["t1"]
    t2[gm_mask] = tissue_values["gm"]["t2"]
    pd[gm_mask] = tissue_values["gm"]["pd"]

    t1[csf_mask] = tissue_values["csf"]["t1"]
    t2[csf_mask] = tissue_values["csf"]["t2"]
    pd[csf_mask] = tissue_values["csf"]["pd"]

    if pd.max() > 0:
        pd = pd / pd.max()

    param_map = np.stack([t1, t2, pd], axis=-1)

    if save_path is not None:
        save_npy(param_map, save_path)
    if figure_path is not None:
        plot_phantom(t1, t2, pd, figure_path)

    return param_map


def load_dicom_image(dicom_path: str | Path) -> np.ndarray:
    """Load a DICOM slice as a float32 image with rescale tags applied."""

    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to read DICOM files. Install the project dependencies first.") from exc

    dataset = pydicom.dcmread(str(dicom_path))
    image = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if image.ndim > 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM image after squeezing, got shape {image.shape}.")
    return image.astype(np.float32, copy=False)


def robust_normalize_image(image: np.ndarray) -> np.ndarray:
    """Clip an image by robust percentiles and normalize it into [0, 1]."""

    image = np.asarray(image, dtype=np.float32)
    finite_image = np.where(np.isfinite(image), image, 0.0)
    nonzero = finite_image[np.abs(finite_image) > 0]
    sample = nonzero if nonzero.size >= 16 else finite_image.ravel()
    lo, hi = np.percentile(sample, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite_image))
        hi = float(np.max(finite_image))
    if hi <= lo:
        return np.zeros_like(finite_image, dtype=np.float32)

    normalized = np.clip(finite_image, lo, hi)
    normalized = (normalized - lo) / (hi - lo)
    return normalized.astype(np.float32, copy=False)


def resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D image, using skimage/scipy when available and nearest fallback otherwise."""

    image = np.asarray(image, dtype=np.float32)
    if image.shape == shape:
        return image.astype(np.float32, copy=False)

    try:
        from skimage.transform import resize

        return resize(image, shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)
    except ImportError:
        pass

    try:
        from scipy.ndimage import zoom

        zoom_factors = (shape[0] / image.shape[0], shape[1] / image.shape[1])
        resized = zoom(image, zoom_factors, order=1)
        return _crop_or_pad_to_shape(resized, shape).astype(np.float32)
    except ImportError:
        warnings.warn("Neither scikit-image nor scipy is available; using nearest-neighbor resize fallback.", RuntimeWarning)
        row_index = np.linspace(0, image.shape[0] - 1, shape[0]).round().astype(int)
        col_index = np.linspace(0, image.shape[1] - 1, shape[1]).round().astype(int)
        return image[np.ix_(row_index, col_index)].astype(np.float32)


def create_brain_mask(image: np.ndarray, min_size: int | None = None) -> np.ndarray:
    """Create a coarse foreground brain mask from a normalized anatomical image."""

    image = np.asarray(image, dtype=np.float32)
    positive = image[image > 0]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=bool)

    try:
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(positive))
        threshold = max(threshold, 0.05)
    except ImportError:
        threshold = max(float(np.percentile(positive, 10.0)), 0.05)

    mask = image > threshold
    min_size = int(min_size or max(16, image.size // 200))

    try:
        from skimage.morphology import binary_closing, disk, remove_small_objects

        mask = remove_small_objects(mask, min_size=min_size)
        mask = binary_closing(mask, disk(2))
    except ImportError:
        try:
            from scipy.ndimage import binary_closing

            mask = binary_closing(mask, iterations=2)
            mask = _remove_small_objects_fallback(mask, min_size=min_size)
        except ImportError:
            warnings.warn("Morphology packages unavailable; using a minimal NumPy mask cleanup.", RuntimeWarning)
            mask = _remove_small_objects_fallback(mask, min_size=min_size)

    try:
        from scipy.ndimage import binary_fill_holes

        mask = binary_fill_holes(mask)
    except ImportError:
        pass

    return np.asarray(mask, dtype=bool)


def segment_dicom_tissues(image: np.ndarray, brain_mask: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Segment brain pixels into FLAIR-aware low/mid/high intensity tissue labels."""

    image = np.asarray(image, dtype=np.float32)
    brain_mask = np.asarray(brain_mask, dtype=bool)
    labels = np.zeros(image.shape, dtype=np.uint8)
    values = image[brain_mask]
    if values.size == 0:
        return labels

    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(values.reshape(-1, 1))
        means = np.array([values[clusters == idx].mean() for idx in range(3)])
        ordered_clusters = np.argsort(means)
        mapped = np.zeros(values.shape, dtype=np.uint8)
        mapped[clusters == ordered_clusters[0]] = TISSUE_LABELS["csf_like"]
        mapped[clusters == ordered_clusters[1]] = TISSUE_LABELS["wm_like"]
        mapped[clusters == ordered_clusters[2]] = TISSUE_LABELS["gm_like"]
        labels[brain_mask] = mapped
    except ImportError:
        warnings.warn("scikit-learn is unavailable; using percentile tissue segmentation fallback.", RuntimeWarning)
        q1, q2 = np.percentile(values, [33.0, 66.0])
        labels[brain_mask & (image <= q1)] = TISSUE_LABELS["csf_like"]
        labels[brain_mask & (image > q1) & (image <= q2)] = TISSUE_LABELS["wm_like"]
        labels[brain_mask & (image > q2)] = TISSUE_LABELS["gm_like"]

    return labels


def assign_tissue_parameters(
    labels: np.ndarray,
    t1_grid: np.ndarray | None = None,
    t2_grid: np.ndarray | None = None,
    tissue_values: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    """Map tissue labels to a ``(H, W, 3)`` T1/T2/PD parameter map."""

    labels = np.asarray(labels)
    tissue_values = {
        "wm_like": {"t1": 800.0, "t2": 70.0, "pd": 0.70},
        "gm_like": {"t1": 1300.0, "t2": 100.0, "pd": 0.85},
        "csf_like": {"t1": 4000.0, "t2": 2000.0, "pd": 1.00},
    } if tissue_values is None else tissue_values

    param_map = np.zeros(labels.shape + (3,), dtype=np.float32)
    for tissue_name in ("csf_like", "wm_like", "gm_like"):
        values = tissue_values[tissue_name]
        tissue_mask = labels == TISSUE_LABELS[tissue_name]
        param_map[tissue_mask, 0] = _nearest_grid_or_clip(values["t1"], t1_grid)
        param_map[tissue_mask, 1] = _nearest_grid_or_clip(values["t2"], t2_grid)
        param_map[tissue_mask, 2] = values["pd"]
    return param_map


def create_dicom_derived_param_map(
    dicom_path: str | Path,
    shape: tuple[int, int],
    t1_grid: np.ndarray | None = None,
    t2_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create an anatomical DICOM-derived phantom with preset tissue parameters."""

    raw_image = load_dicom_image(dicom_path)
    normalized = robust_normalize_image(raw_image)
    normalized = resize_image(normalized, shape)
    normalized = robust_normalize_image(normalized)
    brain_mask = create_brain_mask(normalized)
    labels = segment_dicom_tissues(normalized, brain_mask)
    param_map = assign_tissue_parameters(labels, t1_grid=t1_grid, t2_grid=t2_grid)

    preview_data = {
        "normalized": normalized,
        "brain_mask": brain_mask.astype(np.float32),
        "labels": labels,
        "t1": param_map[..., 0],
        "t2": param_map[..., 1],
        "pd": param_map[..., 2],
    }
    return param_map, preview_data


def _nearest_grid_or_clip(value: float, grid: np.ndarray | None) -> float:
    if grid is None:
        return float(value)
    grid = np.asarray(grid, dtype=np.float32)
    if grid.size == 0:
        return float(value)
    return float(grid[int(np.argmin(np.abs(grid - value)))])


def _crop_or_pad_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=image.dtype)
    rows = min(shape[0], image.shape[0])
    cols = min(shape[1], image.shape[1])
    output[:rows, :cols] = image[:rows, :cols]
    return output


def _remove_small_objects_fallback(mask: np.ndarray, min_size: int) -> np.ndarray:
    try:
        from scipy.ndimage import label

        labeled, num_features = label(mask)
        if num_features == 0:
            return mask
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_size
        keep[0] = False
        return keep[labeled]
    except ImportError:
        if int(mask.sum()) < min_size:
            return np.zeros_like(mask, dtype=bool)
        return mask.astype(bool)

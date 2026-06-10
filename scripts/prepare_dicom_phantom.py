from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.forward.phantom import create_dicom_derived_param_map
from mri_project.forward.io import save_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a DICOM-derived anatomical phantom for the 2D MRF pipeline.")
    parser.add_argument(
        "--dicom-path",
        type=Path,
        default=None,
        help="Path to the DICOM slice. Defaults to the first .dcm file found under the project root.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/brain_param_map_2d.npy"),
        help="Output .npy parameter map path.",
    )
    parser.add_argument(
        "--preview-path",
        type=Path,
        default=Path("data/processed/dicom_phantom_preview.png"),
        help="Output preview PNG path.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the preview without opening an interactive window.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing parameter map.")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_dicom_path(project_root: Path) -> Path:
    candidates = sorted(project_root.rglob("*.dcm"))
    if not candidates:
        raise FileNotFoundError("No .dcm files were found under the project root. Pass --dicom-path explicitly.")
    if len(candidates) > 1:
        print("[DICOM] Multiple .dcm files found; using the first by default:")
        for index, candidate in enumerate(candidates, start=1):
            marker = "  <- selected" if index == 1 else ""
            print(f"  {index}. {candidate}{marker}")
    return candidates[0]


def save_preview(preview_data: dict[str, np.ndarray], save_path: Path, show: bool) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Normalized DICOM", preview_data["normalized"], "gray", None, None),
        ("Brain mask", preview_data["brain_mask"], "gray", 0, 1),
        ("Tissue labels", preview_data["labels"], "tab10", 0, 3),
        ("T1 map (ms)", preview_data["t1"], "magma", 0, None),
        ("T2 map (ms)", preview_data["t2"], "viridis", 0, None),
        ("PD map", preview_data["pd"], "bone", 0, 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("DICOM-derived anatomical phantom", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Preview] Saved preview to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dicom_path = project_path(args.dicom_path) if args.dicom_path is not None else find_dicom_path(PROJECT_ROOT)
    output_path = project_path(args.output_path)
    preview_path = project_path(args.preview_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite to replace it.")

    dictionary_path = PROJECT_ROOT / "data" / "processed" / "mrf_dictionary_data.npz"
    if dictionary_path.exists():
        dictionary_data = np.load(dictionary_path)
        t1_grid = dictionary_data["t1"]
        t2_grid = dictionary_data["t2"]
        print(f"[Grid] Using dictionary grids from: {dictionary_path}")
    else:
        t1_grid = config.t1_grid()
        t2_grid = config.t2_grid()
        print("[Grid] Dictionary package not found; using grids from pipeline_config.py.")

    print(f"[DICOM] Loading: {dicom_path}")
    param_map, preview_data = create_dicom_derived_param_map(
        dicom_path=dicom_path,
        shape=config.IMG_SHAPE,
        t1_grid=t1_grid,
        t2_grid=t2_grid,
    )
    save_npy(param_map, output_path)
    save_preview(preview_data, preview_path, show=not args.no_show)

    labels, counts = np.unique(preview_data["labels"], return_counts=True)
    label_summary = ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(labels, counts))
    print(f"[Output] Saved DICOM-derived phantom to: {output_path}")
    print(f"[Output] param_map shape={param_map.shape}, dtype={param_map.dtype}, label counts={label_summary}")


if __name__ == "__main__":
    main()

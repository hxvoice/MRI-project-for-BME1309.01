from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_project.quantification import execute_template_matching


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MRF template matching and render quantitative maps.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("data/output/quantitative_maps.png"),
        help="Path used to save the rendered T1/T2/PD map figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening the interactive popup window.",
    )
    return parser.parse_args()


def plot_quantitative_maps(
    t1_map: np.ndarray,
    t2_map: np.ndarray,
    pd_map: np.ndarray,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Render final T1, T2, and PD maps."""

    print("\n[Render] Rendering final quantitative maps...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(t1_map, cmap="magma", vmin=0, vmax=2000)
    axes[0].set_title("T1 Map (ms)")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(t2_map, cmap="viridis", vmin=0, vmax=150)
    axes[1].set_title("T2 Map (ms)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(pd_map, cmap="bone")
    axes[2].set_title("PD Map (a.u.)")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("MR Fingerprinting Quantitative Maps", fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Render] Saved quantitative maps to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def load_dictionary(dict_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the compressed dictionary package used by template matching."""

    dictionary_data = np.load(dict_path)
    return (
        dictionary_data["compressed_dict"],
        dictionary_data["t1"],
        dictionary_data["t2"],
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    print(">>> Starting MRF template matching pipeline <<<")

    dict_path = project_root / "data" / "processed" / "mrf_dictionary_data.npz"
    print(f"\n[Init] Loading compressed dictionary:\n{dict_path}")
    dict_compressed, t1_grid, t2_grid = load_dictionary(dict_path)
    print(f"[Init] Dictionary shape: {dict_compressed.shape}")

    coeff_maps_path = project_root / "data" / "output" / "reconstructed_coeff_maps.npy"
    print(f"\n[Data] Loading reconstructed coefficient maps:\n{coeff_maps_path}")
    coeff_maps_input = np.load(coeff_maps_path)
    print(f"[Data] Coefficient map shape: {coeff_maps_input.shape}")

    print("\n[Match] Running template matching...")
    start_time = time.time()
    t1_result, t2_result, pd_result = execute_template_matching(
        coeff_maps_input,
        dict_compressed,
        t1_grid,
        t2_grid,
    )
    num_pixels = coeff_maps_input.shape[0] * coeff_maps_input.shape[1]
    elapsed = time.time() - start_time
    print(f"[Match] Matched {num_pixels} pixels in {elapsed:.3f} seconds.")

    save_path = args.save_path
    if not save_path.is_absolute():
        save_path = project_root / save_path
    plot_quantitative_maps(t1_result, t2_result, pd_result, save_path=save_path, show=not args.no_show)
    print("\n>>> Template matching pipeline finished <<<")


if __name__ == "__main__":
    main()

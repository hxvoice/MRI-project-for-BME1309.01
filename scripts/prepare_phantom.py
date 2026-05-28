from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_project.forward import prepare_phantom


def main() -> None:
    output_dir = Path("data/processed")
    prepare_phantom(
        save_path=output_dir / "brain_param_map_2d.npy",
        figure_path=output_dir / "simulated_brain_phantom.png",
    )
    print(f"Saved phantom to {output_dir / 'brain_param_map_2d.npy'}")


if __name__ == "__main__":
    main()

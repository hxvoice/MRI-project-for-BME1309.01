from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.forward import prepare_phantom


def main() -> None:
    output_dir = Path("data/processed")
    prepare_phantom(
        shape=config.IMG_SHAPE,
        save_path=output_dir / "brain_param_map_2d.npy",
        figure_path=output_dir / "simulated_brain_phantom.png",
    )
    print(f"Saved phantom to {output_dir / 'brain_param_map_2d.npy'}")


if __name__ == "__main__":
    main()

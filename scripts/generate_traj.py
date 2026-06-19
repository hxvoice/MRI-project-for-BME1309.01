from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.forward import generate_tga_trajectory
from mri_project.forward.io import plot_traj_2d, save_npy


def main() -> None:
    output_dir = Path("data/processed")
    traj = generate_tga_trajectory(
        n_tr=config.N_TR,
        tiny_golden_angle=config.TINY_GOLDEN_ANGLE,
        fov=config.SPIRAL_FOV,
        res=config.SPIRAL_RES,
        readout_time=config.SPIRAL_READOUT_TIME,
        undersample_center=config.SPIRAL_UNDERSAMPLE_CENTER,
        undersample_edge=config.SPIRAL_UNDERSAMPLE_EDGE,
        adc_rate=config.SPIRAL_ADC_RATE,
    )
    save_npy(traj, output_dir / "traj_full_2d.npy")
    plot_traj_2d(traj[:5].reshape(-1, 2), "First 5 TGA spiral TRs", output_dir / "first5_traj.png")
    print(f"Saved trajectory to {output_dir / 'traj_full_2d.npy'}")


if __name__ == "__main__":
    main()

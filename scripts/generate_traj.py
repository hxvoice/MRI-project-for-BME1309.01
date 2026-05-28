from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_project.forward import generate_tga_trajectory
from mri_project.forward.io import plot_traj_2d, save_npy


def main() -> None:
    output_dir = Path("data/processed")
    traj = generate_tga_trajectory()
    save_npy(traj, output_dir / "traj_full_2d.npy")
    plot_traj_2d(traj[:5].reshape(-1, 2), "First 5 TGA spiral TRs", output_dir / "first5_traj.png")
    print(f"Saved trajectory to {output_dir / 'traj_full_2d.npy'}")


if __name__ == "__main__":
    main()

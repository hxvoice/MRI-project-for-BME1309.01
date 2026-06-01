from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.forward.simulation import simulate_kspace
from mri_project.forward.io import save_npy


def main() -> None:
    processed_dir = Path("data/processed")
    output_dir = Path("data/output")

    param_map = np.load(processed_dir / "brain_param_map_2d.npy")
    traj = np.load(processed_dir / "traj_full_2d.npy")
    dictionary_data = np.load(processed_dir / "mrf_dictionary_data.npz")

    kspace = simulate_kspace(
        param_map=param_map,
        signal_dict=dictionary_data["signal_dict"],
        t1_grid=dictionary_data["t1"],
        t2_grid=dictionary_data["t2"],
        traj=traj,
        n_coils=config.N_COILS,
        noise_level=config.NOISE_LEVEL,
        seed=config.RANDOM_SEED,
    )
    save_npy(kspace, output_dir / "mrf_kspace_2d_noisy.npy")
    print(f"Saved noisy k-space to {output_dir / 'mrf_kspace_2d_noisy.npy'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_project.dictionary import (
    MRFEPGSimulator,
    build_signal_dictionary,
    compress_dictionary,
    generate_mrf_fa_train_spline,
)


def main() -> None:
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    fa_train = generate_mrf_fa_train_spline()
    simulator = MRFEPGSimulator(num_states=200)
    signal_dict, t1_grid, t2_grid = build_signal_dictionary(simulator, fa_train)
    compressed_dict, bases = compress_dictionary(signal_dict, rank=5)

    output_path = output_dir / "mrf_dictionary_data.npz"
    np.savez(
        output_path,
        signal_dict=signal_dict,
        compressed_dict=compressed_dict,
        bases=bases,
        t1=t1_grid,
        t2=t2_grid,
        fa=fa_train,
    )
    print(f"Saved dictionary package to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.dictionary import (
    MRFEPGSimulator,
    build_signal_dictionary,
    compress_dictionary,
    generate_mrf_fa_train_spline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and compress the MRF dictionary.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Array backend for dictionary generation.")
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA device id used when --device cuda is selected.")
    parser.add_argument(
        "--dictionary-batch-size",
        type=int,
        default=None,
        help="Valid dictionary entries per GPU batch. Uses all valid entries when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUPY_CACHE_DIR", str(PROJECT_ROOT / ".cupy_cache"))
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    fa_train = generate_mrf_fa_train_spline(num_trs=config.N_TR)
    simulator = MRFEPGSimulator(num_states=config.EPG_NUM_STATES)
    signal_dict, t1_grid, t2_grid = build_signal_dictionary(
        simulator,
        fa_train,
        device=args.device,
        device_id=args.gpu_device,
        batch_size=args.dictionary_batch_size,
    )
    compressed_dict, bases = compress_dictionary(
        signal_dict,
        rank=config.SUBSPACE_RANK,
        device=args.device,
        device_id=args.gpu_device,
    )

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

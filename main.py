from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 2D MRF pipeline.")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of LLR reconstruction iterations.",
    )
    parser.add_argument(
        "--lambda-llr",
        type=float,
        default=1e-4,
        help="LLR regularization weight.",
    )
    parser.add_argument("--step-size", type=float, default=1e-3, help="Gradient step size.")
    parser.add_argument("--img-shape", type=int, nargs=2, default=(220, 220), metavar=("H", "W"))
    parser.add_argument("--patch-shape", type=int, nargs=2, default=(8, 8), metavar=("H", "W"))
    parser.add_argument(
        "--center-width",
        type=int,
        default=24,
        help="Central k-space width for calibration gridding.",
    )
    parser.add_argument("--calib-width", type=int, default=24, help="ESPIRiT calibration width.")
    return parser.parse_args()


def run_step(
    project_root: Path,
    env: dict[str, str],
    label: str,
    script: str,
    args: Sequence[str] = (),
    outputs: Sequence[Path] = (),
) -> None:
    output_paths = [project_root / output for output in outputs]

    print(f"\n=== {label} ===")
    if output_paths and all(output.exists() for output in output_paths):
        print("Skip: output file(s) already exist.")
        for output in output_paths:
            print(f"  {output.relative_to(project_root)}")
        return

    command = [sys.executable, script, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NUMBA_CACHE_DIR", str(project_root / ".numba_cache"))

    recon_args = [
        "--n-iter",
        str(args.n_iter),
        "--lambda-llr",
        str(args.lambda_llr),
        "--step-size",
        str(args.step_size),
        "--img-shape",
        str(args.img_shape[0]),
        str(args.img_shape[1]),
        "--patch-shape",
        str(args.patch_shape[0]),
        str(args.patch_shape[1]),
        "--center-width",
        str(args.center_width),
        "--calib-width",
        str(args.calib_width),
    ]

    print(">>> Starting full 2D MRF pipeline <<<")
    run_step(
        project_root,
        env,
        "1/6 Generate trajectory",
        "scripts/generate_traj.py",
        outputs=[
            Path("data/processed/traj_full_2d.npy"),
            Path("data/processed/first5_traj.png"),
        ],
    )
    run_step(
        project_root,
        env,
        "2/6 Prepare phantom",
        "scripts/prepare_phantom.py",
        outputs=[
            Path("data/processed/brain_param_map_2d.npy"),
            Path("data/processed/simulated_brain_phantom.png"),
        ],
    )
    run_step(
        project_root,
        env,
        "3/6 Build dictionary",
        "scripts/build_dictionary.py",
        outputs=[Path("data/processed/mrf_dictionary_data.npz")],
    )
    run_step(
        project_root,
        env,
        "4/6 Forward simulation",
        "scripts/run_forward_sim.py",
        outputs=[Path("data/output/mrf_kspace_2d_noisy.npy")],
    )
    run_step(
        project_root,
        env,
        "5/6 Subspace LLR reconstruction",
        "scripts/run_recon.py",
        args=recon_args,
        outputs=[Path("data/output/reconstructed_coeff_maps.npy")],
    )
    run_step(project_root, env, "6/6 Template matching and plotting", "scripts/template_matching.py")
    print("\n>>> Full 2D MRF pipeline finished <<<")


if __name__ == "__main__":
    main()

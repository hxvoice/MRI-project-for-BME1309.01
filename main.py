from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import pipeline_config as config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 2D MRF pipeline.")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=config.N_ITER,
        help="Number of LLR reconstruction iterations.",
    )
    parser.add_argument(
        "--lambda-llr",
        type=float,
        default=config.LAMBDA_LLR,
        help="LLR regularization weight.",
    )
    parser.add_argument("--step-size", type=float, default=config.STEP_SIZE, help="Gradient step size.")
    parser.add_argument("--img-shape", type=int, nargs=2, default=config.IMG_SHAPE, metavar=("H", "W"))
    parser.add_argument("--patch-shape", type=int, nargs=2, default=config.PATCH_SHAPE, metavar=("H", "W"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Array backend for reconstruction/matching.")
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA device id used when --device cuda is selected.")
    parser.add_argument(
        "--matching-batch-size",
        type=int,
        default=None,
        help="Pixels per template-matching similarity batch. Uses the full image when omitted.",
    )
    parser.add_argument(
        "--dictionary-batch-size",
        type=int,
        default=None,
        help="Valid dictionary entries per GPU batch. Uses all valid entries when omitted.",
    )
    parser.add_argument(
        "--center-width",
        type=int,
        default=config.CENTER_WIDTH,
        help="Central k-space width for calibration gridding.",
    )
    parser.add_argument("--calib-width", type=int, default=config.CALIB_WIDTH, help="ESPIRiT calibration width.")
    return parser.parse_args()


def npy_shape_is(expected_shape: tuple[int, ...]) -> Callable[[Path], bool]:
    def check(path: Path) -> bool:
        return np.load(path, mmap_mode="r").shape == expected_shape

    return check


def dictionary_matches_config(path: Path) -> bool:
    dictionary_data = np.load(path)
    signal_dict = dictionary_data["signal_dict"]
    compressed_dict = dictionary_data["compressed_dict"]
    bases = dictionary_data["bases"]
    return (
        signal_dict.ndim == 3
        and signal_dict.shape[2] == config.N_TR
        and compressed_dict.ndim == 3
        and compressed_dict.shape[2] == config.SUBSPACE_RANK
        and bases.shape == (config.SUBSPACE_RANK, config.N_TR)
    )


def traj_matches_recon_grid(img_shape: tuple[int, int], n_tr: int = config.N_TR) -> Callable[[Path], bool]:
    def check(path: Path) -> bool:
        traj = np.load(path, mmap_mode="r")
        expected_samples = max(2, int(round(config.SPIRAL_READOUT_TIME * config.SPIRAL_ADC_RATE)))
        if traj.ndim != 3 or traj.shape != (n_tr, expected_samples, 2):
            return False
        max_radius = float(np.max(np.linalg.norm(traj, axis=-1)))
        return max_radius <= max(img_shape) / 2.0 + 1.0

    return check


def kspace_matches_lightweight_defaults(n_tr: int = config.N_TR) -> Callable[[Path], bool]:
    def check(path: Path) -> bool:
        kspace = np.load(path, mmap_mode="r")
        return kspace.ndim == 3 and kspace.shape[0] == config.N_COILS and kspace.shape[1] == n_tr

    return check


def outputs_are_valid(project_root: Path, checks: Sequence[tuple[Path, Callable[[Path], bool]]]) -> bool:
    for output, check in checks:
        output_path = project_root / output
        if not output_path.exists():
            return False
        try:
            if not check(output_path):
                print(f"Will rerun: {output} has an incompatible shape or grid.")
                return False
        except (OSError, ValueError, KeyError) as error:
            print(f"Will rerun: could not validate {output}: {error}")
            return False
    return True


def outputs_are_current(project_root: Path, outputs: Sequence[Path], inputs: Sequence[Path]) -> bool:
    if not inputs:
        return True

    input_paths = [project_root / input_path for input_path in inputs]
    if any(not input_path.exists() for input_path in input_paths):
        return True

    newest_input = max(input_path.stat().st_mtime for input_path in input_paths)
    for output in outputs:
        output_path = project_root / output
        if output_path.stat().st_mtime < newest_input:
            print(f"Will rerun: {output} is older than its inputs.")
            return False
    return True


def run_step(
    project_root: Path,
    env: dict[str, str],
    label: str,
    script: str,
    args: Sequence[str] = (),
    inputs: Sequence[Path] = (),
    outputs: Sequence[Path] = (),
    output_checks: Sequence[tuple[Path, Callable[[Path], bool]]] = (),
) -> None:
    output_paths = [project_root / output for output in outputs]

    print(f"\n=== {label} ===")
    if (
        output_paths
        and all(output.exists() for output in output_paths)
        and outputs_are_valid(project_root, output_checks)
        and outputs_are_current(project_root, outputs, inputs)
    ):
        print("Skip: output file(s) already exist.")
        for output in output_paths:
            print(f"  {output.relative_to(project_root)}")
        return

    command = [sys.executable, script, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def main() -> None:
    args = parse_args()
    img_shape = tuple(args.img_shape)
    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NUMBA_CACHE_DIR", str(project_root / ".numba_cache"))
    env.setdefault("CUPY_CACHE_DIR", str(project_root / ".cupy_cache"))

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
        "--device",
        args.device,
        "--gpu-device",
        str(args.gpu_device),
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
        output_checks=[
            (Path("data/processed/traj_full_2d.npy"), traj_matches_recon_grid(img_shape)),
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
        output_checks=[
            (Path("data/processed/brain_param_map_2d.npy"), npy_shape_is((*img_shape, 3))),
        ],
    )
    run_step(
        project_root,
        env,
        "3/6 Build dictionary",
        "scripts/build_dictionary.py",
        args=[
            "--device",
            args.device,
            "--gpu-device",
            str(args.gpu_device),
            *(
                ["--dictionary-batch-size", str(args.dictionary_batch_size)]
                if args.dictionary_batch_size is not None
                else []
            ),
        ],
        outputs=[Path("data/processed/mrf_dictionary_data.npz")],
        output_checks=[
            (Path("data/processed/mrf_dictionary_data.npz"), dictionary_matches_config),
        ],
    )
    run_step(
        project_root,
        env,
        "4/6 Forward simulation",
        "scripts/run_forward_sim.py",
        inputs=[
            Path("data/processed/brain_param_map_2d.npy"),
            Path("data/processed/traj_full_2d.npy"),
            Path("data/processed/mrf_dictionary_data.npz"),
        ],
        outputs=[Path("data/output/mrf_kspace_2d_noisy.npy")],
        output_checks=[
            (Path("data/output/mrf_kspace_2d_noisy.npy"), kspace_matches_lightweight_defaults()),
        ],
    )
    run_step(
        project_root,
        env,
        "5/6 Subspace LLR reconstruction",
        "scripts/run_recon.py",
        args=recon_args,
        inputs=[
            Path("data/output/mrf_kspace_2d_noisy.npy"),
            Path("data/processed/traj_full_2d.npy"),
            Path("data/processed/mrf_dictionary_data.npz"),
        ],
        outputs=[Path("data/output/reconstructed_coeff_maps.npy")],
        output_checks=[
            (Path("data/output/reconstructed_coeff_maps.npy"), npy_shape_is((*img_shape, config.SUBSPACE_RANK))),
        ],
    )
    run_step(
        project_root,
        env,
        "6/6 Template matching and plotting",
        "scripts/template_matching.py",
        args=[
            "--save-path",
            "data/output/quantitative_maps.png",
            "--device",
            args.device,
            "--gpu-device",
            str(args.gpu_device),
            *(
                ["--batch-size", str(args.matching_batch_size)]
                if args.matching_batch_size is not None
                else []
            ),
        ],
    )
    print("\n>>> Full 2D MRF pipeline finished <<<")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import main as pipeline_main
import pipeline_config as config


DICOM_FINGERPRINT_PATH = Path("data/processed/pipeline_config_fingerprint_dicom.json")
DICOM_PHANTOM_CONFIG_KEYS = (
    "IMG_SHAPE",
    "T1_GRID_RANGES",
    "T2_GRID_RANGES",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 2D MRF pipeline with a DICOM-derived phantom.")
    parser.add_argument("--dicom-path", type=Path, default=None, help="DICOM slice used to derive anatomy.")
    parser.add_argument(
        "--preview-path",
        type=Path,
        default=Path("data/processed/dicom_phantom_preview.png"),
        help="Preview PNG for the DICOM-derived phantom.",
    )
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
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Array backend for reconstruction/matching. Defaults to CUDA; pass cpu to run without GPU.",
    )
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
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening interactive windows.")
    return parser.parse_args()


def find_dicom_path(project_root: Path) -> Path:
    candidates = sorted(project_root.rglob("*.dcm"))
    if not candidates:
        raise FileNotFoundError("No .dcm files were found under the project root. Pass --dicom-path explicitly.")
    if len(candidates) > 1:
        print("[DICOM] Multiple .dcm files found; using the first by default:")
        for index, candidate in enumerate(candidates, start=1):
            marker = "  <- selected" if index == 1 else ""
            print(f"  {index}. {candidate}{marker}")
    return candidates[0]


def dicom_phantom_matches(img_shape: tuple[int, int]) -> Callable[[Path], bool]:
    def check(path: Path) -> bool:
        phantom = np.load(path, mmap_mode="r")
        if phantom.shape != (*img_shape, 3):
            return False
        foreground = phantom[..., 2] > 0
        if not np.any(foreground):
            return False
        expected = np.array(
            [
                (800.0, 70.0, 0.70),
                (1300.0, 100.0, 0.85),
                (4000.0, 2000.0, 1.0),
            ],
            dtype=np.float32,
        )
        actual = np.unique(np.asarray(phantom[foreground]), axis=0)
        if actual.shape != expected.shape:
            return False
        actual = actual[np.lexsort((actual[:, 2], actual[:, 1], actual[:, 0]))]
        expected = expected[np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))]
        return bool(np.allclose(actual, expected, rtol=0.0, atol=1e-4))

    return check


def run_step(
    project_root: Path,
    env: dict[str, str],
    fingerprint: dict[str, Any],
    step_id: str,
    step_config: dict[str, Any],
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
        and pipeline_main.outputs_are_valid(project_root, output_checks)
        and pipeline_main.outputs_are_current(project_root, outputs, inputs)
        and pipeline_main.step_config_is_current(fingerprint, step_id, step_config)
    ):
        print("Skip: output file(s) already exist, are valid, and config is unchanged.")
        for output in output_paths:
            print(f"  {output.relative_to(project_root)}")
        return

    command = [sys.executable, script, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)
    fingerprint.setdefault("steps", {})[step_id] = copy.deepcopy(step_config)
    pipeline_main.save_fingerprint(project_root, fingerprint)


def main() -> None:
    args = parse_args()
    img_shape = tuple(args.img_shape)
    project_root = Path(__file__).resolve().parent
    dicom_path = args.dicom_path if args.dicom_path is not None else find_dicom_path(project_root)
    if not dicom_path.is_absolute():
        dicom_path = project_root / dicom_path

    pipeline_main.FINGERPRINT_PATH = DICOM_FINGERPRINT_PATH
    fingerprint = pipeline_main.load_fingerprint(project_root)

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

    step_configs = {
        "trajectory": pipeline_main.selected_config(pipeline_main.TRAJECTORY_CONFIG_KEYS),
        "dicom_phantom": {
            **pipeline_main.selected_config(DICOM_PHANTOM_CONFIG_KEYS),
            "dicom_path": str(dicom_path.resolve()),
        },
        "dictionary": {
            **pipeline_main.selected_config(pipeline_main.DICTIONARY_CONFIG_KEYS),
            "device": args.device,
            "gpu_device": args.gpu_device,
            "dictionary_batch_size": args.dictionary_batch_size,
        },
        "forward": {
            **pipeline_main.selected_config(pipeline_main.FORWARD_CONFIG_KEYS),
            "phantom_source": "dicom",
            "dicom_path": str(dicom_path.resolve()),
        },
        "recon": {
            **pipeline_main.selected_config(pipeline_main.RECON_CONFIG_KEYS),
            "n_iter": args.n_iter,
            "lambda_llr": args.lambda_llr,
            "step_size": args.step_size,
            "img_shape": pipeline_main.normalize_for_json(args.img_shape),
            "patch_shape": pipeline_main.normalize_for_json(args.patch_shape),
            "center_width": args.center_width,
            "calib_width": args.calib_width,
            "device": args.device,
            "gpu_device": args.gpu_device,
            "phantom_source": "dicom",
        },
        "matching": {
            **pipeline_main.selected_config(pipeline_main.MATCHING_CONFIG_KEYS),
            "device": args.device,
            "gpu_device": args.gpu_device,
            "matching_batch_size": args.matching_batch_size,
            "phantom_source": "dicom",
        },
    }

    print(">>> Starting full 2D MRF pipeline with DICOM-derived phantom <<<")
    print(f"[DICOM] Phantom source: {dicom_path}")
    pipeline_main.print_config_summary(step_configs)

    run_step(
        project_root,
        env,
        fingerprint,
        "trajectory",
        step_configs["trajectory"],
        "1/6 Generate trajectory",
        "scripts/generate_traj.py",
        outputs=[
            Path("data/processed/traj_full_2d.npy"),
            Path("data/processed/first5_traj.png"),
        ],
        output_checks=[
            (Path("data/processed/traj_full_2d.npy"), pipeline_main.traj_matches_recon_grid(img_shape)),
        ],
    )
    run_step(
        project_root,
        env,
        fingerprint,
        "dicom_phantom",
        step_configs["dicom_phantom"],
        "2/6 Prepare DICOM-derived phantom",
        "scripts/prepare_dicom_phantom.py",
        args=[
            "--dicom-path",
            str(dicom_path),
            "--preview-path",
            str(args.preview_path),
            "--overwrite",
            *(["--no-show"] if args.no_show else []),
        ],
        inputs=[dicom_path],
        outputs=[
            Path("data/processed/brain_param_map_2d.npy"),
            args.preview_path,
        ],
        output_checks=[
            (Path("data/processed/brain_param_map_2d.npy"), dicom_phantom_matches(img_shape)),
        ],
    )
    run_step(
        project_root,
        env,
        fingerprint,
        "dictionary",
        step_configs["dictionary"],
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
            (Path("data/processed/mrf_dictionary_data.npz"), pipeline_main.dictionary_matches_config),
        ],
    )
    run_step(
        project_root,
        env,
        fingerprint,
        "forward",
        step_configs["forward"],
        "4/6 Forward simulation",
        "scripts/run_forward_sim.py",
        inputs=[
            Path("data/processed/brain_param_map_2d.npy"),
            Path("data/processed/traj_full_2d.npy"),
            Path("data/processed/mrf_dictionary_data.npz"),
        ],
        outputs=[Path("data/output/mrf_kspace_2d_noisy.npy")],
        output_checks=[
            (Path("data/output/mrf_kspace_2d_noisy.npy"), pipeline_main.kspace_matches_lightweight_defaults()),
        ],
    )
    run_step(
        project_root,
        env,
        fingerprint,
        "recon",
        step_configs["recon"],
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
            (Path("data/output/reconstructed_coeff_maps.npy"), pipeline_main.npy_shape_is((*img_shape, config.SUBSPACE_RANK))),
        ],
    )
    run_step(
        project_root,
        env,
        fingerprint,
        "matching",
        step_configs["matching"],
        "6/6 Template matching and plotting",
        "scripts/template_matching.py",
        args=[
            "--save-path",
            "data/output/quantitative_maps.png",
            "--device",
            args.device,
            "--gpu-device",
            str(args.gpu_device),
            *(["--no-show"] if args.no_show else []),
            *(
                ["--batch-size", str(args.matching_batch_size)]
                if args.matching_batch_size is not None
                else []
            ),
        ],
        inputs=[
            Path("data/output/reconstructed_coeff_maps.npy"),
            Path("data/processed/mrf_dictionary_data.npz"),
        ],
        outputs=[Path("data/output/quantitative_maps.png")],
    )
    print("\n>>> Full DICOM-derived 2D MRF pipeline finished <<<")


if __name__ == "__main__":
    main()

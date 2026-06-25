from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

import pipeline_config as config


FINGERPRINT_PATH = Path("data/processed/pipeline_config_fingerprint.json")

TRAJECTORY_CONFIG_KEYS = (
    "N_TR",
    "IMG_SHAPE",
    "SPIRAL_FOV",
    "SPIRAL_RES",
    "SPIRAL_READOUT_TIME",
    "SPIRAL_UNDERSAMPLE_CENTER",
    "SPIRAL_UNDERSAMPLE_EDGE",
    "SPIRAL_ADC_RATE",
    "TINY_GOLDEN_ANGLE",
)
PHANTOM_CONFIG_KEYS = (
    "IMG_SHAPE",
    "PHANTOM_BRAIN_RADIUS",
    "PHANTOM_WM_RADIUS",
    "PHANTOM_CSF_RADIUS",
    "PHANTOM_TISSUES",
)
DICTIONARY_CONFIG_KEYS = (
    "N_TR",
    "SUBSPACE_RANK",
    "EPG_NUM_STATES",
    "EPG_TR",
    "EPG_TE",
    "EPG_TI",
    "FA_MIN",
    "FA_MAX",
    "FA_NUM_ANCHORS",
    "FA_RANDOM_SEED",
    "T1_GRID_RANGES",
    "T2_GRID_RANGES",
)
FORWARD_CONFIG_KEYS = (
    "N_COILS",
    "NOISE_LEVEL",
    "RANDOM_SEED",
)
RECON_CONFIG_KEYS = (
    "N_ITER",
    "LAMBDA_LLR",
    "STEP_SIZE",
    "PATCH_SHAPE",
    "CENTER_WIDTH",
    "CALIB_WIDTH",
    "SUBSPACE_RANK",
    "ESPIRIT_THRESH",
    "ESPIRIT_KERNEL_WIDTH",
    "ESPIRIT_CROP",
    "ESPIRIT_MAX_ITER",
    "ESPIRIT_SHOW_PBAR",
)
MATCHING_CONFIG_KEYS = ("SUBSPACE_RANK",)


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


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): normalize_for_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    return value


def selected_config(keys: Sequence[str]) -> dict[str, Any]:
    return {key: normalize_for_json(getattr(config, key)) for key in keys}


def config_diff(old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    if old is None:
        return sorted(new)
    changed = sorted(key for key, value in new.items() if old.get(key) != value)
    removed = sorted(key for key in old if key not in new)
    return [*changed, *removed]


def load_fingerprint(project_root: Path) -> dict[str, Any]:
    path = project_root / FINGERPRINT_PATH
    if not path.exists():
        return {"steps": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Will refresh config fingerprint: could not read {FINGERPRINT_PATH}: {error}")
        return {"steps": {}}
    if not isinstance(data, dict) or not isinstance(data.get("steps"), dict):
        print(f"Will refresh config fingerprint: {FINGERPRINT_PATH} has an unexpected format.")
        return {"steps": {}}
    return data


def save_fingerprint(project_root: Path, fingerprint: dict[str, Any]) -> None:
    path = project_root / FINGERPRINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")


def print_config_summary(step_configs: dict[str, dict[str, Any]]) -> None:
    print("\n[Config] Current key parameters")
    summary = {
        "image": selected_config(("IMG_SHAPE", "N_TR", "SUBSPACE_RANK", "N_COILS")),
        "recon": selected_config(("N_ITER", "LAMBDA_LLR", "STEP_SIZE", "PATCH_SHAPE")),
        "trajectory": selected_config(("SPIRAL_READOUT_TIME", "SPIRAL_ADC_RATE", "TINY_GOLDEN_ANGLE")),
        "noise": selected_config(("NOISE_LEVEL", "RANDOM_SEED")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[Config] Tracking fingerprints for {len(step_configs)} pipeline steps.")


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
            print(f"Will rerun: {output} is missing.")
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


def step_config_is_current(
    fingerprint: dict[str, Any],
    step_id: str,
    current_config: dict[str, Any],
) -> bool:
    saved_config = fingerprint.get("steps", {}).get(step_id)
    changed_keys = config_diff(saved_config, current_config)
    if changed_keys:
        print(f"Will rerun: config changed for {step_id}: {', '.join(changed_keys)}.")
        return False
    return True


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
        and outputs_are_valid(project_root, output_checks)
        and outputs_are_current(project_root, outputs, inputs)
        and step_config_is_current(fingerprint, step_id, step_config)
    ):
        print("Skip: output file(s) already exist, are valid, and config is unchanged.")
        for output in output_paths:
            print(f"  {output.relative_to(project_root)}")
        return

    command = [sys.executable, script, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)
    fingerprint.setdefault("steps", {})[step_id] = copy.deepcopy(step_config)
    save_fingerprint(project_root, fingerprint)


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

    step_configs = {
        "trajectory": selected_config(TRAJECTORY_CONFIG_KEYS),
        "phantom": selected_config(PHANTOM_CONFIG_KEYS),
        "dictionary": {
            **selected_config(DICTIONARY_CONFIG_KEYS),
            "device": args.device,
            "gpu_device": args.gpu_device,
            "dictionary_batch_size": args.dictionary_batch_size,
        },
        "forward": selected_config(FORWARD_CONFIG_KEYS),
        "recon": {
            **selected_config(RECON_CONFIG_KEYS),
            "n_iter": args.n_iter,
            "lambda_llr": args.lambda_llr,
            "step_size": args.step_size,
            "img_shape": normalize_for_json(args.img_shape),
            "patch_shape": normalize_for_json(args.patch_shape),
            "center_width": args.center_width,
            "calib_width": args.calib_width,
            "device": args.device,
            "gpu_device": args.gpu_device,
        },
        "matching": {
            **selected_config(MATCHING_CONFIG_KEYS),
            "device": args.device,
            "gpu_device": args.gpu_device,
            "matching_batch_size": args.matching_batch_size,
        },
    }
    fingerprint = load_fingerprint(project_root)

    print(">>> Starting full 2D MRF pipeline <<<")
    print_config_summary(step_configs)
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
            (Path("data/processed/traj_full_2d.npy"), traj_matches_recon_grid(img_shape)),
        ],
    )
    run_step(
        project_root,
        env,
        fingerprint,
        "phantom",
        step_configs["phantom"],
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
            (Path("data/processed/mrf_dictionary_data.npz"), dictionary_matches_config),
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
            (Path("data/output/mrf_kspace_2d_noisy.npy"), kspace_matches_lightweight_defaults()),
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
            (Path("data/output/reconstructed_coeff_maps.npy"), npy_shape_is((*img_shape, config.SUBSPACE_RANK))),
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
            "--dict-path",
            "data/processed/mrf_dictionary_data.npz",
            "--coeff-maps",
            "data/output/reconstructed_coeff_maps.npy",
            "--save-path",
            "data/output/quantitative_maps.png",
            "--save-npz",
            "data/output/quantitative_maps.npz",
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
        inputs=[
            Path("data/output/reconstructed_coeff_maps.npy"),
            Path("data/processed/mrf_dictionary_data.npz"),
        ],
        outputs=[Path("data/output/quantitative_maps.png")],
    )
    print("\n>>> Full 2D MRF pipeline finished <<<")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pipeline_config as config


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real cMRF pipeline: extract -> recon -> matching.")
    parser.add_argument("--base-dir", type=Path, default=config.CMRF_DATA_ROOT, help="Root directory of the cMRF dataset.")
    parser.add_argument("--scanner", default=config.CMRF_SCANNER, help="Scanner subdirectory under --base-dir.")
    parser.add_argument("--scan-name", default=config.CMRF_SCAN_NAME, help="Raw cMRF HDF5 filename.")
    parser.add_argument("--fa-file", type=Path, default=Path(config.CMRF_FA_FILE), help="Flip-angle text file path.")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Processed cMRF directory.")
    parser.add_argument("--output-dir", type=Path, default=config.CMRF_OUTPUT_DIR, help="Pipeline output directory.")
    parser.add_argument("--n-iter", type=int, default=config.CMRF_N_ITER, help="LLR reconstruction iteration count.")
    parser.add_argument("--lambda-llr", type=float, default=config.CMRF_LAMBDA_LLR, help="LLR regularization weight.")
    parser.add_argument("--step-size", type=float, default=config.STEP_SIZE, help="Gradient step size.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=config.CMRF_DEVICE, help="Compute backend.")
    parser.add_argument("--gpu-device", type=int, default=config.CMRF_GPU_DEVICE, help="CUDA device id.")
    parser.add_argument(
        "--matching-batch-size",
        type=int,
        default=config.CMRF_MATCHING_BATCH_SIZE,
        help="Pixels matched per template-matching batch.",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Skip cMRF extraction even if outputs are stale.")
    parser.add_argument("--skip-recon", action="store_true", help="Skip reconstruction even if outputs are stale.")
    parser.add_argument("--skip-matching", action="store_true", help="Skip template matching even if outputs are stale.")
    parser.add_argument("--force", action="store_true", help="Run all non-skipped steps even when outputs look current.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_paths(args: argparse.Namespace) -> dict[str, Path]:
    base_dir = resolve_path(args.base_dir)
    scanner_dir = base_dir / args.scanner
    processed_dir = resolve_path(args.processed_dir) if args.processed_dir is not None else scanner_dir / "processed"
    output_dir = resolve_path(args.output_dir)
    fa_path = resolve_path(args.fa_file) if args.fa_file.is_absolute() else base_dir / args.fa_file
    return {
        "base_dir": base_dir,
        "scanner_dir": scanner_dir,
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "scan_path": scanner_dir / args.scan_name,
        "fa_path": fa_path,
        "dict_path": processed_dir / "mrf_dictionary_data.npz",
        "kspace_path": processed_dir / "mrf_kspace_noisy.npy",
        "traj_path": processed_dir / "traj_full_2d.npy",
        "csm_path": processed_dir / "csm.npy",
        "coeff_maps_path": output_dir / "reconstructed_coeff_maps.npy",
        "quant_png_path": output_dir / "quantitative_maps.png",
        "quant_npz_path": output_dir / "quantitative_maps.npz",
        "fingerprint_path": output_dir / "cmrf_pipeline_fingerprint.json",
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"steps": {}}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {"steps": {}}


def save_fingerprint(path: Path, fingerprint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(fingerprint, file, indent=2, sort_keys=True)


def files_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def outputs_are_current(outputs: list[Path], inputs: list[Path]) -> bool:
    if not outputs or not files_exist(outputs):
        return False
    if not inputs:
        return True
    if not files_exist(inputs):
        return False
    output_mtime = min(path.stat().st_mtime for path in outputs)
    input_mtime = max(path.stat().st_mtime for path in inputs)
    return output_mtime >= input_mtime


def should_run(
    fingerprint: dict[str, Any],
    step_id: str,
    config_blob: dict[str, Any],
    inputs: list[Path],
    outputs: list[Path],
    force: bool,
) -> bool:
    if force:
        return True
    if not files_exist(outputs):
        return True
    if not outputs_are_current(outputs, inputs):
        return True
    return fingerprint.get("steps", {}).get(step_id) != json_ready(config_blob)


def run_step(
    step_id: str,
    label: str,
    command: list[str],
    config_blob: dict[str, Any],
    inputs: list[Path],
    outputs: list[Path],
    fingerprint: dict[str, Any],
    fingerprint_path: Path,
    force: bool,
    env: dict[str, str],
) -> None:
    print(f"\n=== {label} ===")
    if not should_run(fingerprint, step_id, config_blob, inputs, outputs, force):
        print("Skip: outputs exist, inputs are not newer, and config is unchanged.")
        return

    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    fingerprint.setdefault("steps", {})[step_id] = copy.deepcopy(json_ready(config_blob))
    save_fingerprint(fingerprint_path, fingerprint)


def main() -> None:
    args = parse_args()
    paths = selected_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
    env.setdefault("CUPY_CACHE_DIR", str(PROJECT_ROOT / ".cupy_cache"))

    fingerprint = load_fingerprint(paths["fingerprint_path"])

    extract_config = {
        "base_dir": paths["base_dir"],
        "scanner": args.scanner,
        "scan_name": args.scan_name,
        "fa_path": paths["fa_path"],
        "processed_dir": paths["processed_dir"],
        "subspace_rank": config.CMRF_SUBSPACE_RANK,
        "t1_grid_ranges": config.CMRF_T1_GRID_RANGES,
        "t2_grid_ranges": config.CMRF_T2_GRID_RANGES,
    }
    recon_config = {
        "input_dir": paths["processed_dir"],
        "output_dir": paths["output_dir"],
        "n_iter": args.n_iter,
        "lambda_llr": args.lambda_llr,
        "step_size": args.step_size,
        "device": args.device,
        "gpu_device": args.gpu_device,
        "patch_shape": config.PATCH_SHAPE,
        "subspace_rank": config.CMRF_SUBSPACE_RANK,
    }
    matching_config = {
        "dict_path": paths["dict_path"],
        "coeff_maps": paths["coeff_maps_path"],
        "save_path": paths["quant_png_path"],
        "save_npz": paths["quant_npz_path"],
        "device": args.device,
        "gpu_device": args.gpu_device,
        "batch_size": args.matching_batch_size,
        "t1_vmax": config.CMRF_T1_VMAX,
        "t2_vmax": config.CMRF_T2_VMAX,
    }

    if args.skip_extract:
        print("\n=== 1/3 Extract cMRF data ===")
        print("Skip requested with --skip-extract.")
    else:
        run_step(
            "extract",
            "1/3 Extract cMRF data",
            [
                sys.executable,
                "scripts/extract_cmrf_data.py",
                "--base-dir",
                str(paths["base_dir"]),
                "--scanner",
                args.scanner,
                "--scan-name",
                args.scan_name,
                "--fa-file",
                str(paths["fa_path"]),
                "--output-dir",
                str(paths["processed_dir"]),
                "--subspace-rank",
                str(config.CMRF_SUBSPACE_RANK),
            ],
            extract_config,
            [paths["scan_path"], paths["fa_path"]],
            [paths["dict_path"], paths["kspace_path"], paths["traj_path"], paths["csm_path"]],
            fingerprint,
            paths["fingerprint_path"],
            args.force,
            env,
        )

    if args.skip_recon:
        print("\n=== 2/3 Reconstruct cMRF coefficient maps ===")
        print("Skip requested with --skip-recon.")
    else:
        run_step(
            "recon",
            "2/3 Reconstruct cMRF coefficient maps",
            [
                sys.executable,
                "scripts/run_recon_cmrf.py",
                "--input-dir",
                str(paths["processed_dir"]),
                "--output-dir",
                str(paths["output_dir"]),
                "--n-iter",
                str(args.n_iter),
                "--lambda-llr",
                str(args.lambda_llr),
                "--step-size",
                str(args.step_size),
                "--device",
                args.device,
                "--gpu-device",
                str(args.gpu_device),
            ],
            recon_config,
            [paths["dict_path"], paths["kspace_path"], paths["traj_path"], paths["csm_path"]],
            [paths["coeff_maps_path"]],
            fingerprint,
            paths["fingerprint_path"],
            args.force,
            env,
        )

    if args.skip_matching:
        print("\n=== 3/3 Template matching ===")
        print("Skip requested with --skip-matching.")
    else:
        run_step(
            "matching",
            "3/3 Template matching",
            [
                sys.executable,
                "scripts/template_matching.py",
                "--dict-path",
                str(paths["dict_path"]),
                "--coeff-maps",
                str(paths["coeff_maps_path"]),
                "--save-path",
                str(paths["quant_png_path"]),
                "--save-npz",
                str(paths["quant_npz_path"]),
                "--device",
                args.device,
                "--gpu-device",
                str(args.gpu_device),
                "--batch-size",
                str(args.matching_batch_size),
                "--t1-vmax",
                str(config.CMRF_T1_VMAX),
                "--t2-vmax",
                str(config.CMRF_T2_VMAX),
            ],
            matching_config,
            [paths["dict_path"], paths["coeff_maps_path"]],
            [paths["quant_png_path"], paths["quant_npz_path"]],
            fingerprint,
            paths["fingerprint_path"],
            args.force,
            env,
        )

    print("\n>>> cMRF pipeline finished.")


if __name__ == "__main__":
    main()

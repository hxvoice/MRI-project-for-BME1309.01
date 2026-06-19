from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.recon.iterative import reconstruct_subspace_llr


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cMRF subspace LLR reconstruction.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.CMRF_PROCESSED_DIR,
        help="Processed cMRF input directory containing k-space, trajectory, CSM, and dictionary files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.CMRF_OUTPUT_DIR,
        help="Directory used for reconstructed coefficient maps and loss history.",
    )
    parser.add_argument("--n-iter", type=int, default=config.CMRF_N_ITER, help="FISTA iteration count.")
    parser.add_argument("--lambda-llr", type=float, default=config.CMRF_LAMBDA_LLR, help="LLR regularization weight.")
    parser.add_argument("--step-size", type=float, default=config.STEP_SIZE, help="Gradient step size.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=config.CMRF_DEVICE, help="Compute backend.")
    parser.add_argument("--gpu-device", type=int, default=config.CMRF_GPU_DEVICE, help="CUDA device id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kspace_path = input_dir / "mrf_kspace_noisy.npy"
    traj_path = input_dir / "traj_full_2d.npy"
    csm_path = input_dir / "csm.npy"
    dict_path = input_dir / "mrf_dictionary_data.npz"

    require_file(kspace_path, "processed k-space")
    require_file(traj_path, "processed trajectory")
    require_file(csm_path, "coil sensitivity map")
    require_file(dict_path, "processed dictionary")

    print(">>> Loading processed cMRF arrays...")
    kspace = np.load(kspace_path)
    traj = np.load(traj_path)
    csm = np.load(csm_path)
    dict_data = np.load(dict_path)
    bases = dict_data["bases"]

    if bases.shape[0] == config.CMRF_SUBSPACE_RANK:
        bases = bases.T

    print(f"    K-space:    {kspace.shape}")
    print(f"    Trajectory: {traj.shape}")
    print(f"    CSM:        {csm.shape}")
    print(f"    Bases:      {bases.shape}")
    print(
        f"\n>>> Running cMRF LLR reconstruction "
        f"(device={args.device}, n_iter={args.n_iter}, lambda_llr={args.lambda_llr})..."
    )

    recon_result = reconstruct_subspace_llr(
        kspace=kspace,
        coord=traj,
        basis=bases,
        sens_maps=csm,
        img_shape=(csm.shape[-2], csm.shape[-1]),
        n_iter=args.n_iter,
        step_size=args.step_size,
        lambda_llr=args.lambda_llr,
        patch_shape=getattr(config, "PATCH_SHAPE", (8, 8)),
        device=args.device,
        device_id=args.gpu_device,
    )

    if isinstance(recon_result, tuple):
        coeff_maps = recon_result[0]
        if len(recon_result) > 1:
            loss_history = recon_result[1]
            if hasattr(loss_history, "get"):
                loss_history = loss_history.get()
            np.save(output_dir / "recon_loss.npy", loss_history)
    else:
        coeff_maps = recon_result

    if hasattr(coeff_maps, "get"):
        coeff_maps = coeff_maps.get()
    elif hasattr(coeff_maps, "cpu"):
        coeff_maps = coeff_maps.cpu().numpy()

    coeff_maps = np.asarray(coeff_maps)
    if coeff_maps.shape[0] == config.CMRF_SUBSPACE_RANK:
        coeff_maps = np.moveaxis(coeff_maps, 0, -1)

    output_path = output_dir / "reconstructed_coeff_maps.npy"
    np.save(output_path, coeff_maps)
    print(f"\n>>> Reconstruction finished. Saved coefficient maps to: {output_path}")


if __name__ == "__main__":
    main()

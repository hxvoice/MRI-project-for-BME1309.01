from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_config as config
from mri_project.recon import estimate_sens_maps_espirit, reconstruct_subspace_llr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D MRF subspace LLR reconstruction.")
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
        "--center-width",
        type=int,
        default=config.CENTER_WIDTH,
        help="Central k-space width for calibration gridding.",
    )
    parser.add_argument("--calib-width", type=int, default=config.CALIB_WIDTH, help="ESPIRiT calibration width.")
    return parser.parse_args()


def load_inputs(
    kspace_path: Path,
    traj_path: Path,
    dictionary_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kspace = np.load(kspace_path)
    traj = np.load(traj_path)
    dictionary_data = np.load(dictionary_path)
    basis = dictionary_data["bases"].T
    return kspace, traj, basis


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "output"

    kspace_path = output_dir / "mrf_kspace_2d_noisy.npy"
    traj_path = processed_dir / "traj_full_2d.npy"
    dictionary_path = processed_dir / "mrf_dictionary_data.npz"
    coeff_maps_path = output_dir / "reconstructed_coeff_maps.npy"

    img_shape = tuple(args.img_shape)
    patch_shape = tuple(args.patch_shape)

    print(">>> 启动 MRF 子空间 LLR 重建流程 <<<")
    print(f"\n[Init] 正在加载 k-space:\n{kspace_path}")
    print(f"[Init] 正在加载轨迹:\n{traj_path}")
    print(f"[Init] 正在加载字典子空间基:\n{dictionary_path}")
    kspace, traj, basis = load_inputs(kspace_path, traj_path, dictionary_path)
    print(f"[Init] k-space 维度: {kspace.shape}")
    print(f"[Init] 轨迹维度: {traj.shape}")
    print(f"[Init] 重建 basis 维度: {basis.shape}")

    print("\n[Sens] 正在估计线圈敏感度图...")
    sens_maps = estimate_sens_maps_espirit(
        kspace,
        traj,
        img_shape,
        center_width=args.center_width,
        calib_width=args.calib_width,
    )
    print(f"[Sens] 敏感度图维度: {sens_maps.shape}")

    print("\n[Recon] 开始 LLR 子空间重建...")
    coeff_maps, losses = reconstruct_subspace_llr(
        kspace,
        basis,
        traj,
        img_shape,
        n_iter=args.n_iter,
        step_size=args.step_size,
        lambda_llr=args.lambda_llr,
        patch_shape=patch_shape,
        sens_maps=sens_maps,
    )
    print(f"[Recon] 重建系数图维度: {coeff_maps.shape}")
    if losses:
        print(f"[Recon] 初始 loss: {losses[0]:.6g}, 最终 loss: {losses[-1]:.6g}")

    coeff_maps_for_matching = np.moveaxis(coeff_maps, 0, -1)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(coeff_maps_path, coeff_maps_for_matching)
    print(f"\n[Save] 已保存模板匹配输入系数图: {coeff_maps_path}")
    print(f"[Save] 保存维度: {coeff_maps_for_matching.shape}")
    print("\n>>> 重建流程执行完毕 <<<")


if __name__ == "__main__":
    main()

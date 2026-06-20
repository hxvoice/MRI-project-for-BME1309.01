from __future__ import annotations

import argparse
import os
from pathlib import Path

# ============================================================
# Windows/Anaconda 下避免 OpenMP 冲突
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from mrpro.data import KData, IData, CsmData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.operators.models import MonoExponentialDecay, InversionRecovery
from mrpro.operators import MagnitudeOp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spin-echo reference T1/T2 maps from ref_t1.h5 and ref_t2.h5 using MRpro."
    )

    parser.add_argument(
        "--scanner-dir",
        type=Path,
        required=True,
        help="Path to scanner folder containing ref_t1.h5 and ref_t2.h5.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/output/reference_maps"),
        help="Output directory for generated reference maps.",
    )

    parser.add_argument(
        "--ref-t1-name",
        type=str,
        default="ref_t1.h5",
        help="Reference T1 h5 file name.",
    )

    parser.add_argument(
        "--ref-t2-name",
        type=str,
        default="ref_t2.h5",
        help="Reference T2 h5 file name.",
    )

    parser.add_argument(
        "--recon-matrix-y",
        type=int,
        default=128,
        help="Reconstruction matrix y size. Official notebook uses 128.",
    )

    parser.add_argument(
        "--recon-matrix-x",
        type=int,
        default=None,
        help="Optional reconstruction matrix x size. If not set, keep MRpro default.",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for dictionary matching. Reconstruction is still handled by MRpro.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening popup window.",
    )

    return parser.parse_args()


def check_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            return torch.device("cpu")

        dev = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return dev

    print("[Device] Using CPU.")
    return torch.device("cpu")


def extract_contrast_times_ms(
    value,
    n_contrasts: int,
    name: str,
) -> torch.Tensor:
    """
    Extract one TE/TI value per reconstructed contrast image.

    Some MRpro versions store img.header.te as acquisition-level values,
    e.g. thousands of repeated entries. But reconstructed image data may only
    contain 5 echoes for T2 or 7 inversion images for T1.

    Returned unit:
        milliseconds
    """
    t = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32).reshape(-1)

    t = t[torch.isfinite(t)]
    t = t[t > 0]

    if t.numel() == 0:
        raise ValueError(f"{name}: no positive finite timing values found.")

    # Unit heuristic:
    # If median is smaller than 10, value is probably in seconds.
    # Otherwise assume already in ms.
    med = float(torch.median(t))
    if med < 10.0:
        t_ms = t * 1000.0
    else:
        t_ms = t

    # Remove tiny floating point noise
    t_ms_round = torch.round(t_ms * 1000.0) / 1000.0

    unique_vals = torch.unique(t_ms_round)
    unique_vals, _ = torch.sort(unique_vals)

    print(f"[Timing] {name}: raw count       = {t.numel()}")
    print(f"[Timing] {name}: unique count    = {unique_vals.numel()}")
    print(f"[Timing] {name}: expected count  = {n_contrasts}")

    if unique_vals.numel() <= 30:
        print(f"[Timing] {name}: unique values ms = {unique_vals.tolist()}")
    else:
        print(
            f"[Timing] {name}: first 10 unique values ms = {unique_vals[:10].tolist()} "
            f"... last 10 = {unique_vals[-10:].tolist()}"
        )

    if unique_vals.numel() == n_contrasts:
        print(f"[Timing] {name}: using unique values.")
        return unique_vals.to(dtype=torch.float32)

    if t_ms_round.numel() == n_contrasts:
        print(f"[Timing] {name}: using raw values.")
        return t_ms_round.to(dtype=torch.float32)

    if unique_vals.numel() > n_contrasts:
        vals = unique_vals[:n_contrasts]
        print(
            f"[Warning] {name}: unique count {unique_vals.numel()} > expected {n_contrasts}. "
            f"Using first {n_contrasts} unique values."
        )
        print(f"[Timing] {name}: used values ms = {vals.tolist()}")
        return vals.to(dtype=torch.float32)

    raise ValueError(
        f"{name}: cannot extract {n_contrasts} timing values.\n"
        f"raw count = {t.numel()}\n"
        f"unique count = {unique_vals.numel()}\n"
        f"unique values = {unique_vals.tolist()}"
    )


def dictionary_matching(
    img_data: torch.Tensor,
    model,
    dictionary_values: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Dictionary matching.

    Expected img_data shape:
        [other, 1, z, y, x]

    After flatten:
        img_flat shape = [z*y*x, other]

    Expected signal_dictionary shape:
        [other, n_dict]
    """
    dictionary_values = dictionary_values.to(dtype=torch.float32, device=device)
    img_data = img_data.to(device)

    signal_dictionary, = model(torch.ones(1, device=device), dictionary_values)
    signal_dictionary = signal_dictionary.to(dtype=torch.complex64)

    n_other = int(img_data.shape[0])
    n_dict = int(dictionary_values.numel())

    print(f"[Matching] img_data shape              = {tuple(img_data.shape)}")
    print(f"[Matching] signal_dictionary raw shape = {tuple(signal_dictionary.shape)}")
    print(f"[Matching] expected other              = {n_other}")
    print(f"[Matching] expected n_dict             = {n_dict}")

    if signal_dictionary.ndim != 2:
        signal_dictionary = signal_dictionary.reshape(signal_dictionary.shape[0], -1)

    # Standardize dictionary to [other, n_dict]
    if signal_dictionary.shape == (n_other, n_dict):
        pass
    elif signal_dictionary.shape == (n_dict, n_other):
        print("[Matching] Transposing signal_dictionary to [other, n_dict].")
        signal_dictionary = signal_dictionary.T
    else:
        raise ValueError(
            "signal_dictionary shape mismatch.\n"
            f"    img_data other dimension = {n_other}\n"
            f"    dictionary_values size   = {n_dict}\n"
            f"    signal_dictionary shape  = {tuple(signal_dictionary.shape)}\n"
            "Expected [other, n_dict] or [n_dict, other]."
        )

    # Normalize dictionary columns
    vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0, keepdim=True)
    vector_norm = torch.clamp(vector_norm, min=1e-12)
    signal_dictionary = signal_dictionary / vector_norm

    if torch.is_complex(img_data):
        signal_dictionary = signal_dictionary.to(img_data.dtype)
    else:
        signal_dictionary = signal_dictionary.real.to(img_data.dtype)

    n_y, n_x = img_data.shape[-2:]

    img_flat = rearrange(
        img_data,
        "other 1 z y x -> (z y x) other",
    )

    print(f"[Matching] img_flat shape              = {tuple(img_flat.shape)}")
    print(f"[Matching] dictionary final shape      = {tuple(signal_dictionary.shape)}")

    dot_product = torch.mm(img_flat, signal_dictionary)
    idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)

    matched_values = dictionary_values[idx_best_match]

    return rearrange(
        matched_values,
        "(y x) -> 1 1 y x",
        y=n_y,
        x=n_x,
    ).detach().cpu()


def multi_image_reco(kdata: KData) -> IData:
    """
    Reconstruct multi-image spin echo data using DirectReconstruction and coil sensitivity maps.

    Same logic as official notebook:
        1. First reconstruction without CSM.
        2. Estimate CSM from the first image.
        3. Reconstruct again with CSM.
    """
    reco = DirectReconstruction(kdata=kdata, csm=None)
    img = reco(kdata)

    first_image = IData(data=img.data[0, None], header=img.header)
    csm_first_image = CsmData.from_idata_inati(first_image)

    reco = DirectReconstruction(kdata=kdata, csm=csm_first_image)
    return reco(kdata)


def dictionary_matching_spin_echo_t2(
    img: IData,
    t2: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    T2 fitting using mono-exponential decay model.

    Important:
        Some MRpro versions store img.header.te as acquisition-level values.
        We reduce them to one TE per reconstructed echo.
    """
    n_echoes = int(img.data.shape[0])

    print(f"[T2] img.data shape = {tuple(img.data.shape)}")
    print(f"[T2] number of reconstructed echoes = {n_echoes}")

    te_ms = extract_contrast_times_ms(img.header.te, n_echoes, name="TE")
    te_ms = te_ms.to(device=device)

    model = MonoExponentialDecay(decay_time=te_ms)

    return dictionary_matching(
        img.data,
        model,
        dictionary_values=t2,
        device=device,
    )


def dictionary_matching_spin_echo_t1(
    img: IData,
    t1: torch.Tensor,
    device: torch.device,
    ti_ms_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    T1 fitting using inversion recovery model.

    For this dataset, the official TI values are:
        [25, 50, 300, 600, 1200, 2400, 4800] ms

    We pass these values directly to the model instead of writing them into
    kdata.header.ti before reconstruction.

    This avoids MRpro version-specific error:
        ValueError: Dimension mismatch. Spatial or coil dimension should be reduced...
    """
    n_ti = int(img.data.shape[0])

    print(f"[T1] img.data shape = {tuple(img.data.shape)}")
    print(f"[T1] number of reconstructed TI images = {n_ti}")

    if ti_ms_override is not None:
        ti_ms = torch.as_tensor(ti_ms_override).detach().cpu().to(dtype=torch.float32).reshape(-1)

        if ti_ms.numel() != n_ti:
            raise ValueError(
                f"Provided TI count does not match reconstructed image count.\n"
                f"    ti_ms_override count = {ti_ms.numel()}\n"
                f"    reconstructed n_ti   = {n_ti}\n"
                f"    ti_ms_override       = {ti_ms.tolist()}"
            )

        print(f"[T1] Using override TI values ms = {ti_ms.tolist()}")

    else:
        ti_ms = extract_contrast_times_ms(img.header.ti, n_ti, name="TI")

    ti_ms = ti_ms.to(device=device)

    model = MagnitudeOp() @ InversionRecovery(ti=ti_ms)

    return dictionary_matching(
        img.data.abs(),
        model,
        dictionary_values=t1,
        device=device,
    )


def reco_ref_scans(
    scanner_dir: Path,
    fname_t1: str,
    fname_t2: str,
    t1: torch.Tensor,
    t2: torch.Tensor,
    recon_matrix_x: int | None,
    recon_matrix_y: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct reference T1 and T2 maps from ref_t1.h5/ref_t2.h5.
    """
    ref_t1_path = scanner_dir / fname_t1
    ref_t2_path = scanner_dir / fname_t2

    if not ref_t1_path.exists():
        raise FileNotFoundError(f"Cannot find reference T1 file: {ref_t1_path}")

    if not ref_t2_path.exists():
        raise FileNotFoundError(f"Cannot find reference T2 file: {ref_t2_path}")

    # ------------------------------------------------------------
    # T2 reference
    # ------------------------------------------------------------
    print(f"\n[Ref T2] Loading: {ref_t2_path}")
    kdata_t2 = KData.from_file(ref_t2_path, KTrajectoryCartesian())

    print(f"[Ref T2] Setting recon_matrix.y = {recon_matrix_y}")
    kdata_t2.header.recon_matrix.y = recon_matrix_y

    if recon_matrix_x is not None:
        print(f"[Ref T2] Setting recon_matrix.x = {recon_matrix_x}")
        kdata_t2.header.recon_matrix.x = recon_matrix_x

    print("[Ref T2] Reconstructing multi-echo spin echo images...")
    img_multi_echo_se = multi_image_reco(kdata_t2)

    print("[Ref T2] Dictionary matching T2 map...")
    t2_map_ref = dictionary_matching_spin_echo_t2(
        img_multi_echo_se,
        t2,
        device=device,
    )[0, 0, ...]

    # ------------------------------------------------------------
    # T1 reference
    # ------------------------------------------------------------
    print(f"\n[Ref T1] Loading: {ref_t1_path}")
    kdata_t1 = KData.from_file(ref_t1_path, KTrajectoryCartesian())

    print(f"[Ref T1] Setting recon_matrix.y = {recon_matrix_y}")
    kdata_t1.header.recon_matrix.y = recon_matrix_y

    if recon_matrix_x is not None:
        print(f"[Ref T1] Setting recon_matrix.x = {recon_matrix_x}")
        kdata_t1.header.recon_matrix.x = recon_matrix_x

    # 注意：
    # 这里不要写：
    # kdata_t1.header.ti = ...
    #
    # 因为你的本地 MRpro 版本会在 DirectReconstruction 阶段因为 header.ti 维度报错。
    # TI 只需要在 dictionary matching 时传给 InversionRecovery 模型即可。
    official_ti_ms = torch.as_tensor(
        [25, 50, 300, 600, 1200, 2400, 4800],
        dtype=torch.float32,
    )

    print(f"[Ref T1] Official TI values ms = {official_ti_ms.tolist()}")

    print("[Ref T1] Reconstructing multi-TI spin echo images...")
    img_multi_ti_se = multi_image_reco(kdata_t1)

    print("[Ref T1] Dictionary matching T1 map...")
    t1_map_ref = dictionary_matching_spin_echo_t1(
        img_multi_ti_se,
        t1,
        device=device,
        ti_ms_override=official_ti_ms,
    )[0, 0, ...]

    return t1_map_ref, t2_map_ref


def save_reference_maps(
    t1_map: torch.Tensor,
    t2_map: torch.Tensor,
    out_dir: Path,
    show: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_np = t1_map.detach().cpu().numpy().astype(np.float32)
    t2_np = t2_map.detach().cpu().numpy().astype(np.float32)

    print("\n[Save] Reference map summary:")
    print(f"    T1 shape: {t1_np.shape}")
    print(f"    T1 min/max: {np.nanmin(t1_np):.6g} / {np.nanmax(t1_np):.6g}")
    print(f"    T2 shape: {t2_np.shape}")
    print(f"    T2 min/max: {np.nanmin(t2_np):.6g} / {np.nanmax(t2_np):.6g}")

    t1_npy = out_dir / "ref_t1_map.npy"
    t2_npy = out_dir / "ref_t2_map.npy"
    npz_path = out_dir / "reference_maps.npz"
    png_path = out_dir / "reference_maps.png"

    np.save(t1_npy, t1_np)
    np.save(t2_npy, t2_np)

    np.savez_compressed(
        npz_path,
        ref_t1_map=t1_np,
        ref_t2_map=t2_np,
    )

    print(f"[Save] Saved T1 map to: {t1_npy}")
    print(f"[Save] Saved T2 map to: {t2_npy}")
    print(f"[Save] Saved npz to:    {npz_path}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axes[0].imshow(t1_np, cmap="magma", vmin=0, vmax=2000)
    axes[0].set_title("Reference T1 Map (ms)")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(t2_np, cmap="viridis", vmin=0, vmax=200)
    axes[1].set_title("Reference T2 Map (ms)")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Spin-Echo Reference Quantitative Maps", fontsize=15)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"[Save] Saved figure to: {png_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    scanner_dir = args.scanner_dir
    out_dir = args.out_dir
    device = check_device(args.device)

    if not scanner_dir.exists():
        raise FileNotFoundError(f"Scanner directory does not exist: {scanner_dir}")

    print(">>> Generating reference T1/T2 maps using MRpro <<<")
    print(f"[Input] Scanner dir: {scanner_dir}")
    print(f"[Output] Out dir:    {out_dir}")
    print(f"[Device] Matching device: {device}")

    # Same dictionary grid as official notebook
    t1 = torch.cat(
        (
            torch.arange(50, 2000 + 10, 10),
            torch.arange(2020, 3000 + 20, 20),
            torch.arange(3050, 5000 + 50, 50),
        )
    ).to(dtype=torch.float32)

    t2 = torch.cat(
        (
            torch.arange(6, 100 + 2, 2),
            torch.arange(105, 200 + 5, 5),
            torch.arange(220, 500 + 20, 20),
        )
    ).to(dtype=torch.float32)

    print(f"[Dict] T1 grid size: {t1.numel()}, range: {float(t1.min())} - {float(t1.max())} ms")
    print(f"[Dict] T2 grid size: {t2.numel()}, range: {float(t2.min())} - {float(t2.max())} ms")

    t1_map_ref, t2_map_ref = reco_ref_scans(
        scanner_dir=scanner_dir,
        fname_t1=args.ref_t1_name,
        fname_t2=args.ref_t2_name,
        t1=t1,
        t2=t2,
        recon_matrix_x=args.recon_matrix_x,
        recon_matrix_y=args.recon_matrix_y,
        device=device,
    )

    save_reference_maps(
        t1_map=t1_map_ref,
        t2_map=t2_map_ref,
        out_dir=out_dir,
        show=not args.no_show,
    )

    print("\n>>> Reference map generation finished <<<")


if __name__ == "__main__":
    main()
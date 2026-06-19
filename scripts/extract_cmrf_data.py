from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as config


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract real cMRF scanner data, build the EPG dictionary, and save processed arrays."
    )
    parser.add_argument("--base-dir", type=Path, default=config.CMRF_DATA_ROOT, help="Root directory of the cMRF dataset.")
    parser.add_argument("--scanner", default=config.CMRF_SCANNER, help="Scanner subdirectory under --base-dir.")
    parser.add_argument("--scan-name", default=config.CMRF_SCAN_NAME, help="Raw cMRF HDF5 filename inside the scanner directory.")
    parser.add_argument("--fa-file", type=Path, default=Path(config.CMRF_FA_FILE), help="Flip-angle text file path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for processed outputs. Defaults to <base-dir>/<scanner>/processed.",
    )
    parser.add_argument(
        "--subspace-rank",
        type=int,
        default=config.CMRF_SUBSPACE_RANK,
        help="Number of dictionary subspace basis vectors to keep.",
    )
    return parser.parse_args()


def generate_subspace_dictionary(full_dict_np: np.ndarray, subspace_rank: int = 5) -> tuple[np.ndarray, np.ndarray]:
    import scipy.linalg

    print(f">>> Running SVD compression, input dictionary shape: {full_dict_np.shape}")
    u_matrix, _, _ = scipy.linalg.svd(full_dict_np, full_matrices=False)
    bases = u_matrix[:, :subspace_rank]
    compressed_dict = bases.conj().T @ full_dict_np
    print(f">>> SVD complete. bases={bases.shape}, compressed_dict={compressed_dict.shape}")
    return bases, compressed_dict


def process_and_intercept_cmrf_data(
    scanner_dir: Path,
    scan_name: str,
    fa: Any,
    t1: Any,
    t2: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from einops import rearrange
    from mrpro.algorithms.reconstruction import DirectReconstruction
    from mrpro.data import KData
    from mrpro.data.CsmData import CsmData
    from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
    from mrpro.operators.models.EPG import (
        DelayBlock,
        EPGSequence,
        FispBlock,
        InversionBlock,
        Parameters,
        T2PrepBlock,
    )

    scan_path = scanner_dir / scan_name
    require_file(scan_path, "raw cMRF scan")
    print(f"--- Processing {scan_path} ---")

    kdata = KData.from_file(scan_path, KTrajectoryIsmrmrd())

    print(">>> Estimating coil sensitivity maps (CSM)...")
    avg_recon = DirectReconstruction(kdata, csm=None)
    avg_img = avg_recon(kdata)
    csm = CsmData.from_idata_walsh(avg_img)
    csm_np = np.squeeze(csm.data.detach().cpu().numpy())

    print(">>> Extracting native k-space and trajectory arrays...")
    k_data_pt = kdata.data.detach().cpu()
    kspace_np = rearrange(k_data_pt.numpy(), "... coils k2 k1 k0 -> coils (... k2 k1) k0")

    ky_pt = kdata.traj.ky.detach().cpu()
    kx_pt = kdata.traj.kx.detach().cpu()
    ky_np = rearrange(ky_pt.numpy(), "... 1 k2 k1 k0 -> (... k2 k1) k0")
    kx_np = rearrange(kx_pt.numpy(), "... 1 k2 k1 k0 -> (... k2 k1) k0")
    traj_np = np.stack((ky_np, kx_np), axis=-1)

    print(f"    K-space shape:    {kspace_np.shape}")
    print(f"    Trajectory shape: {traj_np.shape}")
    print(f"    CSM shape:        {csm_np.shape}")

    print(">>> Building EPG sequence and signal dictionary...")
    t1, t2 = torch.broadcast_tensors(t1[None, :], t2[:, None])
    t1_all = t1.flatten().to(dtype=torch.float32)
    t2_all = t2.flatten().to(dtype=torch.float32)

    valid = t1_all >= t2_all
    t1_valid = t1_all[valid]
    t2_valid = t2_all[valid]
    m0 = torch.ones_like(t1_valid)

    n_rf_pulses_per_block = 47
    acq_t_ms = kdata.header.acq_info.acquisition_time_stamp.flatten().detach().cpu().numpy() * 2.5
    delay_between_blocks = [
        acq_t_ms[n_block * n_rf_pulses_per_block] - acq_t_ms[n_block * n_rf_pulses_per_block - 1]
        for n_block in range(1, 3 * 5)
    ]
    delay_between_blocks.append(delay_between_blocks[-1])

    tr_raw = kdata.header.tr
    tr_val = tr_raw[0] if isinstance(tr_raw, (list, tuple)) else tr_raw
    tr = float(tr_val.item() if hasattr(tr_val, "item") else tr_val) * 1000

    inv_prep_ti = [21, None, None, None, None] * 3
    t2_prep_te = [None, None, 30, 50, 100] * 3
    delay_due_to_prep = [0, 30, 50, 100, 21] * 3
    delay_after_block = [trig_delay - prep_delay for prep_delay, trig_delay in zip(delay_due_to_prep, delay_between_blocks)]

    epg_blocks = []
    for block_index in range(15):
        if inv_prep_ti[block_index] is not None:
            epg_blocks.append(InversionBlock(inversion_time=inv_prep_ti[block_index]))
        if t2_prep_te[block_index] is not None:
            epg_blocks.append(T2PrepBlock(te=t2_prep_te[block_index]))

        start = block_index * n_rf_pulses_per_block
        stop = start + n_rf_pulses_per_block
        epg_blocks.append(FispBlock(flip_angles=fa[start:stop], rf_phases=0.0, te=1.52, tr=tr))

        if delay_after_block[block_index] > 0:
            epg_blocks.append(DelayBlock(delay_time=delay_after_block[block_index]))

    epg_sequence = EPGSequence(epg_blocks)
    params = Parameters(m0=m0, t1=t1_valid, t2=t2_valid)
    _, signals = epg_sequence(params)

    signal_dictionary = torch.stack(signals, dim=0)
    signal_dictionary /= torch.linalg.vector_norm(signal_dictionary, dim=0)

    return (
        signal_dictionary.detach().cpu().numpy(),
        t1_valid.detach().cpu().numpy(),
        t2_valid.detach().cpu().numpy(),
        kspace_np,
        traj_np,
        csm_np,
    )


def main() -> None:
    args = parse_args()

    base_dir = resolve_path(args.base_dir)
    scanner_dir = base_dir / args.scanner
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else scanner_dir / "processed"
    fa_path = resolve_path(args.fa_file) if args.fa_file.is_absolute() else base_dir / args.fa_file

    require_file(fa_path, "flip-angle file")
    require_file(scanner_dir / args.scan_name, "raw cMRF scan")

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    t1_grid = torch.as_tensor(config.cmrf_t1_grid(), dtype=torch.float32)
    t2_grid = torch.as_tensor(config.cmrf_t2_grid(), dtype=torch.float32)

    with fa_path.open("r", encoding="utf-8") as file:
        fa = torch.as_tensor([float(line) for line in file.readlines()], dtype=torch.float32) / 180 * torch.pi

    full_dict_np, t1_valid_np, t2_valid_np, kspace_np, traj_np, csm_np = process_and_intercept_cmrf_data(
        scanner_dir=scanner_dir,
        scan_name=args.scan_name,
        fa=fa,
        t1=t1_grid,
        t2=t2_grid,
    )

    bases, compressed_dict = generate_subspace_dictionary(full_dict_np, subspace_rank=args.subspace_rank)

    print(f"\n>>> Saving processed cMRF data to: {output_dir}")
    np.savez(
        output_dir / "mrf_dictionary_data.npz",
        signal_dict=full_dict_np,
        compressed_dict=compressed_dict,
        bases=bases,
        t1=t1_valid_np,
        t2=t2_valid_np,
    )
    np.save(output_dir / "mrf_kspace_noisy.npy", kspace_np)
    np.save(output_dir / "traj_full_2d.npy", traj_np)
    np.save(output_dir / "csm.npy", csm_np)
    print(">>> cMRF extraction finished.")


if __name__ == "__main__":
    main()

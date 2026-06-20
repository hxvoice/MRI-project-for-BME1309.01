"""Central default settings for the lightweight MRF pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

IMG_SHAPE = (220, 220)
N_COILS = 8
N_TR = 500
SUBSPACE_RANK = 5

N_ITER = 30
LAMBDA_LLR = 1e-3
STEP_SIZE = 1e-3
PATCH_SHAPE = (8, 8)
CENTER_WIDTH = 24
CALIB_WIDTH = 24

SPIRAL_FOV = 96e-3
SPIRAL_RES = 1e-3
SPIRAL_READOUT_TIME = 7e-3
SPIRAL_UNDERSAMPLE_CENTER = 12
SPIRAL_UNDERSAMPLE_EDGE = 24
SPIRAL_ADC_RATE = 1e6
TINY_GOLDEN_ANGLE = 23.63

EPG_NUM_STATES = 200
EPG_TR = 12.0
EPG_TE = 0.7
EPG_TI = 20.0

FA_MIN = 10.0
FA_MAX = 90.0
FA_NUM_ANCHORS = 15
FA_RANDOM_SEED = 42

T1_GRID_RANGES = (
    (20, 3001, 20),
    (3200, 5001, 200),
)
T2_GRID_RANGES = (
    (10, 201, 2),
    (220, 1001, 20),
    (1050, 2001, 50),
    (2100, 4001, 100),
)

PHANTOM_BRAIN_RADIUS = 0.8
PHANTOM_WM_RADIUS = 0.5
PHANTOM_CSF_RADIUS = 0.85
PHANTOM_TISSUES = {
    "wm": {"t1": 850.0, "t2": 60.0, "pd": 0.8},
    "gm": {"t1": 1350.0, "t2": 80.0, "pd": 0.9},
    "csf": {"t1": 4000.0, "t2": 2000.0, "pd": 1.0},
}

ESPIRIT_THRESH = 0.02
ESPIRIT_KERNEL_WIDTH = 6
ESPIRIT_CROP = 0.95
ESPIRIT_MAX_ITER = 100
ESPIRIT_SHOW_PBAR = False

NOISE_LEVEL = 0.25
RANDOM_SEED = 42


# Real cMRF scanner-comparison pipeline defaults.
# Set CMRF_DATA_ROOT in the environment on a server to avoid editing this file.
CMRF_DATA_ROOT = Path(
    os.environ.get(
        "CMRF_DATA_ROOT",
        PROJECT_ROOT / "data" / "cmrf_scanner_comparison",
    )
)
CMRF_SCANNER = "scanner1"
CMRF_SCAN_NAME = "cMRF.h5"
CMRF_FA_FILE = "cMRF_fa_705rep.txt"

CMRF_SCANNER_DIR = CMRF_DATA_ROOT / CMRF_SCANNER
CMRF_PROCESSED_DIR = CMRF_SCANNER_DIR / "processed"
CMRF_DICT_PATH = CMRF_PROCESSED_DIR / "mrf_dictionary_data.npz"
CMRF_KSPACE_PATH = CMRF_PROCESSED_DIR / "mrf_kspace_noisy.npy"
CMRF_TRAJ_PATH = CMRF_PROCESSED_DIR / "traj_full_2d.npy"
CMRF_CSM_PATH = CMRF_PROCESSED_DIR / "csm.npy"
CMRF_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
CMRF_COEFF_MAPS_PATH = CMRF_OUTPUT_DIR / "reconstructed_coeff_maps.npy"
CMRF_RECON_LOSS_PATH = CMRF_OUTPUT_DIR / "recon_loss.npy"
CMRF_QUANT_PNG_PATH = CMRF_OUTPUT_DIR / "quantitative_maps.png"
CMRF_QUANT_NPZ_PATH = CMRF_OUTPUT_DIR / "quantitative_maps.npz"
CMRF_FINGERPRINT_PATH = CMRF_OUTPUT_DIR / "cmrf_pipeline_fingerprint.json"

CMRF_SUBSPACE_RANK = 5
CMRF_N_ITER = 40
CMRF_LAMBDA_LLR = 0.003
CMRF_DEVICE = "cuda"
CMRF_GPU_DEVICE = 0
CMRF_MATCHING_BATCH_SIZE = 1024
CMRF_T1_GRID_RANGES = (
    (50, 2001, 10),
    (2020, 3001, 20),
    (3050, 5001, 50),
)
CMRF_T2_GRID_RANGES = (
    (6, 101, 2),
    (105, 201, 5),
    (220, 501, 20),
)
CMRF_T1_VMAX = 2000.0
CMRF_T2_VMAX = 150.0


def build_grid(ranges: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    """Build a 1D grid from inclusive-style numpy arange ranges."""

    return np.concatenate([np.arange(start, stop, step) for start, stop, step in ranges])


def t1_grid() -> np.ndarray:
    return build_grid(T1_GRID_RANGES)


def t2_grid() -> np.ndarray:
    return build_grid(T2_GRID_RANGES)


def cmrf_t1_grid() -> np.ndarray:
    return build_grid(CMRF_T1_GRID_RANGES)


def cmrf_t2_grid() -> np.ndarray:
    return build_grid(CMRF_T2_GRID_RANGES)


# Rerun guide after changing parameters:
#
# Reconstruction-only parameters:
#   N_ITER, LAMBDA_LLR, STEP_SIZE, PATCH_SHAPE, CENTER_WIDTH, CALIB_WIDTH,
#   ESPIRIT_THRESH, ESPIRIT_KERNEL_WIDTH, ESPIRIT_CROP, ESPIRIT_MAX_ITER
#   Rerun:
#     python scripts/run_recon.py
#     python scripts/template_matching.py
#     python validate_results.py
#
# Forward-simulation parameters:
#   N_COILS, NOISE_LEVEL, RANDOM_SEED
#   Rerun:
#     python scripts/run_forward_sim.py
#     python scripts/run_recon.py
#     python scripts/template_matching.py
#     python validate_results.py
#
# Phantom/grid parameters:
#   IMG_SHAPE, PHANTOM_* values
#   Rerun:
#     python scripts/prepare_phantom.py
#     python scripts/run_forward_sim.py
#     python scripts/run_recon.py
#     python scripts/template_matching.py
#     python validate_results.py
#
# Trajectory parameters:
#   SPIRAL_FOV, SPIRAL_RES, SPIRAL_READOUT_TIME,
#   SPIRAL_UNDERSAMPLE_CENTER, SPIRAL_UNDERSAMPLE_EDGE,
#   SPIRAL_ADC_RATE, TINY_GOLDEN_ANGLE
#   Rerun:
#     python scripts/generate_traj.py
#     python scripts/run_forward_sim.py
#     python scripts/run_recon.py
#     python scripts/template_matching.py
#     python validate_results.py
#
# Dictionary/subspace parameters:
#   N_TR, SUBSPACE_RANK, EPG_*, FA_*, T1_GRID_RANGES, T2_GRID_RANGES
#   Rerun:
#     python scripts/generate_traj.py      # needed when N_TR changes
#     python scripts/build_dictionary.py
#     python scripts/run_forward_sim.py
#     python scripts/run_recon.py
#     python scripts/template_matching.py
#     python validate_results.py
#
# Note: python main.py records per-step config fingerprints and reruns affected
# steps when these parameters change.

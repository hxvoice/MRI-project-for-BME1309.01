"""Central default settings for the lightweight MRF pipeline."""

IMG_SHAPE = (96, 96)
N_COILS = 4
N_TR = 500
SUBSPACE_RANK = 5

N_ITER = 2
LAMBDA_LLR = 1e-4
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
NOISE_LEVEL = 0.25
RANDOM_SEED = 42


# Rerun guide after changing parameters:
#
# Reconstruction-only parameters:
#   N_ITER, LAMBDA_LLR, STEP_SIZE, PATCH_SHAPE, CENTER_WIDTH, CALIB_WIDTH
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
#   IMG_SHAPE
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
#   N_TR, SUBSPACE_RANK, EPG_NUM_STATES
#   Rerun:
#     python scripts/generate_traj.py      # needed when N_TR changes
#     python scripts/build_dictionary.py
#     python scripts/run_forward_sim.py
#     python scripts/run_recon.py
#     python scripts/template_matching.py
#     python validate_results.py
#
# Note: python main.py skips existing output files when their shapes still look
# valid, so use the scripts above when you want to force results to update after
# parameter tuning.

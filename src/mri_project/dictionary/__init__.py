"""MRF dictionary generation utilities."""

from .epg import (
    MRFEPGSimulator,
    build_and_compress_dictionary,
    build_signal_dictionary,
    compress_dictionary,
    generate_mrf_fa_train_spline,
)

MRF_EPG_Simulator = MRFEPGSimulator

__all__ = [
    "MRFEPGSimulator",
    "MRF_EPG_Simulator",
    "build_signal_dictionary",
    "compress_dictionary",
    "build_and_compress_dictionary",
    "generate_mrf_fa_train_spline",
]

"""
Utility functions for music source separation.

This module provides various utility functions including audio processing,
evaluation metrics, and helper functions.
"""

from .audio import (
    load_audio,
    save_audio,
    compute_stft,
    compute_istft,
    apply_mask,
    normalize_audio,
    peak_normalize,
    mix_sources,
    compute_energy,
    apply_window_compensation,
    trim_silence,
)

from .metrics import (
    sdr,
    si_sdr,
    bss_eval_sources_torch,
    evaluate_separation,
    compute_aggregate_metrics,
    SeparationMetrics,
)

__all__ = [
    # Audio utilities
    "load_audio",
    "save_audio",
    "compute_stft",
    "compute_istft",
    "apply_mask",
    "normalize_audio",
    "peak_normalize",
    "mix_sources",
    "compute_energy",
    "apply_window_compensation",
    "trim_silence",
    # Metrics
    "sdr",
    "si_sdr",
    "bss_eval_sources_torch",
    "evaluate_separation",
    "compute_aggregate_metrics",
    "SeparationMetrics",
]

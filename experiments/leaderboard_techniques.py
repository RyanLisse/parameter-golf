"""Proven leaderboard techniques from the parameter-golf competition.

These techniques come from the #1 entry (1.1748 val_bpb) and other top entries.
Each is individually toggleable via environment variables.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


# =============================================================================
# a) Spectral/Overtone Embedding Init
# =============================================================================
# From #1 entry. Shapes embedding spectrum to power-law decay using SVD.
# Env var: SPECTRAL_EMBED_INIT=1, SPECTRAL_EXPONENT=-0.5


def apply_spectral_embed_init_np(
    weight: np.ndarray, exponent: float = -0.5
) -> np.ndarray:
    """SVD-based power-law spectrum shaping for embeddings.

    U, S, V = svd(weight)
    target_S = S[0] * (1/arange(1, len(S)+1))^exponent
    weight = (U * target_S) @ V

    Args:
        weight: 2D numpy array (vocab_size, dim).
        exponent: Power-law exponent. -0.5 gives 1/sqrt(k) decay.

    Returns:
        Reshaped weight with power-law singular value spectrum.
    """
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    ranks = np.arange(1, len(s) + 1, dtype=np.float64)
    target_s = float(s[0]) * np.power(ranks, exponent).astype(np.float32)
    return (u * target_s[None, :]) @ vt


# =============================================================================
# b) Phase-Transition Residual Mix Init
# =============================================================================
# From #1 entry. Sigmoid-scheduled init for residual mixing per layer.
# Env var: PHASE_TRANSITION_INIT=1


def compute_phase_transition_schedule(
    num_layers: int, steepness: float = 3.0
) -> List[Tuple[float, float]]:
    """Compute sigmoid-scheduled residual mix values per layer.

    Early layers trust x0 more (mix[0] high, mix[1] low).
    Late layers trust the accumulated residual more.

    Args:
        num_layers: Total number of transformer blocks.
        steepness: Controls sigmoid transition sharpness.

    Returns:
        List of (resid_mix_0, resid_mix_1) tuples per layer.
    """
    schedule = []
    for i in range(num_layers):
        # phase goes from ~0 (early) to ~1 (late)
        phase = 1.0 / (1.0 + math.exp(-steepness * (i / max(num_layers - 1, 1) - 0.5)))
        schedule.append((phase, 1.0 - phase))
    return schedule


# =============================================================================
# c) Sliding Window Evaluation
# =============================================================================
# From #1 and #2 entries (-0.034 BPB). Evaluate with overlapping windows.
# Env vars: EVAL_STRIDE=64, SLIDING_WINDOW_EVAL=1


def compute_sliding_window_ranges(
    total_tokens: int, seq_len: int, stride: int
) -> List[Tuple[int, int]]:
    """Compute sliding window ranges for evaluation.

    Each window is seq_len tokens. Windows overlap by (seq_len - stride).
    This gives each token near-full context for scoring.

    Args:
        total_tokens: Total number of tokens in validation set.
        seq_len: Model sequence length.
        stride: Step size between windows. Smaller = more overlap = better scores.

    Returns:
        List of (start, end) index tuples.
    """
    if stride <= 0:
        stride = seq_len
    ranges = []
    start = 0
    while start + seq_len <= total_tokens:
        ranges.append((start, start + seq_len))
        start += stride
    # If we haven't covered the last position, add a final window
    if ranges and ranges[-1][1] < total_tokens:
        final_start = total_tokens - seq_len
        if final_start >= 0:
            ranges.append((final_start, total_tokens))
    return ranges


# =============================================================================
# d) Muon Weight Decay
# =============================================================================
# From #1 entry. Decoupled weight decay applied post-optimizer.
# Env var: MUON_WD=0.02


def apply_muon_weight_decay_np(
    matrix_params: List[np.ndarray], lr: float, wd: float = 0.02
) -> None:
    """Decoupled weight decay for Muon optimizer (in-place).

    Args:
        matrix_params: List of 2D parameter arrays to decay.
        lr: Current learning rate.
        wd: Weight decay coefficient.
    """
    if wd <= 0.0:
        return
    factor = 1.0 - wd * lr
    for i in range(len(matrix_params)):
        matrix_params[i] *= factor


# =============================================================================
# e) FP16 Embedding Passthrough
# =============================================================================
# From #1 and warmdown entries (-0.004 BPB). Skip int8 quantization for embeddings.
# Env var: EMBED_FP16_PASSTHROUGH=1

_EMBED_PATTERNS = ("tok_emb", "embed", "embedding")


def should_keep_fp16(name: str) -> bool:
    """Return True if this tensor should stay fp16 during quantization.

    Embeddings benefit from higher precision because they are the
    first and last layer (tied), so quantization noise gets amplified.

    Args:
        name: Parameter name (e.g. 'tok_emb.weight').

    Returns:
        True if this is an embedding tensor.
    """
    name_lower = name.lower()
    return any(pat in name_lower for pat in _EMBED_PATTERNS)


# =============================================================================
# f) Extended Warmdown Schedule
# =============================================================================
# From warmdown entry (-0.009 BPB). Set WARMDOWN_ITERS >= total training steps.
# This is already supported by existing warmdown code. The key insight is that
# when warmdown_iters >= iterations, the LR decays linearly from a partial peak
# from the very first step, producing tighter weight distributions for better
# quantization. No new code needed -- just the right preset value.


# =============================================================================
# g) Championship Preset Configs
# =============================================================================

CHAMPIONSHIP_CONFIG: dict[str, str] = {
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "10",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "2",
    "TRAIN_SEQ_LEN": "1024",
    "TRAIN_BATCH_TOKENS": "524288",
    "VAL_BATCH_SIZE": "524288",
    "ITERATIONS": "20000",
    "MAX_WALLCLOCK_SECONDS": "600",
    "WARMDOWN_ITERS": "20000",  # Extended warmdown >= iterations
    "WARMUP_STEPS": "20",
    "MATRIX_LR": "0.06",
    "TIED_EMBED_LR": "0.07",
    "SCALAR_LR": "0.06",
    "MUON_MOMENTUM": "0.95",
    "MUON_BACKEND_STEPS": "5",
    "GRAD_CLIP_NORM": "1.0",
    "QK_GAIN_INIT": "1.5",
    "LOGIT_SOFTCAP": "30.0",
    "TIED_EMBED_INIT_STD": "0.005",
    # Leaderboard technique toggles
    "SPECTRAL_EMBED_INIT": "1",
    "SPECTRAL_EXPONENT": "-0.5",
    "PHASE_TRANSITION_INIT": "1",
    "MUON_WD": "0.02",
    "EMBED_FP16_PASSTHROUGH": "1",
    "EVAL_STRIDE": "64",
}

CHAMPIONSHIP_LONGCTX_CONFIG: dict[str, str] = {
    **CHAMPIONSHIP_CONFIG,
    "TRAIN_SEQ_LEN": "4096",
    "TRAIN_BATCH_TOKENS": "131072",  # Reduced to fit longer sequences
    "VAL_BATCH_SIZE": "131072",
    "MATRIX_LR": "0.04",  # Lower LR for longer context stability
    "TIED_EMBED_LR": "0.05",
    "SCALAR_LR": "0.04",
    "ITERATIONS": "25000",
    "WARMDOWN_ITERS": "25000",
    "MUON_MOMENTUM": "0.99",
    "GRAD_CLIP_NORM": "0.3",
}

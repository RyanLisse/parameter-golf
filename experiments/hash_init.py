"""Hash-Based Initialization & Deterministic Dropout (RJC-38).

Uses cryptographic hash properties (SipHash) for deterministic weight initialization
and dropout masks without GPU RNG overhead.

Environment variables:
- HASH_INIT_ENABLED=1: Enable hash-based initialization
- HASH_DROPOUT_ENABLED=1: Enable hash-based dropout
- HASH_DROPOUT_PCT=10: Dropout percentage for hash-based dropout
- CHAOTIC_LR_ENABLED=0: Enable chaotic learning rate schedule (logistic map)
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np


def _siphash_like_hash(layer: int, row: int, col: int, seed: int) -> int:
    """
    Generate a hash value using SHA-256 (cascade of SipHash-inspired properties).
    Takes layer, row, col, and seed as inputs and returns a deterministic integer.
    """
    key = struct.pack("<QIIII", seed, layer, row, col, seed)
    h = hashlib.sha256(key).digest()
    return struct.unpack("<Q", h[:8])[0]


def siphash_init(
    layer_idx: int,
    rows: int,
    cols: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate weight matrix using hash-based initialization.

    Args:
        layer_idx: Layer index in the model
        rows: Number of rows in weight matrix
        cols: Number of columns in weight matrix
        seed: Random seed for reproducibility

    Returns:
        Weight matrix (rows, cols) with zero mean and unit variance.
    """
    weights = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            h = _siphash_like_hash(layer_idx, i, j, seed)
            # Map hash to normal distribution via Box-Muller-like approach
            weights[i, j] = float(h) / 2**64 * 2.0 - 1.0

    # Normalize to zero mean, unit variance
    mean = np.mean(weights)
    std = np.std(weights)
    if std > 0:
        weights = (weights - mean) / std

    return weights


def hash_dropout_mask(
    layer_idx: int,
    step: int,
    seq_len: int,
    dropout_pct: int,
) -> np.ndarray:
    """
    Generate deterministic dropout mask using hash function.

    Args:
        layer_idx: Layer index in the model
        step: Training step
        seq_len: Sequence length
        dropout_pct: Dropout percentage (0-100)

    Returns:
        Binary mask (seq_len,) where 1 = keep, 0 = drop.
    """
    mask = np.ones(seq_len, dtype=np.uint8)

    for pos in range(seq_len):
        h = _siphash_like_hash(layer_idx, step, pos, dropout_pct)
        # Map hash to percentage: if (hash % 100) < dropout_pct, then drop (0)
        if (h % 100) < dropout_pct:
            mask[pos] = 0

    return mask


def chaotic_lr_schedule(step: int, x0: float = 0.5) -> float:
    """
    Compute learning rate using chaotic logistic map.

    The logistic map: x_{n+1} = 3.99 * x_n * (1 - x_n)
    Produces pseudo-random oscillations in (0, 1) for LR scheduling.

    Args:
        step: Training step
        x0: Initial condition (default 0.5)

    Returns:
        Learning rate value in (0, 1).
    """
    x = x0
    for _ in range(step + 1):
        x = 3.99 * x * (1.0 - x)
    return x


def apply_hash_init(model: dict[str, Any], seed: int = 42) -> None:
    """
    Replace standard weight initialization with hash-based initialization.

    Modifies the model dict in-place by replacing weight matrices with
    hash-initialized versions for all 'weight' keys.

    Args:
        model: Dictionary of model weights (layer_name -> weight_array)
        seed: Random seed for initialization
    """
    layer_idx = 0
    for name, param in model.items():
        if "weight" in name.lower() and isinstance(param, np.ndarray):
            if param.ndim >= 2:
                rows, cols = param.shape[0], param.shape[1]
                model[name] = siphash_init(layer_idx, rows, cols, seed=seed)
                layer_idx += 1

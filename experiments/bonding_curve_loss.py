"""Bonding Curve Token-Weighted Loss Function (RJC-35).

This module implements DeFi-inspired bonding curve weighting for token loss.
Rare tokens receive higher loss weight during training, controlled via:
  BONDING_LOSS_ENABLED: Enable/disable (1/0)
  BONDING_ALPHA_START: Initial alpha (default 0.0, no weighting)
  BONDING_ALPHA_END: Final alpha (default 0.5, strong rare-token emphasis)

Weight formula: w(token) = 1 / (freq^alpha) where alpha schedules from start to end.
"""

import glob
import os
from pathlib import Path

import numpy as np


class TokenFrequencyWeighter:
    """Weight tokens by frequency using bonding curve schedule.

    Common tokens (high frequency) get weight < 1.0.
    Rare tokens (low frequency) get weight >= 1.0.
    Alpha parameter schedules from alpha_start to alpha_end over training.
    """

    def __init__(self, vocab_size, alpha_start=0.0, alpha_end=0.5):
        """Initialize TokenFrequencyWeighter.

        Args:
            vocab_size: Size of vocabulary
            alpha_start: Initial alpha exponent (0.0 = no weighting)
            alpha_end: Final alpha exponent (0.5 = strong weighting)
        """
        self.vocab_size = vocab_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.frequencies = None

    def build_frequency_table(self, data_path):
        """Build token frequency table from training data.

        Args:
            data_path: Path to directory containing tokenized *.bin files
        """
        self.frequencies = np.zeros(self.vocab_size, dtype=np.float32)

        # Find all .bin files in directory
        files = glob.glob(os.path.join(data_path, "*.bin"))

        for file_path in files:
            try:
                # Load tokenized data (uint16)
                data = np.fromfile(file_path, dtype=np.uint16)

                # Count frequencies
                unique, counts = np.unique(data, return_counts=True)
                for token_id, count in zip(unique, counts):
                    if token_id < self.vocab_size:
                        self.frequencies[token_id] += count
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")

        # Avoid zero frequencies
        self.frequencies = np.maximum(self.frequencies, 1.0)

    def get_weights(self, step, total_steps):
        """Get token weights at current training step.

        Alpha schedules linearly from alpha_start to alpha_end.

        Args:
            step: Current training step
            total_steps: Total training steps

        Returns:
            Weight array of shape (vocab_size,) with w(token) = 1 / (freq^alpha)
        """
        if self.frequencies is None:
            # No frequency table: uniform weights
            return np.ones(self.vocab_size, dtype=np.float32)

        # Linear schedule for alpha
        progress = step / max(total_steps, 1)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress

        # Compute weights: w = 1 / freq^alpha
        # Add small epsilon to avoid division by zero
        weights = 1.0 / np.power(self.frequencies + 1e-8, alpha)

        # Normalize so mean weight is 1.0 (for stable loss magnitude)
        mean_weight = np.mean(weights)
        weights = weights / (mean_weight + 1e-8)

        return weights.astype(np.float32)


class EntropyWeighter:
    """Weight tokens by prediction entropy (hard vs. easy examples).

    High-entropy predictions (uncertain) get higher weight.
    Low-entropy predictions (confident) get lower weight.
    """

    def __init__(self, vocab_size, entropy_scale=0.1):
        """Initialize EntropyWeighter.

        Args:
            vocab_size: Size of vocabulary
            entropy_scale: Scale factor for entropy weighting (larger = stronger effect)
        """
        self.vocab_size = vocab_size
        self.entropy_scale = entropy_scale


def weighted_cross_entropy_torch(logits, targets, weights):
    """Compute weighted cross-entropy loss with PyTorch.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        targets: Tensor of shape (batch, seq_len) with token IDs
        weights: Tensor of shape (vocab_size,) with per-token weights

    Returns:
        Scalar loss (mean over batch and sequence)
    """
    import torch
    import torch.nn.functional as F

    # Flatten batch and sequence dimensions
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute cross-entropy per token
    log_probs = F.log_softmax(logits_flat, dim=-1)
    ce_loss = -log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

    # Apply token weights
    token_weights = weights[targets_flat]
    weighted_loss = ce_loss * token_weights

    # Average over all tokens
    return weighted_loss.mean()


def weighted_cross_entropy_mlx(logits, targets, weights):
    """Compute weighted cross-entropy loss with MLX.

    Args:
        logits: Array of shape (batch, seq_len, vocab_size)
        targets: Array of shape (batch, seq_len) with token IDs
        weights: Array of shape (vocab_size,) with per-token weights

    Returns:
        Scalar loss (mean over batch and sequence)
    """
    import mlx.core as mx

    # Flatten batch and sequence dimensions
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute log softmax
    log_probs = logits_flat - mx.log(mx.sum(mx.exp(logits_flat), axis=1, keepdims=True) + 1e-8)

    # Cross-entropy loss for each token
    batch_indices = mx.arange(logits_flat.shape[0])
    ce_loss = -log_probs[batch_indices, targets_flat]

    # Apply token weights
    token_weights = weights[targets_flat]
    weighted_loss = ce_loss * token_weights

    # Average over all tokens
    return mx.mean(weighted_loss)


def entropy_weights_torch(logits, entropy_scale=0.1):
    """Compute per-example entropy weights with PyTorch.

    Hard examples (high entropy) get higher weight.
    Easy examples (low entropy) get lower weight.

    Args:
        logits: Tensor of shape (batch, vocab_size) or (batch, seq_len, vocab_size)
        entropy_scale: Scale factor for weighting

    Returns:
        Weights of shape (batch,) or (batch, seq_len) with values in [entropy_scale, 1.0]
    """
    import torch
    import torch.nn.functional as F

    # Handle both 2D and 3D logits
    original_shape = logits.shape[:-1]
    logits_2d = logits.reshape(-1, logits.shape[-1])

    # Compute softmax probabilities
    probs = F.softmax(logits_2d, dim=-1)

    # Compute entropy
    entropy = -(probs * (F.log_softmax(logits_2d, dim=-1) + 1e-8)).sum(dim=-1)

    # Normalize entropy to [0, 1]
    max_entropy = float(torch.log(torch.tensor(logits_2d.shape[-1], dtype=logits.dtype)))
    normalized_entropy = entropy / max_entropy

    # Weight: high entropy -> high weight, low entropy -> low weight
    weights = entropy_scale + (1.0 - entropy_scale) * normalized_entropy

    return weights.reshape(original_shape)


def entropy_weights_mlx(logits, entropy_scale=0.1):
    """Compute per-example entropy weights with MLX.

    Hard examples (high entropy) get higher weight.
    Easy examples (low entropy) get lower weight.

    Args:
        logits: Array of shape (batch, vocab_size) or (batch, seq_len, vocab_size)
        entropy_scale: Scale factor for weighting

    Returns:
        Weights of shape (batch,) or (batch, seq_len) with values in [entropy_scale, 1.0]
    """
    import mlx.core as mx

    # Handle both 2D and 3D logits
    original_shape = logits.shape[:-1]
    logits_2d = logits.reshape(-1, logits.shape[-1])

    # Compute softmax probabilities
    log_probs = logits_2d - mx.log(mx.sum(mx.exp(logits_2d), axis=1, keepdims=True) + 1e-8)
    probs = mx.exp(log_probs)

    # Compute entropy
    entropy = -mx.sum(probs * log_probs, axis=-1)

    # Normalize entropy to [0, 1]
    vocab_size = logits_2d.shape[-1]
    max_entropy = float(mx.log(mx.array(vocab_size)))
    normalized_entropy = entropy / max_entropy

    # Weight: high entropy -> high weight, low entropy -> low weight
    weights = entropy_scale + (1.0 - entropy_scale) * normalized_entropy

    return weights.reshape(original_shape)

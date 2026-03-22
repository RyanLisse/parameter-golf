"""Gas Optimization — Kernel Fusion & Dead Code Elimination (RJC-30).

Maximizes GPU FLOPs per iteration by eliminating waste, enabling more iterations
within the 10-minute training budget.

Environment variables:
    GAS_OPT_ENABLED: Enable gas optimization (0/1)
    GAS_SKIP_DROPOUT_WARMDOWN: Skip dropout during warmdown phase (0/1)
    GAS_FUSE_OPS: Enable operation fusion hints (0/1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class GasOptConfig:
    """Configuration for gas optimization read from environment variables."""

    enabled: bool = False
    skip_dropout_warmdown: bool = False
    fuse_ops: bool = False

    def __init__(self):
        """Initialize config from environment variables."""
        self.enabled = os.environ.get("GAS_OPT_ENABLED", "0") == "1"
        self.skip_dropout_warmdown = (
            os.environ.get("GAS_SKIP_DROPOUT_WARMDOWN", "0") == "1"
        )
        self.fuse_ops = os.environ.get("GAS_FUSE_OPS", "0") == "1"


def should_skip_dropout(step: int, total_steps: int, warmdown_ratio: float = 0.2) -> bool:
    """Determine if dropout should be skipped during warmdown phase.

    During the final warmdown phase of training, dropout can be skipped to reduce
    computation without significantly impacting final model quality.

    Args:
        step: Current training step
        total_steps: Total number of training steps
        warmdown_ratio: Fraction of training considered "warmdown" (default 0.2 = 20%)

    Returns:
        True if dropout should be skipped, False otherwise
    """
    warmdown_start = total_steps * (1 - warmdown_ratio)
    return step >= warmdown_start


def fused_relu_squared(x: torch.Tensor) -> torch.Tensor:
    """Compute ReLU(x)² as a single fused operation.

    This provides a hint to torch.compile to fuse ReLU and square operations,
    reducing memory traffic and kernel launch overhead.

    Args:
        x: Input tensor

    Returns:
        ReLU(x)² computed element-wise
    """
    # Clamp negative values to zero, then square
    # This is equivalent to: relu(x) ** 2, but hints at fusion
    return torch.clamp(x, min=0.0) ** 2


def adaptive_grad_clip(
    grad_norms_history: list[float], threshold_multiplier: float = 2.0
) -> bool:
    """Determine if gradient clipping can be skipped based on norm history.

    When gradient norms are consistently below a threshold, clipping can be skipped
    to save computation. Uses exponential moving average to detect stable gradients.

    Args:
        grad_norms_history: Recent history of gradient norms
        threshold_multiplier: Multiplier for EMA threshold (default 2.0)

    Returns:
        True if clipping can be skipped (norms consistently low), False otherwise
    """
    if len(grad_norms_history) < 3:
        # Not enough history - be conservative
        return False

    # Compute exponential moving average of gradient norms
    alpha = 0.1  # Smoothing factor
    ema = grad_norms_history[0]
    for norm in grad_norms_history[1:]:
        ema = alpha * norm + (1 - alpha) * ema

    # Check if recent norms are consistently below threshold
    threshold = ema * threshold_multiplier
    recent_norms = grad_norms_history[-3:]  # Last 3 norms

    all_below_threshold = all(norm < threshold for norm in recent_norms)
    return all_below_threshold


def optimize_batch_size(
    gpu_memory_gb: float, model_params: int, seq_len: int
) -> int:
    """Calculate optimal batch size to maximize GPU memory utilization.

    Estimates memory usage for model parameters, activations, and gradients,
    then computes the largest batch size that fits in GPU memory.

    Args:
        gpu_memory_gb: Available GPU memory in gigabytes
        model_params: Number of model parameters
        seq_len: Sequence length

    Returns:
        Optimal batch size (at least 1)
    """
    # Convert to bytes (1 GB = 1e9 bytes)
    gpu_memory_bytes = gpu_memory_gb * 1e9

    # Reserve some memory for framework overhead (20%)
    available_memory = gpu_memory_bytes * 0.8

    # Memory per parameter:
    # - Model weights: 4 bytes (fp32) or 2 bytes (fp16)
    # - Gradients: same as weights
    # - Optimizer state (Adam): 8 bytes per param (momentum + variance)
    bytes_per_param = 4 + 4 + 8  # weights + grads + optimizer state

    model_memory = model_params * bytes_per_param

    # Memory per sample:
    # - Activations: roughly 2x model size per layer for forward+backward
    # - Assume average of 4 bytes per activation element
    # Rough estimate: seq_len * model_dim * num_layers * 4 bytes
    # Activations scale with sequence length
    base_activation_memory = (model_params // 10) * 4
    activation_per_sample = base_activation_memory * (seq_len / 1024)  # Scale with seq_len

    # Calculate batch size
    memory_for_batch = available_memory - model_memory
    if memory_for_batch <= 0:
        return 1  # Minimum batch size

    batch_size = int(memory_for_batch / activation_per_sample)

    # Ensure at least batch size of 1
    return max(1, batch_size)


def measure_iterations_per_second(
    train_fn: callable, num_warmup: int = 5, num_measure: int = 20
) -> float:
    """Measure training iterations per second for performance comparison.

    Args:
        train_fn: Function that performs one training iteration
        num_warmup: Number of warmup iterations (not measured)
        num_measure: Number of iterations to measure

    Returns:
        Average iterations per second
    """
    import time

    # Warmup iterations (not measured)
    for _ in range(num_warmup):
        train_fn()

    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure iterations
    start_time = time.perf_counter()

    for _ in range(num_measure):
        train_fn()

    # Synchronize again
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    iterations_per_sec = num_measure / elapsed

    return iterations_per_sec

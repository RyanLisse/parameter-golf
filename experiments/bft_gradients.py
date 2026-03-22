"""BFT Trimmed-Mean Gradient Aggregation (RJC-36).

Byzantine Fault Tolerant gradient aggregation using trimmed mean across GPUs.
Replaces standard AllReduce with robust aggregation that tolerates outliers.

Environment variables:
    BFT_ENABLED: Enable BFT aggregation (0/1)
    BFT_TRIM_COUNT: Number of extreme values to trim from each end (default: 1)
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


class BFTAggregator:
    """Byzantine Fault Tolerant gradient aggregator using trimmed mean.

    For N GPUs, trims the top and bottom k gradient values, then averages
    the remaining values. This provides robustness against Byzantine failures
    or hardware issues causing gradient outliers.

    Example:
        8 GPUs with trim_count=1:
        - Gather all 8 gradient values per parameter
        - Sort values
        - Remove highest and lowest values
        - Average remaining 6 values
    """

    def __init__(self, world_size: int, trim_count: int = 1):
        """Initialize BFT aggregator.

        Args:
            world_size: Total number of GPUs in distributed training
            trim_count: Number of values to trim from each end (top and bottom)
        """
        self.world_size = world_size
        self.trim_count = trim_count

    def should_use_fallback(self) -> bool:
        """Check if we should fallback to standard AllReduce.

        Returns:
            True if configuration requires fallback (not enough GPUs for trimming)
        """
        # Need at least 2*trim_count + 1 GPUs to have values left after trimming
        min_required = 2 * self.trim_count + 1
        return self.world_size <= min_required

    def requires_distributed(self) -> bool:
        """Check if BFT requires distributed training.

        Returns:
            True (BFT only works in distributed mode)
        """
        return True

    def fallback_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fallback reduction when distributed is not available.

        Args:
            tensor: Input tensor

        Returns:
            Input tensor unchanged (no reduction needed in non-distributed mode)
        """
        return tensor

    def _compute_trimmed_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Compute trimmed mean of 1D tensor.

        Args:
            values: 1D tensor of values to aggregate

        Returns:
            Trimmed mean (scalar)
        """
        # Sort values
        sorted_vals, _ = torch.sort(values)

        # Trim top and bottom
        trimmed = sorted_vals[self.trim_count : -self.trim_count]

        # Return mean of remaining values
        return trimmed.mean()

    def _compute_trimmed_mean_multidim(self, stacked: torch.Tensor) -> torch.Tensor:
        """Compute trimmed mean across first dimension of multi-dimensional tensor.

        Args:
            stacked: Tensor with shape [world_size, ...] where first dim is GPU index

        Returns:
            Trimmed mean with shape [...] (first dimension reduced)
        """
        # Flatten all dims except first (GPU dimension)
        original_shape = stacked.shape[1:]
        flat = stacked.flatten(start_dim=1)  # [world_size, num_elements]

        # Sort along GPU dimension
        sorted_vals, _ = torch.sort(flat, dim=0)

        # Trim top and bottom along GPU dimension
        trimmed = sorted_vals[self.trim_count : -self.trim_count, :]

        # Mean along GPU dimension
        result_flat = trimmed.mean(dim=0)

        # Reshape back to original shape
        return result_flat.reshape(original_shape)

    def trimmed_mean_reduce(
        self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None
    ) -> torch.Tensor:
        """Perform trimmed-mean reduction across all GPUs.

        Args:
            tensor: Local gradient tensor
            group: Process group for distributed communication (None = default)

        Returns:
            Aggregated tensor using trimmed mean
        """
        if not dist.is_initialized():
            # Not in distributed mode - return unchanged
            return self.fallback_reduce(tensor)

        if self.should_use_fallback():
            # Not enough GPUs for trimming - use standard AllReduce
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=group)
            return tensor

        # Gather tensors from all GPUs
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=group)

        # Stack into single tensor: [world_size, *tensor.shape]
        stacked = torch.stack(gathered)

        # Compute trimmed mean
        result = self._compute_trimmed_mean_multidim(stacked)

        # Copy result back to input tensor
        tensor.copy_(result)
        return tensor


class QuorumEarlyStopping:
    """Quorum-based early stopping using loss consensus across GPUs.

    Stops training when a quorum of GPUs agree that loss has plateaued,
    providing robustness against individual GPU variance.
    """

    def __init__(
        self,
        world_size: int,
        quorum_size: int = 6,
        patience: int = 5,
        min_delta: float = 0.001,
    ):
        """Initialize quorum early stopping.

        Args:
            world_size: Total number of GPUs
            quorum_size: Number of GPUs that must agree to trigger stopping
            patience: Number of steps to wait before stopping
            min_delta: Minimum improvement to reset patience counter
        """
        self.world_size = world_size
        self.quorum_size = quorum_size
        self.patience = patience
        self.min_delta = min_delta

        # Track per-GPU state
        self.best_losses = [float("inf")] * world_size
        self.plateau_counters = [0] * world_size

    def should_stop(self, losses: torch.Tensor) -> bool:
        """Check if training should stop based on quorum consensus.

        Args:
            losses: Tensor of current losses from all GPUs [world_size]

        Returns:
            True if quorum agrees training should stop
        """
        losses_list = losses.tolist()

        # Update state for each GPU
        for i, loss in enumerate(losses_list):
            if loss < self.best_losses[i] - self.min_delta:
                # Improvement - reset counter
                self.best_losses[i] = loss
                self.plateau_counters[i] = 0
            else:
                # No improvement - increment counter
                self.plateau_counters[i] += 1

        # Count how many GPUs have reached patience threshold
        gpus_ready_to_stop = sum(
            1 for counter in self.plateau_counters if counter >= self.patience
        )

        # Stop if quorum agrees
        return gpus_ready_to_stop >= self.quorum_size

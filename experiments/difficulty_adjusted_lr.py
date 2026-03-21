"""Difficulty-Adjusted Dynamic Curriculum Learning (RJC-33).

This module implements Bitcoin difficulty retargeting-inspired learning rate adjustment.
As training progresses and loss reduction rate changes, the learning rate adapts to maintain
a target loss reduction velocity. Controlled via environment variables:
  DIFFICULTY_LR_ENABLED: Enable/disable (1/0)
  DIFFICULTY_TARGET_RATE: Target per-batch loss reduction rate (default 0.001)
  DIFFICULTY_EMA_ALPHA: Exponential moving average smoothing (default 0.1)
  DIFFICULTY_LR_UP: Factor to increase LR when learning fast (default 1.05)
  DIFFICULTY_LR_DOWN: Factor to decrease LR when learning slow (default 0.95)
"""

import math


class DifficultyAdjuster:
    """Dynamically adjusts learning rate based on loss reduction trajectory.

    Tracks the rate at which loss decreases per batch and adjusts learning rate
    to maintain target loss reduction velocity (like Bitcoin difficulty retargeting).
    """

    def __init__(self, target_rate=0.001, ema_alpha=0.1, lr_up_factor=1.05, lr_down_factor=0.95):
        """Initialize DifficultyAdjuster.

        Args:
            target_rate: Target per-batch loss reduction rate (e.g., 0.001 = 0.1% per batch)
            ema_alpha: EMA smoothing factor for loss and loss_reduction_rate (0 < alpha <= 1)
            lr_up_factor: Multiplier when learning faster than target (e.g., 1.05 = +5%)
            lr_down_factor: Multiplier when learning slower than target (e.g., 0.95 = -5%)
        """
        self.target_rate = target_rate
        self.ema_alpha = ema_alpha
        self.lr_up_factor = lr_up_factor
        self.lr_down_factor = lr_down_factor

        # Track EMA of loss and loss reduction rate
        self.loss_ema = None
        self.loss_reduction_rate_ema = 0.0

    def update(self, loss_value):
        """Update loss trajectory with new batch loss.

        Args:
            loss_value: Scalar loss from current batch.
        """
        if self.loss_ema is None:
            # First update: initialize EMA
            self.loss_ema = loss_value
            return

        # Calculate loss reduction rate for this batch
        prev_loss = self.loss_ema
        reduction = max(0.0, (prev_loss - loss_value) / (prev_loss + 1e-8))

        # Update loss reduction rate EMA
        self.loss_reduction_rate_ema = (
            self.ema_alpha * reduction + (1 - self.ema_alpha) * self.loss_reduction_rate_ema
        )

        # Update loss EMA
        self.loss_ema = self.ema_alpha * loss_value + (1 - self.ema_alpha) * prev_loss

    def get_lr_multiplier(self):
        """Get learning rate multiplier based on current loss reduction rate.

        Returns:
            Multiplier to apply to learning rate:
              > 1.0 if learning faster than target (increase LR)
              < 1.0 if learning slower than target (decrease LR)
              ≈ 1.0 if at target rate
        """
        if self.loss_reduction_rate_ema >= self.target_rate:
            # Learning fast: increase LR
            return self.lr_up_factor
        else:
            # Learning slow: decrease LR
            return self.lr_down_factor

    def get_batch_size_multiplier(self):
        """Get batch size multiplier for curriculum adjustment.

        Larger batches when loss drops fast (confident phase),
        smaller batches for exploration when struggling.

        Returns:
            Multiplier to apply to batch size (clamped to [0.5, 2.0])
        """
        ratio = self.loss_reduction_rate_ema / (self.target_rate + 1e-8)
        # Linear scaling: ratio of 2x target -> batch size 2x, ratio of 0.5x -> batch size 0.5x
        multiplier = max(0.5, min(2.0, ratio))
        return multiplier


class HalvingSchedule:
    """Discrete learning rate schedule: halving at 25%, 50%, 75% of training.

    Alternative to cosine/linear warmdown. Used with DifficultyAdjuster
    for coarse-grained schedule + fine-grained adaptive adjustment.
    """

    def __init__(self):
        """Initialize HalvingSchedule."""
        pass

    def get_lr(self, step, total_steps):
        """Get learning rate multiplier at given step.

        Schedule:
        - [0%, 25%): multiplier = 1.0
        - [25%, 50%): multiplier = 0.5
        - [50%, 75%): multiplier = 0.25
        - [75%, 100%): multiplier = 0.125

        Args:
            step: Current training step (0-indexed)
            total_steps: Total training steps

        Returns:
            Learning rate multiplier
        """
        progress = step / max(total_steps, 1)

        if progress < 0.25:
            return 1.0
        elif progress < 0.5:
            return 0.5
        elif progress < 0.75:
            return 0.25
        else:
            return 0.125

#!/usr/bin/env python3
"""Progressive Sequence Length Curriculum (RJC-52 / EXP-19).

Start training at short sequence lengths (e.g., 512), progressively increase to longer
sequences (e.g., 4096). Gets more iterations early (shorter sequences = faster steps),
more context late (longer sequences = better quality).

Combines ideas from:
- The 4096-seq entry that showed -0.023 BPB improvement
- Curriculum learning: easy-to-hard task progression

Key insight: At seq_len=512 → ~43ms/step → ~14,000 steps in 10 min
             At seq_len=4096 → ~71ms/step → ~8,400 steps in 10 min
             Progressive: start fast, end with quality → best of both worlds

Environment Variables:
    PROGRESSIVE_SEQ_ENABLED: 1 to enable, 0 to disable (default: 0)
    SEQ_START: Starting sequence length, must be multiple of 64 (default: 512)
    SEQ_END: Ending sequence length, must be multiple of 64 (default: 4096)
    SEQ_SCHEDULE: Schedule type - linear, cosine, step, exponential (default: linear)
    SEQ_WARMUP_FRAC: Fraction of training to keep at start length (default: 0.0)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Literal


def _round_to_multiple(value: int, multiple: int) -> int:
    """Round value to nearest multiple."""
    return ((value + multiple // 2) // multiple) * multiple


def _nearest_power_of_2(value: int) -> int:
    """Find nearest power of 2."""
    if value <= 0:
        return 1
    power = round(math.log2(value))
    return 2 ** power


@dataclass
class ProgressiveConfig:
    """Configuration for progressive sequence length curriculum.

    Attributes:
        enabled: Whether progressive scheduling is enabled
        seq_start: Starting sequence length (must be multiple of 64)
        seq_end: Ending sequence length (must be multiple of 64)
        schedule: Type of schedule (linear, cosine, step, exponential)
        warmup_frac: Fraction of training to keep at start length
    """
    enabled: bool
    seq_start: int
    seq_end: int
    schedule: Literal["linear", "cosine", "step", "exponential"]
    warmup_frac: float

    @classmethod
    def from_env(cls) -> ProgressiveConfig:
        """Create config from environment variables.

        Returns:
            ProgressiveConfig instance

        Raises:
            ValueError: If validation fails
        """
        enabled = os.environ.get("PROGRESSIVE_SEQ_ENABLED", "0") == "1"
        seq_start = int(os.environ.get("SEQ_START", "512"))
        seq_end = int(os.environ.get("SEQ_END", "4096"))
        schedule = os.environ.get("SEQ_SCHEDULE", "linear")
        warmup_frac = float(os.environ.get("SEQ_WARMUP_FRAC", "0.0"))

        if enabled:
            # Validate seq_start and seq_end
            if seq_start % 64 != 0:
                raise ValueError(f"SEQ_START must be multiple of 64, got {seq_start}")
            if seq_end % 64 != 0:
                raise ValueError(f"SEQ_END must be multiple of 64, got {seq_end}")
            if seq_start > seq_end:
                raise ValueError(f"SEQ_START ({seq_start}) must be <= SEQ_END ({seq_end})")

            # Validate schedule
            if schedule not in ["linear", "cosine", "step", "exponential"]:
                raise ValueError(f"SEQ_SCHEDULE must be linear/cosine/step/exponential, got {schedule}")

            # Validate warmup_frac
            if not 0.0 <= warmup_frac < 1.0:
                raise ValueError(f"SEQ_WARMUP_FRAC must be in [0, 1), got {warmup_frac}")

        return cls(
            enabled=enabled,
            seq_start=seq_start,
            seq_end=seq_end,
            schedule=schedule,  # type: ignore
            warmup_frac=warmup_frac,
        )

    def __post_init__(self):
        """Additional validation after init."""
        if self.enabled:
            if self.seq_start > self.seq_end:
                raise ValueError(f"seq_start ({self.seq_start}) must be <= seq_end ({self.seq_end})")
            if self.seq_start % 64 != 0 or self.seq_end % 64 != 0:
                raise ValueError("seq_start and seq_end must be multiples of 64")


class ProgressiveSeqSchedule:
    """Progressive sequence length scheduler.

    Manages the sequence length schedule during training, progressively increasing
    from seq_start to seq_end according to the chosen schedule type.

    Args:
        start: Starting sequence length (must be multiple of 64)
        end: Ending sequence length (must be multiple of 64)
        schedule: Type of schedule ("linear", "cosine", "step", "exponential")
        warmup_frac: Fraction of training to keep at start length (default: 0.0)
    """

    def __init__(
        self,
        start: int = 512,
        end: int = 4096,
        schedule: Literal["linear", "cosine", "step", "exponential"] = "linear",
        warmup_frac: float = 0.0,
    ):
        if start % 64 != 0 or end % 64 != 0:
            raise ValueError("start and end must be multiples of 64")
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        if not 0.0 <= warmup_frac < 1.0:
            raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}")

        self.start = start
        self.end = end
        self.schedule = schedule
        self.warmup_frac = warmup_frac

    @classmethod
    def from_env(cls) -> ProgressiveSeqSchedule:
        """Create scheduler from environment variables."""
        config = ProgressiveConfig.from_env()
        if not config.enabled:
            # Return a no-op scheduler that always returns a fixed length
            # Read from TRAIN_SEQ_LEN if available
            fixed_len = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
            return cls(start=fixed_len, end=fixed_len, schedule="linear")

        return cls(
            start=config.seq_start,
            end=config.seq_end,
            schedule=config.schedule,
            warmup_frac=config.warmup_frac,
        )

    def get_seq_len(self, step: int, total_steps: int) -> int:
        """Get sequence length for current step.

        Args:
            step: Current training step (0-indexed)
            total_steps: Total number of training steps

        Returns:
            Sequence length for this step (multiple of 64)
        """
        if total_steps <= 0:
            return self.start

        # Handle warmup phase
        warmup_steps = int(self.warmup_frac * total_steps)
        if step < warmup_steps:
            return self.start

        # Normalize progress to [0, 1] after warmup
        effective_step = step - warmup_steps
        effective_total = total_steps - warmup_steps
        if effective_total <= 0:
            return self.start

        progress = min(1.0, effective_step / effective_total)

        # Calculate raw sequence length based on schedule
        if self.schedule == "linear":
            raw_seq_len = self.start + (self.end - self.start) * progress

        elif self.schedule == "cosine":
            # Cosine annealing: slow start, fast middle, slow end
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2
            raw_seq_len = self.start + (self.end - self.start) * cosine_progress

        elif self.schedule == "step":
            # Discrete jumps at 33% and 66%
            if progress < 0.333:
                raw_seq_len = self.start
            elif progress < 0.667:
                # Middle value
                raw_seq_len = self.start + (self.end - self.start) / 2
            else:
                raw_seq_len = self.end

        elif self.schedule == "exponential":
            # Exponential growth: slow start, rapid acceleration
            # Use exponential interpolation: start * (end/start)^progress
            if self.start > 0:
                ratio = self.end / self.start
                raw_seq_len = self.start * (ratio ** progress)
            else:
                raw_seq_len = self.end * progress

        else:
            # Fallback to linear
            raw_seq_len = self.start + (self.end - self.start) * progress

        # Round to nearest multiple of 64
        seq_len = _round_to_multiple(int(raw_seq_len), 64)

        # Clamp to valid range
        return max(self.start, min(self.end, seq_len))


class AdaptiveBatchSchedule:
    """Adaptive batch size scheduler.

    Maintains roughly constant total tokens per step by adjusting batch size
    as sequence length changes. When seq_len is shorter, we can fit more sequences
    per batch; when seq_len is longer, we fit fewer sequences.

    Args:
        target_tokens: Target total tokens per batch (default: 524288)
        min_batch_size: Minimum batch size to enforce (default: 8)
        round_to_pow2: Whether to round batch sizes to powers of 2 (default: True)
    """

    def __init__(
        self,
        target_tokens: int = 524288,
        min_batch_size: int = 8,
        round_to_pow2: bool = True,
    ):
        self.target_tokens = target_tokens
        self.min_batch_size = min_batch_size
        self.round_to_pow2 = round_to_pow2

    def get_batch_size(self, seq_len: int) -> int:
        """Get batch size for given sequence length.

        Args:
            seq_len: Current sequence length

        Returns:
            Batch size (number of sequences per batch)
        """
        if seq_len <= 0:
            return self.min_batch_size

        # Calculate ideal batch size
        ideal_batch_size = self.target_tokens // seq_len

        # Enforce minimum
        batch_size = max(self.min_batch_size, ideal_batch_size)

        # Optionally round to nearest power of 2
        if self.round_to_pow2:
            batch_size = _nearest_power_of_2(batch_size)

        return batch_size


class CurriculumDataLoader:
    """Data loader wrapper for progressive sequence length curriculum.

    Wraps existing data loader to produce variable-length sequences according
    to the progressive schedule.

    Args:
        base_loader: Base data loader to wrap
        seq_schedule: Sequence length scheduler
        batch_schedule: Batch size scheduler
    """

    def __init__(
        self,
        base_loader: Any,
        seq_schedule: ProgressiveSeqSchedule,
        batch_schedule: AdaptiveBatchSchedule,
    ):
        self.base_loader = base_loader
        self.seq_schedule = seq_schedule
        self.batch_schedule = batch_schedule

    def next_batch(
        self,
        step: int,
        total_steps: int,
    ) -> tuple[Any, Any]:
        """Get next batch with dynamic sequence length.

        Args:
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Tuple of (input_batch, target_batch)
        """
        # Get current sequence length from schedule
        seq_len = self.seq_schedule.get_seq_len(step, total_steps)

        # Get adaptive batch size
        batch_size = self.batch_schedule.get_batch_size(seq_len)

        # Delegate to base loader
        # Most training scripts will need to handle seq_len dynamically
        # This is a simple wrapper that passes through
        x, y = self.base_loader.next_batch()

        # In a real implementation, we would slice/pad x and y to seq_len
        # For now, we just return what the base loader gives us
        # The training script will need to handle the dynamic seq_len

        return x, y


def estimate_throughput(
    seq_lens: list[int],
    step_times: list[float],
    total_time: float = 600.0,
    schedule: str = "linear",
) -> float:
    """Estimate total iterations achievable with given sequence length schedule.

    Args:
        seq_lens: List of sequence lengths to benchmark
        step_times: Corresponding step times in seconds
        total_time: Total training time in seconds (default: 600 = 10 minutes)
        schedule: Schedule type to simulate (default: "linear")

    Returns:
        Estimated total iterations achievable
    """
    if not seq_lens or not step_times or len(seq_lens) != len(step_times):
        return 0.0

    # Simple linear interpolation estimate
    # In a real implementation, this would simulate the full schedule
    # For now, we just return a rough estimate based on average

    # Create a simple interpolation function
    def interpolate_time(seq_len: int) -> float:
        """Interpolate step time for given seq_len."""
        if seq_len <= seq_lens[0]:
            return step_times[0]
        if seq_len >= seq_lens[-1]:
            return step_times[-1]

        # Find bracketing points
        for i in range(len(seq_lens) - 1):
            if seq_lens[i] <= seq_len <= seq_lens[i + 1]:
                # Linear interpolation
                frac = (seq_len - seq_lens[i]) / (seq_lens[i + 1] - seq_lens[i])
                return step_times[i] + frac * (step_times[i + 1] - step_times[i])

        return step_times[-1]

    # Simulate the schedule with 100 steps
    total_steps = 100
    scheduler = ProgressiveSeqSchedule(
        start=seq_lens[0],
        end=seq_lens[-1],
        schedule=schedule,  # type: ignore
    )

    total_simulated_time = 0.0
    for step in range(total_steps):
        seq_len = scheduler.get_seq_len(step, total_steps)
        step_time = interpolate_time(seq_len)
        total_simulated_time += step_time

    # Average time per step in the simulation
    avg_time_per_step = total_simulated_time / total_steps

    # Estimate total iterations in total_time
    estimated_iterations = total_time / avg_time_per_step

    return estimated_iterations


# Example usage when imported:
if __name__ == "__main__":
    # Example: Create a progressive schedule
    schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="linear")

    print("Progressive Sequence Length Schedule (Linear)")
    print("=" * 60)

    total_steps = 10000
    checkpoints = [0, 1000, 2500, 5000, 7500, 9999]

    for step in checkpoints:
        seq_len = schedule.get_seq_len(step, total_steps)
        progress = (step / total_steps) * 100
        print(f"Step {step:5d} ({progress:5.1f}%): seq_len = {seq_len:4d}")

    print("\nBatch Size Adaptation")
    print("=" * 60)

    batch_schedule = AdaptiveBatchSchedule(target_tokens=524288)
    for seq_len in [512, 1024, 2048, 4096]:
        batch_size = batch_schedule.get_batch_size(seq_len)
        total_tokens = batch_size * seq_len
        print(f"seq_len={seq_len:4d} → batch_size={batch_size:4d} → total_tokens={total_tokens:7d}")

    print("\nThroughput Estimation")
    print("=" * 60)

    # Example timings from the issue description
    seq_lens = [512, 4096]
    step_times = [0.043, 0.071]  # seconds

    for sched in ["linear", "cosine", "step", "exponential"]:
        iters = estimate_throughput(seq_lens, step_times, total_time=600, schedule=sched)
        print(f"{sched:12s}: ~{iters:6.0f} iterations in 10 minutes")

#!/usr/bin/env python3
"""Tests for Progressive Sequence Length Curriculum (RJC-52 / EXP-19).

Tests the progressive sequence length scheduling system that starts training
at short sequences (e.g., 512) and progressively increases to longer sequences
(e.g., 4096) to maximize early iterations while achieving quality late in training.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

# Import will fail until we implement - that's expected for TDD RED phase
try:
    from experiments.progressive_seq_len import (
        AdaptiveBatchSchedule,
        CurriculumDataLoader,
        ProgressiveConfig,
        ProgressiveSeqSchedule,
        estimate_throughput,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipIf(not IMPORTS_AVAILABLE, "progressive_seq_len module not yet implemented")
class ProgressiveSeqScheduleTests(unittest.TestCase):
    """Test the ProgressiveSeqSchedule class for sequence length scheduling."""

    def test_linear_schedule_interpolates_correctly(self):
        """Linear schedule should interpolate from start to end."""
        schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="linear")

        # At step 0, should be at start
        self.assertEqual(schedule.get_seq_len(0, 1000), 512)

        # At halfway, should be midpoint
        mid_seq = schedule.get_seq_len(500, 1000)
        self.assertGreater(mid_seq, 512)
        self.assertLess(mid_seq, 4096)
        # Should be close to 2304 (midpoint between 512 and 4096)
        self.assertAlmostEqual(mid_seq, 2304, delta=64)

        # At end, should be at end
        self.assertEqual(schedule.get_seq_len(999, 1000), 4096)

    def test_cosine_schedule_follows_expected_curve(self):
        """Cosine schedule should follow cosine curve (slow start, fast middle, slow end)."""
        schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="cosine")

        # At start
        self.assertEqual(schedule.get_seq_len(0, 1000), 512)

        # At 25%, should be less than linear midpoint (cosine is slower early)
        quarter_seq = schedule.get_seq_len(250, 1000)
        linear_quarter = 512 + (4096 - 512) * 0.25
        self.assertLess(quarter_seq, linear_quarter)

        # At 50%, should be close to midpoint
        mid_seq = schedule.get_seq_len(500, 1000)
        self.assertAlmostEqual(mid_seq, 2304, delta=128)

        # At end
        self.assertEqual(schedule.get_seq_len(999, 1000), 4096)

    def test_step_schedule_has_discrete_jumps(self):
        """Step schedule should have discrete jumps at 33% and 66%."""
        schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="step")

        # First third: should stay at start
        self.assertEqual(schedule.get_seq_len(0, 1000), 512)
        self.assertEqual(schedule.get_seq_len(330, 1000), 512)

        # Second third: should jump to middle
        mid_val = schedule.get_seq_len(500, 1000)
        self.assertGreater(mid_val, 512)
        self.assertLess(mid_val, 4096)

        # Last third: should jump to end
        self.assertEqual(schedule.get_seq_len(670, 1000), 4096)
        self.assertEqual(schedule.get_seq_len(999, 1000), 4096)

    def test_exponential_schedule_grows_exponentially(self):
        """Exponential schedule should grow exponentially from start to end."""
        schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="exponential")

        # At start
        self.assertEqual(schedule.get_seq_len(0, 1000), 512)

        # Exponential grows slowly at first, faster later
        quarter_seq = schedule.get_seq_len(250, 1000)
        linear_quarter = 512 + (4096 - 512) * 0.25
        # Exponential should be slower in early stages
        self.assertLess(quarter_seq, linear_quarter)

        # At halfway, exponential is still catching up to linear
        mid_seq = schedule.get_seq_len(500, 1000)
        linear_mid = 512 + (4096 - 512) * 0.5
        # Should be close to or just below linear midpoint
        self.assertLessEqual(mid_seq, linear_mid + 128)

        # At end
        self.assertEqual(schedule.get_seq_len(999, 1000), 4096)

    def test_seq_lengths_always_multiples_of_64(self):
        """All sequence lengths must be multiples of 64 for GPU efficiency."""
        schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="linear")

        for step in [0, 100, 250, 500, 750, 999]:
            seq_len = schedule.get_seq_len(step, 1000)
            self.assertEqual(seq_len % 64, 0,
                           f"seq_len {seq_len} at step {step} not multiple of 64")

    def test_warmup_delays_schedule_start(self):
        """Warmup fraction should delay the start of sequence length increase."""
        schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="linear", warmup_frac=0.1)

        # During warmup (first 10%), should stay at start
        self.assertEqual(schedule.get_seq_len(0, 1000), 512)
        self.assertEqual(schedule.get_seq_len(50, 1000), 512)
        self.assertEqual(schedule.get_seq_len(99, 1000), 512)

        # After warmup, should start increasing
        post_warmup = schedule.get_seq_len(200, 1000)
        self.assertGreater(post_warmup, 512)


@unittest.skipIf(not IMPORTS_AVAILABLE, "progressive_seq_len module not yet implemented")
class AdaptiveBatchScheduleTests(unittest.TestCase):
    """Test the AdaptiveBatchSchedule class for maintaining constant batch tokens."""

    def test_batch_tokens_stay_roughly_constant(self):
        """As seq_len changes, batch size should adjust to keep tokens/step constant."""
        schedule = AdaptiveBatchSchedule(target_tokens=524288)

        # At seq_len=512, should fit 1024 sequences (524288 / 512 = 1024)
        batch_size_512 = schedule.get_batch_size(512)
        self.assertEqual(batch_size_512, 1024)

        # At seq_len=1024, should fit 512 sequences
        batch_size_1024 = schedule.get_batch_size(1024)
        self.assertEqual(batch_size_1024, 512)

        # At seq_len=4096, should fit 128 sequences
        batch_size_4096 = schedule.get_batch_size(4096)
        self.assertEqual(batch_size_4096, 128)

        # Total tokens should be constant
        self.assertEqual(batch_size_512 * 512, 524288)
        self.assertEqual(batch_size_1024 * 1024, 524288)
        self.assertEqual(batch_size_4096 * 4096, 524288)

    def test_batch_size_rounds_down_to_power_of_2(self):
        """Batch sizes should be rounded to powers of 2 for efficiency."""
        schedule = AdaptiveBatchSchedule(target_tokens=500000)

        # At seq_len=768 (not power of 2), should still get power-of-2 batch
        batch_size = schedule.get_batch_size(768)
        # 500000 / 768 ≈ 651, should round to 512 (nearest power of 2)
        self.assertIn(batch_size, [256, 512, 1024])

    def test_minimum_batch_size_enforced(self):
        """Batch size should never go below minimum (e.g., 8 or 16)."""
        schedule = AdaptiveBatchSchedule(target_tokens=524288, min_batch_size=16)

        # Even with very long sequences, should respect minimum
        batch_size = schedule.get_batch_size(65536)  # Extremely long
        self.assertGreaterEqual(batch_size, 16)


@unittest.skipIf(not IMPORTS_AVAILABLE, "progressive_seq_len module not yet implemented")
class CurriculumDataLoaderTests(unittest.TestCase):
    """Test the CurriculumDataLoader wrapper for dynamic sequence lengths."""

    def test_produces_correct_sequence_length(self):
        """Data loader should produce batches with the scheduled sequence length."""
        # Mock base loader
        base_loader = MagicMock()
        base_loader.next_batch.return_value = (
            [[1, 2, 3, 4] * 256],  # Dummy data
            [[2, 3, 4, 5] * 256]
        )

        seq_schedule = ProgressiveSeqSchedule(start=512, end=4096, schedule="linear")
        batch_schedule = AdaptiveBatchSchedule(target_tokens=524288)
        loader = CurriculumDataLoader(base_loader, seq_schedule, batch_schedule)

        # At step 0, should request 512-length sequences
        x, y = loader.next_batch(step=0, total_steps=1000)
        # Verify base loader was called with correct seq_len
        # (This is a mock test - real implementation would slice data)
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)

    def test_handles_transition_smoothly(self):
        """Should handle sequence length transitions without data loss."""
        base_loader = MagicMock()
        # Configure mock to return proper tuple
        base_loader.next_batch.return_value = (
            [[1, 2, 3, 4] * 256],  # Dummy x data
            [[2, 3, 4, 5] * 256]   # Dummy y data
        )

        seq_schedule = ProgressiveSeqSchedule(start=512, end=1024, schedule="step")
        batch_schedule = AdaptiveBatchSchedule(target_tokens=524288)
        loader = CurriculumDataLoader(base_loader, seq_schedule, batch_schedule)

        # Request batches before and after transition point
        # Should not crash or lose data
        for step in range(330, 340):
            x, y = loader.next_batch(step=step, total_steps=1000)
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)


@unittest.skipIf(not IMPORTS_AVAILABLE, "progressive_seq_len module not yet implemented")
class ThroughputEstimationTests(unittest.TestCase):
    """Test throughput estimation for optimal schedule prediction."""

    def test_estimates_total_iterations(self):
        """Should estimate total achievable iterations given seq_len timings."""
        # Example: 512-length takes 43ms, 4096-length takes 71ms
        seq_lens = [512, 1024, 2048, 4096]
        step_times = [0.043, 0.050, 0.060, 0.071]  # seconds per step
        total_time = 600  # 10 minutes

        iterations = estimate_throughput(seq_lens, step_times, total_time)

        # Should be a reasonable estimate
        self.assertGreater(iterations, 8000)  # Minimum from all-4096
        self.assertLess(iterations, 14000)    # Maximum from all-512

    def test_predicts_optimal_schedule(self):
        """Should predict which schedule gives most iterations."""
        seq_lens = [512, 4096]
        step_times = [0.043, 0.071]

        # Linear schedule should balance early speed with late quality
        linear_iters = estimate_throughput(
            seq_lens, step_times, total_time=600, schedule="linear"
        )

        # All-512 should give most iterations but poor quality
        # All-4096 should give fewest iterations but best quality
        # Linear should be in between
        self.assertIsNotNone(linear_iters)


@unittest.skipIf(not IMPORTS_AVAILABLE, "progressive_seq_len module not yet implemented")
class ProgressiveConfigTests(unittest.TestCase):
    """Test the ProgressiveConfig class for environment variable integration."""

    def test_from_env_reads_variables(self):
        """Should read all progressive seq config from environment."""
        os.environ["PROGRESSIVE_SEQ_ENABLED"] = "1"
        os.environ["SEQ_START"] = "512"
        os.environ["SEQ_END"] = "4096"
        os.environ["SEQ_SCHEDULE"] = "cosine"

        try:
            config = ProgressiveConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.seq_start, 512)
            self.assertEqual(config.seq_end, 4096)
            self.assertEqual(config.schedule, "cosine")
        finally:
            # Clean up env
            for key in ["PROGRESSIVE_SEQ_ENABLED", "SEQ_START", "SEQ_END", "SEQ_SCHEDULE"]:
                os.environ.pop(key, None)

    def test_defaults_when_disabled(self):
        """When disabled, should return None or disabled config."""
        os.environ["PROGRESSIVE_SEQ_ENABLED"] = "0"

        try:
            config = ProgressiveConfig.from_env()
            self.assertFalse(config.enabled)
        finally:
            os.environ.pop("PROGRESSIVE_SEQ_ENABLED", None)

    def test_validates_start_less_than_end(self):
        """Should validate that seq_start <= seq_end."""
        os.environ["PROGRESSIVE_SEQ_ENABLED"] = "1"
        os.environ["SEQ_START"] = "4096"
        os.environ["SEQ_END"] = "512"

        try:
            with self.assertRaises(ValueError):
                ProgressiveConfig.from_env()
        finally:
            for key in ["PROGRESSIVE_SEQ_ENABLED", "SEQ_START", "SEQ_END"]:
                os.environ.pop(key, None)

    def test_validates_multiple_of_64(self):
        """Should validate that seq lengths are multiples of 64."""
        os.environ["PROGRESSIVE_SEQ_ENABLED"] = "1"
        os.environ["SEQ_START"] = "500"  # Not multiple of 64
        os.environ["SEQ_END"] = "4096"

        try:
            with self.assertRaises(ValueError):
                ProgressiveConfig.from_env()
        finally:
            for key in ["PROGRESSIVE_SEQ_ENABLED", "SEQ_START", "SEQ_END"]:
                os.environ.pop(key, None)

    def test_defaults_to_linear_schedule(self):
        """Should default to linear schedule if not specified."""
        os.environ["PROGRESSIVE_SEQ_ENABLED"] = "1"
        os.environ["SEQ_START"] = "512"
        os.environ["SEQ_END"] = "4096"

        try:
            config = ProgressiveConfig.from_env()
            self.assertEqual(config.schedule, "linear")
        finally:
            for key in ["PROGRESSIVE_SEQ_ENABLED", "SEQ_START", "SEQ_END"]:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()

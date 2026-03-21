"""Tests for difficulty-adjusted dynamic curriculum learning (RJC-33)."""

import math
import unittest

from experiments.difficulty_adjusted_lr import DifficultyAdjuster, HalvingSchedule


class TestDifficultyAdjuster(unittest.TestCase):
    """Test DifficultyAdjuster class."""

    def setUp(self):
        """Initialize a DifficultyAdjuster for testing."""
        self.adjuster = DifficultyAdjuster(
            target_rate=0.001, ema_alpha=0.1, lr_up_factor=1.05, lr_down_factor=0.95
        )

    def test_initialization(self):
        """Test that DifficultyAdjuster initializes with correct parameters."""
        self.assertEqual(self.adjuster.target_rate, 0.001)
        self.assertEqual(self.adjuster.ema_alpha, 0.1)
        self.assertEqual(self.adjuster.lr_up_factor, 1.05)
        self.assertEqual(self.adjuster.lr_down_factor, 0.95)
        self.assertEqual(self.adjuster.loss_ema, None)
        self.assertEqual(self.adjuster.loss_reduction_rate_ema, 0.0)

    def test_first_loss_update(self):
        """Test that first loss update initializes EMA."""
        self.adjuster.update(1.0)
        # First update should initialize EMA
        self.assertIsNotNone(self.adjuster.loss_ema)
        self.assertAlmostEqual(self.adjuster.loss_ema, 1.0)

    def test_subsequent_loss_updates(self):
        """Test that subsequent updates use EMA formula."""
        self.adjuster.update(1.0)
        self.adjuster.update(0.9)
        # EMA should be: 0.1 * 0.9 + 0.9 * 1.0 = 0.99
        expected_ema = 0.1 * 0.9 + 0.9 * 1.0
        self.assertAlmostEqual(self.adjuster.loss_ema, expected_ema, places=5)

    def test_loss_reduction_rate_calculation(self):
        """Test that loss reduction rate is correctly calculated."""
        self.adjuster.update(1.0)
        self.adjuster.update(0.95)
        # Loss reduction rate = (1.0 - 0.95) / 1.0 = 0.05
        # This should be tracked in loss_reduction_rate_ema
        self.assertGreater(self.adjuster.loss_reduction_rate_ema, 0.0)

    def test_get_lr_multiplier_when_learning_fast(self):
        """Test that LR multiplier increases when learning fast."""
        self.adjuster.loss_reduction_rate_ema = 0.005  # Above target 0.001
        multiplier = self.adjuster.get_lr_multiplier()
        self.assertGreater(multiplier, 1.0)
        self.assertAlmostEqual(multiplier, 1.05, places=2)  # Should be close to lr_up_factor

    def test_get_lr_multiplier_when_learning_slow(self):
        """Test that LR multiplier decreases when learning slow."""
        self.adjuster.loss_reduction_rate_ema = 0.0002  # Below target 0.001
        multiplier = self.adjuster.get_lr_multiplier()
        self.assertLess(multiplier, 1.0)
        self.assertAlmostEqual(multiplier, 0.95, places=2)  # Should be close to lr_down_factor

    def test_get_lr_multiplier_near_target(self):
        """Test that LR multiplier is at least 1.0 when at or above target rate."""
        self.adjuster.loss_reduction_rate_ema = 0.001  # Equal to target
        multiplier = self.adjuster.get_lr_multiplier()
        # At target rate, should switch to up factor
        self.assertEqual(multiplier, self.adjuster.lr_up_factor)

    def test_get_batch_size_multiplier_when_learning_fast(self):
        """Test that batch size multiplier increases when learning fast."""
        self.adjuster.loss_reduction_rate_ema = 0.01  # Much above target
        multiplier = self.adjuster.get_batch_size_multiplier()
        self.assertGreater(multiplier, 1.0)

    def test_get_batch_size_multiplier_when_learning_slow(self):
        """Test that batch size multiplier decreases when learning slow."""
        self.adjuster.loss_reduction_rate_ema = 0.00001  # Much below target
        multiplier = self.adjuster.get_batch_size_multiplier()
        self.assertLess(multiplier, 1.0)


class TestHalvingSchedule(unittest.TestCase):
    """Test HalvingSchedule class."""

    def setUp(self):
        """Initialize a HalvingSchedule for testing."""
        self.schedule = HalvingSchedule()

    def test_schedule_at_start(self):
        """Test that LR is 1.0 at the start."""
        multiplier = self.schedule.get_lr(0, 10000)
        self.assertAlmostEqual(multiplier, 1.0, places=2)

    def test_schedule_at_25_percent(self):
        """Test that LR is halved at 25% of training."""
        multiplier = self.schedule.get_lr(2500, 10000)
        self.assertAlmostEqual(multiplier, 0.5, places=2)

    def test_schedule_at_50_percent(self):
        """Test that LR is halved again at 50% of training."""
        multiplier = self.schedule.get_lr(5000, 10000)
        self.assertAlmostEqual(multiplier, 0.25, places=2)

    def test_schedule_at_75_percent(self):
        """Test that LR is halved again at 75% of training."""
        multiplier = self.schedule.get_lr(7500, 10000)
        self.assertAlmostEqual(multiplier, 0.125, places=2)

    def test_schedule_at_end(self):
        """Test that LR is very small at end of training."""
        multiplier = self.schedule.get_lr(9999, 10000)
        self.assertEqual(multiplier, 0.125)  # Should be 0.125 in final phase

    def test_schedule_monotonic_decrease(self):
        """Test that LR schedule decreases monotonically."""
        prev_multiplier = 1.0
        for step in [0, 2000, 5000, 7500, 9999]:
            multiplier = self.schedule.get_lr(step, 10000)
            self.assertLessEqual(multiplier, prev_multiplier + 1e-6)
            prev_multiplier = multiplier


if __name__ == "__main__":
    unittest.main()

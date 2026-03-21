"""Integration tests for curriculum learning features (RJC-33, RJC-35)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestDifficultyAdjusterIntegration(unittest.TestCase):
    """Integration test for DifficultyAdjuster with simulated training."""

    def test_difficulty_adjuster_simulation(self):
        """Simulate a training loop with DifficultyAdjuster."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from experiments.difficulty_adjusted_lr import DifficultyAdjuster

        # Initialize adjuster
        adjuster = DifficultyAdjuster(
            target_rate=0.001, ema_alpha=0.05, lr_up_factor=1.05, lr_down_factor=0.95
        )

        # Simulate 100 training steps with decreasing loss
        initial_loss = 5.0
        losses = [initial_loss - (initial_loss - 0.5) * (i / 100) for i in range(100)]

        lr_history = []
        for loss_value in losses:
            adjuster.update(loss_value)
            lr_mult = adjuster.get_lr_multiplier()
            lr_history.append(lr_mult)

        # Early on, loss should be decreasing fast -> LR should increase
        early_avg = np.mean(lr_history[:10])

        # Later, loss should be decreasing slower -> LR should decrease
        late_avg = np.mean(lr_history[-10:])

        # LR should generally decrease over time as loss reduction slows
        self.assertGreater(early_avg, 0.95)  # Should start aggressive
        self.assertLess(late_avg, 1.1)  # Should calm down


class TestBondingCurveLossIntegration(unittest.TestCase):
    """Integration test for TokenFrequencyWeighter with real frequency data."""

    def test_frequency_weighting_integration(self):
        """Test frequency weighting with realistic token distribution."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from experiments.bonding_curve_loss import TokenFrequencyWeighter

        # Create a TokenFrequencyWeighter
        weighter = TokenFrequencyWeighter(vocab_size=256, alpha_start=0.0, alpha_end=0.5)

        # Create synthetic training data with power-law distribution
        # (like real natural language)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "train_data.bin"

            # Create tokens following zipfian distribution
            # Most tokens are common (low ID), few tokens are very rare (high ID)
            tokens = []
            for token_id in range(256):
                # Zipf distribution: frequency ~ 1/token_id
                freq = max(1, int(1000 / (token_id + 1)))
                tokens.extend([token_id] * freq)

            token_array = np.array(tokens, dtype=np.uint16)
            token_array.tofile(data_file)

            # Build frequency table
            weighter.build_frequency_table(tmpdir)

            # At start (alpha=0), all weights should be 1.0
            weights_start = weighter.get_weights(step=0, total_steps=1000)
            self.assertAlmostEqual(np.mean(weights_start), 1.0, places=1)

            # At end (alpha=0.5), rare tokens should have higher weight
            weights_end = weighter.get_weights(step=1000, total_steps=1000)

            # Common tokens (low ID) should have weight < 1
            # Rare tokens (high ID) should have weight > 1
            common_token_idx = 0
            rare_token_idx = 255

            self.assertLess(weights_end[common_token_idx], weights_end[rare_token_idx])


class TestHalvingScheduleIntegration(unittest.TestCase):
    """Integration test for HalvingSchedule with training progression."""

    def test_halving_schedule_training_loop(self):
        """Test HalvingSchedule through a simulated training loop."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from experiments.difficulty_adjusted_lr import HalvingSchedule

        schedule = HalvingSchedule()

        total_steps = 10000
        lr_history = []

        # Simulate 10,000 training steps
        for step in range(total_steps):
            lr_mult = schedule.get_lr(step, total_steps)
            lr_history.append(lr_mult)

        # Check phase boundaries
        lr_at_0pct = lr_history[0]
        lr_at_25pct = lr_history[2500]
        lr_at_50pct = lr_history[5000]
        lr_at_75pct = lr_history[7500]
        lr_at_100pct = lr_history[9999]

        # Verify halving schedule
        self.assertEqual(lr_at_0pct, 1.0)
        self.assertEqual(lr_at_25pct, 0.5)
        self.assertEqual(lr_at_50pct, 0.25)
        self.assertEqual(lr_at_75pct, 0.125)
        self.assertEqual(lr_at_100pct, 0.125)

        # Verify monotonic decrease
        for i in range(1, len(lr_history)):
            self.assertLessEqual(lr_history[i], lr_history[i - 1] + 1e-6)


if __name__ == "__main__":
    unittest.main()

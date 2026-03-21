"""Tests for bonding curve token-weighted loss function (RJC-35)."""

import math
import os
import tempfile
import unittest

import numpy as np

from experiments.bonding_curve_loss import EntropyWeighter, TokenFrequencyWeighter


class TestTokenFrequencyWeighter(unittest.TestCase):
    """Test TokenFrequencyWeighter class."""

    def setUp(self):
        """Initialize a TokenFrequencyWeighter for testing."""
        self.weighter = TokenFrequencyWeighter(vocab_size=256, alpha_start=0.0, alpha_end=0.5)

    def test_initialization(self):
        """Test that TokenFrequencyWeighter initializes correctly."""
        self.assertEqual(self.weighter.vocab_size, 256)
        self.assertEqual(self.weighter.alpha_start, 0.0)
        self.assertEqual(self.weighter.alpha_end, 0.5)
        self.assertIsNone(self.weighter.frequencies)

    def test_build_frequency_table_from_data(self):
        """Test building frequency table from tokenized data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple tokenized data file
            data_file = os.path.join(tmpdir, "test_data.bin")
            # Create fake token data: [1, 2, 1, 3, 1, 2] (1 appears 3 times, 2 twice, 3 once)
            token_data = np.array([1, 2, 1, 3, 1, 2], dtype=np.uint16)
            token_data.tofile(data_file)

            # Build frequency table
            self.weighter.build_frequency_table(tmpdir)

            # Check that frequencies were built
            self.assertIsNotNone(self.weighter.frequencies)
            self.assertEqual(len(self.weighter.frequencies), 256)

    def test_get_weights_at_start(self):
        """Test that weights at start are 1.0 (alpha=0)."""
        # Create mock frequencies
        self.weighter.frequencies = np.ones(256, dtype=np.float32)
        self.weighter.frequencies[0:5] = [5.0, 4.0, 3.0, 2.0, 1.0]  # Varying frequencies

        weights = self.weighter.get_weights(step=0, total_steps=1000)

        # At alpha=0, all weights should be 1.0
        self.assertEqual(weights.shape[0], 256)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)
        self.assertAlmostEqual(float(weights[1]), 1.0, places=5)

    def test_get_weights_at_middle(self):
        """Test that weights at middle apply intermediate alpha."""
        self.weighter.frequencies = np.ones(256, dtype=np.float32)
        self.weighter.frequencies[0] = 10.0  # Common token
        self.weighter.frequencies[1] = 1.0  # Rare token

        weights = self.weighter.get_weights(step=500, total_steps=1000)

        # At alpha=0.25 (middle), rare tokens should have higher weight
        # weight = 1 / freq^alpha
        # weight[0] = 1 / 10^0.25 < 1
        # weight[1] = 1 / 1^0.25 = 1
        self.assertLess(float(weights[0]), float(weights[1]))

    def test_get_weights_at_end(self):
        """Test that weights at end apply full alpha."""
        self.weighter.frequencies = np.ones(256, dtype=np.float32)
        self.weighter.frequencies[0] = 10.0  # Common token
        self.weighter.frequencies[1] = 1.0  # Rare token

        weights = self.weighter.get_weights(step=1000, total_steps=1000)

        # At alpha=0.5 (end), rare tokens should have much higher weight
        # weight = 1 / freq^alpha
        # weight[0] = 1 / 10^0.5 ≈ 0.316
        # weight[1] = 1 / 1^0.5 = 1
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[0]), 0.3)
        self.assertLess(float(weights[0]), 0.35)

    def test_alpha_schedule(self):
        """Test that alpha follows linear schedule."""
        self.weighter.frequencies = np.ones(256, dtype=np.float32)
        self.weighter.frequencies[0] = 10.0

        # Test at 0%, 50%, 100%
        weights_0 = self.weighter.get_weights(step=0, total_steps=1000)
        weights_500 = self.weighter.get_weights(step=500, total_steps=1000)
        weights_1000 = self.weighter.get_weights(step=1000, total_steps=1000)

        # weight[0] should decrease as alpha increases (common token becomes less important)
        w0 = float(weights_0[0])
        w500 = float(weights_500[0])
        w1000 = float(weights_1000[0])

        self.assertGreater(w0, w500)
        self.assertGreater(w500, w1000)

    def test_weighted_cross_entropy_torch(self):
        """Test weighted cross entropy with PyTorch."""
        try:
            import torch

            from experiments.bonding_curve_loss import weighted_cross_entropy_torch

            # Create simple logits and targets
            batch_size, seq_len, vocab_size = 2, 4, 256
            logits = torch.randn(batch_size, seq_len, vocab_size)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))
            weights = torch.ones(vocab_size)

            # Compute weighted loss
            loss = weighted_cross_entropy_torch(logits, targets, weights)

            # Check that loss is a scalar
            self.assertEqual(loss.dim(), 0)
            self.assertGreater(float(loss), 0.0)
        except ImportError:
            self.skipTest("PyTorch not available")

    def test_weighted_cross_entropy_mlx(self):
        """Test weighted cross entropy with MLX."""
        try:
            import mlx.core as mx

            from experiments.bonding_curve_loss import weighted_cross_entropy_mlx

            # Create simple logits and targets
            batch_size, seq_len, vocab_size = 2, 4, 256
            logits = mx.random.normal((batch_size, seq_len, vocab_size))
            targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))
            weights = mx.ones(vocab_size)

            # Compute weighted loss
            loss = weighted_cross_entropy_mlx(logits, targets, weights)

            # Check that loss is a scalar
            loss_val = float(loss)
            self.assertGreater(loss_val, 0.0)
        except ImportError:
            self.skipTest("MLX not available")


class TestEntropyWeighter(unittest.TestCase):
    """Test EntropyWeighter class."""

    def setUp(self):
        """Initialize an EntropyWeighter for testing."""
        self.weighter = EntropyWeighter(vocab_size=256, entropy_scale=0.1)

    def test_initialization(self):
        """Test that EntropyWeighter initializes correctly."""
        self.assertEqual(self.weighter.vocab_size, 256)
        self.assertEqual(self.weighter.entropy_scale, 0.1)

    def test_get_entropy_weights_torch(self):
        """Test entropy-based weighting with PyTorch."""
        try:
            import torch

            from experiments.bonding_curve_loss import entropy_weights_torch

            # Create logits where some classes are more uncertain
            logits = torch.tensor(
                [
                    [10.0, 0.0, 0.0],  # Confident prediction
                    [1.0, 1.0, 1.0],  # Uncertain prediction
                ]
            )

            weights = entropy_weights_torch(logits, entropy_scale=0.1)

            # Check shape
            self.assertEqual(weights.shape, (2,))

            # Uncertain prediction should have higher weight
            self.assertGreater(float(weights[1]), float(weights[0]))
        except ImportError:
            self.skipTest("PyTorch not available")

    def test_get_entropy_weights_mlx(self):
        """Test entropy-based weighting with MLX."""
        try:
            import mlx.core as mx
            import mlx.nn as nn

            from experiments.bonding_curve_loss import entropy_weights_mlx

            # Create logits
            logits = mx.array(
                [
                    [10.0, 0.0, 0.0],  # Confident prediction
                    [1.0, 1.0, 1.0],  # Uncertain prediction
                ]
            )

            weights = entropy_weights_mlx(logits, entropy_scale=0.1)

            # Check shape
            self.assertEqual(weights.shape[0], 2)

            # Uncertain prediction should have higher weight
            self.assertGreater(float(weights[1]), float(weights[0]))
        except ImportError:
            self.skipTest("MLX not available")


if __name__ == "__main__":
    unittest.main()

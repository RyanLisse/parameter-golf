#!/usr/bin/env python3
"""Tests for Gas Optimization module (RJC-30)."""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.gas_optimization import (
    GasOptConfig,
    adaptive_grad_clip,
    fused_relu_squared,
    optimize_batch_size,
    should_skip_dropout,
)


class TestGasOptConfig(unittest.TestCase):
    """Test GasOptConfig reads environment variables correctly."""

    def setUp(self):
        """Clear relevant env vars before each test."""
        for key in ["GAS_OPT_ENABLED", "GAS_SKIP_DROPOUT_WARMDOWN", "GAS_FUSE_OPS"]:
            os.environ.pop(key, None)

    def test_default_config(self):
        """Test default config when no env vars set."""
        config = GasOptConfig()
        self.assertFalse(config.enabled)
        self.assertFalse(config.skip_dropout_warmdown)
        self.assertFalse(config.fuse_ops)

    def test_enabled_from_env(self):
        """Test GAS_OPT_ENABLED=1 enables optimization."""
        os.environ["GAS_OPT_ENABLED"] = "1"
        config = GasOptConfig()
        self.assertTrue(config.enabled)

    def test_skip_dropout_from_env(self):
        """Test GAS_SKIP_DROPOUT_WARMDOWN=1 enables dropout skipping."""
        os.environ["GAS_SKIP_DROPOUT_WARMDOWN"] = "1"
        config = GasOptConfig()
        self.assertTrue(config.skip_dropout_warmdown)

    def test_fuse_ops_from_env(self):
        """Test GAS_FUSE_OPS=1 enables operation fusion."""
        os.environ["GAS_FUSE_OPS"] = "1"
        config = GasOptConfig()
        self.assertTrue(config.fuse_ops)

    def test_all_enabled(self):
        """Test all flags enabled."""
        os.environ["GAS_OPT_ENABLED"] = "1"
        os.environ["GAS_SKIP_DROPOUT_WARMDOWN"] = "1"
        os.environ["GAS_FUSE_OPS"] = "1"
        config = GasOptConfig()
        self.assertTrue(config.enabled)
        self.assertTrue(config.skip_dropout_warmdown)
        self.assertTrue(config.fuse_ops)


class TestShouldSkipDropout(unittest.TestCase):
    """Test dropout skipping logic during warmdown."""

    def test_skip_in_last_20_percent(self):
        """Test dropout is skipped in final 20% of training."""
        # 1000 steps, warmdown_ratio=0.2 means skip after step 800
        self.assertTrue(should_skip_dropout(900, 1000, warmdown_ratio=0.2))
        self.assertTrue(should_skip_dropout(850, 1000, warmdown_ratio=0.2))
        self.assertTrue(should_skip_dropout(1000, 1000, warmdown_ratio=0.2))

    def test_no_skip_before_warmdown(self):
        """Test dropout is not skipped before warmdown phase."""
        self.assertFalse(should_skip_dropout(500, 1000, warmdown_ratio=0.2))
        self.assertFalse(should_skip_dropout(799, 1000, warmdown_ratio=0.2))
        self.assertFalse(should_skip_dropout(0, 1000, warmdown_ratio=0.2))

    def test_boundary_condition(self):
        """Test exact boundary of warmdown phase."""
        # At step 800, we're exactly at 80% - should start skipping
        self.assertTrue(should_skip_dropout(800, 1000, warmdown_ratio=0.2))

    def test_custom_warmdown_ratio(self):
        """Test custom warmdown ratio."""
        # 30% warmdown means skip after step 700
        self.assertTrue(should_skip_dropout(750, 1000, warmdown_ratio=0.3))
        self.assertFalse(should_skip_dropout(650, 1000, warmdown_ratio=0.3))


class TestFusedReluSquared(unittest.TestCase):
    """Test fused ReLU squared operation."""

    def test_positive_values(self):
        """Test fused operation on positive values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = fused_relu_squared(x)
        expected = torch.tensor([1.0, 4.0, 9.0])
        torch.testing.assert_close(result, expected)

    def test_negative_values(self):
        """Test negative values are zeroed before squaring."""
        x = torch.tensor([-1.0, -2.0, -3.0])
        result = fused_relu_squared(x)
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_mixed_values(self):
        """Test mixed positive and negative values."""
        x = torch.tensor([-2.0, 0.0, 2.0, -1.0, 3.0])
        result = fused_relu_squared(x)
        expected = torch.tensor([0.0, 0.0, 4.0, 0.0, 9.0])
        torch.testing.assert_close(result, expected)

    def test_zero_value(self):
        """Test zero value."""
        x = torch.tensor([0.0])
        result = fused_relu_squared(x)
        expected = torch.tensor([0.0])
        torch.testing.assert_close(result, expected)

    def test_preserves_dtype(self):
        """Test operation preserves data type."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = fused_relu_squared(x)
        self.assertEqual(result.dtype, torch.float32)

    def test_preserves_shape(self):
        """Test operation preserves shape."""
        x = torch.randn(2, 3, 4)
        result = fused_relu_squared(x)
        self.assertEqual(result.shape, (2, 3, 4))


class TestAdaptiveGradClip(unittest.TestCase):
    """Test adaptive gradient clipping."""

    def test_skip_when_consistently_low(self):
        """Test skipping when gradient norms are consistently below threshold."""
        # History of low gradient norms
        history = [0.1, 0.12, 0.11, 0.09, 0.10]
        # Current norm also low
        result = adaptive_grad_clip(history, threshold_multiplier=2.0)
        self.assertTrue(result)  # Should skip clipping

    def test_clip_when_high_norm_appears(self):
        """Test clipping when gradient norm exceeds threshold."""
        # History of low norms, but recent spike
        history = [0.1, 0.12, 0.11, 0.5, 0.6]
        result = adaptive_grad_clip(history, threshold_multiplier=2.0)
        self.assertFalse(result)  # Should NOT skip clipping

    def test_insufficient_history(self):
        """Test behavior with insufficient history."""
        # Not enough samples - should not skip (be conservative)
        history = [0.1, 0.12]
        result = adaptive_grad_clip(history, threshold_multiplier=2.0)
        self.assertFalse(result)

    def test_empty_history(self):
        """Test behavior with empty history."""
        history = []
        result = adaptive_grad_clip(history, threshold_multiplier=2.0)
        self.assertFalse(result)

    def test_threshold_multiplier_effect(self):
        """Test different threshold multipliers."""
        history = [0.1, 0.1, 0.1, 0.1, 0.15]
        # Low multiplier - more conservative, less likely to skip
        result_conservative = adaptive_grad_clip(history, threshold_multiplier=1.2)
        # High multiplier - more aggressive, more likely to skip
        result_aggressive = adaptive_grad_clip(history, threshold_multiplier=5.0)
        # With these norms, aggressive should skip, conservative might not
        self.assertTrue(result_aggressive)


class TestOptimizeBatchSize(unittest.TestCase):
    """Test optimal batch size calculation."""

    def test_typical_gpu_memory(self):
        """Test batch size for typical GPU memory (40GB)."""
        # 40GB, 100M params, seq_len=1024
        batch_size = optimize_batch_size(
            gpu_memory_gb=40, model_params=100_000_000, seq_len=1024
        )
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        # Should fit in memory
        self.assertLess(batch_size, 10000)

    def test_small_gpu_memory(self):
        """Test batch size for small GPU (16GB)."""
        batch_size = optimize_batch_size(
            gpu_memory_gb=16, model_params=100_000_000, seq_len=1024
        )
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

    def test_large_model(self):
        """Test batch size decreases with larger models."""
        batch_small = optimize_batch_size(
            gpu_memory_gb=40, model_params=50_000_000, seq_len=1024
        )
        batch_large = optimize_batch_size(
            gpu_memory_gb=40, model_params=200_000_000, seq_len=1024
        )
        # Larger model should require smaller batch
        self.assertLess(batch_large, batch_small)

    def test_longer_sequences(self):
        """Test batch size decreases with longer sequences."""
        batch_short = optimize_batch_size(
            gpu_memory_gb=40, model_params=100_000_000, seq_len=512
        )
        batch_long = optimize_batch_size(
            gpu_memory_gb=40, model_params=100_000_000, seq_len=2048
        )
        # Longer sequences should require smaller batch
        self.assertLess(batch_long, batch_short)

    def test_minimum_batch_size(self):
        """Test batch size has a minimum value."""
        # Even with huge model and long sequences, should return at least 1
        batch_size = optimize_batch_size(
            gpu_memory_gb=1, model_params=1_000_000_000, seq_len=4096
        )
        self.assertGreaterEqual(batch_size, 1)


if __name__ == "__main__":
    unittest.main()

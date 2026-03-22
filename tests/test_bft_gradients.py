#!/usr/bin/env python3
"""Tests for BFT Trimmed-Mean Gradient Aggregation (RJC-36)."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.bft_gradients import BFTAggregator, QuorumEarlyStopping


class TestBFTAggregator(unittest.TestCase):
    """Test BFT trimmed-mean gradient aggregation."""

    def test_initialization(self):
        """Test BFTAggregator initializes correctly."""
        aggregator = BFTAggregator(world_size=8, trim_count=1)
        self.assertEqual(aggregator.world_size, 8)
        self.assertEqual(aggregator.trim_count, 1)

    def test_fallback_small_world(self):
        """Test fallback to AllReduce when world size too small."""
        # With 3 GPUs and trim_count=1, we'd trim 2 and have only 1 left
        # Should fallback to AllReduce
        aggregator = BFTAggregator(world_size=3, trim_count=1)
        # The aggregator should recognize this and use standard averaging
        self.assertTrue(aggregator.should_use_fallback())

    def test_valid_configuration(self):
        """Test valid configuration doesn't fallback."""
        # 8 GPUs, trim 2 (top 1, bottom 1), average 6 remaining
        aggregator = BFTAggregator(world_size=8, trim_count=1)
        self.assertFalse(aggregator.should_use_fallback())

    def test_trimmed_mean_calculation(self):
        """Test trimmed mean calculation logic."""
        aggregator = BFTAggregator(world_size=8, trim_count=1)

        # Simulate 8 gradient values from different GPUs
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])

        # Trim top 1 (100.0) and bottom 1 (1.0)
        # Average remaining: (2+3+4+5+6+7)/6 = 27/6 = 4.5
        result = aggregator._compute_trimmed_mean(values)
        expected = torch.tensor(4.5)
        torch.testing.assert_close(result, expected)

    def test_trimmed_mean_removes_outliers(self):
        """Test that trimmed mean effectively removes Byzantine outliers."""
        aggregator = BFTAggregator(world_size=8, trim_count=1)

        # Most GPUs report ~5.0, but one Byzantine attacker reports 1000.0
        values = torch.tensor([4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 1000.0])

        result = aggregator._compute_trimmed_mean(values)

        # Result should be close to 5.0, not affected by the 1000.0 outlier
        # Trimming removes 4.8 and 1000.0, leaving 4.9,5.0,5.1,5.2,5.3,5.4
        expected = (4.9 + 5.0 + 5.1 + 5.2 + 5.3 + 5.4) / 6
        torch.testing.assert_close(result, torch.tensor(expected), rtol=1e-4, atol=1e-4)

    def test_higher_trim_count(self):
        """Test with higher trim count for more aggressive filtering."""
        aggregator = BFTAggregator(world_size=8, trim_count=2)

        # Trim top 2 and bottom 2, average middle 4
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = aggregator._compute_trimmed_mean(values)
        # Removes 1,2,7,8 → average 3,4,5,6 = 4.5
        expected = torch.tensor(4.5)
        torch.testing.assert_close(result, expected)

    def test_multi_dimensional_tensor(self):
        """Test trimmed mean works with multi-dimensional tensors."""
        aggregator = BFTAggregator(world_size=8, trim_count=1)

        # Create a 2x3 tensor for each of 8 GPUs
        # We'll test element-wise trimming
        tensors = [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + i
            for i in range(8)
        ]

        # Stack along new dimension: [8, 2, 3]
        stacked = torch.stack(tensors)

        # Compute trimmed mean along GPU dimension (dim=0)
        result = aggregator._compute_trimmed_mean_multidim(stacked)

        # Should have shape [2, 3] after reduction
        self.assertEqual(result.shape, (2, 3))

        # Each element should be trimmed mean across 8 GPUs
        # For first element: [1, 2, 3, 4, 5, 6, 7, 8]
        # Trim 1 and 8, average [2,3,4,5,6,7] = 4.5
        expected_first = 4.5
        self.assertAlmostEqual(result[0, 0].item(), expected_first, places=4)


class TestQuorumEarlyStopping(unittest.TestCase):
    """Test quorum-based early stopping."""

    def test_initialization(self):
        """Test QuorumEarlyStopping initializes correctly."""
        stopper = QuorumEarlyStopping(
            world_size=8, quorum_size=6, patience=5, min_delta=0.001
        )
        self.assertEqual(stopper.world_size, 8)
        self.assertEqual(stopper.quorum_size, 6)
        self.assertEqual(stopper.patience, 5)

    def test_no_stop_initially(self):
        """Test no early stopping when losses are improving."""
        stopper = QuorumEarlyStopping(
            world_size=8, quorum_size=6, patience=3, min_delta=0.01
        )

        # All GPUs show improving loss
        losses = torch.tensor([2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8])
        should_stop = stopper.should_stop(losses)
        self.assertFalse(should_stop)

    def test_stop_when_quorum_plateaus(self):
        """Test early stopping when quorum agrees loss has plateaued."""
        stopper = QuorumEarlyStopping(
            world_size=8, quorum_size=6, patience=2, min_delta=0.01
        )

        # Simulate multiple steps where 6/8 GPUs see plateau
        # Step 1: losses around 2.0 for most GPUs
        losses1 = torch.tensor([2.0, 2.01, 2.0, 2.02, 2.0, 2.01, 2.0, 1.5])
        stopper.should_stop(losses1)

        # Step 2: similar losses (plateau continues)
        losses2 = torch.tensor([2.0, 2.01, 2.0, 2.01, 2.0, 2.0, 2.01, 1.4])
        stopper.should_stop(losses2)

        # Step 3: plateau persists for patience steps
        losses3 = torch.tensor([2.0, 2.0, 2.01, 2.0, 2.01, 2.0, 2.0, 1.3])
        should_stop = stopper.should_stop(losses3)

        # After patience steps with quorum plateau, should stop
        self.assertTrue(should_stop)

    def test_no_stop_without_quorum(self):
        """Test no early stopping when quorum doesn't agree."""
        stopper = QuorumEarlyStopping(
            world_size=8, quorum_size=6, patience=2, min_delta=0.01
        )

        # Only 4/8 GPUs see plateau (not enough for quorum of 6)
        # Other GPUs keep improving
        for i in range(5):
            # First 4 GPUs plateau at 2.0, last 4 GPUs keep improving
            base_loss = 2.0 - (i * 0.1)  # Keeps decreasing
            losses = torch.tensor([2.0, 2.0, 2.0, 2.0, base_loss, base_loss - 0.1, base_loss - 0.2, base_loss - 0.3])
            should_stop = stopper.should_stop(losses)
            self.assertFalse(should_stop)

    def test_reset_on_improvement(self):
        """Test patience counter resets when loss improves."""
        stopper = QuorumEarlyStopping(
            world_size=8, quorum_size=6, patience=3, min_delta=0.01
        )

        # Step 1: plateau
        losses1 = torch.tensor([2.0] * 8)
        stopper.should_stop(losses1)

        # Step 2: still plateau
        losses2 = torch.tensor([2.0] * 8)
        stopper.should_stop(losses2)

        # Step 3: improvement!
        losses3 = torch.tensor([1.8] * 8)
        stopper.should_stop(losses3)

        # Step 4: back to plateau (but patience should have reset)
        losses4 = torch.tensor([1.8] * 8)
        stopper.should_stop(losses4)

        # Should not stop yet since we haven't hit patience after the improvement
        losses5 = torch.tensor([1.8] * 8)
        should_stop = stopper.should_stop(losses5)
        self.assertFalse(should_stop)

    def test_min_delta_threshold(self):
        """Test min_delta threshold for considering improvement."""
        stopper = QuorumEarlyStopping(
            world_size=8, quorum_size=6, patience=2, min_delta=0.1
        )

        # Step 1: loss at 2.0
        losses1 = torch.tensor([2.0] * 8)
        stopper.should_stop(losses1)

        # Step 2: tiny improvement (less than min_delta), should count as plateau
        losses2 = torch.tensor([1.99] * 8)
        stopper.should_stop(losses2)

        # Step 3: still tiny improvements
        losses3 = torch.tensor([1.98] * 8)
        should_stop = stopper.should_stop(losses3)

        # Should trigger early stop since improvements < min_delta
        self.assertTrue(should_stop)


class TestBFTWithoutDistributed(unittest.TestCase):
    """Test BFT handles non-distributed environments gracefully."""

    @patch("torch.distributed.is_initialized")
    def test_disabled_without_distributed(self, mock_is_init):
        """Test BFT gracefully handles non-distributed environment."""
        mock_is_init.return_value = False

        aggregator = BFTAggregator(world_size=8, trim_count=1)

        # Should recognize distributed is not available
        self.assertTrue(aggregator.requires_distributed())

        # In non-distributed mode, should return input unchanged
        tensor = torch.randn(10, 10)
        result = aggregator.fallback_reduce(tensor)
        torch.testing.assert_close(result, tensor)


if __name__ == "__main__":
    unittest.main()

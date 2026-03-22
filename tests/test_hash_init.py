"""Tests for hash-based initialization and deterministic dropout (RJC-38)."""

from __future__ import annotations

import unittest
import numpy as np

from experiments import hash_init


class HashInitTests(unittest.TestCase):
    """Test hash-based weight initialization using SipHash."""

    def test_siphash_init_deterministic(self) -> None:
        """Verify that siphash_init produces deterministic outputs for same seed."""
        w1 = hash_init.siphash_init(layer_idx=0, rows=4, cols=4, seed=42)
        w2 = hash_init.siphash_init(layer_idx=0, rows=4, cols=4, seed=42)
        np.testing.assert_array_almost_equal(w1, w2)

    def test_siphash_init_different_layers(self) -> None:
        """Verify that different layers produce different weights."""
        w0 = hash_init.siphash_init(layer_idx=0, rows=4, cols=4, seed=42)
        w1 = hash_init.siphash_init(layer_idx=1, rows=4, cols=4, seed=42)
        assert not np.allclose(w0, w1), "Different layers should have different weights"

    def test_siphash_init_shape(self) -> None:
        """Verify that output shape matches requested dimensions."""
        w = hash_init.siphash_init(layer_idx=0, rows=10, cols=5, seed=42)
        self.assertEqual(w.shape, (10, 5))

    def test_siphash_init_normalized(self) -> None:
        """Verify that weights are normalized to zero mean, unit variance."""
        w = hash_init.siphash_init(layer_idx=0, rows=100, cols=100, seed=42)
        mean = np.mean(w)
        std = np.std(w)
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(std, 1.0, places=5)

    def test_hash_dropout_mask_deterministic(self) -> None:
        """Verify that hash_dropout_mask produces deterministic outputs."""
        m1 = hash_init.hash_dropout_mask(layer_idx=0, step=0, seq_len=10, dropout_pct=25)
        m2 = hash_init.hash_dropout_mask(layer_idx=0, step=0, seq_len=10, dropout_pct=25)
        np.testing.assert_array_equal(m1, m2)

    def test_hash_dropout_mask_shape(self) -> None:
        """Verify that dropout mask has correct shape."""
        m = hash_init.hash_dropout_mask(layer_idx=5, step=100, seq_len=32, dropout_pct=20)
        self.assertEqual(m.shape, (32,))

    def test_hash_dropout_mask_binary(self) -> None:
        """Verify that dropout mask contains only 0 and 1."""
        m = hash_init.hash_dropout_mask(layer_idx=0, step=0, seq_len=100, dropout_pct=50)
        assert np.all((m == 0) | (m == 1)), "Mask should contain only 0 and 1"

    def test_hash_dropout_mask_percentage(self) -> None:
        """Verify that dropout percentage is approximately correct over many positions."""
        pct = 25
        m = hash_init.hash_dropout_mask(layer_idx=0, step=0, seq_len=1000, dropout_pct=pct)
        actual_pct = 100 * np.mean(m == 0)
        self.assertAlmostEqual(actual_pct, pct, delta=5)

    def test_hash_dropout_different_steps(self) -> None:
        """Verify that different steps produce different masks."""
        m0 = hash_init.hash_dropout_mask(layer_idx=0, step=0, seq_len=100, dropout_pct=50)
        m1 = hash_init.hash_dropout_mask(layer_idx=0, step=1, seq_len=100, dropout_pct=50)
        assert not np.array_equal(m0, m1), "Different steps should have different masks"

    def test_chaotic_lr_schedule_deterministic(self) -> None:
        """Verify that chaotic_lr_schedule is deterministic."""
        lr1 = hash_init.chaotic_lr_schedule(step=0, x0=0.5)
        lr2 = hash_init.chaotic_lr_schedule(step=0, x0=0.5)
        self.assertAlmostEqual(lr1, lr2)

    def test_chaotic_lr_schedule_progression(self) -> None:
        """Verify that LR schedule produces valid values in (0, 1)."""
        for step in range(10):
            lr = hash_init.chaotic_lr_schedule(step=step, x0=0.5)
            self.assertGreater(lr, 0.0)
            self.assertLess(lr, 1.0)

    def test_chaotic_lr_oscillates(self) -> None:
        """Verify that chaotic schedule produces oscillating values."""
        values = [hash_init.chaotic_lr_schedule(step=s, x0=0.5) for s in range(6)]
        # Check that we have variation in the sequence
        self.assertGreater(max(values), min(values))

    def test_apply_hash_init_modifies_weights(self) -> None:
        """Verify that apply_hash_init modifies model weights (numpy mock)."""
        # Create mock model with linear layers (use dict to simulate)
        model = {
            'layer1.weight': np.random.randn(10, 20),
            'layer2.weight': np.random.randn(5, 10),
        }
        original = {k: v.copy() for k, v in model.items()}

        hash_init.apply_hash_init(model, seed=42)

        # Weights should be different after hash init
        for key in model:
            assert not np.allclose(original[key], model[key]), f"{key} should be modified"

    def test_apply_hash_init_deterministic(self) -> None:
        """Verify that apply_hash_init is deterministic with same seed."""
        model1 = {'layer.weight': np.random.randn(8, 16)}
        model2 = {'layer.weight': np.random.randn(8, 16)}

        hash_init.apply_hash_init(model1, seed=42)
        hash_init.apply_hash_init(model2, seed=42)

        np.testing.assert_array_almost_equal(model1['layer.weight'], model2['layer.weight'])


if __name__ == "__main__":
    unittest.main()

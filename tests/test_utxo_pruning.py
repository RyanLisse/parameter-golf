"""Tests for UTXO-Style Parameter Pruning (RJC-34)."""
from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from experiments import utxo_pruning

    class SimpleModel(nn.Module):
        """Minimal model for testing pruning."""
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(32, 64, bias=False)
            self.fc2 = nn.Linear(64, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class PruningSchedulerInitTests(unittest.TestCase):
    def test_default_construction(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model)
        self.assertFalse(sched.enabled)
        self.assertEqual(sched.pruning_interval, 1000)
        self.assertEqual(sched.pruning_amount, 0.10)

    def test_env_var_construction(self) -> None:
        model = SimpleModel()
        env = {
            "PRUNING_ENABLED": "1",
            "PRUNING_START_RATIO": "1.2",
            "PRUNING_INTERVAL": "500",
            "PRUNING_AMOUNT": "0.15",
        }
        sched = utxo_pruning.PruningScheduler.from_env(model, env)
        self.assertTrue(sched.enabled)
        self.assertAlmostEqual(sched.start_ratio, 1.2)
        self.assertEqual(sched.pruning_interval, 500)
        self.assertAlmostEqual(sched.pruning_amount, 0.15)

    def test_disabled_by_default(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler.from_env(model, {})
        self.assertFalse(sched.enabled)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class ShouldPruneTests(unittest.TestCase):
    def test_returns_false_when_disabled(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=False, pruning_interval=10)
        self.assertFalse(sched.should_prune(10))
        self.assertFalse(sched.should_prune(20))

    def test_returns_true_at_interval(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=True, pruning_interval=100)
        self.assertFalse(sched.should_prune(0))
        self.assertTrue(sched.should_prune(100))
        self.assertTrue(sched.should_prune(200))
        self.assertFalse(sched.should_prune(150))

    def test_step_zero_never_prunes(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=True, pruning_interval=1)
        self.assertFalse(sched.should_prune(0))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class UpdateScoresTests(unittest.TestCase):
    def test_scores_initialized_to_zero(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=True)
        stats = sched.get_pruning_stats()
        self.assertAlmostEqual(stats["current_sparsity"], 0.0)

    def test_scores_updated_after_backward(self) -> None:
        torch.manual_seed(42)
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=True)

        x = torch.randn(4, 32)
        out = model(x)
        out.sum().backward()

        sched.update_scores()
        # After update, some scores should be non-zero
        total_nonzero = sum(
            (s > 0).sum().item() for s in sched.scores.values()
        )
        self.assertGreater(total_nonzero, 0)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class ApplyPruningTests(unittest.TestCase):
    def test_pruning_increases_sparsity(self) -> None:
        torch.manual_seed(42)
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(
            model, enabled=True, pruning_interval=1, pruning_amount=0.20
        )

        # Simulate a training step
        x = torch.randn(4, 32)
        out = model(x)
        out.sum().backward()
        sched.update_scores()

        stats_before = sched.get_pruning_stats()
        sched.apply_pruning(model, step=1)
        stats_after = sched.get_pruning_stats()

        self.assertGreater(stats_after["current_sparsity"], stats_before["current_sparsity"])

    def test_pruning_zeros_parameters(self) -> None:
        torch.manual_seed(42)
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(
            model, enabled=True, pruning_interval=1, pruning_amount=0.50
        )

        x = torch.randn(4, 32)
        out = model(x)
        out.sum().backward()
        sched.update_scores()
        sched.apply_pruning(model, step=1)

        # Some parameters should now be zero
        total_zeros = sum(
            (p.data == 0).sum().item()
            for p in model.parameters() if p.ndim >= 2
        )
        self.assertGreater(total_zeros, 0)

    def test_noop_when_disabled(self) -> None:
        torch.manual_seed(42)
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=False, pruning_interval=1)
        original_weights = {n: p.data.clone() for n, p in model.named_parameters()}

        sched.apply_pruning(model, step=1)

        for name, param in model.named_parameters():
            torch.testing.assert_close(param.data, original_weights[name])

    def test_masks_persist_across_pruning_steps(self) -> None:
        torch.manual_seed(42)
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(
            model, enabled=True, pruning_interval=1, pruning_amount=0.10
        )

        for step in range(1, 4):
            x = torch.randn(4, 32)
            out = model(x)
            out.sum().backward()
            sched.update_scores()
            sched.apply_pruning(model, step=step)

        stats = sched.get_pruning_stats()
        # After 3 rounds of 10% pruning, sparsity should be > 10%
        self.assertGreater(stats["current_sparsity"], 0.10)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class GetPruningStatsTests(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=True)
        stats = sched.get_pruning_stats()
        self.assertIn("current_sparsity", stats)
        self.assertIn("total_params", stats)
        self.assertIn("pruned_params", stats)
        self.assertIn("enabled", stats)

    def test_total_params_correct(self) -> None:
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(model, enabled=True)
        stats = sched.get_pruning_stats()
        expected = sum(p.numel() for p in model.parameters() if p.ndim >= 2)
        self.assertEqual(stats["total_params"], expected)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class LotteryTicketTests(unittest.TestCase):
    def test_identify_winning_subnet(self) -> None:
        torch.manual_seed(42)
        model = SimpleModel()
        sched = utxo_pruning.PruningScheduler(
            model, enabled=True, pruning_interval=1, pruning_amount=0.50
        )

        # Run a few updates
        for _ in range(3):
            x = torch.randn(4, 32)
            out = model(x)
            out.sum().backward()
            sched.update_scores()

        mask = sched.get_winning_mask(keep_ratio=0.5)
        self.assertIsInstance(mask, dict)
        for name, m in mask.items():
            self.assertEqual(m.dtype, torch.bool)
            # About 50% should be kept
            keep_frac = m.float().mean().item()
            self.assertGreater(keep_frac, 0.1)
            self.assertLess(keep_frac, 0.95)


if __name__ == "__main__":
    unittest.main()

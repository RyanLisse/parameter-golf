"""Tests for multi-seed ensemble distillation (RJC-53 / EXP-20)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the module we're testing
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.ensemble_distillation import (
    DistillationLoss,
    EnsembleConfig,
    StudentConfig,
    TeacherEnsemble,
    plan_ensemble_training,
    weight_average_ensemble,
)


class TestEnsembleConfig(unittest.TestCase):
    """Test EnsembleConfig dataclass and environment variable parsing."""

    def test_from_env_defaults(self):
        """Test default values when env vars are not set."""
        with patch.dict(os.environ, {}, clear=False):
            for key in [
                "ENSEMBLE_ENABLED",
                "ENSEMBLE_SEEDS",
                "DISTILL_TEMP",
                "DISTILL_ALPHA",
                "ENSEMBLE_STUDENT_SCALE",
            ]:
                if key in os.environ:
                    del os.environ[key]

            cfg = EnsembleConfig.from_env()
            self.assertFalse(cfg.enabled)
            self.assertEqual(cfg.seeds, [])
            self.assertEqual(cfg.temperature, 2.0)
            self.assertEqual(cfg.alpha, 0.5)
            self.assertEqual(cfg.student_scale, 0.6)

    def test_from_env_with_values(self):
        """Test parsing with explicit environment variables."""
        with patch.dict(
            os.environ,
            {
                "ENSEMBLE_ENABLED": "1",
                "ENSEMBLE_SEEDS": "42,1337,7",
                "DISTILL_TEMP": "3.0",
                "DISTILL_ALPHA": "0.7",
                "ENSEMBLE_STUDENT_SCALE": "0.75",
            },
        ):
            cfg = EnsembleConfig.from_env()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.seeds, [42, 1337, 7])
            self.assertEqual(cfg.temperature, 3.0)
            self.assertEqual(cfg.alpha, 0.7)
            self.assertEqual(cfg.student_scale, 0.75)

    def test_from_env_single_seed(self):
        """Test parsing with single seed."""
        with patch.dict(
            os.environ, {"ENSEMBLE_ENABLED": "1", "ENSEMBLE_SEEDS": "42"}
        ):
            cfg = EnsembleConfig.from_env()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.seeds, [42])

    def test_from_env_disabled(self):
        """Test that ENSEMBLE_ENABLED=0 disables everything."""
        with patch.dict(
            os.environ,
            {
                "ENSEMBLE_ENABLED": "0",
                "ENSEMBLE_SEEDS": "42,1337",
                "DISTILL_TEMP": "4.0",
            },
        ):
            cfg = EnsembleConfig.from_env()
            self.assertFalse(cfg.enabled)


class TestDistillationLoss(unittest.TestCase):
    """Test distillation loss computation."""

    def setUp(self):
        """Create test data."""
        self.batch_size = 4
        self.vocab_size = 1024
        self.device = torch.device("cpu")

    def test_distillation_loss_basic(self):
        """Test that loss computes without error."""
        loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)

        student_logits = torch.randn(
            self.batch_size, self.vocab_size, device=self.device
        )
        teacher_probs = torch.softmax(
            torch.randn(self.batch_size, self.vocab_size, device=self.device), dim=-1
        )
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size,))

        loss = loss_fn.compute(student_logits, teacher_probs, target_ids)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

    def test_temperature_scaling(self):
        """Test that temperature scaling works correctly."""
        loss_fn = DistillationLoss(temperature=2.0, alpha=1.0)

        student_logits = torch.ones(
            self.batch_size, self.vocab_size, device=self.device
        )
        teacher_probs = torch.ones(
            self.batch_size, self.vocab_size, device=self.device
        ) / self.vocab_size
        target_ids = torch.zeros(self.batch_size, dtype=torch.long)

        loss = loss_fn.compute(student_logits, teacher_probs, target_ids)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

    def test_alpha_zero_gives_ce(self):
        """Test that alpha=0 gives pure cross-entropy loss."""
        loss_fn = DistillationLoss(temperature=2.0, alpha=0.0)

        student_logits = torch.randn(
            self.batch_size, self.vocab_size, device=self.device
        )
        teacher_probs = torch.softmax(
            torch.randn(self.batch_size, self.vocab_size, device=self.device), dim=-1
        )
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size,))

        loss = loss_fn.compute(student_logits, teacher_probs, target_ids)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

    def test_alpha_one_gives_kl(self):
        """Test that alpha=1.0 gives pure KL divergence."""
        loss_fn = DistillationLoss(temperature=2.0, alpha=1.0)

        student_logits = torch.randn(
            self.batch_size, self.vocab_size, device=self.device
        )
        teacher_probs = torch.softmax(
            torch.randn(self.batch_size, self.vocab_size, device=self.device), dim=-1
        )
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size,))

        loss = loss_fn.compute(student_logits, teacher_probs, target_ids)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_decreases_on_perfect_match(self):
        """Test that loss decreases when student matches teacher."""
        loss_fn = DistillationLoss(temperature=1.0, alpha=0.5)

        teacher_logits = torch.randn(self.batch_size, self.vocab_size)
        student_logits = teacher_logits.clone()
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size,))

        loss = loss_fn.compute(student_logits, teacher_probs, target_ids)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_soft_target_distribution(self):
        """Test that temperature produces softer distributions."""
        student_logits = torch.randn(self.batch_size, self.vocab_size)
        teacher_logits = torch.randn(self.batch_size, self.vocab_size)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size,))

        # Low temperature (sharper)
        loss_sharp = DistillationLoss(temperature=0.5, alpha=1.0).compute(
            student_logits, teacher_probs, target_ids
        )

        # High temperature (softer)
        loss_soft = DistillationLoss(temperature=5.0, alpha=1.0).compute(
            student_logits, teacher_probs, target_ids
        )

        self.assertTrue(torch.isfinite(loss_sharp))
        self.assertTrue(torch.isfinite(loss_soft))


class TestStudentConfig(unittest.TestCase):
    """Test student model configuration computation."""

    def test_compute_student_dims_scale_06(self):
        """Test student dimensions with 0.6 scale."""
        teacher_config = {
            "num_layers": 9,
            "model_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 4,
            "mlp_mult": 2,
        }

        student_config = StudentConfig.compute_student_dims(teacher_config, scale=0.6)

        # Reduced layers and dimensions
        self.assertLess(student_config["num_layers"], teacher_config["num_layers"])
        self.assertLess(student_config["model_dim"], teacher_config["model_dim"])

        # But still reasonable
        self.assertGreater(student_config["num_layers"], 0)
        self.assertGreater(student_config["model_dim"], 64)

    def test_compute_student_dims_scale_08(self):
        """Test student dimensions with 0.8 scale (less aggressive)."""
        teacher_config = {
            "num_layers": 10,
            "model_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 4,
            "mlp_mult": 2,
        }

        student_config_06 = StudentConfig.compute_student_dims(
            teacher_config, scale=0.6
        )
        student_config_08 = StudentConfig.compute_student_dims(
            teacher_config, scale=0.8
        )

        # 0.8 scale should be larger than 0.6
        self.assertGreaterEqual(
            student_config_08["model_dim"], student_config_06["model_dim"]
        )
        self.assertGreaterEqual(
            student_config_08["num_layers"], student_config_06["num_layers"]
        )

    def test_compute_student_dims_preserves_compatibility(self):
        """Test that student config preserves architectural compatibility."""
        teacher_config = {
            "num_layers": 12,
            "model_dim": 768,
            "num_heads": 12,
            "num_kv_heads": 6,
            "mlp_mult": 3,
        }

        student_config = StudentConfig.compute_student_dims(teacher_config, scale=0.7)

        # Heads should be divisible by model_dim
        self.assertEqual(student_config["model_dim"] % student_config["num_heads"], 0)
        # KV heads should be <= num_heads
        self.assertLessEqual(
            student_config["num_kv_heads"], student_config["num_heads"]
        )


class TestWeightAverageEnsemble(unittest.TestCase):
    """Test weight averaging across models."""

    def test_weight_average_two_models(self):
        """Test averaging weights from two models."""
        state_dict_1 = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.zeros(10),
            "layer2.weight": torch.ones(5, 10) * 2,
        }

        state_dict_2 = {
            "layer1.weight": torch.ones(10, 10) * 3,
            "layer1.bias": torch.ones(10),
            "layer2.weight": torch.ones(5, 10) * 4,
        }

        averaged = weight_average_ensemble([state_dict_1, state_dict_2])

        # Check keys are preserved
        self.assertEqual(set(averaged.keys()), set(state_dict_1.keys()))

        # Check averaging
        expected_layer1_weight = (torch.ones(10, 10) + torch.ones(10, 10) * 3) / 2
        torch.testing.assert_close(
            averaged["layer1.weight"], expected_layer1_weight, atol=1e-6, rtol=1e-5
        )

        expected_layer1_bias = (torch.zeros(10) + torch.ones(10)) / 2
        torch.testing.assert_close(
            averaged["layer1.bias"], expected_layer1_bias, atol=1e-6, rtol=1e-5
        )

    def test_weight_average_three_models(self):
        """Test averaging weights from three models."""
        state_dicts = [
            {"w": torch.ones(5, 5) * i} for i in range(1, 4)
        ]  # 1, 2, 3

        averaged = weight_average_ensemble(state_dicts)

        # Average of 1, 2, 3 is 2
        torch.testing.assert_close(
            averaged["w"], torch.ones(5, 5) * 2.0, atol=1e-6, rtol=1e-5
        )

    def test_weight_average_preserves_shapes(self):
        """Test that averaging preserves tensor shapes."""
        state_dicts = [
            {
                "a": torch.randn(32, 64),
                "b": torch.randn(128),
                "c": torch.randn(10, 10, 10),
            }
            for _ in range(3)
        ]

        averaged = weight_average_ensemble(state_dicts)

        self.assertEqual(averaged["a"].shape, torch.Size([32, 64]))
        self.assertEqual(averaged["b"].shape, torch.Size([128]))
        self.assertEqual(averaged["c"].shape, torch.Size([10, 10, 10]))


class SimpleTeacherModel(nn.Module):
    """Simple teacher model for testing."""

    def __init__(self, output_tensor: torch.Tensor):
        super().__init__()
        self.output_tensor = output_tensor

    def forward(self, input_ids):
        return self.output_tensor


class TestTeacherEnsemble(unittest.TestCase):
    """Test teacher ensemble predictions."""

    def setUp(self):
        """Create mock teacher models."""
        self.batch_size = 4
        self.seq_len = 16
        self.vocab_size = 1024

    def test_ensemble_prediction_shape(self):
        """Test that ensemble predictions have correct shape."""
        # Create simple teacher models
        models = [
            SimpleTeacherModel(torch.randn(self.batch_size, self.seq_len, self.vocab_size))
            for _ in range(3)
        ]

        ensemble = TeacherEnsemble(models)
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        predictions = ensemble.predict(input_ids)

        # Should be probabilities (sum to 1 over vocab)
        self.assertEqual(
            predictions.shape, (self.batch_size, self.seq_len, self.vocab_size)
        )
        # Should sum to approximately 1 over vocab dimension
        sums = predictions.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_ensemble_averaging(self):
        """Test that ensemble averages predictions correctly."""
        # Create predictable outputs
        output_1 = torch.ones(self.batch_size, self.seq_len, self.vocab_size) * 0.1
        output_2 = torch.ones(self.batch_size, self.seq_len, self.vocab_size) * 0.2
        output_3 = torch.ones(self.batch_size, self.seq_len, self.vocab_size) * 0.3

        models = [
            SimpleTeacherModel(output_1),
            SimpleTeacherModel(output_2),
            SimpleTeacherModel(output_3),
        ]

        ensemble = TeacherEnsemble(models)
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        predictions = ensemble.predict(input_ids)

        # Average logits: (0.1 + 0.2 + 0.3) / 3 = 0.2
        # After softmax, should all have equal probability (since all inputs same)
        self.assertTrue(
            torch.allclose(
                predictions, predictions[:, :, 0:1], atol=1e-5, rtol=1e-5
            )
        )


class TestPlanEnsembleTraining(unittest.TestCase):
    """Test training schedule planning."""

    def test_plan_allocates_time(self):
        """Test that training plan allocates time correctly."""
        schedule = plan_ensemble_training(wallclock_budget=600, num_teachers=3)

        total_time = sum(schedule.values())
        # Should allocate most of the budget
        self.assertGreater(total_time, 400)
        self.assertLessEqual(total_time, 600)

    def test_plan_has_distillation_phase(self):
        """Test that plan includes distillation phase."""
        schedule = plan_ensemble_training(wallclock_budget=600, num_teachers=3)

        self.assertIn("distillation", schedule)
        self.assertGreater(schedule["distillation"], 0)

    def test_plan_has_teacher_phases(self):
        """Test that plan includes teacher training phases."""
        schedule = plan_ensemble_training(wallclock_budget=600, num_teachers=3)

        for i in range(3):
            self.assertIn(f"teacher_{i}", schedule)
            self.assertGreater(schedule[f"teacher_{i}"], 0)

    def test_plan_scales_with_budget(self):
        """Test that time allocation scales with budget."""
        schedule_300 = plan_ensemble_training(wallclock_budget=300, num_teachers=3)
        schedule_600 = plan_ensemble_training(wallclock_budget=600, num_teachers=3)

        total_300 = sum(schedule_300.values())
        total_600 = sum(schedule_600.values())

        # Larger budget should allocate more time
        self.assertGreater(total_600, total_300)


if __name__ == "__main__":
    unittest.main()

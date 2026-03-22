"""Multi-Seed Ensemble Distillation (RJC-53 / EXP-20).

Train multiple teacher models with different seeds, distill into a single smaller
student model. Averaging predictions across seeds consistently reduces BPB.
The student model fits in 16MB while benefiting from teacher ensemble diversity.

Environment variables:
  ENSEMBLE_ENABLED: Enable ensemble distillation (0/1). Default: 0
  ENSEMBLE_SEEDS: Comma-separated seed values for teachers. Default: "" (disabled)
  DISTILL_TEMP: Temperature for knowledge distillation (float). Default: 2.0
  DISTILL_ALPHA: Weight for KL loss vs CE loss (0-1). Default: 0.5
  ENSEMBLE_STUDENT_SCALE: Student model scale relative to teacher (0-1). Default: 0.6
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class EnsembleConfig:
    """Configuration for ensemble distillation from environment variables."""

    enabled: bool = False
    seeds: list[int] = field(default_factory=list)
    temperature: float = 2.0
    alpha: float = 0.5
    student_scale: float = 0.6

    @classmethod
    def from_env(cls) -> EnsembleConfig:
        """Load configuration from environment variables."""
        enabled = bool(int(os.environ.get("ENSEMBLE_ENABLED", "0")))
        seeds_str = os.environ.get("ENSEMBLE_SEEDS", "")

        seeds = []
        if seeds_str.strip():
            try:
                seeds = [int(s.strip()) for s in seeds_str.split(",")]
            except ValueError:
                seeds = []

        temperature = float(os.environ.get("DISTILL_TEMP", "2.0"))
        alpha = float(os.environ.get("DISTILL_ALPHA", "0.5"))
        student_scale = float(os.environ.get("ENSEMBLE_STUDENT_SCALE", "0.6"))

        return cls(
            enabled=enabled,
            seeds=seeds,
            temperature=temperature,
            alpha=alpha,
            student_scale=student_scale,
        )


@dataclass
class StudentConfig:
    """Configuration for student model architecture."""

    @staticmethod
    def compute_student_dims(
        teacher_config: dict[str, Any], scale: float = 0.6
    ) -> dict[str, Any]:
        """
        Compute student model dimensions based on teacher config and scale factor.

        Args:
            teacher_config: Teacher model configuration dict with keys:
                num_layers, model_dim, num_heads, num_kv_heads, mlp_mult
            scale: Scale factor (0-1) for reducing model size

        Returns:
            Student model configuration dict with same keys, scaled appropriately
        """
        student_config = {}

        # Scale layers (at least 1, roughly scale*teacher_layers)
        teacher_layers = teacher_config.get("num_layers", 9)
        student_config["num_layers"] = max(
            1, int(teacher_layers * scale * 0.9)
        )  # Slightly more aggressive for layers

        # Scale model dimension (must maintain divisibility)
        teacher_dim = teacher_config.get("model_dim", 512)
        scaled_dim = int(teacher_dim * scale)
        # Round to nearest multiple of 64 for better hardware efficiency
        student_config["model_dim"] = max(64, (scaled_dim // 64) * 64)

        # Scale heads proportionally
        teacher_heads = teacher_config.get("num_heads", 8)
        scaled_heads = max(1, int(teacher_heads * scale))
        # Ensure divisibility by model_dim
        student_config["num_heads"] = min(
            scaled_heads, student_config["model_dim"] // 64
        )
        student_config["num_heads"] = max(
            1, student_config["num_heads"]
        )  # At least 1

        # Scale KV heads
        teacher_kv_heads = teacher_config.get("num_kv_heads", 4)
        student_config["num_kv_heads"] = min(
            max(1, int(teacher_kv_heads * scale)),
            student_config["num_heads"],
        )

        # Keep MLP multiplier same (compresses less aggressively)
        student_config["mlp_mult"] = teacher_config.get("mlp_mult", 2)

        return student_config


class DistillationLoss(torch.nn.Module):
    """Knowledge distillation loss combining KL divergence and cross-entropy."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        Initialize distillation loss.

        Args:
            temperature: Temperature for soft target distribution (>1 = softer)
            alpha: Weight for KL divergence (0=pure CE, 1=pure KL)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute blended distillation loss.

        Args:
            student_logits: Student model logits, shape (batch, vocab) or (batch, seq, vocab)
            teacher_probs: Teacher ensemble probabilities, same shape as student_logits
            target_ids: Ground truth token IDs

        Returns:
            Scalar loss tensor
        """
        # Flatten if needed for easier computation
        original_shape = student_logits.shape
        if len(original_shape) == 3:
            batch, seq_len, vocab = original_shape
            student_logits = student_logits.reshape(batch * seq_len, vocab)
            teacher_probs = teacher_probs.reshape(batch * seq_len, vocab)
            target_ids = target_ids.reshape(batch * seq_len)

        # Soft targets: apply temperature scaling to teacher probabilities
        # Higher temperature -> softer distribution (more uniform)
        soft_targets = torch.pow(teacher_probs, 1.0 / self.temperature)
        soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-10)

        # KL divergence: KL(soft_targets || student_soft)
        student_soft = F.softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            soft_targets,
            reduction="batchmean",
        )

        # Cross-entropy: CE(student, targets)
        ce_loss = F.cross_entropy(student_logits, target_ids, reduction="mean")

        # Blended loss
        loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return loss


class TeacherEnsemble(torch.nn.Module):
    """Ensemble of teacher models for knowledge distillation."""

    def __init__(self, teacher_models: list[torch.nn.Module]):
        """
        Initialize ensemble with teacher models.

        Args:
            teacher_models: List of trained teacher models
        """
        super().__init__()
        self.teachers = torch.nn.ModuleList(teacher_models)

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get averaged probability predictions from ensemble.

        Args:
            input_ids: Input token IDs

        Returns:
            Averaged softmax probabilities across teachers
        """
        logits_list = []

        for teacher in self.teachers:
            with torch.no_grad():
                logits = teacher(input_ids)  # (batch, seq, vocab)
                logits_list.append(logits)

        # Average logits before softmax (more stable)
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return F.softmax(avg_logits, dim=-1)

    def predict_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get averaged logits from ensemble.

        Args:
            input_ids: Input token IDs

        Returns:
            Averaged logits across teachers
        """
        logits_list = []

        for teacher in self.teachers:
            with torch.no_grad():
                logits = teacher(input_ids)
                logits_list.append(logits)

        return torch.stack(logits_list, dim=0).mean(dim=0)


def weight_average_ensemble(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Average weights across multiple model state dictionaries.

    Simple alternative to distillation when teacher architectures are identical.

    Args:
        state_dicts: List of model state dicts to average

    Returns:
        Averaged state dict
    """
    if not state_dicts:
        raise ValueError("No state dicts provided")

    averaged = {}
    num_models = len(state_dicts)

    for key in state_dicts[0].keys():
        # Stack tensors and average
        tensors = [sd[key] for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)
        averaged[key] = stacked.mean(dim=0)

    return averaged


def plan_ensemble_training(
    wallclock_budget: float = 600.0, num_teachers: int = 3
) -> dict[str, float]:
    """
    Plan training schedule for ensemble distillation.

    Allocates time budget across teacher training and distillation.

    Args:
        wallclock_budget: Total wallclock time budget in seconds
        num_teachers: Number of teacher models to train

    Returns:
        Dict mapping phase names to allocated time in seconds
    """
    # Allocate 40% per teacher (if parallel, can run in parallel time windows)
    # Total: 3 * 40% = 120% conceptually, but we only have 600s total
    # So: split into teacher time + distill time
    # Strategy: 45% for parallel teachers (each gets 45%/3 = 15% = 90s)
    # 20% for distillation (120s)
    # 35% buffer/overhead

    teacher_fraction = 0.45  # 45% total for all teachers
    distill_fraction = 0.20  # 20% for distillation
    # 35% buffer for overhead, eval, checkpointing

    schedule = {}

    # Each teacher gets equal time
    time_per_teacher = (wallclock_budget * teacher_fraction) / num_teachers
    for i in range(num_teachers):
        schedule[f"teacher_{i}"] = time_per_teacher

    # Distillation phase
    schedule["distillation"] = wallclock_budget * distill_fraction

    # Overhead/buffer (evaluation, checkpointing)
    schedule["overhead"] = wallclock_budget * 0.35

    return schedule

"""UTXO-Style Parameter Pruning (RJC-34).

Tracks parameter contribution scores and progressively prunes during training.
Controlled by env vars: PRUNING_ENABLED, PRUNING_START_RATIO, PRUNING_INTERVAL, PRUNING_AMOUNT.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class PruningScheduler:
    """Track parameter contribution scores and progressively prune.

    Contribution score: EMA of ``|weight * gradient|``.
    Every ``pruning_interval`` steps, zeros out the bottom ``pruning_amount`` fraction
    of parameters (by score) across all prunable layers.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        enabled: bool = False,
        start_ratio: float = 1.0,
        pruning_interval: int = 1000,
        pruning_amount: float = 0.10,
        ema_decay: float = 0.99,
    ) -> None:
        self.enabled = enabled
        self.start_ratio = start_ratio
        self.pruning_interval = pruning_interval
        self.pruning_amount = pruning_amount
        self.ema_decay = ema_decay

        # Track scores and masks for prunable params (ndim >= 2)
        self.scores: dict[str, Tensor] = {}
        self.masks: dict[str, Tensor] = {}
        self._prunable_names: list[str] = []

        self._model_ref: nn.Module | None = model

        for name, param in model.named_parameters():
            if param.ndim >= 2:
                self._prunable_names.append(name)
                self.scores[name] = torch.zeros_like(param.data)
                self.masks[name] = torch.ones_like(param.data, dtype=torch.bool)

    @classmethod
    def from_env(cls, model: nn.Module, env: dict[str, str] | None = None) -> PruningScheduler:
        """Construct from environment variables (or provided dict)."""
        e = env if env is not None else dict(os.environ)
        return cls(
            model,
            enabled=e.get("PRUNING_ENABLED", "0") == "1",
            start_ratio=float(e.get("PRUNING_START_RATIO", "1.0")),
            pruning_interval=int(e.get("PRUNING_INTERVAL", "1000")),
            pruning_amount=float(e.get("PRUNING_AMOUNT", "0.10")),
        )

    def should_prune(self, step: int) -> bool:
        """Return True if pruning should be applied at this step."""
        if not self.enabled or step == 0:
            return False
        return step % self.pruning_interval == 0

    def update_scores(self, model: nn.Module | None = None) -> None:
        """Update EMA of |weight * gradient| importance scores.

        Must be called after ``loss.backward()`` and before ``optimizer.step()``.
        If *model* is not passed, uses the model reference from construction.
        """
        m = model if model is not None else self._model_ref
        if m is None:
            return
        for name, param in m.named_parameters():
            if name not in self.scores:
                continue
            if param.grad is None:
                continue
            importance = (param.data * param.grad).abs()
            self.scores[name] = (
                self.ema_decay * self.scores[name] + (1 - self.ema_decay) * importance
            )

    def apply_pruning(self, model: nn.Module, step: int) -> None:
        """Zero out the bottom ``pruning_amount`` fraction of parameters by score.

        Masks are cumulative: once pruned, a parameter stays pruned.
        """
        if not self.enabled:
            return
        if not self.should_prune(step):
            return

        # Gather all scores for currently-alive parameters
        all_scores: list[float] = []
        for name in self._prunable_names:
            alive = self.masks[name]
            if alive.any():
                all_scores.extend(self.scores[name][alive].flatten().tolist())

        if not all_scores:
            return

        all_scores_t = torch.tensor(all_scores)
        # Find the threshold: bottom pruning_amount fraction
        k = max(1, int(len(all_scores) * self.pruning_amount))
        threshold = torch.topk(all_scores_t, k, largest=False).values[-1].item()

        # Apply masks
        param_dict = dict(model.named_parameters())
        for name in self._prunable_names:
            if name not in param_dict:
                continue
            # Prune where score <= threshold AND currently alive
            prune_mask = (self.scores[name] <= threshold) & self.masks[name]
            self.masks[name] = self.masks[name] & ~prune_mask
            param_dict[name].data[prune_mask] = 0.0

    def get_pruning_stats(self) -> dict[str, Any]:
        """Return current sparsity metrics."""
        total = 0
        pruned = 0
        for name in self._prunable_names:
            total += self.masks[name].numel()
            pruned += (~self.masks[name]).sum().item()

        return {
            "enabled": self.enabled,
            "total_params": total,
            "pruned_params": int(pruned),
            "current_sparsity": pruned / total if total > 0 else 0.0,
            "pruning_interval": self.pruning_interval,
            "pruning_amount": self.pruning_amount,
        }

    def get_winning_mask(self, keep_ratio: float = 0.5) -> dict[str, Tensor]:
        """Identify the winning sub-network (lottery ticket).

        Returns boolean masks where True = keep the parameter.
        """
        all_scores: list[float] = []
        for name in self._prunable_names:
            all_scores.extend(self.scores[name].flatten().tolist())

        if not all_scores:
            return {}

        all_scores_t = torch.tensor(all_scores)
        k = max(1, int(len(all_scores) * keep_ratio))
        threshold = torch.topk(all_scores_t, k, largest=True).values[-1].item()

        result: dict[str, Tensor] = {}
        for name in self._prunable_names:
            result[name] = self.scores[name] >= threshold
        return result

    def enforce_masks(self, model: nn.Module) -> None:
        """Re-apply masks after optimizer step to keep pruned weights at zero."""
        param_dict = dict(model.named_parameters())
        for name in self._prunable_names:
            if name in param_dict:
                param_dict[name].data[~self.masks[name]] = 0.0

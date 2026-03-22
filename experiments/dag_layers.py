"""DAG Multi-Path Transformer Layers (RJC-32).

Replace sequential layer stack with a DAG where layers run in parallel branches.
Inspired by DenseNet-style skip connections and multi-path architectures.

Environment variables:
    DAG_ENABLED: Enable DAG architecture (0 or 1, default 0)
    DAG_BRANCH_START: Layer index where branching begins (default 3)
    DAG_BRANCH_END: Layer index where branching ends (default 5)
    DAG_NUM_BRANCHES: Number of parallel branches (default 2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DAGConfig:
    """Configuration for DAG multi-path architecture."""

    enabled: bool = False
    branch_start: int = 3
    branch_end: int = 5
    num_branches: int = 2

    def __init__(self):
        """Initialize from environment variables."""
        self.enabled = bool(int(os.environ.get("DAG_ENABLED", "0")))
        self.branch_start = int(os.environ.get("DAG_BRANCH_START", "3"))
        self.branch_end = int(os.environ.get("DAG_BRANCH_END", "5"))
        self.num_branches = int(os.environ.get("DAG_NUM_BRANCHES", "2"))


class ParallelBranch(nn.Module):
    """Wrapper for a Block with independent parameters in a parallel branch."""

    def __init__(self, block_factory: Callable):
        """Initialize a parallel branch.

        Args:
            block_factory: Callable that creates a new Block instance
        """
        super().__init__()
        self.block = block_factory()
        # Store layers for this branch
        self.layers = nn.ModuleList()

    def add_layer(self, block):
        """Add a layer to this branch."""
        self.layers.append(block)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Forward pass through all layers in this branch.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            x0: Initial residual from embedding

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        for layer in self.layers:
            x = layer(x, x0)
        return x


class BranchMerger(nn.Module):
    """Merge multiple branch outputs with learned mixing weights."""

    def __init__(self, num_branches: int, dim: int):
        """Initialize branch merger.

        Args:
            num_branches: Number of branches to merge
            dim: Model dimension
        """
        super().__init__()
        self.num_branches = num_branches
        self.dim = dim
        # Learnable mixing weights (will be softmax normalized)
        self.merge_weights = nn.Parameter(torch.ones(num_branches))

    def forward(self, branch_outputs: list[Tensor]) -> Tensor:
        """Merge branch outputs using learned weights.

        Args:
            branch_outputs: List of tensors from each branch, shape (batch, seq_len, dim)

        Returns:
            Merged tensor of shape (batch, seq_len, dim)
        """
        # Normalize weights with softmax
        weights = torch.softmax(self.merge_weights, dim=0)

        # Weighted sum of branches
        output = sum(w * branch_out for w, branch_out in zip(weights, branch_outputs))
        return output


class DAGBlock(nn.Module):
    """DAG-based multi-path transformer block with DenseNet-style skip connections.

    Architecture:
        - Layers 0 to branch_start-1: Sequential processing (local patterns)
        - Layers branch_start to branch_end: Parallel branches (diverse representations)
        - Layers branch_end+1 onwards: Sequential with merged branch output
        - Dense skip connections: Each layer receives concat of all previous via bottleneck
    """

    def __init__(
        self,
        num_layers: int,
        block_factory: Callable,
        config: dict,
        dim: int,
    ):
        """Initialize DAG block.

        Args:
            num_layers: Total number of transformer layers
            block_factory: Callable that creates a new Block instance
            config: DAG configuration dict with enabled, branch_start, branch_end, num_branches
            dim: Model dimension
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.enabled = config.get("enabled", False)
        self.branch_start = config.get("branch_start", 3)
        self.branch_end = config.get("branch_end", 5)
        self.num_branches = config.get("num_branches", 2)

        if not self.enabled:
            # Fall back to sequential processing
            self.sequential_blocks = nn.ModuleList([block_factory() for _ in range(num_layers)])
            self.branches = None
            self.merger = None
            self.skip_bottlenecks = None
            self.post_branch_blocks = None
            return

        # Pre-branch sequential layers
        self.pre_branch_blocks = nn.ModuleList([block_factory() for _ in range(self.branch_start)])

        # Parallel branches
        self.branches = nn.ModuleList()
        branch_layers = self.branch_end - self.branch_start + 1
        for _ in range(self.num_branches):
            branch = ParallelBranch(block_factory)
            for _ in range(branch_layers):
                branch.add_layer(block_factory())
            self.branches.append(branch)

        # Branch merger
        self.merger = BranchMerger(self.num_branches, dim)

        # Post-branch sequential layers
        post_branch_count = num_layers - (self.branch_end + 1)
        if post_branch_count > 0:
            self.post_branch_blocks = nn.ModuleList([block_factory() for _ in range(post_branch_count)])
        else:
            self.post_branch_blocks = None

        # DenseNet-style skip connections via bottleneck
        # Bottleneck projects concatenated previous layers back to dim
        # Maximum concat size is num_layers * dim (all previous layers)
        max_concat_dim = dim * num_layers
        self.skip_bottlenecks = nn.ModuleList(
            [nn.Linear(max_concat_dim, dim) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Forward pass through DAG architecture.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            x0: Initial residual from embedding

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        if not self.enabled:
            # Sequential fallback
            for block in self.sequential_blocks:
                x = block(x, x0)
            return x

        # Track all layer outputs for dense skip connections
        layer_outputs = []

        # Pre-branch sequential layers
        for i, block in enumerate(self.pre_branch_blocks):
            # Dense skip: concat all previous layers, pad to max_concat_dim
            if layer_outputs:
                skip_input = torch.cat([x] + layer_outputs, dim=-1)
                # Pad to expected dimension
                pad_size = (self.num_layers * self.dim) - skip_input.size(-1)
                if pad_size > 0:
                    skip_input = torch.nn.functional.pad(skip_input, (0, pad_size))
                x = x + self.skip_bottlenecks[i](skip_input)
            x = block(x, x0)
            layer_outputs.append(x)

        # Parallel branches
        branch_input = x
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(branch_input, x0)
            branch_outputs.append(branch_out)

        # Merge branches
        x = self.merger(branch_outputs)
        layer_outputs.append(x)

        # Post-branch sequential layers
        if self.post_branch_blocks:
            for i, block in enumerate(self.post_branch_blocks):
                idx = self.branch_end + 1 + i
                if layer_outputs:
                    skip_input = torch.cat([x] + layer_outputs, dim=-1)
                    # Pad to expected dimension
                    pad_size = (self.num_layers * self.dim) - skip_input.size(-1)
                    if pad_size > 0:
                        skip_input = torch.nn.functional.pad(skip_input, (0, pad_size))
                    x = x + self.skip_bottlenecks[idx](skip_input)
                x = block(x, x0)
                layer_outputs.append(x)

        return x


def build_dag_model(config: DAGConfig, base_gpt_args: dict):
    """Build a GPT model with DAG topology if enabled.

    Args:
        config: DAG configuration
        base_gpt_args: Dictionary of GPT constructor arguments

    Returns:
        Modified GPT model with DAG architecture, or None if disabled
    """
    if not config.enabled:
        return None

    # Import here to avoid circular dependency
    import sys
    from pathlib import Path

    # Add parent directory to path to import train_gpt
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from train_gpt import Block, GPT

    # Create modified GPT with DAG blocks
    class DAGGPT(GPT):
        """GPT with DAG multi-path architecture."""

        def __init__(self, **kwargs):
            # Extract DAG config
            dag_config_dict = {
                "enabled": config.enabled,
                "branch_start": config.branch_start,
                "branch_end": config.branch_end,
                "num_branches": config.num_branches,
            }

            # Initialize parent without blocks
            num_layers = kwargs["num_layers"]
            model_dim = kwargs["model_dim"]
            num_heads = kwargs["num_heads"]
            num_kv_heads = kwargs["num_kv_heads"]
            mlp_mult = kwargs["mlp_mult"]
            rope_base = kwargs["rope_base"]
            qk_gain_init = kwargs["qk_gain_init"]

            # Call parent init
            super().__init__(**kwargs)

            # Replace blocks with DAG
            moe_experts = kwargs.get("moe_experts", 1)
            moe_deploy_expert = kwargs.get("moe_deploy_expert", -1)

            def block_factory():
                return Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    moe_experts,
                    moe_deploy_expert,
                )

            self.dag_blocks = DAGBlock(num_layers, block_factory, dag_config_dict, model_dim)

        def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
            """Forward pass through DAG model."""
            x = self.tok_emb(input_ids)
            x = torch.nn.functional.rms_norm(x, (x.size(-1),))
            x0 = x

            # Use DAG blocks instead of sequential
            x = self.dag_blocks(x, x0)

            # Final norm and head
            x = self.final_norm(x)
            if self.tie_embeddings:
                logits = torch.nn.functional.linear(x, self.tok_emb.weight)
            else:
                logits = self.lm_head(x)

            # Apply softcap
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )
            return loss

    return DAGGPT(**base_gpt_args)

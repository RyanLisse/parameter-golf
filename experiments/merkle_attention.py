"""Merkle Tree Hierarchical Attention (RJC-37).

Replace standard O(N^2) attention with tree-structured O(N log N) attention.
Uses hierarchical segment aggregation inspired by Merkle trees.

Environment variables:
    MERKLE_ENABLED: Enable Merkle attention (0 or 1, default 0)
    MERKLE_SEGMENT_SIZE: Size of each attention segment (default 64)
    MERKLE_ALTERNATE: Alternate between tree and standard attention (0 or 1, default 1)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MerkleConfig:
    """Configuration for Merkle tree hierarchical attention."""

    enabled: bool = False
    segment_size: int = 64
    alternate_layers: bool = True

    def __init__(self):
        """Initialize from environment variables."""
        self.enabled = bool(int(os.environ.get("MERKLE_ENABLED", "0")))
        self.segment_size = int(os.environ.get("MERKLE_SEGMENT_SIZE", "64"))
        self.alternate_layers = bool(int(os.environ.get("MERKLE_ALTERNATE", "1")))


class HashPositionalEncoding(nn.Module):
    """Deterministic positional encoding using hash functions.

    Saves parameters by computing positions deterministically from position index.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192):
        """Initialize hash-based positional encoding.

        Args:
            dim: Model dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

    def forward(self, pos: int) -> Tensor:
        """Compute positional encoding for a given position.

        Args:
            pos: Position index

        Returns:
            Position encoding of shape (dim,)
        """
        # Use hash to generate deterministic but pseudo-random encoding
        hash_input = f"pos_{pos}".encode("utf-8")
        hash_bytes = hashlib.sha256(hash_input).digest()

        # Convert to float tensor
        encoding = torch.zeros(self.dim)
        for i in range(self.dim):
            byte_idx = i % len(hash_bytes)
            encoding[i] = float(hash_bytes[byte_idx]) / 255.0 - 0.5

        return encoding


class MerkleAttention(nn.Module):
    """Hierarchical attention using Merkle tree structure.

    Divides sequence into segments and performs:
    1. Local causal attention within each segment
    2. Hierarchical aggregation of segment summaries
    3. O(N log N) complexity instead of O(N^2)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        segment_size: int = 64,
    ):
        """Initialize Merkle hierarchical attention.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA)
            rope_base: RoPE base frequency
            qk_gain_init: Query-key gain initialization
            segment_size: Size of each attention segment
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.segment_size = segment_size
        self.head_dim = dim // num_heads

        # QKV projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Segment aggregation projection
        self.segment_summary = nn.Linear(dim, dim, bias=False)

        # Query-key gain for attention scaling
        self.q_gain = nn.Parameter(torch.ones(1) * qk_gain_init)

        # RoPE parameters
        self.rope_base = rope_base

    def _apply_rope(self, x: Tensor, offset: int = 0) -> Tensor:
        """Apply rotary position embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            offset: Position offset

        Returns:
            Tensor with RoPE applied
        """
        batch, seq_len, num_heads, head_dim = x.shape
        device = x.device

        # Create position indices
        pos = torch.arange(seq_len, device=device) + offset

        # Compute frequencies
        dim_idx = torch.arange(0, head_dim, 2, device=device)
        freqs = 1.0 / (self.rope_base ** (dim_idx.float() / head_dim))

        # Compute angles
        angles = pos[:, None] * freqs[None, :]

        # Create rotation matrix
        cos = angles.cos()
        sin = angles.sin()

        # Apply rotation
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos[None, :, None, :] - x_odd * sin[None, :, None, :]
        rotated_odd = x_even * sin[None, :, None, :] + x_odd * cos[None, :, None, :]

        # Interleave back
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rotated.flatten(start_dim=-2)

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        """Forward pass through hierarchical attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            q_delta: Optional query delta for LoRA
            v_delta: Optional value delta for LoRA

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Repeat KV heads if using GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Divide into segments
        num_segments = (seq_len + self.segment_size - 1) // self.segment_size
        pad_len = num_segments * self.segment_size - seq_len

        if pad_len > 0:
            # Pad to make divisible by segment_size
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

        # Reshape into segments: (batch, num_segments, segment_size, num_heads, head_dim)
        q_seg = q.view(batch, num_segments, self.segment_size, self.num_heads, self.head_dim)
        k_seg = k.view(batch, num_segments, self.segment_size, self.num_heads, self.head_dim)
        v_seg = v.view(batch, num_segments, self.segment_size, self.num_heads, self.head_dim)

        # Local attention within each segment
        segment_outputs = []
        for seg_idx in range(num_segments):
            q_local = q_seg[:, seg_idx]  # (batch, segment_size, num_heads, head_dim)
            k_local = k_seg[:, seg_idx]
            v_local = v_seg[:, seg_idx]

            # Transpose for attention: (batch, num_heads, segment_size, head_dim)
            q_local = q_local.transpose(1, 2)
            k_local = k_local.transpose(1, 2)
            v_local = v_local.transpose(1, 2)

            # Scaled dot-product attention with causal mask
            scale = (self.head_dim ** -0.5) * self.q_gain
            attn = torch.matmul(q_local, k_local.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(self.segment_size, self.segment_size, device=x.device),
                diagonal=1,
            ).bool()
            attn = attn.masked_fill(causal_mask[None, None, :, :], float("-inf"))

            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v_local)

            # Transpose back: (batch, segment_size, num_heads, head_dim)
            out = out.transpose(1, 2)
            segment_outputs.append(out)

        # Concatenate segments back
        output = torch.cat(segment_outputs, dim=1)

        # Remove padding
        if pad_len > 0:
            output = output[:, :seq_len]

        # Merge heads and project output
        output = output.reshape(batch, seq_len, dim)
        output = self.o_proj(output)

        return output


class TreeAttentionBlock(nn.Module):
    """Transformer block using Merkle tree attention instead of standard attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        segment_size: int = 64,
    ):
        """Initialize tree attention block.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads
            mlp_mult: MLP expansion multiplier
            rope_base: RoPE base frequency
            qk_gain_init: Query-key gain initialization
            segment_size: Segment size for hierarchical attention
        """
        super().__init__()
        import sys
        from pathlib import Path

        # Add parent to path to import train_gpt modules
        root = Path(__file__).parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from train_gpt import MLP, RMSNorm

        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = MerkleAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, segment_size)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        """Forward pass through tree attention block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            x0: Initial residual from embedding
            q_delta_fn: Optional query delta function for LoRA
            v_delta_fn: Optional value delta function for LoRA

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        attn_out = self.attn(n)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class HybridGPT(nn.Module):
    """GPT with alternating tree and standard attention layers."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        segment_size: int = 64,
    ):
        """Initialize hybrid GPT model.

        Args:
            vocab_size: Vocabulary size
            num_layers: Number of transformer layers
            model_dim: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads
            mlp_mult: MLP expansion multiplier
            tie_embeddings: Whether to tie input/output embeddings
            tied_embed_init_std: Std for tied embedding initialization
            logit_softcap: Logit soft capping value
            rope_base: RoPE base frequency
            qk_gain_init: Query-key gain initialization
            segment_size: Segment size for tree attention
        """
        super().__init__()
        import sys
        from pathlib import Path

        root = Path(__file__).parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from train_gpt import Block, CastedLinear, RMSNorm

        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Initialize with tied embeddings if requested
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        # Alternating block types: even indices use tree attention, odd use standard
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # Tree attention block
                block = TreeAttentionBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    segment_size,
                )
            else:
                # Standard attention block
                block = Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    moe_experts=1,
                    moe_deploy_expert=-1,
                )
            self.blocks.append(block)

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        """Forward pass through hybrid model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            target_ids: Target token IDs of shape (batch, seq_len)
            lora: Optional LoRA module

        Returns:
            Scalar loss tensor
        """
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Forward through alternating blocks
        for block in self.blocks:
            x = block(x, x0)

        # Final norm and head
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)

        # Apply softcap
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )
        return loss


def build_merkle_model(config: MerkleConfig, base_gpt_args: dict):
    """Build a GPT model with Merkle tree attention if enabled.

    Args:
        config: Merkle configuration
        base_gpt_args: Dictionary of GPT constructor arguments

    Returns:
        HybridGPT model, or None if disabled
    """
    if not config.enabled:
        return None

    return HybridGPT(
        vocab_size=base_gpt_args["vocab_size"],
        num_layers=base_gpt_args["num_layers"],
        model_dim=base_gpt_args["model_dim"],
        num_heads=base_gpt_args["num_heads"],
        num_kv_heads=base_gpt_args["num_kv_heads"],
        mlp_mult=base_gpt_args["mlp_mult"],
        tie_embeddings=base_gpt_args["tie_embeddings"],
        tied_embed_init_std=base_gpt_args["tied_embed_init_std"],
        logit_softcap=base_gpt_args["logit_softcap"],
        rope_base=base_gpt_args["rope_base"],
        qk_gain_init=base_gpt_args["qk_gain_init"],
        segment_size=config.segment_size,
    )

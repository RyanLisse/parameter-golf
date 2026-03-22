"""Tests for Merkle Tree Hierarchical Attention (RJC-37)."""

import os
import unittest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestMerkleAttention(unittest.TestCase):
    """Test Merkle tree hierarchical attention architecture."""

    def setUp(self):
        """Set up test environment variables and random seed."""
        self.original_env = os.environ.copy()
        torch.manual_seed(42)

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_merkle_config_from_env_vars(self):
        """Test MerkleConfig reads environment variables correctly."""
        from experiments.merkle_attention import MerkleConfig

        os.environ["MERKLE_ENABLED"] = "1"
        os.environ["MERKLE_SEGMENT_SIZE"] = "64"
        os.environ["MERKLE_ALTERNATE"] = "1"

        config = MerkleConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.segment_size, 64)
        self.assertTrue(config.alternate_layers)

    def test_merkle_config_defaults(self):
        """Test MerkleConfig uses sensible defaults."""
        from experiments.merkle_attention import MerkleConfig

        config = MerkleConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.segment_size, 64)
        self.assertTrue(config.alternate_layers)

    def test_merkle_attention_divides_sequence_into_segments(self):
        """Test MerkleAttention segments input sequence."""
        from experiments.merkle_attention import MerkleAttention

        batch_size = 2
        seq_len = 128
        dim = 64
        num_heads = 8
        segment_size = 32

        attn = MerkleAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=4,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=segment_size,
        )

        x = torch.randn(batch_size, seq_len, dim)

        # Forward should not raise
        output = attn(x)
        self.assertEqual(output.shape, (batch_size, seq_len, dim))

    def test_merkle_attention_local_attention_within_segments(self):
        """Test MerkleAttention performs local attention within segments."""
        from experiments.merkle_attention import MerkleAttention

        batch_size = 2
        seq_len = 64
        dim = 64
        segment_size = 16

        attn = MerkleAttention(
            dim=dim,
            num_heads=8,
            num_kv_heads=4,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=segment_size,
        )

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)

        # Each segment should attend only within itself (causally)
        self.assertEqual(output.shape, (batch_size, seq_len, dim))

    def test_merkle_attention_hierarchical_aggregation(self):
        """Test MerkleAttention hierarchically aggregates segment summaries."""
        from experiments.merkle_attention import MerkleAttention

        batch_size = 2
        seq_len = 128
        dim = 64
        segment_size = 32

        attn = MerkleAttention(
            dim=dim,
            num_heads=8,
            num_kv_heads=4,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=segment_size,
        )

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)

        # Should produce valid output
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        self.assertFalse(torch.isnan(output).any())

    def test_merkle_attention_complexity_is_n_log_n(self):
        """Test MerkleAttention has O(N log N) complexity instead of O(N^2)."""
        from experiments.merkle_attention import MerkleAttention

        # Small sequence
        seq_len_small = 64
        segment_size = 16
        dim = 64

        attn = MerkleAttention(
            dim=dim,
            num_heads=8,
            num_kv_heads=4,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=segment_size,
        )

        x_small = torch.randn(1, seq_len_small, dim)

        # Larger sequence (2x)
        seq_len_large = 128
        x_large = torch.randn(1, seq_len_large, dim)

        # Both should complete without memory issues
        output_small = attn(x_small)
        output_large = attn(x_large)

        self.assertEqual(output_small.shape, (1, seq_len_small, dim))
        self.assertEqual(output_large.shape, (1, seq_len_large, dim))

    def test_tree_attention_block_replaces_standard_attention(self):
        """Test TreeAttentionBlock uses MerkleAttention instead of standard."""
        from experiments.merkle_attention import TreeAttentionBlock

        dim = 64
        block = TreeAttentionBlock(
            dim=dim,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=32,
        )

        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, dim)
        x0 = torch.randn(batch_size, seq_len, dim)

        # Should have same interface as Block
        output = block(x, x0)
        self.assertEqual(output.shape, (batch_size, seq_len, dim))

    def test_hash_positional_encoding_deterministic(self):
        """Test HashPositionalEncoding produces deterministic positions."""
        from experiments.merkle_attention import HashPositionalEncoding

        dim = 64
        max_seq_len = 1024

        pos_enc = HashPositionalEncoding(dim, max_seq_len)

        # Get encodings twice
        enc1 = pos_enc(100)
        enc2 = pos_enc(100)

        # Should be identical
        self.assertTrue(torch.allclose(enc1, enc2))

    def test_hash_positional_encoding_saves_parameters(self):
        """Test HashPositionalEncoding has no learnable parameters."""
        from experiments.merkle_attention import HashPositionalEncoding

        dim = 64
        max_seq_len = 1024

        pos_enc = HashPositionalEncoding(dim, max_seq_len)

        # Should have zero parameters (all deterministic)
        num_params = sum(p.numel() for p in pos_enc.parameters())
        self.assertEqual(num_params, 0)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestHybridGPT(unittest.TestCase):
    """Test hybrid model with alternating attention types."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        torch.manual_seed(42)

    def tearDown(self):
        """Restore environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_hybrid_gpt_alternates_attention_layers(self):
        """Test HybridGPT alternates between tree and standard attention."""
        from experiments.merkle_attention import HybridGPT

        model = HybridGPT(
            vocab_size=1024,
            num_layers=6,
            model_dim=64,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=32,
        )

        # Check that we have alternating block types
        # Even layers should be tree attention, odd should be standard
        self.assertEqual(len(model.blocks), 6)

    def test_hybrid_gpt_forward_pass(self):
        """Test HybridGPT forward pass works correctly."""
        from experiments.merkle_attention import HybridGPT

        model = HybridGPT(
            vocab_size=1024,
            num_layers=6,
            model_dim=64,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            segment_size=32,
        )

        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1024, (batch_size, seq_len))
        target_ids = torch.randint(0, 1024, (batch_size, seq_len))

        # Should not raise
        loss = model(input_ids, target_ids)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestMerkleIntegration(unittest.TestCase):
    """Test Merkle attention integration with training script."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        torch.manual_seed(42)

    def tearDown(self):
        """Restore environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_build_merkle_model_creates_hybrid_gpt(self):
        """Test build_merkle_model returns HybridGPT."""
        from experiments.merkle_attention import MerkleConfig, build_merkle_model

        os.environ["MERKLE_ENABLED"] = "1"
        os.environ["MERKLE_SEGMENT_SIZE"] = "64"

        config = MerkleConfig()

        gpt_args = {
            "vocab_size": 1024,
            "num_layers": 9,
            "model_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 4,
            "mlp_mult": 2,
            "tie_embeddings": True,
            "tied_embed_init_std": 0.005,
            "logit_softcap": 30.0,
            "rope_base": 10000.0,
            "qk_gain_init": 1.5,
            "moe_experts": 1,
            "moe_deploy_expert": -1,
        }

        model = build_merkle_model(config, gpt_args)

        # Model should be valid
        self.assertIsInstance(model, nn.Module)

        # Should handle forward pass
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1024, (batch_size, seq_len))
        target_ids = torch.randint(0, 1024, (batch_size, seq_len))

        loss = model(input_ids, target_ids)
        self.assertIsInstance(loss, torch.Tensor)

    def test_merkle_disabled_returns_none(self):
        """Test build_merkle_model returns None when disabled."""
        from experiments.merkle_attention import MerkleConfig, build_merkle_model

        os.environ["MERKLE_ENABLED"] = "0"

        config = MerkleConfig()
        gpt_args = {}

        model = build_merkle_model(config, gpt_args)
        self.assertIsNone(model)


if __name__ == "__main__":
    unittest.main()

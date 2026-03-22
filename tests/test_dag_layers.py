"""Tests for DAG Multi-Path Transformer Layers (RJC-32)."""

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
class TestDAGLayers(unittest.TestCase):
    """Test DAG multi-path transformer layer architecture."""

    def setUp(self):
        """Set up test environment variables and random seed."""
        self.original_env = os.environ.copy()
        torch.manual_seed(42)

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_dag_config_from_env_vars(self):
        """Test DAGConfig reads environment variables correctly."""
        from experiments.dag_layers import DAGConfig

        os.environ["DAG_ENABLED"] = "1"
        os.environ["DAG_BRANCH_START"] = "3"
        os.environ["DAG_BRANCH_END"] = "5"
        os.environ["DAG_NUM_BRANCHES"] = "2"

        config = DAGConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.branch_start, 3)
        self.assertEqual(config.branch_end, 5)
        self.assertEqual(config.num_branches, 2)

    def test_dag_config_defaults(self):
        """Test DAGConfig uses sensible defaults."""
        from experiments.dag_layers import DAGConfig

        config = DAGConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.branch_start, 3)
        self.assertEqual(config.branch_end, 5)
        self.assertEqual(config.num_branches, 2)

    def test_parallel_branch_creates_independent_parameters(self):
        """Test ParallelBranch wraps a Block with independent parameters."""
        from experiments.dag_layers import ParallelBranch

        # Create dummy block class for testing
        class DummyBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(dim, dim))

            def forward(self, x, x0):
                return x @ self.weight

        branch1 = ParallelBranch(lambda: DummyBlock(64))
        branch2 = ParallelBranch(lambda: DummyBlock(64))

        # Verify branches have independent parameters
        self.assertIsNot(
            branch1.block.weight,
            branch2.block.weight,
            "Branches should have independent parameters",
        )

    def test_branch_merger_creates_learned_mixing_weights(self):
        """Test BranchMerger has trainable mixing weights."""
        from experiments.dag_layers import BranchMerger

        num_branches = 3
        dim = 64
        merger = BranchMerger(num_branches, dim)

        # Check merge_weights shape
        self.assertEqual(merger.merge_weights.shape, (num_branches,))

        # Verify weights are trainable parameters
        self.assertTrue(merger.merge_weights.requires_grad)

    def test_branch_merger_output_shape(self):
        """Test BranchMerger produces correct output shape."""
        from experiments.dag_layers import BranchMerger

        batch_size = 4
        seq_len = 16
        dim = 64
        num_branches = 2

        merger = BranchMerger(num_branches, dim)
        branch_outputs = [torch.randn(batch_size, seq_len, dim) for _ in range(num_branches)]

        output = merger(branch_outputs)

        self.assertEqual(output.shape, (batch_size, seq_len, dim))

    def test_dag_block_sequential_layers_before_branch(self):
        """Test DAGBlock processes layers 0 to branch_start-1 sequentially."""
        from experiments.dag_layers import DAGBlock

        num_layers = 9
        dim = 64
        config_dict = {
            "enabled": True,
            "branch_start": 3,
            "branch_end": 5,
            "num_branches": 2,
        }

        # Mock block factory
        def make_block():
            class SimpleBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(dim, dim)

                def forward(self, x, x0):
                    return self.linear(x)

            return SimpleBlock()

        dag = DAGBlock(num_layers, make_block, config_dict, dim)

        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, dim)

        # Forward should not raise
        output = dag(x, x)
        self.assertEqual(output.shape, (batch_size, seq_len, dim))

    def test_dag_block_parallel_branches(self):
        """Test DAGBlock creates parallel branches from branch_start to branch_end."""
        from experiments.dag_layers import DAGBlock

        num_layers = 9
        dim = 64
        config_dict = {
            "enabled": True,
            "branch_start": 3,
            "branch_end": 5,
            "num_branches": 2,
        }

        def make_block():
            class SimpleBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(dim, dim)

                def forward(self, x, x0):
                    return self.linear(x)

            return SimpleBlock()

        dag = DAGBlock(num_layers, make_block, config_dict, dim)

        # Should have 2 branches
        self.assertEqual(len(dag.branches), 2)

        # Each branch should have (branch_end - branch_start + 1) = 3 layers
        expected_branch_layers = 5 - 3 + 1
        for branch in dag.branches:
            self.assertEqual(len(branch.layers), expected_branch_layers)

    def test_dag_block_merge_after_branches(self):
        """Test DAGBlock merges branches with learned weights."""
        from experiments.dag_layers import DAGBlock

        num_layers = 9
        dim = 64
        config_dict = {
            "enabled": True,
            "branch_start": 3,
            "branch_end": 5,
            "num_branches": 2,
        }

        def make_block():
            class SimpleBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(dim, dim)

                def forward(self, x, x0):
                    return self.linear(x)

            return SimpleBlock()

        dag = DAGBlock(num_layers, make_block, config_dict, dim)

        # Should have a merger
        self.assertIsNotNone(dag.merger)

    def test_dag_block_dense_skip_connections(self):
        """Test DAGBlock implements DenseNet-style skip connections."""
        from experiments.dag_layers import DAGBlock

        num_layers = 6
        dim = 64
        config_dict = {
            "enabled": True,
            "branch_start": 2,
            "branch_end": 3,
            "num_branches": 2,
        }

        def make_block():
            class SimpleBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(dim, dim)

                def forward(self, x, x0):
                    return self.linear(x)

            return SimpleBlock()

        dag = DAGBlock(num_layers, make_block, config_dict, dim)

        # Bottleneck layers should exist for skip connections
        self.assertIsNotNone(dag.skip_bottlenecks)

    def test_dag_block_disabled_fallback(self):
        """Test DAGBlock falls back to sequential processing when disabled."""
        from experiments.dag_layers import DAGBlock

        num_layers = 9
        dim = 64
        config_dict = {
            "enabled": False,
            "branch_start": 3,
            "branch_end": 5,
            "num_branches": 2,
        }

        def make_block():
            class SimpleBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(dim, dim)

                def forward(self, x, x0):
                    return self.linear(x)

            return SimpleBlock()

        dag = DAGBlock(num_layers, make_block, config_dict, dim)

        # Should not have branches when disabled
        self.assertIsNone(dag.branches)
        self.assertIsNone(dag.merger)

        # Should still process correctly
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, dim)
        output = dag(x, x)
        self.assertEqual(output.shape, (batch_size, seq_len, dim))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDAGIntegration(unittest.TestCase):
    """Test DAG integration with training script."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        torch.manual_seed(42)

    def tearDown(self):
        """Restore environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_build_dag_model_creates_modified_gpt(self):
        """Test build_dag_model returns a GPT with DAG topology."""
        from experiments.dag_layers import DAGConfig, build_dag_model

        os.environ["DAG_ENABLED"] = "1"
        os.environ["DAG_BRANCH_START"] = "3"
        os.environ["DAG_BRANCH_END"] = "5"
        os.environ["DAG_NUM_BRANCHES"] = "2"

        config = DAGConfig()

        # Mock GPT args
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

        model = build_dag_model(config, gpt_args)

        # Model should be a valid nn.Module
        self.assertIsInstance(model, nn.Module)

        # Should be able to forward
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1024, (batch_size, seq_len))
        target_ids = torch.randint(0, 1024, (batch_size, seq_len))

        # Should not raise
        loss = model(input_ids, target_ids)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()

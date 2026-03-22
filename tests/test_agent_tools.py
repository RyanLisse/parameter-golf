"""Tests for agent-native experiment tools (RJC-39 + RJC-41)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import agent_tools


class AgentToolsTests(unittest.TestCase):
    """Test atomic tools for agent-driven experiment orchestration."""

    def test_mutate_hyperparams_returns_dict(self) -> None:
        """Verify that mutate_hyperparams returns a JSON-serializable dict."""
        config = {"NUM_LAYERS": "9", "MODEL_DIM": "512"}
        mutated = agent_tools.mutate_hyperparams(config, mutation_rate=0.5)

        self.assertIsInstance(mutated, dict)
        json.dumps(mutated)  # Should not raise

    def test_mutate_hyperparams_preserves_keys(self) -> None:
        """Verify that mutation includes original keys."""
        config = {"NUM_LAYERS": "9", "MODEL_DIM": "512", "MLP_MULT": "2"}
        mutated = agent_tools.mutate_hyperparams(config, mutation_rate=1.0)

        for key in config:
            self.assertIn(key, mutated)

    def test_mutate_hyperparams_changes_some_values(self) -> None:
        """Verify that mutation with high rate actually changes values."""
        config = {"NUM_LAYERS": "9", "MODEL_DIM": "512"}
        mutated = agent_tools.mutate_hyperparams(config, mutation_rate=1.0)

        # At least one value should be different with mutation_rate=1.0
        changed = sum(1 for k in config if config[k] != mutated.get(k, ""))
        self.assertGreater(changed, 0, "High mutation rate should change some values")

    def test_mutate_hyperparams_respects_mutation_rate(self) -> None:
        """Verify that low mutation rate causes fewer changes."""
        config = {"NUM_LAYERS": "9", "MODEL_DIM": "512"}
        mutated = agent_tools.mutate_hyperparams(config, mutation_rate=0.0)

        # mutation_rate=0.0 should mean no changes
        self.assertEqual(config, mutated)

    def test_compare_runs_returns_dict(self) -> None:
        """Verify that compare_runs returns a JSON-serializable dict."""
        result = agent_tools.compare_runs(run_ids=["run1", "run2"])

        self.assertIsInstance(result, dict)
        json.dumps(result)  # Should not raise

    def test_compare_runs_includes_status(self) -> None:
        """Verify that compare_runs result includes status field."""
        result = agent_tools.compare_runs(run_ids=["run1"])

        self.assertIn("status", result)
        self.assertIsInstance(result["status"], str)

    def test_get_experiment_status_returns_dict(self) -> None:
        """Verify that get_experiment_status returns a JSON-serializable dict."""
        result = agent_tools.get_experiment_status(run_id="test_run")

        self.assertIsInstance(result, dict)
        json.dumps(result)  # Should not raise

    def test_get_experiment_status_includes_stage(self) -> None:
        """Verify that status result includes 'stage' field."""
        result = agent_tools.get_experiment_status(run_id="test_run")

        self.assertIn("stage", result)
        stage = result.get("stage", "")
        valid_stages = {"configuring", "training", "validating", "compressing", "scoring", "unknown"}
        self.assertIn(stage, valid_stages)

    def test_read_val_bpb_returns_dict(self) -> None:
        """Verify that read_val_bpb returns a JSON-serializable dict."""
        result = agent_tools.read_val_bpb(run_id="test_run")

        self.assertIsInstance(result, dict)
        json.dumps(result)  # Should not raise

    def test_read_val_bpb_includes_status(self) -> None:
        """Verify that read_val_bpb result includes 'status' field."""
        result = agent_tools.read_val_bpb(run_id="test_run")

        self.assertIn("status", result)
        self.assertIsInstance(result["status"], str)

    def test_check_artifact_size_returns_dict(self) -> None:
        """Verify that check_artifact_size returns a JSON-serializable dict."""
        result = agent_tools.check_artifact_size(run_id="test_run")

        self.assertIsInstance(result, dict)
        json.dumps(result)  # Should not raise

    def test_check_artifact_size_includes_fits(self) -> None:
        """Verify that artifact size check includes 'fits_16mb' field."""
        result = agent_tools.check_artifact_size(run_id="test_run")

        # Should have some indication of size constraint
        keys = set(result.keys())
        self.assertTrue(
            {"fits_16mb", "fits", "valid", "within_limit"} & keys,
            "Should have size constraint indicator"
        )

    def test_submit_training_run_returns_dict(self) -> None:
        """Verify that submit_training_run returns a JSON-serializable dict."""
        config = {"NUM_LAYERS": "9", "MODEL_DIM": "512"}
        result = agent_tools.submit_training_run(config, backend="mlx")

        self.assertIsInstance(result, dict)
        json.dumps(result)  # Should not raise

    def test_submit_training_run_includes_run_id(self) -> None:
        """Verify that submit_training_run result includes 'run_id' field."""
        config = {"NUM_LAYERS": "9"}
        result = agent_tools.submit_training_run(config, backend="mlx")

        self.assertIn("run_id", result)
        self.assertIsInstance(result["run_id"], str)

    def test_submit_training_run_includes_status(self) -> None:
        """Verify that submit_training_run result includes 'status' field."""
        config = {"NUM_LAYERS": "9"}
        result = agent_tools.submit_training_run(config, backend="mlx")

        self.assertIn("status", result)
        self.assertIsInstance(result["status"], str)

    def test_submit_training_run_backend_validation(self) -> None:
        """Verify that submit_training_run accepts both backends."""
        config = {"NUM_LAYERS": "9"}

        result_mlx = agent_tools.submit_training_run(config, backend="mlx")
        result_cuda = agent_tools.submit_training_run(config, backend="cuda")

        self.assertIsInstance(result_mlx["run_id"], str)
        self.assertIsInstance(result_cuda["run_id"], str)


if __name__ == "__main__":
    unittest.main()

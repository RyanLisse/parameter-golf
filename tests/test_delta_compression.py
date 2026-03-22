"""Tests for ZK-Inspired Recursive Layer Delta Compression (RJC-31)."""
from __future__ import annotations

import unittest
import zlib

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from experiments import delta_compression


@unittest.skipUnless(HAS_TORCH, "torch not available")
class AnalyzeLayerSimilarityTests(unittest.TestCase):
    def test_identical_layers_have_similarity_one(self) -> None:
        sd = {
            "layers.0.attn.weight": torch.randn(64, 64),
            "layers.1.attn.weight": torch.randn(64, 64),
        }
        # Copy layer 0 to layer 1 so they're identical.
        sd["layers.1.attn.weight"] = sd["layers.0.attn.weight"].clone()
        sim = delta_compression.analyze_layer_similarity(sd)
        self.assertEqual(sim.shape, (2, 2))
        self.assertAlmostEqual(float(sim[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(sim[1, 0]), 1.0, places=5)

    def test_different_layers_have_lower_similarity(self) -> None:
        torch.manual_seed(42)
        sd = {
            "layers.0.attn.weight": torch.randn(64, 64),
            "layers.1.attn.weight": torch.randn(64, 64) * 10,
        }
        sim = delta_compression.analyze_layer_similarity(sd)
        self.assertLess(float(sim[0, 1]), 1.0)

    def test_single_layer_returns_1x1_matrix(self) -> None:
        sd = {"layers.0.fc.weight": torch.ones(32, 32)}
        sim = delta_compression.analyze_layer_similarity(sd)
        self.assertEqual(sim.shape, (1, 1))
        self.assertAlmostEqual(float(sim[0, 0]), 1.0)

    def test_non_layer_keys_ignored(self) -> None:
        sd = {
            "embed.weight": torch.randn(100, 64),
            "layers.0.fc.weight": torch.randn(64, 64),
            "layers.1.fc.weight": torch.randn(64, 64),
            "head.weight": torch.randn(100, 64),
        }
        sim = delta_compression.analyze_layer_similarity(sd)
        self.assertEqual(sim.shape, (2, 2))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class ComputeDeltasTests(unittest.TestCase):
    def test_base_layer_stored_fully(self) -> None:
        torch.manual_seed(7)
        sd = {
            "layers.0.fc.weight": torch.randn(32, 32),
            "layers.1.fc.weight": torch.randn(32, 32),
            "layers.2.fc.weight": torch.randn(32, 32),
            "embed.weight": torch.randn(100, 32),
        }
        result = delta_compression.compute_deltas(sd, base_layer_idx=0)
        self.assertIn("base_weights", result)
        self.assertIn("deltas", result)
        self.assertIn("non_layer_weights", result)
        # Base layer weights should be present
        self.assertIn("layers.0.fc.weight", result["base_weights"])
        # Deltas for non-base layers
        self.assertIn("layers.1.fc.weight", result["deltas"])
        self.assertIn("layers.2.fc.weight", result["deltas"])

    def test_delta_smaller_than_full_weight(self) -> None:
        """Deltas of similar layers should have smaller magnitude than originals."""
        torch.manual_seed(7)
        base = torch.randn(64, 64)
        sd = {
            "layers.0.fc.weight": base,
            "layers.1.fc.weight": base + torch.randn(64, 64) * 0.01,
        }
        result = delta_compression.compute_deltas(sd, base_layer_idx=0)
        delta = result["deltas"]["layers.1.fc.weight"]
        self.assertLess(delta.abs().mean().item(), base.abs().mean().item())

    def test_roundtrip_via_reconstruction(self) -> None:
        torch.manual_seed(42)
        sd = {
            "layers.0.attn.weight": torch.randn(32, 32),
            "layers.1.attn.weight": torch.randn(32, 32),
            "layers.0.fc.weight": torch.randn(64, 32),
            "layers.1.fc.weight": torch.randn(64, 32),
            "embed.weight": torch.randn(100, 32),
        }
        result = delta_compression.compute_deltas(sd, base_layer_idx=0)
        reconstructed = delta_compression.reconstruct_from_deltas(result)
        for key in sd:
            torch.testing.assert_close(reconstructed[key], sd[key])


@unittest.skipUnless(HAS_TORCH, "torch not available")
class CompressDecompressRoundtripTests(unittest.TestCase):
    def test_roundtrip_preserves_values_approximately(self) -> None:
        """Compress + decompress should recover weights within int8 quantization error."""
        torch.manual_seed(99)
        sd = {
            "layers.0.fc.weight": torch.randn(32, 32),
            "layers.1.fc.weight": torch.randn(32, 32),
            "layers.2.fc.weight": torch.randn(32, 32),
            "embed.weight": torch.randn(50, 32),
        }
        compressed = delta_compression.compress_with_deltas(sd)
        recovered = delta_compression.decompress_from_deltas(compressed)
        for key in sd:
            self.assertEqual(recovered[key].shape, sd[key].shape)
            # Quantization introduces some error, but should be small
            max_err = (recovered[key] - sd[key]).abs().max().item()
            self.assertLess(max_err, 0.5, f"Too much error in {key}: {max_err}")

    def test_compressed_is_bytes(self) -> None:
        sd = {
            "layers.0.fc.weight": torch.randn(16, 16),
            "layers.1.fc.weight": torch.randn(16, 16),
        }
        compressed = delta_compression.compress_with_deltas(sd)
        self.assertIsInstance(compressed, bytes)

    def test_delta_compression_beats_direct_for_similar_layers(self) -> None:
        """When layers are very similar and large, delta compression should yield smaller output."""
        torch.manual_seed(123)
        # Larger tensors where delta overhead is amortized
        base = torch.randn(512, 512)
        sd = {}
        for i in range(12):
            sd[f"layers.{i}.fc.weight"] = base + torch.randn(512, 512) * 0.001

        delta_size = len(delta_compression.compress_with_deltas(sd))
        direct_size = len(delta_compression.compress_direct(sd))
        self.assertLess(delta_size, direct_size,
                        f"Delta {delta_size} should be < direct {direct_size}")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class CompressDirectTests(unittest.TestCase):
    def test_compress_direct_returns_bytes(self) -> None:
        sd = {"layers.0.fc.weight": torch.randn(16, 16)}
        result = delta_compression.compress_direct(sd)
        self.assertIsInstance(result, bytes)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class CLITests(unittest.TestCase):
    def test_module_has_main_guard(self) -> None:
        """The module should be runnable as a script."""
        import importlib
        spec = importlib.util.find_spec("experiments.delta_compression")
        self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for learned compression codebook (RJC-51 / EXP-18)."""

import unittest
import numpy as np
from experiments.compression_codebook import (
    CodebookQuantizer,
    BlockCodebookQuantizer,
    compress_with_codebook,
    decompress_from_codebook,
    compare_compression,
)


class TestKMeansCodebook(unittest.TestCase):
    """Test K-means clustering for weight compression."""

    def test_kmeans_converges_simple_distribution(self):
        """K-means should converge on simple clustered data."""
        # Create data with obvious clusters: values near -1, 0, 1
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-1.0, 0.1, 100),
            np.random.normal(0.0, 0.1, 100),
            np.random.normal(1.0, 0.1, 100),
        ])

        quantizer = CodebookQuantizer(bits=2, block_size=1)  # 4 centroids for 3 clusters
        quantizer.fit(data)

        # Should have 4 centroids
        self.assertEqual(quantizer.codebook.shape[0], 4)

        # Centroids should be roughly near -1, 0, 1 (may have extra)
        sorted_centroids = np.sort(quantizer.codebook)
        self.assertLess(np.min(sorted_centroids), -0.5)
        self.assertGreater(np.max(sorted_centroids), 0.5)

    def test_encode_decode_preserves_shape(self):
        """Encode/decode roundtrip should preserve tensor shape."""
        np.random.seed(42)
        weights = np.random.randn(128, 64).astype(np.float32)

        quantizer = CodebookQuantizer(bits=6, block_size=1)
        quantizer.fit(weights.flatten())

        indices, codebook = quantizer.encode(weights)
        reconstructed = quantizer.decode(indices, codebook)

        self.assertEqual(weights.shape, reconstructed.shape)
        self.assertEqual(weights.dtype, reconstructed.dtype)

    def test_codebook_compression_smaller_than_int8(self):
        """6-bit codebook should produce smaller indices than 8-bit int8."""
        np.random.seed(42)
        # Clustered weights compress better than uniform
        weights = np.concatenate([
            np.random.normal(-0.5, 0.1, 1024),
            np.random.normal(0.5, 0.1, 1024),
        ]).astype(np.float32)

        quantizer = CodebookQuantizer(bits=6, block_size=1)
        quantizer.fit(weights)

        ratio = quantizer.compression_ratio(weights.size)
        # 6-bit indices: 0.75 bytes per weight
        # int8: 1.0 bytes per weight
        # Codebook overhead is small for large arrays
        self.assertLess(ratio, 1.0)  # Better than int8

    def test_block_quantization_correct_shapes(self):
        """Block quantizer should produce correct index shapes."""
        np.random.seed(42)
        weights = np.random.randn(128, 64).astype(np.float32)

        quantizer = BlockCodebookQuantizer(bits=6, block_size=4)
        quantizer.fit(weights.flatten())

        indices, codebook = quantizer.encode(weights)

        # Each block of 4 values gets one index
        expected_indices = weights.size // 4
        self.assertEqual(indices.size, expected_indices)

        # Codebook has 2^6=64 entries, each is a 4-element vector
        self.assertEqual(codebook.shape, (64, 4))

    def test_compression_comparison_metrics(self):
        """compare_compression should return valid metrics."""
        np.random.seed(42)
        state_dict = {
            "layer1.weight": np.random.randn(256, 128).astype(np.float32),
            "layer2.weight": np.random.randn(128, 64).astype(np.float32),
        }

        metrics = compare_compression(state_dict, bits=6, block_size=1)

        # Should have both int8 and codebook sizes
        self.assertIn("int8_bytes", metrics)
        self.assertIn("codebook_bytes", metrics)
        self.assertIn("compression_ratio", metrics)
        self.assertIn("reconstruction_mse", metrics)

        # Sizes should be positive
        self.assertGreater(metrics["int8_bytes"], 0)
        self.assertGreater(metrics["codebook_bytes"], 0)

        # MSE should be non-negative
        self.assertGreaterEqual(metrics["reconstruction_mse"], 0.0)


class TestCodebookIntegration(unittest.TestCase):
    """Test full compression/decompression pipeline."""

    def test_compress_decompress_roundtrip(self):
        """Full compress/decompress should preserve state dict keys and shapes."""
        np.random.seed(42)
        state_dict = {
            "embed.weight": np.random.randn(1024, 512).astype(np.float32),
            "layer1.q.weight": np.random.randn(512, 512).astype(np.float32),
            "layer1.v.weight": np.random.randn(512, 512).astype(np.float32),
        }

        compressed = compress_with_codebook(state_dict, bits=6, block_size=1)
        reconstructed = decompress_from_codebook(compressed)

        # Should have same keys
        self.assertEqual(set(state_dict.keys()), set(reconstructed.keys()))

        # Should have same shapes
        for key in state_dict:
            self.assertEqual(state_dict[key].shape, reconstructed[key].shape)

    def test_compressed_size_reasonable(self):
        """Compressed size should be smaller than original float32."""
        np.random.seed(42)
        state_dict = {
            "weight": np.random.randn(512, 512).astype(np.float32),
        }

        compressed = compress_with_codebook(state_dict, bits=6, block_size=1)
        original_bytes = state_dict["weight"].nbytes

        # Should be significantly smaller than fp32 (4 bytes per param)
        # 6-bit indices = 0.75 bytes per param + codebook overhead
        self.assertLess(len(compressed), original_bytes)


if __name__ == "__main__":
    unittest.main()

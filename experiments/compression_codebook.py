#!/usr/bin/env python3
"""Learned Compression Codebook (RJC-51 / EXP-18).

K-means weight clustering for better-than-int8 compression.
Env vars: CODEBOOK_ENABLED=1, CODEBOOK_BITS=6, CODEBOOK_BLOCK_SIZE=4

Key idea: Instead of uniform int8 quantization, learn a codebook of common
weight values during training. Map weights to their nearest centroid index.
This can reduce quantization error and compress better than fixed-precision int8.

For a 6-bit codebook (64 centroids):
- Each weight index: 6 bits = 0.75 bytes
- vs int8: 8 bits = 1.0 bytes
- 25% size reduction on indices alone

Codebook overhead is small: 64 centroids × 4 bytes = 256 bytes per layer.
For a 9-layer model with ~30 weight tensors: ~7.5KB total codebook overhead.
"""

from __future__ import annotations

import io
import pickle
import zlib
from typing import Any

import numpy as np


class CodebookQuantizer:
    """Scalar quantizer using K-means clustering.

    Args:
        bits: Number of bits for indices (2^bits centroids)
        block_size: Always 1 for scalar quantization
    """

    def __init__(self, bits: int = 6, block_size: int = 1):
        if block_size != 1:
            raise ValueError("CodebookQuantizer only supports block_size=1 (scalar)")
        self.bits = bits
        self.num_centroids = 2 ** bits
        self.block_size = 1
        self.codebook: np.ndarray | None = None

    def fit(self, weights: np.ndarray, max_iters: int = 20, tol: float = 1e-4):
        """Run K-means to find optimal centroids.

        Args:
            weights: Flattened weight array
            max_iters: Maximum K-means iterations
            tol: Convergence tolerance for centroid movement
        """
        flat = weights.flatten().astype(np.float32)

        # Initialize centroids with random samples
        indices = np.random.choice(flat.size, size=self.num_centroids, replace=False)
        centroids = flat[indices].copy()

        for _ in range(max_iters):
            # Assignment step: find nearest centroid for each weight
            distances = np.abs(flat[:, np.newaxis] - centroids[np.newaxis, :])
            assignments = np.argmin(distances, axis=1)

            # Update step: recompute centroids as cluster means
            new_centroids = centroids.copy()
            for i in range(self.num_centroids):
                mask = assignments == i
                if np.sum(mask) > 0:
                    new_centroids[i] = flat[mask].mean()

            # Check convergence
            movement = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if movement < tol:
                break

        self.codebook = centroids

    def encode(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode weights to codebook indices.

        Args:
            weights: Weight tensor (any shape)

        Returns:
            (indices, codebook) where indices has same shape as weights
        """
        if self.codebook is None:
            raise RuntimeError("Must call fit() before encode()")

        original_shape = weights.shape
        flat = weights.flatten().astype(np.float32)

        # Find nearest centroid for each weight
        distances = np.abs(flat[:, np.newaxis] - self.codebook[np.newaxis, :])
        indices = np.argmin(distances, axis=1)

        return indices.reshape(original_shape), self.codebook

    def decode(self, indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Decode indices back to approximate weights.

        Args:
            indices: Codebook indices
            codebook: Codebook array

        Returns:
            Reconstructed weights (same shape as indices)
        """
        return codebook[indices.flatten()].reshape(indices.shape).astype(np.float32)

    def compression_ratio(self, num_weights: int) -> float:
        """Compute compression ratio vs int8.

        Args:
            num_weights: Total number of weights

        Returns:
            Ratio of codebook size to int8 size (< 1.0 is better)
        """
        if self.codebook is None:
            raise RuntimeError("Must call fit() before compression_ratio()")

        # Codebook storage: num_centroids × 4 bytes (float32)
        codebook_bytes = self.num_centroids * 4

        # Index storage: num_weights × (bits / 8) bytes
        index_bytes = num_weights * (self.bits / 8.0)

        total_codebook = codebook_bytes + index_bytes

        # Int8 baseline: num_weights × 1 byte
        int8_bytes = num_weights * 1

        return total_codebook / int8_bytes


class BlockCodebookQuantizer:
    """Vector quantizer using block-wise K-means.

    Groups weights into blocks of size `block_size` and maps each block
    to a codebook entry. Higher quality than scalar but larger codebook.

    Args:
        bits: Number of bits for indices (2^bits centroids)
        block_size: Number of weights per block (e.g., 4)
    """

    def __init__(self, bits: int = 6, block_size: int = 4):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.bits = bits
        self.num_centroids = 2 ** bits
        self.block_size = block_size
        self.codebook: np.ndarray | None = None

    def fit(self, weights: np.ndarray, max_iters: int = 20, tol: float = 1e-4):
        """Run K-means on weight blocks.

        Args:
            weights: Flattened weight array
            max_iters: Maximum K-means iterations
            tol: Convergence tolerance
        """
        flat = weights.flatten().astype(np.float32)

        # Pad to multiple of block_size
        remainder = flat.size % self.block_size
        if remainder != 0:
            pad_size = self.block_size - remainder
            flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0.0)

        # Reshape into blocks
        blocks = flat.reshape(-1, self.block_size)

        # Initialize centroids with random blocks
        num_blocks = blocks.shape[0]
        indices = np.random.choice(num_blocks, size=min(self.num_centroids, num_blocks), replace=False)
        centroids = blocks[indices].copy()

        # If we have fewer blocks than centroids, duplicate some
        while centroids.shape[0] < self.num_centroids:
            centroids = np.vstack([centroids, centroids[: self.num_centroids - centroids.shape[0]]])

        for _ in range(max_iters):
            # Assignment: find nearest centroid for each block (L2 distance)
            distances = np.sum((blocks[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update: recompute centroids as block means
            new_centroids = centroids.copy()
            for i in range(self.num_centroids):
                mask = assignments == i
                if np.sum(mask) > 0:
                    new_centroids[i] = blocks[mask].mean(axis=0)

            # Check convergence
            movement = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if movement < tol:
                break

        self.codebook = centroids

    def encode(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode weights to block indices.

        Args:
            weights: Weight tensor (any shape)

        Returns:
            (indices, codebook) where indices is 1D with one index per block
        """
        if self.codebook is None:
            raise RuntimeError("Must call fit() before encode()")

        original_shape = weights.shape
        flat = weights.flatten().astype(np.float32)

        # Pad to multiple of block_size
        remainder = flat.size % self.block_size
        if remainder != 0:
            pad_size = self.block_size - remainder
            flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0.0)

        blocks = flat.reshape(-1, self.block_size)

        # Find nearest centroid for each block
        distances = np.sum((blocks[:, np.newaxis, :] - self.codebook[np.newaxis, :, :]) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)

        return indices, self.codebook

    def decode(self, indices: np.ndarray, codebook: np.ndarray, original_size: int) -> np.ndarray:
        """Decode block indices back to weights.

        Args:
            indices: Block indices (1D)
            codebook: Codebook array (num_centroids, block_size)
            original_size: Original weight count (before padding)

        Returns:
            Reconstructed weights (1D, size=original_size)
        """
        blocks = codebook[indices]
        flat = blocks.flatten()
        return flat[:original_size].astype(np.float32)


def compress_with_codebook(
    state_dict: dict[str, np.ndarray], bits: int = 6, block_size: int = 1
) -> bytes:
    """Compress state dict with learned codebook.

    Args:
        state_dict: Dictionary of weight tensors
        bits: Codebook size (2^bits centroids)
        block_size: Weights per block (1=scalar, 4=vector)

    Returns:
        Compressed bytes (pickle + zlib)
    """
    compressed_state = {}

    for name, weights in state_dict.items():
        if block_size == 1:
            quantizer = CodebookQuantizer(bits=bits, block_size=1)
        else:
            quantizer = BlockCodebookQuantizer(bits=bits, block_size=block_size)

        quantizer.fit(weights.flatten())
        indices, codebook = quantizer.encode(weights)

        compressed_state[name] = {
            "indices": indices,
            "codebook": codebook,
            "shape": weights.shape,
            "block_size": block_size,
        }

    # Serialize and compress
    buffer = io.BytesIO()
    pickle.dump(compressed_state, buffer)
    return zlib.compress(buffer.getvalue(), level=9)


def decompress_from_codebook(compressed_bytes: bytes) -> dict[str, np.ndarray]:
    """Decompress codebook-compressed state dict.

    Args:
        compressed_bytes: Compressed bytes from compress_with_codebook

    Returns:
        Reconstructed state dict
    """
    decompressed = zlib.decompress(compressed_bytes)
    compressed_state = pickle.loads(decompressed)

    state_dict = {}
    for name, meta in compressed_state.items():
        indices = meta["indices"]
        codebook = meta["codebook"]
        shape = meta["shape"]
        block_size = meta["block_size"]

        if block_size == 1:
            quantizer = CodebookQuantizer(bits=0, block_size=1)  # bits unused for decode
            weights = quantizer.decode(indices, codebook)
        else:
            quantizer = BlockCodebookQuantizer(bits=0, block_size=block_size)
            original_size = np.prod(shape)
            weights = quantizer.decode(indices, codebook, original_size)
            weights = weights.reshape(shape)

        state_dict[name] = weights

    return state_dict


def compare_compression(
    state_dict: dict[str, np.ndarray], bits: int = 6, block_size: int = 1
) -> dict[str, Any]:
    """Compare codebook compression vs int8+zlib baseline.

    Args:
        state_dict: Dictionary of weight tensors
        bits: Codebook size
        block_size: Weights per block

    Returns:
        Metrics dict with int8_bytes, codebook_bytes, compression_ratio, reconstruction_mse
    """
    # Baseline: int8 + zlib
    int8_state = {}
    for name, weights in state_dict.items():
        # Per-row int8 quantization (simplified version)
        w_min = weights.min()
        w_max = weights.max()
        scale = (w_max - w_min) / 255.0
        if scale == 0:
            scale = 1.0
        quantized = np.round((weights - w_min) / scale).astype(np.uint8)
        int8_state[name] = {"quantized": quantized, "scale": scale, "min": w_min}

    int8_bytes = len(zlib.compress(pickle.dumps(int8_state), level=9))

    # Codebook compression
    codebook_bytes_compressed = len(compress_with_codebook(state_dict, bits=bits, block_size=block_size))

    # Reconstruction error
    reconstructed = decompress_from_codebook(compress_with_codebook(state_dict, bits=bits, block_size=block_size))
    mse = 0.0
    total_weights = 0
    for name, weights in state_dict.items():
        diff = weights - reconstructed[name]
        mse += np.sum(diff**2)
        total_weights += weights.size
    mse /= total_weights

    return {
        "int8_bytes": int8_bytes,
        "codebook_bytes": codebook_bytes_compressed,
        "compression_ratio": codebook_bytes_compressed / int8_bytes if int8_bytes > 0 else 1.0,
        "reconstruction_mse": float(mse),
    }

"""ZK-Inspired Recursive Layer Delta Compression (RJC-31).

Stores model as base layer + small deltas for dramatic compression improvement.
Usage: python -m experiments.delta_compression <checkpoint_path>
"""
from __future__ import annotations

import io
import re
import sys
import zlib
from typing import Any

import torch
from torch import Tensor

# Match keys like "layers.0.attn.weight", "layers.12.fc.bias"
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def _group_by_layer(state_dict: dict[str, Tensor]) -> tuple[dict[int, dict[str, Tensor]], dict[str, Tensor]]:
    """Split state dict into per-layer groups and non-layer tensors."""
    layers: dict[int, dict[str, Tensor]] = {}
    non_layer: dict[str, Tensor] = {}
    for key, val in state_dict.items():
        m = _LAYER_RE.match(key)
        if m:
            idx = int(m.group(1))
            layers.setdefault(idx, {})[key] = val
        else:
            non_layer[key] = val
    return layers, non_layer


def _layer_flat_vector(layer_weights: dict[str, Tensor]) -> Tensor:
    """Concatenate all tensors in a layer into a single flat vector."""
    parts = [t.detach().float().flatten() for t in sorted(layer_weights.items()).__iter__()
             if isinstance(t, Tensor)]
    # Sort by key for determinism
    parts = [v.detach().float().flatten() for _, v in sorted(layer_weights.items())]
    return torch.cat(parts) if parts else torch.tensor([])


def analyze_layer_similarity(state_dict: dict[str, Tensor]) -> Tensor:
    """Return a cosine-similarity matrix between transformer layers.

    Only considers keys matching ``layers.N.*``. Non-layer keys are ignored.
    Returns shape ``(num_layers, num_layers)``.
    """
    layers, _ = _group_by_layer(state_dict)
    if not layers:
        return torch.zeros(0, 0)

    indices = sorted(layers.keys())
    vecs = [_layer_flat_vector(layers[i]) for i in indices]

    n = len(vecs)
    sim = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if vecs[i].numel() == vecs[j].numel() and vecs[i].numel() > 0:
                cos = torch.nn.functional.cosine_similarity(
                    vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
                )
                sim[i, j] = cos.item()
            elif i == j:
                sim[i, j] = 1.0
    return sim


def compute_deltas(
    state_dict: dict[str, Tensor],
    base_layer_idx: int = 0,
) -> dict[str, Any]:
    """Compute deltas relative to a base layer.

    Returns dict with:
        - ``base_weights``: full weights for the base layer
        - ``deltas``: ``{key: tensor - base_tensor}`` for other layers
        - ``non_layer_weights``: weights that don't belong to any layer
        - ``base_layer_idx``: which layer is the base
        - ``layer_key_map``: maps each layer's keys to the base layer's corresponding keys
    """
    layers, non_layer = _group_by_layer(state_dict)
    if base_layer_idx not in layers:
        raise ValueError(f"Base layer {base_layer_idx} not in state dict (available: {sorted(layers.keys())})")

    base_prefix = f"layers.{base_layer_idx}."
    base_suffix_map: dict[str, Tensor] = {}
    for key, val in layers[base_layer_idx].items():
        suffix = key[len(base_prefix):]
        base_suffix_map[suffix] = val

    base_weights = dict(layers[base_layer_idx])
    deltas: dict[str, Tensor] = {}

    for idx in sorted(layers.keys()):
        if idx == base_layer_idx:
            continue
        prefix = f"layers.{idx}."
        for key, val in layers[idx].items():
            suffix = key[len(prefix):]
            if suffix in base_suffix_map and val.shape == base_suffix_map[suffix].shape:
                deltas[key] = val.float() - base_suffix_map[suffix].float()
            else:
                # Shape mismatch or missing base key -- store as-is
                deltas[key] = val

    return {
        "base_weights": base_weights,
        "deltas": deltas,
        "non_layer_weights": non_layer,
        "base_layer_idx": base_layer_idx,
    }


def reconstruct_from_deltas(delta_result: dict[str, Any]) -> dict[str, Tensor]:
    """Reconstruct full state dict from base + deltas (lossless, float)."""
    base_layer_idx = delta_result["base_layer_idx"]
    base_prefix = f"layers.{base_layer_idx}."
    base_suffix_map: dict[str, Tensor] = {}
    for key, val in delta_result["base_weights"].items():
        suffix = key[len(base_prefix):]
        base_suffix_map[suffix] = val

    out: dict[str, Tensor] = {}
    # Base layer
    out.update(delta_result["base_weights"])
    # Non-layer weights
    out.update(delta_result["non_layer_weights"])
    # Reconstruct from deltas
    for key, delta in delta_result["deltas"].items():
        m = _LAYER_RE.match(key)
        if m:
            idx = int(m.group(1))
            prefix = f"layers.{idx}."
            suffix = key[len(prefix):]
            if suffix in base_suffix_map and delta.shape == base_suffix_map[suffix].shape:
                out[key] = (base_suffix_map[suffix].float() + delta.float()).to(
                    dtype=base_suffix_map[suffix].dtype
                )
            else:
                out[key] = delta
        else:
            out[key] = delta
    return out


def _quantize_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a float tensor to int8 with per-row (2D) or per-tensor scale."""
    t32 = t.detach().float()
    if t32.ndim >= 2:
        row_max = t32.abs().amax(dim=-1).clamp(min=1e-12)
        scale = row_max / 127.0
        q = torch.clamp(torch.round(t32 / scale.unsqueeze(-1)), -127, 127).to(torch.int8)
        return q, scale.to(torch.float16)
    else:
        amax = t32.abs().max().clamp(min=1e-12).item()
        scale = torch.tensor(amax / 127.0, dtype=torch.float32)
        q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8)
        return q, scale


def _dequantize_tensor_int8(q: Tensor, scale: Tensor) -> Tensor:
    """Reverse int8 quantization."""
    if scale.ndim > 0 and q.ndim >= 2:
        return (q.float() * scale.float().unsqueeze(-1))
    return q.float() * scale.float()


def compress_with_deltas(state_dict: dict[str, Tensor]) -> bytes:
    """Apply delta compression + int8 quantization + zlib.

    Returns compressed bytes blob.
    """
    delta_result = compute_deltas(state_dict, base_layer_idx=0)

    # Quantize base weights
    quantized_base: dict[str, tuple[Tensor, Tensor]] = {}
    for key, val in delta_result["base_weights"].items():
        quantized_base[key] = _quantize_tensor_int8(val)

    # Quantize deltas (they tend to be small, so int8 works well)
    quantized_deltas: dict[str, tuple[Tensor, Tensor]] = {}
    for key, val in delta_result["deltas"].items():
        quantized_deltas[key] = _quantize_tensor_int8(val)

    # Quantize non-layer weights
    quantized_non_layer: dict[str, tuple[Tensor, Tensor]] = {}
    for key, val in delta_result["non_layer_weights"].items():
        quantized_non_layer[key] = _quantize_tensor_int8(val)

    obj = {
        "format": "delta_int8_v1",
        "base_layer_idx": delta_result["base_layer_idx"],
        "base": {k: {"q": v[0], "s": v[1]} for k, v in quantized_base.items()},
        "deltas": {k: {"q": v[0], "s": v[1]} for k, v in quantized_deltas.items()},
        "non_layer": {k: {"q": v[0], "s": v[1]} for k, v in quantized_non_layer.items()},
    }
    buf = io.BytesIO()
    torch.save(obj, buf)
    return zlib.compress(buf.getvalue(), level=9)


def decompress_from_deltas(compressed: bytes) -> dict[str, Tensor]:
    """Decompress a delta-compressed blob back to a state dict."""
    raw = zlib.decompress(compressed)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

    base_layer_idx = obj["base_layer_idx"]
    base_prefix = f"layers.{base_layer_idx}."

    # Dequantize base
    base_suffix_map: dict[str, Tensor] = {}
    out: dict[str, Tensor] = {}
    for key, qdata in obj["base"].items():
        val = _dequantize_tensor_int8(qdata["q"], qdata["s"])
        out[key] = val
        suffix = key[len(base_prefix):]
        base_suffix_map[suffix] = val

    # Dequantize and reconstruct from deltas
    for key, qdata in obj["deltas"].items():
        delta = _dequantize_tensor_int8(qdata["q"], qdata["s"])
        m = _LAYER_RE.match(key)
        if m:
            idx = int(m.group(1))
            prefix = f"layers.{idx}."
            suffix = key[len(prefix):]
            if suffix in base_suffix_map and delta.shape == base_suffix_map[suffix].shape:
                out[key] = base_suffix_map[suffix] + delta
            else:
                out[key] = delta
        else:
            out[key] = delta

    # Non-layer weights
    for key, qdata in obj["non_layer"].items():
        out[key] = _dequantize_tensor_int8(qdata["q"], qdata["s"])

    return out


def compress_direct(state_dict: dict[str, Tensor]) -> bytes:
    """Baseline: int8 quantize + zlib without delta compression."""
    quantized: dict[str, dict[str, Tensor]] = {}
    for key, val in state_dict.items():
        q, s = _quantize_tensor_int8(val)
        quantized[key] = {"q": q, "s": s}

    obj = {"format": "direct_int8_v1", "tensors": quantized}
    buf = io.BytesIO()
    torch.save(obj, buf)
    return zlib.compress(buf.getvalue(), level=9)


def compare_compression(state_dict: dict[str, Tensor]) -> dict[str, Any]:
    """Compare direct vs delta compression ratios."""
    direct_blob = compress_direct(state_dict)
    delta_blob = compress_with_deltas(state_dict)

    raw_bytes = sum(t.nelement() * t.element_size() for t in state_dict.values())
    return {
        "raw_bytes": raw_bytes,
        "direct_int8_zlib_bytes": len(direct_blob),
        "delta_int8_zlib_bytes": len(delta_blob),
        "direct_ratio": raw_bytes / len(direct_blob) if len(direct_blob) > 0 else 0,
        "delta_ratio": raw_bytes / len(delta_blob) if len(delta_blob) > 0 else 0,
        "delta_savings_pct": (1 - len(delta_blob) / len(direct_blob)) * 100 if len(direct_blob) > 0 else 0,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m experiments.delta_compression <checkpoint_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both raw state_dict and wrapped formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("layers.") for k in ckpt):
        sd = ckpt
    else:
        sd = ckpt

    print("\n--- Layer Similarity Analysis ---")
    sim = analyze_layer_similarity(sd)
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  layers {i} <-> {j}: cosine sim = {sim[i,j]:.4f}")

    print("\n--- Compression Comparison ---")
    stats = compare_compression(sd)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

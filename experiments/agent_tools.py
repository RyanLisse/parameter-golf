"""Agent-Native Experiment Tools (RJC-39 + RJC-41).

Atomic tools for agent-driven experiment orchestration following agent-native principles:
- Parity: Each tool has clear input/output contracts
- Granularity: One concept per tool, composable
- Composability: Tools combine to form complex workflows
- Emergent Capability: Tool chains enable discovery
- Improvement Over Time: Learning from trial results

All functions return JSON-serializable dicts for agent integration.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import subprocess
import uuid
from pathlib import Path
from typing import Any


def submit_training_run(config: dict[str, str], backend: str = "mlx") -> dict[str, Any]:
    """
    Submit a training run subprocess with given hyperparameters.

    Args:
        config: Hyperparameter configuration dict (e.g., {"NUM_LAYERS": "9"})
        backend: "mlx" or "cuda"

    Returns:
        JSON-serializable dict with:
        - run_id: Unique identifier
        - status: "submitted" or "error"
        - message: Status message
    """
    run_id = f"ar_{uuid.uuid4().hex[:8]}"

    # Build env vars from config
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    for key, value in config.items():
        env[key] = str(value)

    try:
        # Launch training in background
        if backend == "mlx":
            cmd = ["uv", "run", "python3", "train_gpt_mlx.py"]
        else:
            cmd = ["uv", "run", "torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"]

        log_path = Path("logs/autoresearch") / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as log_file:
            subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

        return {
            "run_id": run_id,
            "status": "submitted",
            "message": f"Training run {run_id} submitted to {backend}",
            "log_path": str(log_path),
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "error",
            "message": f"Failed to submit run: {e}",
            "error": str(e),
        }


def read_val_bpb(run_id: str) -> dict[str, Any]:
    """
    Read validation bits-per-byte (val_bpb) from a completed run's log.

    Args:
        run_id: Run identifier

    Returns:
        JSON-serializable dict with:
        - status: "found", "not_found", or "error"
        - val_bpb: Float value or None
        - val_loss: Float value or None
        - log_path: Path to log file
    """
    log_path = Path("logs/autoresearch") / f"{run_id}.log"

    if not log_path.exists():
        return {
            "status": "not_found",
            "val_bpb": None,
            "val_loss": None,
            "log_path": str(log_path),
            "message": f"Log not found: {log_path}",
        }

    try:
        with open(log_path) as f:
            content = f.read()

        # Match pattern: final_int8_zlib_roundtrip.*val_loss: X.*val_bpb: Y
        pattern = (
            r"final_int8_zlib_roundtrip.*?"
            r"val_loss:\s*([-+0-9.eE]+).*?"
            r"val_bpb:\s*([-+0-9.eE]+)"
        )
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        if match:
            return {
                "status": "found",
                "val_loss": float(match.group(1)),
                "val_bpb": float(match.group(2)),
                "log_path": str(log_path),
            }
        else:
            return {
                "status": "not_found",
                "val_bpb": None,
                "val_loss": None,
                "log_path": str(log_path),
                "message": "val_bpb/val_loss not found in log",
            }
    except Exception as e:
        return {
            "status": "error",
            "val_bpb": None,
            "val_loss": None,
            "log_path": str(log_path),
            "error": str(e),
        }


def check_artifact_size(run_id: str) -> dict[str, Any]:
    """
    Check if the trained artifact fits within 16MB constraint.

    Args:
        run_id: Run identifier

    Returns:
        JSON-serializable dict with:
        - status: "found", "not_found", or "error"
        - fits_16mb: Boolean (true if size <= 16,000,000 bytes)
        - total_bytes: Integer size or None
        - artifact_limit: 16,000,000
    """
    log_path = Path("logs/autoresearch") / f"{run_id}.log"
    ARTIFACT_LIMIT = 16_000_000

    if not log_path.exists():
        return {
            "status": "not_found",
            "fits_16mb": False,
            "total_bytes": None,
            "artifact_limit": ARTIFACT_LIMIT,
        }

    try:
        with open(log_path) as f:
            content = f.read()

        # Look for total submission size pattern
        patterns = [
            r"total\s+submission\s+size\s+int8\+zlib:\s*(\d+)\s*bytes",
            r"total_bytes:\s*(\d+)",
        ]

        total_bytes = None
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                total_bytes = int(match.group(1))
                break

        if total_bytes is None:
            return {
                "status": "not_found",
                "fits_16mb": False,
                "total_bytes": None,
                "artifact_limit": ARTIFACT_LIMIT,
                "message": "Total artifact size not found in log",
            }

        return {
            "status": "found",
            "fits_16mb": total_bytes <= ARTIFACT_LIMIT,
            "total_bytes": total_bytes,
            "artifact_limit": ARTIFACT_LIMIT,
            "message": f"Size: {total_bytes} bytes ({'OK' if total_bytes <= ARTIFACT_LIMIT else 'OVER LIMIT'})",
        }
    except Exception as e:
        return {
            "status": "error",
            "fits_16mb": False,
            "total_bytes": None,
            "artifact_limit": ARTIFACT_LIMIT,
            "error": str(e),
        }


def mutate_hyperparams(
    config: dict[str, str],
    mutation_rate: float = 0.1,
) -> dict[str, str]:
    """
    Randomly mutate hyperparameters within plausible search ranges.

    Args:
        config: Current hyperparameter configuration
        mutation_rate: Probability of mutating each parameter (0-1)

    Returns:
        Mutated configuration dict (JSON-serializable).
    """
    from autoresearch.run_search import SEARCH_CHOICES

    mutated = dict(config)
    backend = "mlx" if config.get("TRAIN_BATCH_TOKENS", "524288") in [
        "8192", "16384", "32768", "65536"
    ] else "cuda"

    search_space = SEARCH_CHOICES.get(backend, {})

    for key in mutated:
        if random.random() < mutation_rate and key in search_space:
            choices = search_space[key]
            mutated[key] = random.choice(choices)

    return mutated


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """
    Compare multiple training runs by metrics.

    Args:
        run_ids: List of run identifiers

    Returns:
        JSON-serializable dict with:
        - status: "ok" or "error"
        - comparison: List of run results with val_bpb, artifact size
        - best_run_id: Run ID with best val_bpb
    """
    results = []
    best_run = None
    best_val_bpb = float("inf")

    for run_id in run_ids:
        metrics = read_val_bpb(run_id)
        size_check = check_artifact_size(run_id)

        result = {
            "run_id": run_id,
            "val_bpb": metrics.get("val_bpb"),
            "val_loss": metrics.get("val_loss"),
            "total_bytes": size_check.get("total_bytes"),
            "fits_16mb": size_check.get("fits_16mb"),
        }
        results.append(result)

        # Track best run
        if metrics.get("val_bpb") is not None:
            if metrics["val_bpb"] < best_val_bpb and size_check.get("fits_16mb", False):
                best_val_bpb = metrics["val_bpb"]
                best_run = run_id

    return {
        "status": "ok",
        "comparison": results,
        "best_run_id": best_run,
        "num_runs": len(run_ids),
    }


def get_experiment_status(run_id: str) -> dict[str, Any]:
    """
    Get current stage of an experiment.

    Args:
        run_id: Run identifier

    Returns:
        JSON-serializable dict with:
        - run_id: The run identifier
        - stage: One of "configuring", "training", "validating", "compressing", "scoring", "unknown"
        - status: Descriptive status message
    """
    log_path = Path("logs/autoresearch") / f"{run_id}.log"

    if not log_path.exists():
        return {
            "run_id": run_id,
            "stage": "unknown",
            "status": "No log file found",
        }

    try:
        with open(log_path) as f:
            content = f.read()

        # Infer stage from log content
        if "final_int8_zlib_roundtrip" in content:
            stage = "scoring"
        elif "val_bpb" in content or "val_loss" in content:
            stage = "validating"
        elif "compressing" in content.lower() or "int8" in content:
            stage = "compressing"
        elif "training" in content.lower() or "loss" in content:
            stage = "training"
        else:
            stage = "configuring"

        return {
            "run_id": run_id,
            "stage": stage,
            "status": f"Run {run_id} in stage: {stage}",
            "log_size_bytes": log_path.stat().st_size,
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "stage": "unknown",
            "status": f"Error reading log: {e}",
            "error": str(e),
        }

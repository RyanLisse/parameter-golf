#!/usr/bin/env python3
"""Modal GPU training dispatcher for Parameter Golf experiments.

Reads experiment config from env vars, runs training on Modal H100s,
and reports results back to Linear.

Usage:
    EXPERIMENT_ID=RJC-29 TRAINING_CONFIG='{"NUM_LAYERS":"9",...}' python scripts/modal_train.py

Environment variables:
    EXPERIMENT_ID: Linear issue ID (required)
    TRAINING_CONFIG: JSON dict of hyperparameter env vars (required)
    BACKEND: 'cuda' or 'mlx' (default: cuda)
    MODAL_TOKEN_ID: Modal authentication token ID (required)
    MODAL_TOKEN_SECRET: Modal authentication token secret (required)
    LINEAR_API_KEY: Linear API key for posting results (optional)
"""

from __future__ import annotations

import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import modal

# Modal app configuration
app = modal.App("parameter-golf-training")

# Training image with PyTorch, CUDA, and dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "requests",
    )
    .run_commands("pip install --upgrade pip")
)

# Volume for data persistence
data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    gpu=modal.gpu.H100(count=8),  # 8xH100 as per challenge spec
    timeout=900,  # 15 min max (10 min training + 5 min overhead)
    image=training_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("parameter-golf-secrets")],
)
def run_training(
    config: Dict[str, str],
    experiment_id: str,
    backend: str = "cuda",
) -> Dict[str, Any]:
    """Run training on Modal H100s with specified hyperparameters.

    Args:
        config: Dict of environment variable name -> value for hyperparameters
        experiment_id: Linear issue ID for tracking
        backend: 'cuda' or 'mlx'

    Returns:
        Dict with keys: success, val_bpb, model_size_bytes, total_bytes, duration_sec, error
    """
    import subprocess
    import time
    import re
    from pathlib import Path

    start_time = time.time()
    result = {
        "success": False,
        "experiment_id": experiment_id,
        "val_bpb": None,
        "model_size_bytes": None,
        "total_bytes": None,
        "duration_sec": None,
        "error": None,
    }

    try:
        # Clone repository (Modal container starts fresh)
        print(f"[modal] Cloning parameter-golf repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/openai/parameter-golf.git", "/workspace"],
            check=True,
            capture_output=True,
        )

        # Change to workspace directory
        os.chdir("/workspace")

        # Install Python dependencies
        print(f"[modal] Installing dependencies with uv...")
        subprocess.run(["pip", "install", "uv"], check=True, capture_output=True)

        if backend == "cuda":
            subprocess.run(["uv", "sync", "--extra", "cuda"], check=True, capture_output=True)
        else:
            subprocess.run(["uv", "sync", "--extra", "mlx"], check=True, capture_output=True)

        # Download training data if not cached
        if not Path("/data/datasets/fineweb10B_sp1024/train_000000.bin").exists():
            print(f"[modal] Downloading training data...")
            subprocess.run(
                ["uv", "run", "python", "download_fineweb10B.py", "10"],
                check=True,
                capture_output=True,
            )
            # Copy to persistent volume
            subprocess.run(
                ["cp", "-r", "data/datasets", "/data/"],
                check=True,
            )
            subprocess.run(
                ["cp", "-r", "data/tokenizers", "/data/"],
                check=True,
            )
        else:
            print(f"[modal] Using cached training data from volume")
            # Link to workspace
            subprocess.run(["ln", "-s", "/data/datasets", "data/datasets"], check=True)
            subprocess.run(["ln", "-s", "/data/tokenizers", "data/tokenizers"], check=True)

        # Build environment with hyperparameters
        env = os.environ.copy()
        env.update(config)

        print(f"[modal] Starting training with config: {json.dumps(config, indent=2)}")

        # Run training with torchrun (multi-GPU)
        if backend == "cuda":
            cmd = [
                "torchrun",
                "--nproc_per_node=8",  # 8xH100
                "train_gpt.py",
            ]
        else:
            # MLX doesn't use torchrun
            cmd = ["uv", "run", "python", "train_gpt_mlx.py"]

        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for training
        )

        output = proc.stdout + "\n" + proc.stderr
        print(output)  # Log full output to Modal logs

        if proc.returncode != 0:
            result["error"] = f"Training failed with exit code {proc.returncode}"
            return result

        # Parse output for metrics
        # Look for: final_int8_zlib_roundtrip val_bpb=X.XXXX model_bytes=NNNN total_bytes=NNNN
        pattern = r'final_int8_zlib_roundtrip\s+val_bpb=([\d.]+)\s+model_bytes=(\d+)\s+total_bytes=(\d+)'
        match = re.search(pattern, output)

        if match:
            result["val_bpb"] = float(match.group(1))
            result["model_size_bytes"] = int(match.group(2))
            result["total_bytes"] = int(match.group(3))
            result["success"] = True
        else:
            result["error"] = "Failed to parse val_bpb from training output"
            return result

        # Validate 16MB constraint
        if result["total_bytes"] > 16_000_000:
            result["error"] = f"Artifact size {result['total_bytes']} exceeds 16MB limit"
            result["success"] = False
            return result

        result["duration_sec"] = time.time() - start_time
        print(f"[modal] Training completed successfully:")
        print(f"  val_bpb: {result['val_bpb']}")
        print(f"  model_bytes: {result['model_size_bytes']}")
        print(f"  total_bytes: {result['total_bytes']}")
        print(f"  duration: {result['duration_sec']:.1f}s")

        return result

    except subprocess.TimeoutExpired:
        result["error"] = "Training exceeded 10 minute timeout"
        result["duration_sec"] = time.time() - start_time
        return result
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        result["duration_sec"] = time.time() - start_time
        return result


def post_result_to_linear(
    experiment_id: str,
    result: Dict[str, Any],
    api_key: Optional[str] = None,
) -> bool:
    """Post training results back to Linear issue as a comment.

    Args:
        experiment_id: Linear issue ID (e.g., RJC-29)
        result: Training result dict from run_training
        api_key: Linear API key (reads from env if not provided)

    Returns:
        True if posted successfully, False otherwise
    """
    if not api_key:
        api_key = os.environ.get("LINEAR_API_KEY")

    if not api_key:
        print("[linear] No LINEAR_API_KEY found, skipping result posting")
        return False

    import requests

    # Format result as markdown comment
    if result["success"]:
        comment = f"""## Training Results ✅

**Experiment:** {experiment_id}
**Status:** SUCCESS

### Metrics
- **val_bpb:** {result['val_bpb']:.4f}
- **Model Size:** {result['model_size_bytes']:,} bytes
- **Total Size:** {result['total_bytes']:,} bytes ({result['total_bytes']/16_000_000*100:.1f}% of 16MB limit)
- **Duration:** {result['duration_sec']:.1f}s

### Configuration
```json
{json.dumps(json.loads(os.environ.get('TRAINING_CONFIG', '{}')), indent=2)}
```

*Run by Modal GPU training workflow*
"""
    else:
        comment = f"""## Training Results ❌

**Experiment:** {experiment_id}
**Status:** FAILED

### Error
```
{result.get('error', 'Unknown error')}
```

### Partial Metrics
- **val_bpb:** {result['val_bpb'] or 'N/A'}
- **Duration:** {result['duration_sec']:.1f}s

*Run by Modal GPU training workflow*
"""

    # GraphQL mutation to add comment to issue
    query = """
    mutation AddComment($issueId: String!, $body: String!) {
      commentCreate(input: {issueId: $issueId, body: $body}) {
        success
        comment {
          id
        }
      }
    }
    """

    # First, get issue ID from identifier (e.g., RJC-29)
    search_query = """
    query GetIssue($identifier: String!) {
      issue(id: $identifier) {
        id
      }
    }
    """

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    try:
        # Get issue ID
        resp = requests.post(
            "https://api.linear.app/graphql",
            json={"query": search_query, "variables": {"identifier": experiment_id}},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            print(f"[linear] GraphQL error: {data['errors']}")
            return False

        issue_id = data.get("data", {}).get("issue", {}).get("id")
        if not issue_id:
            print(f"[linear] Issue {experiment_id} not found")
            return False

        # Post comment
        resp = requests.post(
            "https://api.linear.app/graphql",
            json={
                "query": query,
                "variables": {"issueId": issue_id, "body": comment},
            },
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("data", {}).get("commentCreate", {}).get("success"):
            print(f"[linear] Posted results to {experiment_id}")
            return True
        else:
            print(f"[linear] Failed to post comment: {data}")
            return False

    except Exception as e:
        print(f"[linear] Error posting to Linear: {e}")
        return False


@app.local_entrypoint()
def main():
    """Entry point for Modal app - reads config from env and runs training."""
    experiment_id = os.environ.get("EXPERIMENT_ID")
    config_json = os.environ.get("TRAINING_CONFIG")
    backend = os.environ.get("BACKEND", "cuda")

    if not experiment_id:
        print("Error: EXPERIMENT_ID environment variable required", file=sys.stderr)
        sys.exit(1)

    if not config_json:
        print("Error: TRAINING_CONFIG environment variable required", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in TRAINING_CONFIG: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Modal training for {experiment_id}")
    print(f"Backend: {backend}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Run training on Modal
    result = run_training.remote(config, experiment_id, backend)

    print(f"\nTraining completed:")
    print(json.dumps(result, indent=2))

    # Post results to Linear
    if os.environ.get("LINEAR_API_KEY"):
        post_result_to_linear(experiment_id, result)

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

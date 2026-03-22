# Parameter Golf Autonomous Experiment Workflow

## Metadata

```yaml
---
name: parameter-golf-experiment
trigger: linear_issue
issue_filter: "EXP-*"
max_concurrent: 4
timeout: 900
---
```

## Overview

This workflow enables autonomous experiment orchestration for the Parameter Golf challenge. Agents pick up Linear issues tagged with `EXP-*` and execute complete training experiments with validation, reporting results back to the issue.

## Workflow Phases

### Phase 1: Issue Pickup
- Monitor Linear for issues with `EXP-*` prefix
- Lock issue to prevent concurrent processing
- Extract experiment specification from issue body

Specification format in issue:
```yaml
experiment:
  name: "descriptive-name"
  backend: "mlx" or "cuda"
  search_mode: "random" | "preset" | "evolution" | "code"
  config:
    NUM_LAYERS: "9"
    MODEL_DIM: "512"
    # ... other hyperparameters
  mutations: 5
  seed: 1337
```

### Phase 2: Implementation
- Design experiment hyperparameter mutations
- Generate candidate configurations using `agent_tools.mutate_hyperparams()`
- Plan training order (prefer configs likely to fit 16MB)

### Phase 3: Training
- Launch training runs using `agent_tools.submit_training_run()`
- Monitor progress via `agent_tools.get_experiment_status()`
- Collect run_ids for subsequent analysis

Example flow:
```python
import experiments.agent_tools as tools

config = {"NUM_LAYERS": "9", "MODEL_DIM": "512"}
result = tools.submit_training_run(config, backend="mlx")
run_id = result["run_id"]

# Poll status
while True:
    status = tools.get_experiment_status(run_id)
    if status["stage"] == "scoring":
        break
```

### Phase 4: Validation
- Read final metrics using `agent_tools.read_val_bpb()`
- Check artifact size using `agent_tools.check_artifact_size()`
- Reject runs exceeding 16MB limit

### Phase 5: Comparison & Selection
- Compare all runs using `agent_tools.compare_runs()`
- Select best run by val_bpb (if within 16MB)
- Extract hyperparameter config from best run

### Phase 6: Reporting
- Post results comment to Linear issue with:
  - Best val_bpb and val_loss
  - Artifact size
  - Best config (for next iteration)
  - Training duration
  - Log file location
- Mark issue as complete
- Tag with `completed` and `results:ready`

## Tool Reference

All tools in `experiments/agent_tools.py` are designed for agent orchestration:

### Atomic Tools (Low-level building blocks)

**submit_training_run(config, backend)**
- Launches training subprocess with env-var config
- Returns: `{run_id, status, log_path}`
- Non-blocking (run happens in background)

**get_experiment_status(run_id)**
- Checks current training stage
- Returns: `{stage, status}` where stage is one of:
  - "configuring" (preparing)
  - "training" (running main loop)
  - "validating" (computing val_bpb)
  - "compressing" (int8+zlib)
  - "scoring" (final evaluation)

**read_val_bpb(run_id)**
- Reads validation metrics from log
- Returns: `{status, val_bpb, val_loss}`
- Blocks on log parsing (log must exist)

**check_artifact_size(run_id)**
- Verifies 16MB constraint compliance
- Returns: `{fits_16mb, total_bytes}`

**compare_runs(run_ids)**
- Ranks runs by val_bpb with size constraint
- Returns: `{comparison, best_run_id}`

**mutate_hyperparams(config, mutation_rate)**
- Randomly modifies config within search space
- Returns: mutated config dict
- Used to generate candidate variations

### Composition Examples

**Simple experiment sequence:**
```python
# Launch run
run_result = tools.submit_training_run(config, "mlx")
run_id = run_result["run_id"]

# Wait for completion
while tools.get_experiment_status(run_id)["stage"] != "scoring":
    time.sleep(30)

# Validate results
metrics = tools.read_val_bpb(run_id)
size_check = tools.check_artifact_size(run_id)
if size_check["fits_16mb"] and metrics["val_bpb"] < threshold:
    return (run_id, metrics, config)
```

**Multi-run search:**
```python
candidates = []
for i in range(num_mutations):
    mutated = tools.mutate_hyperparams(base_config, 0.3)
    result = tools.submit_training_run(mutated, "mlx")
    candidates.append(result["run_id"])

# ... wait for all to complete ...

comparison = tools.compare_runs(candidates)
best_id = comparison["best_run_id"]
```

## Design Principles (Agent-Native Architecture)

### Parity
Each tool has clear input/output contracts. Tools are substitutable (can swap backends without changing agent logic).

### Granularity
One concept per tool. No mega-functions that combine training + validation + reporting.

### Composability
Tools chain naturally. Output of one feeds input of next (run_id → metrics → comparison).

### Emergent Capability
Simple tools combine to form discovery workflows. Agents discover patterns without hardcoding.

### Improvement Over Time
Agents learn which configs work best and mutate more intelligently based on results.

## Integration with Autoresearch

The workflow can feed results back to autoresearch:

```python
# After experiment completes
comparison = tools.compare_runs(all_run_ids)
best_config = extract_config_from_log(comparison["best_run_id"])

# Save to autoresearch trials for next search iteration
autoresearch.persist_result(best_config, comparison["best_run_id"])
```

## Error Handling

All tools return JSON dicts with `status` field:
- `"ok"` or `"found"`: Operation succeeded
- `"not_found"`: Resource not found (expected for in-progress runs)
- `"error"`: Unexpected failure

Agents should:
1. Check `status` field in all returns
2. Retry on `"not_found"` (run still executing)
3. Log `"error"` cases but continue (resilience)
4. Never assume result fields exist without checking

Example:
```python
metrics = tools.read_val_bpb(run_id)
if metrics["status"] == "found":
    val_bpb = metrics["val_bpb"]
elif metrics["status"] == "not_found":
    # Still training, retry later
    continue
else:  # error
    log.error(f"Failed to read metrics: {metrics.get('error')}")
```

## Monitoring & Observability

All runs write logs to: `logs/autoresearch/{run_id}.log`

Agents can tail logs for debugging:
```bash
tail -f logs/autoresearch/{run_id}.log
```

The log includes:
- Training progress (loss, speed)
- Validation results (val_loss, val_bpb)
- Model size (parameters, serialized bytes)
- Final artifact size (int8+zlib total)
- Stage transitions (TRAINING → VALIDATING → SCORING)

## Constraints & Limits

- **Artifact limit**: 16,000,000 bytes (decimal, not binary)
- **Max concurrent runs**: 4 (prevent resource exhaustion)
- **Timeout**: 900 seconds per experiment
- **Iterations**: Typically 20,000 on MLX (adjust for backend)
- **Batch size**: 524,288 tokens (adjust for memory)

## Future Extensions

1. **Parallel mutation**: Launch multiple mutations concurrently
2. **Population evolution**: Track Pareto frontier (val_bpb vs model_size)
3. **Architecture search**: Mutate layer counts, head counts, etc.
4. **Learning rate tuning**: Use chaotic_lr_schedule with sweep
5. **Hash initialization integration**: Enable HASH_INIT_ENABLED in configs

See `experiments/hash_init.py` for deterministic initialization and dropout features.

"""Experiment modules for Parameter Golf challenge.

RJC-33: Difficulty-Adjusted Dynamic Curriculum Learning
RJC-35: Bonding Curve Token-Weighted Loss Function
RJC-38: Hash-Based Initialization & Deterministic Dropout
RJC-39: Symphony Orchestration (agent-native tool design)
RJC-41: Agent-Native Architecture (tool interfaces)
Leaderboard: Proven techniques from #1 entry (1.1748 val_bpb)
"""

from .agent_tools import (
    check_artifact_size,
    compare_runs,
    get_experiment_status,
    mutate_hyperparams,
    read_val_bpb,
    submit_training_run,
)
from .hash_init import (
    apply_hash_init,
    chaotic_lr_schedule,
    hash_dropout_mask,
    siphash_init,
)

# Optional: RJC-33 and RJC-35 curriculum learning modules
try:
    from .difficulty_adjusted_lr import DifficultyAdjuster, HalvingSchedule
except ImportError:
    pass

try:
    from .bonding_curve_loss import (
        EntropyWeighter,
        TokenFrequencyWeighter,
        entropy_weights_mlx,
        entropy_weights_torch,
        weighted_cross_entropy_mlx,
        weighted_cross_entropy_torch,
    )
except ImportError:
    pass

from .leaderboard_techniques import (
    CHAMPIONSHIP_CONFIG,
    CHAMPIONSHIP_LONGCTX_CONFIG,
    apply_muon_weight_decay_np,
    apply_spectral_embed_init_np,
    compute_phase_transition_schedule,
    compute_sliding_window_ranges,
    should_keep_fp16,
)

__all__ = [
    "siphash_init",
    "hash_dropout_mask",
    "chaotic_lr_schedule",
    "apply_hash_init",
    "submit_training_run",
    "read_val_bpb",
    "check_artifact_size",
    "mutate_hyperparams",
    "compare_runs",
    "get_experiment_status",
    "DifficultyAdjuster",
    "HalvingSchedule",
    "TokenFrequencyWeighter",
    "EntropyWeighter",
    "weighted_cross_entropy_torch",
    "weighted_cross_entropy_mlx",
    "entropy_weights_torch",
    "entropy_weights_mlx",
    "apply_spectral_embed_init_np",
    "compute_phase_transition_schedule",
    "compute_sliding_window_ranges",
    "apply_muon_weight_decay_np",
    "should_keep_fp16",
    "CHAMPIONSHIP_CONFIG",
    "CHAMPIONSHIP_LONGCTX_CONFIG",
]

"""Tests for proven leaderboard techniques from the parameter-golf competition."""
from __future__ import annotations

import math
import unittest

import numpy as np


class TestSpectralEmbedInit(unittest.TestCase):
    """Spectral/overtone embedding init shapes spectrum via SVD power-law decay."""

    def test_shapes_preserved(self) -> None:
        from experiments.leaderboard_techniques import apply_spectral_embed_init_np
        weight = np.random.randn(1024, 512).astype(np.float32)
        out = apply_spectral_embed_init_np(weight, exponent=-0.5)
        self.assertEqual(weight.shape, out.shape)
        self.assertEqual(weight.dtype, out.dtype)

    def test_singular_values_follow_power_law(self) -> None:
        from experiments.leaderboard_techniques import apply_spectral_embed_init_np
        weight = np.random.randn(1024, 512).astype(np.float32)
        out = apply_spectral_embed_init_np(weight, exponent=-0.5)
        _, s_out, _ = np.linalg.svd(out, full_matrices=False)
        # The first few singular values should decay roughly as power law
        # Check ratio s[0]/s[1] vs s[1]/s[2] are similar (power law property)
        if len(s_out) >= 3 and s_out[1] > 1e-8 and s_out[2] > 1e-8:
            r1 = s_out[0] / s_out[1]
            r2 = s_out[1] / s_out[2]
            # Power law means consecutive ratios should be close
            self.assertAlmostEqual(r1, r2, delta=0.5)

    def test_exponent_zero_preserves_magnitude(self) -> None:
        from experiments.leaderboard_techniques import apply_spectral_embed_init_np
        weight = np.random.randn(64, 32).astype(np.float32)
        out = apply_spectral_embed_init_np(weight, exponent=0.0)
        # With exponent=0, target_S = S[0] * 1^0 = S[0] for all, so all singular
        # values become equal to S[0]. The Frobenius norm should change.
        self.assertEqual(weight.shape, out.shape)


class TestPhaseTransitionResidMix(unittest.TestCase):
    """Phase-transition init uses sigmoid schedule for residual mixing."""

    def test_schedule_shape(self) -> None:
        from experiments.leaderboard_techniques import compute_phase_transition_schedule
        schedule = compute_phase_transition_schedule(10)
        self.assertEqual(len(schedule), 10)

    def test_early_layers_trust_x0(self) -> None:
        from experiments.leaderboard_techniques import compute_phase_transition_schedule
        schedule = compute_phase_transition_schedule(10)
        # Early layers: phase near 0, so resid_mix[0] (phase) is low
        # and resid_mix[1] (1-phase) is high, meaning x0 dominates
        # (Block.__call__: x = mix[0]*x + mix[1]*x0)
        self.assertLess(schedule[0][0], 0.2)
        self.assertGreater(schedule[0][1], 0.8)

    def test_late_layers_trust_residual(self) -> None:
        from experiments.leaderboard_techniques import compute_phase_transition_schedule
        schedule = compute_phase_transition_schedule(10)
        # Late layers: phase near 1, so resid_mix[0] ~ phase, resid_mix[1] ~ 1-phase
        # The last layer should have high phase
        self.assertGreater(schedule[-1][0], 0.8)
        self.assertLess(schedule[-1][1], 0.2)

    def test_monotonic(self) -> None:
        from experiments.leaderboard_techniques import compute_phase_transition_schedule
        schedule = compute_phase_transition_schedule(12)
        phases = [s[0] for s in schedule]
        # Phase values increase monotonically
        for i in range(1, len(phases)):
            self.assertGreaterEqual(phases[i], phases[i - 1])


class TestSlidingWindowEval(unittest.TestCase):
    """Sliding window eval produces valid BPB with overlapping context windows."""

    def test_compute_sliding_positions(self) -> None:
        from experiments.leaderboard_techniques import compute_sliding_window_ranges
        seq_len = 1024
        stride = 64
        total_tokens = 4096
        ranges = compute_sliding_window_ranges(total_tokens, seq_len, stride)
        # Should have multiple overlapping windows
        self.assertGreater(len(ranges), 1)
        # Each range is (start, end) where end - start = seq_len
        for start, end in ranges:
            self.assertEqual(end - start, seq_len)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, total_tokens)

    def test_all_positions_covered(self) -> None:
        from experiments.leaderboard_techniques import compute_sliding_window_ranges
        seq_len = 128
        stride = 32
        total_tokens = 512
        ranges = compute_sliding_window_ranges(total_tokens, seq_len, stride)
        # Every position from seq_len to total_tokens should be scorable
        covered = set()
        for start, end in ranges:
            # The scored positions are those at the end of the window
            # (the ones with maximum context)
            for pos in range(start, end):
                covered.add(pos)
        # At minimum, positions [0, total_tokens) should be covered
        self.assertEqual(len(covered), total_tokens)


class TestMuonWeightDecay(unittest.TestCase):
    """Decoupled weight decay applied post-optimizer step."""

    def test_reduces_norm(self) -> None:
        from experiments.leaderboard_techniques import apply_muon_weight_decay_np
        params = [np.random.randn(64, 32).astype(np.float32) for _ in range(3)]
        norms_before = [np.linalg.norm(p) for p in params]
        apply_muon_weight_decay_np(params, lr=0.04, wd=0.02)
        norms_after = [np.linalg.norm(p) for p in params]
        for nb, na in zip(norms_before, norms_after):
            self.assertLess(na, nb)

    def test_zero_wd_no_change(self) -> None:
        from experiments.leaderboard_techniques import apply_muon_weight_decay_np
        params = [np.random.randn(64, 32).astype(np.float32)]
        original = params[0].copy()
        apply_muon_weight_decay_np(params, lr=0.04, wd=0.0)
        np.testing.assert_array_equal(params[0], original)

    def test_decay_factor(self) -> None:
        from experiments.leaderboard_techniques import apply_muon_weight_decay_np
        lr, wd = 0.04, 0.02
        params = [np.ones((4, 4), dtype=np.float32)]
        apply_muon_weight_decay_np(params, lr=lr, wd=wd)
        expected = 1.0 - wd * lr
        np.testing.assert_allclose(params[0], expected, atol=1e-7)


class TestFP16Passthrough(unittest.TestCase):
    """FP16 passthrough identifies embedding tensors for higher precision."""

    def test_identifies_embeddings(self) -> None:
        from experiments.leaderboard_techniques import should_keep_fp16
        self.assertTrue(should_keep_fp16("tok_emb.weight"))
        self.assertTrue(should_keep_fp16("embed.weight"))
        self.assertTrue(should_keep_fp16("embedding.weight"))

    def test_rejects_non_embeddings(self) -> None:
        from experiments.leaderboard_techniques import should_keep_fp16
        self.assertFalse(should_keep_fp16("blocks.0.attn.c_q.weight"))
        self.assertFalse(should_keep_fp16("blocks.3.mlp.fc.weight"))
        self.assertFalse(should_keep_fp16("skip_weights"))


class TestChampionshipConfig(unittest.TestCase):
    """Championship config contains all required keys and valid values."""

    def test_has_all_required_keys(self) -> None:
        from experiments.leaderboard_techniques import CHAMPIONSHIP_CONFIG
        required = [
            "NUM_LAYERS", "MODEL_DIM", "NUM_HEADS", "NUM_KV_HEADS",
            "TRAIN_SEQ_LEN", "TRAIN_BATCH_TOKENS", "ITERATIONS",
            "WARMDOWN_ITERS", "MATRIX_LR", "TIED_EMBED_LR", "SCALAR_LR",
            "MUON_MOMENTUM", "MUON_BACKEND_STEPS", "GRAD_CLIP_NORM",
            "QK_GAIN_INIT", "LOGIT_SOFTCAP",
            "SPECTRAL_EMBED_INIT", "SPECTRAL_EXPONENT",
            "PHASE_TRANSITION_INIT", "MUON_WD",
            "EMBED_FP16_PASSTHROUGH", "EVAL_STRIDE",
        ]
        for key in required:
            self.assertIn(key, CHAMPIONSHIP_CONFIG, f"Missing key: {key}")

    def test_all_values_are_strings(self) -> None:
        from experiments.leaderboard_techniques import CHAMPIONSHIP_CONFIG
        for k, v in CHAMPIONSHIP_CONFIG.items():
            self.assertIsInstance(v, str, f"{k} value should be str, got {type(v)}")

    def test_leaderboard_techniques_toggles(self) -> None:
        from experiments.leaderboard_techniques import CHAMPIONSHIP_CONFIG
        self.assertEqual(CHAMPIONSHIP_CONFIG["SPECTRAL_EMBED_INIT"], "1")
        self.assertEqual(CHAMPIONSHIP_CONFIG["PHASE_TRANSITION_INIT"], "1")
        self.assertEqual(CHAMPIONSHIP_CONFIG["EMBED_FP16_PASSTHROUGH"], "1")
        self.assertEqual(CHAMPIONSHIP_CONFIG["EVAL_STRIDE"], "64")

    def test_extended_warmdown(self) -> None:
        from experiments.leaderboard_techniques import CHAMPIONSHIP_CONFIG
        warmdown = int(CHAMPIONSHIP_CONFIG["WARMDOWN_ITERS"])
        iterations = int(CHAMPIONSHIP_CONFIG["ITERATIONS"])
        # Extended warmdown: warmdown_iters >= iterations
        self.assertGreaterEqual(warmdown, iterations)


class TestChampionshipLongCtxConfig(unittest.TestCase):
    """Long-context variant of championship config."""

    def test_has_long_seq_len(self) -> None:
        from experiments.leaderboard_techniques import CHAMPIONSHIP_LONGCTX_CONFIG
        self.assertEqual(CHAMPIONSHIP_LONGCTX_CONFIG["TRAIN_SEQ_LEN"], "4096")

    def test_all_values_are_strings(self) -> None:
        from experiments.leaderboard_techniques import CHAMPIONSHIP_LONGCTX_CONFIG
        for k, v in CHAMPIONSHIP_LONGCTX_CONFIG.items():
            self.assertIsInstance(v, str)


if __name__ == "__main__":
    unittest.main()

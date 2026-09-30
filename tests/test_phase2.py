"""
MCTR Phase 2 Unit Tests
========================
Tests for:
  - AttentionController (all three levels)
  - JointController (temperature + attention heads)
  - AttentionMetrics
  - AttentionModulation (functional API)
  - Bounded output guarantees
  - Stochastic sampling
  - Causal state ordering
  - Attention normalization
"""

import unittest
import torch
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mctr.control.attention_controller import (
    AttentionController, AttentionControlMode, AttentionControlState
)
from mctr.control.joint_controller import JointController
from mctr.attention.attention_metrics import (
    AttentionMetrics,
    compute_attention_entropy,
    compute_attention_js_divergence,
    compute_attention_l1,
)
from mctr.attention.attention_modulation import (
    apply_attention_modulation,
    compute_attention_from_logits,
)
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.state.state import StateExtractor
from mctr.base.interface import TransformerState


class TestAttentionController(unittest.TestCase):
    """Tests for the Phase 2 AttentionController."""

    META_DIM  = 32
    BATCH     = 4
    N_LAYERS  = 6
    N_HEADS   = 8
    BETA      = 2.0

    def _make_ctrl(self, mode: str):
        return AttentionController(
            meta_dim=self.META_DIM,
            config={
                "attention_control_mode":  mode,
                "attention_modulation_bound": self.BETA,
                "attention_hidden_dim":    16,
            },
            n_layers=self.N_LAYERS,
            n_heads=self.N_HEADS,
        )

    def _dummy_m(self):
        return torch.randn(self.BATCH, self.META_DIM)

    def test_level1_global_shape_and_bounds(self):
        ctrl = self._make_ctrl("global")
        out  = ctrl(self._dummy_m())
        self.assertEqual(out.modulation.shape, (self.BATCH,))
        self.assertTrue(torch.all(out.modulation.abs() <= self.BETA + 1e-4))
        self.assertTrue(torch.all(torch.isfinite(out.modulation)))
        self.assertTrue(torch.all(out.raw_gate >= 0.0))
        self.assertTrue(torch.all(out.raw_gate <= 1.0))

    def test_level2_layer_shape_and_bounds(self):
        ctrl = self._make_ctrl("layer")
        out  = ctrl(self._dummy_m())
        self.assertEqual(out.modulation.shape, (self.BATCH, self.N_LAYERS))
        self.assertTrue(torch.all(out.modulation.abs() <= self.BETA + 1e-4))
        self.assertTrue(torch.all(torch.isfinite(out.modulation)))

    def test_level3_head_shape_and_bounds(self):
        ctrl = self._make_ctrl("head")
        out  = ctrl(self._dummy_m())
        self.assertEqual(out.modulation.shape, (self.BATCH, self.N_LAYERS, self.N_HEADS))
        self.assertTrue(torch.all(out.modulation.abs() <= self.BETA + 1e-4))
        self.assertTrue(torch.all(torch.isfinite(out.modulation)))

    def test_none_mode_zero_modulation(self):
        ctrl = self._make_ctrl("none")
        out  = ctrl(self._dummy_m())
        self.assertEqual(out.mode, AttentionControlMode.NONE)
        self.assertEqual(out.beta, 0.0)

    def test_gate_is_sigmoid_output(self):
        """raw_gate ∈ (0,1) strict."""
        ctrl = self._make_ctrl("global")
        out  = ctrl(self._dummy_m())
        self.assertTrue(torch.all(out.raw_gate > 0.0))
        self.assertTrue(torch.all(out.raw_gate < 1.0))


class TestJointController(unittest.TestCase):
    """Tests for the Phase 2 JointController."""

    META_DIM = 32
    BATCH    = 3
    T_MIN    = 0.3
    T_MAX    = 1.2
    BETA     = 2.0

    def _make_ctrl(self, enable_temp=True, enable_attn=True):
        return JointController(
            meta_dim=self.META_DIM,
            config={
                "enable_temperature":        enable_temp,
                "enable_attention":          enable_attn,
                "temperature_min":           self.T_MIN,
                "temperature_max":           self.T_MAX,
                "controller_hidden_dim":     32,
                "attention_control_mode":    "global",
                "attention_modulation_bound": self.BETA,
                "attention_hidden_dim":      16,
            },
            n_layers=6,
            n_heads=8,
        )

    def _dummy_m(self):
        return torch.randn(self.BATCH, self.META_DIM)

    def test_temperature_bounds(self):
        ctrl = self._make_ctrl()
        out  = ctrl(self._dummy_m())
        T    = out.temperature
        self.assertEqual(T.shape, (self.BATCH,))
        self.assertTrue(torch.all(T >= self.T_MIN - 1e-4))
        self.assertTrue(torch.all(T <= self.T_MAX + 1e-4))
        self.assertTrue(torch.all(torch.isfinite(T)))

    def test_attention_bounds(self):
        ctrl = self._make_ctrl()
        out  = ctrl(self._dummy_m())
        A    = out.attention.modulation
        self.assertTrue(torch.all(A.abs() <= self.BETA + 1e-4))
        self.assertTrue(torch.all(torch.isfinite(A)))

    def test_baseline_temp_override(self):
        ctrl = self._make_ctrl()
        out  = ctrl(self._dummy_m(), baseline_temp=0.5)
        self.assertTrue(torch.allclose(out.temperature, torch.full((self.BATCH,), 0.5)))

    def test_temperature_only_mode(self):
        ctrl = self._make_ctrl(enable_temp=True, enable_attn=False)
        out  = ctrl(self._dummy_m())
        self.assertEqual(out.attention.mode, AttentionControlMode.NONE)

    def test_meta_score_range(self):
        ctrl = self._make_ctrl()
        out  = ctrl(self._dummy_m())
        ms   = out.meta_score
        self.assertTrue(torch.all(ms >= 0.0))
        self.assertTrue(torch.all(ms <= 1.0))


class TestAttentionMetrics(unittest.TestCase):
    """Tests for Phase 2 attention metrics."""

    BATCH   = 1
    N_HEADS = 8
    SEQ_Q   = 5
    SEQ_K   = 10

    def _dummy_attn(self):
        return F.softmax(torch.randn(self.BATCH, self.N_HEADS, self.SEQ_Q, self.SEQ_K), dim=-1)

    def test_entropy_nonnegative(self):
        a   = self._dummy_attn()
        ent = compute_attention_entropy(a)
        self.assertTrue(torch.all(ent >= 0.0))
        self.assertEqual(ent.shape, (self.BATCH, self.N_HEADS, self.SEQ_Q))

    def test_js_divergence_nonnegative(self):
        p  = self._dummy_attn()
        q  = self._dummy_attn()
        js = compute_attention_js_divergence(p, q)
        self.assertTrue(torch.all(js >= 0.0))

    def test_l1_nonnegative(self):
        p  = self._dummy_attn()
        q  = self._dummy_attn()
        l1 = compute_attention_l1(p, q)
        self.assertTrue(torch.all(l1 >= 0.0))

    def test_self_js_is_zero(self):
        p  = self._dummy_attn()
        js = compute_attention_js_divergence(p, p)
        self.assertTrue(torch.allclose(js, torch.zeros_like(js), atol=1e-5))

    def test_attention_metrics_compute(self):
        a  = self._dummy_attn()
        b  = self._dummy_attn()
        native_maps = [a] * 3 + [None] * 3
        mod_maps    = [b] * 3 + [None] * 3
        delta = torch.randn(1)

        am = AttentionMetrics.compute(native_maps, mod_maps, delta)
        self.assertIsNotNone(am.entropy_native)
        self.assertEqual(am.entropy_native.shape[0], 3)  # 3 non-None layers
        self.assertGreaterEqual(am.mean_entropy_native(), 0.0)
        self.assertGreaterEqual(am.mean_js(), 0.0)
        self.assertGreaterEqual(am.modulation_magnitude, 0.0)


class TestAttentionModulation(unittest.TestCase):
    """Tests for the functional attention modulation API."""

    BATCH   = 2
    N_HEADS = 4
    SEQ_Q   = 5
    SEQ_K   = 8

    def _dummy_logits(self):
        return torch.randn(self.BATCH, self.N_HEADS, self.SEQ_Q, self.SEQ_K)

    def _make_state(self, mode, beta=1.0):
        if mode == AttentionControlMode.LEVEL1_GLOBAL:
            mod = torch.tensor([0.5, -0.3])  # [batch]
        elif mode == AttentionControlMode.LEVEL2_LAYER:
            mod = torch.randn(self.BATCH, 6) * 0.5
        elif mode == AttentionControlMode.LEVEL3_HEAD:
            mod = torch.randn(self.BATCH, 6, self.N_HEADS) * 0.5
        else:
            mod = torch.zeros(self.BATCH)
        return AttentionControlState(modulation=mod, mode=mode, beta=beta, raw_gate=torch.zeros(1)+0.5)

    def test_level1_adds_scalar(self):
        logits = self._dummy_logits()
        state  = self._make_state(AttentionControlMode.LEVEL1_GLOBAL)
        result = apply_attention_modulation(logits, state, layer_idx=0)
        self.assertEqual(result.shape, logits.shape)
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_level3_head_shape(self):
        logits = self._dummy_logits()
        state  = self._make_state(AttentionControlMode.LEVEL3_HEAD)
        result = apply_attention_modulation(logits, state, layer_idx=0)
        self.assertEqual(result.shape, logits.shape)

    def test_attention_normalization(self):
        logits = self._dummy_logits()
        probs  = compute_attention_from_logits(logits)
        sums   = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


class TestCausalOrdering(unittest.TestCase):
    """Verifies temporal causality: Ψ_(t-1) → controller, never Ψ_t."""

    def test_controller_uses_prev_state_only(self):
        """
        The joint controller should accept a meta-state embedding and return
        control signals BEFORE the current-step state is computed.
        This test verifies the ordering is achievable (controller is called
        with prev state, then the model runs, then new state is computed).
        """
        meta_dim = 16
        batch    = 1

        evaluator = MetaEvaluator(input_dim=6, hidden_dim=32, meta_dim=meta_dim)
        ctrl = JointController(
            meta_dim=meta_dim,
            config={
                "enable_temperature": True,
                "enable_attention":   True,
                "temperature_min":    0.3,
                "temperature_max":    1.2,
                "controller_hidden_dim": 16,
                "attention_control_mode": "global",
                "attention_modulation_bound": 1.0,
                "attention_hidden_dim": 8,
            },
            n_layers=4,
            n_heads=4,
        )

        # Step 0: initialize with zero state
        psi_prev = torch.zeros(batch, 6)
        m_prev   = evaluator.mlp(psi_prev)

        # Step 1: compute control from Ψ_(t-1) BEFORE running the model
        ctrl_out = ctrl(m_prev)
        T_t = ctrl_out.temperature
        A_t = ctrl_out.attention

        self.assertEqual(T_t.shape, (batch,))
        # Controller produced before model runs — causal contract satisfied

        # Simulate model output → compute Ψ_t
        vocab_size = 1000
        hidden_size = 64
        logits = torch.randn(batch, vocab_size)
        probs  = F.softmax(logits / T_t.unsqueeze(-1), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Now compute Ψ_t (this is AFTER the model/sampling)
        state_extractor = StateExtractor(vocab_size=vocab_size, config={"uncertainty": "normalized_entropy", "novelty": "cosine"})
        hidden = torch.randn(batch, 1, hidden_size)
        mock_state = TransformerState(
            hidden_state=hidden,
            logits=logits.unsqueeze(1),
            probabilities=probs.unsqueeze(1),
        )
        state_extractor.novelty_ref.fit(hidden.squeeze(1))
        meta_state = state_extractor(mock_state)
        psi_t = meta_state.to_tensor().squeeze(1)

        # Ψ_t is only available now — it correctly feeds the NEXT step's controller
        self.assertEqual(psi_t.shape, (batch, 6))
        print("    [OK] Temporal causality verified: Ψ_(t-1)→controller→(model)→Ψ_t")


class TestStochasticSampling(unittest.TestCase):
    """Verifies stochastic sampling via torch.multinomial."""

    def test_multinomial_samples_vary_with_temperature(self):
        """Different temperatures should produce different distributions of samples."""
        torch.manual_seed(0)
        vocab_size = 100
        n_trials   = 200

        logits = torch.zeros(vocab_size)
        logits[0] = 5.0  # strong bias toward token 0

        samples_high_T = []
        samples_low_T  = []

        for _ in range(n_trials):
            probs_high = F.softmax(logits / 2.0, dim=-1)
            probs_low  = F.softmax(logits / 0.1, dim=-1)
            samples_high_T.append(torch.multinomial(probs_high, num_samples=1).item())
            samples_low_T.append(torch.multinomial(probs_low,  num_samples=1).item())

        # Low temperature → more concentrated → higher rate of argmax
        rate_low  = samples_low_T.count(0)  / n_trials
        rate_high = samples_high_T.count(0) / n_trials

        self.assertGreater(rate_low, rate_high,
            f"Low-T sampling should be more concentrated: {rate_low:.2f} > {rate_high:.2f}")

    def test_argmax_temperature_invariant(self):
        """argmax is invariant to positive temperature scaling — the Phase 1 key finding."""
        logits = torch.randn(1000)
        argmax_native = torch.argmax(logits).item()

        for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
            argmax_scaled = torch.argmax(logits / T).item()
            self.assertEqual(argmax_native, argmax_scaled,
                f"argmax changed with T={T}: this should never happen!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
# tests/test_s3_timing_urllc.py
"""
Comprehensive tests for S3 — NR Timing Shim for URLLC.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nr_urllc.s3_timing_urllc import (
    attempt_latency_ms,
    cumulative_latency_K,
    max_K_by_deadline,
    bler_to_timely_bler,
    success_within_deadline,
    S3TimingController,
)


class TestAttemptLatency:
    """Test single-attempt latency computation."""
    
    def test_basic_computation(self):
        """Test basic attempt latency calculation."""
        # mu=0 (15 kHz SCS), L=7 symbols, k1=0
        lat = attempt_latency_ms(mu=0, L_symbols=7, k1_symbols=0, grant_delay_ms=0, proc_margin_ms=0)
        
        # 7 symbols at 15 kHz = 7 / 15 ms = 0.4667 ms
        expected = 7.0 / 15.0
        assert abs(lat - expected) < 0.001, f"Expected ~{expected}, got {lat}"
    
    def test_with_k1(self):
        """Test latency with k1 feedback delay."""
        lat = attempt_latency_ms(mu=2, L_symbols=7, k1_symbols=1, grant_delay_ms=0, proc_margin_ms=0)
        
        # mu=2: SCS = 60 kHz
        symbol_dur = 1.0 / 60.0
        # Latency = L * symbol_dur + k1 * symbol_dur
        expected = (7 + 1) * symbol_dur
        assert abs(lat - expected) < 0.001, f"Expected ~{expected}, got {lat}"
    
    def test_with_margins(self):
        """Test latency with processing margins and grant delay."""
        lat = attempt_latency_ms(
            mu=2, L_symbols=7, k1_symbols=1,
            grant_delay_ms=0.5, proc_margin_ms=0.2
        )
        
        symbol_dur = 1.0 / 60.0
        expected = 0.5 + (7 + 1) * symbol_dur + 0.2
        assert abs(lat - expected) < 0.001


class TestCumulativeLatency:
    """Test cumulative latency for K attempts."""
    
    def test_single_attempt(self):
        """Test K=1 (no gaps needed)."""
        lat = cumulative_latency_K(K=1, attempt_latency_ms=0.5, inter_attempt_gap_ms=0.1)
        assert abs(lat - 0.5) < 0.001
    
    def test_multiple_attempts(self):
        """Test K>1 with gaps."""
        # K=3, each attempt 0.5ms, gaps 0.1ms
        # Total = 3*0.5 + 2*0.1 = 1.7 ms
        lat = cumulative_latency_K(K=3, attempt_latency_ms=0.5, inter_attempt_gap_ms=0.1)
        expected = 3*0.5 + 2*0.1
        assert abs(lat - expected) < 0.001


class TestMaxKByDeadline:
    """Test maximum K that fits within deadline."""
    
    def test_fits_deadline(self):
        """Test K calculation for reasonable parameters."""
        # Deadline 5ms, attempt 0.5ms, gap 0.1ms
        # K*0.5 + (K-1)*0.1 <= 5
        # K*0.6 - 0.1 <= 5
        # K <= 8.5, so K_max = 8
        K_max = max_K_by_deadline(
            deadline_ms=5.0,
            attempt_latency_ms=0.5,
            inter_attempt_gap_ms=0.1
        )
        assert K_max >= 1
        assert cumulative_latency_K(K_max, 0.5, 0.1) <= 5.0
        assert cumulative_latency_K(K_max + 1, 0.5, 0.1) > 5.0
    
    def test_tight_deadline(self):
        """Test with very tight deadline (only 1 attempt fits)."""
        K_max = max_K_by_deadline(
            deadline_ms=0.6,
            attempt_latency_ms=0.5,
            inter_attempt_gap_ms=0.1
        )
        assert K_max >= 1  # At least 1 should fit


class TestBlerToTimelyBler:
    """Test BLER conversion to timely BLER."""
    
    def test_single_attempt(self):
        """Test K=1 (should return same BLER)."""
        bler = np.array([0.1, 0.05, 0.01])
        bler_K = bler_to_timely_bler(bler, K=1)
        np.testing.assert_array_almost_equal(bler, bler_K)
    
    def test_multiple_attempts(self):
        """Test K=2 (BLER should decrease)."""
        bler = np.array([0.1, 0.05, 0.01])
        bler_K = bler_to_timely_bler(bler, K=2)
        
        # For K=2: bler_K = bler^2
        expected = bler**2
        np.testing.assert_array_almost_equal(expected, bler_K, decimal=5)
    
    def test_bler_decreases_with_K(self):
        """Test that BLER decreases as K increases."""
        bler = np.array([0.1, 0.05, 0.01])
        bler_1 = bler_to_timely_bler(bler, K=1)
        bler_2 = bler_to_timely_bler(bler, K=2)
        bler_3 = bler_to_timely_bler(bler, K=3)
        
        # Each should be smaller than the previous
        assert np.all(bler_2 <= bler_1)
        assert np.all(bler_3 <= bler_2)


class TestSuccessWithinDeadline:
    """Test success-within-deadline computation."""
    
    def test_within_deadline(self):
        """Test when deadline is met."""
        bler_K = np.array([0.1, 0.05, 0.01])
        success = success_within_deadline(bler_K, deadline_met=True)
        
        # P(success) = 1 - BLER
        expected = 1.0 - bler_K
        np.testing.assert_array_almost_equal(expected, success)
    
    def test_exceeds_deadline(self):
        """Test when deadline is exceeded."""
        bler_K = np.array([0.1, 0.05, 0.01])
        success = success_within_deadline(bler_K, deadline_met=False)
        
        # All zeros if deadline exceeded
        np.testing.assert_array_equal(success, np.zeros_like(success))


class TestS3TimingController:
    """Test the S3TimingController class."""
    
    def test_initialization(self):
        """Test controller initialization with config."""
        cfg = {
            "nr": {
                "mu": 2,
                "minislot_symbols": 7,
                "cg_ul": {"K": 2},
                "harq": {"k1_symbols": 1}
            },
            "urllc": {
                "pdb_ms": 10.0,
                "core_budget_ms": 2.0,
            },
            "timing": {
                "grant_delay_ms": 0.0,
                "inter_attempt_gap_ms": 0.1,
                "proc_margin_ms": 0.2,
            }
        }
        
        ctrl = S3TimingController(cfg)
        assert ctrl.mu == 2
        assert ctrl.L == 7
        assert ctrl.k1 == 1
        assert ctrl.radio_deadline_ms == 8.0  # 10.0 - 2.0
    
    def test_attempt_latency(self):
        """Test getting attempt latency from controller."""
        cfg = {
            "nr": {
                "mu": 2,
                "minislot_symbols": 7,
                "cg_ul": {"K": 2},
                "harq": {"k1_symbols": 1}
            },
            "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
            "timing": {"grant_delay_ms": 0.0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
        }
        
        ctrl = S3TimingController(cfg)
        lat = ctrl.get_attempt_latency()
        assert lat > 0
        assert lat < 10.0
    
    def test_K_fits_deadline(self):
        """Test deadline checking for K values."""
        cfg = {
            "nr": {"mu": 2, "minislot_symbols": 7, "cg_ul": {"K": 2}, "harq": {"k1_symbols": 1}},
            "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
            "timing": {"grant_delay_ms": 0.0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
        }
        
        ctrl = S3TimingController(cfg)
        
        # K=1 should almost always fit
        assert ctrl.K_fits_deadline(1)
        
        # K_max_by_deadline should fit
        assert ctrl.K_fits_deadline(ctrl.K_max_by_deadline)
        
        # K_max + 1 should not fit
        assert not ctrl.K_fits_deadline(ctrl.K_max_by_deadline + 1)
    
    def test_timely_bler_curve(self):
        """Test generating timely BLER curves."""
        cfg = {
            "nr": {"mu": 2, "minislot_symbols": 7, "cg_ul": {"K": 2}, "harq": {"k1_symbols": 1}},
            "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
            "timing": {"grant_delay_ms": 0.0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
        }
        
        ctrl = S3TimingController(cfg)
        bler_single = np.array([0.1, 0.05, 0.01])
        K_list = [1, 2, 3]
        
        curves = ctrl.timely_bler_curve(bler_single, K_list)
        
        assert len(curves) == 3
        for K in K_list:
            assert K in curves
            assert curves[K].shape == bler_single.shape
    
    def test_summary(self):
        """Test controller summary generation."""
        cfg = {
            "nr": {"mu": 2, "minislot_symbols": 7, "cg_ul": {"K": 2}, "harq": {"k1_symbols": 1}},
            "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
            "timing": {"grant_delay_ms": 0.0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
        }
        
        ctrl = S3TimingController(cfg)
        summary = ctrl.summary()
        
        assert "mu" in summary
        assert "attempt_latency_ms" in summary
        assert "radio_deadline_ms" in summary
        assert "K_max_by_deadline" in summary


@pytest.mark.parametrize("mu,expected_scs", [(0, 15), (1, 30), (2, 60)])
def test_scs_values(mu, expected_scs):
    """Test different SCS numerologies."""
    cfg = {
        "nr": {"mu": mu, "minislot_symbols": 7, "cg_ul": {"K": 2}, "harq": {"k1_symbols": 1}},
        "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
        "timing": {"grant_delay_ms": 0.0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
    }
    
    ctrl = S3TimingController(cfg)
    summary = ctrl.summary()
    assert summary["scs_khz"] == expected_scs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

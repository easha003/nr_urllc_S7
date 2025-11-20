# nr_urllc/s3_timing_urllc.py
"""
S3 — NR Timing Shim for URLLC Reliability & Latency.

Computes per-attempt and cumulative latencies, tracks success-by-deadline,
and generates timely BLER curves across K repetitions.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


def attempt_latency_ms(
    mu: int,
    L_symbols: int,
    k1_symbols: int = 0,
    grant_delay_ms: float = 0.0,
    proc_margin_ms: float = 0.2,
) -> float:
    """
    Compute latency for a single transmission attempt.
    
    Args:
        mu: SCS numerology (0=15kHz, 1=30kHz, 2=60kHz, etc.)
        L_symbols: Number of symbols in mini-slot
        k1_symbols: k1 HARQ feedback delay (symbols); converted to ms
        grant_delay_ms: Scheduling delay before transmission (ms)
        proc_margin_ms: Processing margin (DSP, encoding, etc.) (ms)
    
    Returns:
        Total latency for one attempt in ms
    """
    # SCS in kHz
    scs_khz = 15.0 * (2 ** int(mu))
    
    # One symbol duration in ms
    symbol_duration_ms = 1.0 / scs_khz  # e.g., 1/15 ≈ 0.0667 ms at mu=0
    
    # Mini-slot transmission time (L symbols)
    tx_time_ms = L_symbols * symbol_duration_ms
    
    # k1 feedback delay (in symbols) → ms
    k1_delay_ms = k1_symbols * symbol_duration_ms
    
    # Total: grant delay + TX + k1 + processing margin
    total_latency_ms = grant_delay_ms + tx_time_ms + k1_delay_ms + proc_margin_ms
    
    return float(total_latency_ms)


def cumulative_latency_K(
    K: int,
    attempt_latency_ms: float,
    inter_attempt_gap_ms: float = 0.1,
) -> float:
    """
    Compute cumulative latency for K sequential attempts with inter-attempt gaps.
    
    Args:
        K: Number of transmission attempts
        attempt_latency_ms: Latency of one attempt (from attempt_latency_ms())
        inter_attempt_gap_ms: Gap between end of attempt i and start of attempt i+1 (ms)
    
    Returns:
        Total latency for K attempts in ms
    """
    if K <= 0:
        return 0.0
    
    # K attempts + (K-1) gaps between them
    total = K * float(attempt_latency_ms) + (K - 1) * float(inter_attempt_gap_ms)
    return float(total)


def max_K_by_deadline(
    deadline_ms: float,
    attempt_latency_ms: float,
    inter_attempt_gap_ms: float = 0.1,
    min_K: int = 1,
) -> int:
    """
    Determine the maximum number of attempts K that fit within the deadline.
    
    Args:
        deadline_ms: Radio deadline (ms)
        attempt_latency_ms: Latency of one attempt
        inter_attempt_gap_ms: Gap between attempts
        min_K: Minimum K to return (at least 1)
    
    Returns:
        Maximum K such that cumulative_latency_K(K, ...) <= deadline_ms
    """
    deadline = float(deadline_ms)
    attempt_lat = float(attempt_latency_ms)
    gap = float(inter_attempt_gap_ms)
    
    if attempt_lat <= 0 or deadline <= 0:
        return int(max(min_K, 1))
    
    # Solve: K * attempt_lat + (K-1) * gap <= deadline
    # => K * (attempt_lat + gap) - gap <= deadline
    # => K <= (deadline + gap) / (attempt_lat + gap)
    K_max = (deadline + gap) / (attempt_lat + gap)
    return int(max(min_K, math.floor(K_max)))


def bler_to_timely_bler(
    bler: np.ndarray,
    K: int,
    policy: str = "independent",
) -> np.ndarray:
    """
    Convert single-attempt BLER to K-attempt timely BLER.
    
    Two policies:
      - "independent": BLER_K = 1 - (1 - BLER)^K (K independent tries; one success)
      - "union_bound": BLER_K = 1 - Product[(1 - BLER_i)] where BLER_i estimated per SNR
    
    Args:
        bler: Single-attempt BLER per SNR (array)
        K: Number of independent attempts
        policy: "independent" or "union_bound" (currently both use independent model)
    
    Returns:
        K-attempt BLER (probability of failure across all K tries)
    """
    bler = np.asarray(bler, dtype=float)
    K = int(K)
    
    if K <= 0:
        return np.ones_like(bler)
    if K == 1:
        return bler
    
    # Independent attempts: at least one success required
    # P(fail all K) = (BLER)^K
    bler_K = np.power(np.clip(bler, 0.0, 1.0), K)
    
    return bler_K.astype(float)


def success_within_deadline(
    bler_K: np.ndarray,
    deadline_met: bool = True,
) -> np.ndarray:
    """
    Compute probability of success AND within deadline.
    
    Args:
        bler_K: BLER for K attempts (so P(success) = 1 - BLER_K)
        deadline_met: If False, multiply by 0 (simulate deadline miss)
    
    Returns:
        P(success ∧ timely)
    """
    p_success = 1.0 - np.asarray(bler_K, dtype=float)
    if deadline_met:
        return p_success
    else:
        return np.zeros_like(p_success)


class S3TimingController:
    """
    High-level S3 timing controller for a given NR configuration.
    Tracks attempts, deadlines, and produces timely BLER curves.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize from S1/S3 config block.
        
        Expected cfg keys:
          - nr: {mu, minislot_symbols, cg_ul: {K, period_ms}, harq: {k1_symbols}}
          - urllc: {pdb_ms, core_budget_ms, ...} → radio_deadline_ms = pdb - core_budget
          - timing: {grant_delay_ms, inter_attempt_gap_ms, proc_margin_ms}
        """
        self.cfg = cfg
        
        nr = cfg.get("nr", {})
        urllc = cfg.get("urllc", {})
        timing = cfg.get("timing", {})
        
        self.mu = int(nr.get("mu", 2))
        self.L = int(nr.get("minislot_symbols", 7))
        self.K_max = int(nr.get("cg_ul", {}).get("K", 2))
        self.k1 = int(nr.get("harq", {}).get("k1_symbols", 0))
        
        self.pdb_ms = float(urllc.get("pdb_ms", 10.0))
        self.core_budget_ms = float(urllc.get("core_budget_ms", 2.0))
        self.radio_deadline_ms = self.pdb_ms - self.core_budget_ms
        
        self.grant_delay_ms = float(timing.get("grant_delay_ms", 0.0))
        self.inter_attempt_gap_ms = float(timing.get("inter_attempt_gap_ms", 0.1))
        self.proc_margin_ms = float(timing.get("proc_margin_ms", 0.2))
        
        # Compute single-attempt latency
        self.attempt_latency_ms = attempt_latency_ms(
            self.mu, self.L, self.k1, self.grant_delay_ms, self.proc_margin_ms
        )
        
        # Compute K_max that fits in deadline
        self.K_max_by_deadline = max_K_by_deadline(
            self.radio_deadline_ms,
            self.attempt_latency_ms,
            self.inter_attempt_gap_ms,
            min_K=1
        )
        
    def get_attempt_latency(self) -> float:
        """Return single-attempt latency in ms."""
        return self.attempt_latency_ms
    
    def get_cumulative_latency(self, K: int) -> float:
        """Return cumulative latency for K attempts."""
        return cumulative_latency_K(K, self.attempt_latency_ms, self.inter_attempt_gap_ms)
    
    def K_fits_deadline(self, K: int) -> bool:
        """Check if K attempts fit within radio deadline."""
        return self.get_cumulative_latency(K) <= self.radio_deadline_ms
    
    def timely_bler_curve(self, bler_single: np.ndarray, K_list: List[int]) -> Dict[int, np.ndarray]:
        """
        Generate timely BLER curves for a range of K values.
        
        Args:
            bler_single: Single-attempt BLER per SNR (array)
            K_list: List of K values to evaluate
        
        Returns:
            {K: timely_bler_array} for each K
        """
        result = {}
        for K in K_list:
            fits = self.K_fits_deadline(K)
            bler_K = bler_to_timely_bler(bler_single, K)
            p_timely = success_within_deadline(bler_K, deadline_met=fits)
            result[K] = 1.0 - p_timely  # Convert back to BLER form for consistency
        return result
    
    def summary(self) -> Dict:
        """Return a summary of timing parameters."""
        return {
            "mu": self.mu,
            "L_symbols": self.L,
            "k1_symbols": self.k1,
            "scs_khz": 15.0 * (2 ** self.mu),
            "attempt_latency_ms": self.attempt_latency_ms,
            "radio_deadline_ms": self.radio_deadline_ms,
            "K_max_by_deadline": self.K_max_by_deadline,
            "inter_attempt_gap_ms": self.inter_attempt_gap_ms,
            "proc_margin_ms": self.proc_margin_ms,
        }


# Example usage & testing
if __name__ == "__main__":
    # Test config
    test_cfg = {
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
    
    ctrl = S3TimingController(test_cfg)
    print("S3 Timing Summary:")
    print(ctrl.summary())
    
    # Test BLER conversion
    bler_single = np.array([0.1, 0.05, 0.01, 0.001])
    K_list = [1, 2, 3]
    timely_curves = ctrl.timely_bler_curve(bler_single, K_list)
    
    print("\nTimely BLER curves:")
    for K, curve in timely_curves.items():
        print(f"  K={K}: {curve}")

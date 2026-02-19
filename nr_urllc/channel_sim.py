"""
Channel simulator for hybrid RF-VLC system.
Implements realistic fading models with complementary characteristics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np


@dataclass
class ChannelConfig:
    """Configuration for channel simulation."""
    # AR(1) parameters
    phi_rf: float = 0.94
    q_rf: float = 0.3
    m_rf: float = 8.0
    
    phi_vlc: float = 0.88
    q_vlc: float = 0.5
    m_vlc: float = 12.5
    
    # Correlation between RF and VLC
    rho: float = -0.3  # Negative = complementary fading
    
    # VLC blockage model (2-state Markov)
    blockage_prob_enter: float = 0.05   # P(blocked | not blocked)
    blockage_prob_stay: float = 0.7     # P(blocked | blocked)
    blockage_snr: float = -15.0         # SNR when blocked
    
    # Measurement noise
    r_meas: float = 0.5


class ChannelSimulator:
    """
    Simulates correlated RF and VLC channels with realistic blockage.
    
    Key features:
    - Complementary fading (negative correlation)
    - Markov blockage model for VLC
    - Mean-reverting AR(1) dynamics
    """
    
    def __init__(self, cfg: ChannelConfig, seed: int = 123):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        
        # Current state
        self.snr_rf = cfg.m_rf
        self.snr_vlc = cfg.m_vlc
        self.vlc_blocked = False
        
        # History for analysis
        self.history = {
            'snr_rf': [],
            'snr_vlc': [],
            'blocked': []
        }
    
    def step(self) -> Tuple[float, float]:
        """
        Advance channel by one timestep.
        
        Returns:
            (snr_rf, snr_vlc) in dB
        """
        # Generate correlated Gaussian noise
        z_rf = self.rng.normal(0, 1)
        z_vlc = (self.cfg.rho * z_rf + 
                 np.sqrt(1 - self.cfg.rho**2) * self.rng.normal(0, 1))
        
        # AR(1) evolution for RF
        self.snr_rf = (self.cfg.m_rf + 
                       self.cfg.phi_rf * (self.snr_rf - self.cfg.m_rf) +
                       np.sqrt(self.cfg.q_rf) * z_rf)
        
        # AR(1) evolution for VLC (if not blocked)
        self.snr_vlc = (self.cfg.m_vlc + 
                        self.cfg.phi_vlc * (self.snr_vlc - self.cfg.m_vlc) +
                        np.sqrt(self.cfg.q_vlc) * z_vlc)
        
        # Update blockage state (Markov model)
        if self.vlc_blocked:
            # Currently blocked: stay blocked with prob blockage_prob_stay
            self.vlc_blocked = (self.rng.random() < self.cfg.blockage_prob_stay)
        else:
            # Not blocked: enter blockage with prob blockage_prob_enter
            self.vlc_blocked = (self.rng.random() < self.cfg.blockage_prob_enter)
        
        # Apply blockage to VLC SNR
        snr_vlc_actual = self.cfg.blockage_snr if self.vlc_blocked else self.snr_vlc
        
        # Record history
        self.history['snr_rf'].append(self.snr_rf)
        self.history['snr_vlc'].append(snr_vlc_actual)
        self.history['blocked'].append(self.vlc_blocked)
        
        return self.snr_rf, snr_vlc_actual
    
    def measure(self, measure_rf: bool = True, measure_vlc: bool = True) -> Tuple[float | None, float | None]:
        """
        Get noisy measurements of current SNR.
        
        Args:
            measure_rf: Whether to measure RF link
            measure_vlc: Whether to measure VLC link
            
        Returns:
            (snr_rf_meas, snr_vlc_meas) with None for unmeasured links
        """
        snr_rf_actual = self.snr_rf
        snr_vlc_actual = self.cfg.blockage_snr if self.vlc_blocked else self.snr_vlc
        
        snr_rf_meas = None
        if measure_rf:
            snr_rf_meas = snr_rf_actual + self.rng.normal(0, np.sqrt(self.cfg.r_meas))
        
        snr_vlc_meas = None
        if measure_vlc:
            snr_vlc_meas = snr_vlc_actual + self.rng.normal(0, np.sqrt(self.cfg.r_meas))
        
        return snr_rf_meas, snr_vlc_meas
    
    def get_stats(self) -> dict:
        """Get statistics of simulated channels."""
        if not self.history['snr_rf']:
            return {}
        
        # ✅ NEW: Compute predictability metrics
        rf_hist = np.array(self.history['snr_rf'])
        vlc_hist = np.array(self.history['snr_vlc'])
        
        # Lag-1 autocorrelation (measure of predictability)
        rf_autocorr = np.corrcoef(rf_hist[:-1], rf_hist[1:])[0, 1] if len(rf_hist) > 1 else 0
        vlc_autocorr = np.corrcoef(vlc_hist[:-1], vlc_hist[1:])[0, 1] if len(vlc_hist) > 1 else 0
        
        # Count blockage transitions (measure of volatility)
        blocked = np.array(self.history['blocked'])
        transitions = np.sum(np.abs(np.diff(blocked.astype(int))))
        
        return {
            'rf_mean': np.mean(rf_hist),
            'rf_std': np.std(rf_hist),
            'rf_autocorr': float(rf_autocorr),  # ← NEW
            'vlc_mean': np.mean(vlc_hist),
            'vlc_std': np.std(vlc_hist),
            'vlc_autocorr': float(vlc_autocorr),  # ← NEW
            'blockage_rate': np.mean(blocked),
            'blockage_transitions': int(transitions),  # ← NEW
            'correlation': np.corrcoef(rf_hist, vlc_hist)[0, 1]
        }


@dataclass
class ScenarioConfig:
    """Pre-configured channel scenarios for evaluation."""
    name: str
    rf_offset: float = 0.0
    vlc_offset: float = 0.0
    rho: float = -0.3
    blockage_prob_enter: float = 0.05
    blockage_prob_stay: float = 0.7


# Standard scenarios from S7 milestones
SCENARIOS = {
    'rf_good': ScenarioConfig(
        name='RF-Good / VLC-Poor',
        rf_offset=5.0,
        vlc_offset=-5.0,
        rho=-0.3
    ),
    'vlc_good': ScenarioConfig(
        name='VLC-Good / RF-Poor',
        rf_offset=-5.0,
        vlc_offset=5.0,
        rho=-0.3
    ),
    'complementary': ScenarioConfig(
        name='Complementary (Independent)',
        rf_offset=0.0,
        vlc_offset=0.0,
        rho=-0.3  # Negative correlation for diversity
    ),
    'correlated_blockage': ScenarioConfig(
        name='Correlated Blockage',
        rf_offset=-2.0,  # RF slightly degraded when VLC blocked
        vlc_offset=0.0,
        rho=-0.3,
        blockage_prob_enter=0.1,  # More frequent blockage
        blockage_prob_stay=0.8     # Longer blockage duration
    ),
    'balanced': ScenarioConfig(
        name='Balanced',
        rf_offset=0.0,
        vlc_offset=0.0,
        rho=0.0,  # Independent
        blockage_prob_enter=0.05,
        blockage_prob_stay=0.7
    )
}


def create_channel_simulator(scenario: str, seed: int = 123) -> ChannelSimulator:
    """
    Factory function to create channel simulator for a given scenario.
    
    Args:
        scenario: One of 'rf_good', 'vlc_good', 'complementary', 'correlated_blockage', 'balanced'
        seed: Random seed
        
    Returns:
        Configured ChannelSimulator
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(SCENARIOS.keys())}")
    
    sc = SCENARIOS[scenario]
    
    cfg = ChannelConfig(
        m_rf=8.0 + sc.rf_offset,
        m_vlc=12.5 + sc.vlc_offset,
        rho=sc.rho,
        blockage_prob_enter=sc.blockage_prob_enter,
        blockage_prob_stay=sc.blockage_prob_stay
    )
    
    return ChannelSimulator(cfg, seed=seed)

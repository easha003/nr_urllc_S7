"""
Baseline policies for performance comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .adapters import LinkLUT


@dataclass
class BaselinePolicy:
    """Base class for simple baseline policies."""
    K_total: int = 2
    
    def decide(self, snr_rf: float, snr_vlc: float) -> Dict:
        """Make decision given current SNR observations."""
        raise NotImplementedError


class RFOnlyPolicy(BaselinePolicy):
    """Always use RF link with full K budget."""
    
    def decide(self, snr_rf: float, snr_vlc: float) -> Dict:
        return {
            'action': 'RF',
            'K_rf': self.K_total,
            'K_vlc': 0,
            'p_est': 0.0  # Would need LUT to compute
        }


class VLCOnlyPolicy(BaselinePolicy):
    """Always use VLC link with full K budget."""
    
    def decide(self, snr_rf: float, snr_vlc: float) -> Dict:
        return {
            'action': 'VLC',
            'K_rf': 0,
            'K_vlc': self.K_total,
            'p_est': 0.0
        }


class AlwaysDUPPolicy(BaselinePolicy):
    """Always use DUP with split budget."""
    
    def __init__(self, K_total: int = 2, split: tuple = (1, 1)):
        super().__init__(K_total)
        self.split = split
    
    def decide(self, snr_rf: float, snr_vlc: float) -> Dict:
        return {
            'action': 'DUP',
            'K_rf': self.split[0],
            'K_vlc': self.split[1],
            'p_est': 0.0
        }


class OraclePolicy:
    """
    Oracle policy with perfect future knowledge.
    
    This represents an upper bound on performance.
    """
    
    def __init__(self, rf_lut: LinkLUT, vlc_lut: LinkLUT, K_total: int = 2):
        self.rf = rf_lut
        self.vlc = vlc_lut
        self.K_total = K_total
    
    def decide(self, snr_rf: float, snr_vlc: float) -> Dict:
        """Choose best action with perfect SNR knowledge."""
        # Try all options and pick best
        p_rf = self.rf.p_timely(snr_rf, self.K_total)
        p_vlc = self.vlc.p_timely(snr_vlc, self.K_total)
        
        # Try DUP with different splits
        best_p = 0.0
        best_action = None
        
        # Option 1: RF only
        if p_rf > best_p:
            best_p = p_rf
            best_action = {'action': 'RF', 'K_rf': self.K_total, 'K_vlc': 0, 'p_est': p_rf}
        
        # Option 2: VLC only
        if p_vlc > best_p:
            best_p = p_vlc
            best_action = {'action': 'VLC', 'K_rf': 0, 'K_vlc': self.K_total, 'p_est': p_vlc}
        
        # Option 3-N: All possible DUP splits
        for k_rf in range(1, self.K_total):
            k_vlc = self.K_total - k_rf
            p_rf_k = self.rf.p_timely(snr_rf, k_rf)
            p_vlc_k = self.vlc.p_timely(snr_vlc, k_vlc)
            p_dup = 1.0 - (1.0 - p_rf_k) * (1.0 - p_vlc_k)
            
            if p_dup > best_p:
                best_p = p_dup
                best_action = {'action': 'DUP', 'K_rf': k_rf, 'K_vlc': k_vlc, 'p_est': p_dup}
        
        return best_action


class SNRThresholdPolicy:
    """
    Simple threshold-based policy.
    
    Use RF if SNR_RF > threshold_rf AND SNR_VLC < threshold_vlc,
    use VLC if SNR_VLC > threshold_vlc AND SNR_RF < threshold_rf,
    otherwise use DUP.
    """
    
    def __init__(self, rf_lut: LinkLUT, vlc_lut: LinkLUT, 
                 K_total: int = 2,
                 threshold_rf: float = 9.0,
                 threshold_vlc: float = 11.0):
        self.rf = rf_lut
        self.vlc = vlc_lut
        self.K_total = K_total
        self.threshold_rf = threshold_rf
        self.threshold_vlc = threshold_vlc
    
    def decide(self, snr_rf: float, snr_vlc: float) -> Dict:
        """Threshold-based decision."""
        if snr_rf >= self.threshold_rf and snr_vlc < self.threshold_vlc:
            # RF is good, VLC is poor -> use RF
            p_est = self.rf.p_timely(snr_rf, self.K_total)
            return {'action': 'RF', 'K_rf': self.K_total, 'K_vlc': 0, 'p_est': p_est}
        
        elif snr_vlc >= self.threshold_vlc and snr_rf < self.threshold_rf:
            # VLC is good, RF is poor -> use VLC
            p_est = self.vlc.p_timely(snr_vlc, self.K_total)
            return {'action': 'VLC', 'K_rf': 0, 'K_vlc': self.K_total, 'p_est': p_est}
        
        else:
            # Both good or both poor -> use DUP
            k_rf, k_vlc = 1, 1
            p_rf = self.rf.p_timely(snr_rf, k_rf)
            p_vlc = self.vlc.p_timely(snr_vlc, k_vlc)
            p_dup = 1.0 - (1.0 - p_rf) * (1.0 - p_vlc)
            return {'action': 'DUP', 'K_rf': k_rf, 'K_vlc': k_vlc, 'p_est': p_dup}


def create_baseline(name: str, rf_lut: LinkLUT = None, vlc_lut: LinkLUT = None, 
                   K_total: int = 2, **kwargs) -> BaselinePolicy:
    """
    Factory function to create baseline policies.
    
    Args:
        name: One of 'rf_only', 'vlc_only', 'always_dup', 'oracle', 'threshold'
        rf_lut: RF lookup table (required for oracle and threshold)
        vlc_lut: VLC lookup table (required for oracle and threshold)
        K_total: Total attempt budget
        **kwargs: Additional policy-specific arguments
        
    Returns:
        Baseline policy instance
    """
    if name == 'rf_only':
        return RFOnlyPolicy(K_total)
    elif name == 'vlc_only':
        return VLCOnlyPolicy(K_total)
    elif name == 'always_dup':
        split = kwargs.get('split', (1, 1))
        return AlwaysDUPPolicy(K_total, split)
    elif name == 'oracle':
        if rf_lut is None or vlc_lut is None:
            raise ValueError("Oracle policy requires rf_lut and vlc_lut")
        return OraclePolicy(rf_lut, vlc_lut, K_total)
    elif name == 'threshold':
        if rf_lut is None or vlc_lut is None:
            raise ValueError("Threshold policy requires rf_lut and vlc_lut")
        threshold_rf = kwargs.get('threshold_rf', 9.0)
        threshold_vlc = kwargs.get('threshold_vlc', 11.0)
        return SNRThresholdPolicy(rf_lut, vlc_lut, K_total, threshold_rf, threshold_vlc)
    else:
        raise ValueError(f"Unknown baseline: {name}")

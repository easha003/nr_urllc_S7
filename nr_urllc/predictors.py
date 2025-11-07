from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math

@dataclass
class AR1SnrPredictor:
    phi: float = 0.9
    q: float = 1.0
    r: float = 0.5
    mu_rf: float = 10.0
    var_rf: float = 4.0
    mu_vlc: float = 10.0
    var_vlc: float = 4.0

    def update(self, snr_rf_meas: float | None, snr_vlc_meas: float | None):
        for link in ("rf", "vlc"):
            mu = getattr(self, f"mu_{link}")
            var = getattr(self, f"var_{link}")
            mu_pred = self.phi * mu
            var_pred = (self.phi * self.phi) * var + self.q
            meas = snr_rf_meas if link == "rf" else snr_vlc_meas
            if meas is None:
                setattr(self, f"mu_{link}", mu_pred)
                setattr(self, f"var_{link}", var_pred)
                continue
            k = var_pred / (var_pred + self.r)
            mu_upd = mu_pred + k * (meas - mu_pred)
            var_upd = (1.0 - k) * var_pred
            setattr(self, f"mu_{link}", mu_upd)
            setattr(self, f"var_{link}", var_upd)

    def predict_next(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.mu_rf, math.sqrt(max(self.var_rf, 1e-9))), (self.mu_vlc, math.sqrt(max(self.var_vlc, 1e-9)))

# --- PATCH: mean-reverting AR(1) predictor ---
@dataclass
class AR1MeanReverting:
    # State-space & noise
    phi: float = 0.9
    q: float = 1.0
    r: float = 0.5

    # Long-run means (reversion anchors), in dB
    m_rf: float = 8.0
    m_vlc: float = 12.0

    # Posterior state at current step (means & variances), in dB
    mu_rf: float = 10.0
    var_rf: float = 4.0
    mu_vlc: float = 10.0
    var_vlc: float = 4.0

    def update(self, snr_rf_meas: float | None, snr_vlc_meas: float | None) -> None:
        """
        One Kalman-like predict→(optional)update for both links.
        If a measurement is None, we do prediction-only for that link.
        """
        for link in ("rf", "vlc"):
            mu = getattr(self, f"mu_{link}")
            var = getattr(self, f"var_{link}")
            m = self.m_rf if link == "rf" else self.m_vlc

            # Predict step (mean reversion)
            mu_pred = m + self.phi * (mu - m)
            var_pred = (self.phi * self.phi) * var + self.q

            # Measurement (if available)
            meas = snr_rf_meas if link == "rf" else snr_vlc_meas
            if meas is None:
                setattr(self, f"mu_{link}", mu_pred)
                setattr(self, f"var_{link}", var_pred)
                continue

            K = var_pred / (var_pred + self.r)
            mu_upd = mu_pred + K * (meas - mu_pred)
            var_upd = (1.0 - K) * var_pred
            setattr(self, f"mu_{link}", mu_upd)
            setattr(self, f"var_{link}", var_upd)

    def predict_next(self):
        """
        Returns ((mu_rf, sig_rf), (mu_vlc, sig_vlc)) where sig = sqrt(var).
        Matches the API your controller already uses.
        """
        return (
            (self.mu_rf,  math.sqrt(max(self.var_rf,  1e-9))),
            (self.mu_vlc, math.sqrt(max(self.var_vlc, 1e-9))),
        )
# --- END PATCH ---


# --- PATCH: Per-link mean-reverting AR(1) predictor ---
@dataclass
class AR1MeanRevertingPerLink:
    """
    Mean-reverting AR(1) with different dynamics per link.
    RF: Slow fading, low noise (stable)
    VLC: Fast fading, high noise (volatile)
    """
    # RF state-space & noise
    phi_rf: float = 0.94      # Slower fading
    q_rf: float = 0.3         # Less process noise
    m_rf: float = 8.0         # Lower mean SNR
    
    # VLC state-space & noise  
    phi_vlc: float = 0.82     # Faster fading
    q_vlc: float = 1.2        # More process noise
    m_vlc: float = 12.0       # Higher mean SNR
    
    # Shared measurement noise
    r: float = 0.5
    
    # Posterior state (means & variances), in dB
    mu_rf: float = 8.0
    var_rf: float = 4.0
    mu_vlc: float = 12.0
    var_vlc: float = 4.0

    def update(self, snr_rf_meas: float | None, snr_vlc_meas: float | None) -> None:
        """
        Kalman-like predict→update for both links with per-link dynamics.
        """
        # RF update
        mu_rf_pred = self.m_rf + self.phi_rf * (self.mu_rf - self.m_rf)
        var_rf_pred = (self.phi_rf ** 2) * self.var_rf + self.q_rf
        
        if snr_rf_meas is not None:
            K_rf = var_rf_pred / (var_rf_pred + self.r)
            self.mu_rf = mu_rf_pred + K_rf * (snr_rf_meas - mu_rf_pred)
            self.var_rf = (1.0 - K_rf) * var_rf_pred
        else:
            self.mu_rf = mu_rf_pred
            self.var_rf = var_rf_pred
        
        # VLC update (different dynamics)
        mu_vlc_pred = self.m_vlc + self.phi_vlc * (self.mu_vlc - self.m_vlc)
        var_vlc_pred = (self.phi_vlc ** 2) * self.var_vlc + self.q_vlc
        
        if snr_vlc_meas is not None:
            K_vlc = var_vlc_pred / (var_vlc_pred + self.r)
            self.mu_vlc = mu_vlc_pred + K_vlc * (snr_vlc_meas - mu_vlc_pred)
            self.var_vlc = (1.0 - K_vlc) * var_vlc_pred
        else:
            self.mu_vlc = mu_vlc_pred
            self.var_vlc = var_vlc_pred

    def predict_next(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns ((mu_rf, sig_rf), (mu_vlc, sig_vlc)) where sig = sqrt(var).
        """
        return (
            (self.mu_rf, math.sqrt(max(self.var_rf, 1e-9))),
            (self.mu_vlc, math.sqrt(max(self.var_vlc, 1e-9))),
        )
# --- END PATCH ---


# --- PATCH: predictor factory ---
def make_predictor(cfg):
    """
    Factory function to create predictor from config.
    Supports: ar1, ar1_mean_revert, ar1_per_link
    """
    px = (cfg.get("predictor") or {})
    model = str(px.get("model", "ar1")).lower()

    # Per-link mean-reverting model (BEST for hybrid systems)
    if model in ("ar1_per_link", "ar1_perlink", "ar1pl"):
        rf_cfg = px.get("rf", {})
        vlc_cfg = px.get("vlc", {})
        
        return AR1MeanRevertingPerLink(
            # RF parameters (slow, stable)
            phi_rf=float(rf_cfg.get("phi", 0.94)),
            q_rf=float(rf_cfg.get("q", 0.3)),
            m_rf=float(rf_cfg.get("m", 8.0)),
            mu_rf=float(px.get("mu_rf", 8.0)),
            var_rf=float(px.get("var_rf", 4.0)),
            
            # VLC parameters (fast, volatile)
            phi_vlc=float(vlc_cfg.get("phi", 0.82)),
            q_vlc=float(vlc_cfg.get("q", 1.2)),
            m_vlc=float(vlc_cfg.get("m", 12.0)),
            mu_vlc=float(px.get("mu_vlc", 12.0)),
            var_vlc=float(px.get("var_vlc", 4.0)),
            
            # Shared measurement noise
            r=float(px.get("r", 0.5)),
        )
    
    # Unified mean-reverting model (same dynamics for both links)
    if model in ("ar1_mean_revert", "ar1mr", "ar1_mr"):
        revert = px.get("revert_to") or {}
        return AR1MeanReverting(
            phi=float(px.get("phi", 0.92)),
            q=float(px.get("q", 0.5)),
            r=float(px.get("r", 0.5)),
            m_rf=float(revert.get("rf", px.get("mu_rf", 10.0))),
            m_vlc=float(revert.get("vlc", px.get("mu_vlc", 10.0))),
            mu_rf=float(px.get("mu_rf", 10.0)),
            var_rf=float(px.get("var_rf", 4.0)),
            mu_vlc=float(px.get("mu_vlc", 10.0)),
            var_vlc=float(px.get("var_vlc", 4.0)),
        )

    # Fallback: original drifting AR1 (deprecated, kept for compatibility)
    try:
        return AR1SnrPredictor(
            phi=float(px.get("phi", 0.92)),
            q=float(px.get("q", 0.8)),
            r=float(px.get("r", 0.5)),
            mu_rf=float(px.get("mu_rf", 10.0)),
            var_rf=float(px.get("var_rf", 4.0)),
            mu_vlc=float(px.get("mu_vlc", 10.0)),
            var_vlc=float(px.get("var_vlc", 4.0)),
        )
    except TypeError:
        return AR1SnrPredictor(
            phi=float(px.get("phi", 0.92)),
            q=float(px.get("q", 0.8)),
            r=float(px.get("r", 0.5)),
        )
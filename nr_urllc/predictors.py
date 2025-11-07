
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


# --- PATCH: predictor factory ---
# --- BEGIN PATCH: predictor factory ---
# nr_urllc/predictors.py
def make_predictor(cfg):
    px = (cfg.get("predictor") or {})
    model = str(px.get("model", "ar1")).lower()

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

    # Fallback: keep using your original AR1SnrPredictor
    try:
        # If your AR1SnrPredictor accepts these fields, pass them through
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
        # Otherwise, call it with the minimal (original) signature
        return AR1SnrPredictor(
            phi=float(px.get("phi", 0.92)),
            q=float(px.get("q", 0.8)),
            r=float(px.get("r", 0.5)),
        )



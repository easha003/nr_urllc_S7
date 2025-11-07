
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from .adapters import LinkLUT
from .predictors import AR1SnrPredictor
import random


PolicyName = Literal["rf_only", "vlc_only", "best_link", "conditional_dup"]

@dataclass
class PolicyConfig:
    name: PolicyName = "best_link"
    K_total: int = 2
    p_gate: float = 0.97
    dup_split: Tuple[int, int] = (1, 1)
    allow_switch: bool = True
    hysteresis_margin_prob: float = 0.05
    epsilon: float = 0.0

@dataclass
class ControllerState:
    last_choice: Optional[str] = None

class HybridController:
    def __init__(self, rf_lut: LinkLUT, vlc_lut: LinkLUT, policy: PolicyConfig,
                 predictor: AR1SnrPredictor | None = None, conf_k: float = 1.0):
        self.rf = rf_lut
        self.vlc = vlc_lut
        self.cfg = policy
        self.state = ControllerState()
        self.pred = predictor
        self.conf_k = float(conf_k)

    def _hysteresis_pick(self, p_rf: float, p_vl: float, prefer: str) -> str:
        if not self.cfg.allow_switch or self.state.last_choice is None:
            self.state.last_choice = prefer
            return prefer
        if self.state.last_choice != prefer:
            margin = abs(p_rf - p_vl)
            if margin < self.cfg.hysteresis_margin_prob:
                prefer = self.state.last_choice
        self.state.last_choice = prefer
        return prefer

    def _score_dup(self, snr_rf: float, snr_vl: float, split: Tuple[int,int]) -> float:
        k_rf, k_vl = split
        p_rf = self.rf.p_timely(snr_rf, k_rf) if k_rf > 0 else 0.0
        p_vl = self.vlc.p_timely(snr_vl, k_vl) if k_vl > 0 else 0.0
        return 1.0 - (1.0 - p_rf) * (1.0 - p_vl)

    def decide_early(self):
        if self.pred is None:
            raise RuntimeError("decide_early called but no predictor provided.")
        K = self.cfg.K_total
        (mu_rf, sig_rf), (mu_vl, sig_vl) = self.pred.predict_next()
        snr_rf_eff = float(mu_rf - self.conf_k * sig_rf)
        snr_vl_eff = float(mu_vl - self.conf_k * sig_vl)
        name = self.cfg.name
        if name == "rf_only":
            return {"action":"RF","K_rf":K,"K_vlc":0,"p_est": self.rf.p_timely(snr_rf_eff, K)}
        if name == "vlc_only":
            return {"action":"VLC","K_rf":0,"K_vlc":K,"p_est": self.vlc.p_timely(snr_vl_eff, K)}
        p_rf = self.rf.p_timely(snr_rf_eff, K)
        p_vl = self.vlc.p_timely(snr_vl_eff, K)
        if name == "best_link":
            prefer = self._hysteresis_pick(p_rf, p_vl, "RF" if p_rf >= p_vl else "VLC")
            if self.cfg.epsilon > 0.0 and prefer in ("RF","VLC"):
                if random.random() < self.cfg.epsilon:
                    prefer = "VLC" if prefer == "RF" else "RF"

            return {"action":prefer,
                    "K_rf": K if prefer=="RF" else 0,
                    "K_vlc": K if prefer=="VLC" else 0,
                    "p_est": p_rf if prefer=="RF" else p_vl}

        if name == "conditional_dup":
            k_rf, k_vl = self.cfg.dup_split
            p_rf_k = self.rf.p_timely(snr_rf_eff, k_rf) if k_rf > 0 else 0.0
            p_vl_k = self.vlc.p_timely(snr_vl_eff, k_vl) if k_vl > 0 else 0.0
            p_dup = 1.0 - (1.0 - p_rf_k) * (1.0 - p_vl_k)
            if (p_dup >= self.cfg.p_gate) or (p_rf >= self.cfg.p_gate and p_vl >= self.cfg.p_gate):
                self.state.last_choice = "DUP"
                return {"action":"DUP","K_rf":k_rf,"K_vlc":k_vl,"p_est":p_dup}
            prefer = self._hysteresis_pick(p_rf, p_vl, "RF" if p_rf >= p_vl else "VLC")
            if self.cfg.epsilon > 0.0 and prefer in ("RF","VLC"):
                if random.random() < self.cfg.epsilon:
                    prefer = "VLC" if prefer == "RF" else "RF"

            return {"action":prefer,
                    "K_rf": K if prefer=="RF" else 0,
                    "K_vlc": K if prefer=="VLC" else 0,
                    "p_est": p_rf if prefer=="RF" else p_vl}

        raise ValueError(f"Unknown policy: {name}")

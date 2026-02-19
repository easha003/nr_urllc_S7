
# nr_urllc/nr_timing.py  (S1 meta)
from __future__ import annotations
from typing import Dict, Any

def scs_khz(mu: int) -> float:
    return 15.0 * (2 ** int(mu))

def minislot_tti_ms(mu: int, L_symbols: int, cp: float) -> float:
    # TTI(ms) ~= L*(1+cp)/Deltaf_kHz
    return float(L_symbols) * (1.0 + float(cp)) / scs_khz(int(mu))

def _nr_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    nr = dict(cfg.get("nr", {}))
    ofdm = dict(cfg.get("ofdm", {}))
    mu   = int(nr.get("mu", 2))
    L    = int(nr.get("minislot_symbols", ofdm.get("minislot_symbols", 7)))
    cp   = float(ofdm.get("cp", 0.125))
    cg   = dict(nr.get("cg_ul", {}))
    harq = dict(nr.get("harq", {}))
    return {
        "mu": mu, "L": L, "cp": cp,
        "period_ms": float(cg.get("period_ms", 1.0)),
        "K": int(cg.get("K", 2)),
        "early_stop": bool(cg.get("early_stop", True)),
        "harq_enabled": bool(harq.get("enabled", False)),
        "k1_symbols": int(harq.get("k1_symbols", 0)),
        "k2_symbols": int(harq.get("k2_symbols", 0)),
    }

def build_step1_meta(cfg: Dict[str, Any], urllc_block: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = _nr_defaults(cfg)
    tti = minislot_tti_ms(p["mu"], p["L"], p["cp"])
    max_reps_in_deadline = None
    span_fits_deadline   = None
    k_span_ms            = p["K"] * tti
    if urllc_block:
        deadline = float(urllc_block.get("radio_deadline_ms", 0.0))
        max_reps_in_deadline = int(max(1, int(deadline // tti)))
        span_fits_deadline = (k_span_ms <= deadline)
    return {
        "mu": p["mu"],
        "scs_khz": scs_khz(p["mu"]),
        "minislot_symbols": p["L"],
        "cp": p["cp"],
        "tti_ms": tti,
        "cg_ul": {
            "period_ms": p["period_ms"],
            "K": p["K"],
            "early_stop": p["early_stop"],
            "K_span_ms": k_span_ms,
            "max_reps_in_deadline": max_reps_in_deadline,
            "span_fits_deadline": span_fits_deadline,
        },
        "harq": {
            "enabled": p["harq_enabled"],
            "k1_symbols": p["k1_symbols"],
            "k2_symbols": p["k2_symbols"],
        }
    }

def attach_step1_meta(result: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    meta = result.setdefault("meta", {})
    urllc_block = meta.get("urllc", None)
    meta["nr_step1"] = build_step1_meta(cfg, urllc_block)

# Compatibility alias so other code can call tti_ms(...)
def tti_ms(mu: int, L: int, cp_fraction: float) -> float:
    return minislot_tti_ms(mu, L, cp_fraction)

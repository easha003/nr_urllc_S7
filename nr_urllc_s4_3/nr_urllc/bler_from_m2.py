"""
BLER from M2 BER curve (single-run analytic mapping).

We estimate TB error probability per SNR from BER(SNR) measured in M2,
avoiding a second PHY loop. Two methods:
- "uncoded": pessimistic upper bound BLER = 1 - (1 - BER)^{TB_bits}
- "coded_approx": apply an effective coding transform p_b_res = scale * BER^{d_eff} first

Returned result matches plotting schema:
{ "snr_db": [...], "bler": [...], "meta": {...} }
"""
from typing import Dict, List, Optional
import numpy as np
import math

def _lookup(d: dict, dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def derive_bler_from_m2(m2_result: Dict, cfg: Dict) -> Dict:
    # Extract arrays
    snr = np.asarray(m2_result.get("snr_db", []), dtype=float)
    ber = np.asarray(m2_result.get("ber", []), dtype=float)
    if snr.size == 0 or ber.size == 0 or snr.size != ber.size:
        raise ValueError("M2 result missing or inconsistent 'snr_db'/'ber' arrays for BLER derivation.")

    # Pull BLER settings (with safe defaults)
    bler_cfg = cfg.get("bler", {}) if isinstance(cfg, dict) else {}
    method = str(bler_cfg.get("method", "uncoded")).lower()
    # Allow referencing URLLC payload TB size if present
    payload_bytes = bler_cfg.get("payload_bytes")
    if payload_bytes is None:
        # Try url_lc/url_lc/urllc fields
        payload_bytes = _lookup(cfg, "urllc.tb_payload_bytes", None)
    if payload_bytes is None:
        payload_bytes = _lookup(cfg, "url_lc.payload_bytes", None)
    if payload_bytes is None:
        payload_bytes = _lookup(cfg, "urllc.app_payload_bytes", None)
    if payload_bytes is None:
        payload_bytes = 32  # final fallback

    crc_bits_tb = bler_cfg.get("crc_bits_tb", 24)

    # Compute TB length (bits)
    tb_bits = int(8 * int(payload_bytes) + int(crc_bits_tb))

    # Optional coded approx
    if method == "coded_approx":
        approx = bler_cfg.get("coded_approx", {}) if isinstance(bler_cfg, dict) else {}
        d_eff = float(approx.get("d_eff", 4.0))
        scale = float(approx.get("scale", 1.0))
        # avoid zero underflow
        ber_eff = np.clip(ber, 1e-20, 1.0)
        ber_eff = scale * np.power(ber_eff, d_eff)
        p_b = np.minimum(ber_eff, 1.0)
    else:
        p_b = np.minimum(np.clip(ber, 0.0, 1.0), 1.0)

    # BLER = 1 - (1 - p_b)^(tb_bits), use log1p for stability
    one_minus_pb = np.clip(1.0 - p_b, 1e-20, 1.0)
    bler = 1.0 - np.exp(tb_bits * np.log(one_minus_pb))

    # Build result
    res = {
        "snr_db": snr.tolist(),
        "bler": bler.tolist(),
        "meta": {
            "method": method,
            "tb_bits": int(tb_bits),
            "tb_payload_bytes": int(payload_bytes),
            "crc_bits_tb": int(crc_bits_tb),
            "M": int(cfg.get("tx", {}).get("M", 4)),
            "note": "Derived from M2 BER without re-running PHY"
        }
    }
    return res

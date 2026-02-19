
# nr_urllc/urllc.py  (S1-solid)
from __future__ import annotations
import math, json
from pathlib import Path
from typing import Dict, Any, Optional

# Minimal 5QI subset (3GPP TS 23.501 Table 5.7.4-1/2, simplified for URLLC-like profiles)
_5QI_TABLE = {
    83: {"pdb_ms": 10.0, "per_target": 1e-4, "mdbv_bytes": None},
    84: {"pdb_ms": 10.0, "per_target": 1e-5, "mdbv_bytes": None},
    85: {"pdb_ms": 5.0,  "per_target": 1e-5, "mdbv_bytes": 255},
    86: {"pdb_ms": 5.0,  "per_target": 1e-4, "mdbv_bytes": 1354},
}

def _is_placeholder(x) -> bool:
    if isinstance(x, str) and ("${" in x or "}" in x):
        return True
    return False

def _as_float(name: str, v, *, allow_none: bool=False) -> Optional[float]:
    if v is None:
        if allow_none: return None
        raise ValueError(f"Missing required numeric field: {name}")
    if _is_placeholder(v):
        raise ValueError(f"Field '{name}' uses an unresolved template value: {v!r}")
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Field '{name}' must be numeric (got {v!r})")

def _as_int(name: str, v, *, allow_none: bool=False) -> Optional[int]:
    if v is None:
        if allow_none: return None
        raise ValueError(f"Missing required integer field: {name}")
    if _is_placeholder(v):
        raise ValueError(f"Field '{name}' uses an unresolved template value: {v!r}")
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Field '{name}' must be an integer (got {v!r})")

def _minislot_tti_ms(mu: int, L_symbols: int, cp: float) -> float:
    """
    Very simple TTI estimate for an NR mini-slot:
      scs_kHz = 15 * 2^mu
      TTI(ms) ~= L * (1 + cp) / scs_kHz
    cp is the cyclic-prefix fraction of useful symbol time (e.g., 0.125).
    """
    scs_khz = 15.0 * (2 ** int(mu))
    return float(L_symbols) * (1.0 + float(cp)) / scs_khz

def _nr_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    nr = dict(cfg.get("nr", {}))
    ofdm = dict(cfg.get("ofdm", {}))
    mu   = int(nr.get("mu", 2))
    L    = int(nr.get("minislot_symbols", ofdm.get("minislot_symbols", 7)))
    cp   = float(ofdm.get("cp", 0.125))
    cg_ul = dict(nr.get("cg_ul", {}))
    period_ms = float(cg_ul.get("period_ms", 1.0))
    K = int(cg_ul.get("K", 2))
    early_stop = bool(cg_ul.get("early_stop", True))
    harq = dict(nr.get("harq", {}))
    harq_enabled = bool(harq.get("enabled", False))
    k1_symbols = int(harq.get("k1_symbols", 0))
    k2_symbols = int(harq.get("k2_symbols", 0))
    return {
        "mu": mu, "L": L, "cp": cp,
        "period_ms": period_ms, "K": K, "early_stop": early_stop,
        "harq_enabled": harq_enabled, "k1_symbols": k1_symbols, "k2_symbols": k2_symbols
    }

def compute_budget(cfg_urllc: Dict[str, Any], *, cfg_all: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """
    Step S1: Compute radio-deadline, number of tries that fit, and per-try BLER target,
    using 5QI or explicit pdb/per plus numerology-derived TTI.
    Safe, standalone, and friendly to unresolved YAML templates.
    """
    # 1) Resolve PDB/PER (from 5QI if present)
    five_qi = cfg_urllc.get("five_qi", None)
    pdb_ms = cfg_urllc.get("pdb_ms", None)
    per_target = cfg_urllc.get("per_target", None)
    mdbv = None

    if five_qi is not None:
        five_qi = _as_int("urllc.five_qi", five_qi)
        entry = _5QI_TABLE.get(five_qi)
        if entry is None and (pdb_ms is None or per_target is None):
            raise ValueError(f"5QI {five_qi} not in built-in table and no pdb_ms/per_target provided.")
        if entry is not None:
            if pdb_ms is None:     pdb_ms = entry["pdb_ms"]
            if per_target is None: per_target = entry["per_target"]
            mdbv = entry["mdbv_bytes"]
    # If 5QI not given, require explicit pdb/per
    pdb_ms = _as_float("urllc.pdb_ms", pdb_ms)
    per_target = _as_float("urllc.per_target", per_target)

    # 2) Core/app payloads & tries (inputs)
    core_budget_ms = _as_float("urllc.core_budget_ms", cfg_urllc.get("core_budget_ms", 2.0))
    app_payload_bytes = _as_int("urllc.app_payload_bytes", cfg_urllc.get("app_payload_bytes", 32))
    tb_payload_bytes  = cfg_urllc.get("tb_payload_bytes", None)
    if tb_payload_bytes is None:
        # simple overhead model: +16 bytes
        tb_payload_bytes = app_payload_bytes + 16
    else:
        tb_payload_bytes = _as_int("urllc.tb_payload_bytes", tb_payload_bytes)
    n_independent_tries_cfg = cfg_urllc.get("n_independent_tries", None)
    n_independent_tries_cfg = _as_int("urllc.n_independent_tries", n_independent_tries_cfg, allow_none=True)

    # 3) Numerology -> TTI and deadline
    nr_params = _nr_defaults(cfg_all or {})
    tti_ms = _minislot_tti_ms(nr_params["mu"], nr_params["L"], nr_params["cp"])
    radio_deadline_ms = pdb_ms - core_budget_ms
    if radio_deadline_ms <= 0:
        raise ValueError(f"Radio deadline is non-positive (pdb_ms={pdb_ms}, core_budget_ms={core_budget_ms}).")

    # 4) Tries that fit + per-try BLER
    max_tries_by_deadline = max(1, int(radio_deadline_ms // tti_ms))  # floor, at least 1
    n_independent_tries = max_tries_by_deadline if n_independent_tries_cfg is None else int(min(n_independent_tries_cfg, max_tries_by_deadline))
    if not (0.0 < per_target < 1.0):
        raise ValueError(f"per_target must be in (0,1). Got {per_target}.")
    bler_per_try_target = 1.0 - (1.0 - per_target) ** (1.0 / float(n_independent_tries))

    # 5) Warnings (MDBV etc.)
    warnings = []
    if mdbv is not None and tb_payload_bytes > mdbv:
        warnings.append(f"tb_payload_bytes={tb_payload_bytes} exceeds MDBV={mdbv} for 5QI={five_qi}.")

    # 6) Assemble block
    block = {
        "five_qi": five_qi,
        "pdb_ms": pdb_ms,
        "per_target": per_target,
        "core_budget_ms": core_budget_ms,
        "radio_deadline_ms": radio_deadline_ms,
        "nr": {
            "mu": nr_params["mu"],
            "minislot_symbols": nr_params["L"],
            "cp": nr_params["cp"],
            "tti_ms": tti_ms,
            "cg_ul": {
                "period_ms": nr_params["period_ms"],
                "K": nr_params["K"],
                "early_stop": nr_params["early_stop"],
            },
            "harq": {
                "enabled": nr_params["harq_enabled"],
                "k1_symbols": nr_params["k1_symbols"],
                "k2_symbols": nr_params["k2_symbols"],
            }
        },
        "payload": {
            "app_payload_bytes": app_payload_bytes,
            "tb_payload_bytes": tb_payload_bytes,
            "mdbv_bytes": mdbv,
        },
        "tries": {
            "max_tries_by_deadline": max_tries_by_deadline,
            "n_independent_tries": n_independent_tries,
            "bler_per_try_target": bler_per_try_target,
        },
        "notes": {
            "formula": "per_try = 1 - (1 - PER_target)^(1 / N_tries) with N_tries <= floor(deadline/TTI)",
            "tti_model": "TTI ~= L*(1+cp)/(15*2^mu) ms (approx.)"
        }
    }
    if warnings:
        block["warnings"] = warnings
    return block

def attach_and_maybe_write(result: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    Compute S1 budget, attach to result['meta']['urllc'], add step1 meta,
    and optionally write artifacts/urllc_budget.json if io.write_json is enabled.
    """
    urllc_cfg = cfg.get("urllc", {}) or {}
    if not urllc_cfg:
        return
    block = compute_budget(urllc_cfg, cfg_all=cfg)
    meta = result.setdefault("meta", {})
    meta["urllc"] = block

    from .nr_timing import build_step1_meta
    meta["nr_step1"] = build_step1_meta(cfg, block)

    io_cfg = cfg.get("io", {})
    if bool(io_cfg.get("write_json", False)):
        path = io_cfg.get("out_urllc_json", "artifacts/urllc_budget.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(block, f, indent=2)
        print("[info] URLLC budget written to", path)

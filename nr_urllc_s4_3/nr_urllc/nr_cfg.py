
from __future__ import annotations
from typing import Dict, Any

def mixin_nr_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of cfg where NR params (minislot, DMRS) are applied to OFDM/pilots.
    - nr.minislot_symbols -> ofdm.minislot_symbols
    - nr.dmrs.{freq_density, offset_sc, seed, power_boost_db, power_mode/budgeted} -> pilots.*
    """
    import copy
    out = copy.deepcopy(cfg) if isinstance(cfg, dict) else dict(cfg)

    nr   = dict(out.get("nr", {}))
    ofdm = out.setdefault("ofdm", {})
    pilots = out.setdefault("pilots", {})

    # Minislot binding
    if "minislot_symbols" in nr:
        ofdm["minislot_symbols"] = int(nr.get("minislot_symbols"))

    # DMRS mapping
    dmrs = nr.get("dmrs", {})
    if dmrs:
        # Map common fields
        if "freq_density" in dmrs:
            pilots["spacing"] = int(dmrs.get("freq_density"))
        if "offset_sc" in dmrs:
            pilots["offset"] = int(dmrs.get("offset_sc"))
        if "seed" in dmrs:
            pilots["seed"] = int(dmrs.get("seed"))
        if "power_boost_db" in dmrs:
            pilots["power_boost_db"] = float(dmrs.get("power_boost_db"))
        # Budgeting: either explicit power_mode or boolean 'budgeted'
        if "power_mode" in dmrs:
            pilots["power_mode"] = str(dmrs.get("power_mode"))
        elif bool(dmrs.get("budgeted", False)):
            pilots["power_mode"] = "budgeted"
        else:
            pilots.setdefault("power_mode", "unconstrained")
    return out

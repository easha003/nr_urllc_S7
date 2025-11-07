#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import yaml

from nr_urllc.adapters import LinkLUT
from nr_urllc.hybrid import HybridController, PolicyConfig
from nr_urllc.predictors import AR1SnrPredictor
from nr_urllc.predictors import make_predictor

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    rf = LinkLUT.from_json(cfg["adapters"]["rf_lut"])
    vlc = LinkLUT.from_json(cfg["adapters"]["vlc_lut"])
    pc = PolicyConfig(
        name=cfg["hybrid"]["policy"],
        K_total=cfg["hybrid"]["K_total"],
        p_gate=cfg["hybrid"].get("p_gate", 0.97),
        dup_split=tuple(cfg["hybrid"].get("dup_split", [1,1])),
        allow_switch=cfg["hybrid"].get("allow_switch", True),
        hysteresis_margin_prob=cfg["hybrid"].get("hysteresis_margin_prob", 0.05),
        epsilon=cfg["hybrid"].get("epsilon", 0.0),
    )
    pred = make_predictor(cfg)
    ctrl = HybridController(rf, vlc, pc, predictor=pred, conf_k=cfg["hybrid"].get("conf_k", 1.0))

    # ✅ FIXED: Define probe before using it
    hy = cfg.get("hybrid", {})
    probe = hy.get("probe", {}) if isinstance(hy.get("probe", {}), dict) else {}
    probe_period = int(probe.get("period", 0))          # 0 = disabled
    probe_on_switch = bool(probe.get("on_switch", False))
    probe_r = float(probe.get("r_probe", pred.r))       # Now probe is defined

    # ✅ ISSUE #6 FIX: Mean-reverting trajectories with VLC blockage
    T = 200
    rng = np.random.default_rng(123)
    
    # Get predictor parameters
    pred_cfg = cfg["predictor"]
    phi = pred_cfg["phi"]
    q = pred_cfg["q"]
    
    # Check if using mean-reverting model
    model = str(pred_cfg.get("model", "ar1")).lower()
    if model in ("ar1_mean_revert", "ar1mr", "ar1_mr"):
        revert = pred_cfg.get("revert_to", {})
        m_rf = float(revert.get("rf", pred_cfg.get("mu_rf", 10.0)))
        m_vlc = float(revert.get("vlc", pred_cfg.get("mu_vlc", 10.0)))
    else:
        m_rf = 0.0  # Original AR1 drifts to zero
        m_vlc = 0.0
    
    # Initialize trajectories
    snr_rf_true = np.zeros(T)
    snr_vlc_true = np.zeros(T)
    snr_rf_true[0] = pred_cfg["mu_rf"]
    snr_vlc_true[0] = pred_cfg["mu_vlc"]
    
    # VLC blockage parameters (Issue #6: channel diversity)
    vlc_blockage_prob = 0.1   # 10% blockage probability per timestep
    vlc_blockage_snr = -15.0  # Deep fade when blocked
    
    # Generate trajectories with mean reversion + VLC blockage
    for t in range(1, T):
        # RF: Normal mean-reverting AR(1)
        snr_rf_true[t] = m_rf + phi * (snr_rf_true[t-1] - m_rf) + rng.normal(0, np.sqrt(q))
        
        # VLC: Mean-reverting AR(1) + Bernoulli blockage
        if rng.random() < vlc_blockage_prob:
            snr_vlc_true[t] = vlc_blockage_snr  # Blocked
        else:
            snr_vlc_true[t] = m_vlc + phi * (snr_vlc_true[t-1] - m_vlc) + rng.normal(0, np.sqrt(q))
    
    last_action = None
    records = []
    
    for t in range(T):
        d = ctrl.decide_early()
        
        # ✅ Check switch BEFORE updating last_action
        switch_happened = (last_action is not None and last_action != d["action"])
        
        # Determine which links are naturally measured by this action
        measured_rf = d["action"] in ("RF", "DUP")
        measured_vlc = d["action"] in ("VLC", "DUP")
        
        # Probing triggers
        need_periodic = (probe_period > 0 and t % probe_period == 0)
        need_switch = probe_on_switch and switch_happened
        
        if need_periodic or need_switch:
            # Probe both links
            meas_rf = float(snr_rf_true[t] + rng.normal(0, np.sqrt(probe_r)))
            meas_vlc = float(snr_vlc_true[t] + rng.normal(0, np.sqrt(probe_r)))
        else:
            # Only measure what we're using (with measurement noise - Issue #5)
            meas_rf = float(snr_rf_true[t] + rng.normal(0, np.sqrt(pred.r))) if measured_rf else None
            meas_vlc = float(snr_vlc_true[t] + rng.normal(0, np.sqrt(pred.r))) if measured_vlc else None
        
        pred.update(meas_rf, meas_vlc)
        last_action = d["action"]  # ✅ Update AFTER checking switch
        
        d["t"] = t
        d["snr_rf_true"] = float(snr_rf_true[t])
        d["snr_vlc_true"] = float(snr_vlc_true[t])
        records.append(d)

    out = Path("artifacts/s6"); out.mkdir(parents=True, exist_ok=True)
    (out / "predecide_trace.json").write_text(json.dumps({"records": records}, indent=2))
    
    # ✅ ADD: Summary statistics
    actions = [r["action"] for r in records]
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
    
    print(f"\n[S6] Simulation Summary:")
    print(f"  Total timesteps: {T}")
    print(f"  RF:   {actions.count('RF'):3d} ({actions.count('RF')/T*100:5.1f}%)")
    print(f"  VLC:  {actions.count('VLC'):3d} ({actions.count('VLC')/T*100:5.1f}%)")
    print(f"  DUP:  {actions.count('DUP'):3d} ({actions.count('DUP')/T*100:5.1f}%)")
    print(f"  Switches: {switches}")
    print(f"\n[S6] wrote {out/'predecide_trace.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
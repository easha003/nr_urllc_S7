
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

    hy = cfg.get("hybrid", {})
    probe = hy.get("probe", {}) if isinstance(hy.get("probe", {}), dict) else {}
    probe_period = int(probe.get("period", 0))          # 0 = disabled
    probe_on_switch = bool(probe.get("on_switch", False))

    # hidden SNR trajectories (for logging)
    T = 200
    rng = np.random.default_rng(123)
    phi = cfg["predictor"]["phi"]; q = cfg["predictor"]["q"]
    snr_rf_true = np.zeros(T); snr_vlc_true = np.zeros(T)
    snr_rf_true[0]  = cfg["predictor"]["mu_rf"]
    snr_vlc_true[0] = cfg["predictor"]["mu_vlc"]
    last_action = None

    for t in range(1, T):
        snr_rf_true[t]  = phi * snr_rf_true[t-1]  + rng.normal(0, np.sqrt(q))
        snr_vlc_true[t] = phi * snr_vlc_true[t-1] + rng.normal(0, np.sqrt(q))

    records = []
    for t in range(T):
        d = ctrl.decide_early()
        switch_happened = (
        last_action is not None and
        d["action"] in ("RF","VLC") and
        last_action in ("RF","VLC") and
        d["action"] != last_action
        )
        last_action = d["action"]

        # updates with sparse measurements on used link(s)
        meas_rf  = float(snr_rf_true[t])  if d["action"] in ("RF","DUP") else None
        meas_vlc = float(snr_vlc_true[t]) if d["action"] in ("VLC","DUP") else None
        
        need_probe = (probe_period and (t % probe_period == 0)) or (probe_on_switch and switch_happened)
        if need_probe:
            if meas_rf is None:
                meas_rf = float(snr_rf_true[t])
            if meas_vlc is None:
                meas_vlc = float(snr_vlc_true[t])

        
        pred.update(meas_rf, meas_vlc)
        d["t"] = t
        d["snr_rf_true"]  = float(snr_rf_true[t])
        d["snr_vlc_true"] = float(snr_vlc_true[t])
        records.append(d)

    out = Path("artifacts/s6"); out.mkdir(parents=True, exist_ok=True)
    (out / "predecide_trace.json").write_text(json.dumps({"records": records}, indent=2))
    print(f"[S6] wrote {out/'predecide_trace.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)

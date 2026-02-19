import argparse, yaml, json, copy
import numpy as np
from nr_urllc import simulate
from nr_urllc import urllc as urllc_mod
from nr_urllc import nr_timing as nr_time_mod
from nr_urllc.sweep import autoramp_ofdm_qpsk_sweep, autoramp_sc_qpsk_sweep
from nr_urllc.sweep import autoramp_ofdm_m2_sweep
from nr_urllc.sweep import bler_ofdm_sweep
from nr_urllc.plots import plot_m2_curves
from nr_urllc.plots import plot_bler_curve
from nr_urllc.urllc import attach_and_maybe_write


def _result_from_non_autoramp(cfg: dict) -> dict:
    """
    Run the simulation in non-autoramp mode and return a uniform result dict.
    Expects simulate.run(cfg) to return a dict with keys like:
      snr_db, ber, evm_percent, mse_H, bits_per_frame (optional)
    """
    # If a composite sim.type was passed (e.g., 'm2_then_bler'), coerce to 'ofdm_m2'
    sim_cfg_local = dict(cfg.get("sim", {}))
    mode_local = str(sim_cfg_local.get("type", "")).lower()
    if mode_local in ("m2_then_bler", "ofdm_m2_then_bler"):
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("sim", {})
        cfg["sim"]["type"] = "ofdm_m2"
        cfg["sim"]["autoramp"] = False

    r = simulate.run(cfg)  # delegate to your existing dispatcher
    sim_cfg = cfg.get("sim", {})

    return {
        "success": True,
        "reps_used": 1,
        "latency_ms": 0.0,
        "crc_ok": True,
        "meta": {
            "mode": sim_cfg.get("type", ""),
            "seed": sim_cfg.get("seed", 0),
            "bits_per_frame": r.get("bits_per_frame", None),
        },
        "snr_db": r.get("snr_db", []),
        # Keep "ber" as the canonical key; if the impl used ber_curve, pick that up too.
        "ber": r.get("ber", r.get("ber_curve", [])),
        "evm_percent": r.get("evm_percent", []),
        "mse_H": r.get("mse_H", []),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--out", default="artifacts/result.json")
    args = p.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply NR→pilots/minislot mixin
    from nr_urllc import nr_cfg as _nr_cfg
    cfg = _nr_cfg.mixin_nr_params(cfg)
    # --------- Resolve ${a.b.c} placeholders (if any) BEFORE any int() casts ---------
    import re as _re

    def _get_by_path(d, dotted):
        cur = d
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    _pattern = _re.compile(r"\$\{([^}]+)\}")
    def _resolve_placeholders(obj, root):
        if isinstance(obj, dict):
            return {k: _resolve_placeholders(v, root) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve_placeholders(v, root) for v in obj]
        if isinstance(obj, str):
            def _sub(m):
                key = m.group(1).strip()
                val = _get_by_path(root, key)
                return str(val) if val is not None else m.group(0)
            return _pattern.sub(_sub, obj)
        return obj

    cfg = _resolve_placeholders(cfg, cfg)

    # IO section needed by multiple branches later
    io_cfg = cfg.get("io", {})


    from nr_urllc.urllc import attach_and_maybe_write as _attach_s1
    _result_holder = locals().get('result', {}) or {}
    if not isinstance(_result_holder, dict):
        _result_holder = {}
    _attach_s1(_result_holder, cfg)          # computes S1 and (optionally) writes artifacts/urllc_budget.json
    locals()['result'] = _result_holder

    sim_cfg = cfg.get("sim", {})            # nested
    tx_cfg  = cfg.get("tx", {})
    ofdm    = cfg.get("ofdm", {})
    ch      = cfg.get("channel", {})
    auto    = cfg.get("autoramp", {})

    mode = str(sim_cfg.get("type", "")).lower()
    use_autoramp = bool(sim_cfg.get("autoramp", False))

    
    # ---------------------------
    # Composite: M2 then BLER
    # ---------------------------
    if mode in ("m2_then_bler", "ofdm_m2_then_bler"):
        # 1) M2 non-autoramp (or autoramp if enabled)
        if use_autoramp:
            # Reuse autoramp M2 path
            res_auto = autoramp_ofdm_m2_sweep(
                cfg,
                target_errs=auto.get("target_errs", 200),
                min_bits=auto.get("min_bits", 50_000),
                max_bits=auto.get("max_bits", 10_000_000),
                growth=auto.get("growth", 2.0),
            )
            m2_result = {
                "success": res_auto.success, "snr_db": res_auto.snr_db, "ber": res_auto.ber_curve,
                "evm_percent": res_auto.evm_curve, "mse_H": res_auto.mse_curve,
                "n_bits_curve": res_auto.n_bits_curve, "n_errs_curve": res_auto.n_errs_curve,
            }
        else:
            m2_result = _result_from_non_autoramp(cfg)
        #2) BLER sweep with same cfg
        #Run full TB+CRC (+FEC if configured) BLER over the same channel
        bler_res = bler_ofdm_sweep(cfg)

        # 3) Write artifacts
        io_m2  = io_cfg.get("m2", {})
        io_blr = io_cfg.get("bler", {})
        out_json_m2  = io_m2.get("out_json",  "artifacts/m2_then_bler/m2_curves.json")
        out_plot_m2  = io_m2.get("out_plot",  "artifacts/m2_then_bler/m2_curves.png")
        out_json_blr = io_blr.get("out_json", "artifacts/m2_then_bler/bler_vs_snr.json")
        out_plot_blr = io_blr.get("out_plot", "artifacts/m2_then_bler/bler_vs_snr.png")

        if bool(io_cfg.get("write_json", True)):
            from pathlib import Path as _P
            _P(out_json_m2).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json_m2, "w") as f: json.dump(m2_result, f, indent=2)
            _P(out_json_blr).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json_blr, "w") as f: json.dump(bler_res, f, indent=2)
            print("JSONs saved to", out_json_m2, "and", out_json_blr)

        if bool(io_cfg.get("plot", True)):
            label=f"TB={bler_res.get('meta',{}).get('tb_payload_bytes','?')}B, M={cfg.get('tx',{}).get('M',4)}",
            plot_m2_curves(m2_result, label=label, title="M2 — BER / EVM / MSE vs SNR",
                           save_path=out_plot_m2, show=bool(io_cfg.get("show_plot", False)))
            plot_bler_curve(bler_res, label=f"TB={bler_res['meta']['tb_payload_bytes']}B, M={bler_res['meta']['M']}",
                            title="S2 — BLER vs SNR", save_path=out_plot_blr, show=bool(io_cfg.get("show_plot", False)))
            print("Plots saved to", out_plot_m2, "and", out_plot_blr)
        return

# ---------------------------
    # AUTORAMP PATHS (unchanged)
    # ---------------------------
    if mode == "ofdm_awgn" and use_autoramp:
        res = autoramp_ofdm_qpsk_sweep(
            snr_db_list=ch["snr_db_list"],
            seed=sim_cfg["seed"],
            M=tx_cfg.get("M", 4),
            nfft=ofdm["nfft"],
            cp=ofdm["cp"],
            n_subcarriers=ofdm["n_subcarriers"],
            minislot_symbols=ofdm["minislot_symbols"],
            target_errs=auto.get("target_errs", 100),
            min_bits=auto.get("min_bits", 20_000),
            max_bits=auto.get("max_bits", 2_000_000),
        )
        result = {
            "success": res.success,
            "reps_used": 1,
            "latency_ms": 0.0,
            "crc_ok": True,
            "meta": getattr(res, "meta", {}),
            "ber_curve": res.ber_curve,
            "n_bits_curve": res.n_bits_curve,
            "n_errs_curve": res.n_errs_curve,
        }

    elif mode == "sc_awgn" and use_autoramp:
        res = autoramp_sc_qpsk_sweep(
            ebn0_db_list=ch["ebn0_db_list"],
            seed=sim_cfg["seed"],
            M=tx_cfg.get("M", 4),
            target_errs=auto.get("target_errs", 100),
            min_bits=auto.get("min_bits", 20_000),
            max_bits=auto.get("max_bits", 2_000_000),
        )
        result = {
            "success": res.success,
            "reps_used": 1,
            "latency_ms": 0.0,
            "crc_ok": True,
            "meta": getattr(res, "meta", {}),
            "ber_curve": res.ber_curve,
            "n_bits_curve": res.n_bits_curve,
            "n_errs_curve": res.n_errs_curve,
        }

    elif mode == "ofdm_m2" and use_autoramp:
        res = autoramp_ofdm_m2_sweep(
            cfg,
            target_errs=auto.get("target_errs", 200),
            min_bits=auto.get("min_bits", 50_000),
            max_bits=auto.get("max_bits", 10_000_000),
            growth=auto.get("growth", 2.0),
        )
        result = {
            "success": res.success,
            "reps_used": 1,
            "latency_ms": 0.0,
            "crc_ok": True,
            "meta": getattr(res, "meta", {}),
            "snr_db": res.snr_db,
            "ber": res.ber_curve,
            "n_bits_curve": res.n_bits_curve,
            "n_errs_curve": res.n_errs_curve,
        }

        # ---- OPTIONAL DIAGNOSTIC ADD-ON (no new files) ----
        # Append EVM/MSE per SNR by running a light, non-autoramp pass per SNR
        # using the same numerology/pilots/equalizer.
        
        import numpy as np  # Add this import at the top if not already there
        
        evm_curve = []  # Initialize the lists BEFORE the loop
        mseH_curve = []
        
        print("\n[EVM/MSE Diagnostic] Running enhanced diagnostic sweeps...")
        
        for snr in result.get("snr_db", []):  # Loop over each SNR point
            # SNR-adaptive bit budget: more bits at high SNR where variance is larger
            if snr >= 16:
                diag_bits = 2_000_000
            elif snr >= 10:
                diag_bits = 1_000_000
            else:
                diag_bits = 500_000

            n_diagnostic_runs = 5  # More runs = smoother curve
            evm_samples = []
            mse_samples = []
            
            print(f"  SNR={snr:.1f} dB: Running {n_diagnostic_runs} diagnostic runs with {diag_bits:,} bits each...")

            for run_idx in range(n_diagnostic_runs):
                cfg2 = copy.deepcopy(cfg)
                cfg2.setdefault("sim", {})["autoramp"] = False
                cfg2.setdefault("channel", {})["snr_db_list"] = [float(snr)]
                cfg2.setdefault("tx", {})["n_bits"] = int(diag_bits)
                cfg2.setdefault("io", {})["write_json"] = False
                cfg2["io"]["plot"] = False
                cfg2["io"]["show_plot"] = False
                
                # IMPORTANT: Use different seed for each run
                base_seed = cfg.get("sim", {}).get("seed", 0)
                cfg2["sim"]["seed"] = base_seed + run_idx + int(snr * 1000)
                
                r2 = simulate.run(cfg2)
                evm_val = r2.get("evm_percent", [float("nan")])
                mse_val = r2.get("mse_H", [float("nan")])
                
                # Handle both list and scalar returns
                if isinstance(evm_val, list):
                    evm_val = evm_val[0] if len(evm_val) > 0 else float("nan")
                if isinstance(mse_val, list):
                    mse_val = mse_val[0] if len(mse_val) > 0 else float("nan")
                
                evm_samples.append(float(evm_val))
                mse_samples.append(float(mse_val))

            # Take median (more robust than mean to outliers)
            evm_median = float(np.median(evm_samples))
            mse_median = float(np.median(mse_samples))
            
            print(f"    → EVM: {evm_median:.2f}% (samples: {[f'{x:.1f}' for x in evm_samples]})")
            print(f"    → MSE: {mse_median:.6f}")
            
            evm_curve.append(evm_median)
            mseH_curve.append(mse_median)

        result["evm_percent"] = evm_curve  # EVM in %
        result["mse_H"] = mseH_curve
        
        print("[EVM/MSE Diagnostic] Complete!\n")

    # ---------------------------------------
    # NON-AUTORAMP PATH (all modes supported)
    # ---------------------------------------
    else:
        # For any mode with autoramp disabled, we delegate to simulate.run(cfg)
        result = _result_from_non_autoramp(cfg)

    
    # Attach URLLC budget if requested
    try:
        attach_and_maybe_write(result, cfg)
    except Exception as e:
        print('[warn] URLLC budget attach failed:', e)

    # --- Persist result JSON (prefer config.io.out_json if enabled) ---
    io_cfg = cfg.get("io", {})
    if bool(io_cfg.get("write_json", False)):
        out_json_path = io_cfg.get("out_json", args.out)
        with open(out_json_path, "w") as f:
            json.dump(result, f, indent=2)
        print("Simulation result saved to", out_json_path)
    else:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("Simulation result saved to", args.out)

    # --- Plot curves (BER / EVM / MSE) to the path in config.io.out_plot ---
    do_plot = bool(io_cfg.get("plot", False))
    if do_plot:
        # Construct a readable label similar to simulate.run(...)
        eq_type  = str(cfg.get("eq", {}).get("type", "zf")).upper()
        ch_model = str(cfg.get("channel", {}).get("model", "tdl")).upper()
        label    = f"M={int(cfg.get('tx',{}).get('M',4))} {ch_model} {eq_type}"
        out_png  = io_cfg.get("out_plot", "artifacts/m2_curves.png")
        plot_m2_curves(result, label=label, title="M2 — BER / EVM / MSE vs SNR",
                       save_path=out_png, show=bool(io_cfg.get("show_plot", False)))
        print("Plot saved to", out_png)


if __name__ == "__main__":
    main()
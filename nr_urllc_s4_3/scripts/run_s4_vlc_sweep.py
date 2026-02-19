#!/usr/bin/env python3
# scripts/run_s4_vlc_sweep.py
"""
S4 VLC Runner: Generate VLC BLER and timely BLER curves using DCO-OFDM over LED channel.

This script parallels run_s3_timely_sweep.py but uses VLC channel instead of RF.
It integrates:
- S4 VLC channel (DCO-OFDM, LED, photodetector)
- S3 timing framework (same mini-slot grid as RF)
- S2 OFDM infrastructure (pilots, MMSE, demod)

Usage:
  # Generate VLC BLER + timely curves
  python scripts/run_s4_vlc_sweep.py --cfg configs/s4_vlc_example.yaml \\
                                      --out artifacts/s4/s4_vlc_results.json

  # Compare with existing RF results
  python scripts/run_s4_vlc_sweep.py --cfg configs/s4_vlc_example.yaml \\
                                      --compare-rf artifacts/s3/s3_timely_bler_results.json
"""

from __future__ import annotations
import argparse
import json
import yaml
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# Adjust import paths if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

# S3 timing (reused for VLC!)
from nr_urllc.s3_timing_urllc import S3TimingController
from nr_urllc.s3_timely_bler_sweep import (
    s3_timely_bler_sweep,
    S3TimelySweepResult,
    plot_timely_bler_curves,
    plot_latency_cdf,
    plot_s3_heatmap,
    plot_timely_bler_curves_combined,
)

# S4 VLC channel
from nr_urllc.s4_vlc_channel import vlc_ofdm_link



def run_vlc_bler_sweep(cfg: Dict) -> Dict[str, Any]:
    """
    True BLER (TB+CRC) vs SNR for VLC using the S4 channel (IM/DD DCO‑OFDM).
    Mirrors nr_urllc.sweep.bler_ofdm_sweep but swaps the RF channel with vlc_ofdm_link.
    Returns a dict with arrays: {snr_db: [...], bler: [...], packets: [...], meta: {...}}.
    """
    import numpy as np
    from nr_urllc.tb import append_crc, check_crc, bytes_to_bits, bits_to_bytes
    from nr_urllc import utils, equalize as eq, pilots as pilots_mod, interp as interp_mod, fec

    sim = cfg.get("sim", {}); tx = cfg.get("tx", {}); ofdm_cfg = cfg.get("ofdm", {})
    pil = cfg.get("pilots", {}); urc = cfg.get("urllc", {}) or cfg.get("url_lc", {})
    vlc_cfg = cfg.get("vlc", {}); s4_sweep_cfg = cfg.get("s4_sweep", {})

    rng_seed = int(sim.get("seed", 0))
    rng = utils.get_rng(rng_seed)
    rng_payload = utils.get_rng(rng_seed + 2000)

    # --- OFDM + modulation params ---
    M = int(tx.get("M", 4)); k = int(np.log2(M))
    nfft = int(ofdm_cfg.get("nfft", 256))
    cp = float(ofdm_cfg.get("cp", 0.125))
    K = int(ofdm_cfg.get("n_subcarriers", 120))
    Lslot = int(ofdm_cfg.get("minislot_symbols", 7))

    # --- Pilots ---
    spacing = int(pil.get("spacing", 4))
    offset  = int(pil.get("offset", 0))
    pseed   = int(pil.get("seed", 0))
    boost   = float(pil.get("power_boost_db", 0.0))
    pmode   = str(pil.get("power_mode", "constrained"))

    # Initial placement to know data energy scaling
    tmp = np.zeros((1, K), dtype=np.complex64)
    _grid1, pilot_mask1, pilot_vals1, data_Es1, pilot_Es1 = pilots_mod.place(
        tmp, spacing, offset=offset, seed=pseed, power_boost_db=boost, power_mode=pmode
    )
    pilots_per_sym = int(pilot_mask1[0].sum()); data_RE_per_sym = K - pilots_per_sym
    if data_RE_per_sym <= 0:
        raise RuntimeError("Pilot pattern leaves no data REs")
    print(f"[S4/VLC] Pilot mode={pmode}, data_Es scale={data_Es1:.4f} ({10*np.log10(data_Es1):.2f} dB)")

    # --- TB size (URLLC payload + CRC) ---
    app_bytes = int(urc.get("app_payload_bytes", urc.get("payload_bytes", 32)) or 32)
    tb_bytes  = int(urc.get("tb_payload_bytes", app_bytes + 16))

    # --- Sweep control ---
    # Prefer explicit s4_sweep.snr_db_range, else fall back to channel.snr_db_list
    snr_range = s4_sweep_cfg.get("snr_db_range", None)
    if snr_range is not None:
        if len(snr_range) == 3:
            snr_start, snr_stop, snr_step = snr_range
            snr_db_list = np.arange(snr_start, snr_stop + snr_step/2, snr_step)
        else:
            snr_db_list = np.array(snr_range, dtype=float)
    else:
        snr_db_list = np.array(cfg.get("channel", {}).get("snr_db_list", [0,2,4,6,8,10]), dtype=float)

    packets_per_snr = int(cfg.get("bler", {}).get("packets_per_snr", 400))
    min_errors      = int(cfg.get("bler", {}).get("min_errors", 20))

    # --- VLC physical parameters ---
    led_bw_mhz     = float(vlc_cfg.get("led_bandwidth_mhz", 20.0))
    dc_bias        = float(vlc_cfg.get("dc_bias", 0.5))
    clipping_ratio = float(vlc_cfg.get("clipping_ratio", 0.95))
    responsivity   = float(vlc_cfg.get("responsivity", 0.5))
    area_cm2       = float(vlc_cfg.get("area_cm2", 1.0))
    sample_rate_hz = float(vlc_cfg.get("sample_rate_hz", 100e6))
    noise_type     = str(vlc_cfg.get("noise_type", "awgn"))

    # --- Outputs ---
    bler_list = []
    sent_list = []

    for snr_db in snr_db_list:
        # Adaptive packet budget if SNR likely yields very low BLER
        base_pps = packets_per_snr
        if bler_list and bler_list[-1] < 1e-4:
            base_pps = max(packets_per_snr, packets_per_snr*10)
        n_fail = 0; n_sent = 0

        while (n_sent < base_pps) and (n_fail < min_errors):
            # 1) Generate a TB
            payload = rng_payload.integers(0, 256, size=tb_bytes, dtype=np.uint8).tobytes()
            tb      = append_crc(payload)
            tb_bits = bytes_to_bits(tb)
            Lbits   = int(tb_bits.size)

            # 2) Encode
            fec_cfg = cfg.get("fec", {})
            code_bits, fec_meta = fec.encode(tb_bits, fec_cfg)
            Lcode = int(code_bits.size)

            # 3) Build an OFDM frame with pilots; grow symbols if needed to fit the codeword
            n_syms = Lslot
            while True:
                base = np.zeros((n_syms, K), dtype=np.complex64)
                _grid, pilot_mask, pilot_vals, data_Es, pilot_Es = pilots_mod.place(
                    base, spacing, offset=offset, seed=pseed, power_boost_db=boost, power_mode=pmode
                )
                data_mask = ~pilot_mask
                capacity_bits = int(np.sum(data_mask)) * k
                if capacity_bits >= Lcode:
                    break
                n_syms += Lslot if Lslot > 1 else 1

            # 4) Map bits to QAM on data REs; pad remainder with zeros
            pad = capacity_bits - Lcode
            bits_tx = np.pad(code_bits, (0, pad), constant_values=0) if pad > 0 else code_bits
            syms_data = utils.mod(bits_tx, M).reshape(-1)
            tx_grid = pilot_vals.astype(np.complex64)
            tx_grid[data_mask] = syms_data[: int(np.sum(data_mask))]

            # 5) VLC channel: frequency grid -> Y_freq (same shape)
            Y_freq, _ = vlc_ofdm_link(
                tx_grid,
                snr_db=float(snr_db),
                nfft=nfft,
                n_subcarriers=K,
                sample_rate_hz=sample_rate_hz,
                led_bandwidth_mhz=led_bw_mhz,
                dc_bias=dc_bias,
                clipping_ratio=clipping_ratio,
                responsivity=responsivity,
                area_cm2=area_cm2,
                noise_type=noise_type,
                rng=rng,
            )

            # 6) Pilot-based LS + interpolation + robust MMSE equalization
            H_p = np.zeros_like(Y_freq, dtype=np.complex64)
            H_p[pilot_mask] = Y_freq[pilot_mask] / (pilot_vals[pilot_mask] + 1e-12)
            H_est = interp_mod.interp_freq_linear(H_p, pilot_mask)
            H_est = interp_mod.smooth_time_triangular(H_est)

            # After H_est and before equalization
            if np.any(pilot_mask):
                noise_res = Y_freq[pilot_mask] - H_est[pilot_mask] * pilot_vals[pilot_mask]
                sigma2 = float(np.mean(np.abs(noise_res)**2)) + 1e-12
            else:
                snr_linear = 10.0 ** (float(snr_db) / 10.0)
                Es_data    = float(np.mean(np.abs(Y_freq[data_mask])**2)) + 1e-12
                sigma2     = Es_data / snr_linear

            Y_eq = eq.equalize_mmse_robust(Y_freq, H_est, sigma2, 1e-12)

            # 7) Extract equalized data symbols and LLRs for exactly Lcode bits
            y_data = Y_eq[data_mask].reshape(-1)
            llr = utils.qam_llr_maxlog(y_data, M, sigma2=sigma2)[:Lcode]

            # 8) Soft decode + CRC
            dec_bits = fec.decode_soft(llr, fec_cfg, meta=fec_meta, info_len=Lbits)
            tb_rx = bits_to_bytes(dec_bits[:Lbits])
            ok = check_crc(tb_rx)
            if not ok:
                n_fail += 1
            n_sent += 1

        bler = n_fail / float(max(n_sent, 1))
        bler_list.append(float(bler))
        sent_list.append(int(n_sent))
        print(f"[S4/VLC] SNR={snr_db:.1f} dB → BLER={bler:.3e} ({n_fail}/{n_sent} TB fails)")

    meta = {
        "mode": "tb_bler_vlc",
        "seed": rng_seed,
        "M": M,
        "nfft": nfft,
        "cp": cp,
        "n_subcarriers": K,
        "minislot_symbols": Lslot,
        "pilot_spacing": spacing,
        "pilot_offset": offset,
        "pilot_seed": pseed,
        "pilot_power_boost_db": boost,
        "pilot_power_mode": pmode,
        "tb_payload_bytes": tb_bytes,
        "packets_per_snr": packets_per_snr,
        "min_errors": min_errors,
        "vlc": {"led_bandwidth_mhz": led_bw_mhz, "dc_bias": dc_bias, "clipping_ratio": clipping_ratio, "responsivity": responsivity, "area_cm2": area_cm2, "noise_type": noise_type}
    }
    return {"success": True, "snr_db": list(map(float, snr_db_list)), "bler": bler_list, "packets": sent_list, "meta": meta}
def run_s4_sweep(
    cfg: Dict,
    bler_results: Dict[float, float],
) -> Tuple[S3TimelySweepResult, S3TimingController]:
    """
    Run S4 timely BLER sweep using VLC BLER + S3 timing.
    
    This is identical to S3 sweep logic because timing is the same!
    """
    print("\n=== S4 VLC Timely BLER Sweep (using S3 timing) ===")
    
    # Initialize S3 timing controller (SAME as RF!)
    timing_ctrl = S3TimingController(cfg)
    
    print("\n=== Timing Controller (Shared RF+VLC) ===")
    summary = timing_ctrl.summary()
    for key, val in summary.items():
        print(f"  {key}: {val}")
    
    # Get K list
    s4_sweep_cfg = cfg.get("s4_sweep", cfg.get("s3_sweep", {}))
    K_list = s4_sweep_cfg.get("K_list", [1, 2, 3])
    
    print(f"\n=== S4 Sweep (K_list={K_list}) ===")
    
    # Run S3 sweep function (reused!)
    s4_result = s3_timely_bler_sweep(bler_results, timing_ctrl, K_list=K_list)
    
    return s4_result, timing_ctrl


def compare_rf_vlc(rf_json: str, vlc_json: str, out_dir: str):
    """Generate comparison plots between RF and VLC."""
    print("\n=== RF vs VLC Comparison ===")
    
    with open(rf_json, "r") as f:
        rf_data = json.load(f)
    with open(vlc_json, "r") as f:
        vlc_data = json.load(f)
    
    # TODO: Create side-by-side comparison plots
    # - BLER vs SNR (RF and VLC on same plot)
    # - Latency CDFs (should be identical)
    # - Performance gap (SNR difference for same BLER)
    
    print(f"[S4] Comparison plots saved to {out_dir}/")
    print("  (Comparison plotting not fully implemented yet)")


def main():
    parser = argparse.ArgumentParser(
        description="S4 — VLC OFDM with DCO-OFDM modulation and LED channel"
    )
    parser.add_argument("--cfg", required=True, help="S4 VLC config YAML file")
    parser.add_argument(
        "--bler-json",
        default=None,
        help="Pre-computed VLC BLER results JSON (skip VLC sweep if provided)"
    )
    parser.add_argument(
        "--skip-bler",
        action="store_true",
        help="Skip VLC BLER sweep, use dummy values (for testing)"
    )
    parser.add_argument(
        "--out",
        default="artifacts/s4/s4_vlc_timely_results.json",
        help="Output JSON file for S4 results"
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/s4",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--compare-rf",
        default=None,
        help="RF results JSON for comparison (optional)"
    )
    args = parser.parse_args()

    # Load S4 config
    print(f"[S4 VLC] Loading config from {args.cfg}")
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Get VLC BLER results
    if args.bler_json:
        print(f"[S4] Loading VLC BLER from {args.bler_json}")
        with open(args.bler_json, "r") as f:
            data = json.load(f)
        if "bler" in data and isinstance(data["bler"], dict):
            bler_results = {float(k): float(v) for k, v in data["bler"].items()}
        elif "snr_db" in data and "bler" in data:
            bler_results = {float(s): float(b) for s, b in zip(data["snr_db"], data["bler"])}
        else:
            raise ValueError(f"Unexpected format in {args.bler_json}")
    elif args.skip_bler:
        print("[S4] Using dummy VLC BLER for testing")
        bler_results = {
            5.0: 0.20,
            10.0: 0.10,
            15.0: 0.05,
            20.0: 0.01,
            25.0: 0.001,
        }
    else:
        # Run VLC BLER sweep
        print("[S4] Running VLC BLER sweep...")
        try:
            bler_results = run_vlc_bler_sweep(cfg)
        except Exception as e:
            print(f"[S4 ERROR] VLC BLER sweep failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    

    # Normalize to a BLER result dict for plotting/saving
    if args.bler_json:
        if isinstance(data, dict) and "snr_db" in data and "bler" in data:
            bler_result_dict = data
        else:
            # Convert mapping → arrays
            snr_vals = sorted(bler_results.keys())
            bler_result_dict = {"success": True, "snr_db": snr_vals, "bler": [bler_results[s] for s in snr_vals], "packets": [], "meta": {"source": "external"}}
    elif args.skip_bler:
        snr_vals = sorted(bler_results.keys())
        bler_result_dict = {"success": True, "snr_db": snr_vals, "bler": [bler_results[s] for s in snr_vals], "packets": [], "meta": {"source": "dummy"}}
    else:
        # run_vlc_bler_sweep returns a full dict already
        bler_result_dict = bler_results

    # Always emit a proper VLC BLER plot + JSON
    from nr_urllc.plots import plot_bler_curve
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_bler_json = Path(args.out_dir) / "vlc_bler_vs_snr.json"
    with open(out_bler_json, "w") as f:
        json.dump(bler_result_dict, f, indent=2)
    plot_bler_curve(bler_result_dict, label=f"VLC M={cfg.get('tx',{}).get('M',4)}", save_path=str(Path(args.out_dir) / "vlc_bler_vs_snr.png"))

    # Build mapping for S3 timing shim
    if isinstance(bler_result_dict.get("snr_db", None), list):
        bler_results = {float(s): float(b) for s, b in zip(bler_result_dict["snr_db"], bler_result_dict["bler"])}

    print(f"\n[S4] VLC BLER points: {len(bler_results)}")

    # Run S4 timely sweep (uses S3 timing!)
    try:
        s4_result, timing_ctrl = run_s4_sweep(cfg, bler_results)
    except Exception as e:
        print(f"[S4 ERROR] Timely sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    s4_result.to_json_file(args.out)
    print(f"\n[S4] Saved results to {args.out}")

    # Generate plots (reuse S3 plotting functions!)
    print(f"[S4] Generating VLC plots in {args.out_dir}...")
    try:
        # Add "vlc_" prefix to distinguish from RF plots
        plot_timely_bler_curves(s4_result, save_dir=args.out_dir)
        # Rename output files to vlc_*
        K_iter = getattr(s4_result, "K_values",
         getattr(s4_result, "K_list",
           sorted({getattr(r, "K", 1) for r in getattr(s4_result, "records", [])}) or [1]
         )
       )
        for k in K_iter:
            old_path = Path(args.out_dir) / f"timely_bler_vs_snr_K{k}.png"
            new_path = Path(args.out_dir) / f"vlc_timely_bler_vs_snr_K{k}.png"
            if old_path.exists():
                old_path.rename(new_path)
                print(f"[S4] Renamed to {new_path.name}")

        plot_timely_bler_curves_combined(s4_result, save_dir=args.out_dir)

        # Then rename the output file to match VLC naming:
        old_combined = Path(args.out_dir) / "timely_bler_vs_snr_combined.png"
        new_combined = Path(args.out_dir) / "vlc_timely_bler_vs_snr_combined.png"
        if old_combined.exists():
            old_combined.rename(new_combined)
            print(f"[S4] Renamed to {new_combined.name}")
        
        plot_latency_cdf(timing_ctrl, save_dir=args.out_dir)
        # Rename latency CDF
        old_cdf = Path(args.out_dir) / "latency_cdf.png"
        new_cdf = Path(args.out_dir) / "vlc_latency_cdf.png"
        if old_cdf.exists():
            old_cdf.rename(new_cdf)
            print(f"[S4] Renamed to {new_cdf.name}")
        
        plot_s3_heatmap(s4_result, timing_ctrl, save_dir=args.out_dir)
        # Rename heatmap
        old_hm = Path(args.out_dir) / "s3_timely_heatmap.png"
        new_hm = Path(args.out_dir) / "s4_vlc_timely_heatmap.png"
        if old_hm.exists():
            old_hm.rename(new_hm)
            print(f"[S4] Renamed to {new_hm.name}")
        
    except Exception as e:
        print(f"[S4 WARNING] Plot generation had issues: {e}")
        import traceback
        traceback.print_exc()

    # Optional: Compare with RF
    if args.compare_rf:
        compare_rf_vlc(args.compare_rf, args.out, args.out_dir)

    print(f"\n[S4 VLC COMPLETE] All outputs in {args.out_dir}/")
    print(f"  - vlc_timely_bler_vs_snr_K*.png (VLC timely BLER curves per K)")
    print(f"  - vlc_latency_cdf.png (latency distribution, same as RF)")
    print(f"  - s4_vlc_timely_heatmap.png (success probability heatmap)")
    print(f"  - s4_vlc_timely_results.json (detailed results)")
    
    print("\n[S4 SUCCESS] VLC sweep complete. Compare with RF results from S3!")


if __name__ == "__main__":
    main()
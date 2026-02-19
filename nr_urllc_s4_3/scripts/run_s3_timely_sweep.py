#!/usr/bin/env python3
# scripts/run_s3_timely_sweep.py
"""
S3 Runner: Integrate S2 BLER results with S3 timing to generate timely BLER curves.

Usage:
  python scripts/run_s3_timely_sweep.py --bler-json artifacts/bler_vs_snr.json \\
                                         --cfg configs/s3_timing_example.yaml \\
                                         --out artifacts/s3/results.json

Or run end-to-end:
  python scripts/run_s3_timely_sweep.py --cfg configs/s3_timing_example.yaml \\
                                         --run-bler \\  # First run S2 BLER
                                         --out artifacts/s3/results.json
"""

from __future__ import annotations
import argparse
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Optional

# Adjust import paths if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from nr_urllc.s3_timing_urllc import S3TimingController
from nr_urllc.s3_timely_bler_sweep import (
    s3_timely_bler_sweep,
    S3TimelySweepResult,
    plot_timely_bler_curves,
    plot_latency_cdf,
    plot_s3_heatmap,
    plot_timely_bler_curves_combined,
)


def load_bler_from_json(path: str | Path) -> Dict[float, float]:
    """Load BLER results from JSON (output of S2)."""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle different possible formats
    if "bler" in data and isinstance(data["bler"], dict):
        # Format: {snr_db: bler_value} (dict)
        return {float(k): float(v) for k, v in data["bler"].items()}
    elif "snr_db" in data and "bler" in data:
        # Format: {snr_db: [...], bler: [...]} (arrays)
        snr_list = data["snr_db"]
        bler_list = data["bler"]
        return {float(snr): float(bler) for snr, bler in zip(snr_list, bler_list)}
    else:
        raise ValueError(f"Unexpected BLER JSON format in {path}")


def run_s3_sweep(
    cfg: Dict,
    bler_results: Dict[float, float],
) -> S3TimelySweepResult:
    """Run S3 sweep with given config and BLER results."""
    
    # Initialize timing controller
    timing_ctrl = S3TimingController(cfg)
    
    print("\n=== S3 Timing Controller ===")
    summary = timing_ctrl.summary()
    for key, val in summary.items():
        print(f"  {key}: {val}")
    
    # Get K list from config, or use defaults
    s3_sweep_cfg = cfg.get("s3_sweep", {})
    K_list = s3_sweep_cfg.get("K_list", [1, 2, 3])
    
    print(f"\n=== S3 Sweep (K_list={K_list}) ===")
    
    # Run S3 sweep
    s3_result = s3_timely_bler_sweep(bler_results, timing_ctrl, K_list=K_list)
    
    return s3_result, timing_ctrl


def main():
    parser = argparse.ArgumentParser(
        description="S3 — NR Timing Shim for URLLC Timely BLER curves"
    )
    parser.add_argument("--cfg", required=True, help="S3 config YAML file")
    parser.add_argument(
        "--bler-json",
        default=None,
        help="S2 BLER results JSON (if not provided, uses config defaults)"
    )
    parser.add_argument(
        "--run-bler",
        action="store_true",
        help="Run S2 BLER sweep first (requires simulate module)"
    )
    parser.add_argument(
        "--out",
        default="artifacts/s3/s3_timely_bler_results.json",
        help="Output JSON file for S3 results"
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/s3",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    # Load S3 config
    print(f"[S3] Loading config from {args.cfg}")
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Optionally run S2 first
    if args.run_bler:
        print("[S3] Running S2 BLER sweep first...")
        try:
            from nr_urllc.sweep import bler_ofdm_sweep
            bler_res_dict = bler_ofdm_sweep(cfg)
            # bler_ofdm_sweep returns {snr_db: [...], bler: [...], ...}
            snr_list = bler_res_dict.get("snr_db", [])
            bler_list = bler_res_dict.get("bler", [])
            bler_results = {float(s): float(b) for s, b in zip(snr_list, bler_list)}
        except Exception as e:
            print(f"[S3 ERROR] Failed to run S2: {e}")
            sys.exit(1)
    elif args.bler_json:
        print(f"[S3] Loading BLER from {args.bler_json}")
        bler_results = load_bler_from_json(args.bler_json)
    else:
        # Use dummy BLER for testing
        print("[S3] Using dummy BLER curve for demo")
        bler_results = {
            0.0: 0.15,
            2.0: 0.10,
            4.0: 0.05,
            6.0: 0.01,
            8.0: 0.001,
        }

    print(f"[S3] BLER points: {len(bler_results)}")

    # Run S3 sweep
    try:
        s3_result, timing_ctrl = run_s3_sweep(cfg, bler_results)
    except Exception as e:
        print(f"[S3 ERROR] Sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    s3_result.to_json_file(args.out)
    print(f"\n[S3] Saved results to {args.out}")

    # Generate plots
    print(f"[S3] Generating plots in {args.out_dir}...")
    try:
        plot_timely_bler_curves(s3_result, save_dir=args.out_dir)
        plot_latency_cdf(timing_ctrl, save_dir=args.out_dir)
        plot_s3_heatmap(s3_result, timing_ctrl, save_dir=args.out_dir)
        plot_timely_bler_curves_combined(s3_result)
    except Exception as e:
        print(f"[S3 WARNING] Plot generation had issues: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[S3 COMPLETE] All outputs in {args.out_dir}/")
    print(f"  - timely_bler_vs_snr_K*.png (timely BLER curves per K)")
    print(f"  - latency_cdf.png (cumulative latency distribution)")
    print(f"  - s3_timely_heatmap.png (success probability heatmap)")
    print(f"  - s3_timely_bler_results.json (detailed results)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick validation script to test S7 evaluation framework.
Runs a mini-evaluation (fewer runs) to verify everything works.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from nr_urllc.channel_sim import create_channel_simulator
from nr_urllc.adapters import LinkLUT
from nr_urllc.baselines import create_baseline
from nr_urllc.metrics import MetricsCalculator


def test_channel_simulator():
    """Test channel simulator."""
    print("Testing channel simulator...")
    
    for scenario_name in ['complementary', 'rf_good', 'correlated_blockage']:
        sim = create_channel_simulator(scenario_name, seed=42)
        
        # Run for 100 steps
        for _ in range(100):
            snr_rf, snr_vlc = sim.step()
        
        stats = sim.get_stats()
        print(f"  {scenario_name:20s}: RF={stats['rf_mean']:.1f}±{stats['rf_std']:.1f} dB, "
              f"VLC={stats['vlc_mean']:.1f}±{stats['vlc_std']:.1f} dB, "
              f"Blockage={stats['blockage_rate']:.1%}")
    
    print("  ✓ Channel simulator working\n")


def test_baselines():
    """Test baseline policies."""
    print("Testing baseline policies...")
    
    # Load LUTs
    rf_lut = LinkLUT.from_json('artifacts/adapters/rf_lut.json')
    vlc_lut = LinkLUT.from_json('artifacts/adapters/vlc_lut.json')
    
    baselines = ['rf_only', 'vlc_only', 'always_dup', 'oracle']
    
    for baseline_name in baselines:
        policy = create_baseline(baseline_name, rf_lut, vlc_lut, K_total=2)
        decision = policy.decide(10.0, 12.0)
        
        assert 'action' in decision
        print(f"  {baseline_name:15s}: {decision['action']}")
    
    print("  ✓ Baselines working\n")


def test_metrics_calculator():
    """Test metrics calculator."""
    print("Testing metrics calculator...")
    
    # Create fake records
    records = []
    for t in range(100):
        records.append({
            't': t,
            'action': 'RF' if t % 2 == 0 else 'VLC',
            'K_rf': 2 if t % 2 == 0 else 0,
            'K_vlc': 0 if t % 2 == 0 else 2,
            'snr_rf_true': 10.0,
            'snr_vlc_true': 12.0,
            'p_est': 0.85
        })
    
    calc = MetricsCalculator(deadline_ms=5.0)
    metrics = calc.compute_metrics(records)
    
    print(f"  Timely Delivery: {metrics.timely_delivery_ratio:.3f}")
    print(f"  Mean Energy: {metrics.mean_energy:.2f}")
    print(f"  RF/VLC/DUP: {metrics.frac_rf:.1%}/{metrics.frac_vlc:.1%}/{metrics.frac_dup:.1%}")
    print("  ✓ Metrics calculator working\n")


def main():
    """Run all tests."""
    print("="*60)
    print("VALIDATING S7 EVALUATION FRAMEWORK")
    print("="*60 + "\n")
    
    try:
        test_channel_simulator()
        test_baselines()
        test_metrics_calculator()
        
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYou can now run the full evaluation:")
        print("  python scripts/run_s7_evaluation.py --cfg configs/s7_evaluation.yaml")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

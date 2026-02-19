#!/usr/bin/env python3
"""
Quick validation script to verify prediction improvements.
Run this BEFORE full S7 evaluation.
"""
from pathlib import Path
import yaml
import numpy as np

from nr_urllc.adapters import LinkLUT
from nr_urllc.hybrid import HybridController, PolicyConfig
from nr_urllc.predictors import make_predictor
from nr_urllc.channel_sim import create_channel_simulator

def test_policy(policy_name: str, policy_cfg: dict, scenario: str = 'complementary', seed: int = 42):
    """Test a single policy configuration."""
    print(f"\nTesting {policy_name} on {scenario}...")
    
    # Load LUTs
    rf_lut = LinkLUT.from_json('artifacts/adapters/rf_lut.json')
    vlc_lut = LinkLUT.from_json('artifacts/adapters/vlc_lut.json')
    
    # Create predictor
    predictor = make_predictor(policy_cfg) if 'predictor' in policy_cfg else None
    
    # Create policy
    pc = PolicyConfig(
        name=policy_cfg['hybrid']['policy'],
        K_total=policy_cfg['hybrid']['K_total'],
        p_gate=policy_cfg['hybrid'].get('p_gate', 0.78),
        dup_split=tuple(policy_cfg['hybrid'].get('dup_split', [1,1])),
        allow_switch=policy_cfg['hybrid'].get('allow_switch', True),
        hysteresis_margin_prob=policy_cfg['hybrid'].get('hysteresis_margin_prob', 0.0),
        epsilon=policy_cfg['hybrid'].get('epsilon', 0.0),
    )
    conf_k = policy_cfg['hybrid'].get('conf_k', 0.8)
    policy = HybridController(rf_lut, vlc_lut, pc, predictor=predictor, conf_k=conf_k)
    
    # Create channel
    channel = create_channel_simulator(scenario, seed=seed)
    
    # Run simulation
    T = 200
    successes = 0
    pred_errors_rf = []
    pred_errors_vlc = []
    
    for t in range(T):
        snr_rf_true, snr_vlc_true = channel.step()
        
        # Make decision
        decision = policy.decide_early()
        
        # Measure and update
        measured_rf = decision['action'] in ('RF', 'DUP')
        measured_vlc = decision['action'] in ('VLC', 'DUP')
        meas_rf, meas_vlc = channel.measure(measured_rf, measured_vlc)
        
        if predictor:
            # Track prediction error before update
            if meas_rf is not None:
                pred_errors_rf.append(abs(predictor.mu_rf - snr_rf_true))
            if meas_vlc is not None:
                pred_errors_vlc.append(abs(predictor.mu_vlc - snr_vlc_true))
            
            predictor.update(meas_rf, meas_vlc)
        
        # Simulate success
        if decision['action'] == 'RF':
            p_success = rf_lut.p_timely(snr_rf_true, decision['K_rf'])
        elif decision['action'] == 'VLC':
            p_success = vlc_lut.p_timely(snr_vlc_true, decision['K_vlc'])
        else:  # DUP
            p_rf = rf_lut.p_timely(snr_rf_true, decision['K_rf']) if decision['K_rf'] > 0 else 0
            p_vlc = vlc_lut.p_timely(snr_vlc_true, decision['K_vlc']) if decision['K_vlc'] > 0 else 0
            p_success = 1.0 - (1.0 - p_rf) * (1.0 - p_vlc)
        
        if np.random.random() < p_success:
            successes += 1
    
    tdr = successes / T
    
    print(f"  Timely Delivery: {tdr:.1%}")
    if pred_errors_rf:
        print(f"  RF Pred MAE: {np.mean(pred_errors_rf):.2f} dB")
    if pred_errors_vlc:
        print(f"  VLC Pred MAE: {np.mean(pred_errors_vlc):.2f} dB")
    
    return tdr

def main():
    print("="*60)
    print("VALIDATING PREDICTION POLICY FIXES")
    print("="*60)
    
    # Load config
    cfg = yaml.safe_load(open('configs/s7_evaluation.yaml'))
    
    # Test scenarios
    scenarios = ['complementary', 'correlated_blockage', 'balanced']
    
    results = {}
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.upper()}")
        print('='*60)
        
        scenario_results = {}
        
        # Test Best-Link
        tdr_bl = test_policy('best_link', cfg['policies']['best_link'], scenario)
        scenario_results['best_link'] = tdr_bl
        
        # Test Conditional-DUP
        tdr_cd = test_policy('conditional_dup_balanced', 
                            cfg['policies']['conditional_dup_balanced'], scenario)
        scenario_results['conditional_dup'] = tdr_cd
        
        results[scenario] = scenario_results
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for scenario, res in results.items():
        print(f"\n{scenario}:")
        print(f"  Best-Link: {res['best_link']:.1%}")
        print(f"  Conditional-DUP: {res['conditional_dup']:.1%}")
    
    # Overall averages
    bl_avg = np.mean([r['best_link'] for r in results.values()])
    cd_avg = np.mean([r['conditional_dup'] for r in results.values()])
    
    print(f"\nOVERALL AVERAGES:")
    print(f"  Best-Link: {bl_avg:.1%}")
    print(f"  Conditional-DUP: {cd_avg:.1%}")
    print(f"\nTarget: Beat Threshold at 76%")
    print(f"Best-Link improvement needed: {max(0, 0.76 - bl_avg):.1%}")
    print(f"Conditional-DUP improvement needed: {max(0, 0.76 - cd_avg):.1%}")

if __name__ == "__main__":
    main()
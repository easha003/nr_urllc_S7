#!/usr/bin/env python3
"""
S7: Comprehensive Evaluation of Hybrid RF-VLC System

Evaluates multiple policies across multiple scenarios with statistical rigor.
Generates publication-ready results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
from tqdm import tqdm

from nr_urllc.adapters import LinkLUT
from nr_urllc.hybrid import HybridController, PolicyConfig
from nr_urllc.predictors import make_predictor
from nr_urllc.channel_sim import create_channel_simulator, SCENARIOS
from nr_urllc.baselines import create_baseline
from nr_urllc.metrics import MetricsCalculator, aggregate_multi_run, format_metrics


def run_single_simulation(policy, channel_sim, predictor, rf_lut, vlc_lut, 
                         T: int = 200, probe_config: Dict = None) -> List[Dict]:
    """
    Run a single simulation episode.
    
    Args:
        policy: Policy object (HybridController or baseline)
        channel_sim: ChannelSimulator instance
        predictor: SNR predictor (None for baselines)
        rf_lut: RF lookup table
        vlc_lut: VLC lookup table
        T: Number of timesteps
        probe_config: Probing configuration dict
        
    Returns:
        List of records, one per timestep
    """
    records = []
    last_action = None
    
    # Default probe config
    if probe_config is None:
        probe_config = {'period': 0, 'on_switch': False, 'r_probe': 0.5}
    
    for t in range(T):
        # Advance channel
        snr_rf_true, snr_vlc_true = channel_sim.step()
        
        # Make decision
        if hasattr(policy, 'decide_early'):
            # HybridController with predictor
            decision = policy.decide_early()
        else:
            # Baseline policy
            decision = policy.decide(snr_rf_true, snr_vlc_true)
        
        # Determine which links are measured
        measured_rf = decision['action'] in ('RF', 'DUP')
        measured_vlc = decision['action'] in ('VLC', 'DUP')
        
        # Check for switch
        switch_happened = (last_action is not None and last_action != decision['action'])
        
        # Probing logic
        need_periodic = (probe_config['period'] > 0 and t % probe_config['period'] == 0)
        need_switch = probe_config['on_switch'] and switch_happened
        
        if need_periodic or need_switch:
            # Probe both links
            meas_rf, meas_vlc = channel_sim.measure(True, True)
        else:
            # Only measure what we're using
            meas_rf, meas_vlc = channel_sim.measure(measured_rf, measured_vlc)
        
        # Update predictor
        if predictor is not None:
            predictor.update(meas_rf, meas_vlc)
            
            # ✅ NEW: Adaptive parameter tuning every 50 steps
            if hasattr(predictor, 'adapt_parameters') and t > 0 and t % 50 == 0:
                # Extract SNR history from records
                rf_history = [r['snr_rf_true'] for r in records[-50:]]
                vlc_history = [r['snr_vlc_true'] for r in records[-50:]]
                predictor.adapt_parameters(rf_history, vlc_history)
        
        # Record
        record = {
            't': t,
            'action': decision['action'],
            'K_rf': decision['K_rf'],
            'K_vlc': decision['K_vlc'],
            'p_est': decision['p_est'],
            'snr_rf_true': float(snr_rf_true),
            'snr_vlc_true': float(snr_vlc_true)
        }
        records.append(record)

        # ✅ NEW: Track prediction quality
        if predictor is not None and t > 0:
            # Get previous prediction
            prev_record = records[-1] if len(records) > 0 else None
            if prev_record and 'mu_rf_pred' in prev_record:
                # Compute prediction errors
                rf_error = abs(snr_rf_true - prev_record['mu_rf_pred'])
                vlc_error = abs(snr_vlc_true - prev_record['mu_vlc_pred'])
                record['rf_pred_error'] = float(rf_error)
                record['vlc_pred_error'] = float(vlc_error)
            
            # Store current prediction for next step
            (mu_rf, sig_rf), (mu_vlc, sig_vlc) = predictor.predict_next()
            record['mu_rf_pred'] = float(mu_rf)
            record['mu_vlc_pred'] = float(mu_vlc)
            record['sig_rf_pred'] = float(sig_rf)
            record['sig_vlc_pred'] = float(sig_vlc)
        
        last_action = decision['action']
    
    return records


def evaluate_policy(policy_name: str, policy_config: Dict, 
                   scenario: str, rf_lut: LinkLUT, vlc_lut: LinkLUT,
                   n_runs: int = 100, T: int = 200, seed_offset: int = 0) -> List[Dict]:
    """
    Evaluate a single policy on a single scenario across multiple runs.
    
    Args:
        policy_name: Name of policy
        policy_config: Policy configuration dict
        scenario: Scenario name
        rf_lut: RF lookup table
        vlc_lut: VLC lookup table
        n_runs: Number of independent runs
        T: Timesteps per run
        seed_offset: Offset for random seeds
        
    Returns:
        List of record lists, one per run
    """
    all_run_records = []
    
    for run_idx in tqdm(range(n_runs), desc=f"{policy_name} on {scenario}", leave=False):
        seed = seed_offset + run_idx
        
        # Create channel simulator
        channel_sim = create_channel_simulator(scenario, seed=seed)
        
        # Create predictor (if needed)
        if 'predictor' in policy_config:
            predictor = make_predictor(policy_config)
        else:
            predictor = None
        
        # Create policy
        if policy_name in ['rf_only', 'vlc_only', 'always_dup', 'oracle', 'threshold']:
            # Baseline policy
            K_total = policy_config.get('K_total', 2)
            policy = create_baseline(policy_name, rf_lut, vlc_lut, K_total)
        else:
            # HybridController
            pc = PolicyConfig(
                name=policy_config['hybrid']['policy'],
                K_total=policy_config['hybrid']['K_total'],
                p_gate=policy_config['hybrid'].get('p_gate', 0.97),
                dup_split=tuple(policy_config['hybrid'].get('dup_split', [1,1])),
                allow_switch=policy_config['hybrid'].get('allow_switch', True),
                hysteresis_margin_prob=policy_config['hybrid'].get('hysteresis_margin_prob', 0.05),
                epsilon=policy_config['hybrid'].get('epsilon', 0.0),
            )
            conf_k = policy_config['hybrid'].get('conf_k', 1.0)
            policy = HybridController(rf_lut, vlc_lut, pc, predictor=predictor, conf_k=conf_k)
        
        # Run simulation
        probe_cfg = policy_config['hybrid'].get('probe') if 'hybrid' in policy_config else None
        records = run_single_simulation(policy, channel_sim, predictor, rf_lut, vlc_lut,
                                       T=T, probe_config=probe_cfg)
        
        all_run_records.append(records)
    
    return all_run_records


def main(cfg_path: str):
    """Main evaluation function."""
    cfg = yaml.safe_load(open(cfg_path, "r"))
    
    # Load LUTs
    rf_lut = LinkLUT.from_json(cfg['adapters']['rf_lut'])
    vlc_lut = LinkLUT.from_json(cfg['adapters']['vlc_lut'])
    
    # Evaluation settings
    eval_cfg = cfg.get('evaluation', {})
    n_runs = eval_cfg.get('n_runs', 100)
    T = eval_cfg.get('timesteps', 200)
    scenarios = eval_cfg.get('scenarios', list(SCENARIOS.keys()))
    
    # Policies to evaluate
    policies = cfg.get('policies', {})
    
    # Output directory
    out_dir = Path(cfg.get('output_dir', 'artifacts/s7'))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics calculator
    deadline_ms = eval_cfg.get('deadline_ms', 5.0)
    metrics_calc = MetricsCalculator(deadline_ms=deadline_ms)
    
    # Store all results
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"S7 COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")
    print(f"Scenarios: {scenarios}")
    print(f"Policies: {list(policies.keys())}")
    print(f"Runs per (scenario, policy): {n_runs}")
    print(f"Timesteps per run: {T}")
    print(f"{'='*60}\n")
    
    # Evaluate each scenario
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {SCENARIOS[scenario].name}")
        print(f"{'='*60}")
        
        scenario_results = {}
        
        # Evaluate each policy
        for policy_name, policy_config in policies.items():
            print(f"\nEvaluating policy: {policy_name}")
            
            # Run multiple simulations
            all_run_records = evaluate_policy(
                policy_name, policy_config, scenario, rf_lut, vlc_lut,
                n_runs=n_runs, T=T, seed_offset=0
            )
            
            # Compute metrics for each run
            metrics_per_run = []
            for records in all_run_records:
                metrics = metrics_calc.compute_with_luts(records, rf_lut, vlc_lut)
                metrics_per_run.append(metrics)
            
            # Aggregate across runs
            aggregated = aggregate_multi_run(metrics_per_run)
            
            # Store results
            scenario_results[policy_name] = {
                'aggregated': aggregated,
                'raw_records': all_run_records[0]  # Save first run for inspection
            }
            
            # Print summary
            print(f"\n  {policy_name} Results (mean ± std):")
            print(f"    Timely Delivery: {aggregated['timely_delivery_ratio']}")
            print(f"    Mean Latency: {aggregated['mean_latency_ms']} ms")
            print(f"    Mean Energy: {aggregated['mean_energy']}")
            print(f"    Action Dist: RF={aggregated['frac_rf'].mean:.1%}, "
                  f"VLC={aggregated['frac_vlc'].mean:.1%}, DUP={aggregated['frac_dup'].mean:.1%}")
        
        all_results[scenario] = scenario_results
        
        # Save scenario results
        scenario_dir = out_dir / scenario
        scenario_dir.mkdir(exist_ok=True)
        
        # Save summary JSON
        summary = {}
        for policy_name, results in scenario_results.items():
            summary[policy_name] = {
                metric_name: {
                    'mean': float(stats.mean),
                    'std': float(stats.std),
                    'ci_95_lower': float(stats.ci_95_lower),
                    'ci_95_upper': float(stats.ci_95_upper)
                }
                for metric_name, stats in results['aggregated'].items()
            }
        
        (scenario_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        
        # Save first run trace for each policy
        for policy_name, results in scenario_results.items():
            trace_file = scenario_dir / f'trace_{policy_name}.json'
            trace_file.write_text(json.dumps({'records': results['raw_records']}, indent=2))
    
    # Overall summary
    print(f"\n\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    
    # Best policy per scenario
    for scenario in scenarios:
        best_policy = max(
            all_results[scenario].keys(),
            key=lambda p: all_results[scenario][p]['aggregated']['timely_delivery_ratio'].mean
        )
        best_tdr = all_results[scenario][best_policy]['aggregated']['timely_delivery_ratio']
        
        print(f"\n{SCENARIOS[scenario].name}:")
        print(f"  Best Policy: {best_policy}")
        print(f"  Timely Delivery: {best_tdr}")
    
    # Save manifest
    manifest = {
        'evaluation': eval_cfg,
        'scenarios': [SCENARIOS[s].name for s in scenarios],
        'policies': list(policies.keys()),
        'n_runs': n_runs,
        'timesteps': T,
        'deadline_ms': deadline_ms
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    
    print(f"\n\nResults saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S7: Comprehensive hybrid system evaluation")
    ap.add_argument("--cfg", required=True, help="YAML configuration file")
    args = ap.parse_args()
    
    main(args.cfg)

#!/usr/bin/env python3
"""
Generate LaTeX tables from S7 evaluation results for publication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict


def load_results(results_dir: Path) -> tuple:
    """Load S7 evaluation results."""
    manifest = json.loads((results_dir / 'manifest.json').read_text())
    
    results = {}
    for scenario in manifest['evaluation']['scenarios']:
        scenario_dir = results_dir / scenario
        if scenario_dir.exists():
            summary_file = scenario_dir / 'summary.json'
            if summary_file.exists():
                results[scenario] = json.loads(summary_file.read_text())
    
    return results, manifest


def generate_main_results_table(results: Dict, manifest: Dict) -> str:
    """
    Generate main results table with timely delivery ratio for all policies and scenarios.
    """
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    
    # Start LaTeX table
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Timely Delivery Ratio across Policies and Scenarios (mean $\pm$ std)}")
    latex.append(r"\label{tab:main_results}")
    latex.append(r"\begin{tabular}{l" + "c" * len(scenarios) + "}")
    latex.append(r"\toprule")
    
    # Header
    header = "Policy & " + " & ".join([s.replace('_', ' ').title() for s in scenarios]) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")
    
    # Data rows
    for policy in policies:
        row = [policy.replace('_', ' ')]
        
        best_in_row = []
        for scenario in scenarios:
            if scenario in results and policy in results[scenario]:
                metric = results[scenario][policy]['timely_delivery_ratio']
                mean = metric['mean']
                std = metric['std']
                row.append(f"{mean:.3f} $\\pm$ {std:.3f}")
                best_in_row.append(mean)
            else:
                row.append("---")
                best_in_row.append(0.0)
        
        latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def generate_energy_table(results: Dict, manifest: Dict) -> str:
    """Generate table with energy consumption metrics."""
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Mean Energy per Packet (mean $\pm$ std)}")
    latex.append(r"\label{tab:energy}")
    latex.append(r"\begin{tabular}{l" + "c" * len(scenarios) + "}")
    latex.append(r"\toprule")
    
    header = "Policy & " + " & ".join([s.replace('_', ' ').title() for s in scenarios]) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")
    
    for policy in policies:
        row = [policy.replace('_', ' ')]
        
        for scenario in scenarios:
            if scenario in results and policy in results[scenario]:
                metric = results[scenario][policy]['mean_energy']
                mean = metric['mean']
                std = metric['std']
                row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
            else:
                row.append("---")
        
        latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def generate_action_dist_table(results: Dict, manifest: Dict) -> str:
    """Generate table with action distribution (RF/VLC/DUP percentages)."""
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Action Distribution: RF\% / VLC\% / DUP\%}")
    latex.append(r"\label{tab:action_dist}")
    latex.append(r"\begin{tabular}{l" + "c" * len(scenarios) + "}")
    latex.append(r"\toprule")
    
    header = "Policy & " + " & ".join([s.replace('_', ' ').title() for s in scenarios]) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")
    
    for policy in policies:
        row = [policy.replace('_', ' ')]
        
        for scenario in scenarios:
            if scenario in results and policy in results[scenario]:
                rf = results[scenario][policy]['frac_rf']['mean'] * 100
                vlc = results[scenario][policy]['frac_vlc']['mean'] * 100
                dup = results[scenario][policy]['frac_dup']['mean'] * 100
                row.append(f"{rf:.0f}/{vlc:.0f}/{dup:.0f}")
            else:
                row.append("---")
        
        latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def generate_summary_statistics(results: Dict, manifest: Dict) -> str:
    """Generate summary statistics text."""
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    
    lines = []
    lines.append("=" * 80)
    lines.append("S7 EVALUATION SUMMARY STATISTICS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total scenarios evaluated: {len(scenarios)}")
    lines.append(f"Total policies evaluated: {len(policies)}")
    lines.append(f"Runs per (scenario, policy): {manifest['n_runs']}")
    lines.append(f"Timesteps per run: {manifest['timesteps']}")
    lines.append("")
    
    # Best policy per scenario
    lines.append("BEST POLICY PER SCENARIO:")
    lines.append("-" * 80)
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        best_policy = None
        best_tdr = 0.0
        
        for policy in policies:
            if policy in results[scenario]:
                tdr = results[scenario][policy]['timely_delivery_ratio']['mean']
                if tdr > best_tdr:
                    best_tdr = tdr
                    best_policy = policy
        
        if best_policy:
            lines.append(f"  {scenario.replace('_', ' ').title():25s}: {best_policy:25s} (TDR = {best_tdr:.4f})")
    
    lines.append("")
    
    # Overall best policy (averaged across scenarios)
    lines.append("OVERALL RANKING (Average TDR across all scenarios):")
    lines.append("-" * 80)
    
    policy_avg_tdr = {}
    for policy in policies:
        tdr_list = []
        for scenario in scenarios:
            if scenario in results and policy in results[scenario]:
                tdr_list.append(results[scenario][policy]['timely_delivery_ratio']['mean'])
        
        if tdr_list:
            policy_avg_tdr[policy] = sum(tdr_list) / len(tdr_list)
    
    ranked_policies = sorted(policy_avg_tdr.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (policy, avg_tdr) in enumerate(ranked_policies, 1):
        lines.append(f"  {rank}. {policy:30s}: {avg_tdr:.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main(results_dir: str, output_dir: str):
    """Generate LaTeX tables and summary."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results, manifest = load_results(results_path)
    
    print(f"Found {len(results)} scenarios, {len(manifest['policies'])} policies")
    
    # Generate tables
    print("\nGenerating LaTeX tables...")
    
    main_table = generate_main_results_table(results, manifest)
    (output_path / 'table_main_results.tex').write_text(main_table)
    print(f"  Saved: table_main_results.tex")
    
    energy_table = generate_energy_table(results, manifest)
    (output_path / 'table_energy.tex').write_text(energy_table)
    print(f"  Saved: table_energy.tex")
    
    action_table = generate_action_dist_table(results, manifest)
    (output_path / 'table_action_dist.tex').write_text(action_table)
    print(f"  Saved: table_action_dist.tex")
    
    # Generate summary statistics
    summary = generate_summary_statistics(results, manifest)
    (output_path / 'summary_statistics.txt').write_text(summary)
    print(f"  Saved: summary_statistics.txt")
    
    # Print summary to console
    print("\n" + summary)
    
    print(f"\nAll tables saved to: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate LaTeX tables from S7 results")
    ap.add_argument("--results", default="artifacts/s7", help="S7 results directory")
    ap.add_argument("--output", default="artifacts/s7/tables", help="Output directory for tables")
    args = ap.parse_args()
    
    main(args.results, args.output)

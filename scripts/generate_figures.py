#!/usr/bin/env python3
"""
Generate publication-quality figures from S7 evaluation results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

# Set publication style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12

DISPLAY_POLICY = {
    "rf_only": "RF_only",
    "vlc_only": "VLC_only",
    "always_dup": "always_DUP",
    "conditional_dup_balanced": "conditional_DUP_balanced",
    "oracle": "oracle",
    "threshold": "threshold",
    "best_link": "best_link",
}
DISPLAY_SCENARIO = {
    "RF_good": "RF Good",
    "VLC_good": "VLC Good",
}

def pretty_scenario(s: str) -> str:
    return DISPLAY_SCENARIO.get(s, s.replace("_", " ").title())


def load_results(results_dir: Path) -> Dict:
    """Load all S7 evaluation results."""
    manifest = json.loads((results_dir / 'manifest.json').read_text())
    
    results = {}
    for scenario in manifest['evaluation']['scenarios']:
        scenario_dir = results_dir / scenario
        if not scenario_dir.exists():
            continue
        
        summary_file = scenario_dir / 'summary.json'
        if summary_file.exists():
            results[scenario] = json.loads(summary_file.read_text())
    
    return results, manifest



def plot_timely_delivery_comparison(results: Dict, manifest: Dict, output_path: Path):
    """
    Bar chart comparing timely delivery ratio across policies and scenarios.
    """
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    policy_labels = [DISPLAY_POLICY.get(p, p) for p in policies]
    
    # Extract data
    data = np.zeros((len(scenarios), len(policies)))
    errors = np.zeros((len(scenarios), len(policies)))
    
    for i, scenario in enumerate(scenarios):
        if scenario not in results:
            continue
        for j, policy in enumerate(policies):
            if policy in results[scenario]:
                metric = results[scenario][policy]['timely_delivery_ratio']
                data[i, j] = metric['mean']
                # Error bars: ±1 std
                errors[i, j] = metric['std']
    
    # Create figure
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 3.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    x = np.arange(len(policies))
    width = 0.7
    
    # Color scheme
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))
    
    for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
        bars = ax.bar(x, data[i, :], width, yerr=errors[i, :],
                     color=colors, capsize=3, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Formatting
        ax.set_xlabel('Policy', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Timely Delivery Ratio', fontweight='bold')
        ax.set_title(pretty_scenario(scenario), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{data[i,j]:.3f}',
                   ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_energy_vs_performance(results: Dict, manifest: Dict, output_path: Path):
    """
    Scatter plot: energy efficiency vs timely delivery ratio.
    """
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    policy_labels = [DISPLAY_POLICY.get(p, p) for p in policies]
    
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 3.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    markers = ['o', 's', 'v', '^', 'd', 'p', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    
    for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
        if scenario not in results:
            continue
        
        for j, policy in enumerate(policies):
            if policy not in results[scenario]:
                continue
            
            tdr = results[scenario][policy]['timely_delivery_ratio']
            energy = results[scenario][policy]['mean_energy']
            
            ax.errorbar(energy['mean'], tdr['mean'],
                       xerr=energy['std'], yerr=tdr['std'],
                       marker=markers[j % len(markers)],
                       markersize=10,  markeredgewidth=0.05,
                       capsize=0.05, capthick=0.05, elinewidth=0.05,
                       color=colors[j], label=policy_labels[j],
                       linestyle='none', alpha=0.8)
        
        ax.set_xlabel('Mean Energy per Packet', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Timely Delivery Ratio', fontweight='bold')
        ax.set_title(pretty_scenario(scenario), fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        if i == len(scenarios) - 1:
            ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_action_distribution(results: Dict, manifest: Dict, output_path: Path):
    """
    Stacked bar chart showing action distribution (RF/VLC/DUP) per policy.
    """
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    policy_labels = [DISPLAY_POLICY.get(p, p) for p in policies]
    
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 3.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    actions = ['frac_rf', 'frac_vlc', 'frac_dup']
    action_labels = ['RF', 'VLC', 'DUP']
    colors_action = ['#d62728', '#2ca02c', '#1f77b4']
    
    for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
        if scenario not in results:
            continue
        
        # Prepare data
        rf_data = []
        vlc_data = []
        dup_data = []
        
        for policy in policies:
            if policy not in results[scenario]:
                rf_data.append(0)
                vlc_data.append(0)
                dup_data.append(0)
                continue
            
            rf_data.append(results[scenario][policy]['frac_rf']['mean'])
            vlc_data.append(results[scenario][policy]['frac_vlc']['mean'])
            dup_data.append(results[scenario][policy]['frac_dup']['mean'])
        
        x = np.arange(len(policies))
        width = 0.7
        
        # Stacked bars
        p1 = ax.bar(x, rf_data, width, label='RF', color=colors_action[0], alpha=0.8)
        p2 = ax.bar(x, vlc_data, width, bottom=rf_data, label='VLC', color=colors_action[1], alpha=0.8)
        
        bottom = np.array(rf_data) + np.array(vlc_data)
        p3 = ax.bar(x, dup_data, width, bottom=bottom, label='DUP', color=colors_action[2], alpha=0.8)
        
        ax.set_xlabel('Policy', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Action Distribution', fontweight='bold')
        ax.set_title(pretty_scenario(scenario), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        
        if i == len(scenarios) - 1:
            ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_comparison(results: Dict, manifest: Dict, output_path: Path):
    """
    Box plot comparing latency distributions across policies.
    """
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    policy_labels = [DISPLAY_POLICY.get(p, p) for p in policies]
    
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 3.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
        if scenario not in results:
            continue
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for policy in policies:
            if policy not in results[scenario]:
                continue
            
            mean_lat = results[scenario][policy]['mean_latency_ms']['mean']
            std_lat = results[scenario][policy]['mean_latency_ms']['std']
            
            # Approximate distribution (Gaussian assumption)
            # For visualization, we show mean ± std
            data_to_plot.append([mean_lat])
            labels.append(policy)
        
        # Since we don't have raw latency samples, use error bars instead
        means = [results[scenario][p]['mean_latency_ms']['mean'] for p in policies if p in results[scenario]]
        stds = [results[scenario][p]['mean_latency_ms']['std'] for p in policies if p in results[scenario]]
        
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Policy', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean Latency (ms)', fontweight='bold')
        ax.set_title(pretty_scenario(scenario), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scenario_heatmap(results: Dict, manifest: Dict, output_path: Path):
    """
    Heatmap: policies (rows) vs scenarios (columns), colored by timely delivery ratio.
    """
    scenarios = manifest['evaluation']['scenarios']
    policies = manifest['policies']
    policy_labels = [DISPLAY_POLICY.get(p, p) for p in policies]
    
    # Prepare data matrix
    data = np.zeros((len(policies), len(scenarios)))
    
    for i, policy in enumerate(policies):
        for j, scenario in enumerate(scenarios):
            if scenario in results and policy in results[scenario]:
                data[i, j] = results[scenario][policy]['timely_delivery_ratio']['mean']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(policies)))
    ax.set_xticklabels([pretty_scenario(s).replace(" ", "\n") for s in scenarios])
    ax.set_yticklabels(policy_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Timely Delivery Ratio', rotation=-90, va="bottom", fontweight='bold')
    
    # Add text annotations
    for i in range(len(policies)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Policy Performance Across Scenarios', fontweight='bold', fontsize=12)
    fig.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main(results_dir: str, output_dir: str):
    """Generate all figures."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results, manifest = load_results(results_path)
    
    print(f"Found {len(results)} scenarios, {len(manifest['policies'])} policies")
    
    # Generate figures
    print("\nGenerating figures...")
    
    plot_timely_delivery_comparison(results, manifest, 
                                    output_path / 'timely_delivery_comparison.png')
    
    plot_energy_vs_performance(results, manifest,
                               output_path / 'energy_vs_performance.png')
    
    plot_action_distribution(results, manifest,
                            output_path / 'action_distribution.png')
    
    plot_latency_comparison(results, manifest,
                           output_path / 'latency_comparison.png')
    
    plot_scenario_heatmap(results, manifest,
                         output_path / 'scenario_heatmap.png')
    
    print(f"\nAll figures saved to: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate publication figures from S7 results")
    ap.add_argument("--results", default="artifacts/s7", help="S7 results directory")
    ap.add_argument("--output", default="artifacts/s7/figures", help="Output directory for figures")
    args = ap.parse_args()
    
    main(args.results, args.output)

"""
Performance metrics for hybrid RF-VLC system evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from scipy import stats


@dataclass
class TransmissionRecord:
    """Record of a single transmission attempt."""
    timestep: int
    action: str  # 'RF', 'VLC', or 'DUP'
    K_rf: int
    K_vlc: int
    snr_rf_true: float
    snr_vlc_true: float
    p_est: float  # Estimated reliability
    success: bool  # Whether packet delivered on time (ground truth)
    latency_ms: float = 0.0  # Actual latency


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for hybrid system."""
    
    # Primary metrics
    timely_delivery_ratio: float = 0.0  # Fraction of packets delivered on time
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Resource utilization
    mean_k_total: float = 0.0  # Average total attempts per packet
    mean_energy: float = 0.0   # Average energy per packet
    
    # Action distribution
    frac_rf: float = 0.0
    frac_vlc: float = 0.0
    frac_dup: float = 0.0
    
    # Switching behavior
    num_switches: int = 0
    switch_rate_hz: float = 0.0  # Assuming 1ms per timestep
    
    # Prediction accuracy (if available)
    prediction_rmse: float = 0.0
    prediction_mae: float = 0.0
    
    # Statistical properties
    std_latency_ms: float = 0.0
    
    # Raw data for further analysis
    latencies: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


class MetricsCalculator:
    """Calculate performance metrics from simulation records."""
    
    def __init__(self, deadline_ms: float = 5.0, 
                 energy_rf: float = 1.0, 
                 energy_vlc: float = 0.3):
        """
        Args:
            deadline_ms: Latency deadline for timely delivery
            energy_rf: Energy cost per RF transmission (normalized)
            energy_vlc: Energy cost per VLC transmission (normalized)
        """
        self.deadline_ms = deadline_ms
        self.energy_rf = energy_rf
        self.energy_vlc = energy_vlc
    
    def compute_metrics(self, records: List[Dict]) -> PerformanceMetrics:
        """
        Compute comprehensive metrics from simulation records.
        
        Args:
            records: List of dicts with keys: t, action, K_rf, K_vlc, snr_rf_true, 
                     snr_vlc_true, p_est, (optional) success, latency_ms
        
        Returns:
            PerformanceMetrics object
        """
        if not records:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # Extract data
        actions = [r['action'] for r in records]
        k_rf_vals = [r['K_rf'] for r in records]
        k_vlc_vals = [r['K_vlc'] for r in records]
        
        # Simulate success based on ground-truth SNR and LUT
        # (This requires LUTs to be passed in - for now, use p_est as proxy)
        successes = []
        latencies = []
        
        for r in records:
            # If success is provided, use it
            if 'success' in r:
                success = r['success']
            else:
                # Simulate: success if random draw < p_est
                success = (np.random.random() < r['p_est'])
            
            successes.append(success)
            
            # Simulate latency based on K and action
            if 'latency_ms' in r:
                latency = r['latency_ms']
            else:
                # Simple model: base latency + attempts × slot duration
                base_latency = 1.0  # ms
                slot_duration = 0.5  # ms per transmission attempt
                k_total = r['K_rf'] + r['K_vlc']
                latency = base_latency + k_total * slot_duration
            
            latencies.append(latency)
        
        metrics.latencies = latencies
        metrics.actions = actions
        
        # Primary metrics
        metrics.timely_delivery_ratio = np.mean(successes)
        
        timely_latencies = [lat for lat, success in zip(latencies, successes) if success]
        if timely_latencies:
            metrics.mean_latency_ms = np.mean(timely_latencies)
            metrics.std_latency_ms = np.std(timely_latencies)
            metrics.p95_latency_ms = np.percentile(timely_latencies, 95)
            metrics.p99_latency_ms = np.percentile(timely_latencies, 99)
        
        # Resource utilization
        k_totals = [k_rf + k_vlc for k_rf, k_vlc in zip(k_rf_vals, k_vlc_vals)]
        metrics.mean_k_total = np.mean(k_totals)
        
        energies = [self.energy_rf * k_rf + self.energy_vlc * k_vlc 
                    for k_rf, k_vlc in zip(k_rf_vals, k_vlc_vals)]
        metrics.mean_energy = np.mean(energies)
        
        # Action distribution
        total = len(actions)
        metrics.frac_rf = actions.count('RF') / total
        metrics.frac_vlc = actions.count('VLC') / total
        metrics.frac_dup = actions.count('DUP') / total
        
        # Switching behavior
        switches = sum(1 for i in range(1, len(actions)) 
                      if actions[i] != actions[i-1])
        metrics.num_switches = switches
        # Assuming 1ms per timestep
        metrics.switch_rate_hz = switches / (len(actions) * 0.001)
        
        return metrics
    
    def compute_with_luts(self, records: List[Dict], rf_lut, vlc_lut) -> PerformanceMetrics:
        """
        Compute metrics with ground-truth success based on LUTs.
        
        This is more accurate than using p_est as proxy.
        """
        # Augment records with ground-truth success
        for r in records:
            snr_rf = r['snr_rf_true']
            snr_vlc = r['snr_vlc_true']
            k_rf = r['K_rf']
            k_vlc = r['K_vlc']
            
            if r['action'] == 'RF':
                p_success = rf_lut.p_timely(snr_rf, k_rf)
            elif r['action'] == 'VLC':
                p_success = vlc_lut.p_timely(snr_vlc, k_vlc)
            elif r['action'] == 'DUP':
                p_rf = rf_lut.p_timely(snr_rf, k_rf) if k_rf > 0 else 0.0
                p_vlc = vlc_lut.p_timely(snr_vlc, k_vlc) if k_vlc > 0 else 0.0
                p_success = 1.0 - (1.0 - p_rf) * (1.0 - p_vlc)
            else:
                p_success = 0.0
            
            # Bernoulli trial
            r['success'] = (np.random.random() < p_success)
        
        return self.compute_metrics(records)


def compare_policies(metrics_dict: Dict[str, PerformanceMetrics]) -> Dict[str, Dict]:
    """
    Compare multiple policies statistically.
    
    Args:
        metrics_dict: Dict mapping policy name to PerformanceMetrics
        
    Returns:
        Dict with comparison results
    """
    comparison = {}
    
    policy_names = list(metrics_dict.keys())
    
    # Primary comparison: timely delivery ratio
    comparison['timely_delivery'] = {
        name: metrics_dict[name].timely_delivery_ratio 
        for name in policy_names
    }
    
    # Best policy
    best_policy = max(policy_names, 
                     key=lambda p: metrics_dict[p].timely_delivery_ratio)
    comparison['best_policy'] = best_policy
    
    # Resource efficiency
    comparison['mean_energy'] = {
        name: metrics_dict[name].mean_energy 
        for name in policy_names
    }
    
    # Action distributions
    comparison['action_dist'] = {
        name: {
            'RF': metrics_dict[name].frac_rf,
            'VLC': metrics_dict[name].frac_vlc,
            'DUP': metrics_dict[name].frac_dup
        }
        for name in policy_names
    }
    
    return comparison


def format_metrics(metrics: PerformanceMetrics, policy_name: str = "") -> str:
    """Format metrics for pretty printing."""
    lines = []
    if policy_name:
        lines.append(f"=== {policy_name} ===")
    lines.append(f"Timely Delivery Ratio: {metrics.timely_delivery_ratio:.4f}")
    lines.append(f"Mean Latency: {metrics.mean_latency_ms:.3f} ms (±{metrics.std_latency_ms:.3f})")
    lines.append(f"P95 Latency: {metrics.p95_latency_ms:.3f} ms")
    lines.append(f"P99 Latency: {metrics.p99_latency_ms:.3f} ms")
    lines.append(f"Mean K Total: {metrics.mean_k_total:.2f}")
    lines.append(f"Mean Energy: {metrics.mean_energy:.3f}")
    lines.append(f"Action Dist: RF={metrics.frac_rf:.1%}, VLC={metrics.frac_vlc:.1%}, DUP={metrics.frac_dup:.1%}")
    lines.append(f"Switches: {metrics.num_switches} ({metrics.switch_rate_hz:.1f} Hz)")
    return "\n".join(lines)


@dataclass
class MultiRunStats:
    """Statistics aggregated across multiple independent runs."""
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    median: float
    min_val: float
    max_val: float
    
    def __str__(self):
        return f"{self.mean:.4f} ± {self.std:.4f} [95% CI: {self.ci_95_lower:.4f}, {self.ci_95_upper:.4f}]"


def aggregate_multi_run(metrics_list: List[PerformanceMetrics]) -> Dict[str, MultiRunStats]:
    """
    Aggregate metrics from multiple independent runs.
    
    Args:
        metrics_list: List of PerformanceMetrics from different runs
        
    Returns:
        Dict mapping metric name to MultiRunStats
    """
    if not metrics_list:
        return {}
    
    def extract_metric(attr_name: str) -> np.ndarray:
        return np.array([getattr(m, attr_name) for m in metrics_list])
    
    def compute_stats(values: np.ndarray) -> MultiRunStats:
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        # 95% confidence interval using t-distribution
        if len(values) > 1 and std_val > 0.0:
            ci = stats.t.interval(
                0.95, len(values)-1,
                loc=mean_val,
                scale=stats.sem(values)
            )
        else:
            ci = (mean_val, mean_val)
        
        return MultiRunStats(
            mean=mean_val,
            std=std_val,
            ci_95_lower=ci[0],
            ci_95_upper=ci[1],
            median=np.median(values),
            min_val=np.min(values),
            max_val=np.max(values)
        )
    
    aggregated = {}
    
    # Key metrics to aggregate
    metric_names = [
        'timely_delivery_ratio',
        'mean_latency_ms',
        'p95_latency_ms',
        'mean_k_total',
        'mean_energy',
        'frac_rf',
        'frac_vlc',
        'frac_dup',
        'num_switches'
    ]
    
    for name in metric_names:
        values = extract_metric(name)
        aggregated[name] = compute_stats(values)
    
    return aggregated

# nr_urllc/s3_timely_bler_sweep.py
"""
S3 — Timely BLER Sweep: Generate BLER vs SNR curves sliced by K repetitions.

Integrates with existing sweep.py and bler_ofdm_sweep logic to produce:
  - timely_bler_vs_snr_K*.png (one curve per K value)
  - latency_cdf.png (packet decode latency distribution)
  - s3_timely_bler_results.json (detailed results)
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import math

from .s3_timing_urllc import S3TimingController, bler_to_timely_bler


@dataclass
class S3Point:
    """Single measurement point in S3 sweep."""
    snr_db: float
    K: int
    n_packets: int
    n_failures_single: int
    n_failures_K: int
    bler_single: float
    bler_K: float
    timely_success_count: int
    timely_success_prob: float
    within_deadline: bool


class S3TimelySweepResult:
    """Container for S3 sweep results."""
    
    def __init__(self):
        self.points: List[S3Point] = []
        self.snr_list: List[float] = []
        self.K_list: List[int] = []
        self.timing_summary: Dict = {}
    
    def add_point(self, pt: S3Point):
        """Add a measurement point."""
        self.points.append(pt)
        if pt.snr_db not in self.snr_list:
            self.snr_list.append(pt.snr_db)
        if pt.K not in self.K_list:
            self.K_list.append(pt.K)
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            "points": [asdict(p) for p in self.points],
            "snr_list": sorted(list(set(self.snr_list))),
            "K_list": sorted(list(set(self.K_list))),
            "timing_summary": self.timing_summary,
        }
    
    def to_json_file(self, path: str | Path):
        """Write to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def derive_K_bler_from_single(
    bler_single: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Given single-attempt BLER curve, compute K-attempt BLER (independent model).
    
    bler_single: array of BLERs at different SNRs (single attempt)
    K: number of independent attempts
    
    Returns: BLER for K attempts (1 - (1-BLER_single)^K)
    """
    bler_single = np.asarray(bler_single, dtype=float)
    return bler_to_timely_bler(bler_single, K, policy="independent")


def s3_timely_bler_sweep(
    bler_results: Dict[float, float],
    timing_ctrl: S3TimingController,
    K_list: Optional[List[int]] = None,
) -> S3TimelySweepResult:
    """
    Convert BLER curve + timing controller into timely BLER curves per K.
    
    Args:
        bler_results: {snr_db: bler_value} from bler_ofdm_sweep
        timing_ctrl: S3TimingController instance
        K_list: List of K values to evaluate (default: [1, 2, 3])
    
    Returns:
        S3TimelySweepResult with curves
    """
    if K_list is None:
        K_list = [1, 2, 3]
    
    result = S3TimelySweepResult()
    result.timing_summary = timing_ctrl.summary()
    
    snr_sorted = sorted(bler_results.keys())
    bler_array = np.array([bler_results[snr] for snr in snr_sorted])
    
    for K in K_list:
        fits_deadline = timing_ctrl.K_fits_deadline(K)
        bler_K_array = derive_K_bler_from_single(bler_array, K)
        
        for i, snr_db in enumerate(snr_sorted):
            bler_single = float(bler_results[snr_db])
            bler_K = float(bler_K_array[i])
            
            # Timely success: (1 - BLER_K) if K fits deadline, else 0
            if fits_deadline:
                p_timely_success = 1.0 - bler_K
            else:
                p_timely_success = 0.0
            
            pt = S3Point(
                snr_db=float(snr_db),
                K=int(K),
                n_packets=0,  # Synthetic; not from measured sweep
                n_failures_single=0,
                n_failures_K=0,
                bler_single=bler_single,
                bler_K=bler_K,
                timely_success_count=0,
                timely_success_prob=p_timely_success,
                within_deadline=fits_deadline,
            )
            result.add_point(pt)
    
    return result


def plot_timely_bler_curves(
    s3_result: S3TimelySweepResult,
    save_dir: str | Path = "artifacts/s3",
    show: bool = False,
):
    """
    Plot timely BLER vs SNR for each K, saving as timely_bler_vs_snr_K*.png.
    
    Args:
        s3_result: S3TimelySweepResult
        save_dir: Directory to save plots
        show: Whether to display (requires GUI)
    """
    import os
    import matplotlib
    if not os.environ.get("ALLOW_GUI_PLOTS", ""):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Group points by K
    by_K = {}
    for pt in s3_result.points:
        if pt.K not in by_K:
            by_K[pt.K] = []
        by_K[pt.K].append(pt)
    
    for K in sorted(by_K.keys()):
        points = sorted(by_K[K], key=lambda x: x.snr_db)
        snr_vals = np.array([p.snr_db for p in points])
        
        # BLER (K) = 1 - timely_success_prob
        bler_K_vals = np.array([p.bler_K for p in points])
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        ax.semilogy(snr_vals, np.clip(bler_K_vals, 1e-12, 1.0), 
                    marker='o', linewidth=2, markersize=6, label=f'K={K}')
        
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("SNR (Eb/N0) [dB]")
        ax.set_ylabel("BLER (TB error rate)")
        
        within_dl = points[0].within_deadline
        title_suffix = "(fits deadline)" if within_dl else "(EXCEEDS deadline)"
        ax.set_title(f"S3 — Timely BLER vs SNR (K={K}) {title_suffix}")
        ax.legend(loc="best")
        
        fig.tight_layout()
        out_path = Path(save_dir) / f"timely_bler_vs_snr_K{K}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=130)
        print(f"[S3] Saved {out_path}")
        
        if show:
            plt.show()
        plt.close(fig)

def plot_timely_bler_curves_combined(
    s3_result: S3TimelySweepResult,
    save_dir: str | Path = "artifacts/s3",
    show: bool = False,
):
    """
    Plot all K curves on the same figure for comparison.
    
    Args:
        s3_result: S3TimelySweepResult
        save_dir: Directory to save plot
        show: Whether to display
    """
    import os
    import matplotlib
    if not os.environ.get("ALLOW_GUI_PLOTS", ""):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Group points by K
    by_K = {}
    for pt in s3_result.points:
        if pt.K not in by_K:
            by_K[pt.K] = []
        by_K[pt.K].append(pt)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    
    for K in sorted(by_K.keys()):
        points = sorted(by_K[K], key=lambda x: x.snr_db)
        snr_vals = np.array([p.snr_db for p in points])
        bler_K_vals = np.array([p.bler_K for p in points])
        
        ax.semilogy(snr_vals, np.clip(bler_K_vals, 1e-12, 1.0), 
                    marker='o', linewidth=2, markersize=6, label=f'K={K}')
    
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("SNR (Eb/N0) [dB]")
    ax.set_ylabel("BLER (TB error rate)")
    ax.set_title("S3 — Timely BLER vs SNR (All K values)")
    ax.legend(loc="best")
    
    fig.tight_layout()
    out_path = Path(save_dir) / "timely_bler_vs_snr_combined.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    print(f"[S3] Saved {out_path}")
    
    if show:
        plt.show()
    plt.close(fig)

def plot_latency_cdf(
    timing_ctrl: S3TimingController,
    K_list: Optional[List[int]] = None,
    save_dir: str | Path = "artifacts/s3",
    show: bool = False,
):
    """
    Plot CDF of cumulative latencies for different K values.
    
    Args:
        timing_ctrl: S3TimingController
        K_list: List of K values to plot (default: [1, 2, 3])
        save_dir: Directory to save plot
        show: Whether to display
    """
    import os
    import matplotlib
    if not os.environ.get("ALLOW_GUI_PLOTS", ""):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if K_list is None:
        K_list = [1, 2, 3]
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    
    for K in sorted(K_list):
        latency = timing_ctrl.get_cumulative_latency(K)
        fits = timing_ctrl.K_fits_deadline(K)
        
        # Artificial CDF: show latency as a point (could be enhanced with actual packet times)
        ax.axvline(latency, linewidth=2, label=f'K={K} ({latency:.2f} ms)' + 
                   (' ✓' if fits else ' ✗'))
    
    ax.axvline(timing_ctrl.radio_deadline_ms, linewidth=2.5, color='red', 
               linestyle='--', label=f'Deadline ({timing_ctrl.radio_deadline_ms:.2f} ms)')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF (cumulative probability)")
    ax.set_title("S3 — Latency CDF by K repetitions")
    ax.legend(loc="best")
    
    fig.tight_layout()
    out_path = Path(save_dir) / "latency_cdf.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    print(f"[S3] Saved {out_path}")
    
    if show:
        plt.show()
    plt.close(fig)


def plot_s3_heatmap(
    s3_result: S3TimelySweepResult,
    timing_ctrl: S3TimingController,
    save_dir: str | Path = "artifacts/s3",
    show: bool = False,
):
    """
    Plot heatmap: SNR vs K, color = timely success probability.
    
    Args:
        s3_result: S3TimelySweepResult
        timing_ctrl: S3TimingController
        save_dir: Directory to save plot
        show: Whether to display
    """
    import os
    import matplotlib
    if not os.environ.get("ALLOW_GUI_PLOTS", ""):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Pivot: rows=SNR, cols=K, values=timely_success_prob
    snr_unique = sorted(list(set(p.snr_db for p in s3_result.points)))
    K_unique = sorted(list(set(p.K for p in s3_result.points)))
    
    data = np.zeros((len(snr_unique), len(K_unique)))
    
    for pt in s3_result.points:
        i = snr_unique.index(pt.snr_db)
        j = K_unique.index(pt.K)
        data[i, j] = pt.timely_success_prob
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(K_unique)))
    ax.set_xticklabels([f'K={k}' for k in K_unique])
    ax.set_yticks(range(len(snr_unique)))
    ax.set_yticklabels([f'{s:.1f}' for s in snr_unique])
    
    ax.set_xlabel("Number of Attempts (K)")
    ax.set_ylabel("SNR (Eb/N0) [dB]")
    ax.set_title("S3 — Timely Success Probability (SNR vs K)")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(success ∧ timely)")
    
    fig.tight_layout()
    out_path = Path(save_dir) / "s3_timely_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    print(f"[S3] Saved {out_path}")
    
    if show:
        plt.show()
    plt.close(fig)


# Example usage
if __name__ == "__main__":
    from .s3_timing_urllc import S3TimingController
    
    # Dummy BLER results (from S2)
    bler_results = {
        0.0: 0.15,
        2.0: 0.10,
        4.0: 0.05,
        6.0: 0.01,
        8.0: 0.001,
    }
    
    # S3 config
    cfg = {
        "nr": {
            "mu": 2,
            "minislot_symbols": 7,
            "cg_ul": {"K": 2},
            "harq": {"k1_symbols": 1}
        },
        "urllc": {
            "pdb_ms": 10.0,
            "core_budget_ms": 2.0,
        },
        "timing": {
            "grant_delay_ms": 0.0,
            "inter_attempt_gap_ms": 0.1,
            "proc_margin_ms": 0.2,
        }
    }
    
    timing_ctrl = S3TimingController(cfg)
    
    print("Timing summary:", timing_ctrl.summary())
    
    # Generate S3 sweep
    s3_result = s3_timely_bler_sweep(bler_results, timing_ctrl, K_list=[1, 2, 3])
    s3_result.to_json_file("artifacts/s3/s3_timely_bler_results.json")
    
    # Plot
    plot_timely_bler_curves(s3_result)
    plot_latency_cdf(timing_ctrl)
    plot_s3_heatmap(s3_result, timing_ctrl)
    
    print("\n[S3 Complete] All plots and JSON generated in artifacts/s3/")

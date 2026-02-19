# nr_urllc/plots.py
import json, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import os, matplotlib
# Prevent GUI popups (which block your sweep) unless explicitly allowed
if not os.environ.get("ALLOW_GUI_PLOTS", ""):
    matplotlib.use("Agg")  # non-interactive backend (PNG only)

import matplotlib.pyplot as plt

# ----------------- Helpers to normalize result dicts ----------------- #

def _arrays_from_m2_result(result: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract SNR, BER, EVM%, MSE(H) from an M2 result dict."""
    snr = np.asarray(result.get("snr_db", []), dtype=float)
    ber = np.asarray(result.get("ber", []), dtype=float)
    evm_pct = result.get("evm_percent", None)
    mse_H   = result.get("mse_H", None)
    evm = np.asarray(evm_pct, dtype=float) if evm_pct is not None else None
    mse = np.asarray(mse_H,   dtype=float) if mse_H   is not None else None
    return snr, ber, evm, mse

def _arrays_from_m1_result(result: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract SNR, BER from an M1 OFDM-over-AWGN result dict (ber_curve map)."""
    curve = result.get("ber_curve", {})
    if isinstance(curve, dict):
        # keys are SNR dB (strings or floats)
        pairs = sorted([(float(k), float(v)) for k, v in curve.items()], key=lambda x: x[0])
        if len(pairs) == 0:
            return np.array([]), np.array([])
        snr, ber = zip(*pairs)
        return np.asarray(snr, dtype=float), np.asarray(ber, dtype=float)
    return np.array([]), np.array([])

def load_result_json(path: str | Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

# ----------------- Core semilog plotting ----------------- #

def plot_semilog_ber_evm_mse(
    curves: List[Dict[str, np.ndarray]],
    title: str = "BER / EVM / MSE vs SNR",
    save_path: Optional[str | Path] = None,
    show: bool = False,
    note_evm_as_fraction: bool = True,
):
    """
    curves: list of dicts with keys:
      - 'label' (str)
      - 'snr_db' (np.ndarray)
      - 'ber' (np.ndarray)
      - optional 'evm_percent' (np.ndarray, %)
      - optional 'mse_H' (np.ndarray)
    All three metrics are drawn on the same semilog y-axis.
    EVM is plotted as FRACTION (EVM% / 100) to coexist with BER/MSE scales.
    """
    if len(curves) == 0:
        raise ValueError("No curves provided to plot.")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.set_yscale("log")

    # Styling cycles (markers/linestyles)
    mark = ["o", "s", "^", "D", "v", "P", "X"]
    dash = ["-", "--", "-.", ":", (0, (1, 1)), (0, (3, 1, 1, 1))]

    def _nz(x):
        return np.maximum(np.asarray(x, dtype=float), 1e-12)

    legend_items = 0

    for idx, c in enumerate(curves):
        label = str(c.get("label", f"cfg{idx+1}"))
        snr   = np.asarray(c["snr_db"], dtype=float)

        # sort by SNR just in case
        order = np.argsort(snr)
        snr = snr[order]

        # BER
        if "ber" in c and c["ber"] is not None:
            ber = _nz(np.asarray(c["ber"], dtype=float)[order])
            ax.semilogy(snr, ber, marker=mark[idx % len(mark)], linestyle=dash[0], linewidth=1.6, markersize=5,
                        label=f"{label} — BER")
            legend_items += 1

        # EVM% (plot as fraction)
        if "evm_percent" in c and c["evm_percent"] is not None:
            evm_pct = np.asarray(c["evm_percent"], dtype=float)[order]
            evm_frac = _nz(evm_pct / 100.0) if note_evm_as_fraction else _nz(evm_pct)
            suffix = "EVM (frac)" if note_evm_as_fraction else "EVM (%)"
            ax.semilogy(snr, evm_frac, marker=mark[idx % len(mark)], linestyle=dash[1], linewidth=1.6, markersize=5,
                        label=f"{label} — {suffix}")
            legend_items += 1

        # MSE(H)
        if "mse_H" in c and c["mse_H"] is not None:
            mse = _nz(np.asarray(c["mse_H"], dtype=float)[order])
            ax.semilogy(snr, mse, marker=mark[idx % len(mark)], linestyle=dash[2], linewidth=1.6, markersize=5,
                        label=f"{label} — MSE(H)")
            legend_items += 1

    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("SNR (Eb/N0) [dB]")
    ax.set_ylabel("Metric value (log scale)")
    ax.set_title(title)

    # Compact legend in multiple columns if many lines
    ncol = 2 if legend_items > 6 else 1
    ax.legend(loc="best", ncol=ncol, fontsize=9)

    # Footnote
    ax.text(0.01, 0.01,
            "Note: EVM plotted as fraction (EVM% / 100) to share axis with BER/MSE.",
            transform=ax.transAxes, fontsize=8, alpha=0.8)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    # Only attempt to show if an interactive backend is active and explicitly requested.
    if show and os.environ.get("ALLOW_GUI_PLOTS", "") and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ----------------- Convenience wrappers for your runners ----------------- #

def plot_m2_curves(result: Dict, label: Optional[str] = None,
                   title: str = "M2 — BER / EVM / MSE vs SNR",
                   save_path: Optional[str | Path] = None,
                   show: bool = False):
    snr, ber, evm, mse = _arrays_from_m2_result(result)
    curve = {"label": (label or "M2"), "snr_db": snr, "ber": ber}
    if evm is not None: curve["evm_percent"] = evm
    if mse  is not None: curve["mse_H"] = mse
    plot_semilog_ber_evm_mse([curve], title=title, save_path=save_path, show=show)

def plot_ofdm_awgn_ber(result: Dict, label: Optional[str] = None,
                       title: str = "M1 — BER vs SNR (OFDM over AWGN)",
                       save_path: Optional[str | Path] = None,
                       show: bool = False):
    snr, ber = _arrays_from_m1_result(result)
    curve = {"label": (label or "M1"), "snr_db": snr, "ber": ber}
    plot_semilog_ber_evm_mse([curve], title=title, save_path=save_path, show=show)

def plot_multi_from_json(json_paths: List[str | Path],
                         labels: Optional[List[str]] = None,
                         title: str = "BER / EVM / MSE vs SNR (multi-config)",
                         save_path: Optional[str | Path] = None,
                         show: bool = False):
    """
    Load multiple JSON result files (M1 or M2) and overlay them in one semilog plot.
    EVM (if present) is drawn as fraction; M1 files will only contribute BER.
    """
    curves = []
    for i, p in enumerate(json_paths):
        res = load_result_json(p)
        lab = labels[i] if labels and i < len(labels) else Path(str(p)).stem

        if "ber_curve" in res:  # M1
            snr, ber = _arrays_from_m1_result(res)
            curves.append({"label": lab, "snr_db": snr, "ber": ber})
        else:                   # M2
            snr, ber, evm, mse = _arrays_from_m2_result(res)
            c = {"label": lab, "snr_db": snr, "ber": ber}
            if evm is not None: c["evm_percent"] = evm
            if mse  is not None: c["mse_H"] = mse
            curves.append(c)

    plot_semilog_ber_evm_mse(curves, title=title, save_path=save_path, show=show)

# ----------------------------- S2: BLER plot ----------------------------- #
def plot_bler_curve(result: Dict, label: Optional[str] = None, title: str = "S2 — BLER vs SNR",
                    save_path: Optional[str] = None, show: bool = False) -> None:
    snr   = np.asarray(result.get("snr_db", []), dtype=float)
    bler  = np.asarray(result.get("bler", []), dtype=float)
    sent  = np.asarray(result.get("packets", []), dtype=float)   # <- added

    # Dynamic floor: ~ 1 / packets_per_snr (fallback to 1e-6 if unknown)
    if sent.size > 0 and np.any(sent > 0):
        floors    = np.where(sent > 0, 1.0 / sent, np.nan)
        min_floor = float(np.nanmin(floors))
        if not np.isfinite(min_floor):
            min_floor = 1e-6
    else:
        min_floor = 1e-6

    point_floor = np.where(sent > 0, 1.0 / sent, 1e-6)
    y_plot = np.maximum(bler, point_floor if point_floor.size == bler.size else 1e-6)


    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    ax.semilogy(snr, y_plot, marker="o", linewidth=1.8, markersize=5,
                label=(label or "TB+CRC"))

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLER (TB error rate)")
    ax.set_title(title)

    # Show the floor in the legend if available
    if np.isfinite(min_floor):
        ax.legend(loc="best", fontsize=9, title=f"Floor ≈ {min_floor:.2e}")
    else:
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


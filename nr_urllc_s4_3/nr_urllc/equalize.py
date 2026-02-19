# nr_urllc/equalize.py
import numpy as np
from typing import Optional

__all__ = [
    "zf",
    "mmse",
    "equalize_zf_robust",
    "equalize_mmse_robust",
    "mmse_equalize",
]

def zf(Y: np.ndarray, H_est: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Classic Zero-Forcing equalizer: Y_eq = Y / H_est.
    Adds small epsilon to avoid division by zero; returns complex64.
    """
    denom = H_est.astype(np.complex64) + np.complex64(eps)
    return (Y.astype(np.complex64) / denom).astype(np.complex64)


def mmse(Y: np.ndarray, H_est: np.ndarray, noise_var: Optional[float], eps: float = 1e-9) -> np.ndarray:
    """
    Diagonal MMSE equalizer per RE (fallback to ZF if noise_var is None).
      G = H* / (|H|^2 + sigma2)
    """
    if noise_var is None:
        return zf(Y, H_est, eps=eps)
    sigma2 = float(max(0.0, noise_var))
    denom = (np.abs(H_est)**2 + sigma2).astype(np.float32)
    denom = np.maximum(denom, eps)
    G = np.conjugate(H_est).astype(np.complex64) / denom
    return (Y.astype(np.complex64) * G).astype(np.complex64)


def equalize_zf_robust(Y: np.ndarray, H_est: np.ndarray, eps: float = 1e-9, floor: float = 1e-3) -> np.ndarray:
    """
    ROBUST Zero-Forcing equalizer with stronger numerical protection.
    - Uses a magnitude floor; deep fades avoid noisy inversion (pass-through).
    """
    Yc = Y.astype(np.complex64)
    Hc = H_est.astype(np.complex64)

    H_mag = np.abs(Hc)
    strong = H_mag > max(eps, floor)

    Y_eq = np.zeros_like(Yc, dtype=np.complex64)
    # ZF where channel is strong
    Y_eq[strong] = (Yc[strong] / (Hc[strong] + np.complex64(eps))).astype(np.complex64)
    # Pass-through on deep fades to avoid infinite noise enhancement
    Y_eq[~strong] = Yc[~strong]
    return Y_eq


def equalize_mmse_robust(Y: np.ndarray, H_est: np.ndarray, noise_var: float, eps: float = 1e-9) -> np.ndarray:
    """
    ROBUST diagonal MMSE equalizer per RE.
      Y_eq = H*(k)/( |H(k)|^2 + sigma2 ) * Y(k)
    """
    Yc = Y.astype(np.complex64)
    Hc = H_est.astype(np.complex64)
    sigma2 = float(max(0.0, noise_var))

    denom = (np.abs(Hc)**2 + sigma2).astype(np.float32)
    denom = np.maximum(denom, eps)

    W = np.conjugate(Hc) / denom  # complex64 / float32 -> complex64
    return (Yc * W).astype(np.complex64)


# --- New: thin wrapper with the exact signature used by simulate.py ---
def mmse_equalize(Y_used: np.ndarray, H_est_used: np.ndarray, sigma2: Optional[float], eps: float = 1e-12) -> np.ndarray:
    """
    Thin wrapper for simulate.py:
      mmse_equalize(Y_used, H_est_used, sigma2) -> Y_eq_used
    Falls back to robust ZF if sigma2 is None.
    """
    if sigma2 is None:
        return equalize_zf_robust(Y_used, H_est_used, eps=eps)
    return equalize_mmse_robust(Y_used, H_est_used, float(sigma2), eps=eps)

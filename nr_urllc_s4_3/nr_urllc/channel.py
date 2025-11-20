
import numpy as np
from typing import Sequence, Tuple, Optional, Dict

# -----------------------------
# AWGN and small helpers
# -----------------------------
def awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator, dtype=np.complex64) -> np.ndarray:
    """Add complex AWGN to a baseband signal given Es/N0 in dB (per complex symbol)."""
    snr_linear = 10 ** (snr_db / 10.0)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / max(snr_linear, 1e-12)
    noise = (rng.normal(0.0, np.sqrt(noise_power/2.0), size=signal.shape)
             + 1j * rng.normal(0.0, np.sqrt(noise_power/2.0), size=signal.shape)).astype(dtype, copy=False)
    return (signal + noise).astype(dtype, copy=False)

def flat_rayleigh(S: int, rng: Optional[np.random.Generator] = None, dtype=np.complex64) -> np.ndarray:
    """Draw one flat fading scalar per OFDM symbol: CN(0,1). Shape [S]."""
    if rng is None:
        rng = np.random.default_rng()
    h = (rng.normal(0.0, 1.0, size=S) + 1j * rng.normal(0.0, 1.0, size=S)) / np.sqrt(2.0)
    return h.astype(dtype, copy=False)

# -----------------------------
# FIR builders
# -----------------------------
def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

def tdl_fir_from_profile(delays: Sequence[int], powers_db: Sequence[float], rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Build a complex FIR from integer-sample delays and per-tap average powers (dB).
    Rayleigh fading taps CN(0, p_n). Returns h of length max(delay)+1.
    """
    rng = _ensure_rng(rng)
    d = np.asarray(delays, dtype=int)
    p_db = np.asarray(powers_db, dtype=float)
    assert d.size == p_db.size and d.size > 0, "delays and powers_db must be same non-zero length"
    L = int(np.max(d)) + 1
    h = np.zeros(L, dtype=np.complex64)
    p_lin = 10.0 ** (p_db / 10.0)
    # Normalize total power to 1 for numerical stability (relative powers preserved)
    if np.sum(p_lin) > 0:
        p_lin = p_lin / np.sum(p_lin)
    for di, pi in zip(d, p_lin):
        tap = (rng.normal(0.0, 1.0) + 1j * rng.normal(0.0, 1.0)) / np.sqrt(2.0)  # CN(0,1)
        h[di] += np.sqrt(pi) * tap
    return h.astype(np.complex64, copy=False)

def apply_fir_per_symbol(x_time: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve each OFDM symbol row with FIR h and crop to input length. x_time: [S, N] -> [S, N]."""
    S, N = x_time.shape
    y = np.zeros_like(x_time, dtype=np.complex64)
    for s in range(S):
        full = np.convolve(x_time[s], h, mode="full").astype(np.complex64, copy=False)
        y[s] = full[:N]
    return y

# -----------------------------
# CDL support (SISO simplification)
# -----------------------------

# A minimal subset of normalized delays and powers from 3GPP TR 38.901 (Tables 7.7.1-1 and 7.7.1-3)
# We only need delays & powers for a SISO-equivalent PDP; angles are irrelevant here.
# Source: ETSI TR 138 901 V16.1.0 (Release 16).
_CDL_PROFILES: Dict[str, Dict[str, list]] = {
    "A": {  # Table 7.7.1-1. CDL-A (normalized delays, power dB)
        "delays_norm": [0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6078, 0.5750, 0.7618,
                        1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5519, 3.0582, 4.0810, 4.4579,
                        4.5695, 4.7966, 5.0066, 5.3043, 9.6586],
        "powers_db":   [-13.4,   0.0,   -2.2,   -4.0,   -6.0,   -8.2,   -9.9,  -10.5,   -7.5,
                        -15.9,  -6.6,  -16.7,  -12.4,  -15.2,  -10.8,  -11.3,  -12.7,  -16.2,
                        -18.3,  -18.9,  -16.6,  -19.9,  -29.7],
    },
    "C": {  # Table 7.7.1-3. CDL-C
        "delays_norm": [0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6484, 0.6560, 0.6584,
                        0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003,
                        5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523],
        "powers_db":   [ -4.4,   -1.2,   -3.5,   -5.2,   -2.5,    0.0,   -2.2,   -3.9,   -7.4,
                         -7.1,  -10.7,  -11.1,   -5.1,   -6.8,   -8.7,  -13.2,  -13.9,  -13.9,
                        -15.8,  -17.1,  -16.0,  -15.7,  -21.6,  -22.8],
    },
    # Profiles D/E include a LOS specular path. For simplicity, we expose a configurable K for the strongest tap.
}

def _cdl_discrete_delays(delays_norm: np.ndarray, ncp: int, scale_samples: Optional[int] = None) -> np.ndarray:
    """
    Map normalized delays to discrete sample delays such that the maximum delay fits within CP.
    If scale_samples is provided, use that directly; otherwise choose scale so max delay <= 0.8 * Ncp.
    """
    max_norm = float(np.max(delays_norm))
    if max_norm <= 0:
        scale = 1
    elif scale_samples is not None and scale_samples > 0:
        scale = int(scale_samples)
    else:
        scale = max(1, int(np.floor(0.8 * max(ncp, 1) / max_norm)))
    d_samp = np.round(delays_norm * scale).astype(int)
    return d_samp

def cdl_fir(profile: str = "C",
            ncp: int = 16,
            scale_samples: Optional[int] = None,
            rice_k_db: Optional[float] = None,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Build a SISO-equivalent CDL FIR (Rayleigh taps) using normalized delays/powers (Tables 7.7.1-x).
    - profile: one of {'A','C'} for now (extendable)
    - ncp: cyclic prefix length, used to auto-scale delays to fit within CP (unless scale_samples is given)
    - scale_samples: explicitly scale normalized delays by this integer factor (overrides ncp-based scaling)
    - rice_k_db: if set, the strongest tap becomes Ricean with K (dB). Others remain Rayleigh.
    """
    rng = _ensure_rng(rng)
    key = str(profile).upper()
    if key not in _CDL_PROFILES:
        raise ValueError(f"Unsupported CDL profile '{profile}'. Available: {list(_CDL_PROFILES.keys())}")
    delays_norm = np.asarray(_CDL_PROFILES[key]["delays_norm"], dtype=float)
    powers_db   = np.asarray(_CDL_PROFILES[key]["powers_db"], dtype=float)

    # Scale to discrete delays
    d = _cdl_discrete_delays(delays_norm, ncp=int(ncp), scale_samples=scale_samples)

    # Convert powers to linear and normalize to sum=1
    p_lin = 10.0 ** (powers_db / 10.0)
    p_lin = p_lin / np.sum(p_lin)

    # Build taps
    L = int(np.max(d)) + 1
    h = np.zeros(L, dtype=np.complex64)

    # Strongest path index (for optional Ricean)
    strongest = int(np.argmax(p_lin))
    K_lin = None if rice_k_db is None else 10.0 ** (float(rice_k_db) / 10.0)

    for i, (di, pi) in enumerate(zip(d, p_lin)):
        if K_lin is not None and i == strongest:
            # Specular + diffuse: scale such that E|h|^2 = pi and K = Ps/Pn
            Ps = pi * (K_lin / (K_lin + 1.0))
            Pn = pi * (1.0 / (K_lin + 1.0))
            spec = np.sqrt(Ps) * 1.0  # deterministic LOS phasor with phase 0; could randomize a global phase
            diff = (rng.normal(0.0, 1.0) + 1j * rng.normal(0.0, 1.0)) / np.sqrt(2.0)
            tap = spec + np.sqrt(Pn) * diff
        else:
            diff = (rng.normal(0.0, 1.0) + 1j * rng.normal(0.0, 1.0)) / np.sqrt(2.0)
            tap = np.sqrt(pi) * diff
        h[di] += tap.astype(np.complex64)

    return h.astype(np.complex64, copy=False)

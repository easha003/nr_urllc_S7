import numpy as np
from typing import Tuple

def generate_pilots(n_subcarriers: int, seed: int = 0, dtype=np.complex64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    const = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    return rng.choice(const, size=n_subcarriers).astype(dtype)

# --- M2 helpers ---
def comb_indices(K: int, spacing: int, offset: int = 0) -> np.ndarray:
    """Return 0-based pilot column indices for a comb pattern across frequency.
    Example: K=64, spacing=4 -> [0,4,8,...,60] (then shifted by offset mod K).
    """
    if spacing <= 0:
        raise ValueError("spacing must be >=1")
    base = np.arange(0, K, spacing)
    return ((base + offset) % K).astype(int)


def place(
    data_grid: np.ndarray,
    spacing: int,
    *,
    offset: int = 0,
    seed: int = 0,
    power_boost_db: float = 0.0,
    power_mode: str = "unconstrained",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Insert QPSK pilots into a [S,K] data grid using a comb pattern.

    Returns:
        tx_grid:   [S,K] grid with pilots inserted
        pilot_mask:[S,K] True at pilot locations
        pilot_vals:[S,K] pilots where mask=True, 0 elsewhere
        data_Es:   per-RE energy of DATA tones after any renormalization
        pilot_Es:  per-RE energy of PILOT tones after any renormalization
    """
    S, K = data_grid.shape
    cols = comb_indices(K, spacing, offset)
    rng = np.random.default_rng(seed)

    # Pilot boost (power)
    b_lin = float(10 ** (power_boost_db / 10.0))
    amp   = float(np.sqrt(b_lin))
    const = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    P     = rng.choice(const, size=(S, cols.size)).astype(np.complex64) * amp

    tx_grid = data_grid.astype(np.complex64, copy=True)
    tx_grid[:, cols] = P

    pilot_mask = np.zeros((S, K), dtype=bool)
    pilot_mask[:, cols] = True

    pilot_vals = np.zeros((S, K), dtype=np.complex64)
    pilot_vals[:, cols] = P

    # Power-constrained: keep average per-RE power = 1
    if str(power_mode).lower() == "constrained":
        Kp = int(pilot_mask.sum())
        Kd = int(S * K - Kp)
        P_avg = (Kd * 1.0 + Kp * b_lin) / float(S * K) if (S * K) > 0 else 1.0
        scale = 1.0 / np.sqrt(P_avg)
        tx_grid   *= scale
        pilot_vals*= scale
        data_Es   = float(scale**2)
        pilot_Es  = float(b_lin * scale**2)
    else:
        data_Es  = 1.0
        pilot_Es = float(b_lin)

    return tx_grid, pilot_mask, pilot_vals, data_Es, pilot_Es



def estimate_ls(Y_used: np.ndarray, pilot_vals: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Least-squares channel estimate on pilot bins: H = Y/X.
    Returns H_pilot (NaN elsewhere) shaped [S,K].
    """
    eps = 1e-12
    H = np.full_like(Y_used, np.nan, dtype=np.complex64)
    Xp = pilot_vals[pilot_mask]
    Yp = Y_used[pilot_mask]
    H[pilot_mask] = Yp / (Xp + eps)
    return H


def _interp1_complex(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolate complex data by applying np.interp to real & imag separately."""
    # Guard: if only one pilot, hold-constant
    if fp.size == 1:
        return np.full(x.shape, fp[0], dtype=np.complex64)
    r = np.interp(x, xp, fp.real)
    i = np.interp(x, xp, fp.imag)
    return (r + 1j * i).astype(np.complex64)


def interp_freq(H_pilot: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Frequency-only linear interpolation of H across K used tones, per OFDM symbol.
    For each row s: interpolate known pilot columns → fill all K columns.
    Edges are held constant (np.interp behavior).
    """
    S, K = H_pilot.shape
    k = np.arange(K)
    H_est = np.zeros_like(H_pilot, dtype=np.complex64)
    for s in range(S):
        cols = np.flatnonzero(pilot_mask[s])
        if cols.size == 0:
            raise ValueError("No pilots in row; cannot interpolate")
        Hp = H_pilot[s, cols]
        H_est[s] = _interp1_complex(cols.astype(float), Hp.astype(np.complex64), k.astype(float))
    return H_est


def data_mask_from_pilots(pilot_mask: np.ndarray) -> np.ndarray:
    """Return boolean mask for data (non-pilot) REs."""
    return ~pilot_mask

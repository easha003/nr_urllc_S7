# nr_urllc/utils.py
import numpy as np
from typing import Union, Sequence
from math import erf, erfc

def get_rng(seed: int | None):
    """Return a reproducible Generator with a seed safely coerced to uint32.
    Accepts arbitrary inputs (None, int, float, str); clamps to [0, 2**32-1].
    """
    if seed is None:
        return np.random.default_rng()
    try:
        s = int(seed)
    except Exception:
        # Fallback: hash arbitrary inputs deterministically
        s = abs(hash(str(seed)))
    # Clamp to valid range for SeedSequence/PCG64
    s = int(s) % (2**32 - 1)
    return np.random.default_rng(np.uint32(s))


def complex_exp(theta: Union[float, np.ndarray, Sequence],
                dtype: np.dtype | None = None) -> np.ndarray:
    """
    Vectorized complex exponential: z = e^{j * theta}.
    If dtype is omitted:
      - float32 input -> complex64 (matches test)
      - otherwise     -> complex128
    """
    theta_arr = np.asarray(theta)
    if dtype is None:
        dtype = np.complex64 if theta_arr.dtype == np.float32 else np.complex128
    return np.exp(1j * theta_arr).astype(dtype, copy=False)



# ------------------------- Modulation / Demodulation ------------------------ #

def mod(bits: np.ndarray, M: int) -> np.ndarray:
    """
    Map bits -> constellation symbols with unit average energy.
    Supports QPSK (M=4) and 16QAM (M=16). Returns complex64.
    - QPSK Gray: 00→+1+1j, 01→-1+1j, 11→-1-1j, 10→+1-1j (normalized by √2)
    - 16QAM Gray (square): I/Q levels ∈ {+3,+1,-1,-3} (normalized by √10)
    """

    bits = np.asarray(bits).astype(np.int8, copy=False)
    if M not in (4, 16):
        raise ValueError(f"Unsupported M={M}; supported: 4 (QPSK), 16 (16QAM).")

    k = 2 if M == 4 else 4  # bits per symbol
    # If length not multiple of k, pad one symbol with zeros (documented behavior)
    pad = (-len(bits)) % k
    if pad:
        bits = np.pad(bits, (0, pad), constant_values=0)

    if M == 4:
        b = bits.reshape(-1, 2)
        # Vectorized Gray mapping via boolean/ints (faster than dict lookup)
        i = 1 - 2*b[:, 1]
        q = 1 - 2*b[:, 0]
        syms = (i + 1j*q) / np.sqrt(2.0)
        return syms.astype(np.complex64, copy=False)

        # M == 16
    b = bits.reshape(-1, 4).astype(np.int8)
    # Gray -> binary index: idx = 2*g1 + (g1 ^ g0)
    levels = np.array([+3, +1, -1, -3], dtype=np.float32)

    gi1, gi0 = b[:, 0], b[:, 1]  # I MSB, I LSB (Gray)
    gq1, gq0 = b[:, 2], b[:, 3]  # Q MSB, Q LSB (Gray)

    idx_i = (gi1 << 1) | (gi1 ^ gi0)
    idx_q = (gq1 << 1) | (gq1 ^ gq0)

    i = levels[idx_i]
    q = levels[idx_q]

    syms = (i + 1j*q) / np.sqrt(10.0)  # unit Es
    return syms.astype(np.complex64, copy=False)




def demod(symbols: np.ndarray, M: int) -> np.ndarray:
    """
    Hard-decision demodulation: nearest neighbor.
    Returns bit array of shape [N_symbols * log2(M)].
    """
    symbols = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    M = int(M)

    if M == 4:  # QPSK
        ref = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
        bits_ref = np.array([[0,0],[0,1],[1,1],[1,0]])
    elif M == 16:  # 16QAM
        levels = np.array([+3, +1, -1, -3])
        ref = []
        bits_ref = []
        # Gray code mapping as in mod()
        def level2bits(val):
            if val == +3: return [0,0]
            if val == +1: return [0,1]
            if val == -1: return [1,1]
            if val == -3: return [1,0]
        for i in levels:
            for q in levels:
                ref.append(i + 1j*q)
                bits_ref.append(level2bits(i) + level2bits(q))
        ref = np.array(ref, dtype=np.complex64) / np.sqrt(10.0)
        bits_ref = np.array(bits_ref, dtype=int)
    else:
        raise ValueError(f"Unsupported M={M}")

    # nearest-neighbor detection
    d2 = abs(symbols.reshape(-1,1) - ref.reshape(1,-1))**2
    idx = d2.argmin(axis=1)
    return bits_ref[idx].reshape(-1)


# --- utils.py (append) -------------------------------------------------------

def ebn0_db_to_sigma_sc(ebn0_db: float, M: int = 4, code_rate: float = 1.0, Es_sym: float = 1.0) -> float:
    """
    Single-carrier AWGN sigma for complex noise n ~ CN(0, sigma_c^2).
    Assumes modulation outputs unit-variance constellation scaled so E[|S|^2] = Es_sym.
      Eb/N0 -> Es/N0 via Es = Eb * log2(M) * code_rate
      sigma_c^2 (per complex sample) = Es_sym / (Es/N0)
    Returns sigma_RI for np.random.normal(0, sigma_RI) on each real/imag (so sigma_RI^2 * 2 = sigma_c^2).
    """
    EsN0_lin = (10.0 ** (ebn0_db / 10.0)) * np.log2(M) * code_rate
    sigma_c2 = Es_sym / EsN0_lin
    sigma_RI = np.sqrt(sigma_c2 / 2.0)
    return sigma_RI

def ebn0_db_to_sigma_ofdm_time(ebn0_db: float,
                               M: int = 4,
                               code_rate: float = 1.0,
                               nfft: int = 256,
                               ifft_norm: str = "numpy",
                               Es_sub: float = 1.0) -> float:
    """
    Time-domain AWGN sigma for OFDM when you add noise BEFORE CP removal and FFT.
    Assumptions:
      - Constellation on used subcarriers has per-subcarrier symbol energy Es_sub (≈1 if normalized).
      - np.fft.ifft uses 1/N scaling (default 'numpy'), and np.fft.fft uses no scaling.
      - With 'unitary' (if you used orthonormal FFT/IFFT), per-bin variance is preserved.

    We want per-used-subcarrier Es/N0 to equal target after FFT:
      EsN0_lin = (Eb/N0)_lin * log2(M) * code_rate
      Desired freq-domain noise variance per bin: sigma_f^2 = Es_sub / EsN0_lin
      Mapping time <-> freq:
        - 'numpy' scaling: var(freq) = nfft * var(time)
        - 'unitary'      : var(freq) = var(time)

      So:
        sigma_t^2 = sigma_f^2 / nfft   (numpy)
        sigma_t^2 = sigma_f^2          (unitary)

      Return sigma_RI for np.random.normal on real/imag (CN with var = 2*sigma_RI^2).
    """
    EsN0_lin = (10.0 ** (ebn0_db / 10.0)) * np.log2(M) * code_rate
    sigma_f2 = Es_sub / EsN0_lin
    if ifft_norm == "numpy":
        sigma_t2 = sigma_f2 / float(nfft)
    elif ifft_norm == "unitary":
        sigma_t2 = sigma_f2
    else:
        raise ValueError("ifft_norm must be 'numpy' or 'unitary'")
    sigma_RI = np.sqrt(sigma_t2 / 2.0)
    return sigma_RI

def _erfc_vec(x):
    """Elementwise erfc that works on scalars or NumPy arrays (no SciPy needed)."""
    x = np.asarray(x, dtype=float)
    return np.vectorize(erfc)(x).astype(float)

def qfunc(x):
    """Q(x) = 0.5 * erfc(x / sqrt(2)) — array-safe."""
    x = np.asarray(x, dtype=float)
    return 0.5 * _erfc_vec(x / np.sqrt(2.0))

def ber_qpsk_theory(ebn0_db):
    """QPSK in AWGN: Pb = 0.5 * erfc(sqrt(Eb/N0)) — array-safe."""
    g = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)  # Eb/N0 (linear)
    return 0.5 * _erfc_vec(np.sqrt(g))

# (Optional) keep this name for backward compatibility if other code imports it
def erfc_vec(x):
    return _erfc_vec(x)
# ---------------------------------------------------------------------------

# --- Soft LLRs for Gray square QAM (max-log) ---

def _gray_bits(M):
    m = int(np.log2(M))
    vals = np.arange(M, dtype=int) ^ (np.arange(M, dtype=int) >> 1)
    return ((vals[:, None] >> np.arange(m)[::-1]) & 1).astype(np.uint8)

def _qam_constellation(M):
    if M == 2:  # BPSK
        const = np.array([-1.0, +1.0]) + 0j
        Es = 1.0
        bits = _gray_bits(2)
        return const.astype(np.complex128), bits
    m = int(np.log2(M))
    K = 2 ** (m // 2)
    pam = np.arange(K) * 2 - (K - 1)              # ...,-3,-1,1,3,...
    g = np.arange(K) ^ (np.arange(K) >> 1)        # Gray reorder
    pam_g = pam[g]
    xv, yv = np.meshgrid(pam_g, pam_g[::-1])
    const = (xv + 1j * yv).reshape(-1)
    Es = np.mean(np.abs(const) ** 2)
    const = const / np.sqrt(Es)                   # normalize Es=1
    bits = _gray_bits(M)
    return const.astype(np.complex128), bits

# nr_urllc/utils.py
def qam_llr_maxlog(sym_eq: np.ndarray, M: int, sigma2: float) -> np.ndarray:
    """
    Max-log LLRs per coded bit, aligned to utils.mod() bit ordering.
    QPSK order: [Q-bit, I-bit]
    16QAM order: [I_MSB, I_LSB, Q_MSB, Q_LSB]
    sigma2: noise variance per complex symbol (post-FFT/equalization)
    """
    y = np.asarray(sym_eq, dtype=np.complex128).reshape(-1)
    s2 = float(max(sigma2, 1e-15))

    if M == 4:
        # QPSK: b0 controls Q, b1 controls I in utils.mod
        # LLR(b0) ~ (2/sigma2) * Im{y}, LLR(b1) ~ (2/sigma2) * Re{y}
        scale = 2.0 / s2
        Lq = y.imag * scale
        Li = y.real * scale
        return np.stack([Lq, Li], axis=1).reshape(-1)

    if M == 16:
        # 4-PAM levels used in utils.mod (normalized by sqrt(10))
        levels = np.array([+3.0, +1.0, -1.0, -3.0]) / np.sqrt(10.0)

        def pam_llr(x):
            x = x.astype(np.float64)
            # MSB (sign): positive set {+3,+1} vs negative set {-1,-3}
            d_pos = np.minimum((x - levels[0])**2, (x - levels[1])**2)
            d_neg = np.minimum((x - levels[2])**2, (x - levels[3])**2)
            L_msb = (d_neg - d_pos) / (2.0 * s2)
            # LSB (magnitude): outer {±3} vs inner {±1}
            d_outer = np.minimum((x - levels[0])**2, (x - levels[3])**2)
            d_inner = np.minimum((x - levels[1])**2, (x - levels[2])**2)
            L_lsb = (d_inner - d_outer) / (2.0 * s2)
            return L_msb, L_lsb

        Li_msb, Li_lsb = pam_llr(y.real)  # I axis
        Lq_msb, Lq_lsb = pam_llr(y.imag)  # Q axis
        return np.stack([Li_msb, Li_lsb, Lq_msb, Lq_lsb], axis=1).reshape(-1)

    # If you later add higher QAMs, extend similarly to match utils.mod’s bit order.
    raise ValueError(f"Unsupported M={M}; supported: 4 (QPSK), 16 (16QAM).")


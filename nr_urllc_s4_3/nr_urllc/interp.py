import numpy as np

def interp_freq_linear(H_p: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    S, K = H_p.shape
    H_est = np.zeros((S, K), dtype=np.complex64)
    xs = np.arange(K)
    for s in range(S):
        m = pilot_mask[s]
        Hrow = H_p[s].astype(np.complex64, copy=False)
        n = int(m.sum())
        if n == 0:
            H_est[s] = 1.0 + 0.0j
        elif n == 1:
            H_est[s] = np.where(m, Hrow, Hrow[m][0]).astype(np.complex64)
        else:
            H_est[s] = (np.interp(xs, xs[m], Hrow[m].real)
                        + 1j*np.interp(xs, xs[m], Hrow[m].imag)).astype(np.complex64)
    return H_est

def smooth_time_triangular(H: np.ndarray) -> np.ndarray:
    # [S,K] → apply 3-tap [0.25, 0.5, 0.25] along S with edge handling
    S, K = H.shape
    out = np.empty_like(H, dtype=np.complex64)
    if S == 1:
        return H.astype(np.complex64)
    out[0]   = (0.75*H[0] + 0.25*H[1]).astype(np.complex64)
    for s in range(1, S-1):
        out[s] = (0.25*H[s-1] + 0.5*H[s] + 0.25*H[s+1]).astype(np.complex64)
    out[-1]  = (0.25*H[-2] + 0.75*H[-1]).astype(np.complex64)
    return out

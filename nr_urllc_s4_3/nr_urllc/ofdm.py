# nr_urllc/ofdm.py
import numpy as np

"""
OFDM conventions:
- FFT/IFFT: NumPy (IFFT has 1/N scaling; FFT has none)
- Used-tone plan: center-aligned, DC skipped for even K
- TX accepts [S,K] (used tones) or [S,Nfft] (full grid)
- RX returns [S,K] by default; set return_full_grid=True for [S,Nfft]
"""


def get_used_bins(nfft: int, n_used: int, skip_dc: bool = True) -> np.ndarray:
    """
    Return positive indices (0..nfft-1) for a center-aligned allocation of n_used tones.
    For even n_used and skip_dc=True, bins are symmetric around DC excluding 0.
    Example (nfft=256, n_used=64): [-32..-1, +1..+32] -> modulo nfft.
    """
    if n_used % 2 != 0 and skip_dc:
        raise ValueError("Use an even n_used when skip_dc=True")
    half = n_used // 2
    if skip_dc:
        bins = np.r_[np.arange(-half, 0), np.arange(1, half + 1)]
    else:
        # include DC; place floor(n_used/2) below and ceil above
        below = np.arange(-half, 0)
        above = np.arange(0, n_used - len(below))
        bins = np.r_[below, above]
    return (bins + nfft) % nfft


def tx(grid: np.ndarray,
       nfft: int,
       cp: float,
       n_subcarriers: int | None = None,
       used_bins: np.ndarray | None = None) -> np.ndarray:
    """
    OFDM TX.
    grid:
      - shape [S, nfft] (full grid): used as-is
      - shape [S, K]    (used tones): will be embedded at 'used_bins' (or default centered)
    Returns time-domain with CP: shape [S, nfft + Ncp]
    """
    grid = np.asarray(grid)
    S = grid.shape[0]
    Ncp = int(round(cp * nfft))

    if grid.shape[1] == nfft:
        # full grid path
        full = grid.astype(np.complex64, copy=False)
    else:
        # used-subcarrier path
        K = grid.shape[1]
        if used_bins is None:
            used_bins = get_used_bins(nfft, n_used=K, skip_dc=True)
        else:
            used_bins = np.asarray(used_bins)
            assert used_bins.size == K
        full = np.zeros((S, nfft), dtype=np.complex64)
        full[:, used_bins] = grid.astype(np.complex64, copy=False)

    # NumPy IFFT has 1/N scaling; FFT has no scaling.
    x_no_cp = np.fft.ifft(full, n=nfft, axis=1).astype(np.complex64, copy=False)

    if Ncp > 0:
        cp_part = x_no_cp[:, -Ncp:]
        x = np.concatenate([cp_part, x_no_cp], axis=1)
    else:
        x = x_no_cp
    return x


def rx(x: np.ndarray,
       nfft: int,
       cp: float,
       n_subcarriers: int | None = None,
       used_bins: np.ndarray | None = None,
       return_full_grid: bool = False) -> np.ndarray:
    """
    OFDM RX.
    Removes CP, FFTs, and returns either:
      - full grid [S, nfft] if return_full_grid=True
      - used tones [S, K]   (with 'used_bins' or inferred from 'n_subcarriers') otherwise
    """
    x = np.asarray(x)
    S = x.shape[0]
    Ncp = int(round(cp * nfft))
    if Ncp > 0:
        x = x[:, Ncp:Ncp + nfft]
    else:
        x = x[:, :nfft]

    Y = np.fft.fft(x, n=nfft, axis=1).astype(np.complex64, copy=False)

    if return_full_grid:
        return Y

    if used_bins is None:
        if n_subcarriers is None:
            raise ValueError("Either used_bins or n_subcarriers must be provided when return_full_grid=False")
        used_bins = get_used_bins(nfft, n_used=int(n_subcarriers), skip_dc=True)
    else:
        used_bins = np.asarray(used_bins)

    return Y[:, used_bins]

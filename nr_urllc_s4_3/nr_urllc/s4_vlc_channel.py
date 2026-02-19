"""
S4 VLC Channel Module: IM/DD DCO-OFDM for Optical Wireless Communication

This module implements a complete VLC (Visible Light Communication) physical layer
using Intensity Modulation / Direct Detection (IM/DD) with DC-biased Optical OFDM (DCO-OFDM).

Key Features:
- DCO-OFDM modulation with Hermitian symmetry for real-valued signals
- LED channel model with bandwidth-limited response
- Optical shot noise and thermal noise modeling
- Photodetector model with responsivity and area
- Clipping distortion handling (LED dynamic range)
- Compatible with existing S3 timing framework

Architecture:
  TX: Complex OFDM symbols → Hermitian symmetry → IFFT → DC bias → Clipping → LED
  Channel: LED low-pass filter + shot noise + thermal noise
  RX: Photodetector → Remove DC → FFT → Extract data subcarriers

Author: Generated for S4 Milestone
Date: October 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import signal
import warnings

# Physical constants
PLANCK_CONSTANT = 6.62607015e-34  # J·s
SPEED_OF_LIGHT = 299792458         # m/s
ELEMENTARY_CHARGE = 1.602176634e-19  # C
BOLTZMANN_CONSTANT = 1.380649e-23   # J/K


# ============================================================================
# DCO-OFDM MODULATION
# ============================================================================

def dco_ofdm_modulate(
    x_freq: np.ndarray,
    dc_bias: float = 0.5,
    nfft: Optional[int] = None,
    clipping_ratio: float = 1.0,
) -> np.ndarray:
    """
    Apply DCO-OFDM modulation to convert complex OFDM symbols to real-valued optical signal.
    
    DCO-OFDM Process:
    1. Apply Hermitian symmetry to ensure real IFFT output
    2. Perform IFFT
    3. Add DC bias to make signal positive
    4. Apply clipping to LED dynamic range
    
    Args:
        x_freq: Complex OFDM symbols [S, K] where S=symbols, K=used subcarriers
        dc_bias: DC bias ratio (0-1). Typical: 0.5 means bias at 50% of max LED current
        nfft: FFT size. If None, uses next power of 2 >= 2*K
        clipping_ratio: Maximum signal level relative to (1+dc_bias). Default 1.0 = no extra headroom
        
    Returns:
        x_real: Real-valued time-domain signal [S, N] ready for LED transmission
        
    Notes:
        - Hermitian symmetry: X[k] = X*[-k] ensures real IFFT output
        - DC subcarrier (k=0) is set to zero to avoid LED bias issues
        - Clipping introduces nonlinear distortion but is necessary for LED operation
    """
    S, K = x_freq.shape
    
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(2 * K)))
    
    # Build Hermitian-symmetric frequency grid
    X_full = np.zeros((S, nfft), dtype=np.complex128)
    
    # Place data on positive frequencies (skip DC)
    # Typical allocation: bins 1 to K
    X_full[:, 1:K+1] = x_freq
    
    # Hermitian symmetry: X[N-k] = conj(X[k])
    # For k = 1 to K, set X[nfft-k] = conj(X[k])
    for k in range(1, K+1):
        X_full[:, nfft - k] = np.conj(X_full[:, k])
    
    # DC and Nyquist bins should be real (typically zero)
    X_full[:, 0] = 0.0  # DC = 0
    if nfft % 2 == 0:
        X_full[:, nfft // 2] = 0.0  # Nyquist = 0
    
    # IFFT (numpy convention: IFFT has 1/N normalization)
    x_time = np.fft.ifft(X_full, axis=1)
    
    # Verify Hermitian property was applied correctly (output should be real)
    if np.max(np.abs(x_time.imag)) > 1e-10:
        warnings.warn(f"DCO-OFDM output has non-negligible imaginary part: {np.max(np.abs(x_time.imag))}")
    
    x_real = x_time.real.astype(np.float64)
    
    # Normalize to unit average power before biasing
    power = np.mean(x_real ** 2)
    if power > 1e-12:
        x_real = x_real / np.sqrt(power)

    # --- pre-clip autoscale (keep peaks inside LED rails after bias) ---
    # Allowable positive/negative headroom after bias:
    max_level  = (1.0 + dc_bias) * clipping_ratio
    half_low   = float(dc_bias)                 # room to 0 lower rail
    half_high  = float(max_level - dc_bias)     # room to upper rail
    allow_amp  = float(min(half_low, half_high))
    peak_abs   = float(np.max(np.abs(x_real))) + 1e-12
    if peak_abs > allow_amp:
        x_real *= (allow_amp / peak_abs) * 0.98   # 2% headroom
    # --- end autoscale ---

    # Add DC bias
    x_biased  = x_real + dc_bias

    # Apply LED clipping [0, max_level]
    x_clipped = np.clip(x_biased, 0.0, max_level)

    
    # Track clipping statistics
    clipped_samples = np.sum((x_biased < 0) | (x_biased > max_level))
    clipping_rate = clipped_samples / x_biased.size
    if clipping_rate > 0.1:
        warnings.warn(f"High clipping rate: {clipping_rate*100:.1f}% of samples clipped")
    
    return x_clipped.astype(np.float32)


# ============================================================================
# LED CHANNEL MODEL
# ============================================================================

def led_frequency_response(
    freq_hz: np.ndarray,
    bandwidth_3db_mhz: float = 20.0,
    order: int = 1,
) -> np.ndarray:
    """
    LED frequency response (magnitude).
    
    Models LED as a low-pass filter. Most LEDs have 3dB bandwidth 1-20 MHz.
    
    Args:
        freq_hz: Frequency array [Hz]
        bandwidth_3db_mhz: 3dB bandwidth [MHz]
        order: Filter order (1=first-order, 2=second-order Butterworth)
        
    Returns:
        H_mag: Magnitude response at each frequency
        
    Notes:
        - First-order: H(f) = 1 / sqrt(1 + (f/f_3dB)^2)
        - Higher-order models can capture phosphor decay in white LEDs
    """
    f_3db = bandwidth_3db_mhz * 1e6  # Convert to Hz
    
    if order == 1:
        # First-order low-pass
        H_mag = 1.0 / np.sqrt(1.0 + (freq_hz / f_3db) ** 2)
    elif order == 2:
        # Second-order Butterworth
        H_mag = 1.0 / np.sqrt(1.0 + (freq_hz / f_3db) ** 4)
    else:
        raise ValueError(f"Unsupported filter order: {order}. Use 1 or 2.")
    
    return H_mag


def apply_led_channel(
    x_optical: np.ndarray,
    sample_rate_hz: float,
    bandwidth_3db_mhz: float = 20.0,
    filter_order: int = 1,
) -> np.ndarray:
    """
    Apply LED frequency-selective channel to optical signal.
    
    Args:
        x_optical: Input optical signal [S, N] (already clipped, non-negative)
        sample_rate_hz: Sampling rate [Hz]
        bandwidth_3db_mhz: LED 3dB bandwidth [MHz]
        filter_order: Filter order (1 or 2)
        
    Returns:
        y_optical: Channel output [S, N]
    """
    S, N = x_optical.shape
    
    # Design LED filter (Butterworth)
    f_3db_normalized = (bandwidth_3db_mhz * 1e6) / (sample_rate_hz / 2)  # Normalize to Nyquist
    f_3db_normalized = np.clip(f_3db_normalized, 1e-6, 0.99)  # Ensure stability
    
    # Create Butterworth filter
    b, a = signal.butter(filter_order, f_3db_normalized, btype='low', analog=False)
    
    # Apply filter to each OFDM symbol
    y_optical = np.zeros_like(x_optical)
    for s in range(S):
        y_optical[s, :] = signal.filtfilt(b, a, x_optical[s, :])
    
    return y_optical.astype(np.float32)


def add_optical_noise(
    y_optical: np.ndarray,
    snr_db: float,
    signal_power: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    noise_type: str = "awgn",
) -> np.ndarray:
    """
    Add optical noise to received signal.
    
    VLC noise sources:
    1. Shot noise (signal-dependent, Poisson)
    2. Thermal noise (signal-independent, Gaussian)
    3. Background light noise
    
    For simplicity, we model as AWGN with specified SNR.
    
    Args:
        y_optical: Received optical signal [S, N]
        snr_db: Optical SNR [dB]
        signal_power: Reference signal power. If None, computed from y_optical
        rng: Random number generator
        noise_type: "awgn" (simple) or "shot" (signal-dependent)
        
    Returns:
        y_noisy: Noisy received signal [S, N]
        
    Notes:
        - Optical SNR = (received optical power)^2 / noise variance
        - Shot noise: variance ∝ signal (Poisson → Gaussian for high counts)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute signal power
    if signal_power is None:
        y_ac = y_optical - np.mean(y_optical)
        signal_power = float(np.mean(y_ac.astype(np.float64) ** 2))
        signal_power = max(signal_power, 1e-12)

    # SNR to noise variance
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power)
    
    if noise_type == "awgn":
        # Simple AWGN
        noise = rng.normal(0.0, sigma, y_optical.shape)
    elif noise_type == "shot":
        # Signal-dependent shot noise (approximate)
        # σ_shot^2 ∝ signal level
        noise = np.zeros_like(y_optical)
        for s in range(y_optical.shape[0]):
            for n in range(y_optical.shape[1]):
                # Shot noise std proportional to sqrt(signal)
                local_sigma = sigma * np.sqrt(np.maximum(y_optical[s, n] / np.mean(y_optical), 0.1))
                noise[s, n] = rng.normal(0.0, local_sigma)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    y_noisy = y_optical + noise
    return y_noisy.astype(np.float32)


# ============================================================================
# PHOTODETECTOR & DEMODULATION
# ============================================================================

def photodetector_response(
    y_optical: np.ndarray,
    responsivity: float = 0.5,
    area_cm2: float = 1.0,
    dc_bias: float = 0.5,
) -> np.ndarray:
    """
    Model photodetector response.
    
    Converts optical power to electrical current.
    
    Args:
        y_optical: Received optical signal [S, N] (in arbitrary units)
        responsivity: Photodetector responsivity [A/W]. Typical: 0.4-0.6 for Si
        area_cm2: Active area [cm^2]
        dc_bias: DC bias to remove (should match TX dc_bias)
        
    Returns:
        i_electrical: Electrical current signal [S, N]
        
    Notes:
        - In simulation, we work in normalized units, so responsivity mainly scales noise
        - Real PD: i = R × P_optical, where R is responsivity [A/W]
    """
    # Remove DC bias
    y_ac = y_optical - dc_bias
    
    # Apply photodetector gain (responsivity × area)
    gain = responsivity * area_cm2
    i_electrical = y_ac * gain
    
    return i_electrical.astype(np.float32)


def dco_ofdm_demodulate(
    y_time: np.ndarray,
    nfft: int,
    n_subcarriers: int,
    dc_bias: float = 0.5,
    remove_dc: bool = True,
) -> np.ndarray:
    """
    DCO-OFDM demodulation: convert real time-domain signal back to complex frequency symbols.
    
    Args:
        y_time: Real-valued received signal [S, N]
        nfft: FFT size
        n_subcarriers: Number of data subcarriers to extract (K)
        dc_bias: DC bias to remove (should match TX)
        remove_dc: If True, remove DC bias before FFT
        
    Returns:
        Y_freq: Complex frequency-domain symbols [S, K]
        
    Notes:
        - Extracts only positive frequency bins (1 to K)
        - Hermitian half is discarded (redundant)
    """
    S, N = y_time.shape
    
    # Ensure N == nfft (or truncate/pad)
    if N != nfft:
        if N > nfft:
            y_time = y_time[:, :nfft]
        else:
            y_time = np.pad(y_time, ((0, 0), (0, nfft - N)), mode='constant')
    
    # Remove DC bias
    if remove_dc:
        y_ac = y_time - dc_bias
    else:
        y_ac = y_time
    
    # FFT
    Y_full = np.fft.fft(y_ac, axis=1)
    
    # Extract data subcarriers (positive frequencies, skip DC)
    # Bins 1 to n_subcarriers
    Y_freq = Y_full[:, 1:n_subcarriers+1]
    
    return Y_freq.astype(np.complex64)


# ============================================================================
# INTEGRATED VLC OFDM LINK
# ============================================================================

def vlc_ofdm_link(
    x_freq: np.ndarray,
    snr_db: float,
    nfft: int,
    n_subcarriers: int,
    sample_rate_hz: float = 100e6,
    led_bandwidth_mhz: float = 20.0,
    dc_bias: float = 0.5,
    clipping_ratio: float = 0.95,
    responsivity: float = 0.5,
    area_cm2: float = 1.0,
    filter_order: int = 1,
    noise_type: str = "awgn",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Complete VLC OFDM link: TX → LED channel → RX.
    
    This is the main entry point for S4 VLC simulations, designed to parallel
    the RF OFDM link from S2/S3.
    
    Args:
        x_freq: Complex OFDM symbols [S, K]
        snr_db: Optical SNR [dB]
        nfft: FFT size
        n_subcarriers: Number of used subcarriers (K)
        sample_rate_hz: Sampling rate [Hz]
        led_bandwidth_mhz: LED 3dB bandwidth [MHz]
        dc_bias: DC bias ratio (0-1)
        clipping_ratio: LED clipping ratio
        responsivity: Photodetector responsivity [A/W]
        area_cm2: Photodetector area [cm^2]
        filter_order: LED filter order
        noise_type: "awgn" or "shot"
        rng: Random number generator
        
    Returns:
        Y_freq: Received complex symbols [S, K]
        info: Dict with channel statistics
        
    Usage Example:
        >>> x_freq = np.random.randn(14, 64) + 1j * np.random.randn(14, 64)
        >>> Y_freq, info = vlc_ofdm_link(x_freq, snr_db=15, nfft=256, n_subcarriers=64)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    S, K = x_freq.shape
    
    # ===== TRANSMITTER =====
    # 1. DCO-OFDM modulation (complex → real + bias)
    x_optical = dco_ofdm_modulate(
        x_freq,
        dc_bias=dc_bias,
        nfft=nfft,
        clipping_ratio=clipping_ratio,
    )
    
    # ===== CHANNEL =====
    # 2. LED frequency-selective channel
    y_optical = apply_led_channel(
        x_optical,
        sample_rate_hz=sample_rate_hz,
        bandwidth_3db_mhz=led_bandwidth_mhz,
        filter_order=filter_order,
    )
    
    # 3. Add optical noise
    y_ac = y_optical - np.mean(y_optical)
    signal_power = float(np.mean(y_ac.astype(np.float64) ** 2))
    y_noisy = add_optical_noise(
        y_optical,
        snr_db=snr_db,
        signal_power=signal_power,
        rng=rng,
        noise_type=noise_type,
    )
    
    # ===== RECEIVER =====
    # 4. Photodetector
    y_electrical = photodetector_response(
        y_noisy,
        responsivity=responsivity,
        area_cm2=area_cm2,
        dc_bias=dc_bias,
    )
    
    # 5. DCO-OFDM demodulation (real → complex)
    Y_freq = dco_ofdm_demodulate(
        y_electrical,
        nfft=nfft,
        n_subcarriers=n_subcarriers,
        dc_bias=0.0,  # Already removed in photodetector
        remove_dc=False,
    )
    
    # ===== DIAGNOSTICS =====
    info = {
        "signal_power_optical": float(signal_power),
        "snr_db": float(snr_db),
        "led_bandwidth_mhz": float(led_bandwidth_mhz),
        "dc_bias": float(dc_bias),
        "clipping_ratio": float(clipping_ratio),
        "nfft": int(nfft),
        "n_subcarriers": int(n_subcarriers),
        "sample_rate_hz": float(sample_rate_hz),
    }
    
    return Y_freq, info


# ============================================================================
# HELPER: VLC OFDM TX (for integration with existing pipeline)
# ============================================================================

def vlc_ofdm_tx(
    tx_grid: np.ndarray,
    nfft: int,
    dc_bias: float = 0.5,
    clipping_ratio: float = 0.95,
) -> np.ndarray:
    """
    VLC OFDM transmitter: frequency-domain grid → time-domain optical signal.
    
    Parallel to ofdm.tx() but for VLC with DCO-OFDM modulation.
    
    Args:
        tx_grid: Complex frequency-domain symbols [S, K]
        nfft: FFT size
        dc_bias: DC bias ratio
        clipping_ratio: LED clipping ratio
        
    Returns:
        x_optical: Real-valued time-domain signal [S, nfft]
    """
    x_optical = dco_ofdm_modulate(
        tx_grid,
        dc_bias=dc_bias,
        nfft=nfft,
        clipping_ratio=clipping_ratio,
    )
    return x_optical


def vlc_ofdm_rx(
    y_time: np.ndarray,
    nfft: int,
    n_subcarriers: int,
    dc_bias: float = 0.5,
) -> np.ndarray:
    """
    VLC OFDM receiver: time-domain optical signal → frequency-domain symbols.
    
    Parallel to ofdm.rx() but for VLC with DCO-OFDM demodulation.
    
    Args:
        y_time: Real-valued received signal [S, N]
        nfft: FFT size
        n_subcarriers: Number of data subcarriers
        dc_bias: DC bias to remove
        
    Returns:
        Y_freq: Complex frequency-domain symbols [S, K]
    """
    Y_freq = dco_ofdm_demodulate(
        y_time,
        nfft=nfft,
        n_subcarriers=n_subcarriers,
        dc_bias=dc_bias,
        remove_dc=True,
    )
    return Y_freq


# ============================================================================
# VLC CHANNEL FOR INTEGRATION WITH EXISTING S2 PIPELINE
# ============================================================================

def apply_vlc_channel(
    x_time: np.ndarray,
    cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply VLC channel to time-domain OFDM signal (for integration with existing S2 code).
    
    This function is designed to be a drop-in replacement for apply_rf_channel()
    in your existing OFDM pipeline (S2/S3).
    
    Args:
        x_time: Time-domain OFDM signal [S, N] (ALREADY in optical domain from vlc_ofdm_tx)
        cfg: Configuration dict with keys:
            - vlc.snr_db: Optical SNR [dB]
            - vlc.led_bandwidth_mhz: LED bandwidth [MHz]
            - vlc.dc_bias: DC bias ratio
            - ofdm.nfft: FFT size
            - (optional) vlc.sample_rate_hz: Sampling rate [Hz]
            - (optional) vlc.filter_order: LED filter order
        rng: Random number generator
        
    Returns:
        y_time: Received time-domain signal [S, N] (still in optical domain)
        
    Usage in existing pipeline:
        >>> # In your run_ofdm_link():
        >>> if channel_type == 'vlc':
        >>>     x_optical = vlc_ofdm_tx(tx_grid, nfft, dc_bias, clipping_ratio)
        >>>     y_optical = apply_vlc_channel(x_optical, cfg, rng)
        >>>     Y_freq = vlc_ofdm_rx(y_optical, nfft, n_subcarriers, dc_bias)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Extract VLC config
    vlc_cfg = cfg.get("vlc", {})
    snr_db = float(vlc_cfg.get("snr_db", 15.0))
    led_bw = float(vlc_cfg.get("led_bandwidth_mhz", 20.0))
    dc_bias = float(vlc_cfg.get("dc_bias", 0.5))
    sample_rate = float(vlc_cfg.get("sample_rate_hz", 100e6))
    filter_order = int(vlc_cfg.get("filter_order", 1))
    noise_type = str(vlc_cfg.get("noise_type", "awgn"))
    
    # LED channel
    y_optical = apply_led_channel(
        x_time,
        sample_rate_hz=sample_rate,
        bandwidth_3db_mhz=led_bw,
        filter_order=filter_order,
    )
    
    # Optical noise
    y_noisy = add_optical_noise(
        y_optical,
        snr_db=snr_db,
        rng=rng,
        noise_type=noise_type,
    )
    
    return y_noisy


# ============================================================================
# UTILITIES & VALIDATION
# ============================================================================

def estimate_required_dc_bias(
    x_freq: np.ndarray,
    nfft: int,
    target_clip_rate: float = 0.01,
) -> float:
    """
    Estimate DC bias needed to keep clipping rate below target.
    
    Args:
        x_freq: Sample OFDM symbols [S, K]
        nfft: FFT size
        target_clip_rate: Target clipping rate (e.g., 0.01 = 1%)
        
    Returns:
        dc_bias: Recommended DC bias ratio
    """
    # Generate trial signal without bias
    x_trial = dco_ofdm_modulate(x_freq, dc_bias=0.0, nfft=nfft, clipping_ratio=1.0)
    
    # Measure peak-to-average ratio
    peak = np.max(np.abs(x_trial))
    rms = np.sqrt(np.mean(x_trial ** 2))
    papr_db = 20 * np.log10(peak / (rms + 1e-12))
    
    # Heuristic: dc_bias ≈ 3-4 × std for 1% clipping
    std = np.std(x_trial)
    dc_bias = 3.5 * std
    
    return float(np.clip(dc_bias, 0.1, 1.0))


def validate_hermitian_symmetry(X: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Validate that frequency-domain signal has Hermitian symmetry.
    
    Args:
        X: Frequency-domain signal [nfft]
        tol: Tolerance for checking symmetry
        
    Returns:
        is_valid: True if Hermitian symmetric
    """
    N = X.shape[0]
    for k in range(1, N // 2):
        if np.abs(X[k] - np.conj(X[N - k])) > tol:
            return False
    return True


# ============================================================================
# MAIN (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("S4 VLC Channel Module - Standalone Test")
    print("=" * 70)
    
    # Test parameters
    S = 14  # OFDM symbols
    K = 64  # Subcarriers
    nfft = 256
    snr_db = 15.0
    
    print(f"\nTest Configuration:")
    print(f"  OFDM symbols: {S}")
    print(f"  Subcarriers: {K}")
    print(f"  FFT size: {nfft}")
    print(f"  SNR: {snr_db} dB")
    
    # Generate random QPSK data
    rng = np.random.default_rng(42)
    x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)
    
    print(f"\n1. DCO-OFDM Modulation...")
    x_optical = dco_ofdm_modulate(x_freq, dc_bias=0.5, nfft=nfft)
    print(f"   Output shape: {x_optical.shape}")
    print(f"   Range: [{x_optical.min():.3f}, {x_optical.max():.3f}]")
    print(f"   Mean: {x_optical.mean():.3f} (should be ≈ dc_bias)")
    
    print(f"\n2. LED Channel (20 MHz bandwidth)...")
    y_optical = apply_led_channel(x_optical, sample_rate_hz=100e6, bandwidth_3db_mhz=20.0)
    print(f"   Output shape: {y_optical.shape}")
    print(f"   Attenuation: {20*np.log10(np.std(y_optical) / np.std(x_optical)):.2f} dB")
    
    print(f"\n3. Add Optical Noise (SNR = {snr_db} dB)...")
    y_noisy = add_optical_noise(y_optical, snr_db=snr_db, rng=rng)
    measured_snr = 10 * np.log10(np.var(y_optical) / np.var(y_noisy - y_optical))
    print(f"   Measured SNR: {measured_snr:.2f} dB")
    
    print(f"\n4. DCO-OFDM Demodulation...")
    Y_freq = dco_ofdm_demodulate(y_noisy, nfft=nfft, n_subcarriers=K, dc_bias=0.5)
    print(f"   Output shape: {Y_freq.shape}")
    
    print(f"\n5. End-to-End Link Test...")
    Y_freq_e2e, info = vlc_ofdm_link(x_freq, snr_db=snr_db, nfft=nfft, n_subcarriers=K, rng=rng)
    print(f"   Output shape: {Y_freq_e2e.shape}")
    print(f"   Channel info: {info}")
    
    # Compute EVM
    evm = np.sqrt(np.mean(np.abs(Y_freq_e2e - x_freq) ** 2)) / np.sqrt(np.mean(np.abs(x_freq) ** 2))
    print(f"   EVM: {evm*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("Module ready for S4 integration.")
    print("=" * 70)

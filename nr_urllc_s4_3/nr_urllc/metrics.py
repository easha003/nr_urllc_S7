import numpy as np


def ber_count(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    return np.mean(tx_bits != rx_bits)


def sinr_db(signal_power: float, noise_power: float) -> float:
    return 10 * np.log10(signal_power / noise_power)

def evm_rms_percent(ref: np.ndarray, est: np.ndarray) -> float:
    """RMS EVM in %, computed over all elements of ref/est."""
    num = np.mean(np.abs(est - ref) ** 2)
    den = np.mean(np.abs(ref) ** 2) + 1e-12
    return float(100.0 * np.sqrt(num / den))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) ** 2))

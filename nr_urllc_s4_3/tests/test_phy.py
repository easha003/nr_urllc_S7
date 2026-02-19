# tests/test_phy.py
import numpy as np
import pytest
from math import erfc

# ---- Adjust these imports to match your repo structure -----------------------
import nr_urllc.utils as utils  # must expose mod(), demod(), get_rng(), and the helper we added
from nr_urllc.utils import ebn0_db_to_sigma_sc, ebn0_db_to_sigma_ofdm_time, ber_qpsk_theory
import nr_urllc.ofdm as ofdm   # must expose tx(), rx()
# -----------------------------------------------------------------------------


def bits_to_bytestr(b: np.ndarray) -> str:
    return "".join(str(int(x)) for x in b[:32])  # for debug

@pytest.mark.parametrize("M", [4, 16])
def test_constellation_roundtrip_noiseless(M):
    """
    TX mod -> RX demod with NO channel must be perfect.
    Catches Gray-map / bit-order mismatches immediately.
    """
    rng = utils.get_rng(1234)
    n_syms = 2048
    k = int(np.log2(M))
    bits = rng.integers(0, 2, size=(n_syms * k,), dtype=np.int8)
    s = utils.mod(bits, M)          # shape [n_syms]
    # simulate perfect channel
    est_bits = utils.demod(s, M)    # shape [n_syms*k]
    assert est_bits.shape == bits.shape
    if not np.array_equal(bits, est_bits):
        # helpful diff
        idx = np.where(bits != est_bits)[0][:16]
        raise AssertionError(f"Roundtrip mismatch for M={M}. "
                             f"First diffs at idx={idx}, "
                             f"tx_head={bits_to_bytestr(bits)}, rx_head={bits_to_bytestr(est_bits)}")

def test_ofdm_noiseless_roundtrip_ber0():
    """
    Put QPSK on used subcarriers, OFDM tx->rx with NO noise.
    Expect exact recovery (BER=0). Verifies subcarrier indexing & IFFT/FFT scaling.
    """
    rng = utils.get_rng(1234)
    M = 4
    nfft = 256
    cp = 0.125  # fraction => CP length = int(cp * nfft)
    n_used = 64
    k = int(np.log2(M))
    n_syms = 1024

    # Choose a simple contiguous allocation (adjust if your ofdm.tx expects indices)
    used_bins = np.arange(-n_used//2, n_used//2)  # centered around DC, may skip DC inside tx()

    bits = rng.integers(0, 2, size=(n_syms * k * n_used,), dtype=np.int8)
    # Map bits -> freq grid [n_syms, nfft] with zeros on unused
    grid = np.zeros((n_syms, nfft), dtype=np.complex64)

    # pack QPSK onto used bins each symbol
    ptr = 0
    for m in range(n_syms):
        symbits = bits[ptr:ptr + (k * n_used)]
        ptr += k * n_used
        qpsk_syms = utils.mod(symbits, M).reshape(n_used)
        # ofdm.tx may accept (grid, cp) or (grid, nfft, cp, used_bins); adapt as needed
        # We'll fill grid and let tx() do IFFT+CP
        # Map used_bins to positive indices mod nfft
        pos = (used_bins + nfft) % nfft
        grid[m, pos] = qpsk_syms

    # TX/RX (adjust signature if your ofdm.py differs)
    x = ofdm.tx(grid, nfft=nfft, cp=cp)
    y = ofdm.rx(x, nfft=nfft, cp=cp, return_full_grid=True)

    # Demap back
    rec_bits = np.empty_like(bits)
    ptr = 0
    pos = (used_bins + nfft) % nfft
    for m in range(n_syms):
        est_syms = y[m, pos]
        rec_bits[ptr:ptr + (k * n_used)] = utils.demod(est_syms, M)
        ptr += k * n_used

    err = np.count_nonzero(bits != rec_bits)
    assert err == 0, f"OFDM noiseless BER should be 0, got {err}/{bits.size}"

@pytest.mark.slow
def test_awgn_calibration_qpsk_matches_theory(within_db: float = 0.5):
    """
    Single-carrier QPSK BER should match theory within ~0.5 dB (mid-BER region).
    This nails down Eb/N0 -> sigma conversion and I/Q normalization.
    """
    rng = utils.get_rng(4321)
    M = 4
    k = int(np.log2(M))
    n_bits = 200_000
    ebn0_dbs = np.array([0, 2, 4, 6, 8], dtype=float)

    # generate bits and symbols once for fairness
    tx_bits = rng.integers(0, 2, size=(n_bits,), dtype=np.int8)
    tx_syms = utils.mod(tx_bits, M)

    ber_meas = []
    for eb in ebn0_dbs:
        sigma_RI = ebn0_db_to_sigma_sc(eb, M=M, code_rate=1.0, Es_sym=1.0)
        n = rng.normal(0.0, sigma_RI, size=tx_syms.shape) + 1j * rng.normal(0.0, sigma_RI, size=tx_syms.shape)
        rx_syms = tx_syms + n
        rx_bits = utils.demod(rx_syms, M)
        ber = np.mean(rx_bits != tx_bits[:rx_bits.size])  # guard shape
        ber_meas.append(ber)

    ber_meas = np.array(ber_meas)
    ber_th = utils.ber_qpsk_theory(ebn0_dbs)

    # Compare by converting BER->effective Eb/N0 and checking offset,
    # or do a looser absolute check in mid region:
    # We'll accept within an order of 20% relative in probability for low Eb/N0 and tighter as SNR rises,
    # but the safest is to check curve monotonic + rough alignment:
    assert np.all(np.diff(ber_meas) < 0), f"Measured BER should decrease with Eb/N0; got {ber_meas}"
    # Check a representative point (4–6 dB) within a small tolerance
    idx = np.where(ebn0_dbs == 4)[0][0]
    assert abs(np.log10(ber_meas[idx]) - np.log10(ber_th[idx])) < 0.2, \
        f"At 4 dB, BER off. meas={ber_meas[idx]:.3e}, th={ber_th[idx]:.3e}"

@pytest.mark.slow
def test_ofdm_awgn_time_sigma_respects_theory_midband():
    """
    Sanity test: OFDM time-domain noise calibrated so that after FFT,
    each used subcarrier's QPSK BER tracks the single-carrier theory trend.
    We don't force an exact dB bound (impl choices vary), but it must be decreasing and roughly aligned.
    """
    rng = utils.get_rng(777)
    M = 4
    k = int(np.log2(M))
    nfft = 256
    cp = 0.125
    n_used = 64
    n_ofdm_syms = 400  # total bits = n_ofdm_syms * n_used * k

    used_bins = np.arange(-n_used//2, n_used//2)
    pos = (used_bins + nfft) % nfft

    ebn0_dbs = np.array([0, 2, 4, 6], dtype=float)
    ber_meas = []

    for eb in ebn0_dbs:
        # make grid
        bits = rng.integers(0, 2, size=(n_ofdm_syms * n_used * k,), dtype=np.int8)
        grid = np.zeros((n_ofdm_syms, nfft), dtype=np.complex64)
        ptr = 0
        for m in range(n_ofdm_syms):
            qpsk_syms = utils.mod(bits[ptr:ptr + k*n_used], M).reshape(n_used)
            ptr += k*n_used
            grid[m, pos] = qpsk_syms

        # tx
        x = ofdm.tx(grid, nfft=nfft, cp=cp)

        # add time-domain AWGN calibrated for Eb/N0 (NumPy FFT/IFFT scaling)
        sigma_RI = ebn0_db_to_sigma_ofdm_time(eb, M=M, code_rate=1.0, nfft=nfft, ifft_norm="numpy", Es_sub=1.0)
        n = rng.normal(0.0, sigma_RI, size=x.shape) + 1j*rng.normal(0.0, sigma_RI, size=x.shape)
        y = x + n

        # rx
        Y = ofdm.rx(y, nfft=nfft, cp=cp, return_full_grid=True)

        # demap and compute BER on used tones only
        rec_bits = np.empty_like(bits)
        ptr = 0
        for m in range(n_ofdm_syms):
            est_syms = Y[m, pos]
            rec_bits[ptr:ptr + k*n_used] = utils.demod(est_syms, M)
            ptr += k*n_used

        ber = np.mean(rec_bits != bits)
        ber_meas.append(ber)

    ber_meas = np.array(ber_meas)
    assert np.all(np.diff(ber_meas) < 0), f"OFDM measured BER must decrease with Eb/N0; got {ber_meas}"
    # very loose alignment check vs. single-carrier at 4 dB (order of magnitude)
    from math import erf as _erf
    th4 = 0.5 * (1.0 - _erf(np.sqrt(10**(4/10))))
    assert 0.05*th4 < ber_meas[2] < 20*th4, f"At 4 dB, OFDM BER should be in the same ballpark as SC QPSK. got {ber_meas[2]:.3e}, th≈{th4:.3e}"

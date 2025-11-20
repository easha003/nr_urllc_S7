import numpy as np
from math import erfc
from nr_urllc.simulate import run_ofdm_awgn  


def qpsk_awgn_theory_ber(snr_db: float) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0))

def test_qpsk_ber_vs_theory():
    snrs = [0, 5, 10]
    res = run_ofdm_awgn(snrs, M=4)  # uses your OFDM path
    for snr_db, ber in res.items():
        th = qpsk_awgn_theory_ber(snr_db)
        ratio_db = 10 * np.log10(max(ber, 1e-12) / max(th, 1e-12))
        assert abs(ratio_db) < 0.5, f"SNR={snr_db}dB: Δ={ratio_db:.2f} dB (sim={ber}, th={th})"

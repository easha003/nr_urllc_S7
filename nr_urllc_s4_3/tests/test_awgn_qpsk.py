import numpy as np
from math import erfc
from nr_urllc import simulate

def qpsk_awgn_theory_ber(snr_db: float) -> float:
    # AWGN QPSK (same as BPSK): Pb = 0.5 * erfc( sqrt(Eb/N0) )
    ebn0 = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0))

def test_awgn_qpsk_ber():
    # Baseline bit-level AWGN (M0 path)
    cfg = {
        "sim": {"type": "baseline_awgn", "seed": 0},
        "tx":  {"M": 4, "n_bits": 1_000_000},
        "channel": {"snr_db": 3},
        "io": {"write_json": False},
    }
    result = simulate.run(cfg)
    sim_ber = result["ber"]
    theory_ber = qpsk_awgn_theory_ber(cfg["channel"]["snr_db"])

    # Compare in dB with ±0.5 dB tolerance (robust across RNG)
    ratio_db = 10 * np.log10(max(sim_ber, 1e-12) / max(theory_ber, 1e-12))
    assert abs(ratio_db) < 0.5, f"Δ={ratio_db:.2f} dB (sim={sim_ber}, th={theory_ber})"

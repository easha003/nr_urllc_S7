# test_robust_m2.py
from nr_urllc import simulate

cfg = {
    "sim": {"type": "ofdm_m2", "seed": 1234},
    "tx": {"M": 4, "n_bits": 25600},
    "ofdm": {"nfft": 256, "cp": 0.125, "n_subcarriers": 64, "minislot_symbols": 7},
    "pilots": {"spacing": 2, "offset": 0, "seed": 999, "power_boost_db": 3.0},
    "channel": {
        "model": "tdl",
        "snr_db_list": [2,4,6,8],
        "tdl": {"delays": [0, 3, 5], "powers_db": [0.0, -4.0, -8.0]}
    },
    "eq": {"type": "mmse"},
    "io": {"write_json": False}
}

result = simulate.run(cfg)
print("Robust M2 Results:")
for i, snr in enumerate(result["snr_db"]):
    print(f"SNR {snr:2.0f} dB: BER={result['ber'][i]:.2e}, EVM={result['evm_percent'][i]:5.1f}%")

max_evm = max(result["evm_percent"])
print(f"\nM2 Status: EVM ≤ 12.5%? {'✓' if max_evm <= 12.5 else '✗'} (got {max_evm:.1f}%)")
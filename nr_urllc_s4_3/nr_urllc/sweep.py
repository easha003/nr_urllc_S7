# nr_urllc/sweep.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Callable, Optional
import math
import numpy as np
from . import utils
from . import ofdm
from . import fec
# ---------- dataclasses ----------

@dataclass
class SweepPoint:
    snr_db: float
    n_bits: int
    n_errs: int
    ber: float
    evm_percent: float= float("nan") # <-- default so other sweep helpers still work
    mse_H: float= float("nan")

@dataclass
class SweepResult:
    success: bool
    meta: Dict[str, Any]
    points: List[SweepPoint]

    @property
    def snr_db(self) -> List[float]:
        return [p.snr_db for p in self.points]

    @property
    def ber_curve(self) -> List[float]:
        return [p.ber for p in self.points]

    @property
    def evm_curve(self) -> List[float]:
        return [p.evm_percent for p in self.points]

    @property
    def mse_curve(self) -> List[float]:
        return [p.mse_H for p in self.points]

    @property
    def n_bits_curve(self) -> List[int]:
        return [p.n_bits for p in self.points]

    @property
    def n_errs_curve(self) -> List[int]:
        return [p.n_errs for p in self.points]



# ---------- theory helpers (array-safe) ----------

def ber_mqam_theory(ebn0_db: float | np.ndarray, M: int) -> np.ndarray:
    if M == 4:
        return utils.ber_qpsk_theory(ebn0_db)
    g = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    k = int(math.log2(M))
    return (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * utils.qfunc(np.sqrt(3.0 * k / (M - 1) * g))


# ---------- planning bits per SNR ----------

def _align_up(n: int, align: int) -> int:
    if align <= 1:
        return int(n)
    return int(((n + align - 1) // align) * align)

def plan_bits_per_snr(
    snr_db_list: List[float],
    M: int,
    target_errs: int = 100,
    min_bits: int = 20_000,
    max_bits: int = 2_000_000,
    floor_ber: float = 1e-6,
    align_to: Optional[int] = None,
) -> Dict[float, int]:
    """
    Choose n_bits per SNR to expect ~target_errs errors, capped into [min_bits, max_bits].
    If align_to is provided (e.g., k for SC; K*k for OFDM), n_bits is rounded up to a multiple of it.
    """
    plan: Dict[float, int] = {}
    # Treat 'snr' as Eb/N0 for SC; Es/N0->Eb/N0 consistency is handled by the measure fn.
    for snr in snr_db_list:
        ber_est = float(ber_mqam_theory(snr, M))
        ber_est = max(ber_est, floor_ber)  # avoid exploding counts at very high SNR
        n = math.ceil(target_errs / ber_est)
        n = max(min_bits, min(n, max_bits))
        if align_to is not None:
            n = _align_up(n, align_to)
        plan[float(snr)] = int(n)
    return plan


# ---------- measurement backends ----------

def measure_sc_awgn_ber(
    ebn0_db: float,
    M: int,
    n_bits: int,
    rng: np.random.Generator,
    code_rate: float = 1.0,
) -> tuple[int, int, float]:
    """Single-carrier BER under AWGN using utils.mod/demod and sigma from Eb/N0."""
    k = int(math.log2(M))
    tx_bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    tx_syms = utils.mod(tx_bits, M)
    sigma_RI = utils.ebn0_db_to_sigma_sc(ebn0_db, M=M, code_rate=code_rate, Es_sym=1.0)
    n = rng.normal(0.0, sigma_RI, tx_syms.shape) + 1j * rng.normal(0.0, sigma_RI, tx_syms.shape)
    rx_syms = tx_syms + n
    rx_bits = utils.demod(rx_syms, M)
    n_eval = rx_bits.size  # should equal n_bits, but guard anyway
    n_errs = int(np.count_nonzero(rx_bits != tx_bits[:n_eval]))
    ber = n_errs / float(n_eval)
    return n_errs, n_eval, ber


def measure_ofdm_awgn_ber(
    snr_db: float,
    M: int,
    n_bits_target: int,
    rng: np.random.Generator,
    nfft: int,
    cp: float,
    n_subcarriers: int,
    minislot_symbols: int,
    code_rate: float = 1.0,
    ifft_norm: str = "numpy",
) -> tuple[int, int, float]:
    """
    OFDM BER: builds enough OFDM symbols to meet/exceed n_bits_target.
    Returns (n_errs, n_bits_eval, ber).
    """
    k = int(math.log2(M))
    bits_per_ofdm_symbol = n_subcarriers * k
    n_syms = math.ceil(n_bits_target / bits_per_ofdm_symbol)
    # optional: force to a multiple of minislot size
    if minislot_symbols > 1:
        n_syms = _align_up(n_syms, minislot_symbols)

    n_bits_eval = n_syms * bits_per_ofdm_symbol

    # Make used-tone matrix [S, K]
    tx_bits = rng.integers(0, 2, size=n_bits_eval, dtype=np.int8)
    syms = utils.mod(tx_bits, M).reshape(n_syms, n_subcarriers)

    x = ofdm.tx(syms, nfft=nfft, cp=cp, n_subcarriers=n_subcarriers)  # time-domain with CP
    sigma_RI = utils.ebn0_db_to_sigma_ofdm_time(
        snr_db, M=M, code_rate=code_rate, nfft=nfft, ifft_norm=ifft_norm, Es_sub=1.0
    )
    noise = rng.normal(0.0, sigma_RI, x.shape) + 1j * rng.normal(0.0, sigma_RI, x.shape)
    y = x + noise
    Y_used = ofdm.rx(y, nfft=nfft, cp=cp, n_subcarriers=n_subcarriers)  # [S, K] used tones
    rx_bits = utils.demod(Y_used.reshape(-1), M)

    n_errs = int(np.count_nonzero(rx_bits != tx_bits))
    ber = n_errs / float(n_bits_eval)
    return n_errs, n_bits_eval, ber


# ---------- top-level sweep helpers ----------

def autoramp_sc_qpsk_sweep(
    ebn0_db_list: List[float],
    seed: int,
    M: int = 4,
    target_errs: int = 100,
    min_bits: int = 100_000,
    max_bits: int = 10_000_000,
) -> SweepResult:
    rng = utils.get_rng(seed)
    k = int(math.log2(M))
    plan = plan_bits_per_snr(
        ebn0_db_list, M=M, target_errs=target_errs, min_bits=min_bits, max_bits=max_bits, align_to=k
    )
    points: List[SweepPoint] = []
    for snr in ebn0_db_list:
        n_bits = plan[snr]
        n_errs, n_eval, ber = measure_sc_awgn_ber(snr, M, n_bits, rng)
        points.append(SweepPoint(snr_db=float(snr), n_bits=n_eval, n_errs=n_errs, ber=ber))
    meta = {"seed": seed, "M": M, "mode": "sc_awgn", "target_errs": target_errs}
    return SweepResult(success=True, meta=meta, points=points)


def autoramp_ofdm_qpsk_sweep(
    snr_db_list: List[float],
    seed: int,
    *,
    M: int = 4,
    nfft: int,
    cp: float,
    n_subcarriers: int,
    minislot_symbols: int,
    target_errs: int = 200,
    min_bits: int = 100_000,
    max_bits: int = 10_000_000,
) -> SweepResult:
    rng = utils.get_rng(seed)
    k = int(math.log2(M))
    align_to = n_subcarriers * k  # make bitcount align with whole OFDM symbols
    plan = plan_bits_per_snr(
        snr_db_list, M=M, target_errs=target_errs, min_bits=min_bits, max_bits=max_bits, align_to=align_to
    )
    points: List[SweepPoint] = []
    for snr in snr_db_list:
        n_bits = plan[snr]
        n_errs, n_eval, ber = measure_ofdm_awgn_ber(
            snr, M, n_bits, rng, nfft=nfft, cp=cp, n_subcarriers=n_subcarriers, minislot_symbols=minislot_symbols
        )
        points.append(SweepPoint(snr_db=float(snr), n_bits=n_eval, n_errs=n_errs, ber=ber))
    meta = {
        "seed": seed, "M": M, "mode": "ofdm_awgn",
        "nfft": nfft, "cp": cp, "n_subcarriers": n_subcarriers, "minislot_symbols": minislot_symbols,
        "target_errs": target_errs
    }
    return SweepResult(success=True, meta=meta, points=points)

# --- M2 autoramp (OFDM QPSK over TDL) ---
from typing import Any, Dict, List, Tuple
from copy import deepcopy
from . import simulate
from .pilots import comb_indices


def _bits_per_frame_M2(cfg: Dict[str, Any]) -> int:
    """Compute DATA bits per frame (payload only) for M2."""
    ofd   = cfg["ofdm"]; pil = cfg["pilots"]; tx = cfg["tx"]
    S     = int(ofd["minislot_symbols"])
    K     = int(ofd["n_subcarriers"])
    M     = int(tx.get("M", 4))
    k     = int(round(math.log2(M)))
    spacing = int(pil["spacing"])
    offset  = int(pil.get("offset", 0))
    # number of pilot columns per symbol for comb pattern
    Kp = len(comb_indices(K, spacing, offset))
    Kd = K - Kp
    if Kd <= 0:
        raise ValueError("No data REs (all-pilot); autoramp M2 requires Kd>0")
    return S * Kd * k

def autoramp_ofdm_m2_sweep(
    cfg_base: Dict[str, Any],
    target_errs: int = 200,
    min_bits: int = 50_000,
    max_bits: int = 10_000_000,
    growth: float = 2.0,
) -> SweepResult:
    """
    Adaptive BER vs SNR for M2 (OFDM+pilots+TDL). Stops at target_errs or N_max per SNR.
    Uses simulate.run() with single-SNR runs and aggregates BER across repetitions.
    """
    cfg0 = deepcopy(cfg_base)
    sim   = cfg0.get("sim", {})
    ch    = cfg0.get("channel", {})
    snr_list = list(ch["snr_db_list"])
    seed0 = int(sim.get("seed", 1234))
    
    chan_seed_fixed = int(cfg0.get('channel', {}).get('seed', seed0 + 7777))  # fixed channel across SNRs
    rng_run = utils.get_rng(seed0)   # one RNG for the whole autoramp sweep
    bits_per_frame = _bits_per_frame_M2(cfg0)

    points: List[SweepPoint] = []

    for idx, snr in enumerate(snr_list):
        # ===== ADD THIS ADAPTIVE TARGET_ERRS =====
        # At high SNR (>16 dB), BER is very low, so we need MINIMUM errors
        # to avoid zero-error instability
        if snr >= 20:
            effective_target_errs = max(50, target_errs // 4)  # Minimum 50 errors
        elif snr >= 16:
            effective_target_errs = max(100, target_errs // 2)  # Minimum 100 errors
        else:
            effective_target_errs = target_errs
        # ==========================================
        total_bits = 0
        total_errs = 0
        n_bits_try = max(min_bits, bits_per_frame)
        snr_seed = seed0 + idx * 12345 

        # Track latest EVM/MSE measured at this SNR
        last_evm_percent = float("nan")
        last_mse_H = float("nan")

        rep = 0
        while total_bits < max_bits and total_errs < target_errs:
            cfg = deepcopy(cfg0)
            # single SNR run
            cfg["channel"]["snr_db_list"] = [float(snr)]
            cfg['channel']['seed'] = chan_seed_fixed
            # grow n_bits each repetition, but simulate.py will round/pad to whole frames
            cfg["tx"]["n_bits"] = int(n_bits_try)
            # vary seed so repetitions are independent
            
            # Disable plotting & JSON during inner autoramp reps
            cfg.setdefault("io", {})["plot"] = False
            cfg["io"]["show_plot"] = False
            cfg["io"]["write_json"] = False
            cfg['sim']['seed'] = (snr_seed + 7919 * rep)
            res = simulate.run(cfg)  # expects "ber": [value] for this single SNR
            ber = float(res["ber"][0])
            # Pull EVM% and MSE from the M2 single-SNR result (use the first point)
            try:
                evm_seq = res.get("evm_percent", [])
                last_evm_percent = float(evm_seq[0] if isinstance(evm_seq, (list, tuple, np.ndarray)) else evm_seq)
            except Exception:
                pass
            try:
                mse_seq = res.get("mse_H", [])
                last_mse_H = float(mse_seq[0] if isinstance(mse_seq, (list, tuple, np.ndarray)) else mse_seq)
            except Exception:
                pass

            # infer evaluated bits = full frames worth (simulate pads/wraps)
            frames = int(math.ceil(n_bits_try / bits_per_frame))
            eval_bits = frames * bits_per_frame
            errs = int(round(ber * eval_bits))

            total_bits += eval_bits
            total_errs += errs

            # increase trial size for next repetition if needed
            if total_errs < target_errs and total_bits < max_bits:
                n_bits_try = int(min(max_bits - total_bits, max(n_bits_try * growth, bits_per_frame)))
            rep += 1

            # avoid infinite loop if ber==0 persistently: let it hit the max_bits cap

        # Final BER estimate (or upper bound if 0 errors)
        if total_errs > 0:
          ber_hat = total_errs / total_bits
        else:
          ber_hat = 0.0    # keep field for backwards-compat
        #  ber_ub  = 1.0 / max(total_bits, 1)  # optional: add this in the result dict

        points.append(SweepPoint(
            snr_db=float(snr),
            n_bits=total_bits,
            n_errs=total_errs,
            ber=ber_hat,
            evm_percent=last_evm_percent,
            mse_H=last_mse_H,
        ))

    meta = {
        "mode": "ofdm_m2",
        "seed": seed0,
        "target_errs": target_errs,
        "min_bits": min_bits,
        "max_bits": max_bits,
        "growth": growth,
        "bits_per_frame": bits_per_frame,
    }
    return SweepResult(success=True, meta=meta, points=points)

# ----------------------------- S2: BLER over SNR (with TDL/CDL) ----------------------------- #
from . import ofdm, utils, pilots as pilots_mod, equalize as eq, interp as interp_mod, channel

def _align_up(x: int, m: int) -> int:
    if m <= 0: return x
    r = x % m
    return x if r == 0 else (x + (m - r))

def bler_ofdm_sweep(cfg: dict) -> dict:
    """BLER vs SNR using TB+CRC over the OFDM+pilots+ZF pipeline with TDL/CDL."""
    import numpy as np
    from .tb import append_crc, check_crc, bytes_to_bits, bits_to_bytes

    sim = cfg.get("sim", {}); tx = cfg.get("tx", {}); of = cfg.get("ofdm", {})
    nr  = cfg.get("nr", {});   ch = cfg.get("channel", {}); pil = cfg.get("pilots", {})
    io  = cfg.get("io", {});   urc = cfg.get("urllc", {})

    rng_seed = int(sim.get("seed", 0)); 
    rng = utils.get_rng(rng_seed)

    
    chan_seed_fixed = int(ch.get('seed', rng_seed + 1000))  # fixed channel RNG seed across SNRs
# Independent RNGs for different random processes
    # rng_channel will be instantiated per-SNR with chan_seed_fixed
    rng_payload = utils.get_rng(rng_seed + 2000)  # For payload data
    rng_noise = utils.get_rng(rng_seed + 3000)    # For AWGN noise

    M = int(tx.get("M", 4)); k = int(np.log2(M))
    nfft = int(of.get("nfft", 256)); cp = float(of.get("cp", 0.125))
    K = int(of.get("n_subcarriers", 120)); Lslot = int(of.get("minislot_symbols", nr.get("minislot_symbols", 7)))
    Ncp = int(round(cp * nfft))

    spacing = int(pil.get("spacing", 4)); offset = int(pil.get("offset", 0))
    pseed = int(pil.get("seed", 123)); boost = float(pil.get("power_boost_db", 0.0))
    pmode = str(pil.get("power_mode", "constrained"))

    # Add warning for unconstrained mode
    if pmode.lower() == "unconstrained":
        print("[WARNING] Using 'unconstrained' pilot power mode. This violates power regulations!")
        print("          Consider using 'power_mode: constrained' for realistic results.")

    app_bytes = int(urc.get("app_payload_bytes", 32) or 32)
    tb_bytes = int(urc.get("tb_payload_bytes", app_bytes + 16))

    snr_list = ch.get("snr_db_list", [0, 2, 4, 6, 8, 10])
    base_packets_per_snr = int(cfg.get("bler", {}).get("packets_per_snr", 400))
    min_errors = int(cfg.get("bler", {}).get("min_errors", 20))

    tmp = np.zeros((1, K), dtype=np.complex64)
    _grid1, pilot_mask1, pilot_vals1, data_Es1, pilot_Es1 = pilots_mod.place(
        tmp, spacing, offset=offset, seed=pseed, power_boost_db=boost, power_mode=pmode
    )
   
    pilots_per_sym = int(pilot_mask1[0].sum()); data_RE_per_sym = K - pilots_per_sym
    if data_RE_per_sym <= 0:
        raise RuntimeError("Pilot pattern leaves no data REs")

    # In bler_ofdm_sweep, after pilot placement:
    print(f"[INFO] Pilot power mode: {pmode}")
    print(f"       Data RE energy scaling: {data_Es1:.4f} ({10*np.log10(data_Es1):.2f} dB)")
    print(f"       Pilot RE energy: {pilot_Es1:.4f} ({10*np.log10(pilot_Es1):.2f} dB)")
    if pmode.lower() == "unconstrained":
        total_power = (pilots_per_sym * pilot_Es1 + data_RE_per_sym * data_Es1) / K
        print(f"       Total TX power: {total_power:.4f} ({10*np.log10(total_power):.2f} dB) - EXCEEDS LIMIT!")
    
    # ADD: Adaptive scaling based on expected BLER
    adaptive_packets = cfg.get("bler", {}).get("adaptive_packets", True)
    data_Es_actual = data_Es1
    bler = []; packets = []
    for i_snr, snr_db in enumerate(snr_list):
        
        
        # Reset RNGs per SNR so channel/data/noise sequences are identical across SNRs
        rng_channel = utils.get_rng(chan_seed_fixed)
        rng_payload = utils.get_rng(rng_seed + 2000)
        rng_noise   = utils.get_rng(rng_seed + 3000)
# Adaptively increase packets for high SNR
        if adaptive_packets and i_snr > 0 and len(bler) > 0:
            # Estimate required packets based on previous BLER
            last_bler = bler[-1]
            if last_bler > 0:
                # Target 100 errors: packets = 100 / expected_bler
                estimated_packets = int(min_errors / (last_bler * 0.3))  # 0.3 = aggressive factor
                packets_per_snr = min(max(base_packets_per_snr, estimated_packets), 100000)
            else:
                packets_per_snr = base_packets_per_snr * 10  # 10x if previous was zero
        else:
            packets_per_snr = base_packets_per_snr
        
        print(f"[INFO] SNR {snr_db} dB: Running up to {packets_per_snr} packets")

        n_fail = 0; n_sent = 0
        while n_sent < packets_per_snr and n_fail < min_errors:
            model = str(ch.get("model", "tdl")).lower()
            if model == "tdl":
                prof = ch.get("tdl", {})
                delays = prof.get("delays", [0,3,5])
                powers_db = prof.get("powers_db", [0.0, -3.0, -6.0])
                h_fir = channel.tdl_fir_from_profile(delays, powers_db, rng=rng_channel)
            elif model == "cdl":
                prof = ch.get("cdl", {})
                profile_name = str(prof.get("profile", "C")).upper()
                scale_samples = int(prof.get("scale_samples", 0)) or None
                k_db = prof.get("k_db", None)
                k_db = float(k_db) if k_db is not None else None
                h_fir = channel.cdl_fir(profile=profile_name, ncp=Ncp, scale_samples=scale_samples, rice_k_db=k_db, rng=rng_channel)
            else:
                h_fir = np.array([1.0+0j], dtype=np.complex64)

            if (len(h_fir) - 1) > Ncp:
                raise ValueError(f"CP too short: L-1={len(h_fir)-1} > Ncp={Ncp}")

            
            payload = rng_payload.integers(0, 256, size=tb_bytes, dtype=np.uint8).tobytes()
            tb = append_crc(payload); tb_bits = bytes_to_bits(tb); Lbits = len(tb_bits)
            # --- FEC encode (optional) ---
            fec_cfg = cfg.get('fec', {})
            code_bits, fec_meta = fec.encode(tb_bits, fec_cfg)
            Lcode = int(code_bits.size)

            data_bits_per_sym = data_RE_per_sym * k
            n_syms_est = int(np.ceil(Lcode / max(1, data_bits_per_sym)))
            n_syms = ( ( (n_syms_est + Lslot - 1) // Lslot ) * Lslot ) if Lslot > 1 else n_syms_est

            base = np.zeros((n_syms, K), dtype=np.complex64)
            _grid, pilot_mask, pilot_vals, data_Es, pilot_Es = pilots_mod.place(
                base, spacing, offset=offset, seed=pseed, power_boost_db=boost, power_mode=pmode
            )
            data_mask = ~pilot_mask
            capacity_bits = int(np.sum(data_mask)) * k
            if capacity_bits < Lcode:
                add = Lslot if Lslot > 1 else 1
                n_syms2 = n_syms + add
                base = np.zeros((n_syms2, K), dtype=np.complex64)
                _grid, pilot_mask, pilot_vals, data_Es, pilot_Es = pilots_mod.place(
                    base, spacing, offset=offset, seed=pseed, power_boost_db=boost, power_mode=pmode
                )
                data_mask = ~pilot_mask
                n_syms = n_syms2
                capacity_bits = int(np.sum(data_mask)) * k

            pad = capacity_bits - Lcode
            bits_tx = np.pad(code_bits, (0, pad), constant_values=0) if pad > 0 else code_bits
            syms_data = utils.mod(bits_tx, M).reshape(-1)

            tx_grid = pilot_vals.astype(np.complex64)
            tx_grid[data_mask] = syms_data[: int(np.sum(data_mask))]

            x = ofdm.tx(tx_grid, nfft=nfft, cp=cp, n_subcarriers=K)
            y_c = channel.apply_fir_per_symbol(x, h_fir)

            sigma_RI = utils.ebn0_db_to_sigma_ofdm_time(
                snr_db, M=M, code_rate=fec.get_rate(fec_cfg), nfft=nfft, ifft_norm="numpy", Es_sub=data_Es_actual
            )

            # --- Timely BLER + HARQ (Chase) with soft LLR + soft LDPC ---
            from . import nr_timing
            from .tb import bits_to_bytes

            # Numerology and timing (use your existing key scheme)
            mu = int(cfg.get('nr', {}).get('mu', cfg.get('numerology', {}).get('mu', 2)))
            Lsym = int(
                cfg.get('nr', {}).get('minislot_symbols',
                cfg.get('numerology', {}).get('L_symbols',
                cfg.get('ofdm', {}).get('minislot_symbols', 7)))
            )
            cp_frac = float(cfg.get('ofdm', {}).get('cp', cfg.get('numerology', {}).get('cp_fraction', 0.125)))
            tti_ms = nr_timing.minislot_tti_ms(mu, Lsym, cp_frac)

            # URLLC block (accept both deadline_ms and radio_deadline_ms)
            urllc_cfg   = cfg.get('urllc', {}) or {}
            deadline_ms = float(urllc_cfg.get('deadline_ms', urllc_cfg.get('radio_deadline_ms', 8.0)))

            # HARQ settings (accept under urllc.harq or nr.harq)
            harq_cfg     = dict(urllc_cfg.get('harq') or cfg.get('nr', {}).get('harq') or {})
            harq_enabled = bool(harq_cfg.get('enabled', False))
            max_retx     = int(harq_cfg.get('max_retx', 0))          # extra attempts allowed
            combining    = str(harq_cfg.get('combining', 'none')).lower()

            attempts = 0
            elapsed  = 0.0
            ok = False
            _llr_acc = None
            m_bits   = int(np.log2(M))

            while True:
                attempts += 1


                # New noise on each HARQ attempt
                noise = rng_noise.normal(0.0, sigma_RI, y_c.shape) + 1j * rng_noise.normal(0.0, sigma_RI, y_c.shape)
                y = y_c + noise

                # Receiver per attempt
                Y = ofdm.rx(y, nfft=nfft, cp=cp, n_subcarriers=K)

                # Channel estimation using pilots (noise-affected per attempt)
                H_p = np.zeros_like(Y, dtype=np.complex64)
                mask = pilot_mask
                H_p[mask] = (Y[mask] / (pilot_vals[mask] + 1e-12)).astype(np.complex64)
                H_est = interp_mod.interp_freq_linear(H_p, mask)
                H_est = interp_mod.smooth_time_triangular(H_est)

                # MMSE equalization
                sigma2 = float(2.0 * (sigma_RI ** 2)*nfft)  # complex noise variance
                Y_eq = eq.equalize_mmse_robust(Y, H_est, float(sigma2), 1e-12)

                # Equalized data symbols
                y_data = Y_eq[data_mask].reshape(-1)

                # Post-eq noise variance per complex symbol for LLRs.
                # (sigma_RI is your per-real/imag std you already computed for AWGN)
                sigma2_llr = sigma2

                # Soft LLRs from equalized symbols (trim to coded length)
                llr_cur = utils.qam_llr_maxlog(y_data, M, sigma2=sigma2_llr)
                llr_cur = llr_cur[: Lcode]

                # HARQ Chase combining (LLR addition)
                if harq_enabled and combining == "chase":
                    _llr_acc = llr_cur if _llr_acc is None else (_llr_acc + llr_cur)
                    use_llr  = _llr_acc
                else:
                    use_llr  = llr_cur

                # Soft-decision LDPC (min-sum) via fec.decode_soft; falls back for none/repeat
                dec_bits = fec.decode_soft(use_llr, fec_cfg, meta=fec_meta, info_len=Lbits)
                tb_rx    = bits_to_bytes(dec_bits)
                ok_now   = check_crc(tb_rx)

                # Timeliness check
                elapsed += tti_ms
                if ok_now and (elapsed <= deadline_ms):
                    ok = True
                    break

                # Stop if no HARQ, retries exhausted, or deadline reached
                if (not harq_enabled) or (attempts > (max_retx + 1)) or (elapsed >= deadline_ms):
                    break

            # Update counters after the HARQ/Deadline loop
            if not ok:
                n_fail += 1
            n_sent += 1

            # Progress report every 1000 packets
            if n_sent % 1000 == 0:
                current_bler = n_fail / n_sent if n_sent > 0 else 0
                print(f"  Progress: {n_sent}/{packets_per_snr} packets, "
                      f"{n_fail} errors, BLER={current_bler:.2e}")
            # --- end timely BLER block ---


        bler.append(n_fail / float(max(n_sent,1)))
        packets.append(n_sent)

    meta = {
        "mode": "tb_bler", "seed": rng_seed, "M": M, "nfft": nfft, "cp": cp, "n_subcarriers": K,
        "minislot_symbols": Lslot, "pilot_spacing": spacing, "pilot_offset": offset, "pilot_seed": pseed,
        "pilot_power_boost_db": boost, "pilot_power_mode": pmode, "tb_payload_bytes": tb_bytes,
        "packets_per_snr": packets_per_snr, "min_errors": min_errors, "channel_model": ch.get("model", "tdl")
    }
    return {"success": True, "snr_db": list(map(float, snr_list)), "bler": bler, "packets": packets, "meta": meta}

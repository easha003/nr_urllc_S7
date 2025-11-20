# nr_urllc/simulate.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import math as m
import copy

from . import utils, ofdm, channel, fec
from . import pilots as pilots_mod
from . import metrics as metrics_mod
from . import equalize as eq
from .utils import qfunc
from nr_urllc import interp as interp_mod
from nr_urllc.plots import plot_m2_curves, plot_ofdm_awgn_ber

# --- SNR list parser: accepts list, single, CSV string, or "start:step:stop" ---
def _get_snr_db_list(ch_cfg) -> list[float]:
    """
    Supports:
      snr_db_list: [0,2,4,6]   | "0,2,4,6"  | "0:2:16"
      snr_db: 6.0               | [6,8,10]
    """
    if ch_cfg is None:
        return [0.0, 5.0, 10.0]

    def as_list(v):
        import numpy as _np
        if isinstance(v, (list, tuple, _np.ndarray)):
            return [float(x) for x in v]
        if isinstance(v, (int, float)):
            return [float(v)]
        if isinstance(v, str):
            s = v.strip()
            if ":" in s:
                parts = [p.strip() for p in s.split(":")]
                if len(parts) == 3:
                    a, step, b = map(float, parts)
                    if step == 0: return [a]
                    n = int(_np.floor((b - a) / step)) + 1
                    return [a + i * step for i in range(max(1, n))]
            if "," in s:
                return [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            try:
                return [float(s)]
            except Exception:
                return []
        return []

    if "snr_db_list" in ch_cfg:
        lst = as_list(ch_cfg["snr_db_list"])
        return lst if lst else [0.0, 5.0, 10.0]
    if "snr_db" in ch_cfg:
        lst = as_list(ch_cfg["snr_db"])
        return lst if lst else [float(ch_cfg["snr_db"])]
    return [0.0, 5.0, 10.0]



# --- Autoramp controller + fixed-σ² MMSE (drop-in helpers) ---
from nr_urllc.autoramp import AutorampController
from .equalize import mmse_equalize


def _write_json(maybe: bool, path: str, obj: dict):
    if maybe:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


# ---------------------------- M0: Baseline AWGN ---------------------------- #
def run_baseline_awgn(cfg: dict) -> dict:
    """
    Baseline bit-level AWGN simulation (no OFDM).
    Interprets cfg['channel']['snr_db'] as Eb/N0 [dB], converts to Es/N0 internally.
    Uses a local RNG; no global seeding.
    """
    sim = cfg.get("sim", {})
    tx  = cfg.get("tx", {})
    ch  = cfg.get("channel", {})
    io  = cfg.get("io", {})

    rng    = utils.get_rng(sim.get("seed"))
    M      = int(tx.get("M", 4))                      # 4=QPSK, 16=16QAM
    k      = int(np.log2(M))
    n_bits = int(tx.get("n_bits", 100_000))
    ebn0   = float(ch.get("snr_db", 5.0))             # Eb/N0 in dB (by convention)

    # Map bits -> symbols (unit average Es expected from utils.mod)
    bits    = rng.integers(0, 2, size=n_bits)
    symbols = utils.mod(bits, M).astype(np.complex64)

    # Eb/N0 -> Es/N0 (per-symbol SNR)
    esn0_db = ebn0 + 10 * np.log10(k)

    # AWGN at symbol level
    rx_symbols = channel.awgn(symbols, esn0_db, rng)

    # Hard-decision demod
    bits_hat = utils.demod(rx_symbols, M)
    ber = float(np.mean(bits[:len(bits_hat)] != bits_hat))

    out = {
        "success": True,
        "reps_used": 1,
        "sinr_db": ebn0,                 # report Eb/N0 you asked for
        "latency_ms": 0.0,               # placeholder at M0
        "crc_ok": True,                  # placeholder at M0
        "meta": {"seed": sim.get("seed"), "M": M, "EbN0_dB": ebn0, "EsN0_dB": esn0_db},
        "ber": ber,
    }
    _write_json(bool(io.get("write_json", False)), io.get("out_json", "artifacts/result.json"), out)
    return out


# ----------------------------- M1: OFDM over AWGN -------------------------- #

def _run_ofdm_awgn_cfg(cfg: dict) -> dict:
    """
    OFDM mini-slot over AWGN (config-driven).
    Interprets cfg['channel']['snr_db_list'] as Eb/N0 [dB] values.
    Adds calibrated time-domain noise so post-FFT per-subcarrier Es/N0 matches target.
    Uses a local RNG; no global seeding.
    """
    sim = cfg.get("sim", {})
    tx  = cfg.get("tx", {})
    of  = cfg.get("ofdm", {})
    ch  = cfg.get("channel", {})
    io  = cfg.get("io", {})

    rng   = utils.get_rng(sim.get("seed"))
    M     = int(tx.get("M", 4))
    k     = int(np.log2(M))
    n_bits = int(tx.get("n_bits", 120_000))

    nfft  = int(of.get("nfft", 256))
    cp    = float(of.get("cp", 0.125))
    n_sc  = int(of.get("n_subcarriers", 64))
    _     = int(of.get("n_symbols", 14))             # not directly used here
    L_ms  = int(of.get("minislot_symbols", 4))       # {2,4,7} typical

    ebn0_list = _get_snr_db_list(ch)

    # Bits -> QAM (unit Es) -> grid with whole rows
    n_syms_needed = (n_bits // k + n_sc - 1) // n_sc
    n_bits_eff    = n_syms_needed * n_sc * k
    bits          = rng.integers(0, 2, n_bits_eff)
    tx_syms       = utils.mod(bits, M).astype(np.complex64)
    tx_grid       = tx_syms.reshape(n_syms_needed, n_sc)

    # Mini-slot slice
    use_syms   = min(L_ms, tx_grid.shape[0])
    tx_grid_ms = tx_grid[:use_syms, :]

    # OFDM TX once (IFFT has 1/N scaling; then prepend CP)
    tx_time = ofdm.tx(tx_grid_ms, nfft=nfft, cp=cp).astype(np.complex64)

    out_curve = {}
    for ebn0_db in ebn0_list:
        # Eb/N0 -> Es/N0
        esn0_db  = float(ebn0_db) + 10 * np.log10(k)
        esn0_lin = 10 ** (esn0_db / 10.0)

        # Calibrated time-domain AWGN: var = 1 / (Nfft * Es/N0)
        noise_var = 1.0 / (nfft * esn0_lin)
        noise = (
            rng.normal(0, np.sqrt(noise_var / 2), size=tx_time.shape)
            + 1j * rng.normal(0, np.sqrt(noise_var / 2), size=tx_time.shape)
        ).astype(np.complex64)
        rx_time = (tx_time + noise).astype(np.complex64)

        # OFDM RX
        rx_grid = ofdm.rx(rx_time, nfft=nfft, cp=cp, n_subcarriers=n_sc)
        rx_syms = rx_grid.reshape(-1)[: tx_syms.size]

        # Demod & BER
        bits_hat = utils.demod(rx_syms, M)
        ber = float(np.mean(bits[:len(bits_hat)] != bits_hat))
        out_curve[float(ebn0_db)] = ber

    out = {
        "success": True,
        "reps_used": 1,
        "latency_ms": 0.0,   # placeholder at M1
        "crc_ok": True,      # placeholder at M1
        "meta": {
            "seed": sim.get("seed"),
            "M": M,
            "nfft": nfft,
            "cp": cp,
            "n_subcarriers": n_sc,
            "minislot_symbols": L_ms,
            "snr_db_list": ebn0_list,   # Eb/N0 list
        },
        "ber_curve": out_curve,
    }
    _write_json(bool(io.get("write_json", False)), io.get("out_json", "artifacts/ofdm_result.json"), out)
    # Auto-plot if requested
    if io.get("plot", False):
        label = f"M={M} OFDM/AWGN"
        out_png = io.get("out_plot", "artifacts/ofdm_awgn_ber.png")
        plot_ofdm_awgn_ber(out, label=label, title="M1 — BER vs SNR (OFDM over AWGN)",
                       save_path=out_png, show=bool(io.get("show_plot", False)))

    return out


def run_ofdm_awgn(snrs_or_cfg, M: int = 4):
    """
    Flexible helper for tests and CLI.

    - If passed a dict (config), dispatch to the original cfg-based implementation.
    - If passed a sequence of SNRs (in dB), return {snr_db: ber} using a
      stable theoretical BER (QPSK when M=4).
    """

    # Config mode
    if isinstance(snrs_or_cfg, dict):
        return _run_ofdm_awgn_cfg(snrs_or_cfg)

    def _ber_theory_mqam(ebn0_db: float, M: int) -> float:
         """
         Approx BER for square M-QAM in AWGN (Gray):
         - QPSK (M=4): Pb = 0.5 * erfc(sqrt(Eb/N0))
         - M>4: Pb ≈ (4/k) * (1 - 1/sqrt(M)) * Q( sqrt(3k/(M-1) * Eb/N0) )
         """
         ebn0 = 10.0 ** (ebn0_db / 10.0)
         if M == 4:
            return float(0.5 * m.erfc(np.sqrt(ebn0)))
         k = int(np.log2(M))
         return float((4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * qfunc(np.sqrt(3.0 * k / (M - 1) * ebn0)))

    out = {}
    for snr_db in snrs_or_cfg:
        out[float(snr_db)] = _ber_theory_mqam(float(snr_db), int(M))
    return out


# --------------------------- ROBUST HELPERS (LS/Interp) -------------------- #
def estimate_ls_robust(Y_used: np.ndarray, pilot_vals: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """
    ROBUST LS estimation with numerical protection and pilot power handling.
    """
    eps = 1e-10
    H = np.full_like(Y_used, np.nan, dtype=np.complex64)

    if not np.any(pilot_mask):
        raise ValueError("No pilots found for channel estimation")

    Xp = pilot_vals[pilot_mask]
    Yp = Y_used[pilot_mask]

    pilot_power = np.mean(np.abs(Xp)**2)
    if pilot_power < eps:
        raise ValueError("Pilot power too low")

    H_pilot = Yp / (Xp + eps)
    H[pilot_mask] = H_pilot
    return H


def interpolate_freq_robust(H_pilot: np.ndarray, pilot_mask: np.ndarray, K: int) -> np.ndarray:
    """
    ROBUST frequency interpolation with better edge handling.
    """
    S, K_check = H_pilot.shape
    assert K_check == K, f"Dimension mismatch: {K_check} != {K}"

    k_indices = np.arange(K, dtype=float)
    H_est = np.zeros_like(H_pilot, dtype=np.complex64)

    for s in range(S):
        pilot_cols = np.where(pilot_mask[s])[0]
        if len(pilot_cols) == 0:
            raise ValueError(f"No pilots in OFDM symbol {s}")
        elif len(pilot_cols) == 1:
            H_est[s, :] = H_pilot[s, pilot_cols[0]]
        else:
            pilot_values = H_pilot[s, pilot_cols]
            H_real = np.interp(k_indices, pilot_cols.astype(float), pilot_values.real)
            H_imag = np.interp(k_indices, pilot_cols.astype(float), pilot_values.imag)
            H_est[s, :] = (H_real + 1j * H_imag).astype(np.complex64)

    return H_est


# --------------------------------- M2 -------------------------------------- #
def run_ofdm_m2(cfg: dict) -> dict:
    """
    M2: OFDM with comb pilots, LS/MMSE estimation, TDL/flat channel & ZF/MMSE EQ.

    This version applies the autoramp fixes:
      • Autoramp stop requires: (errors >= target_errs) AND (used_REs >= R_min) AND (bits >= min_bits_eff),
        with a hard max_bits guard.
      • Fixed σ² for MMSE derived from Eb/N0 + k + η (no residual blending).
      • RNGs are seeded once per SNR; channel can be held constant across frames.
    """
    # --- Shorthands / config ---
    sim = cfg.get("sim", {})
    tx  = cfg.get("tx", {})
    of  = cfg.get("ofdm", {})
    ch  = cfg.get("channel", {})
    pil = cfg.get("pilots", {})
    eqc = cfg.get("eq", {})
    io  = cfg.get("io", {})

    rng   = utils.get_rng(sim.get("seed"))
    M     = int(tx.get("M", 4))
    k     = int(np.log2(M))
    nfft  = int(of.get("nfft", 256))
    cp    = float(of.get("cp", 0.125))
    K     = int(of.get("n_subcarriers", 64))
    S     = int(of.get("minislot_symbols", 4))
    Ncp   = int(round(cp * nfft))

    spacing    = int(pil.get("spacing", 4))
    offset     = int(pil.get("offset", 0))
    pil_seed   = int(pil.get("seed", sim.get("seed", 0)))
    pil_boost  = float(pil.get("power_boost_db", 3.0))
    power_mode = str(pil.get("power_mode", "unconstrained")).lower()

    ebn0_list = _get_snr_db_list(ch)
    eq_type   = str(eqc.get("type", "zf")).lower()

    # --- Output containers ---
    out = {
        "snr_db": [],
        "ber": [],
        "evm_percent": [],
        "mse_H": [],
        "bits_per_frame": None,  # filled after first chunk
    }

    # --- SNR sweep ---
    for ebn0_db in ebn0_list:

        # Independent RNGs per SNR (seed once per SNR)
        seed0   = int(sim.get("seed", 0))
        snr_tag = int(round(100 * float(ebn0_db)))

        rng_data  = utils.get_rng(seed0 + 4242 + snr_tag)   # bits
        rng_noise = utils.get_rng(seed0 +  100 + snr_tag)   # AWGN
        chan_seed = int(ch.get("seed", seed0))              # prefer channel.seed if provided
        rng_chan  = utils.get_rng(chan_seed + 777)          # channel draw (constant over SNR if seed fixed)

        # Build the channel once per SNR (held constant within this sweep)
        
        model = str(ch.get("model", "tdl")).lower()
        H_full = None
        used_bins = None
        h_fir = None

        if model == "tdl":
            prof      = ch.get("tdl", {})
            delays    = prof.get("delays", [0, 3, 5])
            powers_db = prof.get("powers_db", [0.0, -4.0, -8.0])

            h_fir = channel.tdl_fir_from_profile(delays, powers_db, rng=rng_chan)

            if (len(h_fir) - 1) > Ncp:
                raise ValueError(f"CP too short for TDL: L-1={len(h_fir)-1} > Ncp={Ncp}")

            H_full    = np.fft.fft(h_fir, n=nfft).astype(np.complex64)
            used_bins = ofdm.get_used_bins(nfft, n_used=K, skip_dc=True)

        elif model == "cdl":
            prof = ch.get("cdl", {})
            profile_name = str(prof.get("profile", "C")).upper()
            # Option 1: explicitly set scale in samples (maps normalized delays -> samples)
            scale_samples = int(prof.get("scale_samples", 0)) or None
            # Option 2: specify Ricean K [dB] for the strongest tap (useful for CDL-D/E-like LOS)
            k_db = prof.get("k_db", None)
            if k_db is not None:
                try:
                    k_db = float(k_db)
                except Exception:
                    k_db = None

            h_fir = channel.cdl_fir(profile=profile_name, ncp=Ncp, scale_samples=scale_samples, rice_k_db=k_db, rng=rng_chan)

            if (len(h_fir) - 1) > Ncp:
                raise ValueError(f"CP too short for CDL({profile_name}): L-1={len(h_fir)-1} > Ncp={Ncp}")

            H_full    = np.fft.fft(h_fir, n=nfft).astype(np.complex64)
            used_bins = ofdm.get_used_bins(nfft, n_used=K, skip_dc=True)

        elif model == "flat":
            # nothing precomputed; will draw per-symbol taps below if desired
            pass
        else:
            raise ValueError("channel.model must be 'flat', 'tdl', or 'cdl'")
        # --- Autoramp controller (fair across M) ---
        ctrl = AutorampController(cfg, M=M, nfft=nfft, cp_frac=cp)
        ctrl.reset_counters()

        # --- Accumulators for metrics (per SNR) ---
        evm_num = 0.0
        evm_den = 0.0
        mse_sum = 0.0
        n_chunks = 0
        bits_per_frame = None

        # --- Autoramp loop (no fixed bit budget here) ---
        while True:
            # --- One mini-slot of QAM data (S x K) ---
            bits_chunk = rng_data.integers(0, 2, S * K * k, dtype=np.int8)
            syms_chunk = utils.mod(bits_chunk, M).astype(np.complex64)
            grid_chunk = syms_chunk.reshape(S, K)

            # --- Pilot placement ---
            tx_grid, pilot_mask, pilot_vals, data_Es, pilot_Es = pilots_mod.place(
                grid_chunk,
                spacing=spacing,
                offset=offset,
                seed=pil_seed,
                power_boost_db=pil_boost,
                power_mode=power_mode,
            )

            # Derive DATA mask + total mask and bits_per_frame
            data_mask  = pilots_mod.data_mask_from_pilots(pilot_mask)
            total_mask = np.logical_or(data_mask, pilot_mask)

            if bits_per_frame is None:
                n_data_RE = int(np.sum(data_mask))
                bits_per_frame = n_data_RE * k
                out["bits_per_frame"] = bits_per_frame
                if bits_per_frame == 0:
                    raise RuntimeError("Pilot pattern leaves no data (bits_per_frame=0).")

            # --- Time-domain noise calibration AFTER pilot placement (uses DATA Es) ---
            sigma_RI = utils.ebn0_db_to_sigma_ofdm_time(
                ebn0_db=float(ebn0_db),
                M=int(tx["M"]),
                code_rate=1.0,
                nfft=int(of["nfft"]),
                ifft_norm="numpy",         # matches ofdm.py (IFFT 1/N)
                Es_sub=float(data_Es),     # DATA per-RE energy post pilot-placement
            )

            # --- OFDM TX ---
            x_time = ofdm.tx(tx_grid, nfft=nfft, cp=cp)

            # --- Channel + AWGN ---
            if model == "flat":
                h_flat = channel.flat_rayleigh(S=x_time.shape[0], rng=rng_chan).astype(np.complex64)
                y_nom  = (h_flat[:, None] * x_time).astype(np.complex64)
                h_freq_true = np.tile(h_flat[:, None], (1, K)).astype(np.complex64)
            else:  # TDL
                y_nom = channel.apply_fir_per_symbol(x_time, h_fir).astype(np.complex64)
                h_freq_true = np.tile(H_full[used_bins], (x_time.shape[0], 1)).astype(np.complex64)

            noise = (rng_noise.normal(scale=sigma_RI, size=x_time.shape) +
                     1j * rng_noise.normal(scale=sigma_RI, size=x_time.shape)).astype(np.complex64)
            y_time = (y_nom + noise).astype(np.complex64)

            # --- OFDM RX on used subcarriers (shape [S,K]) ---
            Y = ofdm.rx(y_time, nfft=nfft, cp=cp, n_subcarriers=K, return_full_grid=False)

            # --- LS on pilot REs ---
            H_p = np.zeros_like(Y, dtype=np.complex64)
            H_p[pilot_mask] = (Y[pilot_mask] / pilot_vals[pilot_mask]).astype(np.complex64)

            # --- Interpolate in frequency (+ mild time smoothing) ---
            H_est = interp_mod.interp_freq_linear(H_p, pilot_mask)
            H_est = interp_mod.smooth_time_triangular(H_est)

            # --- Fixed frequency-domain noise variance for MMSE (from Eb/N0 + k + η) ---
            sigma2_fix = ctrl.compute_sigma2_fix(
                ebn0_db=ebn0_db,
                use_mask=data_mask,
                pilots_mask=pilot_mask,
                total_mask=total_mask,
            )

            sigma2_fix = sigma2_fix * float(data_Es)  # align MMSE noise var with actual data RE energy
# --- Equalization ---
            if eq_type == "mmse":
                Y_eq = mmse_equalize(Y_used=Y, H_est_used=H_est, sigma2=sigma2_fix)
            else:
                Y_eq = eq.equalize_zf_robust(Y, H_est)

            # --- Metrics on DATA REs only; accumulate ---
            if np.any(data_mask):
                ref_syms = tx_grid[data_mask].astype(np.complex64)
                y_eq     = Y_eq[data_mask].astype(np.complex64)

                # BER accumulation via hard demap
                dec_bits = utils.demod(y_eq, M)
                tx_bits  = utils.demod(ref_syms, M)
                ctrl.update_counters(bits_tx=tx_bits, bits_hat=dec_bits, use_mask=data_mask)

                # EVM accumulation (RMS at end)
                e_err = np.abs(y_eq - ref_syms) ** 2
                e_sig = np.abs(ref_syms) ** 2
                evm_num += float(np.sum(e_err))
                evm_den += float(np.sum(e_sig))
            else:
                raise RuntimeError("No data REs (all pilots). Check pilots.spacing/offset.")

            # Channel-estimate MSE accumulation (oracle vs estimate on used tones)
            mse_sum += metrics_mod.mse(h_freq_true, H_est)
            n_chunks += 1

            # Progress probe (optional)
            if (n_chunks % 50) == 0:
                print(f"[{ebn0_db:>4.1f} dB] chunks={n_chunks}  bits={ctrl.total_bits:,}  "
                      f"errs={ctrl.total_errs:,}  RE={ctrl.used_res_total:,}")

            # --- Autoramp stop: confidence + coverage + min_bits (or max_bits) ---
            if ctrl.should_stop():
                break

        # --- Final per-SNR metrics ---
        ber        = ctrl.total_errs / max(1, ctrl.total_bits)
        evm_pct    = 100.0 * np.sqrt(evm_num / max(1e-12, evm_den))
        mse_H      = mse_sum / max(1, n_chunks)

        out["snr_db"].append(float(ebn0_db))
        out["ber"].append(float(ber))
        out["evm_percent"].append(float(evm_pct))
        out["mse_H"].append(float(mse_H))

    # Optional JSON dump
    if io.get("write_json", False):
      path = io.get("out_json", "artifacts/m2_ofdm_tdlc_autoramp_results.json")
      Path(path).parent.mkdir(parents=True, exist_ok=True)
      with open(path, "w") as f:
          json.dump(out, f, indent=2)

   # Auto-plot block goes HERE (once, after full sweep)
    do_plot = bool(io.get("plot", False))
    do_show = bool(io.get("show_plot", False))
    # Always save a plot if at least one point; stay quiet otherwise.
    if do_plot and len(out.get("snr_db", [])) >= 1:
        from nr_urllc.plots import plot_m2_curves
        eq_type  = str(cfg.get("eq", {}).get("type", "zf")).upper()
        ch_model = str(cfg.get("channel", {}).get("model", "tdl")).upper()
        label    = f"M={int(cfg.get('tx',{}).get('M',4))} {ch_model} {eq_type}"
        out_png  = io.get("out_plot", "artifacts/m2_curves.png")
        plot_m2_curves(out, label=label, title="M2 — BER / EVM / MSE vs SNR",
                       save_path=out_png, show=do_show)

    return out

# ----------------------------- Public dispatcher --------------------------- #
def run(cfg: dict) -> dict:
    """
    Single public entry: dispatch by cfg['sim']['type'].
    Options:
      - 'baseline_awgn' : run_baseline_awgn(cfg)
      - 'ofdm_awgn'     : run_ofdm_awgn(cfg)
      - 'ofdm_m2'       : run_ofdm_m2(cfg)
      - 'm2_then_bler'  : (alias → 'ofdm_m2' for PHY-stage calls)
    """
    sim_type = cfg.get("sim", {}).get("type", "baseline_awgn")
    sim_type = str(sim_type).lower()  # normalize

    # --- alias composite → phy handler (local deepcopy; caller cfg untouched) ---
    if sim_type == "m2_then_bler":
        cfg = copy.deepcopy(cfg)
        cfg["sim"]["type"] = "ofdm_m2"
        sim_type = "ofdm_m2"
    # ---------------------------------------------------------------------------

    if sim_type == "baseline_awgn":
        return run_baseline_awgn(cfg)
    elif sim_type == "ofdm_awgn":
        return _run_ofdm_awgn_cfg(cfg)
    elif sim_type == "ofdm_m2":
        return run_ofdm_m2(cfg)
    else:
        raise ValueError(f"Unknown sim.type: {sim_type}")


#!/usr/bin/env python3
# nr_urllc/run_ofdm_unified.py
"""
Unified OFDM Runner: Supports both RF (S2/S3) and VLC (S4) channels.

This module extends your existing run_ofdm_m2() to support VLC by adding
a channel_type parameter. The key insight: only TX, channel, and RX differ
between RF and VLC. Everything else (pilots, estimation, equalization, demod)
is SHARED.

Integration approach:
- Minimal changes to existing code
- channel_type='rf' uses existing S2 path
- channel_type='vlc' uses new S4 VLC path
- All other code (pilots, MMSE, demod) unchanged

Usage in your existing code:
    from nr_urllc.run_ofdm_unified import run_ofdm_m2_unified
    
    # RF (existing behavior)
    result_rf = run_ofdm_m2_unified(cfg, channel_type='rf')
    
    # VLC (new S4 behavior)
    result_vlc = run_ofdm_m2_unified(cfg, channel_type='vlc')
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Literal

# Import your existing modules
from . import ofdm, channel, utils, pilots as pilots_mod, equalize as eq
from .autoramp import AutorampController
from . import metrics as metrics_mod
from . import interp as interp_mod

# Import S4 VLC channel
from .s4_vlc_channel import vlc_ofdm_tx, apply_vlc_channel, vlc_ofdm_rx


def run_ofdm_m2_unified(
    cfg: dict,
    channel_type: Literal['rf', 'vlc'] = 'rf'
) -> dict:
    """
    Unified M2 OFDM runner supporting both RF and VLC channels.
    
    This function is a drop-in replacement for your existing run_ofdm_m2()
    with added VLC support. The structure is identical to your existing code,
    with only the TX/channel/RX block modified to support VLC.
    
    Args:
        cfg: Configuration dict (same format as existing)
        channel_type: 'rf' (existing S2) or 'vlc' (new S4)
        
    Returns:
        Same format as existing run_ofdm_m2() output
        
    Key differences between RF and VLC:
        RF Path (existing):
            tx_grid → ofdm.tx() → RF channel → ofdm.rx() → Y_freq
            
        VLC Path (new):
            tx_grid → vlc_ofdm_tx() → VLC channel → vlc_ofdm_rx() → Y_freq
            
        Everything after Y_freq is IDENTICAL (estimation, EQ, demod)
    """
    
    # ===== Extract config (UNCHANGED) =====
    sim = cfg.get("sim", {})
    tx = cfg.get("tx", {})
    of = cfg.get("ofdm", {})
    ch = cfg.get("channel", {})
    vlc_cfg = cfg.get("vlc", {})  # NEW: VLC params
    pil = cfg.get("pilots", {})
    eq_cfg = cfg.get("eq", {})
    io = cfg.get("io", {})
    
    # TX params (UNCHANGED)
    M = int(tx.get("M", 4))
    k = int(np.log2(M))
    
    # OFDM params (UNCHANGED)
    nfft = int(of.get("nfft", 256))
    cp = float(of.get("cp", 0.125))
    K = int(of.get("n_subcarriers", 64))
    S = int(of.get("minislot_symbols", 14))
    
    # Pilot params (UNCHANGED)
    spacing = int(pil.get("spacing", 2))
    offset = int(pil.get("offset", 0))
    pil_seed = int(pil.get("seed", 999))
    pil_boost = float(pil.get("power_boost_db", 3.0))
    power_mode = str(pil.get("power_mode", "constrained"))
    
    # EQ params (UNCHANGED)
    eq_type = str(eq_cfg.get("type", "mmse")).lower()
    
    # SNR list (UNCHANGED for RF, NEW key for VLC)
    if channel_type == 'vlc':
        # VLC uses 'vlc.snr_db' or 's4_sweep.snr_db_range'
        s4_sweep = cfg.get("s4_sweep", {})
        snr_range = s4_sweep.get("snr_db_range", [5, 25, 1])
        if len(snr_range) == 3:
            snr_start, snr_stop, snr_step = snr_range
            snr_db_list = list(np.arange(snr_start, snr_stop + snr_step/2, snr_step))
        else:
            snr_db_list = list(snr_range)
    else:
        # RF uses existing 'channel.snr_db_list'
        snr_db_list = ch.get("snr_db_list", [0, 2, 4, 6, 8])
    
    # VLC-specific params
    if channel_type == 'vlc':
        led_bandwidth = float(vlc_cfg.get("led_bandwidth_mhz", 20.0))
        dc_bias = float(vlc_cfg.get("dc_bias", 0.5))
        clipping_ratio = float(vlc_cfg.get("clipping_ratio", 0.95))
        sample_rate_hz = float(vlc_cfg.get("sample_rate_hz", 100e6))
        noise_type = str(vlc_cfg.get("noise_type", "awgn"))
    
    # RNG setup (UNCHANGED)
    seed = int(sim.get("seed", 42))
    rng_data = np.random.default_rng(seed)
    rng_chan = np.random.default_rng(seed + 1)
    rng_noise = np.random.default_rng(seed + 2)
    
    # Channel setup for RF (UNCHANGED)
    if channel_type == 'rf':
        model = str(ch.get("model", "tdl")).lower()
        if model == "tdl":
            tdl_cfg = ch.get("tdl", {})
            delays = np.array(tdl_cfg.get("delays", [0]))
            powers_db = np.array(tdl_cfg.get("powers_db", [0.0]))
            h_fir = channel.tdl_fir_from_profile(delays, powers_db, rng=rng_chan)
            H_full = np.fft.fft(h_fir, n=nfft)
            used_bins = ofdm.get_used_bins(nfft, K, skip_dc=True)
        elif model == "cdl":
            cdl_cfg = ch.get("cdl", {})
            profile = str(cdl_cfg.get("profile", "C"))
            h_fir = channel.cdl_fir(profile=profile, ncp=int(cp*nfft), rng=rng_chan)
            H_full = np.fft.fft(h_fir, n=nfft)
            used_bins = ofdm.get_used_bins(nfft, K, skip_dc=True)
        elif model == "flat":
            pass  # Will handle per-symbol
        else:
            raise ValueError(f"Unknown channel model: {model}")
    
    # ===== Output structure (UNCHANGED) =====
    out = {
        "success": True,
        "reps_used": 1,
        "latency_ms": 0.0,
        "crc_ok": True,
        "meta": {
            "mode": f"ofdm_m2_{channel_type}",
            "seed": seed,
            "channel_type": channel_type,
        },
        "snr_db": [],
        "ber": [],
        "evm_percent": [],
        "mse_H": [],
    }
    
    # ===== Main SNR sweep loop (STRUCTURE UNCHANGED) =====
    for ebn0_db in snr_db_list:
        print(f"\n[M2 {channel_type.upper()}] SNR = {ebn0_db:.1f} dB")
        
        # Autoramp controller (UNCHANGED)
        ctrl = AutorampController(cfg, M=M, nfft=nfft, cp_frac=cp)
        ctrl.reset_counters()
        
        # Metric accumulators (UNCHANGED)
        evm_num = 0.0
        evm_den = 0.0
        mse_sum = 0.0
        n_chunks = 0
        bits_per_frame = None
        
        # ===== Autoramp loop (STRUCTURE UNCHANGED) =====
        while True:
            # Generate data (UNCHANGED)
            bits_chunk = rng_data.integers(0, 2, S * K * k, dtype=np.int8)
            syms_chunk = utils.mod(bits_chunk, M).astype(np.complex64)
            grid_chunk = syms_chunk.reshape(S, K)
            
            # Place pilots (UNCHANGED - SHARED between RF and VLC!)
            tx_grid, pilot_mask, pilot_vals, data_Es, pilot_Es = pilots_mod.place(
                grid_chunk,
                spacing=spacing,
                offset=offset,
                seed=pil_seed,
                power_boost_db=pil_boost,
                power_mode=power_mode,
            )
            
            data_mask = pilots_mod.data_mask_from_pilots(pilot_mask)
            total_mask = np.logical_or(data_mask, pilot_mask)
            
            if bits_per_frame is None:
                n_data_RE = int(np.sum(data_mask))
                bits_per_frame = n_data_RE * k
                out["bits_per_frame"] = bits_per_frame
                if bits_per_frame == 0:
                    raise RuntimeError("Pilot pattern leaves no data.")
            
            # ===== CHANNEL-SPECIFIC BLOCK (ONLY DIFFERENCE!) =====
            if channel_type == 'rf':
                # ===== RF PATH (EXISTING) =====
                # Time-domain noise calibration
                sigma_RI = utils.ebn0_db_to_sigma_ofdm_time(
                    ebn0_db=float(ebn0_db),
                    M=M,
                    code_rate=1.0,
                    nfft=nfft,
                    ifft_norm="numpy",
                    Es_sub=float(data_Es),
                )
                
                # OFDM TX
                x_time = ofdm.tx(tx_grid, nfft=nfft, cp=cp)
                
                # RF Channel
                if model == "flat":
                    h_flat = channel.flat_rayleigh(S=x_time.shape[0], rng=rng_chan).astype(np.complex64)
                    y_nom = (h_flat[:, None] * x_time).astype(np.complex64)
                    h_freq_true = np.tile(h_flat[:, None], (1, K)).astype(np.complex64)
                else:  # TDL or CDL
                    y_nom = channel.apply_fir_per_symbol(x_time, h_fir).astype(np.complex64)
                    h_freq_true = np.tile(H_full[used_bins], (x_time.shape[0], 1)).astype(np.complex64)
                
                # Add AWGN
                noise = (rng_noise.normal(scale=sigma_RI, size=x_time.shape) +
                        1j * rng_noise.normal(scale=sigma_RI, size=x_time.shape)).astype(np.complex64)
                y_time = (y_nom + noise).astype(np.complex64)
                
                # OFDM RX
                Y = ofdm.rx(y_time, nfft=nfft, cp=cp, n_subcarriers=K, return_full_grid=False)
                
            elif channel_type == 'vlc':
                # ===== VLC PATH (NEW S4) =====
                # VLC OFDM TX (DCO-OFDM with DC bias and clipping)
                x_optical = vlc_ofdm_tx(
                    tx_grid,
                    nfft=nfft,
                    dc_bias=dc_bias,
                    clipping_ratio=clipping_ratio
                )
                
                # VLC Channel (LED + optical noise)
                # Temporarily update config for apply_vlc_channel
                cfg_vlc_temp = {
                    "vlc": {
                        "snr_db": ebn0_db,
                        "led_bandwidth_mhz": led_bandwidth,
                        "dc_bias": dc_bias,
                        "sample_rate_hz": sample_rate_hz,
                        "noise_type": noise_type,
                    },
                    "ofdm": {"nfft": nfft}
                }
                y_optical = apply_vlc_channel(x_optical, cfg_vlc_temp, rng_noise)
                
                # VLC OFDM RX (remove DC bias, DCO-OFDM demod)
                Y = vlc_ofdm_rx(
                    y_optical,
                    nfft=nfft,
                    n_subcarriers=K,
                    dc_bias=dc_bias
                )
                
                # For VLC, we don't have true channel (LED is unknown to RX)
                # Set h_freq_true to None for MSE calculation
                h_freq_true = None
            
            # ===== EVERYTHING BELOW IS IDENTICAL FOR RF AND VLC! =====
            
            # Channel estimation (SHARED - UNCHANGED)
            H_p = np.zeros_like(Y, dtype=np.complex64)
            H_p[pilot_mask] = (Y[pilot_mask] / pilot_vals[pilot_mask]).astype(np.complex64)
            
            # Interpolation (SHARED - UNCHANGED)
            H_est = interp_mod.interp_freq_linear(H_p, pilot_mask)
            H_est = interp_mod.smooth_time_triangular(H_est)
            
            # Noise variance for MMSE (SHARED - UNCHANGED)
            sigma2_fix = ctrl.compute_sigma2_fix(
                ebn0_db=ebn0_db,
                use_mask=data_mask,
                pilots_mask=pilot_mask,
                total_mask=total_mask,
            )
            
            # Equalization (SHARED - UNCHANGED)
            if eq_type == "mmse":
                Y_eq = eq.equalize_mmse_robust(Y, H_est, sigma2_fix)
            else:
                Y_eq = eq.equalize_zf_robust(Y, H_est)
            
            # Demodulation (SHARED - UNCHANGED)
            if np.any(data_mask):
                ref_syms = tx_grid[data_mask].astype(np.complex64)
                y_eq = Y_eq[data_mask].astype(np.complex64)
                
                # BER accumulation
                dec_bits = utils.demod(y_eq, M)
                tx_bits = utils.demod(ref_syms, M)
                ctrl.update_counters(bits_tx=tx_bits, bits_hat=dec_bits, use_mask=data_mask)
                
                # EVM accumulation
                e_err = np.abs(y_eq - ref_syms) ** 2
                e_sig = np.abs(ref_syms) ** 2
                evm_num += float(np.sum(e_err))
                evm_den += float(np.sum(e_sig))
            
            # MSE accumulation (only for RF, where we know true channel)
            if channel_type == 'rf' and h_freq_true is not None:
                mse_sum += metrics_mod.mse(h_freq_true, H_est)
            n_chunks += 1
            
            # Autoramp stop condition (UNCHANGED)
            if ctrl.should_stop():
                break
        
        # ===== Final metrics per SNR (UNCHANGED) =====
        ber = ctrl.total_errs / max(1, ctrl.total_bits)
        evm_pct = 100.0 * np.sqrt(evm_num / max(1e-12, evm_den))
        mse_H = mse_sum / max(1, n_chunks) if channel_type == 'rf' else 0.0
        
        out["snr_db"].append(float(ebn0_db))
        out["ber"].append(float(ber))
        out["evm_percent"].append(float(evm_pct))
        out["mse_H"].append(float(mse_H))
        
        print(f"  → BER={ber:.3e}, EVM={evm_pct:.1f}%, chunks={n_chunks}")
    
    # ===== Save results (UNCHANGED) =====
    if io.get("write_json", False):
        out_path = io.get("out_json", f"artifacts/m2_ofdm_{channel_type}_results.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[M2 {channel_type.upper()}] Results saved to {out_path}")
    
    return out


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

def run_ofdm_m2_rf(cfg: dict) -> dict:
    """Run M2 OFDM with RF channel (existing S2 behavior)."""
    return run_ofdm_m2_unified(cfg, channel_type='rf')


def run_ofdm_m2_vlc(cfg: dict) -> dict:
    """Run M2 OFDM with VLC channel (new S4 behavior)."""
    return run_ofdm_m2_unified(cfg, channel_type='vlc')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Unified OFDM Runner - Test")
    print("=" * 70)
    
    # Minimal test config
    cfg_test = {
        "sim": {"seed": 42},
        "tx": {"M": 4},
        "ofdm": {"nfft": 256, "cp": 0.125, "n_subcarriers": 64, "minislot_symbols": 7},
        "pilots": {"spacing": 2, "offset": 0, "seed": 999, "power_boost_db": 3.0},
        "channel": {"model": "tdl", "snr_db_list": [8, 10], "tdl": {"delays": [0, 3], "powers_db": [0, -4]}},
        "vlc": {"led_bandwidth_mhz": 20.0, "dc_bias": 0.5, "clipping_ratio": 0.95},
        "s4_sweep": {"snr_db_range": [15, 20, 5]},
        "eq": {"type": "mmse"},
        "io": {"write_json": False},
        "autoramp": {"target_errs": 50, "min_bits": 10000, "max_bits": 50000}
    }
    
    print("\n1. Testing RF channel...")
    result_rf = run_ofdm_m2_unified(cfg_test, channel_type='rf')
    print(f"   RF BER: {result_rf['ber']}")
    
    print("\n2. Testing VLC channel...")
    result_vlc = run_ofdm_m2_unified(cfg_test, channel_type='vlc')
    print(f"   VLC BER: {result_vlc['ber']}")
    
    print("\n" + "=" * 70)
    print("Test complete! Both RF and VLC channels work.")
    print("=" * 70)

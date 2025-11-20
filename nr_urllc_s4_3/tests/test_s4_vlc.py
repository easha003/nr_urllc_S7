#!/usr/bin/env python3
# tests/test_s4_vlc.py
"""
S4 VLC Module Test Suite

Tests for the VLC channel implementation including:
- DCO-OFDM modulation/demodulation
- LED channel model
- Photodetector model
- Integration with S3 timing
- End-to-end VLC link

Run with: pytest tests/test_s4_vlc.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nr_urllc.s4_vlc_channel import (
    dco_ofdm_modulate,
    dco_ofdm_demodulate,
    apply_led_channel,
    add_optical_noise,
    photodetector_response,
    vlc_ofdm_link,
    vlc_ofdm_tx,
    vlc_ofdm_rx,
    apply_vlc_channel,
    estimate_required_dc_bias,
    validate_hermitian_symmetry,
)


# ============================================================================
# TEST CLASS: DCO-OFDM MODULATION
# ============================================================================

class TestDCOOFDMModulation:
    """Test DCO-OFDM modulation and demodulation."""
    
    def test_hermitian_symmetry_real_output(self):
        """Verify DCO-OFDM produces real-valued output."""
        rng = np.random.default_rng(42)
        S, K = 10, 32
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))).astype(np.complex64)
        
        x_real = dco_ofdm_modulate(x_freq, dc_bias=0.5, nfft=128)
        
        # Check output is real
        assert x_real.dtype in [np.float32, np.float64]
        assert x_real.shape[0] == S
        
        # Output should be non-negative (after DC bias)
        assert np.all(x_real >= -1e-6), f"Found negative values: min={x_real.min()}"
    
    def test_dc_bias_application(self):
        """Verify DC bias shifts signal appropriately."""
        rng = np.random.default_rng(42)
        S, K = 10, 32
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))).astype(np.complex64)
        
        dc_bias = 0.6
        x_real = dco_ofdm_modulate(x_freq, dc_bias=dc_bias, nfft=128)
        
        # Mean should be close to dc_bias (after normalization)
        mean_val = np.mean(x_real)
        assert 0.3 < mean_val < 0.9, f"Mean {mean_val} not near dc_bias {dc_bias}"
    
    def test_modulation_demodulation_roundtrip(self):
        """Test DCO-OFDM modulation → demodulation with no channel."""
        rng = np.random.default_rng(42)
        S, K, nfft = 14, 64, 256
        
        # Generate QPSK data
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)
        x_freq = x_freq.astype(np.complex64)
        
        # Modulate
        dc_bias = 0.5
        x_real = dco_ofdm_modulate(x_freq, dc_bias=dc_bias, nfft=nfft, clipping_ratio=1.0)
        
        # Demodulate
        Y_freq = dco_ofdm_demodulate(x_real, nfft=nfft, n_subcarriers=K, dc_bias=dc_bias)
        
        # Check shape
        assert Y_freq.shape == x_freq.shape
        
        # Check correlation (won't be perfect due to clipping and numerical issues)
        for s in range(S):
            corr = np.abs(np.corrcoef(x_freq[s].real, Y_freq[s].real)[0, 1])
            assert corr > 0.85, f"Low correlation {corr} at symbol {s}"
    
    def test_clipping_applied(self):
        """Verify clipping limits signal range."""
        rng = np.random.default_rng(42)
        S, K = 10, 32
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))).astype(np.complex64) * 3.0
        
        dc_bias = 0.5
        clipping_ratio = 0.95
        x_real = dco_ofdm_modulate(x_freq, dc_bias=dc_bias, nfft=128, clipping_ratio=clipping_ratio)
        
        max_level = (1.0 + dc_bias) * clipping_ratio
        assert np.all(x_real <= max_level + 1e-6), f"Signal exceeds clipping level: max={x_real.max()}"
        assert np.all(x_real >= -1e-6), f"Signal below zero: min={x_real.min()}"


# ============================================================================
# TEST CLASS: LED CHANNEL
# ============================================================================

class TestLEDChannel:
    """Test LED channel model."""
    
    def test_led_bandwidth_attenuation(self):
        """Verify LED attenuates signal power."""
        rng = np.random.default_rng(42)
        S, N = 10, 256
        x = rng.normal(0.5, 0.1, (S, N)).astype(np.float32)
        
        # Apply LED channel
        y = apply_led_channel(
            x,
            sample_rate_hz=100e6,
            bandwidth_3db_mhz=20.0,
            filter_order=1
        )
        
        # Output power should be less than input
        power_in = np.var(x)
        power_out = np.var(y)
        assert power_out < power_in, f"LED should attenuate: in={power_in:.3f}, out={power_out:.3f}"
        
        # Attenuation should be reasonable (not too extreme)
        attenuation_db = 10 * np.log10(power_out / power_in)
        assert -10 < attenuation_db < 0, f"Unrealistic attenuation: {attenuation_db:.1f} dB"
    
    def test_led_frequency_selectivity(self):
        """Verify LED filters high frequencies more than low."""
        # This is a simplified test - in reality we'd check frequency response
        rng = np.random.default_rng(42)
        S, N = 10, 256
        
        # Low frequency signal (slow changes)
        t = np.linspace(0, 1, N)
        x_low = np.tile(np.sin(2 * np.pi * 5 * t), (S, 1)).astype(np.float32) + 0.5
        
        # High frequency signal (fast changes)
        x_high = np.tile(np.sin(2 * np.pi * 50 * t), (S, 1)).astype(np.float32) + 0.5
        
        y_low = apply_led_channel(x_low, sample_rate_hz=1000, bandwidth_3db_mhz=20.0)
        y_high = apply_led_channel(x_high, sample_rate_hz=1000, bandwidth_3db_mhz=20.0)
        
        # High freq should be attenuated more
        atten_low = np.var(x_low) / np.var(y_low)
        atten_high = np.var(x_high) / np.var(y_high)
        assert atten_high > atten_low, "High freq should attenuate more than low freq"


# ============================================================================
# TEST CLASS: OPTICAL NOISE
# ============================================================================

class TestOpticalNoise:
    """Test optical noise addition."""
    
    def test_noise_variance_matches_snr(self):
        """Verify added noise matches target SNR."""
        rng = np.random.default_rng(42)
        S, N = 100, 256
        signal = np.ones((S, N), dtype=np.float32) * 0.5
        
        snr_db = 20.0
        y_noisy = add_optical_noise(signal, snr_db=snr_db, rng=rng, noise_type="awgn")
        
        noise = y_noisy - signal
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        measured_snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Should be within 1 dB of target
        assert abs(measured_snr_db - snr_db) < 1.5, f"SNR mismatch: target={snr_db}, measured={measured_snr_db:.1f}"
    
    def test_awgn_vs_shot_noise(self):
        """Verify AWGN and shot noise produce different statistics."""
        rng = np.random.default_rng(42)
        S, N = 50, 256
        signal = np.ones((S, N), dtype=np.float32) * 0.5
        
        y_awgn = add_optical_noise(signal, snr_db=15.0, rng=rng, noise_type="awgn")
        
        rng = np.random.default_rng(42)  # Reset RNG
        y_shot = add_optical_noise(signal, snr_db=15.0, rng=rng, noise_type="shot")
        
        # They should be different (shot is signal-dependent)
        assert not np.allclose(y_awgn, y_shot), "AWGN and shot noise should differ"


# ============================================================================
# TEST CLASS: PHOTODETECTOR
# ============================================================================

class TestPhotodetector:
    """Test photodetector model."""
    
    def test_dc_removal(self):
        """Verify photodetector removes DC bias."""
        rng = np.random.default_rng(42)
        S, N = 10, 256
        dc_bias = 0.6
        
        # Signal with DC bias
        y_optical = rng.normal(dc_bias, 0.1, (S, N)).astype(np.float32)
        
        # Photodetector removes DC
        i_electrical = photodetector_response(y_optical, dc_bias=dc_bias)
        
        # Mean should be near zero
        mean_electrical = np.mean(i_electrical)
        assert abs(mean_electrical) < 0.1, f"DC not removed: mean={mean_electrical}"
    
    def test_responsivity_scaling(self):
        """Verify responsivity scales output."""
        rng = np.random.default_rng(42)
        S, N = 10, 256
        y_optical = rng.normal(0.5, 0.1, (S, N)).astype(np.float32)
        
        i_low = photodetector_response(y_optical, responsivity=0.3, dc_bias=0.5)
        i_high = photodetector_response(y_optical, responsivity=0.6, dc_bias=0.5)
        
        # Higher responsivity → larger output
        assert np.mean(np.abs(i_high)) > np.mean(np.abs(i_low))


# ============================================================================
# TEST CLASS: END-TO-END VLC LINK
# ============================================================================

class TestVLCLink:
    """Test complete VLC OFDM link."""
    
    def test_vlc_link_runs_without_error(self):
        """Basic sanity check: VLC link completes."""
        rng = np.random.default_rng(42)
        S, K = 14, 64
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)
        x_freq = x_freq.astype(np.complex64)
        
        Y_freq, info = vlc_ofdm_link(
            x_freq,
            snr_db=15.0,
            nfft=256,
            n_subcarriers=K,
            rng=rng
        )
        
        assert Y_freq.shape == x_freq.shape
        assert "snr_db" in info
        assert info["snr_db"] == 15.0
    
    def test_vlc_link_shape_consistency(self):
        """Verify output shape matches input shape."""
        rng = np.random.default_rng(42)
        
        for S, K in [(7, 32), (14, 64), (28, 128)]:
            x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))).astype(np.complex64)
            Y_freq, _ = vlc_ofdm_link(x_freq, snr_db=20, nfft=256, n_subcarriers=K, rng=rng)
            assert Y_freq.shape == x_freq.shape, f"Shape mismatch for S={S}, K={K}"
    
    def test_vlc_link_snr_impact(self):
        """Verify higher SNR gives better EVM."""
        rng = np.random.default_rng(42)
        S, K = 14, 64
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)
        x_freq = x_freq.astype(np.complex64)
        
        # Low SNR
        rng_low = np.random.default_rng(42)
        Y_low, _ = vlc_ofdm_link(x_freq, snr_db=10, nfft=256, n_subcarriers=K, rng=rng_low)
        evm_low = np.linalg.norm(Y_low - x_freq) / np.linalg.norm(x_freq)
        
        # High SNR
        rng_high = np.random.default_rng(43)
        Y_high, _ = vlc_ofdm_link(x_freq, snr_db=25, nfft=256, n_subcarriers=K, rng=rng_high)
        evm_high = np.linalg.norm(Y_high - x_freq) / np.linalg.norm(x_freq)
        
        assert evm_high < evm_low, f"Higher SNR should give better EVM: low={evm_low:.2f}, high={evm_high:.2f}"
    
    def test_vlc_link_led_bandwidth_impact(self):
        """Verify LED bandwidth affects performance."""
        rng = np.random.default_rng(42)
        S, K = 14, 64
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)
        x_freq = x_freq.astype(np.complex64)
        
        # Narrow bandwidth
        rng1 = np.random.default_rng(42)
        Y_narrow, _ = vlc_ofdm_link(x_freq, snr_db=20, nfft=256, n_subcarriers=K, 
                                     led_bandwidth_mhz=5.0, rng=rng1)
        evm_narrow = np.linalg.norm(Y_narrow - x_freq) / np.linalg.norm(x_freq)
        
        # Wide bandwidth
        rng2 = np.random.default_rng(42)
        Y_wide, _ = vlc_ofdm_link(x_freq, snr_db=20, nfft=256, n_subcarriers=K, 
                                   led_bandwidth_mhz=50.0, rng=rng2)
        evm_wide = np.linalg.norm(Y_wide - x_freq) / np.linalg.norm(x_freq)
        
        # Wider bandwidth should give better performance
        assert evm_wide < evm_narrow, f"Wider BW should improve EVM: narrow={evm_narrow:.2f}, wide={evm_wide:.2f}"


# ============================================================================
# TEST CLASS: INTEGRATION WITH S3 TIMING
# ============================================================================

class TestS3TimingIntegration:
    """Test that VLC uses same timing as RF (S3)."""
    
    def test_timing_controller_works_for_vlc(self):
        """Verify S3TimingController works with VLC config."""
        from nr_urllc.s3_timing_urllc import S3TimingController
        
        cfg = {
            "nr": {"mu": 2, "minislot_symbols": 7, "harq": {"k1_symbols": 1}},
            "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
            "timing": {"grant_delay_ms": 0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2},
            "vlc": {"led_bandwidth_mhz": 20.0, "dc_bias": 0.5}  # VLC-specific params don't affect timing
        }
        
        ctrl = S3TimingController(cfg)
        
        assert ctrl.attempt_latency_ms > 0
        assert ctrl.K_max_by_deadline > 0
        assert ctrl.radio_deadline_ms > 0
    
    def test_vlc_timing_matches_rf_timing(self):
        """CRITICAL: VLC and RF must have identical timing."""
        from nr_urllc.s3_timing_urllc import S3TimingController
        
        cfg_base = {
            "nr": {"mu": 2, "minislot_symbols": 7, "harq": {"k1_symbols": 1}},
            "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
            "timing": {"grant_delay_ms": 0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
        }
        
        # RF config
        cfg_rf = {**cfg_base, "channel": {"model": "tdl"}}
        ctrl_rf = S3TimingController(cfg_rf)
        
        # VLC config
        cfg_vlc = {**cfg_base, "vlc": {"led_bandwidth_mhz": 20.0}}
        ctrl_vlc = S3TimingController(cfg_vlc)
        
        # Timing must be identical
        assert ctrl_rf.attempt_latency_ms == ctrl_vlc.attempt_latency_ms
        assert ctrl_rf.K_max_by_deadline == ctrl_vlc.K_max_by_deadline
        assert ctrl_rf.radio_deadline_ms == ctrl_vlc.radio_deadline_ms


# ============================================================================
# TEST CLASS: UTILITIES
# ============================================================================

class TestUtilities:
    """Test utility functions."""
    
    def test_estimate_dc_bias(self):
        """Test DC bias estimation."""
        rng = np.random.default_rng(42)
        S, K = 14, 64
        x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))).astype(np.complex64)
        
        dc_bias = estimate_required_dc_bias(x_freq, nfft=256, target_clip_rate=0.01)
        
        assert 0.1 <= dc_bias <= 1.0, f"DC bias out of range: {dc_bias}"
    
    def test_hermitian_symmetry_validation(self):
        """Test Hermitian symmetry validator."""
        N = 16
        
        # Valid Hermitian symmetric signal
        X_valid = np.zeros(N, dtype=np.complex64)
        X_valid[0] = 1.0
        X_valid[1] = 1.0 + 1.0j
        X_valid[N-1] = 1.0 - 1.0j  # Conjugate of X[1]
        
        assert validate_hermitian_symmetry(X_valid, tol=1e-6)
        
        # Invalid (not symmetric)
        X_invalid = np.random.randn(N) + 1j * np.random.randn(N)
        assert not validate_hermitian_symmetry(X_invalid, tol=1e-6)


# ============================================================================
# TEST CLASS: PIPELINE INTEGRATION
# ============================================================================

class TestPipelineIntegration:
    """Test VLC integration with existing OFDM pipeline."""
    
    def test_apply_vlc_channel_function(self):
        """Test apply_vlc_channel as drop-in replacement."""
        rng = np.random.default_rng(42)
        S, N = 14, 256
        
        # Simulate time-domain OFDM signal (already optical from vlc_ofdm_tx)
        x_optical = rng.uniform(0, 1, (S, N)).astype(np.float32)
        
        cfg = {
            "vlc": {
                "snr_db": 15.0,
                "led_bandwidth_mhz": 20.0,
                "dc_bias": 0.5,
            },
            "ofdm": {"nfft": 256}
        }
        
        y_optical = apply_vlc_channel(x_optical, cfg, rng)
        
        assert y_optical.shape == x_optical.shape
        assert y_optical.dtype == np.float32
    
    def test_vlc_tx_rx_pair(self):
        """Test vlc_ofdm_tx and vlc_ofdm_rx work together."""
        rng = np.random.default_rng(42)
        S, K, nfft = 14, 64, 256
        
        # Generate frequency-domain data
        tx_grid = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))).astype(np.complex64)
        
        # TX
        x_optical = vlc_ofdm_tx(tx_grid, nfft=nfft, dc_bias=0.5, clipping_ratio=0.95)
        
        # RX (no channel)
        Y_freq = vlc_ofdm_rx(x_optical, nfft=nfft, n_subcarriers=K, dc_bias=0.5)
        
        assert Y_freq.shape == tx_grid.shape


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

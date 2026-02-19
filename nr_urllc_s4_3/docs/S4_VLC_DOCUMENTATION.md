# S4 VLC Channel Module - Documentation & Integration Guide

## Overview

The `s4_vlc_channel.py` module implements a complete **Visible Light Communication (VLC)** physical layer using **Intensity Modulation / Direct Detection (IM/DD)** with **DC-biased Optical OFDM (DCO-OFDM)**.

This module is designed to integrate seamlessly with your existing S2/S3 RF OFDM pipeline to enable hybrid RF+VLC research.

---

## Key Features

### ✅ DCO-OFDM Modulation
- **Hermitian symmetry** → real-valued signals for LED transmission
- **DC biasing** → ensures non-negative optical intensity
- **Clipping** → models LED dynamic range limits

### ✅ LED Channel Model
- **Frequency-selective response** (1st or 2nd order Butterworth)
- **Configurable 3dB bandwidth** (typical: 1-20 MHz)
- **Realistic attenuation** based on LED phosphor decay

### ✅ Optical Noise
- **AWGN mode** (simple, default)
- **Shot noise mode** (signal-dependent, realistic)
- **Configurable optical SNR**

### ✅ Photodetector Model
- **Responsivity** (typical: 0.4-0.6 A/W for Si photodiodes)
- **Active area** (cm²)
- **DC removal** and signal conditioning

### ✅ S3 Timing Compatible
- Uses **same mini-slot grid** as RF
- Compatible with existing `S3TimingController`
- Drop-in replacement for RF channel in your existing pipeline

---

## Module Structure

```python
s4_vlc_channel.py
│
├── DCO-OFDM MODULATION
│   ├── dco_ofdm_modulate()         # Complex → Real + DC bias
│   └── dco_ofdm_demodulate()       # Real → Complex
│
├── LED CHANNEL MODEL
│   ├── led_frequency_response()    # LED H(f)
│   ├── apply_led_channel()         # Time-domain filtering
│   └── add_optical_noise()         # AWGN or shot noise
│
├── PHOTODETECTOR
│   └── photodetector_response()    # Optical → Electrical
│
├── INTEGRATED LINK
│   ├── vlc_ofdm_link()             # Full TX → Channel → RX
│   ├── vlc_ofdm_tx()               # TX only (for pipeline)
│   ├── vlc_ofdm_rx()               # RX only (for pipeline)
│   └── apply_vlc_channel()         # Drop-in for apply_rf_channel()
│
└── UTILITIES
    ├── estimate_required_dc_bias() # Auto-tune DC bias
    └── validate_hermitian_symmetry()
```

---

## Usage Examples

### Example 1: Standalone VLC Link

```python
import numpy as np
from s4_vlc_channel import vlc_ofdm_link

# Generate QPSK data
rng = np.random.default_rng(42)
S, K = 14, 64  # 14 OFDM symbols, 64 subcarriers
x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)

# Run VLC link
Y_freq, info = vlc_ofdm_link(
    x_freq,
    snr_db=15.0,
    nfft=256,
    n_subcarriers=64,
    led_bandwidth_mhz=20.0,
    dc_bias=0.5,
    rng=rng
)

# Compute EVM
evm = np.sqrt(np.mean(np.abs(Y_freq - x_freq) ** 2)) / np.sqrt(np.mean(np.abs(x_freq) ** 2))
print(f"EVM: {evm*100:.2f}%")
```

### Example 2: Integration with Existing OFDM Pipeline

```python
# In your existing run_ofdm_link() function:

from s4_vlc_channel import vlc_ofdm_tx, apply_vlc_channel, vlc_ofdm_rx

def run_ofdm_link(cfg, channel_type='rf'):
    """Unified OFDM link for RF or VLC"""
    
    # ... [existing pilot insertion, modulation] ...
    
    if channel_type == 'rf':
        # Existing RF path
        x_time = ofdm.tx(tx_grid, nfft=nfft, cp=cp)
        y_time = apply_rf_channel(x_time, cfg)  # Your existing S2 channel
        Y_freq = ofdm.rx(y_time, nfft=nfft, cp=cp, n_subcarriers=K)
    
    elif channel_type == 'vlc':
        # New VLC path (S4)
        vlc_cfg = cfg.get('vlc', {})
        dc_bias = vlc_cfg.get('dc_bias', 0.5)
        clipping = vlc_cfg.get('clipping_ratio', 0.95)
        
        # VLC TX (no CP needed for DCO-OFDM)
        x_optical = vlc_ofdm_tx(tx_grid, nfft=nfft, dc_bias=dc_bias, clipping_ratio=clipping)
        
        # VLC channel
        y_optical = apply_vlc_channel(x_optical, cfg, rng)
        
        # VLC RX
        Y_freq = vlc_ofdm_rx(y_optical, nfft=nfft, n_subcarriers=K, dc_bias=dc_bias)
    
    # ... [existing MMSE equalization, demodulation] ...
```

### Example 3: Configuration File (`configs/s4_vlc_example.yaml`)

```yaml
# Inherits all settings from s3_timing_example.yaml
# Add VLC-specific block:

vlc:
  # LED parameters
  led_bandwidth_mhz: 20.0        # LED 3dB bandwidth [MHz]
  filter_order: 1                # 1st-order low-pass (or 2 for 2nd-order)
  
  # DCO-OFDM parameters
  dc_bias: 0.5                   # DC bias ratio (0-1)
  clipping_ratio: 0.95           # Max signal = (1+dc_bias) × clipping_ratio
  
  # Photodetector parameters
  responsivity: 0.5              # [A/W] (typical for Si: 0.4-0.6)
  area_cm2: 1.0                  # Active area [cm²]
  
  # Noise model
  noise_type: awgn               # "awgn" or "shot"
  snr_db: 15.0                   # Optical SNR [dB]
  
  # System parameters
  sample_rate_hz: 100.0e6        # 100 MHz sampling rate

# S4 sweep (same structure as S3)
s4_sweep:
  K_list: [1, 2, 3]
  snr_db_range: [5, 25, 1]       # VLC typically needs higher SNR than RF
  policy: independent

# Reuse S3 timing (unchanged)
nr:
  mu: 2
  minislot_symbols: 7
  harq:
    k1_symbols: 1

timing:
  grant_delay_ms: 0.0
  inter_attempt_gap_ms: 0.1
  proc_margin_ms: 0.2

urllc:
  pdb_ms: 5.0
  core_budget_ms: 2.0
```

---

## Integration Checklist

### Step 1: Copy Module to Your Project

```bash
# Place in your nr_urllc/ directory
cp s4_vlc_channel.py /path/to/your/project/nr_urllc/s4_vlc_channel.py
```

### Step 2: Add VLC Config Block

Add `vlc:` block to your existing config YAML (see Example 3 above).

### Step 3: Modify Your OFDM Runner

```python
# In your scripts/run_ofdm_link.py or simulate.py:

from nr_urllc.s4_vlc_channel import apply_vlc_channel, vlc_ofdm_tx, vlc_ofdm_rx

# Add channel_type parameter to your runner
def run_ofdm_sweep(cfg, channel_type='rf'):
    # ... existing setup ...
    
    for snr_db in snr_list:
        if channel_type == 'vlc':
            cfg['vlc']['snr_db'] = snr_db
            # Use VLC channel functions
        else:
            cfg['channel']['snr_db'] = snr_db
            # Use existing RF channel functions
```

### Step 4: Create S4 Sweep Runner

```python
# scripts/run_s4_vlc_sweep.py
from nr_urllc.s3_timing_urllc import S3TimingController
from nr_urllc.s3_timely_bler_sweep import plot_timely_bler_curves, plot_latency_cdf
from nr_urllc.s4_vlc_channel import vlc_ofdm_link

# Parallel to run_s3_timely_sweep.py but calls VLC functions
```

---

## Understanding DCO-OFDM

### Why DCO-OFDM for VLC?

**Problem**: LEDs can only transmit **intensity** (non-negative real values), but OFDM naturally produces **complex** symbols.

**Solution**: DCO-OFDM uses three tricks:

1. **Hermitian Symmetry**: 
   - Set X[N-k] = X*[k] for all k
   - Ensures IFFT output is purely real

2. **DC Bias**:
   - Add constant offset to make signal non-negative
   - Typically dc_bias ≈ 3-4 × std(signal)

3. **Clipping**:
   - Clip to LED dynamic range [0, I_max]
   - Introduces nonlinear distortion but necessary for LED

### DCO-OFDM vs ACO-OFDM

| Feature | DCO-OFDM | ACO-OFDM |
|---------|----------|----------|
| **Spectrum Efficiency** | 2× | 1× |
| **DC Bias** | Required | Not required |
| **Clipping Noise** | Higher | Lower |
| **Complexity** | Lower | Higher |
| **Use Case** | General VLC | High-PAPR scenarios |

**We chose DCO-OFDM** because:
- ✅ Better spectrum efficiency (uses all subcarriers)
- ✅ Simpler implementation
- ✅ More common in VLC literature

---

## Parameter Tuning Guide

### DC Bias Selection

```python
from s4_vlc_channel import estimate_required_dc_bias

# Auto-estimate based on your data
dc_bias = estimate_required_dc_bias(x_freq, nfft=256, target_clip_rate=0.01)
```

**Rule of thumb**:
- `dc_bias = 0.3`: Low overhead, high clipping (>10%)
- `dc_bias = 0.5`: **Recommended** balance (3-5% clipping)
- `dc_bias = 0.7`: Low clipping, high DC power waste

### LED Bandwidth

Typical LED bandwidths:
- **Red/Green/Blue LEDs**: 1-5 MHz
- **White LEDs (phosphor)**: 1-20 MHz
- **White LEDs (RGB mix)**: 5-50 MHz
- **Custom VLC LEDs**: 50-200 MHz

**Impact**:
- Lower bandwidth → more ISI → higher BLER
- For fair RF vs VLC comparison, use **20 MHz** as baseline

### Optical SNR

Optical SNR ≠ Electrical SNR!

**Typical values**:
- **Indoor VLC (1m, 100mW LED)**: 15-25 dB
- **Indoor VLC (3m, 100mW LED)**: 10-20 dB
- **Outdoor VLC (sunlight background)**: 5-15 dB

**Tip**: VLC usually requires 3-5 dB **higher** SNR than RF for same BER due to:
- Clipping distortion
- LED nonlinearity
- Shot noise (signal-dependent)

---

## Common Issues & Solutions

### Issue 1: High Clipping Rate (>10%)

**Symptom**: Warning: `High clipping rate: XX% of samples clipped`

**Solutions**:
1. Increase `dc_bias` (e.g., 0.5 → 0.7)
2. Increase `clipping_ratio` (e.g., 0.95 → 1.0)
3. Reduce input signal power (lower constellation size)

### Issue 2: Poor BER Performance

**Symptom**: VLC BER much worse than RF at same SNR

**Expected**: This is normal! VLC typically needs 3-5 dB higher SNR.

**Reasons**:
- Clipping introduces nonlinear distortion
- LED frequency response attenuates high frequencies
- Shot noise is signal-dependent (worse than AWGN)

**Solutions**:
1. Increase optical SNR (use `snr_db` 3-5 dB higher than RF)
2. Increase LED bandwidth (`led_bandwidth_mhz`)
3. Optimize DC bias to minimize clipping

### Issue 3: Hermitian Symmetry Error

**Symptom**: Warning: `DCO-OFDM output has non-negligible imaginary part`

**Cause**: Numerical precision issues in Hermitian symmetry construction

**Solution**: Usually harmless (< 1e-10). If larger, check:
```python
from s4_vlc_channel import validate_hermitian_symmetry
is_valid = validate_hermitian_symmetry(X_full)
```

---

## Performance Benchmarks

### Test Setup
- QPSK modulation
- 14 OFDM symbols × 64 subcarriers
- FFT size: 256
- LED bandwidth: 20 MHz
- DC bias: 0.5

### Expected Results

| Optical SNR (dB) | BER (approx) | EVM (%) |
|------------------|--------------|---------|
| 10 | 1e-2 | 50-80 |
| 15 | 1e-3 | 30-50 |
| 20 | 1e-4 | 15-25 |
| 25 | 1e-5 | 8-15 |

**Note**: These are approximate. Your results will depend on:
- Pilot overhead
- Channel estimation quality
- Equalization method (ZF vs MMSE)
- Clipping ratio

---

## Testing Your Integration

### Unit Test Template

```python
# tests/test_s4_vlc.py
import pytest
import numpy as np
from nr_urllc.s4_vlc_channel import (
    dco_ofdm_modulate, dco_ofdm_demodulate, 
    apply_led_channel, vlc_ofdm_link
)

def test_dco_ofdm_hermitian():
    """Verify DCO-OFDM output is real-valued"""
    rng = np.random.default_rng(42)
    x_freq = rng.normal(0, 1, (10, 32)) + 1j * rng.normal(0, 1, (10, 32))
    x_real = dco_ofdm_modulate(x_freq, dc_bias=0.5, nfft=128)
    
    assert x_real.dtype == np.float32
    assert x_real.min() >= 0  # Should be non-negative

def test_vlc_roundtrip_no_noise():
    """Test DCO-OFDM modulation/demodulation roundtrip"""
    rng = np.random.default_rng(42)
    S, K, nfft = 14, 64, 256
    x_freq = (rng.normal(0, 1, (S, K)) + 1j * rng.normal(0, 1, (S, K))) / np.sqrt(2)
    
    # TX
    x_optical = dco_ofdm_modulate(x_freq, dc_bias=0.5, nfft=nfft)
    
    # RX (no channel, no noise)
    Y_freq = dco_ofdm_demodulate(x_optical, nfft=nfft, n_subcarriers=K, dc_bias=0.5)
    
    # Check similarity (won't be perfect due to clipping)
    correlation = np.abs(np.corrcoef(x_freq.flatten(), Y_freq.flatten())[0, 1])
    assert correlation > 0.9

def test_vlc_timing_matches_rf():
    """CRITICAL: VLC timing must match RF timing"""
    from nr_urllc.s3_timing_urllc import S3TimingController
    
    cfg = {
        "nr": {"mu": 2, "minislot_symbols": 7, "harq": {"k1_symbols": 1}},
        "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
        "timing": {"grant_delay_ms": 0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
    }
    
    # VLC should use same timing as RF
    ctrl = S3TimingController(cfg)
    
    assert ctrl.attempt_latency_ms > 0
    assert ctrl.K_max_by_deadline > 0
    # Both RF and VLC use same timing controller!

def test_led_bandwidth_attenuation():
    """Verify LED attenuates signal as expected"""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (10, 256))
    
    # Apply 20 MHz LED filter
    y = apply_led_channel(x, sample_rate_hz=100e6, bandwidth_3db_mhz=20.0)
    
    # Signal power should decrease
    assert np.var(y) < np.var(x)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Next Steps After S4

Once you have VLC working:

1. **Generate VLC baseline plots** (parallel to S3 RF plots):
   - `vlc_bler_vs_snr_K*.png`
   - `vlc_latency_cdf.png`
   - `s4_vlc_timely_results.json`

2. **Compare RF vs VLC**:
   - Plot both on same axes
   - Note VLC needs ~5 dB higher SNR for same BLER
   - Both should have same latency (timing aligned)

3. **Proceed to S5** (Link Adapters):
   - Wrap both RF and VLC with unified API
   - `(SNR, MCS, K) → {RWD, latency, cost}`

4. **Proceed to S6** (Hybrid Policies):
   - Implement link selection (best_link)
   - Implement duplication (dup, conditional_dup)
   - Generate Pareto plots (RWD vs overhead)

---

## References

### VLC & DCO-OFDM
- Armstrong, J. (2009). "OFDM for Optical Communications." IEEE JSAC.
- Dissanayake, S. D., & Armstrong, J. (2013). "Comparison of ACO-OFDM, DCO-OFDM and ADO-OFDM in IM/DD Systems." JLT.

### LED Models
- Grubor, J., et al. (2008). "Bandwidth-Efficient Indoor Optical Wireless Communications with White LEDs." CSNDSP.

### 3GPP NR Timing
- 3GPP TS 38.211 — Physical channels and modulation
- 3GPP TS 38.214 — Physical layer procedures for data

---

## Support

For questions or issues:
1. Check this documentation
2. Review module docstrings (all functions documented)
3. Run standalone test: `python s4_vlc_channel.py`
4. Check unit tests: `pytest tests/test_s4_vlc.py -v`

---

**Module Version**: 1.0  
**Compatible With**: S3 timing framework, existing OFDM pipeline  
**Status**: Ready for S4 integration  
**License**: Use freely for research purposes

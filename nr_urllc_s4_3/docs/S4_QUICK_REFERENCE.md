# S4 VLC Integration - Quick Reference Card

## 🚀 Quick Start (5 Minutes)

### 1. Copy Module
```bash
cp s4_vlc_channel.py your_project/nr_urllc/s4_vlc_channel.py
```

### 2. Add Config Block to `configs/s4_vlc_example.yaml`
```yaml
vlc:
  led_bandwidth_mhz: 20.0
  dc_bias: 0.5
  clipping_ratio: 0.95
  responsivity: 0.5
  area_cm2: 1.0
  noise_type: awgn
  snr_db: 15.0
  sample_rate_hz: 100.0e6
```

### 3. Modify Your OFDM Runner
```python
from nr_urllc.s4_vlc_channel import vlc_ofdm_tx, apply_vlc_channel, vlc_ofdm_rx

# In your run_ofdm_link():
if channel_type == 'vlc':
    x_optical = vlc_ofdm_tx(tx_grid, nfft, dc_bias=0.5, clipping_ratio=0.95)
    y_optical = apply_vlc_channel(x_optical, cfg, rng)
    Y_freq = vlc_ofdm_rx(y_optical, nfft, n_subcarriers, dc_bias=0.5)
```

### 4. Run VLC Sweep
```bash
python scripts/run_s4_vlc_sweep.py --cfg configs/s4_vlc_example.yaml
```

---

## 📦 Main Functions

### Standalone Link (Easiest)
```python
from s4_vlc_channel import vlc_ofdm_link

Y_freq, info = vlc_ofdm_link(
    x_freq,              # [S, K] complex symbols
    snr_db=15.0,         # Optical SNR [dB]
    nfft=256,            # FFT size
    n_subcarriers=64,    # K
    led_bandwidth_mhz=20.0,
    dc_bias=0.5,
    rng=rng
)
```

### Pipeline Integration (Drop-in Replacement)
```python
from s4_vlc_channel import apply_vlc_channel

# Replace this:
# y_time = apply_rf_channel(x_time, cfg)

# With this:
y_time = apply_vlc_channel(x_time, cfg, rng)
```

---

## 🎯 Key Parameters

| Parameter | Typical Value | Range | Impact |
|-----------|---------------|-------|--------|
| `dc_bias` | **0.5** | 0.3-0.7 | Higher = less clipping, more DC waste |
| `led_bandwidth_mhz` | **20** | 5-50 | Higher = better BER, less realistic |
| `snr_db` | **15-20** | 10-25 | VLC needs ~5dB more than RF |
| `clipping_ratio` | **0.95** | 0.9-1.0 | Lower = more distortion |
| `responsivity` | **0.5** | 0.4-0.6 | Photodetector sensitivity [A/W] |

---

## ⚠️ Common Issues

### Issue: High Clipping Rate (>10%)
**Solution**: Increase `dc_bias` from 0.5 → 0.7

### Issue: Poor BER vs RF
**Expected!** VLC needs 3-5 dB higher SNR due to:
- Clipping distortion
- LED bandwidth limits
- Signal-dependent shot noise

**Solution**: Compare at VLC SNR = RF SNR + 5 dB

### Issue: Warning About Imaginary Part
**Solution**: Usually harmless (< 1e-10). Numerical precision, not a bug.

---

## 🧪 Sanity Checks

### Test 1: Module Works
```bash
python s4_vlc_channel.py
# Should print "All tests completed successfully!"
```

### Test 2: Hermitian Symmetry
```python
from s4_vlc_channel import dco_ofdm_modulate
x_real = dco_ofdm_modulate(x_freq, dc_bias=0.5, nfft=256)
assert x_real.min() >= 0  # Should be non-negative
```

### Test 3: Timing Alignment
```python
from nr_urllc.s3_timing_urllc import S3TimingController
ctrl = S3TimingController(cfg)  # Same cfg for RF and VLC!
print(f"Attempt latency: {ctrl.attempt_latency_ms} ms")  # Should match RF
```

---

## 📊 Expected Performance

| SNR (dB) | QPSK BER | Notes |
|----------|----------|-------|
| 10 | ~1e-2 | Poor, high errors |
| 15 | ~1e-3 | Moderate, usable |
| 20 | ~1e-4 | Good, typical indoor |
| 25 | ~1e-5 | Excellent, URLLC-grade |

**Note**: VLC at 15 dB ≈ RF at 10 dB (due to clipping)

---

## 🔄 S4 Definition of Done

- [ ] `s4_vlc_channel.py` in `nr_urllc/` directory
- [ ] VLC config block in YAML
- [ ] `run_s4_vlc_sweep.py` script works
- [ ] Unit tests pass (`test_s4_vlc.py`)
- [ ] Generated outputs:
  - [ ] `vlc_bler_vs_snr_K*.png`
  - [ ] `vlc_latency_cdf.png`
  - [ ] `s4_vlc_timely_results.json`
- [ ] VLC timing matches RF timing (verified)
- [ ] Documentation updated

---

## 🎓 Key Concepts

### DCO-OFDM in 30 Seconds
1. **Problem**: LEDs need real, non-negative signals
2. **Solution**: 
   - Hermitian symmetry → real IFFT output
   - DC bias → make positive
   - Clipping → fit LED range

### Why VLC Needs Higher SNR Than RF
1. **Clipping distortion** (nonlinear)
2. **LED bandwidth** limited (ISI)
3. **Shot noise** signal-dependent (not white)

### S4 → S5 → S6 Pipeline
- **S4**: VLC physical layer ← **YOU ARE HERE**
- **S5**: Link adapters (unified RF+VLC API)
- **S6**: Hybrid policies (best_link, duplication)

---

## 📞 Help Checklist

Before asking for help:
1. ✅ Ran standalone test: `python s4_vlc_channel.py`
2. ✅ Checked docstrings in functions
3. ✅ Reviewed `S4_VLC_DOCUMENTATION.md`
4. ✅ Tried adjusting `dc_bias` and `snr_db`
5. ✅ Compared VLC vs RF at SNR+5dB offset

---

## 🔗 Quick Links

- **Full Documentation**: `S4_VLC_DOCUMENTATION.md`
- **Module Source**: `s4_vlc_channel.py`
- **S3 Docs (Timing)**: `docs/README_S3.md`
- **Hybrid Plan**: `Hybrid_Plan.docx`

---

**Version**: 1.0  
**Status**: Ready for S4 Integration  
**Last Updated**: October 2025

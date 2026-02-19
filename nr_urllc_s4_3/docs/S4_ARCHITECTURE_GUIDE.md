# S4 VLC Architecture - Visual Integration Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID RF + VLC SYSTEM (S4)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    USER CONFIGURATION (YAML)                    │
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                  │
│  │   RF Settings   │     │   VLC Settings  │ ← NEW (S4)       │
│  │  (from S2/S3)   │     │                 │                  │
│  └─────────────────┘     └─────────────────┘                  │
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │     NR Timing (from S3) - SHARED         │                  │
│  │  mu, L, k1, deadline, gaps, margins      │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    S3 TIMING CONTROLLER                         │
│                     (SHARED BY RF & VLC)                        │
│                                                                 │
│  • attempt_latency_ms      ← Same for both links               │
│  • K_max_by_deadline       ← Same for both links               │
│  • cumulative_latency_K    ← Same for both links               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  Data Bits    │
                    └───────────────┘
                            ↓
                ┌───────────────────────┐
                │   QAM Modulation      │
                │  (M=4/16/64, shared)  │
                └───────────────────────┘
                            ↓
            ┌───────────────────────────────┐
            │   Pilot Insertion (shared)    │
            │   (spacing, offset, boost)    │
            └───────────────────────────────┘
                            ↓
                  ┌──────────┴──────────┐
                  │                     │
         ┌────────▼─────────┐  ┌───────▼──────────┐
         │   RF CHANNEL     │  │   VLC CHANNEL    │
         │   (S2/S3 path)   │  │   (S4 NEW path)  │
         └────────┬─────────┘  └───────┬──────────┘
                  │                     │
                  │                     │
    ┌─────────────▼──────────┐  ┌──────▼────────────────┐
    │   RF OFDM TX/RX        │  │  VLC DCO-OFDM TX/RX   │
    │                        │  │                       │
    │  • ofdm.tx()           │  │  • vlc_ofdm_tx()      │
    │  • ofdm.rx()           │  │  • vlc_ofdm_rx()      │
    │  • Complex symbols     │  │  • Real optical sig   │
    └─────────────┬──────────┘  └──────┬────────────────┘
                  │                     │
    ┌─────────────▼──────────┐  ┌──────▼────────────────┐
    │   RF CHANNEL MODEL     │  │  VLC CHANNEL MODEL    │
    │                        │  │                       │
    │  • CDL/TDL fading      │  │  • LED low-pass       │
    │  • AWGN                │  │  • DC bias/clipping   │
    │  • apply_rf_channel()  │  │  • Shot noise         │
    │                        │  │  • apply_vlc_channel()│
    └─────────────┬──────────┘  └──────┬────────────────┘
                  │                     │
    ┌─────────────▼──────────┐  ┌──────▼────────────────┐
    │  CHANNEL ESTIMATION    │  │  CHANNEL ESTIMATION   │
    │                        │  │                       │
    │  • LS on pilots        │  │  • LS on pilots       │
    │  • Freq interpolation  │  │  • Freq interpolation │
    │  • (SHARED CODE)       │  │  • (SHARED CODE)      │
    └─────────────┬──────────┘  └──────┬────────────────┘
                  │                     │
    ┌─────────────▼──────────┐  ┌──────▼────────────────┐
    │  MMSE EQUALIZATION     │  │  MMSE EQUALIZATION    │
    │                        │  │                       │
    │  • equalize_mmse()     │  │  • equalize_mmse()    │
    │  • (SHARED CODE)       │  │  • (SHARED CODE)      │
    └─────────────┬──────────┘  └──────┬────────────────┘
                  │                     │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   QAM Demodulation  │
                  │   (shared demod)    │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Recovered Bits    │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   BER/BLER/EVM      │
                  │   (metrics)         │
                  └─────────────────────┘
```

---

## Code Flow: RF vs VLC Side-by-Side

```python
# ============================================================
# RF PATH (S2/S3 - Existing)
# ============================================================

# 1. TX
tx_grid = place_pilots(data_symbols, pilot_cfg)  # Shared
x_time = ofdm.tx(tx_grid, nfft, cp)              # RF: Complex I/Q

# 2. Channel
y_time = apply_rf_channel(x_time, cfg, rng)     # CDL/TDL + AWGN

# 3. RX
Y_freq = ofdm.rx(y_time, nfft, cp, K)           # RF: Complex FFT

# 4. Estimation & EQ (Shared)
H_est = estimate_channel(Y_freq, pilot_mask)    # Shared
Y_eq = equalize_mmse(Y_freq, H_est, sigma2)     # Shared

# 5. Demod
bits_hat = demodulate(Y_eq, M)                   # Shared


# ============================================================
# VLC PATH (S4 - New)
# ============================================================

# 1. TX
tx_grid = place_pilots(data_symbols, pilot_cfg)  # Shared (same as RF!)
x_optical = vlc_ofdm_tx(                         # VLC: Real + DC bias
    tx_grid, nfft, 
    dc_bias=0.5, 
    clipping_ratio=0.95
)

# 2. Channel
y_optical = apply_vlc_channel(                   # LED + shot noise
    x_optical, cfg, rng
)

# 3. RX
Y_freq = vlc_ofdm_rx(                            # VLC: Remove DC + FFT
    y_optical, nfft, K, 
    dc_bias=0.5
)

# 4. Estimation & EQ (Shared)
H_est = estimate_channel(Y_freq, pilot_mask)    # Shared (same as RF!)
Y_eq = equalize_mmse(Y_freq, H_est, sigma2)     # Shared (same as RF!)

# 5. Demod
bits_hat = demodulate(Y_eq, M)                   # Shared (same as RF!)
```

**Key Insight**: Only steps 1-3 differ between RF and VLC. Steps 4-5 are identical!

---

## File Organization After S4

```
your_project/
│
├── nr_urllc/                          # Core modules
│   ├── s3_timing_urllc.py             # S3: Timing (shared by RF & VLC) ✓
│   ├── s3_timely_bler_sweep.py        # S3: Sweep & plotting ✓
│   ├── s4_vlc_channel.py              # S4: VLC PHY ← NEW ✓
│   ├── ofdm.py                        # S2: OFDM TX/RX (RF) ✓
│   ├── channel.py                     # S2: RF channel models ✓
│   ├── pilots.py                      # S2: Pilots (shared) ✓
│   ├── eq.py                          # S2: Equalization (shared) ✓
│   └── utils.py                       # S2: Mod/demod (shared) ✓
│
├── scripts/                           # Runners
│   ├── run_s3_timely_sweep.py         # S3: RF sweep ✓
│   ├── run_s4_vlc_sweep.py            # S4: VLC sweep ← TO CREATE
│   └── run_bler.py                    # S2: BLER measurement ✓
│
├── configs/                           # Configuration
│   ├── s3_timing_example.yaml         # S3: RF config ✓
│   ├── s4_vlc_example.yaml            # S4: VLC config ← TO CREATE
│   └── m2_ofdm_tdlc_autoramp.yaml     # S2: example ✓
│
├── tests/                             # Unit tests
│   ├── test_s3_timing_urllc.py        # S3: tests ✓
│   ├── test_s4_vlc.py                 # S4: tests ← TO CREATE
│   └── test_phy.py                    # S2: tests ✓
│
├── docs/                              # Documentation
│   ├── README_S3.md                   # S3: docs ✓
│   ├── S3_COMPLETION_SUMMARY.md       # S3: summary ✓
│   ├── S4_VLC_DOCUMENTATION.md        # S4: docs ← PROVIDED ✓
│   ├── S4_QUICK_REFERENCE.md          # S4: quick ref ← PROVIDED ✓
│   └── S4_DELIVERY_SUMMARY.md         # S4: summary ← PROVIDED ✓
│
└── artifacts/                         # Results
    ├── s3/                            # S3: RF results ✓
    │   ├── timely_bler_vs_snr_K*.png
    │   ├── latency_cdf.png
    │   └── s3_timely_bler_results.json
    └── s4/                            # S4: VLC results ← TO GENERATE
        ├── vlc_timely_bler_vs_snr_K*.png
        ├── vlc_latency_cdf.png
        └── s4_vlc_timely_results.json
```

---

## Integration Checklist (Detailed)

### Phase 1: Setup (30 min)
```bash
# 1. Copy VLC module
cp s4_vlc_channel.py your_project/nr_urllc/

# 2. Verify standalone
cd your_project
python nr_urllc/s4_vlc_channel.py
# Expected: "All tests completed successfully!"

# 3. Read docs
cat docs/S4_QUICK_REFERENCE.md  # 10 min
```

### Phase 2: Configuration (1 hour)
```bash
# 1. Copy S3 config as base
cp configs/s3_timing_example.yaml configs/s4_vlc_example.yaml

# 2. Add VLC block (see template in docs)
vim configs/s4_vlc_example.yaml
```

```yaml
# Add to configs/s4_vlc_example.yaml:
vlc:
  led_bandwidth_mhz: 20.0
  dc_bias: 0.5
  clipping_ratio: 0.95
  responsivity: 0.5
  area_cm2: 1.0
  noise_type: awgn
  snr_db: 15.0
  sample_rate_hz: 100.0e6

s4_sweep:
  K_list: [1, 2, 3]
  snr_db_range: [5, 25, 1]
  policy: independent
```

### Phase 3: Code Integration (3 hours)

**Option A: Minimal (recommended first)**
```python
# Create scripts/test_vlc_standalone.py
from nr_urllc.s4_vlc_channel import vlc_ofdm_link
import numpy as np

rng = np.random.default_rng(42)
x_freq = (rng.normal(0,1,(14,64)) + 1j*rng.normal(0,1,(14,64)))/np.sqrt(2)
Y_freq, info = vlc_ofdm_link(x_freq, snr_db=15, nfft=256, n_subcarriers=64)
print(f"EVM: {np.linalg.norm(Y_freq-x_freq)/np.linalg.norm(x_freq)*100:.1f}%")
```

**Option B: Full Integration**
```python
# Modify your existing scripts/run_bler.py or similar:

from nr_urllc.s4_vlc_channel import vlc_ofdm_tx, apply_vlc_channel, vlc_ofdm_rx

def run_ofdm_link(cfg, channel_type='rf'):
    # ... existing setup ...
    
    if channel_type == 'vlc':
        vlc_cfg = cfg['vlc']
        x_optical = vlc_ofdm_tx(tx_grid, nfft, 
                                dc_bias=vlc_cfg['dc_bias'],
                                clipping_ratio=vlc_cfg['clipping_ratio'])
        y_optical = apply_vlc_channel(x_optical, cfg, rng)
        Y_freq = vlc_ofdm_rx(y_optical, nfft, n_subcarriers, 
                             dc_bias=vlc_cfg['dc_bias'])
    else:  # rf
        # ... existing RF path ...
    
    # ... rest is shared (estimation, equalization, demod) ...
```

### Phase 4: Testing (2 hours)
```bash
# 1. Unit tests
cat > tests/test_s4_vlc.py << 'EOF'
# (see test template in S4_VLC_DOCUMENTATION.md)
EOF

pytest tests/test_s4_vlc.py -v

# 2. Integration test
python scripts/test_vlc_standalone.py
# Expected: EVM ~30-50% at SNR=15 dB

# 3. Config test
python scripts/run_s4_vlc_sweep.py --cfg configs/s4_vlc_example.yaml
# Expected: Generates plots in artifacts/s4/
```

### Phase 5: Sweep & Analysis (2 hours)
```bash
# 1. Full VLC sweep
python scripts/run_s4_vlc_sweep.py \
    --cfg configs/s4_vlc_example.yaml \
    --output-dir artifacts/s4/

# 2. Verify outputs
ls -lh artifacts/s4/
# vlc_timely_bler_vs_snr_K1.png
# vlc_timely_bler_vs_snr_K2.png
# vlc_timely_bler_vs_snr_K3.png
# vlc_latency_cdf.png
# s4_vlc_timely_results.json

# 3. Compare RF vs VLC
python scripts/compare_rf_vlc.py  # You'll create this
```

---

## Common Patterns

### Pattern 1: Unified Runner with Channel Selection
```python
def run_sweep(cfg, channel_type='rf'):
    """Unified sweep for both RF and VLC"""
    if channel_type == 'vlc':
        from nr_urllc.s4_vlc_channel import vlc_ofdm_tx, apply_vlc_channel, vlc_ofdm_rx
        tx_fn, channel_fn, rx_fn = vlc_ofdm_tx, apply_vlc_channel, vlc_ofdm_rx
    else:
        from nr_urllc import ofdm, channel as ch
        tx_fn = lambda grid, nfft, **kw: ofdm.tx(grid, nfft, cp)
        channel_fn = ch.apply_fir_per_symbol
        rx_fn = lambda y, nfft, K, **kw: ofdm.rx(y, nfft, cp, K)
    
    # Common pipeline
    for snr in snr_list:
        x = tx_fn(tx_grid, nfft)
        y = channel_fn(x, cfg, rng)
        Y = rx_fn(y, nfft, n_subcarriers)
        # ... equalization & demod (shared) ...
```

### Pattern 2: Config-Driven Channel Selection
```yaml
# In YAML config:
simulation:
  channel_type: vlc  # or 'rf'
```

```python
# In runner:
channel_type = cfg['simulation']['channel_type']
result = run_sweep(cfg, channel_type=channel_type)
```

---

## Validation Checklist

### ✅ Module Level
- [ ] Standalone test passes
- [ ] All functions have docstrings
- [ ] Hermitian symmetry verified
- [ ] No import errors

### ✅ Integration Level
- [ ] VLC path works without errors
- [ ] Config loads correctly
- [ ] Outputs shape match RF outputs
- [ ] Can switch RF ↔ VLC via config

### ✅ System Level
- [ ] Timing matches RF (same latency)
- [ ] Pilot pattern reused correctly
- [ ] MMSE equalization works
- [ ] BER/BLER curves generated

### ✅ Scientific Level
- [ ] VLC needs ~5 dB more SNR than RF
- [ ] LED bandwidth limits performance
- [ ] Clipping rate reasonable (<10%)
- [ ] Results match expectations

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Module tests pass | 100% | `python s4_vlc_channel.py` |
| Integration tests pass | 100% | `pytest tests/test_s4_vlc.py` |
| VLC vs RF SNR gap | 3-7 dB | Compare BLER curves |
| Latency alignment | Exact match | Compare CDFs |
| Clipping rate | <10% | Check warnings |
| Time to integrate | <12 hours | Your experience |

---

## Troubleshooting Guide

### Problem: Import Error
```
ModuleNotFoundError: No module named 's4_vlc_channel'
```
**Solution**: 
```bash
# Ensure you're running from project root
cd /path/to/your/project
python -c "import nr_urllc.s4_vlc_channel; print('OK')"
```

### Problem: High EVM (>100%)
**Causes**:
1. SNR too low (try SNR=25 first)
2. DC bias too low (increase to 0.7)
3. LED bandwidth too narrow (increase to 50 MHz for test)

### Problem: VLC BER = RF BER
**This is wrong!** VLC should be worse due to clipping.

**Check**:
1. Are you actually calling VLC functions? (add print statements)
2. Is `channel_type='vlc'` being passed correctly?
3. Is clipping actually happening? (check warnings)

---

**Ready to integrate? Start with Phase 1 (Setup) and work your way through!**

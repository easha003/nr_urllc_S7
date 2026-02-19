# S4 Merge Notes (nr_urllc_S4)

This package is your original `nr_urllc_S4` repository plus the **added** files you uploaded, merged without removing or overwriting any of your existing algorithms.

## What I added

**Code**
- `nr_urllc/run_ofdm_unified.py` — drop‑in unified OFDM runner that supports both RF (`channel_type='rf'`) and VLC (`channel_type='vlc'`). It **reuses** your pilots/interp/MMSE/demod path.
  - Patch applied for this repo: `equalize` is imported as `eq` alias and TDL call updated to `channel.tdl_fir_from_profile(...)` to match your API.

**Configs**
- `configs/s4_rf_comparison.yaml` — RF config that mirrors `s4_vlc_example.yaml` numerology/timing to enable apples‑to‑apples RF↔VLC comparisons.

**Docs**
- `docs/S4_ARCHITECTURE_GUIDE.md`
- `docs/S4_QUICK_REFERENCE.md`
- `docs/S4_VLC_DOCUMENTATION.md`

> No existing files were deleted. Existing S4 code (`nr_urllc/s4_vlc_channel.py`, `scripts/run_s4_vlc_sweep.py`, etc.) remains intact.

## Quick smoke tests I ran

Inside this archive I executed a minimal smoke test of `run_ofdm_unified` for both RF and VLC with QPSK, `nfft=256, K=64, L=7`:

```
from nr_urllc.run_ofdm_unified import run_ofdm_m2_unified
res_rf  = run_ofdm_m2_unified(cfg, channel_type='rf')   # OK
res_vlc = run_ofdm_m2_unified(cfg, channel_type='vlc')  # OK (VLC prints clipping warnings by design)
```

Both paths returned BER/EVM arrays without errors.

## How to use

### 1) Baseline RF (existing)
```
python -m scripts.run_sims --cfg configs/m2_ofdm_tdlc.yaml
```

### 2) VLC sweep (S4)
```
python -m scripts.run_s4_vlc_sweep --cfg configs/s4_vlc_example.yaml     --out artifacts/s4/s4_vlc_timely_results.json --out-dir artifacts/s4/
```

Optional: compare against RF timely results (produce an RF JSON via `scripts/run_s3_timely_sweep.py` first), then:
```
python -m scripts.run_s4_vlc_sweep --cfg configs/s4_vlc_example.yaml     --out artifacts/s4/s4_vlc_timely_results.json     --compare-rf artifacts/s3/results.json
```

### 3) Unified runner (side‑by‑side RF/VLC, minimal M2 metrics)
```
from nr_urllc.run_ofdm_unified import run_ofdm_m2_rf, run_ofdm_m2_vlc
rf  = run_ofdm_m2_rf(cfg_dict)
vlc = run_ofdm_m2_vlc(cfg_dict)
```

**Tip:** VLC often needs ~3–5 dB more SNR than RF for the same BER due to clipping and LED bandwidth (see docs).

## Non‑breaking changes only

- No existing algorithms were removed or renamed.
- The new module is additive; your current `simulate.py`, `sweep.py`, `scripts/` continue to operate exactly as before.

— End —

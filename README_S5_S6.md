
# S5 + S6 Drop-in (Adapters + Early-Decision Hybrid Controller)

## S5 — Build per-link LUTs
Ensure your timely sweep JSONs exist (edit paths in `configs/s5_build_luts.yaml` if needed).
```
python scripts/build_luts.py --cfg configs/s5_build_luts.yaml
# or
python -m scripts.build_luts --cfg configs/s5_build_luts.yaml
```
Outputs:
- `artifacts/adapters/rf_lut.json`
- `artifacts/adapters/vlc_lut.json`

## S6 — Predict-before-path controller
Uses AR(1) SNR prediction and lower-confidence-bound scoring on LUTs to decide RF/VLC/DUP **before** probing.
```
python scripts/run_s6_predecide.py --cfg configs/s6_predecide.yaml
# or
python -m scripts.run_s6_predecide --cfg configs/s6_predecide.yaml
```
Output:
- `artifacts/s6/predecide_trace.json`

### Notes
- LUT build accepts flexible JSON layouts: `points`, `records`, or `tables` per K; converts BLER→p automatically.
- Interpolation is linear with monotonicity enforcement to avoid wiggles.
- Adjust `conf_k` in `s6_predecide.yaml` (e.g., 1.64 ~ 95% one-sided) to be more conservative.

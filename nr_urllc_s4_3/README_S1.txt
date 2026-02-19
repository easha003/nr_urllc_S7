
S1 (URLLC budget) — How to run

1) Compute S1 budget only (no PHY) from any config that includes urllc: and nr:/ofdm: blocks:
   python -m scripts.urllc_budget --cfg configs/m2_ofdm_tdlc.yaml --out artifacts/urllc_budget.json

2) Run your normal simulation; S1 is auto-attached to result['meta']['urllc'] and written to artifacts/urllc_budget.json
   when io.write_json: true in the YAML:
   python -m scripts.run_sims --cfg configs/m2_ofdm_tdlc.yaml --out artifacts/m2_result.json

Robustness:
- If a field like urllc.app_payload_bytes is set to an unresolved template string (e.g., "${url_lc.payload_bytes}"),
  the tool fails with a clear error naming that field, rather than a cryptic int()/float() ValueError.
- The per-try BLER target is derived from your service PER and the number of tries that fit the radio deadline:
    per_try = 1 - (1 - PER_target)^(1/N), with N <= floor(deadline_ms / TTI_ms).

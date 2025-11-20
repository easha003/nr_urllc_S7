# S3 — NR Timing Shim for URLLC Reliability & Latency

## Overview

**S3** integrates NR numerology (mini-slots, SCS, k1 feedback delays) with URLLC deadlines to generate **timely BLER curves**. It answers the key question:

> **"How many attempts K fit within the deadline, and what's the resulting P(success ∧ timely) as a function of SNR?"**

## S3 Outputs

S3 produces three canonical plots and one JSON file:

1. **`timely_bler_vs_snr_K*.png`** — Timely BLER (TB failure rate within deadline) vs SNR for each K ∈ {1,2,3,...}
2. **`latency_cdf.png`** — Cumulative latency distribution showing which K values fit the deadline
3. **`s3_timely_heatmap.png`** — 2D heatmap (SNR × K) colored by P(success ∧ timely)
4. **`s3_timely_bler_results.json`** — Detailed results: per-point BLER, latency metrics, deadline status

## Architecture

### Core Concepts

- **Attempt Latency**: Time for one transmission attempt = grant_delay + L·Tsym + k1·Tsym + proc_margin
- **Cumulative Latency**: K attempts = K·attempt_latency + (K−1)·inter_attempt_gap
- **Radio Deadline**: pdb_ms − core_budget_ms (from S1 URLLC budget)
- **Timely BLER**: P(packet fails) given that all K attempts must finish before deadline

### Key Functions

#### 1. `attempt_latency_ms(mu, L_symbols, k1_symbols, grant_delay_ms, proc_margin_ms)`

Compute latency for one attempt:
```python
attempt_latency_ms(mu=2, L_symbols=7, k1_symbols=1, grant_delay_ms=0, proc_margin_ms=0.2)
# Returns: 0.333 ms (for mu=2: 60kHz SCS, so (7+1) symbols × 1/60ms + 0.2ms margin)
```

#### 2. `cumulative_latency_K(K, attempt_latency_ms, inter_attempt_gap_ms)`

Compute total latency for K sequential attempts:
```python
cumulative_latency_K(K=3, attempt_latency_ms=0.333, inter_attempt_gap_ms=0.1)
# Returns: 1.299 ms = 3×0.333 + 2×0.1
```

#### 3. `max_K_by_deadline(deadline_ms, attempt_latency_ms, ...)`

Find maximum K that fits:
```python
max_K = max_K_by_deadline(deadline_ms=8.0, attempt_latency_ms=0.333, inter_attempt_gap_ms=0.1)
# Returns: K_max (e.g., 7 or 8)
```

#### 4. `bler_to_timely_bler(bler_single, K, policy="independent")`

Convert single-attempt BLER to K-attempt BLER:
```python
bler_single = np.array([0.1, 0.05, 0.01])
bler_K = bler_to_timely_bler(bler_single, K=2)
# Returns: bler^K = [0.01, 0.0025, 0.0001]
# (At least one of K independent attempts succeeds)
```

#### 5. `S3TimingController(cfg)`

High-level orchestrator:
```python
from nr_urllc.s3_timing_urllc import S3TimingController

cfg = {
    "nr": {"mu": 2, "minislot_symbols": 7, "harq": {"k1_symbols": 1}},
    "urllc": {"pdb_ms": 10.0, "core_budget_ms": 2.0},
    "timing": {"grant_delay_ms": 0, "inter_attempt_gap_ms": 0.1, "proc_margin_ms": 0.2}
}

ctrl = S3TimingController(cfg)
print(ctrl.summary())
# Output:
# {
#   "mu": 2,
#   "attempt_latency_ms": 0.333,
#   "radio_deadline_ms": 8.0,
#   "K_max_by_deadline": 7,
#   ...
# }

# Generate timely BLER curves
bler_single = np.array([0.1, 0.05, 0.01])  # From S2
timely_curves = ctrl.timely_bler_curve(bler_single, K_list=[1, 2, 3])
# {1: array(...), 2: array(...), 3: array(...)}
```

## Integration: S2 → S3 → S4+

### Step 1: Run S2 BLER Sweep

Generate BLER vs SNR using your existing M2 simulator:

```bash
python scripts/run_sims.py --cfg configs/m2_ofdm_tdlc.yaml --out artifacts/m2_result.json
python scripts/run_bler.py --cfg configs/m2_ofdm_tdlc.yaml --out artifacts/bler_vs_snr.json
```

Output: `artifacts/bler_vs_snr.json` with format:
```json
{
  "snr_db": [0.0, 2.0, 4.0, 6.0, 8.0],
  "bler": [0.15, 0.10, 0.05, 0.01, 0.001],
  ...
}
```

### Step 2: Run S3 Timing Sweep

Use S3 to compute timely BLER for different K:

```bash
python scripts/run_s3_timely_sweep.py \
  --cfg configs/s3_timing_example.yaml \
  --bler-json artifacts/bler_vs_snr.json \
  --out artifacts/s3/results.json
```

Or run S2 + S3 in one go:

```bash
python scripts/run_s3_timely_sweep.py \
  --cfg configs/s3_timing_example.yaml \
  --run-bler \
  --out artifacts/s3/results.json
```

### Step 3: Inspect S3 Outputs

```
artifacts/s3/
├── timely_bler_vs_snr_K1.png     # BLER(K=1) vs SNR
├── timely_bler_vs_snr_K2.png     # BLER(K=2) vs SNR
├── timely_bler_vs_snr_K3.png     # BLER(K=3) vs SNR
├── latency_cdf.png               # Cumulative latency by K
├── s3_timely_heatmap.png         # SNR × K heatmap
└── s3_timely_bler_results.json   # Machine-readable results
```

## Configuration

### Required Blocks

#### `nr:` — Numerology and Timing

```yaml
nr:
  mu: 2                          # SCS: 0=15kHz, 1=30kHz, 2=60kHz, etc.
  minislot_symbols: 7            # L: symbols per mini-slot
  cg_ul:
    K: 2                         # Base K (swept separately in S3)
  harq:
    enabled: true
    k1_symbols: 1                # k1: HARQ feedback delay (symbols)
```

#### `urllc:` — Service Level

```yaml
urllc:
  pdb_ms: 10.0                   # Packet Delay Budget (from S1)
  core_budget_ms: 2.0            # Processing delay (from S1)
  # radio_deadline_ms = pdb_ms - core_budget_ms (computed internally)
```

#### `timing:` — Latency Components

```yaml
timing:
  grant_delay_ms: 0.0            # Scheduling delay before TX
  inter_attempt_gap_ms: 0.1      # Gap between attempts
  proc_margin_ms: 0.2            # Buffer for DSP/encoding
```

#### `s3_sweep:` — Sweep Parameters

```yaml
s3_sweep:
  K_list: [1, 2, 3]              # K values to sweep
  policy: "independent"          # BLER combination model
```

## Example Workflow

### 1. Generate S1 URLLC Budget

```bash
python scripts/urllc_budget.py \
  --cfg configs/s3_timing_example.yaml \
  --out artifacts/urllc_budget.json
```

### 2. Run M2 + S2 (BLER)

```bash
python scripts/run_sims.py --cfg configs/s3_timing_example.yaml
```

### 3. Run S3 (Timely BLER)

```bash
python scripts/run_s3_timely_sweep.py \
  --cfg configs/s3_timing_example.yaml \
  --bler-json artifacts/bler_vs_snr.json
```

### 4. Analyze Results

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load S3 results
with open("artifacts/s3/s3_timely_bler_results.json") as f:
    s3_data = json.load(f)

# Extract timing summary
timing = s3_data["timing_summary"]
print(f"Attempt latency: {timing['attempt_latency_ms']:.3f} ms")
print(f"Radio deadline: {timing['radio_deadline_ms']:.3f} ms")
print(f"Max K that fits: {timing['K_max_by_deadline']}")

# Plot timely BLER
points = s3_data["points"]
for K in [1, 2, 3]:
    pts = [p for p in points if p['K'] == K]
    snr = [p['snr_db'] for p in pts]
    bler = [p['timely_success_prob'] for p in pts]
    plt.semilogy(snr, bler, marker='o', label=f'K={K}')

plt.xlabel("SNR (dB)")
plt.ylabel("P(success ∧ timely)")
plt.legend()
plt.grid(True)
plt.show()
```

## Key Metrics (S3 → S6)

From S3, we define:

- **Reliability Within Deadline (RWD)**: P(TB success ∧ all K attempts finish before deadline)
- **Decoder Miss Rate (DMR)**: 1 − RWD
- **Latency CDF**: Distribution of cumulative latencies
- **Timely Goodput**: Throughput of packets that meet the deadline

## Testing

Run S3 unit tests:

```bash
pytest tests/test_s3_timing_urllc.py -v
```

Example test output:
```
test_s3_timing_urllc.py::TestAttemptLatency::test_basic_computation PASSED
test_s3_timing_urllc.py::TestCumulativeLatency::test_multiple_attempts PASSED
test_s3_timing_urllc.py::TestS3TimingController::test_K_fits_deadline PASSED
...
```

## Extending S3

### Custom k1 Delays

To model variable k1 (e.g., grant-free or scheduled):

```python
def get_k1_delay(attempt: int) -> int:
    """Custom k1 schedule: 1 symbol for first, 2 for second, etc."""
    return min(attempt, 3)

for attempt in range(K_max):
    k1_var = get_k1_delay(attempt)
    lat_attempt = attempt_latency_ms(mu=2, L=7, k1_symbols=k1_var)
    # accumulate...
```

### Custom Deadline Models

To handle multiple service levels:

```python
deadlines = {
    "URLLC": 5.0,      # Ultra-reliable, low-latency
    "eMBB": 50.0,      # Enhanced mobile broadband
}

for service, deadline_ms in deadlines.items():
    K_max = max_K_by_deadline(deadline_ms, attempt_latency_ms)
    print(f"{service}: K_max = {K_max}")
```

## Next Steps (S4+)

After S3 completes, proceed to:

- **S4** — Add VLC shim (optical IM/DD with same timing model)
- **S5** — Link adapters (per-link SNR → RWD mapping)
- **S6** — Hybrid policies (best_link, duplication, conditional_dup)
- **S7** — Scenarios and manifests (Pareto plots, manifests)

## Files Added / Modified

**New Files:**
- `nr_urllc/s3_timing_urllc.py` — Core S3 timing logic
- `nr_urllc/s3_timely_bler_sweep.py` — S3 sweep and plotting
- `scripts/run_s3_timely_sweep.py` — S3 runner
- `tests/test_s3_timing_urllc.py` — S3 unit tests
- `configs/s3_timing_example.yaml` — S3 config template
- `README_S3.md` — This file

**Modified Files:**
- (None in core; S3 is additive)

## References

- 3GPP TS 38.104 — NR numerologies and timing
- 3GPP TS 23.501 — 5QI service levels
- Your S1 URLLC budget derivation
- Your S2 BLER measurement results

---

**Status**: ✓ S3 Complete  
**Ready for**: S4 (VLC shim) or S5 (link adapters)

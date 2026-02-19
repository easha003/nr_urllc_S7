# Hybrid RF-VLC URLLC Simulation Framework

A comprehensive simulation framework for evaluating hybrid Radio Frequency (RF) and Visible Light Communication (VLC) systems designed to meet Ultra-Reliable Low-Latency Communication (URLLC) requirements for industrial IoT applications.

## Overview

This framework implements a predictive hybrid control system that intelligently selects between RF, VLC, or simultaneous transmission (duplication) to achieve ultra-reliable communication while minimizing energy consumption. The system leverages per-link AR(1) SNR prediction with blockage detection and lookup table (LUT)-based reliability mapping to make real-time link selection decisions.

## Key Features

- **Dual Physical Layer Simulation**: 5G NR OFDM for RF and DCO-OFDM for VLC with realistic channel models
- **Advanced Channel Modeling**: AR(1) dynamics with complementary fading and VLC blockage states
- **Predictive Control Framework**: Kalman-based SNR prediction with partial observability handling
- **Adaptive Hybrid Policies**: Best-Link selection and Conditional-DUP strategies
- **Comprehensive Evaluation**: Five diverse channel scenarios with statistical validation (100 runs × 1000 timesteps)
- **Energy-Performance Trade-off Analysis**: Pareto-optimal operation balancing reliability and energy efficiency

## System Architecture

The hybrid system combines two complementary wireless technologies:

### RF Link (5G NR)
- Sub-6 GHz cellular band operation (3.5 GHz)
- Slow fading dynamics (φ_RF = 0.94)
- Reliable baseline connectivity (up to 100m indoors)
- High energy consumption (E_RF = 1.0 normalized)
- HARQ with incremental redundancy

### VLC Link (LED-based Optical)
- Infrared/visible spectrum (850 nm)
- Energy-efficient transmission (E_VLC = 0.3, 3.3× lower than RF)
- Faster fading dynamics (φ_VLC = 0.88)
- High vulnerability to blockage (SNR drops to -15 dB)
- Limited range (5-10m line-of-sight)

## Repository Structure

```
.
├── nr_urllc/                      # Core simulation modules
│   ├── s3_timing_urllc.py         # URLLC timing and latency models
│   ├── s4_vlc_channel.py          # VLC physical layer (DCO-OFDM)
│   ├── adapters.py                # Link adapters and LUT management
│   ├── hybrid.py                  # Hybrid controller and policies
│   ├── predictors.py              # AR(1) SNR prediction
│   └── channel_sim.py             # Channel simulators for scenarios
│
├── scripts/                       # Execution scripts
│   ├── run_sims.py                # RF simulation (M2 BLER)
│   ├── run_s3_timely_sweep.py     # RF timely BLER sweep
│   ├── run_s4_vlc_sweep.py        # VLC simulation and sweep
│   ├── build_luts.py              # Build reliability LUTs (S5)
│   ├── run_s6_predecide.py        # Predictive controller testing
│   ├── run_s7_evaluation.py       # Full scenario evaluation
│   ├── generate_figures.py        # Generate result plots
│   └── generate_tables.py         # Generate performance tables
│
├── configs/                       # Configuration files
│   ├── m2_then_bler.yaml          # RF simulation config
│   ├── s3_timing_example.yaml     # RF timing configuration
│   ├── s4_vlc_example.yaml        # VLC simulation config
│   ├── s5_build_luts.yaml         # LUT construction config
│   ├── s6_predecide.yaml          # Predictive control config
│   └── s7_evaluation.yaml         # Scenario evaluation config
│
├── docs/                          # Documentation
│   ├── README_S3.md               # S3 timing model guide
│   ├── S4_VLC_DOCUMENTATION.md    # VLC implementation details
│   ├── S4_ARCHITECTURE_GUIDE.md   # S4 integration guide
│   └── README_S5_S6.md            # Adapter and controller docs
│
└── artifacts/                     # Output directory (generated)
    ├── m2_then_bler/              # RF BLER results
    ├── s3/                        # RF timely BLER curves
    ├── s4/                        # VLC results
    ├── adapters/                  # LUT files
    ├── s6/                        # Predictive control traces
    └── s7/                        # Full evaluation results
```

## Simulation Pipeline

The framework follows a structured pipeline (S3-S7) for comprehensive system evaluation:

### Stage 3 (S3): RF Physical Layer Simulation
Characterizes RF link performance with 5G NR OFDM including HARQ timing.

**Key Configuration**: `configs/m2_then_bler.yaml`
- **Numerology**: μ=2 (60 kHz subcarrier spacing)
- **Mini-slot**: 7 OFDM symbols
- **Modulation**: QPSK with polar codes (R=1/3)
- **Channel**: TDL-A multipath fading
- **HARQ**: k1=1 symbol feedback delay

```bash
cd nr_urllc_s4_3

# Run RF BLER simulation
python -m scripts.run_sims --cfg configs/m2_then_bler.yaml

# Generate timely BLER curves
python -m scripts.run_s3_timely_sweep \
  --cfg configs/s3_timing_example.yaml \
  --bler-json artifacts/m2_then_bler/bler_vs_snr.json \
  --out-dir artifacts/s3
```

**Outputs**:
- `bler_vs_snr.json`: Raw BLER vs SNR data
- `timely_bler_vs_snr_K*.png`: Timely BLER curves per K
- `latency_cdf.png`: Cumulative latency distribution
- `s3_timely_heatmap.png`: SNR × K success probability

### Stage 4 (S4): VLC Physical Layer Simulation
Implements VLC link with DCO-OFDM and realistic LED/photodetector models.

**Key Features**:
- **DCO-OFDM**: Hermitian symmetry for real-valued signals
- **LED Model**: 1st-order Butterworth (20 MHz 3dB bandwidth)
- **Clipping**: Models LED dynamic range constraints
- **Optical Noise**: Shot noise (signal-dependent) + AWGN

```bash
# Run VLC BLER sweep
python -m scripts.run_s4_vlc_sweep --cfg configs/s4_vlc_example.yaml
```

**Outputs**: 
- `s4_vlc_timely_results.json`: VLC timely BLER data
- VLC BLER curves and latency CDFs

### Stage 5 (S5): LUT Construction
Builds monotonic lookup tables mapping SNR to timely delivery probability for both links.

```bash
# Build reliability LUTs
python -m scripts.build_luts --cfg configs/s5_build_luts.yaml
```

**Outputs**:
- `artifacts/adapters/rf_lut.json`: RF reliability LUT
- `artifacts/adapters/vlc_lut.json`: VLC reliability LUT

**LUT Processing**:
1. Aggregation: Compute empirical p_timely from latency CDFs
2. Monotonicity enforcement: Isotonic regression on SNR
3. Interpolation: Linear interpolation for intermediate SNR values
4. Boundary handling: Clamp queries to [0, 24] dB range

### Stage 6 (S6): Predictive Control Testing
Tests the hybrid controller with AR(1) SNR prediction before full scenario evaluation.

```bash
# Run predictive control test
python -m scripts.run_s6_predecide --cfg configs/s6_predecide.yaml
```

**Controller Features**:
- **Per-link AR(1) prediction**: Mean-reverting SNR models with link-specific dynamics
- **VLC blockage detection**: Two-state Markov chain (p_enter=0.05, p_stay=0.7)
- **Confidence-based scoring**: Lower confidence bound (conf_k=1.64 for 95%)
- **Early decision**: Predict-then-decide before channel probing

**Outputs**:
- `artifacts/s6/predecide_trace.json`: Decision trace and metrics

### Notes on S6
LUT build accepts flexible JSON layouts: points, records, or tables per K; converts BLER→p automatically.
Interpolation is linear with monotonicity enforcement to avoid wiggles.
Adjust conf_k in s6_predecide.yaml (e.g., 1.64 ~ 95% one-sided) to be more conservative.

### Stage 7 (S7): Comprehensive Scenario Evaluation
Evaluates all policies across five diverse channel scenarios with rigorous statistical testing.

```bash
# Install dependencies (if needed)
pip install numpy scipy matplotlib pyyaml tqdm

# Run full evaluation
python -m scripts.run_s7_evaluation --cfg configs/s7_evaluation.yaml

# Generate all figures
python -m scripts.generate_figures

# Generate all tables
python -m scripts.generate_tables

# View summary
cat artifacts/s7/tables/summary_statistics.txt

# View detailed results
python -m json.tool artifacts/s7/complementary/summary.json
```

## Channel Scenarios

Five scenarios representing diverse RF-VLC operating conditions:

| Scenario | m_RF (dB) | m_VLC (dB) | Correlation | Description |
|----------|-----------|------------|-------------|-------------|
| **RF-Good** | 13.0 | 7.5 | ρ=0 | RF-dominated environment |
| **VLC-Good** | 3.0 | 17.5 | ρ=0 | VLC-dominated environment |
| **Complementary** | 8.0 | 12.5 | ρ=-0.3 | Negative correlation (diversity gain) |
| **Correlated Blockage** | 6.0 | 12.5 | Frequent VLC outages | p_enter=0.1, p_stay=0.8 |
| **Balanced** | 10.0 | 10.0 | ρ=0 | Equal link quality |

Each scenario runs for T=1000 timesteps (1 ms slots) over 100 independent runs.

## Hybrid Control Policies

### Baseline Policies
1. **RF-Only**: Always use RF link (K_RF=2, K_VLC=0)
2. **VLC-Only**: Always use VLC link (K_RF=0, K_VLC=2)
3. **Always-DUP**: Fixed duplication (K_RF=1, K_VLC=1)
4. **Oracle**: Perfect future CSI (upper bound)
5. **SNR-Threshold**: Static threshold-based selection

### Predictive Policies (Proposed)
6. **Best-Link**: Selects link with highest predicted reliability
   - AR(1) SNR prediction with link-specific dynamics
   - Lower confidence bound scoring (confidence-aware)
   - Exploration via ε-greedy (ε=0.05)
   - Hysteresis to prevent excessive switching

7. **Conditional-DUP**: Adaptive duplication based on uncertainty
   - Triggers DUP when: p_best < p_gate AND p_DUP ≥ p_gate
   - DUP threshold p_gate = 0.97 (URLLC target)
   - Balances reliability and energy efficiency

## Performance Metrics

- **Timely Delivery Ratio (TDR)**: Primary metric, targeting ≥99.9% for URLLC
- **Energy Efficiency**: Mean energy per packet (E_RF=1.0, E_VLC=0.3)
- **Action Distribution**: Fraction of RF/VLC/DUP transmissions
- **Prediction Accuracy**: Mean absolute error in SNR prediction

## Key Results

Experimental results demonstrate the effectiveness of predictive hybrid control:

### Overall Performance
- **Best-Link**: 74.9% average TDR (93% of Oracle performance)
- **Conditional-DUP**: 72.8% average TDR (90% of Oracle performance)
- **Static baselines**: ~50-70% TDR (scenario-dependent)

### Energy-Reliability Trade-off
- **Best-Link**: 35-50% energy savings vs Always-DUP with minimal TDR loss (<3%)
- **VLC-Good scenario**: VLC-Only, Best-Link achieve optimal energy-performance (TDR≥0.6, E~0.6)

### Scenario-Specific Insights
- **Complementary**: Best-Link (74.5%) and Conditional-DUP (73.3%) both approach Oracle (82.0%)
- **RF-Good**: Policies correctly favor RF (90%+ RF usage)
- **VLC-Good**: Policies adapt to use VLC primarily (60-80% VLC usage)

## Requirements

### Python Dependencies
```bash
pip install numpy scipy matplotlib pyyaml tqdm
```

### Recommended Python Version
- Python 3.8 or higher

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hybrid_rf_vlc

# Install dependencies
pip install numpy scipy matplotlib pyyaml tqdm --break-system-packages

# Verify installation
python -m scripts.run_sims --help
```

## Quick Start Example

```bash
# 1. Run RF simulation
python -m scripts.run_sims --cfg configs/m2_then_bler.yaml

# 2. Generate RF timely BLER curves
python -m scripts.run_s3_timely_sweep \
  --cfg configs/s3_timing_example.yaml \
  --bler-json artifacts/m2_then_bler/bler_vs_snr.json \
  --out-dir artifacts/s3

# 3. Run VLC simulation
python -m scripts.run_s4_vlc_sweep --cfg configs/s4_vlc_example.yaml

# 4. Build LUTs
python -m scripts.build_luts --cfg configs/s5_build_luts.yaml

# 5. Test predictive controller
python -m scripts.run_s6_predecide --cfg configs/s6_predecide.yaml

# 6. Run full evaluation
python -m scripts.run_s7_evaluation --cfg configs/s7_evaluation.yaml

# 7. Generate visualizations
python -m scripts.generate_figures
python -m scripts.generate_tables
```

## Configuration

All simulations are configured via YAML files in the `configs/` directory:

### RF Configuration (`m2_then_bler.yaml`)
- OFDM parameters (FFT size, subcarriers, cyclic prefix)
- Numerology (μ, L_symbols)
- Modulation and coding (QPSK, polar codes)
- Channel model (TDL-A)
- SNR sweep range

### VLC Configuration (`s4_vlc_example.yaml`)
- DCO-OFDM parameters
- LED characteristics (bandwidth, responsivity)
- DC bias and clipping
- Optical noise model
- Uses same timing as RF for fair comparison

### Timing Configuration (`s3_timing_example.yaml`)
- URLLC service level (PDB=10ms, core_budget=2ms)
- Grant delay, inter-attempt gap, processing margin
- HARQ parameters (k1 feedback delay)

### Evaluation Configuration (`s7_evaluation.yaml`)
- Policy definitions and parameters
- Scenario specifications
- Number of runs and timesteps
- Predictor configuration (AR(1) parameters)

## Documentation

Detailed documentation is available in the `docs/` directory:

- **README_S3.md**: Complete guide to RF timing and URLLC models
- **S4_VLC_DOCUMENTATION.md**: VLC implementation details and integration
- **S4_ARCHITECTURE_GUIDE.md**: Visual guide to VLC system architecture
- **README_S5_S6.md**: Link adapters and hybrid controller documentation

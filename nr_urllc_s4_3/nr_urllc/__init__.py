"""nr_urllc: Simulation framework for URLLC link-level experiments."""


# --- S3 timing (URLLC) ---
try:
    from .s3_timing_urllc import (
        attempt_latency_ms,
        cumulative_latency_K,
        max_K_by_deadline,
        bler_to_timely_bler,
        success_within_deadline,
        S3TimingController,
    )
    from .s3_timely_bler_sweep import (
        s3_timely_bler_sweep,
        S3TimelySweepResult,
        plot_timely_bler_curves,
        plot_latency_cdf,
        plot_s3_heatmap,
    )
except Exception as _e:
    # S3 modules may be optional in some environments
    pass

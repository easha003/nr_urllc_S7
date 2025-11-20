
import argparse, yaml, json
from pathlib import Path
from nr_urllc.sweep import bler_ofdm_sweep
from nr_urllc.urllc import attach_and_maybe_write
from nr_urllc.nr_cfg import mixin_nr_params
from nr_urllc.plots import plot_bler_curve

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--out", default="artifacts/bler_vs_snr.json")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    cfg = mixin_nr_params(cfg)

    # Attach S1 (will also write urllc_budget.json if io.write_json)
    result_holder = {}
    attach_and_maybe_write(result_holder, cfg)

    res = bler_ofdm_sweep(cfg)

    out = res  # already contains success, snr_db, bler, packets, meta
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print("[info] BLER saved to", args.out)

    # Optional plot if the YAML enables it
    io = cfg.get("io", {})
    if bool(io.get("plot", False)):
        label = f"TB={res['meta']['tb_payload_bytes']}B, M={res['meta']['M']}"
        out_png = io.get("out_plot", "artifacts/bler_vs_snr.png")
        plot_bler_curve(out, label=label, title="S2 — BLER vs SNR", save_path=out_png, show=bool(io.get("show_plot", False)))
        print("[info] BLER plot saved to", out_png)

if __name__ == "__main__":
    main()

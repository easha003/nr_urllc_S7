
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np, yaml

from nr_urllc.adapters import flatten_points, build_tables

def main():
    ap = argparse.ArgumentParser(description="S5: Build per-link LUTs from timely sweep JSONs.")
    ap.add_argument("--cfg", required=True, help="YAML with adapters.inputs/outputs, interp, snr_bounds.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    a = cfg["adapters"]
    rf_json = Path(a["inputs"]["rf_json"]).expanduser()
    vlc_json = Path(a["inputs"]["vlc_json"]).expanduser()
    rf_lut_out = Path(a["outputs"]["rf_lut"]).expanduser()
    vlc_lut_out = Path(a["outputs"]["vlc_lut"]).expanduser()

    def build(path: Path) -> Dict[str, Any]:
        obj = json.loads(path.read_text())
        pts, K_list = flatten_points(obj)
        lut = build_tables(pts, K_list)
        # override bounds if provided
        if "snr_bounds" in a:
            lut["snr_bounds"] = list(a["snr_bounds"])
        if "interp" in a:
            lut["interp"] = a["interp"]
        return lut

    rf_lut = build(rf_json)
    vlc_lut = build(vlc_json)

    rf_lut_out.parent.mkdir(parents=True, exist_ok=True)
    rf_lut_out.write_text(json.dumps(rf_lut, indent=2))
    vlc_lut_out.write_text(json.dumps(vlc_lut, indent=2))

    # monotonicity asserts
    for name, lut in [("RF", rf_lut), ("VLC", vlc_lut)]:
        for K in lut["K_list"]:
            ys = lut["tables"][str(K)]["p_timely"]
            if any(y2 < y1 for y1, y2 in zip(ys, ys[1:])):
                raise AssertionError(f"{name} LUT monotonicity failure for K={K}")
    print(f"[S5] LUTs built:\n  {rf_lut_out}\n  {vlc_lut_out}\n[S5] Monotonicity checks passed.")

if __name__ == "__main__":
    main()

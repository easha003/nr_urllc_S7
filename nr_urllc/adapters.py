
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import numpy as np

JsonLike = Dict[str, Any]

@dataclass
class LinkLUT:
    """
    Monotone p_timely(SNR, K) lookup built from sweep JSONs.
    """
    K_list: List[int]
    tables: Dict[str, Dict[str, List[float]]]
    interp: str = "linear"
    snr_bounds: Tuple[float, float] = (0.0, 24.0)

    @classmethod
    def from_json(cls, path: str | Path) -> "LinkLUT":
        path = Path(path)
        d = json.loads(path.read_text())
        # If it's already a LUT, just load it
        if "tables" in d and isinstance(d["tables"], dict):
            tables = {str(k): v for k, v in d["tables"].items()}
            klist = d.get("K_list") or sorted(int(k) for k in tables.keys())
            return cls(K_list=list(klist),
                       tables=tables,
                       interp=d.get("interp", "linear"),
                       snr_bounds=tuple(d.get("snr_bounds", [0.0, 24.0])))
        # else, attempt to convert "points" style to LUT on the fly
        points, K_list = flatten_points(d)
        lut = build_tables(points, K_list)
        return cls(**lut)

    def p_timely(self, snr_db: float, K: int) -> float:
        key = str(int(K))
        if key not in self.tables:
            raise KeyError(f"K={K} not in LUT; available={list(self.tables.keys())}")
        xs = np.asarray(self.tables[key]["snr_db"], dtype=float)
        ys = np.asarray(self.tables[key]["p_timely"], dtype=float)
        if xs.size == 0 or ys.size == 0:
            raise ValueError(f"LUT table for K={K} is empty.")
        # enforce monotonicity
        ys = np.maximum.accumulate(ys)
        lo, hi = float(self.snr_bounds[0]), float(self.snr_bounds[1])
        xq = float(np.clip(snr_db, lo, hi))
        val = float(np.interp(xq, xs, ys, left=ys[0], right=ys[-1]))
        return max(0.0, min(1.0, val))


# --------- Helpers shared with builder ---------
def flatten_points(obj: JsonLike) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Normalize various JSON shapes into a flat list of records:
      { "snr_db": float, "K": int, "p_timely": float }
    Returns (points, K_list)
    """
    pts: List[Dict[str, float]] = []
    Kset = set()

    def add_point(snr, K, p=None, bler=None):
        if p is None and bler is not None:
            p = 1.0 - float(bler)
        if p is None:
            return
        rec = {"snr_db": float(snr), "K": int(K), "p_timely": float(p)}
        pts.append(rec); Kset.add(int(K))

    # Case A: explicit "points" list
    if isinstance(obj.get("points"), list) and obj["points"]:
        for p in obj["points"]:
            snr = p.get("snr_db")
            K = p.get("K") or p.get("k") or p.get("attempts")
            if snr is None or K is None: 
                continue
            add_point(snr, K, p.get("timely_success_prob") or p.get("p_timely"),
                      p.get("timely_bler") or p.get("bler_K") or p.get("bler") or p.get("tb_bler"))
        if pts:
            return pts, sorted(Kset)

    # Case B: "records" list
    if isinstance(obj.get("records"), list) and obj["records"]:
        for p in obj["records"]:
            snr = p.get("snr_db")
            K = p.get("K") or p.get("k") or p.get("attempts")
            if snr is None or K is None:
                continue
            add_point(snr, K, p.get("timely_success_prob") or p.get("p_timely"),
                      p.get("timely_bler") or p.get("bler_K") or p.get("bler") or p.get("tb_bler"))
        if pts:
            return pts, sorted(Kset)

    # Case C: tables per K with arrays
    tables = obj.get("tables")
    if isinstance(tables, dict) and tables:
        for K, rec in tables.items():
            Kint = int(K)
            xs = rec.get("snr_db") or rec.get("snr") or rec.get("x")
            # p arrays can be named differently
            p_arr = rec.get("p_timely") or rec.get("timely_success_prob")
            bler_arr = rec.get("timely_bler") or rec.get("bler_K") or rec.get("bler")
            if xs and (p_arr or bler_arr):
                n = min(len(xs), len(p_arr or bler_arr))
                for i in range(n):
                    add_point(xs[i], Kint, (p_arr[i] if p_arr else None), (bler_arr[i] if bler_arr else None))
        if pts:
            return pts, sorted(Kset)

    # Case D: dict of K -> list of points
    for key, val in obj.items():
        try:
            Kint = int(key)
        except Exception:
            continue
        if isinstance(val, list):
            for p in val:
                snr = p.get("snr_db") or p.get("snr")
                add_point(snr, Kint, p.get("timely_success_prob") or p.get("p_timely"),
                          p.get("timely_bler") or p.get("bler_K") or p.get("bler"))
        if pts:
            return pts, sorted(Kset)

    raise ValueError("Unrecognized JSON layout for timely sweep data. Expect 'points' or 'tables'.")


def build_tables(points: List[Dict[str, float]], K_list: List[int] | None) -> Dict[str, Any]:
    if not points:
        raise ValueError("No points provided for LUT build.")
    if not K_list:
        K_list = sorted({int(p["K"]) for p in points})
    out = {"K_list": list(K_list), "tables": {}, "interp": "linear", "snr_bounds": [0.0, 24.0]}
    for K in K_list:
        kp = [p for p in points if int(p["K"]) == int(K)]
        kp.sort(key=lambda p: float(p["snr_db"]))
        xs = np.asarray([float(p["snr_db"]) for p in kp], dtype=float)
        ys = np.asarray([float(p["p_timely"]) for p in kp], dtype=float)
        ys = np.maximum.accumulate(ys)  # enforce monotone
        out["tables"][str(K)] = {"snr_db": xs.tolist(), "p_timely": ys.tolist()}
    return out

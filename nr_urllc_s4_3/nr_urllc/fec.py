import numpy as np
from typing import Dict, Tuple, Optional
from functools import lru_cache
from typing import NamedTuple

class _LDPCStruct(NamedTuple):
    P: np.ndarray
    H: np.ndarray

@lru_cache(maxsize=64)
def _get_struct(k: int, n: int, seed: int, density: float) -> _LDPCStruct:
    r = int(n - k)
    P = _build_P(k, r, seed, density)
    H = np.concatenate([P.T, np.eye(r, dtype=np.uint8)], axis=1)
    return _LDPCStruct(P, H)


# ---------- Public API ----------
def get_rate(cfg: Dict) -> float:
    fec = dict(cfg or {})
    t = str(fec.get("type", "none")).lower()
    if t == "none": return 1.0
    if t == "repeat": return 1.0 / float(max(1, int(fec.get("times", 2))))
    if t in ("ldpc_syst", "ldpc_qc"): return float(fec.get("rate", 0.5))
    return 1.0

def encode(info_bits: np.ndarray, cfg: Dict | None = None) -> Tuple[np.ndarray, Dict]:
    if cfg is None: cfg = {}
    fec = dict(cfg); t = str(fec.get("type", "none")).lower()
    info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
    k = int(info.size)
    if t == "none":
        return info.copy(), {"type": "none", "n": k, "k": k}
    if t == "repeat":
        r = int(max(1, fec.get("times", 2)))
        c = np.repeat(info, r).astype(np.uint8)
        return c, {"type": "repeat", "k": k, "n": int(c.size), "times": r}
    if t == "ldpc_syst":
        return _encode_ldpc_syst(info, fec)
    if t == "ldpc_qc":
        return _encode_ldpc_qc(info, fec)
    return info.copy(), {"type": "none", "n": k, "k": k}

def decode(rx_bits: np.ndarray, cfg: Dict | None = None, meta: Dict | None = None, info_len: Optional[int] = None) -> np.ndarray:
    """Hard-decision fallback (kept for compatibility)."""
    if cfg is None: cfg = {}
    fec = dict(cfg); t = str(fec.get("type", "none")).lower()
    v = np.asarray(rx_bits, dtype=np.uint8).reshape(-1)
    if t == "none": return v if info_len is None else v[:info_len]
    if t == "repeat":
        r = int(max(1, fec.get("times", 2)))
        v = v[: (v.size // r) * r]
        hard = (v.reshape(-1, r).sum(axis=1) > (r // 2)).astype(np.uint8)
        return hard if info_len is None else hard[:info_len]
    if t == "ldpc_syst":
        return _decode_flip_ldpc_syst(v, fec, meta, info_len)
    if t == "ldpc_qc":
        return _decode_flip_ldpc_qc(v, fec, meta, info_len)
    return v if info_len is None else v[:info_len]

def decode_soft(llr: np.ndarray, cfg: Dict | None = None, meta: Dict | None = None, info_len: Optional[int] = None) -> np.ndarray:
    """Soft-decision entry (LLR vector)."""
    if cfg is None: cfg = {}
    fec = dict(cfg); t = str(fec.get("type", "none")).lower()
    L = np.asarray(llr, dtype=float).reshape(-1)
    if t == "none": return (L < 0).astype(np.uint8)[:info_len] if info_len else (L < 0).astype(np.uint8)
    if t == "repeat":
        r = int(max(1, fec.get("times", 2)))
        L = L[: (L.size // r) * r]
        hard = (L.reshape(-1, r).sum(axis=1) < 0).astype(np.uint8)
        return hard if info_len is None else hard[:info_len]
    if t == "ldpc_syst":
        return _decode_min_sum_ldpc_syst(L, fec, meta, info_len)
    if t == "ldpc_qc":
        return _decode_min_sum_ldpc_qc(L, fec, meta, info_len)
    return (L < 0).astype(np.uint8) if info_len is None else (L < 0).astype(np.uint8)[:info_len]

# ---------- LDPC (random systematic) ----------
def _build_P(k, r, seed, density):
    rng = np.random.default_rng(seed)
    P = (rng.random((k, r)) < density).astype(np.uint8)
    for j in range(r):
        if P[:, j].sum() == 0:
            P[rng.integers(0, k), j] = 1
    return P

def _encode_ldpc_syst(info: np.ndarray, fec: Dict):
    k = int(info.size)
    rate = float(fec.get("rate", 0.5))
    n_cfg = int(fec.get("n", 0))
    n = int(n_cfg) if n_cfg > 0 else int(np.ceil(k / max(rate, 1e-6) / 8.0) * 8)
    n = max(n, k); r = n - k
    if r <= 0: 
        return info.copy(), {"type":"none","n":k,"k":k}
    seed = int(fec.get("seed", 1234)); density = float(fec.get("density", 0.05))

    # NEW: reuse parity structure
    P = _get_struct(k, n, seed, density).P
    G = np.concatenate([np.eye(k, dtype=np.uint8), P], axis=1)

    c = (info @ G) % 2
    return c.astype(np.uint8), {"type":"ldpc_syst","k":k,"n":n,"r":r,"seed":seed,"density":density}

def _decode_flip_ldpc_syst(vbits, fec, meta, info_len):
    if meta is None:
        k = int(info_len or vbits.size)
        n = int(fec.get("n", vbits.size))
        seed = int(fec.get("seed", 1234)); density = float(fec.get("density", 0.05))
    else:
        k = int(meta["k"]); n = int(meta["n"])
        seed = int(meta["seed"]); density = float(meta["density"])
    r = n - k; v = vbits[:n].astype(np.uint8)
    P = _build_P(k, r, seed, density)
    H = np.concatenate([P.T, np.eye(r, dtype=np.uint8)], axis=1)
    max_iters = int(fec.get("max_iters", 20))
    x = v.copy()
    for it in range(max_iters):
        s = (H @ x) % 2
        if s.sum() == 0: break
        counts = (H.T @ s)
        idx = np.where(counts == counts.max())[0]
        if idx.size == 0: break
        flip = idx if idx.size <= max(n // 10, 1) else np.random.default_rng(999+it).choice(idx, size=max(n//10,1), replace=False)
        x[flip] ^= 1
    return x[:k] if info_len is None else x[:info_len]

def _decode_min_sum_ldpc_syst(llr, fec, meta, info_len):
    if meta is None:
        rate = float(fec.get("rate", 0.5))
        n = int(fec.get("n", llr.size))
        k = int(np.floor(rate * n))
        seed = int(fec.get("seed", 1234)); density = float(fec.get("density", 0.05))
    else:
        k = int(meta["k"]); n = int(meta["n"])
        seed = int(meta["seed"]); density = float(meta["density"])

    L = llr[:n].astype(float)

    # NEW: reuse H
    H = _get_struct(k, n, seed, density).H

    hard = _min_sum_decode(H, L, max_iters=int(fec.get("max_iters", 40)))
    return hard[:k] if info_len is None else hard[:info_len]


# ---------- QC-LDPC hook (toy BG; swap with 38.212 later) ----------
def _builtin_bg2_tiny():
    return np.array([
        [  0, -1, 13, 22, -1,  7, -1, 19, -1, -1,  5, -1],
        [ -1, 11, -1, -1, 21, -1, 17, -1, -1,  9, -1,  3],
        [ 14, -1, -1, 10, -1, -1, -1,  6, 23, -1,  2, -1],
        [ -1, 20, 12, -1, -1,  4, -1, -1,  8, -1, -1, 16],
        [  9, -1, -1, -1, 15, -1, -1,  1, -1, 18, -1, -1],
        [ -1, -1,  7, -1, -1, -1,  5, -1, 12, -1,  0, -1],
    ], dtype=int)

def _expand_QC_H(B: np.ndarray, Z: int) -> np.ndarray:
    rB, cB = B.shape; n = cB * Z; m = rB * Z
    H = np.zeros((m, n), dtype=np.uint8); I = np.eye(Z, dtype=np.uint8)
    for i in range(rB):
        for j in range(cB):
            s = B[i, j]
            if s < 0: continue
            H[i*Z:(i+1)*Z, j*Z:(j+1)*Z] = np.roll(I, shift=s % Z, axis=1)
    return H

def _invert_binary_matrix(M: np.ndarray) -> np.ndarray:
    M = M.copy().astype(np.uint8); n = M.shape[0]; I = np.eye(n, dtype=np.uint8)
    A = np.concatenate([M, I], axis=1); r = c = 0
    while r < n and c < n:
        piv = np.where(A[r:, c] == 1)[0]
        if piv.size == 0: c += 1; continue
        piv = piv[0] + r
        if piv != r: A[[r, piv]] = A[[piv, r]]
        for rr in range(n):
            if rr != r and A[rr, c] == 1: A[rr, :] ^= A[r, :]
        r += 1; c += 1
    return A[:, n:]

def _encode_ldpc_qc(info: np.ndarray, fec: Dict):
    qc = dict(fec.get("qc", {})); Z = int(qc.get("lifting_Z", 32))
    B = _builtin_bg2_tiny() if bool(qc.get("use_builtin_bg2_tiny", True)) else _load_bg_from_json(qc["base_graph_path"])
    H = _expand_QC_H(B, Z); m, n = H.shape; k = n - m
    A = H[:, :k].astype(np.uint8); Bm = H[:, k:].astype(np.uint8)
    Binv = _invert_binary_matrix(Bm)
    info = info[:k].astype(np.uint8)
    p = (Binv @ (A @ info % 2)) % 2
    code = np.concatenate([info, p], axis=0)
    return code, {"type":"ldpc_qc","n":n,"k":k,"m":m,"Z":Z}

def _decode_flip_ldpc_qc(vbits, fec, meta, info_len):
    L = (1 - 2*vbits.astype(np.float64)) * 10.0
    return _decode_min_sum_ldpc_qc(L, fec, meta, info_len)

def _decode_min_sum_ldpc_qc(llr, fec, meta, info_len):
    qc = dict(fec.get("qc", {})); Z = int(qc.get("lifting_Z", meta.get("Z", 32)))
    B = _builtin_bg2_tiny() if bool(qc.get("use_builtin_bg2_tiny", True)) else _load_bg_from_json(qc["base_graph_path"])
    H = _expand_QC_H(B, Z)
    hard = _min_sum_decode(H, llr[:H.shape[1]].astype(float), max_iters=int(fec.get("max_iters", 40)))
    k = int(meta["k"])
    return hard[:k] if info_len is None else hard[:info_len]

def _load_bg_from_json(pth):
    import json, pathlib
    p = pathlib.Path(pth).expanduser().resolve()
    return np.array(json.loads(p.read_text())["B"], dtype=int)

# ---------- Common min-sum ----------
def _min_sum_decode(H: np.ndarray, Lch: np.ndarray, max_iters: int = 40) -> np.ndarray:
    m, n = H.shape
    rows = [np.where(H[i, :] == 1)[0] for i in range(m)]
    cols = [np.where(H[:, j] == 1)[0] for j in range(n)]
    v2c = {(i, j): 0.0 for j in range(n) for i in cols[j]}
    c2v = {(i, j): 0.0 for i in range(m) for j in rows[i]}
    for _ in range(max_iters):
        # check node
        for i in range(m):
            js = rows[i]
            msgs = np.array([Lch[j] - v2c[(i, j)] for j in js], dtype=float)
            signs = np.sign(msgs); signs[signs == 0] = 1.0
            prod = np.prod(signs)
            mags = np.abs(msgs)
            for idx, j in enumerate(js):
                m_excl = np.min(np.delete(mags, idx)) if js.size > 1 else mags[idx]
                sign_excl = prod * signs[idx]
                c2v[(i, j)] = sign_excl * m_excl
        # variable node
        hard = np.zeros(n, dtype=np.uint8)
        for j in range(n):
            is_ = cols[j]
            Lj = Lch[j] + sum(c2v[(i, j)] for i in is_)
            for i in is_:
                v2c[(i, j)] = Lj - c2v[(i, j)]
            hard[j] = 0 if Lj >= 0 else 1
        if ((H @ hard) % 2).sum() == 0:
            return hard
    return hard

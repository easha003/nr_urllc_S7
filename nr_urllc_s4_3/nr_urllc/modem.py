# nr_urllc/modem.py
import numpy as np

__all__ = ["qpsk_gray_map", "qam16_gray_map"]

def qpsk_gray_map(bits: np.ndarray) -> np.ndarray:
    """
    Gray QPSK mapping with Es=1.
    Input: bits shape (..., 2)
    Output: complex symbols shape (...,)
    """
    b = bits.reshape(-1, 2).astype(np.int8)
    i = 1 - 2*b[:,0]   # 1 -> -1, 0 -> +1
    q = 1 - 2*b[:,1]
    # unnormalized Es per symbol = 2; scale by 1/sqrt(2)
    s = (i + 1j*q) / np.sqrt(2.0)
    return s.astype(np.complex64)

def qam16_gray_map(bits: np.ndarray) -> np.ndarray:
    """
    Square 16-QAM Gray mapping with Es=1.
    Input: bits shape (..., 4)
    Output: complex symbols shape (...,)
    Gray on each axis: b0b1 -> {±1, ±3}
    """
    b = bits.reshape(-1, 4).astype(np.int8)

    # Map 2 bits → PAM4 Gray levels: 00→+3, 01→+1, 11→−1, 10→−3
    def pam4_gray(two_bits):
        b0, b1 = int(two_bits[0]), int(two_bits[1])
        if b0 == 0 and b1 == 0: return +3
        if b0 == 0 and b1 == 1: return +1
        if b0 == 1 and b1 == 1: return -1
        return -3  # b0==1 and b1==0

    I = np.fromiter((pam4_gray(bb) for bb in b[:,0:2]), dtype=np.int8, count=b.shape[0])
    Q = np.fromiter((pam4_gray(bb) for bb in b[:,2:4]), dtype=np.int8, count=b.shape[0])

    # Average energy of {±1,±3}×{±1,±3} is 10 → scale by 1/sqrt(10) for Es=1
    s = (I.astype(np.float32) + 1j*Q.astype(np.float32)) / np.sqrt(10.0, dtype=np.float32)
    return s.astype(np.complex64)

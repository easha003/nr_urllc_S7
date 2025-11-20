
from __future__ import annotations
import numpy as np

CRC24A_POLY = 0x1864CFB
CRC24A_INIT = 0x000000

def crc24a(data: bytes) -> int:
    crc = CRC24A_INIT
    for b in data:
        crc ^= (b << 16)
        for _ in range(8):
            if crc & 0x800000:
                crc = ((crc << 1) & 0xFFFFFF) ^ CRC24A_POLY
            else:
                crc = (crc << 1) & 0xFFFFFF
    return crc & 0xFFFFFF

def append_crc(data: bytes) -> bytes:
    crc = crc24a(data)
    return data + crc.to_bytes(3, byteorder="big")

def check_crc(tb: bytes) -> bool:
    if len(tb) < 3: return False
    data = tb[:-3]
    calc = crc24a(data)
    rx = int.from_bytes(tb[-3:], byteorder="big")
    return calc == rx

def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.int8, copy=False)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) % 8 != 0:
        bits = bits[: len(bits) - (len(bits) % 8)]
    arr = np.packbits(bits, bitorder="big")
    return bytes(arr.tolist())

"""LSB-first deterministic three-bit metadata packing."""

from __future__ import annotations

from .metadata import decode_metadata


def pack_metadata(values: list[int]) -> bytes:
    """Pack traversal-order metadata values into an LSB-first bit stream."""
    for value in values:
        decode_metadata(value)
    output = bytearray((len(values) * 3 + 7) // 8)
    for index, value in enumerate(values):
        for bit in range(3):
            if int(value) & (1 << bit):
                bit_offset = index * 3 + bit
                output[bit_offset // 8] |= 1 << (bit_offset % 8)
    return bytes(output)


def unpack_metadata(payload: bytes, count: int) -> list[int]:
    expected = (count * 3 + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"metadata payload length must be {expected} bytes")
    values: list[int] = []
    for index in range(count):
        value = 0
        for bit in range(3):
            bit_offset = index * 3 + bit
            value |= ((payload[bit_offset // 8] >> (bit_offset % 8)) & 1) << bit
        decode_metadata(value)
        values.append(value)
    if count * 3 % 8 and payload[-1] >> (count * 3 % 8):
        raise ValueError("metadata padding bits must be zero")
    return values

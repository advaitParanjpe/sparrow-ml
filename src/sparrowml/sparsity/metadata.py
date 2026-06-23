"""Sparrow-V-compatible metadata mapping for one 2:4 group."""

from __future__ import annotations

LANES_TO_METADATA = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}
METADATA_TO_LANES = {value: key for key, value in LANES_TO_METADATA.items()}


def encode_lanes(lanes: tuple[int, int] | list[int]) -> int:
    ordered = tuple(int(lane) for lane in lanes)
    if ordered not in LANES_TO_METADATA:
        raise ValueError("2:4 selected lanes must be two ascending distinct lanes in [0, 3]")
    return LANES_TO_METADATA[ordered]


def decode_metadata(metadata: int) -> tuple[int, int]:
    try:
        return METADATA_TO_LANES[int(metadata)]
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid 2:4 metadata; 110 and 111 are reserved") from exc

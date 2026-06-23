"""Small, canonical Phase 4 IR for the two supported linear deployment graphs."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any

IR_FORMAT_VERSION = "sparrowml_ir_v1"
ELEMENT_BYTES = {"int8": 1, "uint8": 1, "int32": 4, "float32": 4}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def ir_hash(ir: dict[str, Any]) -> str:
    return sha256(canonical_json(ir).encode("utf-8")).hexdigest()


def parse_ir(payload: str | bytes) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("IR root must be an object")
    from .validation import validate_ir
    validate_ir(value)
    return value


def serialize_ir(ir: dict[str, Any]) -> bytes:
    from .validation import validate_ir
    validate_ir(ir)
    return canonical_json(ir).encode("utf-8")


def relative_identity(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("source_artifact_identity must be repository-relative")
    return value

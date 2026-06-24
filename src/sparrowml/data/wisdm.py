"""Deterministic, subject-safe preparation of WISDM phone accelerometer data."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CLASS_NAMES = ("walking", "jogging", "sitting", "standing")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
FEATURE_NAMES = (
    "x_mean", "y_mean", "z_mean", "magnitude_mean", "x_std", "y_std", "z_std",
    "magnitude_std", "magnitude_min", "magnitude_max", "magnitude_rms", "magnitude_energy",
    "magnitude_mean_absolute_difference", "magnitude_zero_crossing_rate",
    "magnitude_dominant_frequency_magnitude", "magnitude_spectral_entropy",
)
ACTIVITY_ALIASES = {"walking": "walking", "jogging": "jogging", "sitting": "sitting", "standing": "standing"}
WINDOW_LENGTH, WINDOW_STRIDE, NOMINAL_HZ, GAP_US = 80, 40, 20.0, 125_000_000

@dataclass(frozen=True)
class Record:
    subject_id: str; activity: str; source_activity_code: str; timestamp: int
    x: float; y: float; z: float; source_file: str; source_row: int

def resolve_root(root: str | Path | None = None) -> Path:
    candidate = Path(root or os.environ.get("WISDM_ROOT", "~/Datasets/WISDM/wisdm-dataset")).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError("WISDM dataset root was not found; set WISDM_ROOT or place it at ~/Datasets/WISDM/wisdm-dataset")
    return candidate

def _activity_key(root: Path) -> tuple[dict[str, str], bool]:
    file = root / "activity_key.txt"
    if not file.is_file(): raise ValueError("WISDM activity_key.txt is required; refusing to guess activity-code mappings")
    mapping: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line: continue
        name, code = (part.strip() for part in line.split("=", 1))
        canonical = ACTIVITY_ALIASES.get(name.lower())
        if canonical: mapping[code] = canonical
    absent = set(CLASS_NAMES) - set(mapping.values())
    if absent: raise ValueError("activity key lacks required activities: " + ", ".join(sorted(absent)))
    return mapping, True

def _files(root: Path) -> list[Path]:
    directory = root / "raw/phone/accel"
    if not directory.is_dir(): raise ValueError("WISDM phone accelerometer directory raw/phone/accel was not found")
    return sorted(path for path in directory.glob("*.txt") if path.is_file())

def parse_records(root: Path) -> tuple[list[Record], dict[str, Any], dict[str, str]]:
    mapping, metadata = _activity_key(root); records: list[Record] = []; malformed = 0; unknown: Counter[str] = Counter(); duplicates = 0; seen: set[tuple[Any, ...]] = set()
    for file in _files(root):
        source = file.name
        for row, raw in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            line = raw.strip().rstrip(";").strip()
            if not line: continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6 or not fields[0] or not fields[1]: malformed += 1; continue
            try: subject, code, timestamp = fields[0], fields[1], int(fields[2]); x, y, z = map(float, fields[3:])
            except ValueError: malformed += 1; continue
            if not all(math.isfinite(v) for v in (x, y, z)): malformed += 1; continue
            activity = mapping.get(code)
            if activity is None: unknown[code] += 1; continue
            key = (subject, code, timestamp, x, y, z, source)
            if key in seen: duplicates += 1; continue
            seen.add(key); records.append(Record(subject, activity, code, timestamp, x, y, z, source, row))
    records.sort(key=lambda r: (r.subject_id, r.source_file, r.activity, r.timestamp, r.source_row))
    return records, {"malformed_rows": malformed, "unknown_activity_rows": dict(sorted(unknown.items())), "duplicate_rows": duplicates, "activity_key_exists": metadata}, mapping

def subject_splits(records: Iterable[Record], seed: int = 20260623) -> tuple[dict[str, list[str]], dict[str, Any]]:
    activities: dict[str, set[str]] = defaultdict(set)
    for record in records: activities[record.subject_id].add(record.activity)
    eligible = sorted(subject for subject, seen in activities.items() if set(CLASS_NAMES) <= seen)
    if len(eligible) < 3: raise ValueError("too few eligible WISDM subjects containing all four activities")
    ranked = sorted(eligible, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
    count = len(ranked); validation = max(1, round(count * .15)); test = max(1, round(count * .15)); train = count - validation - test
    if train < 1: raise ValueError("cannot allocate nonempty subject splits")
    splits = {"train": sorted(ranked[:train]), "validation": sorted(ranked[train:train + validation]), "test": sorted(ranked[train + validation:])}
    eligibility = {subject: {"eligible": subject in eligible, "activities": sorted(activities[subject]), "reason": None if subject in eligible else "missing_required_activity"} for subject in sorted(activities)}
    return splits, {"seed": seed, "raw_subject_count": len(activities), "eligible_subject_count": len(eligible), "subjects": eligibility}

def features(samples: list[Record], sampling_rate: float = NOMINAL_HZ) -> list[float]:
    values = np.asarray([[r.x, r.y, r.z] for r in samples], dtype=np.float64); magnitude = np.linalg.norm(values, axis=1)
    centered = magnitude - magnitude.mean(); signs = np.sign(centered); last = 1.0
    normalized = []
    for sign in signs:
        if sign == 0: normalized.append(last)
        else: last = float(sign); normalized.append(last)
    crossings = sum(a != b for a, b in zip(normalized, normalized[1:]))
    spectrum = np.fft.rfft(centered); power = np.abs(spectrum) ** 2
    if len(power) > 1: dominant = float(np.abs(spectrum[1:]).max())
    else: dominant = 0.0
    non_dc = power[1:]; total = float(non_dc.sum())
    if total == 0 or len(non_dc) <= 1:
        entropy = 0.0
    else:
        probabilities = non_dc / total
        entropy = float(-(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0.0)).sum() / math.log2(len(probabilities)))
    output = [*values.mean(axis=0), float(magnitude.mean()), *values.std(axis=0), float(magnitude.std()), float(magnitude.min()), float(magnitude.max()), float(np.sqrt(np.mean(magnitude ** 2))), float(np.mean(magnitude ** 2)), float(np.mean(np.abs(np.diff(magnitude))) if len(magnitude) > 1 else 0.0), float(crossings / max(1, len(magnitude) - 1)), dominant, entropy]
    if len(output) != 16 or not np.isfinite(output).all(): raise ValueError("non-finite WISDM feature vector")
    return [float(v) for v in output]

def prepare(root: str | Path | None = None, output: Path | None = None) -> dict[str, Any]:
    dataset_root = resolve_root(root); records, parse, mapping = parse_records(dataset_root); splits, eligibility = subject_splits(records)
    output = output or Path("artifacts/phase8_wisdm/phase8a"); output.mkdir(parents=True, exist_ok=True)
    subject_to_split = {subject: split for split, subjects in splits.items() for subject in subjects}; accepted: list[dict[str, Any]] = []; rejected: Counter[str] = Counter(); groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for record in records:
        if record.subject_id in subject_to_split: groups[(record.subject_id, record.activity, record.source_file)].append(record)
    for (subject, activity, source), group in sorted(groups.items()):
        group.sort(key=lambda r: (r.timestamp, r.source_row)); segment: list[Record] = []
        def consume(items: list[Record]) -> None:
            if len(items) < WINDOW_LENGTH: rejected["incomplete_segment"] += 1; return
            for start in range(0, len(items) - WINDOW_LENGTH + 1, WINDOW_STRIDE):
                window = items[start:start + WINDOW_LENGTH]
                if len(window) != WINDOW_LENGTH: rejected["incomplete_window"] += 1; continue
                uid = hashlib.sha256(f"{subject}|{activity}|{source}|{window[0].source_row}|{window[-1].source_row}".encode()).hexdigest()[:20]
                accepted.append({"window_id": f"wisdm-{uid}", "subject_id": subject, "activity": activity, "class_id": CLASS_IDS[activity], "split": subject_to_split[subject], "features": features(window), "source_file": source, "source_row_start": window[0].source_row, "source_row_end": window[-1].source_row, "start_timestamp": window[0].timestamp, "end_timestamp": window[-1].timestamp, "sample_count": WINDOW_LENGTH, "sampling_rate_hz": NOMINAL_HZ, "feature_extractor_version": "wisdm_features_v1"})
        previous = None
        for item in group:
            if previous is not None and (item.timestamp <= previous.timestamp or item.timestamp - previous.timestamp > GAP_US):
                rejected["timestamp_discontinuity"] += 1; consume(segment); segment = []
            segment.append(item); previous = item
        consume(segment)
    accepted.sort(key=lambda item: item["window_id"])
    if len({x["window_id"] for x in accepted}) != len(accepted): raise ValueError("duplicate WISDM window IDs")
    counts = lambda key: dict(sorted(Counter(item[key] for item in accepted).items()))
    by_split_class = {split: {activity: sum(x["split"] == split and x["activity"] == activity for x in accepted) for activity in CLASS_NAMES} for split in splits}
    if any(value == 0 for row in by_split_class.values() for value in row.values()): raise ValueError("a subject split lacks a required activity window")
    stats = np.asarray([x["features"] for x in accepted], dtype=np.float64)
    def dump(name: str, payload: Any) -> None: (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    dump("dataset_discovery.json", {"dataset_root_serialized": "WISDM_ROOT", "activity_key_exists": parse["activity_key_exists"], "phone_accelerometer_directory": "raw/phone/accel", "subject_file_count": len(_files(dataset_root)), "subject_ids": sorted({r.subject_id for r in records}), "discovered_activity_codes": sorted({r.source_activity_code for r in records}), "mapped_activities": mapping, **parse, "usable": True})
    dump("activity_mapping.json", {"source": "activity_key.txt", "source_code_to_canonical": mapping, "class_order": list(CLASS_NAMES)})
    dump("subject_eligibility.json", eligibility); dump("subject_splits.json", {"split_seed": 20260623, "splits": splits, "split_sizes": {k: len(v) for k,v in splits.items()}})
    dump("recording_manifest.json", {"source_files": sorted({r.source_file for r in records}), "record_count": len(records), "path_policy": "source identities are basenames; no absolute paths"})
    dump("window_manifest.json", {"format_version": "wisdm_window_manifest_v1", "windows": accepted})
    dump("feature_schema.json", {"version": "wisdm_features_v1", "feature_order": list(FEATURE_NAMES), "sampling_rate_hz": NOMINAL_HZ, "window_length": WINDOW_LENGTH, "stride": WINDOW_STRIDE, "timestamp_gap_nanoseconds": GAP_US, "zero_crossing": "mean-center magnitude; exact zero inherits previous nonzero sign, initial sign positive", "frequency": "mean-removed rFFT; DC excluded; entropy is normalized non-DC power"})
    dump("feature_statistics.json", {"minimum": stats.min(0).tolist(), "maximum": stats.max(0).tolist(), "mean": stats.mean(0).tolist(), "std": stats.std(0).tolist()})
    audit = {"raw_subject_count": eligibility["raw_subject_count"], "eligible_subject_count": eligibility["eligible_subject_count"], "source_files_parsed": len(_files(dataset_root)), "raw_rows_per_class": dict(sorted(Counter(r.activity for r in records).items())), **parse, "accepted_windows": len(accepted), "accepted_windows_per_subject": counts("subject_id"), "accepted_windows_per_class": counts("activity"), "accepted_windows_per_split": counts("split"), "accepted_windows_per_split_class": by_split_class, "rejected_windows_by_reason": dict(sorted(rejected.items())), "feature_finite": bool(np.isfinite(stats).all()), "duplicate_window_ids": 0, "subject_overlap": False, "class_balance_flag": any(max(row.values()) / min(row.values()) > 3 for row in by_split_class.values()), "subject_lists": splits}
    dump("dataset_audit.json", audit)
    (output / "summary.md").write_text("# Phase 8A WISDM preparation\n\nSubject-held-out phone accelerometer windows were prepared without model training. Canonical artifacts contain no absolute dataset paths.\n", encoding="utf-8")
    return {"output": output, "audit": audit, "splits": splits, "windows": accepted}

def doctor(root: str | Path | None = None) -> dict[str, Any]:
    dataset_root = resolve_root(root); records, parse, mapping = parse_records(dataset_root)
    return {"resolved_dataset_root": str(dataset_root), "activity_key_exists": parse["activity_key_exists"], "phone_accelerometer_directory": str(dataset_root / "raw/phone/accel"), "subject_file_count": len(_files(dataset_root)), "subject_ids": sorted({r.subject_id for r in records}), "activity_codes": sorted({r.source_activity_code for r in records}), "mapped_activities": mapping, "malformed_rows": parse["malformed_rows"], "missing_required_activities": sorted(set(CLASS_NAMES) - set(r.activity for r in records)), "usable": bool(records and not (set(CLASS_NAMES) - set(r.activity for r in records)))}

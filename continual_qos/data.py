                                                                

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QoSSpec:
    name: str
    filename: str
    unit: str
    valid_upper_exclusive: float | None
    duplicate_policy: str

    def is_valid(self, value: float) -> bool:
        if not np.isfinite(value) or value <= 0:
            return False
        return self.valid_upper_exclusive is None or value < self.valid_upper_exclusive


QOS_SPECS = {
    "rt": QoSSpec("RT", "rtRate", "seconds", 19.9, "any_valid_record_wins"),
    "tp": QoSSpec("TP", "tpRate", "kbps", None, "last_record_wins"),
}


def build_value_cache(
    raw_path: Path,
    cache_path: Path,
    manifest_path: Path,
    qos_type: str,
    shape: tuple[int, int, int] = (64, 142, 4500),
) -> dict:
                                                                              

                                                                                
                                                                            
       
    if qos_type not in QOS_SPECS:
        raise ValueError(f"unknown qos_type={qos_type!r}")
    spec = QOS_SPECS[qos_type]
    raw_path, cache_path, manifest_path = map(Path, (raw_path, cache_path, manifest_path))
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    num_times, num_users, num_services = shape
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)
    values[:] = np.nan
    seen = np.zeros(shape, dtype=bool)
    digest = hashlib.sha256()
    raw_records = invalid_records = duplicate_records = conflicting_duplicates = 0
    max_user = max_service = max_time = -1

    with raw_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"malformed row {line_number}: expected 4 fields")
            user, service, time_slice = map(int, fields[:3])
            value = float(fields[3])
            if not (0 <= time_slice < num_times and 0 <= user < num_users and 0 <= service < num_services):
                raise ValueError(
                    f"row {line_number} key out of range: user={user}, service={service}, time={time_slice}"
                )
            key = (time_slice, user, service)
            raw_records += 1
            max_user, max_service, max_time = max(max_user, user), max(max_service, service), max(max_time, time_slice)
            valid = spec.is_valid(value)
            if not valid:
                invalid_records += 1
            if seen[key]:
                duplicate_records += 1
                old = float(values[key])
                if np.isfinite(old) and valid and not np.isclose(old, value, rtol=0.0, atol=1e-6):
                    conflicting_duplicates += 1
            seen[key] = True
            if spec.duplicate_policy == "any_valid_record_wins":
                if valid:
                    values[key] = value
            else:
                values[key] = value if valid else np.nan

    values.flush()
    available_entries = int(np.isfinite(values).sum())
    manifest = {
        "schema": "time_user_service_float32_nan_missing_v1",
        "qos_type": qos_type,
        "qos_name": spec.name,
        "unit": spec.unit,
        "raw_path": str(raw_path.resolve()),
        "raw_sha256": digest.hexdigest(),
        "cache_path": str(cache_path.resolve()),
        "shape_time_user_service": list(shape),
        "observed_max_ids": {"user": max_user, "service": max_service, "time": max_time},
        "raw_records": raw_records,
        "unique_raw_keys": int(seen.sum()),
        "available_entries": available_entries,
        "invalid_raw_records": invalid_records,
        "duplicate_raw_records": duplicate_records,
        "conflicting_duplicate_records": conflicting_duplicates,
        "validity_rule": (
            f"0 < {spec.name} < {spec.valid_upper_exclusive}"
            if spec.valid_upper_exclusive is not None
            else f"finite {spec.name} > 0"
        ),
        "duplicate_policy": spec.duplicate_policy,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del values
    return manifest


def load_value_cache(cache_path: Path, expected_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    values = np.load(Path(cache_path), mmap_mode="r")
    if values.ndim != 3:
        raise ValueError(f"expected a 3-D time-user-service cache, got {values.shape}")
    if expected_shape is not None and tuple(values.shape) != tuple(expected_shape):
        raise ValueError(f"cache shape mismatch: {values.shape} != {expected_shape}")
    return values

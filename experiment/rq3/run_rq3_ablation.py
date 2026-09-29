                                                             

                                                                             
                                                                           
                                                                
   

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_PROTOCOL_PATH = HERE / "rq3_ablation_protocol.json"
DEFAULT_VARIANTS_PATH = HERE / "rq3_variants.json"
RUNNER_PATH = HERE / "run_rq3_ablation.py"
DEFAULT_ACCURACY_OUTPUT = REPO_ROOT / "results" / "rq3_ablation_accuracy"
DEFAULT_COMMUNICATION_OUTPUT = REPO_ROOT / "results" / "rq3_ablation_communication"

FROZEN_VARIANT_IDS = ("A0", "A1", "A2", "A3", "A4", "A5")
FROZEN_DENSITIES = (0.01, 0.05, 0.10, 0.15, 0.20)
FROZEN_SEEDS = (42, 3407, 1024, 1, 0)
FROZEN_QOS = ("rt", "tp")
FROZEN_SLICES = 64
FROZEN_PROTOCOL_ID = "fedsaa_rq3_component_ablation_v1"

ACCURACY_QUEUE_FIELDS = (
    "record_type",
    "cell_id",
    "variant_id",
    "paper_name",
    "short_name",
    "implementation_key",
    "qos",
    "density",
    "seed",
    "time_slices",
    "training_required",
    "projection_only",
    "status",
    "error",
)
COMMUNICATION_QUEUE_FIELDS = (
    "record_type",
    "record_id",
    "strategy_id",
    "paper_name",
    "short_name",
    "implementation_key",
    "source_variant_id",
    "source_accuracy_cell_id",
    "qos",
    "density",
    "seed",
    "time_slices",
    "training_required",
    "projection_only",
    "prediction_equivalence_required",
    "status",
    "error",
)
ACCEPTED_CELL_STATUSES = {"COMPLETED", "ACCEPTED", "SKIPPED_COMPLETED"}
TERMINAL_BATCH_STATUS = "BATCH_COMPLETED"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def density_code(density: float) -> str:
    return f"d{int(round(float(density) * 100)):02d}"


def accuracy_cell_id(variant_id: str, qos: str, density: float, seed: int) -> str:
    return f"{variant_id}_{qos}_{density_code(density)}_seed{int(seed)}"


def communication_record_id(strategy_id: str, qos: str, density: float, seed: int) -> str:
    return f"{strategy_id}_{qos}_{density_code(density)}_seed{int(seed)}"


def _variant_map(variants: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for variant in variants:
        variant_id = str(variant.get("variant_id", ""))
        if not variant_id or variant_id in result:
            raise ValueError(f"duplicate or missing variant_id: {variant_id!r}")
        result[variant_id] = variant
    return result


def validate_frozen_spec(protocol: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]) -> None:
                                                                         

    if protocol.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise ValueError(f"unexpected protocol_id: {protocol.get('protocol_id')!r}")
    accuracy = protocol.get("accuracy_matrix")
    communication = protocol.get("communication_matrix")
    if not isinstance(accuracy, Mapping) or not isinstance(communication, Mapping):
        raise ValueError("protocol must contain accuracy_matrix and communication_matrix")
    if tuple(float(v) for v in accuracy.get("densities", ())) != FROZEN_DENSITIES:
        raise ValueError("accuracy density list differs from the frozen five-density matrix")
    if tuple(int(v) for v in accuracy.get("seeds", ())) != FROZEN_SEEDS:
        raise ValueError("accuracy seed list differs from the frozen five-seed matrix")
    if tuple(str(v) for v in accuracy.get("qos_types", ())) != FROZEN_QOS:
        raise ValueError("accuracy QoS list differs from the frozen RT/TP matrix")
    if int(accuracy.get("time_slices", -1)) != FROZEN_SLICES:
        raise ValueError("accuracy time_slices must remain 64")
    if int(accuracy.get("expected_cells", -1)) != 300:
        raise ValueError("accuracy expected_cells must remain 300")
    if int(communication.get("expected_records", -1)) != 150:
        raise ValueError("communication expected_records must remain 150")
    if tuple(str(v.get("variant_id", "")) for v in variants) != FROZEN_VARIANT_IDS:
        raise ValueError("variant IDs/order differ from frozen A0-A5 matrix")
    for variant in variants:
        components = variant.get("components")
        if not isinstance(components, Mapping):
            raise ValueError(f"variant has no components map: {variant.get('variant_id')}")
        if bool(components.get("auxiliary_stabilization")) and not bool(components.get("fedsaa")):
            raise ValueError("auxiliary stabilization cannot be enabled without FedSAA")
    strategies = communication.get("strategies")
    if not isinstance(strategies, Sequence) or tuple(str(s.get("strategy_id", "")) for s in strategies) != (
        "C0",
        "C1",
        "C2",
    ):
        raise ValueError("communication strategies/order must remain C0, C1, C2")
    if communication.get("source_accuracy_variant_id") != "A5":
        raise ValueError("communication projection source must remain A5")


def load_spec(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH, variants_path: Path = DEFAULT_VARIANTS_PATH
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _read_json(Path(protocol_path))
    variants_payload = _read_json(Path(variants_path))
    variants = variants_payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError("variant manifest must contain a list named variants")
    variants_list = [dict(item) for item in variants]
    if variants_payload.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("variant manifest protocol_id does not match protocol")
    validate_frozen_spec(protocol, variants_list)
    return protocol, variants_list


def build_accuracy_cells(
    protocol: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validate_frozen_spec(protocol, variants)
    accuracy = protocol["accuracy_matrix"]
    cells: list[dict[str, Any]] = []
    for variant in variants:
        components = variant["components"]
        for density in accuracy["densities"]:
            for seed in accuracy["seeds"]:
                for qos in accuracy["qos_types"]:
                    cells.append(
                        {
                            "record_type": "accuracy_cell",
                            "cell_id": accuracy_cell_id(
                                str(variant["variant_id"]), str(qos), float(density), int(seed)
                            ),
                            "variant_id": str(variant["variant_id"]),
                            "paper_name": str(variant["paper_name"]),
                            "short_name": str(variant["short_name"]),
                            "implementation_key": str(variant["implementation_key"]),
                            "qos": str(qos),
                            "density": float(density),
                            "seed": int(seed),
                            "time_slices": int(accuracy["time_slices"]),
                            "training_required": True,
                            "projection_only": False,
                            "status": "PLANNED",
                            "error": "",
                            "components": {
                                "rumf": bool(components["rumf"]),
                                "fedsaa": bool(components["fedsaa"]),
                                "auxiliary_stabilization": bool(
                                    components["auxiliary_stabilization"]
                                ),
                            },
                        }
                    )
    return cells


def build_communication_records(
    protocol: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validate_frozen_spec(protocol, variants)
    accuracy = protocol["accuracy_matrix"]
    communication = protocol["communication_matrix"]
    records: list[dict[str, Any]] = []
    for strategy in communication["strategies"]:
        for density in accuracy["densities"]:
            for seed in accuracy["seeds"]:
                for qos in accuracy["qos_types"]:
                    source_cell_id = accuracy_cell_id("A5", str(qos), float(density), int(seed))
                    records.append(
                        {
                            "record_type": "communication_ledger_record",
                            "record_id": communication_record_id(
                                str(strategy["strategy_id"]), str(qos), float(density), int(seed)
                            ),
                            "strategy_id": str(strategy["strategy_id"]),
                            "paper_name": str(strategy["paper_name"]),
                            "short_name": str(strategy["short_name"]),
                            "implementation_key": str(strategy["implementation_key"]),
                            "source_variant_id": "A5",
                            "source_accuracy_cell_id": source_cell_id,
                            "qos": str(qos),
                            "density": float(density),
                            "seed": int(seed),
                            "time_slices": int(accuracy["time_slices"]),
                            "training_required": False,
                            "projection_only": True,
                            "prediction_equivalence_required": True,
                            "status": "PLANNED",
                            "error": "",
                        }
                    )
    return records


def audit_matrix(
    protocol: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    accuracy_cells: Sequence[Mapping[str, Any]] | None = None,
    communication_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
                                                                               

    validate_frozen_spec(protocol, variants)
    expected_accuracy = build_accuracy_cells(protocol, variants)
    expected_communication = build_communication_records(protocol, variants)
    observed_accuracy = list(accuracy_cells) if accuracy_cells is not None else expected_accuracy
    observed_communication = (
        list(communication_records) if communication_records is not None else expected_communication
    )
    expected_accuracy_ids = [str(row["cell_id"]) for row in expected_accuracy]
    expected_communication_ids = [str(row["record_id"]) for row in expected_communication]
    observed_accuracy_ids = [str(row.get("cell_id", "")) for row in observed_accuracy]
    observed_communication_ids = [str(row.get("record_id", "")) for row in observed_communication]
    accuracy_duplicates = sorted(
        {item for item in observed_accuracy_ids if observed_accuracy_ids.count(item) > 1}
    )
    communication_duplicates = sorted(
        {item for item in observed_communication_ids if observed_communication_ids.count(item) > 1}
    )
    expected_accuracy_set = set(expected_accuracy_ids)
    expected_communication_set = set(expected_communication_ids)
    accuracy_missing = sorted(expected_accuracy_set - set(observed_accuracy_ids))
    accuracy_unexpected = sorted(set(observed_accuracy_ids) - expected_accuracy_set)
    communication_missing = sorted(expected_communication_set - set(observed_communication_ids))
    communication_unexpected = sorted(set(observed_communication_ids) - expected_communication_set)
    communication_role_errors = [
        str(row.get("record_id", ""))
        for row in observed_communication
        if row.get("training_required") not in (False, "False", "false", 0, "0")
        or row.get("projection_only") not in (True, "True", "true", 1, "1")
    ]
    return {
        "accuracy_expected": len(expected_accuracy),
        "accuracy_observed": len(observed_accuracy),
        "accuracy_duplicates": accuracy_duplicates,
        "accuracy_missing": accuracy_missing,
        "accuracy_unexpected": accuracy_unexpected,
        "communication_expected": len(expected_communication),
        "communication_observed": len(observed_communication),
        "communication_duplicates": communication_duplicates,
        "communication_missing": communication_missing,
        "communication_unexpected": communication_unexpected,
        "communication_role_errors": communication_role_errors,
        "valid": not (
            accuracy_duplicates
            or accuracy_missing
            or accuracy_unexpected
            or communication_duplicates
            or communication_missing
            or communication_unexpected
            or communication_role_errors
        ),
    }


def _matrix_payload(
    accuracy_cells: Sequence[Mapping[str, Any]], communication_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "accuracy_cells": [dict(row) for row in accuracy_cells],
        "communication_records": [dict(row) for row in communication_records],
    }


def _source_hashes(protocol_path: Path, variants_path: Path) -> dict[str, str]:
    return {
        "protocol": sha256_file(protocol_path),
        "variant_manifest": sha256_file(variants_path),
        "runner": sha256_file(RUNNER_PATH),
    }


def _config_sha256(protocol_path: Path, variants_path: Path) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "protocol": _read_json(protocol_path),
                "variants": _read_json(variants_path),
            }
        )
    )


def build_batch_manifest(
    protocol: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    accuracy_cells: Sequence[Mapping[str, Any]],
    communication_records: Sequence[Mapping[str, Any]],
    *,
    accuracy_output: Path,
    communication_output: Path,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    variants_path: Path = DEFAULT_VARIANTS_PATH,
) -> dict[str, Any]:
    matrix_audit = audit_matrix(protocol, variants, accuracy_cells, communication_records)
    if not matrix_audit["valid"]:
        raise ValueError(f"cannot build manifest from invalid matrix: {matrix_audit}")
    source_hashes = _source_hashes(Path(protocol_path), Path(variants_path))
    accuracy_spec = protocol["accuracy_matrix"]
    communication_spec = protocol["communication_matrix"]
    return {
        "manifest_kind": "rq3_ablation_batch_manifest",
        "status": "PLANNED",
        "protocol": str(protocol["protocol_id"]),
        "protocol_version": str(protocol["protocol_version"]),
        "rq": "RQ2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accuracy_output": str(Path(accuracy_output).resolve()),
        "communication_output": str(Path(communication_output).resolve()),
        "dataset": dict(protocol["dataset"]),
        "accuracy": {
            "variants": [dict(variant) for variant in variants],
            "densities": [float(value) for value in accuracy_spec["densities"]],
            "seeds": [int(value) for value in accuracy_spec["seeds"]],
            "qos_types": [str(value) for value in accuracy_spec["qos_types"]],
            "time_slices": int(accuracy_spec["time_slices"]),
            "expected_cells": len(accuracy_cells),
            "cells": [dict(row) for row in accuracy_cells],
        },
        "communication": {
            "source_accuracy_variant_id": str(communication_spec["source_accuracy_variant_id"]),
            "strategies": [dict(strategy) for strategy in communication_spec["strategies"]],
            "expected_records": len(communication_records),
            "records": [dict(row) for row in communication_records],
            "training_required": False,
            "projection_only": True,
        },
        "matrix_sha256": _sha256_bytes(_canonical_json(_matrix_payload(accuracy_cells, communication_records))),
        "source_sha256": source_hashes,
        "config_sha256": _config_sha256(Path(protocol_path), Path(variants_path)),
        "data_sha256": {
            "rt": None,
            "tp": None,
            "policy": "Executor must bind the frozen cache hashes before formal completion.",
        },
        "required_terminal_artifacts": list(protocol["artifacts"]["accuracy_root_files"]),
        "communication_required_terminal_artifacts": list(
            protocol["artifacts"]["communication_root_files"]
        ),
        "gates": dict(protocol["gates"]),
    }


def _communication_manifest(
    batch_manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]], output: Path
) -> dict[str, Any]:
    return {
        "manifest_kind": "rq3_communication_ledger_manifest",
        "status": "PLANNED",
        "protocol": batch_manifest["protocol"],
        "protocol_version": batch_manifest["protocol_version"],
        "rq": "RQ2",
        "created_at": batch_manifest["created_at"],
        "output": str(Path(output).resolve()),
        "source_accuracy_variant_id": "A5",
        "expected_records": len(records),
        "records": [dict(row) for row in records],
        "training_required": False,
        "projection_only": True,
        "prediction_equivalence_required": True,
        "matrix_sha256": _sha256_bytes(_canonical_json({"records": list(records)})),
        "source_sha256": dict(batch_manifest["source_sha256"]),
        "data_sha256": dict(batch_manifest["data_sha256"]),
        "required_terminal_artifacts": [
            "manifest.json",
            "queue_records.csv",
            "queue_status.json",
            "final_summary.csv",
            "aggregate_mean_std.csv",
            "BATCH_COMPLETED",
        ],
    }


def _accuracy_queue_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PLANNED",
        "protocol": manifest["protocol"],
        "matrix_sha256": manifest["matrix_sha256"],
        "cells_expected": int(manifest["accuracy"]["expected_cells"]),
        "cells_visited": 0,
        "cells_completed": 0,
        "cells_failed": 0,
        "active": None,
        "process_count": 0,
        "required_terminal_artifacts": list(manifest["required_terminal_artifacts"]),
        "created_at": manifest["created_at"],
        "updated_at": manifest["created_at"],
        "note": "Planning scaffold only; no formal training has been launched.",
    }


def _communication_queue_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PLANNED",
        "protocol": manifest["protocol"],
        "matrix_sha256": manifest["matrix_sha256"],
        "records_expected": int(manifest["expected_records"]),
        "records_visited": 0,
        "records_completed": 0,
        "records_failed": 0,
        "active": None,
        "process_count": 0,
        "training_required": False,
        "projection_only": True,
        "required_terminal_artifacts": list(manifest["required_terminal_artifacts"]),
        "created_at": manifest["created_at"],
        "updated_at": manifest["created_at"],
        "note": "Projection ledger only; no training is represented by this queue.",
    }


def _under_experiment_rq3(path: Path) -> bool:
    allowed = (REPO_ROOT / "results").resolve()
    return path.resolve().is_relative_to(allowed)


def _assert_fresh_output(path: Path) -> None:
                                                                            

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty RQ2 output; choose a new root: {path}"
        )


def write_plan(
    accuracy_output: Path = DEFAULT_ACCURACY_OUTPUT,
    communication_output: Path = DEFAULT_COMMUNICATION_OUTPUT,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    variants_path: Path = DEFAULT_VARIANTS_PATH,
    enforce_output_scope: bool = True,
) -> dict[str, Any]:
    protocol, variants = load_spec(protocol_path, variants_path)
    accuracy_cells = build_accuracy_cells(protocol, variants)
    communication_records = build_communication_records(protocol, variants)
    accuracy_output = Path(accuracy_output).resolve()
    communication_output = Path(communication_output).resolve()
    if enforce_output_scope and (
        not _under_experiment_rq3(accuracy_output) or not _under_experiment_rq3(communication_output)
    ):
        raise ValueError("ablation outputs must stay below results")
    _assert_fresh_output(accuracy_output)
    _assert_fresh_output(communication_output)
    manifest = build_batch_manifest(
        protocol,
        variants,
        accuracy_cells,
        communication_records,
        accuracy_output=accuracy_output,
        communication_output=communication_output,
        protocol_path=Path(protocol_path),
        variants_path=Path(variants_path),
    )
    accuracy_output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "logs"):
        (accuracy_output / name).mkdir(exist_ok=True)
    _atomic_json(accuracy_output / "manifest.json", manifest)
    _atomic_csv(accuracy_output / "queue_cells.csv", accuracy_cells, ACCURACY_QUEUE_FIELDS)
    _atomic_json(accuracy_output / "queue_status.json", _accuracy_queue_status(manifest))

    communication_manifest = _communication_manifest(manifest, communication_records, communication_output)
    communication_output.mkdir(parents=True, exist_ok=True)
    (communication_output / "logs").mkdir(exist_ok=True)
    _atomic_json(communication_output / "manifest.json", communication_manifest)
    _atomic_csv(
        communication_output / "queue_records.csv",
        communication_records,
        COMMUNICATION_QUEUE_FIELDS,
    )
    _atomic_json(
        communication_output / "queue_status.json", _communication_queue_status(communication_manifest)
    )
    return {
        "status": "PLANNED",
        "accuracy_output": str(accuracy_output),
        "communication_output": str(communication_output),
        "accuracy_cells": len(accuracy_cells),
        "communication_records": len(communication_records),
        "required_accuracy_terminal_artifacts": manifest["required_terminal_artifacts"],
        "required_communication_terminal_artifacts": communication_manifest["required_terminal_artifacts"],
    }


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _finite_metric_map(summary: Mapping[str, Any]) -> bool:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    candidate = metrics.get("pooled", metrics)
    if not isinstance(candidate, Mapping):
        return False
    return all(name in candidate and math.isfinite(float(candidate[name])) for name in ("mae", "rmse", "mre", "npre"))


def _accuracy_cell_dir(output: Path, row: Mapping[str, Any]) -> Path:
    return (
        output
        / str(row["variant_id"])
        / str(row["qos"])
        / density_code(float(row["density"]))
        / f"seed{int(row['seed'])}"
    )


def _audit_accuracy_cell(output: Path, row: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    cell_dir = _accuracy_cell_dir(output, row)
    summary_path = cell_dir / "summary.json"
    status_path = cell_dir / "status.json"
    result: dict[str, Any] = {
        "cell_id": row.get("cell_id", ""),
        "accepted": False,
        "reason": "missing summary.json",
        "path": str(cell_dir),
    }
    if not summary_path.is_file() or not status_path.is_file():
        return result
    try:
        summary = _read_json(summary_path)
        status = _read_json(status_path)
    except Exception as exc:
        result["reason"] = f"invalid JSON: {type(exc).__name__}: {exc}"
        return result
    if summary.get("status") != "COMPLETED" or status.get("status") != "COMPLETED":
        result["reason"] = "cell status is not COMPLETED"
        return result
    checks = {
        "protocol": summary.get("protocol") == manifest.get("protocol"),
        "variant_id": summary.get("variant_id") == row.get("variant_id"),
        "qos": summary.get("qos") == row.get("qos"),
        "density": float(summary.get("density", -1.0)) == float(row.get("density", -2.0)),
        "seed": int(summary.get("seed", -1)) == int(row.get("seed", -2)),
        "time_slices": int(summary.get("time_slices_completed", -1)) == FROZEN_SLICES,
        "finite_metrics": bool(summary.get("finite_metrics_all_slices")) and _finite_metric_map(summary),
        "query_or_future_used": _bool_value(summary.get("query_or_future_used")) is False,
        "final_state_sha256": bool(summary.get("final_state_sha256")),
    }
    attempt_dir = cell_dir / "attempts"
    attempt_paths = sorted(attempt_dir.glob("attempt_*") if attempt_dir.is_dir() else [])
    per_slice_path = None
    for attempt in reversed(attempt_paths):
        candidate = attempt / "per_slice.csv"
        if candidate.is_file():
            per_slice_path = candidate
            break
    if per_slice_path is None:
        checks["per_slice"] = False
    else:
        try:
            rows = _read_csv(per_slice_path)
            checks["per_slice"] = len(rows) == FROZEN_SLICES and all(
                all(name in item and math.isfinite(float(item[name])) for name in ("mae", "rmse", "mre", "npre"))
                and _bool_value(item.get("query_or_future_used")) is False
                and bool(item.get("mask_sha256"))
                for item in rows
            )
        except Exception:
            checks["per_slice"] = False
    expected_source = manifest.get("source_sha256")
    summary_source = summary.get("source_sha256")
    checks["source_sha256"] = isinstance(summary_source, Mapping) and all(
        summary_source.get(name) == value for name, value in dict(expected_source or {}).items()
    )
    checks["config_sha256"] = summary.get("config_sha256") == manifest.get("config_sha256")
    data_sha = summary.get("data_sha256")
    checks["data_sha256"] = isinstance(data_sha, Mapping) and all(bool(data_sha.get(qos)) for qos in FROZEN_QOS)
    result["checks"] = checks
    result["accepted"] = all(checks.values())
    result["reason"] = "strict complete" if result["accepted"] else "; ".join(
        name for name, passed in checks.items() if not passed
    )
    return result


def _audit_communication(output: Path, expected_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queue_path = output / "queue_records.csv"
    status_path = output / "queue_status.json"
    manifest_path = output / "manifest.json"
    if not (queue_path.is_file() and status_path.is_file() and manifest_path.is_file()):
        return {"status": "MISSING", "expected_records": len(expected_records)}
    rows = _read_csv(queue_path)
    expected_ids = {str(row["record_id"]) for row in expected_records}
    observed_ids = [str(row.get("record_id", "")) for row in rows]
    duplicates = sorted({item for item in observed_ids if observed_ids.count(item) > 1})
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    role_errors = sorted(
        str(row.get("record_id", ""))
        for row in rows
        if row.get("training_required") != "False" or row.get("projection_only") != "True"
    )
    status = _read_json(status_path)
    terminal_files = [
        output / "final_summary.csv",
        output / "aggregate_mean_std.csv",
        output / "BATCH_COMPLETED",
    ]
    complete = (
        len(rows) == len(expected_records)
        and not duplicates
        and not missing
        and not unexpected
        and not role_errors
        and status.get("status") == TERMINAL_BATCH_STATUS
        and all(path.is_file() and path.stat().st_size > 0 for path in terminal_files)
        and status.get("active") is None
        and int(status.get("process_count", -1)) == 0
    )
    return {
        "status": "COMPLETE" if complete else ("PLANNED" if status.get("status") == "PLANNED" else "INCOMPLETE"),
        "expected_records": len(expected_records),
        "observed_records": len(rows),
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
        "role_errors": role_errors,
        "queue_status": status.get("status"),
        "terminal_artifacts_present": all(path.is_file() and path.stat().st_size > 0 for path in terminal_files),
    }


def audit_batch(
    accuracy_output: Path,
    *,
    communication_output: Path | None = None,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    variants_path: Path = DEFAULT_VARIANTS_PATH,
) -> dict[str, Any]:
                                                                              

    protocol, variants = load_spec(protocol_path, variants_path)
    accuracy_output = Path(accuracy_output).resolve()
    communication_output = (
        Path(communication_output).resolve()
        if communication_output is not None
        else DEFAULT_COMMUNICATION_OUTPUT.resolve()
    )
    manifest_path = accuracy_output / "manifest.json"
    queue_path = accuracy_output / "queue_cells.csv"
    status_path = accuracy_output / "queue_status.json"
    if not (manifest_path.is_file() and queue_path.is_file() and status_path.is_file()):
        return {"status": "MISSING", "accuracy": {"expected_cells": 300}, "communication": {}}
    manifest = _read_json(manifest_path)
    expected_accuracy = build_accuracy_cells(protocol, variants)
    expected_communication = build_communication_records(protocol, variants)
    rows = _read_csv(queue_path)
    matrix_audit = audit_matrix(protocol, variants, rows, expected_communication)
    expected_by_id = {str(row["cell_id"]): row for row in expected_accuracy}
    observed_ids = [str(row.get("cell_id", "")) for row in rows]
    duplicate_ids = sorted({item for item in observed_ids if observed_ids.count(item) > 1})
    missing_ids = sorted(set(expected_by_id) - set(observed_ids))
    unexpected_ids = sorted(set(observed_ids) - set(expected_by_id))
    complete_rows = [row for row in rows if row.get("status") in ACCEPTED_CELL_STATUSES]
    cell_audits: list[dict[str, Any]] = []
    if complete_rows:
        for row in complete_rows:
            expected = expected_by_id.get(str(row.get("cell_id", "")))
            if expected is not None:
                cell_audits.append(_audit_accuracy_cell(accuracy_output, expected, manifest))
    terminal_files = [
        accuracy_output / "final_summary.csv",
        accuracy_output / "aggregate_mean_std.csv",
        accuracy_output / "BATCH_COMPLETED",
    ]
    queue_status = _read_json(status_path)
    all_queue_rows_accepted = (
        len(rows) == len(expected_accuracy)
        and not duplicate_ids
        and not missing_ids
        and not unexpected_ids
        and all(row.get("status") in ACCEPTED_CELL_STATUSES for row in rows)
    )
    all_cell_gates = bool(cell_audits) and len(cell_audits) == len(expected_accuracy) and all(
        bool(item.get("accepted")) for item in cell_audits
    )
    terminal_files_present = all(path.is_file() and path.stat().st_size > 0 for path in terminal_files)
    accuracy_complete = (
        all_queue_rows_accepted
        and all_cell_gates
        and queue_status.get("status") == TERMINAL_BATCH_STATUS
        and queue_status.get("active") is None
        and int(queue_status.get("process_count", -1)) == 0
        and terminal_files_present
    )
    if accuracy_complete:
        accuracy_status = "COMPLETE"
    elif queue_status.get("status") == "PLANNED" and not complete_rows:
        accuracy_status = "PLANNED"
    else:
        accuracy_status = "INCOMPLETE"
    communication_audit = _audit_communication(communication_output, expected_communication)
    return {
        "status": "COMPLETE" if accuracy_complete and communication_audit.get("status") == "COMPLETE" else accuracy_status,
        "accuracy": {
            "status": accuracy_status,
            "expected_cells": len(expected_accuracy),
            "queued_cells": len(rows),
            "completed_queue_rows": len(complete_rows),
            "duplicate_ids": duplicate_ids,
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "matrix_valid": matrix_audit["valid"],
            "all_queue_rows_accepted": all_queue_rows_accepted,
            "all_cell_gates": all_cell_gates,
            "terminal_artifacts_present": terminal_files_present,
            "cell_audits": cell_audits,
        },
        "communication": communication_audit,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACCURACY_OUTPUT)
    parser.add_argument("--communication-output", type=Path, default=DEFAULT_COMMUNICATION_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="print frozen matrix counts without writing")
    parser.add_argument("--write-plan", action="store_true", help="write planned queues; never launches training")
    parser.add_argument("--audit", type=Path, help="audit an existing accuracy output root")
    args = parser.parse_args(argv)
    if args.audit is not None:
        _print_json(
            audit_batch(
                args.audit,
                communication_output=args.communication_output,
                protocol_path=args.protocol,
                variants_path=args.variants,
            )
        )
        return 0
    protocol, variants = load_spec(args.protocol, args.variants)
    accuracy_cells = build_accuracy_cells(protocol, variants)
    communication_records = build_communication_records(protocol, variants)
    if args.write_plan:
        _print_json(
            write_plan(
                args.output,
                args.communication_output,
                protocol_path=args.protocol,
                variants_path=args.variants,
            )
        )
        return 0
    _print_json(
        {
            "status": "DRY_RUN",
            "protocol": protocol["protocol_id"],
            "accuracy_cells": len(accuracy_cells),
            "communication_records": len(communication_records),
            "training_launched": False,
            "accuracy_output": str(Path(args.output).resolve()),
            "communication_output": str(Path(args.communication_output).resolve()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

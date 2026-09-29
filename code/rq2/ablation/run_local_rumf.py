                                                      

                                                                        
                                                                         
                                                                        
                                                                       
                                                                         
                                                                          
                                                   

                                                                            
                                                                              
                                                                              
                                                                            
                                      
   

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

                                                                             
                                                                          
                                          
if importlib.util.find_spec("torch") is None and "continual_qos" not in sys.modules:
    package = type(sys)("continual_qos")
    package.__path__ = [str(ROOT / "continual_qos")]                              
    sys.modules["continual_qos"] = package

from continual_qos.data import load_value_cache              
from continual_qos.protocol import make_causal_split              
from local_rumf_model import (              
    LocalRUMFModel,
    cuda_available,
    cuda_device_info,
    prediction_metrics,
    state_hash,
)


PROTOCOL = "fedsaa_rq2_local_rumf_v1"
EXECUTION_MODE = "rq2_local_rumf_gpu_v1"
SHAPE = (64, 142, 4500)
FORMAL_DENSITIES = (0.01, 0.05, 0.10, 0.15, 0.20)
FORMAL_SEEDS = (42, 3407, 1024, 1, 0)
FORMAL_QOS = ("rt", "tp")
FORMAL_SLICES = 64
SMOKE_DENSITIES = (0.10,)
SMOKE_SEEDS = (42,)
SMOKE_QOS = ("rt",)
SMOKE_SLICES = 1
DEFAULT_CONFIG = ROOT / "configs" / "base_config.json"
DEFAULT_FORMAL_OUTPUT = ROOT / "results" / "rq3_local_rumf"
DEFAULT_SMOKE_OUTPUT = (
    ROOT / "results" / "rq3_local_rumf_smoke"
)

VARIANT = {
    "variant_id": "LOCAL_RUMF",
    "variant": "local_rumf",
    "variant_bits": "local",
    "paper_label": "Local RUMF",
    "aggregation": "local",
    "server_aggregation": False,
    "private_residual": True,
    "shared_broadcast": False,
    "communication": "zero_server_exchange",
}

COMMUNICATION_FIELDS = (
    "proposal_count",
    "touched_request_rows",
    "touched_service_rows",
    "edge_summary_count",
    "client_edge_uplink_bytes",
    "client_edge_request_bytes",
    "client_edge_total_uplink_bytes",
    "edge_cloud_uplink_bytes",
    "cloud_edge_bootstrap_rows",
    "cloud_edge_bootstrap_bytes",
    "cloud_edge_delta_rows",
    "cloud_edge_delta_bytes",
    "edge_client_touched_rows",
    "edge_client_touched_row_downlink_bytes",
    "edge_cache_total_bytes",
    "edge_cache_steady_state_total_bytes",
)
METRICS = ("mae", "rmse", "mre", "npre")
PROFILE_FIELDS = (
    "input_upload_seconds",
    "cpsa_seconds",
    "shared_broadcast_seconds",
    "local_update_seconds",
    "carw_srr_seconds",
    "proposal_quantization_seconds",
    "aggregation_seconds",
    "aggregate_quantization_seconds",
    "total_seconds",
    "peak_device_memory_used_bytes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(str(key))
    else:
        names = [str(value) for value in fieldnames]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _cell_id(qos: str, density: float, seed: int) -> str:
    return f"LOCAL_RUMF_{qos}_d{int(round(float(density) * 100)):02d}_seed{int(seed)}"


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    density = int(round(float(cell["density"]) * 100))
    return (
        Path(output)
        / "LOCAL_RUMF"
        / str(cell["qos"])
        / f"d{density:02d}"
        / f"seed{int(cell['seed'])}"
    )


def build_cells(
    densities: Sequence[float] = FORMAL_DENSITIES,
    seeds: Sequence[int] = FORMAL_SEEDS,
    qos_types: Sequence[str] = FORMAL_QOS,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for density in densities:
        density = float(density)
        if not 0.0 < density < 1.0:
            raise ValueError("density must lie strictly between zero and one")
        for seed in seeds:
            for qos in qos_types:
                qos = str(qos).lower()
                if qos not in FORMAL_QOS:
                    raise ValueError(f"unknown QoS type {qos!r}")
                cells.append(
                    {
                        "cell_id": _cell_id(qos, density, int(seed)),
                        "variant_id": VARIANT["variant_id"],
                        "variant": VARIANT["variant"],
                        "variant_bits": VARIANT["variant_bits"],
                        "paper_label": VARIANT["paper_label"],
                        "qos": qos,
                        "density": density,
                        "seed": int(seed),
                        "aggregation": VARIANT["aggregation"],
                        "server_aggregation": False,
                        "private_residual": True,
                    }
                )
    return cells


def _scope(
    smoke: bool,
    densities: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    qos_types: Sequence[str] | None = None,
    slices: int | None = None,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[str, ...], int]:
    expected = (
        SMOKE_DENSITIES,
        SMOKE_SEEDS,
        SMOKE_QOS,
        SMOKE_SLICES,
    ) if smoke else (
        FORMAL_DENSITIES,
        FORMAL_SEEDS,
        FORMAL_QOS,
        FORMAL_SLICES,
    )
    supplied = (
        None if densities is None else tuple(float(value) for value in densities),
        None if seeds is None else tuple(int(value) for value in seeds),
        None if qos_types is None else tuple(str(value).lower() for value in qos_types),
        None if slices is None else int(slices),
    )
    for got, frozen, name in zip(
        supplied, expected, ("densities", "seeds", "qos", "slices")
    ):
        if got is not None and got != tuple(frozen) if name != "slices" else got is not None and got != frozen:
            raise ValueError(
                f"Local RUMF {'smoke' if smoke else 'formal'} {name} are frozen; "
                f"requested {got!r}, expected {frozen!r}"
            )
    return tuple(expected[0]), tuple(expected[1]), tuple(expected[2]), int(expected[3])


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if tuple(config.get("shape_time_user_service", ())) != SHAPE:
        raise ValueError("Local RUMF requires WSDream Dataset 3 shape 64x142x4500")
    if config.get("raw_prediction_caps") != {"rt": 20.0, "tp": 10000.0}:
        raise ValueError("raw QoS caps differ from the frozen v1 protocol")
    required = (
        "rank", "bias_eta", "factor_eta", "l2", "beta", "factor_init_std",
        "num_edges", "bias_epochs", "factor_epochs", "cache_dir",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing frozen config fields: {missing}")
    return config


def _source_hashes(config_path: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "local_model": HERE / "local_rumf_model.py",
        "protocol": ROOT / "continual_qos" / "protocol.py",
        "config": Path(config_path).resolve(),
    }
    return {name: _sha256(path) for name, path in files.items()}


def _new_model(config: Mapping[str, Any], seed: int) -> LocalRUMFModel:
    return LocalRUMFModel(
        num_users=SHAPE[1],
        num_services=SHAPE[2],
        rank=int(config["rank"]),
        seed=int(seed),
        bias_eta=float(config["bias_eta"]),
        factor_eta=float(config["factor_eta"]),
        l2=float(config["l2"]),
        beta=float(config["beta"]),
        factor_init_std=float(config["factor_init_std"]),
        num_edges=int(config["num_edges"]),
    )


def _profile_dict(profile: Any) -> dict[str, Any]:
    if profile is None:
        raise AssertionError("Local RUMF training profile missing")
    if hasattr(profile, "as_dict") and callable(profile.as_dict):
        raw = dict(profile.as_dict())
    elif isinstance(profile, Mapping):
        raw = dict(profile)
    else:
        raw = {
            name: getattr(profile, name)
            for name in PROFILE_FIELDS
            if hasattr(profile, name)
        }
    return raw


def _zero_communication() -> dict[str, Any]:
    return {name: 0 for name in COMMUNICATION_FIELDS} | {
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "mode": "local_only",
    }


def _completed(
    cell_dir: Path,
    cell: Mapping[str, Any],
    slices: int,
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: str,
) -> bool:
    summary = _read_json(Path(cell_dir) / "summary.json")
    if summary is None or summary.get("status") != "COMPLETED":
        return False
    for key in ("cell_id", "variant_id", "variant", "variant_bits", "qos", "seed"):
        if str(summary.get(key)) != str(cell.get(key)):
            return False
    if not np.isclose(float(summary.get("density", -1.0)), float(cell["density"])):
        return False
    if int(summary.get("time_slices_completed", -1)) != int(slices):
        return False
    if summary.get("source_sha256") != dict(source_sha256):
        return False
    if str(summary.get("config_sha256")) != str(config_sha256):
        return False
    if str(summary.get("data_sha256")) != str(data_sha256):
        return False
    if summary.get("query_or_future_used") is not False:
        return False
    communication = summary.get("communication_totals", {})
    return all(int(communication.get(name, 0)) == 0 for name in COMMUNICATION_FIELDS)


def _next_attempt(cell_dir: Path) -> Path:
    attempts = cell_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in attempts.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = attempts / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _summarize_metrics(
    rows: Sequence[Mapping[str, Any]], relative_chunks: Sequence[np.ndarray]
) -> dict[str, Any]:
    if not rows or not relative_chunks:
        raise ValueError("cannot summarize an empty cell")
    count = sum(int(row["query_entries"]) for row in rows)
    absolute = sum(float(row["absolute_error_sum"]) for row in rows)
    squared = sum(float(row["squared_error_sum"]) for row in rows)
    pooled = np.concatenate([np.asarray(chunk, dtype=np.float64) for chunk in relative_chunks])
    if count <= 0 or pooled.size != count or not np.all(np.isfinite(pooled)):
        raise FloatingPointError("invalid metric accumulation")
    pooled.sort()
    return {
        "pooled": {
            "mae": float(absolute / count),
            "rmse": float(np.sqrt(squared / count)),
            "mre": float(np.median(pooled)),
            "npre": float(pooled[min(int(0.9 * count), count - 1)]),
            "count": int(count),
        },
        "time_mean": {
            name: float(np.mean([float(row[name]) for row in rows], dtype=np.float64))
            for name in METRICS
        },
        "definitions": {
            "mre_per_slice": "median(sorted(abs(pred-truth)/truth))",
            "npre_per_slice": "sorted_relative[min(int(0.9*n), n-1)]",
            "pooled": "computed over all query observations across slices",
        },
    }


def run_cell(
    *,
    cell: Mapping[str, Any],
    values: np.ndarray,
    config: Mapping[str, Any],
    output: Path,
    slices: int,
    resume: bool,
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: str,
) -> dict[str, Any]:
    cell_dir = _cell_dir(Path(output), cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if _completed(cell_dir, cell, slices, source_sha256, config_sha256, data_sha256):
        if resume:
            summary = _read_json(cell_dir / "summary.json") or {}
            return {"status": "SKIPPED_COMPLETED", "summary": summary}
        raise FileExistsError(f"completed cell already exists; use --resume: {cell_dir}")
    attempts_dir = cell_dir / "attempts"
    if attempts_dir.exists() and any(attempts_dir.glob("attempt_*")) and not resume:
        raise FileExistsError(f"incomplete prior attempt exists; use --resume: {cell_dir}")

    attempt = _next_attempt(cell_dir)
    status: dict[str, Any] = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "execution_mode": EXECUTION_MODE,
        **dict(cell),
        "time_slices_expected": int(slices),
        "time_slices_completed": 0,
        "started_at": _utc_now(),
        "attempt_dir": str(attempt),
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": str(data_sha256),
        "query_or_future_used": False,
        "query_or_future_used_for_update": False,
        "local_state_only": True,
        "server_aggregation": False,
        "communication_expected_zero": True,
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)
    (attempt / "run.log").write_text(
        f"{_utc_now()} START {cell['cell_id']} slices={slices}\n", encoding="utf-8"
    )

    process = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
    except Exception:                                          
        process = None
    peak_rss = int(process.memory_info().rss) if process is not None else 0
    start_cpu = process.cpu_times() if process is not None else None
    rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    relative_chunks: list[np.ndarray] = []
    mask_hashes: list[str] = []
    started_cell = time.perf_counter()
    model: LocalRUMFModel | None = None
    try:
        model = _new_model(config, int(cell["seed"]))
        raw_cap = float(config["raw_prediction_caps"][str(cell["qos"])])
        for time_slice in range(int(slices)):
            slice_started = time.perf_counter()
            split_started = time.perf_counter()
            split = make_causal_split(
                values[time_slice],
                float(cell["density"]),
                int(cell["seed"]),
                int(time_slice),
            )
            split_seconds = time.perf_counter() - split_started
            if split.support_count <= 0 or split.query_count <= 0:
                raise ValueError("causal split must have non-empty support and query sets")
            support_keys = split.support_users.astype(np.int64) * SHAPE[2] + split.support_services
            query_keys = split.query_users.astype(np.int64) * SHAPE[2] + split.query_services
            if np.intersect1d(support_keys, query_keys).size:
                raise AssertionError("support/query masks overlap")
            if not np.all(np.isfinite(split.support_values)) or not np.all(np.isfinite(split.query_values)):
                raise FloatingPointError("causal split contains non-finite values")
            if np.any(split.support_values <= 0) or np.any(split.query_values <= 0):
                raise ValueError("causal split contains non-positive QoS values")

            update_started = time.perf_counter()
            result = model.advance_support(
                split.support_users,
                split.support_services,
                split.support_values,
                raw_cap,
                int(config["bias_epochs"]),
                int(config["factor_epochs"]),
                int(cell["seed"]) * 1_000_003 + int(time_slice),
            )
            update_wall = time.perf_counter() - update_started

            predict_started = time.perf_counter()
            prediction, predict_timing = model.predict(
                split.query_users,
                split.query_services,
                raw_cap,
            )
            predict_wall = time.perf_counter() - predict_started
            metric_started = time.perf_counter()
            metrics = prediction_metrics(split.query_values, prediction)
            truth = np.asarray(split.query_values, dtype=np.float64)
            estimate = np.asarray(prediction, dtype=np.float64)
            relative = np.abs(estimate - truth) / truth
            if not np.all(np.isfinite(relative)):
                raise FloatingPointError("non-finite relative error")
            relative_chunks.append(relative)
            metric_seconds = time.perf_counter() - metric_started

            communication = _zero_communication()
            communication_from_model = result.communication
            if int(getattr(communication_from_model, "uplink_bytes", 0)) != 0 or int(
                getattr(communication_from_model, "downlink_bytes", 0)
            ) != 0:
                raise AssertionError("Local RUMF model reported non-zero wire bytes")
            profile = _profile_dict(model.last_profile)
            if not all(np.isfinite(float(metrics[name])) for name in METRICS):
                raise FloatingPointError("non-finite Local RUMF metric")
            mask_hashes.append(str(split.mask_sha256))
            if process is not None:
                peak_rss = max(peak_rss, int(process.memory_info().rss))
            row: dict[str, Any] = {
                "protocol": PROTOCOL,
                "execution_mode": EXECUTION_MODE,
                "cell_id": str(cell["cell_id"]),
                "variant_id": str(cell["variant_id"]),
                "variant": str(cell["variant"]),
                "variant_bits": str(cell["variant_bits"]),
                "paper_label": str(cell["paper_label"]),
                "qos": str(cell["qos"]),
                "density": float(cell["density"]),
                "seed": int(cell["seed"]),
                "time_slice": int(time_slice),
                "support_entries": int(split.support_count),
                "query_entries": int(split.query_count),
                "mask_sha256": str(split.mask_sha256),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mre": float(metrics["mre"]),
                "npre": float(metrics["npre"]),
                "absolute_error_sum": float(metrics["absolute_error_sum"]),
                "squared_error_sum": float(metrics["squared_error_sum"]),
                "split_seconds": float(split_seconds),
                "update_wall_seconds": float(update_wall),
                "predict_wall_seconds": float(predict_wall),
                "predict_kernel_seconds": float(predict_timing.get("kernel_seconds", 0.0)),
                "metric_seconds": float(metric_seconds),
                "communication_accounting_seconds": 0.0,
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
                "query_or_future_used": False,
                "query_or_future_used_for_update": False,
                "local_state_only": True,
                "gpu_profile_json": json.dumps(profile, sort_keys=True),
                "communication_json": json.dumps(communication, sort_keys=True),
            }
            for name in COMMUNICATION_FIELDS:
                row[name] = 0
            for name in PROFILE_FIELDS:
                if name in profile:
                    row[f"gpu_{name}"] = float(profile[name])
            rows.append(row)
            state_rows.append(
                {
                    "protocol": PROTOCOL,
                    "cell_id": str(cell["cell_id"]),
                    "qos": str(cell["qos"]),
                    "density": float(cell["density"]),
                    "seed": int(cell["seed"]),
                    "time_slice": int(time_slice),
                    "active_clients": int(len(result.active_clients)),
                    "support_entries": int(split.support_count),
                    "local_state_bytes": int(model.local_state_bytes),
                    "shared_broadcast": False,
                    "server_aggregation": False,
                    "uplink_bytes": 0,
                    "downlink_bytes": 0,
                    "query_or_future_used": False,
                    "state_hash_at_slice": "",
                }
            )
            _atomic_csv(attempt / "per_slice.csv", rows)
            _atomic_csv(attempt / "state_audit.csv", state_rows)
            status["time_slices_completed"] = int(time_slice + 1)
            status["last_mask_sha256"] = str(split.mask_sha256)
            status["updated_at"] = _utc_now()
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)
            with (attempt / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{_utc_now()} SLICE {time_slice} support={split.support_count} "
                    f"query={split.query_count} uplink=0 downlink=0\n"
                )

        digest = hashlib.sha256()
        for value in mask_hashes:
            digest.update(bytes.fromhex(value))
        metrics_summary = _summarize_metrics(rows, relative_chunks)
        communication_totals = {
            name: int(sum(int(row[name]) for row in rows))
            for name in COMMUNICATION_FIELDS
        }
        end_cpu = process.cpu_times() if process is not None else None
        assert model is not None
        summary: dict[str, Any] = {
            "status": "COMPLETED",
            "protocol": PROTOCOL,
            "execution_mode": EXECUTION_MODE,
            **dict(cell),
            "time_slices_completed": len(rows),
            "metrics": metrics_summary,
            "finite_metrics_all_slices": True,
            "query_or_future_used": False,
            "query_or_future_used_for_update": False,
            "local_state_only": True,
            "server_aggregation": False,
            "shared_broadcast": False,
            "communication_expected_zero": True,
            "source_sha256": dict(source_sha256),
            "config_sha256": str(config_sha256),
            "data_sha256": str(data_sha256),
            "mask_audit": {
                "count": len(mask_hashes),
                "unique_count": len(set(mask_hashes)),
                "mask_sha256_by_slice": {str(i): value for i, value in enumerate(mask_hashes)},
                "mask_stream_sha256": digest.hexdigest(),
                "support_query_partition_checked": True,
            },
            "communication_totals": communication_totals,
            "communication_audit": {
                "uplink_bytes": 0,
                "downlink_bytes": 0,
                "all_fields_zero": True,
                "model_reported_zero": True,
            },
            "timing_seconds": {
                name: float(sum(float(row.get(name, 0.0)) for row in rows))
                for name in (
                    "split_seconds", "update_wall_seconds", "predict_wall_seconds",
                    "metric_seconds", "communication_accounting_seconds", "slice_wall_seconds",
                )
            },
            "cell_wall_seconds": float(time.perf_counter() - started_cell),
            "gpu_peak_memory_used_bytes": int(
                max(
                    (int(row.get("gpu_peak_device_memory_used_bytes", 0)) for row in rows),
                    default=0,
                )
            ),
            "local_state_bytes": int(model.local_state_bytes),
            "process_peak_rss_bytes": int(peak_rss),
            "process_cpu_user_seconds": (
                float(end_cpu.user - start_cpu.user)
                if end_cpu is not None and start_cpu is not None
                else 0.0
            ),
            "process_cpu_system_seconds": (
                float(end_cpu.system - start_cpu.system)
                if end_cpu is not None and start_cpu is not None
                else 0.0
            ),
            "final_state_sha256": state_hash(model),
            "attempt_dir": str(attempt),
            "completed_at": _utc_now(),
        }
                                                                          
        _atomic_json(attempt / "summary.json", summary)
        _atomic_json(attempt / "status.json", summary)
        _atomic_json(cell_dir / "summary.json", summary)
        _atomic_json(cell_dir / "status.json", summary)
        (cell_dir / "run.log").write_text(
            (attempt / "run.log").read_text(encoding="utf-8")
            + f"{_utc_now()} COMPLETED\n",
            encoding="utf-8",
        )
        return {"status": "COMPLETED", "summary": summary}
    except Exception as exc:
        failed = {
            **status,
            "status": "FAILED",
            "time_slices_completed": len(rows),
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(attempt / "status.json", failed)
        _atomic_json(cell_dir / "status.json", failed)
        with (attempt / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} FAILED {type(exc).__name__}: {exc}\n")
        raise


def _summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    pooled = summary["metrics"]["pooled"]
    time_mean = summary["metrics"]["time_mean"]
    row: dict[str, Any] = {
        "protocol": str(summary["protocol"]),
        "execution_mode": str(summary["execution_mode"]),
        "cell_id": str(summary["cell_id"]),
        "variant_id": str(summary["variant_id"]),
        "variant": str(summary["variant"]),
        "paper_label": str(summary["paper_label"]),
        "qos": str(summary["qos"]),
        "density": float(summary["density"]),
        "seed": int(summary["seed"]),
        "time_slices": int(summary["time_slices_completed"]),
        "local_state_bytes": int(summary.get("local_state_bytes", 0)),
        "cell_wall_seconds": float(summary.get("cell_wall_seconds", 0.0)),
        "final_state_sha256": str(summary.get("final_state_sha256", "")),
    }
    for metric in METRICS:
        row[f"pooled_{metric}"] = float(pooled[metric])
        row[f"time_mean_{metric}"] = float(time_mean[metric])
    for name, value in summary.get("communication_totals", {}).items():
        row[f"communication_{name}"] = int(value)
    for name, value in summary.get("timing_seconds", {}).items():
        row[f"timing_{name}"] = float(value)
    row["uplink_bytes"] = 0
    row["downlink_bytes"] = 0
    row["query_or_future_used"] = False
    return row


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["qos"]), float(row["density"]))].append(row)
    output: list[dict[str, Any]] = []
    for (qos, density), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "protocol": PROTOCOL,
            "variant_id": VARIANT["variant_id"],
            "variant": VARIANT["variant"],
            "qos": qos,
            "density": density,
            "seed_count": len(group),
            "communication_all_zero": True,
        }
        for metric in METRICS:
            for prefix in ("pooled", "time_mean"):
                values = np.asarray(
                    [float(row[f"{prefix}_{metric}"]) for row in group], dtype=np.float64
                )
                aggregate[f"{prefix}_{metric}_mean"] = float(np.mean(values))
                aggregate[f"{prefix}_{metric}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
        for name in ("cell_wall_seconds", "local_state_bytes"):
            values = np.asarray([float(row[name]) for row in group], dtype=np.float64)
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        output.append(aggregate)
    return output


def _validate_existing_manifest(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    immutable = (
        "protocol", "execution_mode", "scope", "shape", "densities", "seeds",
        "qos_types", "time_slices", "cell_count", "variant_id", "variant",
    )
    for key in immutable:
        if existing.get(key) != current.get(key):
            raise ValueError(f"existing Local RUMF manifest mismatch at {key}")
    if existing.get("source_sha256") != current.get("source_sha256"):
        raise ValueError("existing Local RUMF source hash mismatch; start a new output")
    if existing.get("config_sha256") != current.get("config_sha256"):
        raise ValueError("existing Local RUMF config hash mismatch; do not resume")


def run_queue(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path | None = None,
    smoke: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
) -> dict[str, Any]:
                                                        

    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    densities, seeds, qos_types, slices = _scope(smoke)
    output = Path(output or (DEFAULT_SMOKE_OUTPUT if smoke else DEFAULT_FORMAL_OUTPUT)).resolve()
    cells = build_cells(densities, seeds, qos_types)
    source_sha256 = _source_hashes(config_path)
    config_sha256 = source_sha256["config"]
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "execution_mode": EXECUTION_MODE,
        "created_at": _utc_now(),
        "workspace": str(ROOT),
        "output": str(output),
        "scope": "smoke" if smoke else "formal",
        "shape": list(SHAPE),
        "densities": [float(value) for value in densities],
        "seeds": [int(value) for value in seeds],
        "qos_types": [str(value) for value in qos_types],
        "time_slices": int(slices),
        "variant_id": VARIANT["variant_id"],
        "variant": dict(VARIANT),
        "cell_count": len(cells),
        "cells": cells,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "data_sha256": {},
        "query_or_future_used": False,
        "local_control_contract": {
            "prediction_model": "RUMF with private residual",
            "state_scope": "per_client",
            "server_aggregation": False,
            "shared_broadcast": False,
            "server_upload": False,
            "server_downlink": False,
            "expected_uplink_bytes": 0,
            "expected_downlink_bytes": 0,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
    }
    if dry_run:
        return {"status": "DRY_RUN", "manifest": manifest, "cells": cells}
    if not cuda_available():
        raise RuntimeError("a usable Numba CUDA device is required for Local RUMF")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = _read_json(manifest_path)
    if existing is not None:
        if not resume:
            raise FileExistsError(f"Local RUMF output already initialized; use --resume: {output}")
        _validate_existing_manifest(existing, manifest)
        manifest["created_at"] = existing.get("created_at", manifest["created_at"])
        manifest["data_sha256"] = dict(existing.get("data_sha256", {}))
    elif any(output.iterdir()) and not resume:
        raise FileExistsError(f"output directory is non-empty without a manifest: {output}")

    cache: dict[str, np.ndarray] = {}
    data_sha256: dict[str, str] = {}
    for qos in qos_types:
        data_path = ROOT / str(config["cache_dir"]) / f"{qos}_values.npy"
        cache[str(qos)] = load_value_cache(data_path, SHAPE)
        data_sha256[str(qos)] = _sha256(data_path)
    if existing is not None and existing.get("data_sha256"):
        if dict(existing["data_sha256"]) != data_sha256:
            raise ValueError("existing Local RUMF data hash mismatch; do not resume")
    manifest["data_sha256"] = data_sha256
    manifest["backend"] = cuda_device_info()
    _atomic_json(manifest_path, manifest)

    queue_rows: list[dict[str, Any]] = [
        {"cell_index": index, **dict(cell), "status": "PENDING", "error": ""}
        for index, cell in enumerate(cells)
    ]
    queue_started = time.perf_counter()
    completed_now = skipped = failed = 0
    log_path = output / "run.log"

    def write_queue() -> None:
        _atomic_csv(output / "queue_cells.csv", queue_rows)

    write_queue()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{_utc_now()} START scope={'smoke' if smoke else 'formal'} cells={len(cells)}\n")
    for index, cell in enumerate(cells):
        try:
            result = run_cell(
                cell=cell,
                values=cache[str(cell["qos"])],
                config=config,
                output=output,
                slices=slices,
                resume=resume,
                source_sha256=source_sha256,
                config_sha256=config_sha256,
                data_sha256=data_sha256[str(cell["qos"])],
            )
            queue_rows[index]["status"] = str(result["status"])
            queue_rows[index]["attempt_dir"] = str(result.get("summary", {}).get("attempt_dir", ""))
            if result["status"] == "SKIPPED_COMPLETED":
                skipped += 1
            else:
                completed_now += 1
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{_utc_now()} {result['status']} {cell['cell_id']}\n")
        except Exception as exc:
            failed += 1
            queue_rows[index]["status"] = "FAILED"
            queue_rows[index]["error"] = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{_utc_now()} FAILED {cell['cell_id']} {type(exc).__name__}: {exc}\n")
            if stop_on_error:
                write_queue()
                raise
        write_queue()
        _atomic_json(
            output / "queue_status.json",
            {
                "status": "RUNNING",
                "protocol": PROTOCOL,
                "scope": "smoke" if smoke else "formal",
                "cells_expected": len(cells),
                "cells_visited": index + 1,
                "cells_completed_this_invocation": completed_now,
                "cells_skipped_completed": skipped,
                "cells_failed": failed,
                "updated_at": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - queue_started),
            },
        )

    summaries: list[dict[str, Any]] = []
    invalid: list[str] = []
    for cell in cells:
        summary = _read_json(_cell_dir(output, cell) / "summary.json")
        if summary is None or not _completed(
            _cell_dir(output, cell),
            cell,
            slices,
            source_sha256,
            config_sha256,
            data_sha256[str(cell["qos"])],
        ):
            invalid.append(str(cell["cell_id"]))
        else:
            summaries.append(summary)
    summary_rows = [_summary_row(value) for value in summaries]
    _atomic_csv(output / "final_summary.csv", summary_rows)
    _atomic_csv(output / "aggregate_mean_std.csv", _aggregate_rows(summary_rows))
    valid = len(invalid) == 0 and len(summaries) == len(cells)
    if valid:
        marker = {
            "status": "COMPLETED" if smoke else "BATCH_COMPLETED",
            "protocol": PROTOCOL,
            "scope": "smoke" if smoke else "formal",
            "cells_expected": len(cells),
            "cells_completed": len(summaries),
            "time_slices": int(slices),
            "final_summary_sha256": _sha256(output / "final_summary.csv"),
            "aggregate_mean_std_sha256": _sha256(output / "aggregate_mean_std.csv"),
            "communication_zero": True,
            "completed_at": _utc_now(),
        }
        _atomic_json(output / "BATCH_COMPLETED", marker)
        final_status = "COMPLETED" if smoke else "BATCH_COMPLETED"
    else:
        marker_path = output / "BATCH_COMPLETED"
        if marker_path.exists():
            marker_path.unlink()
        final_status = "COMPLETED_WITH_FAILURES"
    final = {
        "status": final_status,
        "protocol": PROTOCOL,
        "scope": "smoke" if smoke else "formal",
        "cells_expected": len(cells),
        "cells_valid": len(summaries),
        "cells_completed_this_invocation": completed_now,
        "cells_skipped_completed": skipped,
        "cells_failed": failed,
        "invalid_cells": invalid,
        "communication_zero": valid,
        "elapsed_seconds": float(time.perf_counter() - queue_started),
        "completed_at": _utc_now(),
    }
    _atomic_json(output / "queue_status.json", final)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{_utc_now()} {final_status} valid={len(summaries)}/{len(cells)}\n")
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="run D10/seed42/RT/one-slice smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = run_queue(
        config_path=args.config,
        output=args.output,
        smoke=bool(args.smoke),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        stop_on_error=bool(args.stop_on_error),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {"COMPLETED_WITH_FAILURES"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

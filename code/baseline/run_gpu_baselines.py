                                                                          

                                                                              
                                                                          
                                             
   

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_ROOT))

                                                                            
                                                                             
if importlib.util.find_spec("torch") is None and "core" not in sys.modules:
    package = types.ModuleType("core")
    package.__path__ = [str(ROOT / "code" / "core")]
    sys.modules["core"] = package

from core.data import load_value_cache
from core.protocol import make_causal_split
from gpu_models import (              
    create_model,
    cuda_available,
    cuda_device_info,
)


PROTOCOL = "fedarm_baseline_causal_baselines_gpu_v2_support_intercept"
PROTOCOL_VARIANT = "v2_support_intercept"
SUPPORT_INITIALIZATION = "slice0_support_mean_global_intercept"
INITIALIZATION_CLIP_EPSILON = 1e-4
SHAPE = (64, 142, 4500)
FORMAL_METHODS = ("fedavg", "fl_lft")
FORMAL_DENSITIES = (0.01, 0.05, 0.10, 0.15, 0.20)
FORMAL_SEEDS = (42, 3407, 1024, 1, 0)
FORMAL_QOS = ("rt", "tp")
FORMAL_SLICES = 64
DEFAULT_CONFIG = HERE / "baseline_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "baseline_gpu_formal_v2_support_intercept"
SOURCE_KIND = {
    "fedavg": "paper_and_open_reference_adaptation",
    "fl_lft": "paper_equation_independent_causal_reconstruction",
}
SOURCE_PROVENANCE = {
    "fedavg": {
        "primary_specification": "https://proceedings.mlr.press/v54/mcmahan17a.html",
        "official_method_repository_found": False,
        "adaptation_label": "FedAvg causal WSDream QoS regression adaptation",
        "protocol_variant": PROTOCOL_VARIANT,
        "support_initialization": SUPPORT_INITIALIZATION,
        "initialization_clip_epsilon": INITIALIZATION_CLIP_EPSILON,
        "initialization_scope": "slice-0 support only; query labels and future slices excluded",
        "legacy_v1_compatible": False,
    },
    "fl_lft": {
        "paper": "https://arxiv.org/abs/2407.19828",
        "official_repository_found": False,
        "adaptation_label": "FL-LFT paper-equation causal adaptation",
        "protocol_variant": PROTOCOL_VARIANT,
        "support_initialization": "one-shot first-slice support-only balanced trilinear scale",
        "initialization_scope": "slice-0 support only; query labels and future slices excluded",
        "legacy_v1_compatible": False,
    },
}
RESUME_INVARIANTS = (
    "protocol",
    "protocol_variant",
    "execution_mode",
    "workspace",
    "output",
    "shape",
    "methods",
    "densities",
    "seeds",
    "qos_types",
    "time_slices",
    "cell_count",
    "cells",
    "strict_causal_order",
    "query_or_future_used",
    "support_initialization",
    "initialization_clip_epsilon",
    "legacy_v1_compatible",
    "source_sha256",
    "data_sha256",
    "method_configs",
    "source_provenance",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    candidate = getattr(value, "as_dict", None)
    if callable(candidate):
        return dict(candidate())
    if isinstance(candidate, Mapping):
        return dict(candidate)
    raise TypeError(f"cannot convert {type(value).__name__} to a result mapping")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(
            "baseline config protocol must be "
            f"{PROTOCOL!r}; legacy v1 configurations are not accepted"
        )
    if tuple(config.get("shape_time_user_service", ())) != SHAPE:
        raise ValueError("baseline runner requires WSDream Dataset 3 shape 64x142x4500")
    if config.get("raw_prediction_caps") != {"rt": 20.0, "tp": 10000.0}:
        raise ValueError("baseline raw caps differ from the frozen FedARM protocol")
    methods = config.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("baseline config must define method configurations")
    missing = [method for method in FORMAL_METHODS if method not in methods]
    if missing:
        raise ValueError(f"baseline config is missing methods: {missing}")
    for method in ("fedavg",):
        method_config = methods[method]
        if not isinstance(method_config, Mapping):
            raise ValueError(f"baseline config for {method} must be a mapping")
        if method_config.get("support_initialization") != SUPPORT_INITIALIZATION:
            raise ValueError(
                f"{method} must use support initialization "
                f"{SUPPORT_INITIALIZATION!r}"
            )
        epsilon = method_config.get("initialization_clip_epsilon")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError(
                f"{method}.initialization_clip_epsilon must be a finite positive number"
            )
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError(
                f"{method}.initialization_clip_epsilon must be a finite positive number"
            )
        if not math.isclose(
            float(epsilon), INITIALIZATION_CLIP_EPSILON, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                f"{method}.initialization_clip_epsilon must equal "
                f"{INITIALIZATION_CLIP_EPSILON}"
            )
    return config


def _resume_mismatches(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
                                                                            
    return [
        field
        for field in RESUME_INVARIANTS
        if existing.get(field) != current.get(field)
    ]


def _cell_id(method: str, qos: str, density: float, seed: int) -> str:
    return f"{method}_{qos}_d{int(round(density * 100)):02d}_seed{int(seed)}"


def build_cells(
    methods: Sequence[str],
    densities: Sequence[float],
    seeds: Sequence[int],
    qos_types: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": _cell_id(method, qos, density, seed),
            "method": str(method),
            "qos": str(qos),
            "density": float(density),
            "seed": int(seed),
        }
        for method in methods
        for density in densities
        for seed in seeds
        for qos in qos_types
    ]


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    density = int(round(float(cell["density"]) * 100))
    return (
        output
        / str(cell["method"])
        / str(cell["qos"])
        / f"d{density:02d}"
        / f"seed{int(cell['seed'])}"
    )


def _completed(cell_dir: Path, cell: Mapping[str, Any], slices: int) -> bool:
    path = cell_dir / "summary.json"
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        summary.get("status") == "COMPLETED"
        and summary.get("protocol") == PROTOCOL
        and summary.get("protocol_variant") == PROTOCOL_VARIANT
        and summary.get("legacy_v1_compatible") is False
        and summary.get("method") == cell["method"]
        and summary.get("qos") == cell["qos"]
        and int(summary.get("seed", -1)) == int(cell["seed"])
        and math.isclose(
            float(summary.get("density", -1.0)),
            float(cell["density"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and int(summary.get("time_slices_completed", -1)) == int(slices)
        and bool(summary.get("finite_metrics_all_slices"))
        and summary.get("query_or_future_used") is False
        and (
            cell["method"] != "fedavg"
            or (
                summary.get("support_initialization") == SUPPORT_INITIALIZATION
                and summary.get("initialization_clip_epsilon")
                == INITIALIZATION_CLIP_EPSILON
            )
        )
    )


def _next_attempt(cell_dir: Path) -> Path:
    attempts = cell_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for path in attempts.glob("attempt_*"):
        try:
            indices.append(int(path.name.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    attempt = attempts / f"attempt_{max(indices, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or prediction.shape != truth.shape or truth.size == 0:
        raise ValueError("truth and prediction must be aligned non-empty vectors")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise FloatingPointError("non-finite truth or prediction")
    if np.any(truth <= 0.0):
        raise ValueError("relative metrics require strictly positive truth")
    absolute = np.abs(prediction - truth)
    relative = absolute / truth
    ordered = np.sort(relative)
    return {
        "mae": float(np.mean(absolute, dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2, dtype=np.float64))),
        "mre": float(np.median(ordered)),
        "npre": float(ordered[int(0.9 * len(ordered))]),
        "absolute_error_sum": float(np.sum(absolute, dtype=np.float64)),
        "squared_error_sum": float(np.sum((prediction - truth) ** 2, dtype=np.float64)),
        "relative": relative,
    }


def _state_hash(model: Any) -> str:
    snapshot = model.state_snapshot()
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ValueError("GPU baseline state_snapshot must return a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        value = np.ascontiguousarray(snapshot[name])
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"non-finite final state array: {name}")
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _summarize_metrics(
    rows: list[Mapping[str, Any]], relative_chunks: list[np.ndarray]
) -> dict[str, Any]:
    count = sum(int(row["query_entries"]) for row in rows)
    absolute = sum(float(row["absolute_error_sum"]) for row in rows)
    squared = sum(float(row["squared_error_sum"]) for row in rows)
    pooled = np.concatenate(relative_chunks)
    pooled.sort()
    return {
        "pooled": {
            "mae": float(absolute / count),
            "rmse": float(np.sqrt(squared / count)),
            "mre": float(np.median(pooled)),
            "npre": float(pooled[int(0.9 * count)]),
            "count": int(count),
        },
        "time_mean": {
            name: float(np.mean([float(row[name]) for row in rows], dtype=np.float64))
            for name in ("mae", "rmse", "mre", "npre")
        },
        "definitions": {
            "mre_per_slice": "median(sorted(abs(pred-truth)/truth))",
            "npre_per_slice": "sorted_relative[int(0.9*n)]",
            "time_mean": "unweighted arithmetic mean over completed time slices",
            "pooled": "all query observations concatenated across slices",
        },
    }


def _method_config(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    result = dict(config["methods"][method])
    result.update(
        {
            "num_times": SHAPE[0],
            "num_users": SHAPE[1],
            "num_services": SHAPE[2],
            "method": method,
        }
    )
    return result


def _validate_model_initialization_contract(
    model: Any, method: str, method_config: Mapping[str, Any]
) -> None:
                                                                              
    if method != "fedavg":
        return
    model_rule = getattr(model, "support_initialization", None)
    if model_rule != SUPPORT_INITIALIZATION:
        raise RuntimeError(
            f"{method} GPU model does not implement the v2 support initialization "
            f"contract {SUPPORT_INITIALIZATION!r}"
        )
    model_epsilon = getattr(model, "initialization_clip_epsilon", None)
    if isinstance(model_epsilon, bool) or not isinstance(model_epsilon, (int, float)):
        raise RuntimeError(
            f"{method} GPU model does not expose a valid v2 initialization clip epsilon"
        )
    configured_epsilon = float(method_config["initialization_clip_epsilon"])
    if not math.isclose(
        float(model_epsilon), configured_epsilon, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError(
            f"{method} GPU model initialization clip epsilon differs from config: "
            f"{model_epsilon!r} != {configured_epsilon!r}"
        )


def _extract_profile(model: Any) -> dict[str, Any]:
    profile = getattr(model, "last_profile", None)
    if callable(profile):
        profile = profile()
    try:
        return _as_dict(profile)
    except TypeError:
        return {}


def _extract_communication(model: Any, update: Mapping[str, Any]) -> dict[str, int]:
    communication: Any = update.get("communication")
    if communication is None:
        candidate = getattr(model, "communication", None)
        communication = candidate() if callable(candidate) else candidate
    values = _as_dict(communication) if communication is not None else dict(update)
    uplink = int(values.get("uplink_bytes", 0))
    downlink = int(values.get("downlink_bytes", 0))
    total = int(values.get("total_bytes", uplink + downlink))
    return {
        "uplink_bytes": uplink,
        "downlink_bytes": downlink,
        "total_bytes": total,
        "uploaded_parameters": int(values.get("uploaded_parameters", 0)),
        "downloaded_parameters": int(values.get("downloaded_parameters", 0)),
        "participating_clients": int(values.get("participating_clients", 0)),
    }


def run_cell(
    *,
    cell: Mapping[str, Any],
    values: np.ndarray,
    config: Mapping[str, Any],
    output: Path,
    slices: int,
    resume: bool,
) -> dict[str, Any]:
    cell_dir = _cell_dir(output, cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if _completed(cell_dir, cell, slices):
        if resume:
            return {"status": "SKIPPED_COMPLETED", **dict(cell)}
        raise FileExistsError(f"completed cell already exists: {cell_dir}")
    if (cell_dir / "attempts").exists() and any((cell_dir / "attempts").glob("attempt_*")):
        if not resume:
            raise FileExistsError(f"incomplete prior attempt exists; use --resume: {cell_dir}")

    attempt = _next_attempt(cell_dir)
    method = str(cell["method"])
    method_config = _method_config(config, method)
    status: dict[str, Any] = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "protocol_variant": PROTOCOL_VARIANT,
        "legacy_v1_compatible": False,
        "execution_mode": "numba_cuda_training_and_prediction",
        "source_kind": SOURCE_KIND[method],
        **dict(cell),
        "support_initialization": method_config.get(
            "support_initialization", SOURCE_PROVENANCE[method]["support_initialization"]
        ),
        "initialization_clip_epsilon": method_config.get(
            "initialization_clip_epsilon"
        ),
        "initialization_scope": SOURCE_PROVENANCE[method]["initialization_scope"],
        "time_slices_expected": int(slices),
        "time_slices_completed": 0,
        "started_at": _utc_now(),
        "attempt_dir": str(attempt),
        "query_or_future_used": False,
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)

    process = psutil.Process(os.getpid())
    cpu_times_start = process.cpu_times()
    peak_rss = int(process.memory_info().rss)
    rows: list[dict[str, Any]] = []
    relative_chunks: list[np.ndarray] = []
    raw_cap = float(config["raw_prediction_caps"][str(cell["qos"])])
    cell_started = time.perf_counter()

    try:
        model = create_model(method, method_config, int(cell["seed"]))
        _validate_model_initialization_contract(model, method, method_config)
        for time_slice in range(int(slices)):
            slice_started = time.perf_counter()
            split_started = time.perf_counter()
            split = make_causal_split(
                values[time_slice],
                float(cell["density"]),
                int(cell["seed"]),
                time_slice,
            )
            split_seconds = time.perf_counter() - split_started

            update_started = time.perf_counter()
            update = _as_dict(
                model.advance_support(
                    split.support_users,
                    split.support_services,
                    split.support_values,
                    int(time_slice),
                    raw_cap,
                    int(cell["seed"]) * 1_000_003 + int(time_slice),
                )
            )
            update_wall_seconds = time.perf_counter() - update_started

            prediction_started = time.perf_counter()
            prediction_result = model.predict(
                split.query_users, split.query_services, int(time_slice), raw_cap
            )
            if isinstance(prediction_result, tuple):
                prediction = np.asarray(prediction_result[0], dtype=np.float64)
            else:
                prediction = np.asarray(prediction_result, dtype=np.float64)
            predict_wall_seconds = time.perf_counter() - prediction_started

            metric_started = time.perf_counter()
            metrics = _prediction_metrics(split.query_values, prediction)
            relative_chunks.append(np.asarray(metrics.pop("relative"), dtype=np.float64))
            metric_seconds = time.perf_counter() - metric_started
            communication = _extract_communication(model, update)
            profile = _extract_profile(model)
            peak_rss = max(peak_rss, int(process.memory_info().rss))

            row = {
                "protocol": PROTOCOL,
                "protocol_variant": PROTOCOL_VARIANT,
                "execution_mode": "numba_cuda_training_and_prediction",
                "method": method,
                "qos": str(cell["qos"]),
                "density": float(cell["density"]),
                "seed": int(cell["seed"]),
                "support_initialization": method_config.get(
                    "support_initialization",
                    SOURCE_PROVENANCE[method]["support_initialization"],
                ),
                "initialization_clip_epsilon": method_config.get(
                    "initialization_clip_epsilon"
                ),
                "time_slice": int(time_slice),
                "support_entries": int(split.support_count),
                "query_entries": int(split.query_count),
                "mask_sha256": str(split.mask_sha256),
                **metrics,
                **communication,
                "split_seconds": float(split_seconds),
                "update_wall_seconds": float(update_wall_seconds),
                "predict_wall_seconds": float(predict_wall_seconds),
                "metric_seconds": float(metric_seconds),
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
                "gpu_peak_memory_used_bytes": int(
                    profile.get("peak_device_memory_used_bytes", 0)
                ),
                "process_rss_bytes": int(process.memory_info().rss),
                "query_or_future_used": False,
                "gpu_profile_json": json.dumps(profile, sort_keys=True),
                "update_audit_json": json.dumps(update, sort_keys=True),
            }
            if not all(math.isfinite(float(row[name])) for name in ("mae", "rmse", "mre", "npre")):
                raise FloatingPointError("non-finite baseline metric")
            rows.append(row)
            _atomic_csv(attempt / "per_slice.csv", rows)
            status["time_slices_completed"] = int(time_slice + 1)
            status["last_mask_sha256"] = str(split.mask_sha256)
            status["updated_at"] = _utc_now()
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)

        metric_summary = _summarize_metrics(rows, relative_chunks)
        cpu_times_end = process.cpu_times()
        summary = {
            "status": "COMPLETED",
            "protocol": PROTOCOL,
            "protocol_variant": PROTOCOL_VARIANT,
            "legacy_v1_compatible": False,
            "execution_mode": "numba_cuda_training_and_prediction",
            "source_kind": SOURCE_KIND[method],
            **dict(cell),
            "method_config": method_config,
            "support_initialization": method_config.get(
                "support_initialization", SOURCE_PROVENANCE[method]["support_initialization"]
            ),
            "initialization_clip_epsilon": method_config.get(
                "initialization_clip_epsilon"
            ),
            "initialization_scope": SOURCE_PROVENANCE[method]["initialization_scope"],
            "time_slices_completed": len(rows),
            "metrics": metric_summary,
            "finite_metrics_all_slices": True,
            "query_or_future_used": False,
            "communication_totals": {
                name: int(sum(int(row[name]) for row in rows))
                for name in (
                    "uplink_bytes",
                    "downlink_bytes",
                    "total_bytes",
                    "uploaded_parameters",
                    "downloaded_parameters",
                )
            },
            "timing_seconds": {
                name: float(sum(float(row[name]) for row in rows))
                for name in (
                    "split_seconds",
                    "update_wall_seconds",
                    "predict_wall_seconds",
                    "metric_seconds",
                    "slice_wall_seconds",
                )
            },
            "cell_wall_seconds": float(time.perf_counter() - cell_started),
            "gpu_peak_memory_used_bytes": int(
                max(int(row["gpu_peak_memory_used_bytes"]) for row in rows)
            ),
            "process_peak_rss_bytes": int(peak_rss),
            "process_cpu_user_seconds": float(cpu_times_end.user - cpu_times_start.user),
            "process_cpu_system_seconds": float(cpu_times_end.system - cpu_times_start.system),
            "final_state_sha256": _state_hash(model),
            "attempt_dir": str(attempt),
            "completed_at": _utc_now(),
        }
        _atomic_json(attempt / "summary.json", summary)
        _atomic_json(attempt / "status.json", summary)
        _atomic_json(cell_dir / "summary.json", summary)
        _atomic_json(cell_dir / "status.json", summary)
        return summary
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
        _atomic_json(attempt / "failure.json", failed)
        _atomic_json(cell_dir / "status.json", failed)
        raise


def run_queue(
    *,
    config_path: Path,
    output: Path,
    methods: Sequence[str],
    densities: Sequence[float],
    seeds: Sequence[int],
    qos_types: Sequence[str],
    slices: int,
    resume: bool,
    dry_run: bool,
    stop_on_error: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    allowed_output_root = (ROOT / "results").resolve()
    if not output.is_relative_to(allowed_output_root):
        raise ValueError(
            f"baseline outputs must stay below {allowed_output_root}; got {output}"
        )
    config = _load_config(config_path)
    cells = build_cells(methods, densities, seeds, qos_types)
    source_paths = {
        "runner": Path(__file__).resolve(),
        "gpu_models": HERE / "gpu_models.py",
        "config": config_path,
    }
    for method in methods:
        reference = HERE / "methods" / f"{method}.py"
        if reference.exists():
            source_paths[f"reference_{method}"] = reference
        reproduction_note = HERE / f"{method.upper()}_REPRODUCTION.md"
        if method == "fl_lft":
            reproduction_note = HERE / "FL_LFT_REPRODUCTION.md"
        if reproduction_note.exists():
            source_paths[f"provenance_{method}"] = reproduction_note
    data_paths = {
        qos: ROOT / str(config["cache_dir"]) / f"{qos}_values.npy"
        for qos in qos_types
    }
    missing_data = [str(path) for path in data_paths.values() if not path.is_file()]
    if missing_data and not dry_run:
        raise FileNotFoundError(f"missing frozen value caches: {missing_data}")
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_variant": PROTOCOL_VARIANT,
        "legacy_v1_compatible": False,
        "execution_mode": "numba_cuda_training_and_prediction",
        "created_at": _utc_now(),
        "workspace": str(ROOT),
        "output": str(output),
        "shape": list(SHAPE),
        "methods": list(methods),
        "densities": [float(value) for value in densities],
        "seeds": [int(value) for value in seeds],
        "qos_types": [str(value) for value in qos_types],
        "time_slices": int(slices),
        "cell_count": len(cells),
        "cells": cells,
        "strict_causal_order": "history + support[t] -> update -> read-only query[t]",
        "query_or_future_used": False,
        "support_initialization": {
            "methods": ["fedavg"],
            "rule": SUPPORT_INITIALIZATION,
            "initialization_clip_epsilon": INITIALIZATION_CLIP_EPSILON,
            "data_scope": "slice-0 support only",
            "query_labels_used": False,
            "future_slices_used": False,
            "legacy_v1_results_accepted": False,
        },
        "initialization_clip_epsilon": INITIALIZATION_CLIP_EPSILON,
        "backend": cuda_device_info(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "data_cache_paths": {
            qos: str(path.relative_to(ROOT)) for qos, path in data_paths.items()
        },
        "data_sha256": {
            qos: _sha256(path) if path.is_file() else None
            for qos, path in data_paths.items()
        },
        "method_configs": {
            method: _method_config(config, method) for method in methods
        },
        "source_provenance": {
            method: SOURCE_PROVENANCE[method] for method in methods
        },
    }
    if dry_run:
        return {"status": "DRY_RUN", **manifest}
    if not cuda_available():
        raise RuntimeError("a usable Numba CUDA device is required")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"baseline output already initialized; use --resume: {output}"
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = _resume_mismatches(existing_manifest, manifest)
        if mismatches:
            raise RuntimeError(
                "resume refused because frozen manifest fields changed: "
                + ", ".join(mismatches)
            )
        manifest = dict(existing_manifest)
    else:
        preexisting_artifacts = list(output.rglob("summary.json")) + list(
            output.rglob("queue_status.json")
        )
        if preexisting_artifacts:
            raise RuntimeError(
                "baseline output contains pre-existing artifacts without a v2 "
                "manifest; choose a new v2_support_intercept output root"
            )
        _atomic_json(manifest_path, manifest)

    queue_rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    cache: dict[str, np.ndarray] = {}
    queue_started = time.perf_counter()
    for index, cell in enumerate(cells):
        cell_dir = _cell_dir(output, cell)
        if _completed(cell_dir, cell, slices):
            skipped += 1
            queue_rows.append({**cell, "status": "SKIPPED_COMPLETED", "error": ""})
        else:
            try:
                qos = str(cell["qos"])
                if qos not in cache:
                    data_path = data_paths[qos]
                    cache[qos] = load_value_cache(data_path, SHAPE)
                result = run_cell(
                    cell=cell,
                    values=cache[qos],
                    config=config,
                    output=output,
                    slices=slices,
                    resume=resume,
                )
                completed += 1
                queue_rows.append({**cell, "status": result["status"], "error": ""})
            except Exception as exc:
                failed += 1
                queue_rows.append(
                    {
                        **cell,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if stop_on_error:
                    _atomic_csv(output / "queue_cells.csv", queue_rows)
                    _atomic_json(
                        output / "queue_status.json",
                        {
                            "status": "FAILED",
                            "protocol": PROTOCOL,
                            "cells_expected": len(cells),
                            "cells_visited": index + 1,
                            "cells_completed_this_invocation": completed,
                            "cells_skipped_completed": skipped,
                            "cells_failed": failed,
                            "failed_cell": dict(cell),
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_at": _utc_now(),
                            "elapsed_seconds": float(
                                time.perf_counter() - queue_started
                            ),
                        },
                    )
                    raise
        _atomic_csv(output / "queue_cells.csv", queue_rows)
        _atomic_json(
            output / "queue_status.json",
            {
                "status": "RUNNING",
                "protocol": PROTOCOL,
                "cells_expected": len(cells),
                "cells_visited": index + 1,
                "cells_completed_this_invocation": completed,
                "cells_skipped_completed": skipped,
                "cells_failed": failed,
                "updated_at": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - queue_started),
            },
        )

    final = {
        "status": "COMPLETED" if failed == 0 else "COMPLETED_WITH_FAILURES",
        "protocol": PROTOCOL,
        "cells_expected": len(cells),
        "cells_completed_this_invocation": completed,
        "cells_skipped_completed": skipped,
        "cells_failed": failed,
        "elapsed_seconds": float(time.perf_counter() - queue_started),
        "completed_at": _utc_now(),
    }
    _atomic_json(output / "queue_status.json", final)
    return final


def _density(value: str) -> float:
    parsed = float(value)
    if parsed > 1.0:
        parsed /= 100.0
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("density must lie in (0,1), or be a percentage")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=FORMAL_METHODS, default=list(FORMAL_METHODS))
    parser.add_argument("--densities", nargs="+", type=_density, default=list(FORMAL_DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--qos", nargs="+", choices=FORMAL_QOS, default=list(FORMAL_QOS))
    parser.add_argument("--max-time-slices", type=int, default=FORMAL_SLICES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.max_time_slices <= 0 or args.max_time_slices > FORMAL_SLICES:
        raise ValueError("max-time-slices must be in [1,64]")
    result = run_queue(
        config_path=args.config,
        output=args.output,
        methods=tuple(dict.fromkeys(str(value) for value in args.methods)),
        densities=tuple(dict.fromkeys(float(value) for value in args.densities)),
        seeds=tuple(dict.fromkeys(int(value) for value in args.seeds)),
        qos_types=tuple(dict.fromkeys(str(value) for value in args.qos)),
        slices=int(args.max_time_slices),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        stop_on_error=bool(args.stop_on_error),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") != "COMPLETED_WITH_FAILURES" else 1


if __name__ == "__main__":
    raise SystemExit(main())

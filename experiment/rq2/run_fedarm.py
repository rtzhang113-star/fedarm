                                                                             

                                                                           
                                                                             
                                                                   
   

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
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

                                                                            
                                                                             
if importlib.util.find_spec("torch") is None and "core" not in sys.modules:
    package = types.ModuleType("core")
    package.__path__ = [str(ROOT / "code" / "core")]
    sys.modules["core"] = package

from core.data import load_value_cache
from core.gpu_full_training import (
    EXECUTION_MODE,
    GPUFinalStackModel,
    cuda_available,
    cuda_device_info,
    prediction_metrics,
)
from core.hmp_sr16_codec import PRECISION_INT8
from core.protocol import make_causal_split
from core.sdq8_attribution_controls import (
    FixedLinkPrecisionCommunicationAccountant,
)


PROTOCOL = "fedarm_full_gpu_v1"
SHAPE = (64, 142, 4500)
FORMAL_DENSITIES = (0.01, 0.05, 0.10, 0.15, 0.20)
FORMAL_SEEDS = (42, 3407, 1024, 1, 0)
FORMAL_QOS = ("rt", "tp")
FORMAL_SLICES = 64
DEFAULT_DENSITIES = (0.10,)
DEFAULT_SEEDS = (42,)
DEFAULT_CONFIG = ROOT / "experiment" / "rq2" / "fedarm.json"
DEFAULT_OUTPUT = ROOT / "results"
SOURCE_MODULES = (
    "data.py",
    "gpu_full_training.py",
    "hmp_sr16_codec.py",
    "protocol.py",
    "sdq8_attribution_controls.py",
    "edge_cache_communication.py",
    "federated_residual.py",
    "personalized_service_adapter.py",
    "residual_model.py",
    "adaptive_residual_window.py",
    "drift_diagnostic.py",
    "pcs_srr_combo_fedobr.py",
    "drift_residual_recenter.py",
    "proposal_count_scaled_fedobr.py",
    "fedobr_mf.py",
    "hierarchical_stable_plastic.py",
    "hierarchical_fedobr.py",
    "stable_plastic_federated.py",
    "stable_plastic_projection.py",
)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(config_path: Path) -> dict[str, str]:
                                                                              

    paths = [
        Path(__file__).resolve(),
        *(ROOT / "code" / "core" / name for name in SOURCE_MODULES),
        config_path.resolve(),
    ]
    result: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            key = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            key = resolved.as_posix()
        result[key] = _sha256(resolved)
    return result


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


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config.get("shape_time_user_service", ())) != SHAPE:
        raise ValueError("formal GPU runner requires WSDream Dataset 3 shape 64x142x4500")
    if config.get("raw_prediction_caps") != {"rt": 20.0, "tp": 10000.0}:
        raise ValueError("raw QoS caps differ from the frozen final-stack protocol")
    required = (
        "rank", "bias_eta", "factor_eta", "l2", "beta", "factor_init_std",
        "num_edges", "bias_epochs", "factor_epochs", "cache_dir",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing frozen config fields: {missing}")
    return config


def _new_model(config: Mapping[str, Any], seed: int) -> GPUFinalStackModel:
    return GPUFinalStackModel(
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


def _new_accountant(config: Mapping[str, Any]) -> Any:
    return FixedLinkPrecisionCommunicationAccountant(
        num_users=SHAPE[1],
        num_services=SHAPE[2],
        rank=int(config["rank"]),
        num_edges=int(config["num_edges"]),
        proposal_precision=PRECISION_INT8,
        summary_precision=PRECISION_INT8,
        refresh_precision=PRECISION_INT8,
        delivery_precision=PRECISION_INT8,
    )


def _cell_id(qos: str, density: float, seed: int) -> str:
    return f"{qos}_d{int(round(density * 100)):02d}_seed{int(seed)}"


def build_cells(
    densities: Sequence[float], seeds: Sequence[int], qos_types: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": _cell_id(qos, density, seed),
            "qos": str(qos),
            "density": float(density),
            "seed": int(seed),
        }
        for density in densities
        for seed in seeds
        for qos in qos_types
    ]


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    density = int(round(float(cell["density"]) * 100))
    return output / str(cell["qos"]) / f"d{density:02d}" / f"seed{int(cell['seed'])}"


def _completed(cell_dir: Path, slices: int) -> bool:
    summary = cell_dir / "summary.json"
    if not summary.exists():
        return False
    try:
        value = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETED"
        and int(value.get("time_slices_completed", -1)) == int(slices)
        and bool(value.get("finite_metrics_all_slices"))
        and bool(value.get("query_or_future_used")) is False
    )


def _next_attempt(cell_dir: Path) -> Path:
    attempts = cell_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in attempts.glob("attempt_*"):
        try:
            indices.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    attempt = attempts / f"attempt_{max(indices, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _state_hash(model: GPUFinalStackModel) -> str:
    digest = hashlib.sha256()
    snapshot = model.parameter_snapshot()
    for name in sorted(snapshot):
        value = np.ascontiguousarray(snapshot[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    history = model.adaptive_residual_history_snapshot()
    for user, values in sorted(history.items()):
        digest.update(f"carw:{user}".encode("ascii"))
        digest.update(np.asarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _summarize_metrics(
    rows: list[Mapping[str, Any]], relative_chunks: list[np.ndarray]
) -> dict[str, Any]:
    count = sum(int(row["query_entries"]) for row in rows)
    absolute = sum(float(row["absolute_error_sum"]) for row in rows)
    squared = sum(float(row["squared_error_sum"]) for row in rows)
    pooled = np.concatenate(relative_chunks)
    pooled.sort()
    time_mean = {
        name: float(np.mean([float(row[name]) for row in rows], dtype=np.float64))
        for name in ("mae", "rmse", "mre", "npre")
    }
    result = {
        "pooled": {
            "mae": float(absolute / count),
            "rmse": float(np.sqrt(squared / count)),
            "mre": float(np.median(pooled)),
            "npre": float(pooled[int(0.9 * count)]),
            "count": int(count),
        },
        "time_mean": time_mean,
        "definitions": {
            "mre_per_slice": "median(sorted(abs(pred-truth)/truth))",
            "npre_per_slice": "sorted_relative[int(0.9*n)]",
            "time_mean": "unweighted arithmetic mean over completed time slices",
            "pooled": "computed over all query observations concatenated across slices",
        },
    }
    del pooled
    return result


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
    if _completed(cell_dir, slices):
        if resume:
            return {"status": "SKIPPED_COMPLETED", **dict(cell)}
        raise FileExistsError(f"completed cell already exists: {cell_dir}")
    if any((cell_dir / "attempts").glob("attempt_*")) if (cell_dir / "attempts").exists() else False:
        if not resume:
            raise FileExistsError(f"incomplete prior attempt exists; use --resume: {cell_dir}")

    attempt = _next_attempt(cell_dir)
    status = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "execution_mode": EXECUTION_MODE,
        **dict(cell),
        "time_slices_expected": int(slices),
        "time_slices_completed": 0,
        "started_at": _utc_now(),
        "attempt_dir": str(attempt),
        "gpu_only_model_arithmetic": True,
        "cpu_reference_model_constructed": False,
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)

    process = psutil.Process(os.getpid())
    cpu_times_start = process.cpu_times()
    peak_rss = int(process.memory_info().rss)
    rows: list[dict[str, Any]] = []
    relative_chunks: list[np.ndarray] = []
    model = _new_model(config, int(cell["seed"]))
    accountant = _new_accountant(config)
    raw_cap = float(config["raw_prediction_caps"][str(cell["qos"])])
    started_cell = time.perf_counter()

    try:
        for time_slice in range(int(slices)):
            slice_started = time.perf_counter()
            started = time.perf_counter()
            split = make_causal_split(
                values[time_slice],
                float(cell["density"]),
                int(cell["seed"]),
                time_slice,
            )
            split_seconds = time.perf_counter() - started

            started = time.perf_counter()
            result = model.advance_support(
                split.support_users,
                split.support_services,
                split.support_values,
                raw_cap,
                int(config["bias_epochs"]),
                int(config["factor_epochs"]),
                int(cell["seed"]) * 1_000_003 + time_slice,
            )
            update_wall_seconds = time.perf_counter() - started

            started = time.perf_counter()
            prediction, predict_timing = model.predict(
                split.query_users, split.query_services, raw_cap
            )
            predict_wall_seconds = time.perf_counter() - started

            started = time.perf_counter()
            metrics = prediction_metrics(split.query_values, prediction)
            truth = np.asarray(split.query_values, dtype=np.float64)
            relative = np.abs(np.asarray(prediction, dtype=np.float64) - truth) / truth
            relative_chunks.append(relative)
            metric_seconds = time.perf_counter() - started

            started = time.perf_counter()
            communication = accountant.account_round(
                split.support_users, split.support_services, result
            ).as_dict
            accounting_seconds = time.perf_counter() - started
            profile = model.last_profile
            if profile is None:
                raise AssertionError("GPU training profile missing")
            if not all(np.isfinite(float(metrics[name])) for name in ("mae", "rmse", "mre", "npre")):
                raise FloatingPointError("non-finite RQ1 metric")
            peak_rss = max(peak_rss, int(process.memory_info().rss))
            row: dict[str, Any] = {
                "protocol": PROTOCOL,
                "execution_mode": EXECUTION_MODE,
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
                "update_wall_seconds": float(update_wall_seconds),
                "predict_wall_seconds": float(predict_wall_seconds),
                "metric_seconds": float(metric_seconds),
                "communication_accounting_seconds": float(accounting_seconds),
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
                "gpu_peak_memory_used_bytes": int(profile.peak_device_memory_used_bytes),
                "process_rss_bytes": int(process.memory_info().rss),
                "query_or_future_used": False,
                "gpu_profile_json": json.dumps(profile.as_dict(), sort_keys=True),
            }
            for name in COMMUNICATION_FIELDS:
                row[name] = int(communication[name])
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
            "execution_mode": EXECUTION_MODE,
            **dict(cell),
            "time_slices_completed": len(rows),
            "metrics": metric_summary,
            "finite_metrics_all_slices": True,
            "query_or_future_used": False,
            "gpu_only_model_arithmetic": True,
            "cpu_reference_model_constructed": False,
            "communication_totals": {
                name: int(sum(int(row[name]) for row in rows))
                for name in COMMUNICATION_FIELDS
            },
            "timing_seconds": {
                name: float(sum(float(row[name]) for row in rows))
                for name in (
                    "split_seconds", "update_wall_seconds", "predict_wall_seconds",
                    "metric_seconds", "communication_accounting_seconds", "slice_wall_seconds",
                )
            },
            "cell_wall_seconds": float(time.perf_counter() - started_cell),
            "gpu_peak_memory_used_bytes": int(max(int(row["gpu_peak_memory_used_bytes"]) for row in rows)),
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
    config = _load_config(config_path)
    cells = build_cells(densities, seeds, qos_types)
    manifest = {
        "protocol": PROTOCOL,
        "execution_mode": EXECUTION_MODE,
        "created_at": _utc_now(),
        "workspace": str(ROOT),
        "output": str(output),
        "shape": list(SHAPE),
        "densities": [float(value) for value in densities],
        "seeds": [int(value) for value in seeds],
        "qos_types": [str(value) for value in qos_types],
        "time_slices": int(slices),
        "cell_count": len(cells),
        "cells": cells,
        "gpu_only_model_arithmetic": True,
        "cpu_reference_model_constructed": False,
        "gpu_scope": [
            "FedOBR-MF bias/factor updates", "CPSA", "CARW/SRR", "PCS",
            "SAQ8 proposal and aggregate quantization", "query prediction",
        ],
        "cpu_scope": [
            "data load", "causal split/index schedule", "communication accounting",
            "metrics", "telemetry", "artifact I/O",
        ],
        "backend": cuda_device_info(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "source_sha256": _source_sha256(config_path),
    }
    if dry_run:
        return {"status": "DRY_RUN", **manifest}
    if not cuda_available():
        raise RuntimeError("a usable Numba CUDA device is required")
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "manifest.json"
    if existing.exists() and not resume:
        raise FileExistsError(f"formal output already initialized; use --resume: {output}")
    _atomic_json(existing, manifest)

    queue_rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    cache: dict[str, np.ndarray] = {}
    queue_started = time.perf_counter()
    for index, cell in enumerate(cells):
        cell_dir = _cell_dir(output, cell)
        if _completed(cell_dir, slices):
            skipped += 1
            queue_rows.append({**cell, "status": "SKIPPED_COMPLETED", "error": ""})
        else:
            try:
                qos = str(cell["qos"])
                if qos not in cache:
                    data_path = ROOT / str(config["cache_dir"]) / f"{qos}_values.npy"
                    cache[qos] = load_value_cache(data_path, SHAPE)
                    manifest.setdefault("data_sha256", {})[qos] = _sha256(data_path)
                    _atomic_json(existing, manifest)
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
                queue_rows.append({
                    **cell,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if stop_on_error:
                    _atomic_csv(output / "queue_cells.csv", queue_rows)
                    raise
        _atomic_csv(output / "queue_cells.csv", queue_rows)
        queue_status = {
            "status": "RUNNING",
            "protocol": PROTOCOL,
            "cells_expected": len(cells),
            "cells_visited": index + 1,
            "cells_completed_this_invocation": completed,
            "cells_skipped_completed": skipped,
            "cells_failed": failed,
            "updated_at": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - queue_started),
        }
        _atomic_json(output / "queue_status.json", queue_status)

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
    parser.add_argument(
        "--densities",
        nargs="+",
        type=_density,
        default=list(DEFAULT_DENSITIES),
        help="Support densities; pass all five values for the paper grid.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds; pass all five values for the paper grid.",
    )
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
        densities=tuple(dict.fromkeys(float(value) for value in args.densities)),
        seeds=tuple(dict.fromkeys(int(value) for value in args.seeds)),
        qos_types=tuple(dict.fromkeys(str(value) for value in args.qos)),
        slices=int(args.max_time_slices),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        stop_on_error=bool(args.stop_on_error),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {"COMPLETED_WITH_FAILURES"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

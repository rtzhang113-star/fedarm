                                                                               

                                                                        
                                                                           
                                                                            
                   

                                                              

                                                                 
   

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_ROOT))

                                                                              
if importlib.util.find_spec("torch") is None and "core" not in sys.modules:
    package = types.ModuleType("core")
    package.__path__ = [str(ROOT / "code" / "core")]
    sys.modules["core"] = package

from core.data import load_value_cache
from core.protocol import make_causal_split
from baseline.reference_models import create_reference_model


PROTOCOL = "fedarm_baseline_causal_reference_reproductions_v1"
EXECUTION_MODE = "numpy_cpu_reference_reconstruction"
SHAPE = (64, 142, 4500)
FORMAL_METHODS = ("multifed", "fedrecgel", "fedem")
FORMAL_DENSITIES = (0.01, 0.05, 0.10, 0.15, 0.20)
FORMAL_SEEDS = (42, 3407, 1024, 1, 0)
FORMAL_QOS = ("rt", "tp")
FORMAL_SLICES = 64
DEFAULT_CONFIG = HERE / "reproduced_baseline_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced_baselines_formal"
SOURCE_KIND = {
    "multifed": "paper_equation_independent_causal_reconstruction",
    "fedrecgel": "official_repository_semantics_independent_causal_adaptation",
    "fedem": "official_repository_semantics_independent_causal_adaptation",
}
SOURCE_PROVENANCE = {
    "multifed": {
        "paper_doi": "10.1016/j.knosys.2023.110463",
        "official_repository_found": False,
        "adaptation_label": "MultiFed paper-equation causal WSDream QoS reconstruction",
    },
    "fedrecgel": {
        "official_repository": "https://github.com/anonymifish/FedRecGEL",
        "official_commit": "c7f2d929b49bb4112d8623d284c4c521663b0785",
        "official_license": "GPL-3.0",
        "adaptation_label": "FedRecGEL official-mechanism causal WSDream QoS adaptation",
    },
    "fedem": {
        "official_repository": "https://github.com/jundongchen13/FedEM",
        "official_commit": "747ed85b17156ef62b3813a34f08c7ad626c2cf0",
        "official_license_found": False,
        "adaptation_label": "FedEM official-mechanism causal WSDream QoS adaptation",
    },
}


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


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must equal {PROTOCOL!r}")
    methods = value.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("config must define method mappings")
    missing = [method for method in FORMAL_METHODS if method not in methods]
    if missing:
        raise ValueError(f"config is missing methods: {missing}")
    return dict(value)


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
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        summary.get("status") == "COMPLETED"
        and summary.get("protocol") == PROTOCOL
        and summary.get("method") == cell["method"]
        and summary.get("qos") == cell["qos"]
        and int(summary.get("seed", -1)) == int(cell["seed"])
        and math.isclose(
            float(summary.get("density", -1.0)),
            float(cell["density"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and int(summary.get("time_slices_completed", -1)) == int(slices)
        and summary.get("finite_metrics_all_slices") is True
        and summary.get("query_read_only_all_slices") is True
        and summary.get("query_or_future_used") is False
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


def _summarize_metrics(
    rows: list[Mapping[str, Any]], relative_chunks: list[np.ndarray]
) -> dict[str, Any]:
    count = sum(int(row["query_entries"]) for row in rows)
    pooled = np.concatenate(relative_chunks)
    pooled.sort()
    absolute = sum(float(row["absolute_error_sum"]) for row in rows)
    squared = sum(float(row["squared_error_sum"]) for row in rows)
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
    }


def _method_config(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    result = dict(config["methods"][method])
    result.update(
        {
            "method": method,
            "num_times": SHAPE[0],
            "num_users": SHAPE[1],
            "num_services": SHAPE[2],
        }
    )
    return result


def _communication(update: Mapping[str, Any]) -> dict[str, int]:
    value = update.get("communication", update)
    if not isinstance(value, Mapping):
        value = {}
    uplink = int(value.get("uplink_bytes", 0))
    downlink = int(value.get("downlink_bytes", 0))
    return {
        "uplink_bytes": uplink,
        "downlink_bytes": downlink,
        "total_bytes": int(value.get("total_bytes", uplink + downlink)),
        "participating_clients": int(value.get("participating_clients", 0)),
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
        "execution_mode": EXECUTION_MODE,
        "source_kind": SOURCE_KIND[method],
        **dict(cell),
        "time_slices_expected": int(slices),
        "time_slices_completed": 0,
        "query_or_future_used": False,
        "started_at": _utc_now(),
        "attempt_dir": str(attempt),
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)

    model = create_reference_model(method, method_config, int(cell["seed"]))
    raw_cap = float(config["raw_prediction_caps"][str(cell["qos"])])
    process = psutil.Process(os.getpid())
    peak_rss = int(process.memory_info().rss)
    rows: list[dict[str, Any]] = []
    relative_chunks: list[np.ndarray] = []
    cell_started = time.perf_counter()
    try:
        for time_slice in range(int(slices)):
            slice_started = time.perf_counter()
            split = make_causal_split(
                values[time_slice],
                float(cell["density"]),
                int(cell["seed"]),
                int(time_slice),
            )
            update_started = time.perf_counter()
            update = model.advance_support(
                split.support_users,
                split.support_services,
                split.support_values,
                int(time_slice),
                raw_cap,
                int(cell["seed"]) * 1_000_003 + int(time_slice),
            )
            update_seconds = time.perf_counter() - update_started

            query_snapshot = model.query_snapshot()
            prediction_started = time.perf_counter()
            prediction = np.asarray(
                model.predict(
                    split.query_users,
                    split.query_services,
                    int(time_slice),
                    raw_cap,
                ),
                dtype=np.float64,
            )
            prediction_seconds = time.perf_counter() - prediction_started
            query_read_only = bool(model.query_unchanged(query_snapshot))
            if not query_read_only:
                raise AssertionError("query prediction mutated model state")

            metrics = _prediction_metrics(split.query_values, prediction)
            relative_chunks.append(np.asarray(metrics.pop("relative"), dtype=np.float64))
            communication = _communication(update)
            peak_rss = max(peak_rss, int(process.memory_info().rss))
            row = {
                "protocol": PROTOCOL,
                "execution_mode": EXECUTION_MODE,
                "method": method,
                "qos": str(cell["qos"]),
                "density": float(cell["density"]),
                "seed": int(cell["seed"]),
                "time_slice": int(time_slice),
                "support_entries": int(split.support_count),
                "query_entries": int(split.query_count),
                "mask_sha256": str(split.mask_sha256),
                **metrics,
                **communication,
                "update_seconds": float(update_seconds),
                "prediction_seconds": float(prediction_seconds),
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
                "process_rss_bytes": int(process.memory_info().rss),
                "query_read_only": True,
                "query_or_future_used": False,
                "update_audit_json": json.dumps(update, sort_keys=True),
            }
            if not all(math.isfinite(float(row[name])) for name in ("mae", "rmse", "mre", "npre")):
                raise FloatingPointError("non-finite baseline metric")
            rows.append(row)
            _atomic_csv(attempt / "per_slice.csv", rows)
            status["time_slices_completed"] = int(time_slice + 1)
            status["updated_at"] = _utc_now()
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)

        summary = {
            "status": "COMPLETED",
            "protocol": PROTOCOL,
            "execution_mode": EXECUTION_MODE,
            "source_kind": SOURCE_KIND[method],
            **dict(cell),
            "method_config": method_config,
            "time_slices_completed": len(rows),
            "metrics": _summarize_metrics(rows, relative_chunks),
            "finite_metrics_all_slices": True,
            "query_read_only_all_slices": True,
            "query_or_future_used": False,
            "communication_totals": {
                name: int(sum(int(row[name]) for row in rows))
                for name in ("uplink_bytes", "downlink_bytes", "total_bytes")
            },
            "timing_seconds": {
                name: float(sum(float(row[name]) for row in rows))
                for name in ("update_seconds", "prediction_seconds", "slice_wall_seconds")
            },
            "cell_wall_seconds": float(time.perf_counter() - cell_started),
            "process_peak_rss_bytes": int(peak_rss),
            "final_state_sha256": model.state_hash(),
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
        _atomic_json(attempt / "failure.json", failed)
        _atomic_json(attempt / "status.json", failed)
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
        raise ValueError(f"outputs must stay below {allowed_output_root}; got {output}")
    config = _load_config(config_path)
    cells = build_cells(methods, densities, seeds, qos_types)
    data_paths = {
        qos: ROOT / str(config["cache_dir"]) / f"{qos}_values.npy"
        for qos in qos_types
    }
    missing = [str(path) for path in data_paths.values() if not path.is_file()]
    if missing and not dry_run:
        raise FileNotFoundError(f"missing frozen value caches: {missing}")
    source_paths: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "adapter": HERE / "reference_models.py",
        "config": config_path,
    }
    for method in methods:
        method_path = HERE / "methods" / f"{method}.py"
        if method == "multifed":
            method_path = ROOT / "code" / "core" / "multifed.py"
        source_paths[f"method_{method}"] = method_path
        provenance = HERE / f"{method.upper()}_REPRODUCTION.md"
        if provenance.is_file():
            source_paths[f"provenance_{method}"] = provenance
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "execution_mode": EXECUTION_MODE,
        "created_at": _utc_now(),
        "workspace": str(ROOT),
        "output": str(output),
        "shape": list(SHAPE),
        "methods": list(methods),
        "densities": [float(value) for value in densities],
        "seeds": [int(value) for value in seeds],
        "qos_types": list(qos_types),
        "time_slices": int(slices),
        "cell_count": len(cells),
        "cells": cells,
        "strict_causal_order": "history + support[t] -> update -> read-only query[t]",
        "query_or_future_used": False,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "data_cache_paths": {
            qos: str(path.relative_to(ROOT)) for qos, path in data_paths.items()
        },
        "data_sha256": {
            qos: _sha256(path) if path.is_file() else None
            for qos, path in data_paths.items()
        },
        "method_configs": {method: _method_config(config, method) for method in methods},
        "source_provenance": {method: SOURCE_PROVENANCE[method] for method in methods},
    }
    if dry_run:
        return {"status": "DRY_RUN", **manifest}

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"output already initialized; use --resume: {output}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = (
            "protocol", "execution_mode", "workspace", "output", "shape", "methods",
            "densities", "seeds", "qos_types", "time_slices", "cell_count", "cells",
            "strict_causal_order", "query_or_future_used", "source_sha256", "data_sha256",
            "method_configs", "source_provenance",
        )
        mismatches = [name for name in frozen if existing.get(name) != manifest.get(name)]
        if mismatches:
            raise RuntimeError("resume refused; frozen fields changed: " + ", ".join(mismatches))
        manifest = existing
    else:
        _atomic_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    cache: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for cell in cells:
        try:
            qos = str(cell["qos"])
            if qos not in cache:
                cache[qos] = load_value_cache(data_paths[qos], SHAPE)
            result = run_cell(
                cell=cell,
                values=cache[qos],
                config=config,
                output=output,
                slices=slices,
                resume=resume,
            )
            if result["status"] == "SKIPPED_COMPLETED":
                skipped += 1
            else:
                completed += 1
            rows.append({**cell, "status": result["status"], "error": ""})
        except Exception as exc:
            failed += 1
            rows.append({**cell, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            _atomic_csv(output / "queue_cells.csv", rows)
            if stop_on_error:
                raise
        _atomic_csv(output / "queue_cells.csv", rows)
    final = {
        "status": "COMPLETED" if failed == 0 else "COMPLETED_WITH_FAILURES",
        "protocol": PROTOCOL,
        "cells_expected": len(cells),
        "cells_completed_this_invocation": completed,
        "cells_skipped_completed": skipped,
        "cells_failed": failed,
        "elapsed_seconds": float(time.perf_counter() - started),
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
    if not 0 < int(args.max_time_slices) <= FORMAL_SLICES:
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
    return 0 if result["status"] in {"DRY_RUN", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

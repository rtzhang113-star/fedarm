                                                    

                                                                              
                                                                        
                                                                           
                                                                            
                                                 

                                                                             
                                                                       
                                                                              
                                                                              
                                          
   

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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:                                                                     
    import psutil
except Exception:                                                 
    psutil = None                            


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

                                                                              
                                                                
if importlib.util.find_spec("torch") is None and "core" not in sys.modules:
    package = type(sys)("core")
    package.__path__ = [str(ROOT / "code" / "core")]
    sys.modules["core"] = package

from core.data import load_value_cache
from core.protocol import make_causal_split


PROTOCOL = "fedsaa_rq3_accuracy_ablation_v1"
EXECUTION_MODE = "rq3_accuracy_ablation_gpu_v1"
SHAPE = (64, 142, 4500)
FORMAL_DENSITIES = (0.01, 0.05, 0.10, 0.15, 0.20)
FORMAL_SEEDS = (42, 3407, 1024, 1, 0)
FORMAL_QOS = ("rt", "tp")
FORMAL_SLICES = 64
SMOKE_DENSITIES = (0.10,)
SMOKE_SEEDS = (42,)
SMOKE_QOS = ("rt",)
SMOKE_SLICES = 1
DEFAULT_CONFIG = ROOT / "code" / "config" / "base_config.json"
DEFAULT_FORMAL_OUTPUT = ROOT / "results" / "rq3_ablation_accuracy"
DEFAULT_SMOKE_OUTPUT = ROOT / "results" / "rq3_ablation_accuracy_smoke"

                                                                            
                                                                          
VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant_id": "A0",
        "name": "backbone_mean",
        "bits": "000",
        "use_cpsa": False,
        "use_pcs": False,
        "use_carw_srr": False,
        "paper_label": "Base",
    },
    {
        "variant_id": "A1",
        "name": "cpsa_mean",
        "bits": "100",
        "use_cpsa": True,
        "use_pcs": False,
        "use_carw_srr": False,
        "paper_label": "FedARM w/o FedSAA",
    },
    {
        "variant_id": "A2",
        "name": "pcs_only",
        "bits": "010",
        "use_cpsa": False,
        "use_pcs": True,
        "use_carw_srr": False,
        "paper_label": "FedARM w/o RUMF & Aux. Stabilization",
    },
    {
        "variant_id": "A3",
        "name": "cpsa_pcs",
        "bits": "110",
        "use_cpsa": True,
        "use_pcs": True,
        "use_carw_srr": False,
        "paper_label": "FedARM w/o Aux. Stabilization",
    },
    {
        "variant_id": "A4",
        "name": "pcs_carw_srr",
        "bits": "011",
        "use_cpsa": False,
        "use_pcs": True,
        "use_carw_srr": True,
        "paper_label": "FedARM w/o RUMF",
    },
    {
        "variant_id": "A5",
        "name": "full",
        "bits": "111",
        "use_cpsa": True,
        "use_pcs": True,
        "use_carw_srr": True,
        "paper_label": "FedARM (Full)",
    },
)
VARIANT_BY_ID = {str(item["variant_id"]): item for item in VARIANTS}
VARIANT_BY_NAME = {str(item["name"]): item for item in VARIANTS}
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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
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


def _cell_id(variant_id: str, qos: str, density: float, seed: int) -> str:
    return f"{variant_id}_{qos}_d{int(round(float(density) * 100)):02d}_seed{int(seed)}"


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    density = int(round(float(cell["density"]) * 100))
    return (
        Path(output)
        / str(cell["variant_id"])
        / str(cell["qos"])
        / f"d{density:02d}"
        / f"seed{int(cell['seed'])}"
    )


def _variant(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        candidate = dict(value)
        key = str(candidate.get("variant_id", candidate.get("name", "")))
    else:
        key = str(value).strip()
    candidate = VARIANT_BY_ID.get(key) or VARIANT_BY_NAME.get(key)
    if candidate is None:
        valid = ", ".join(item["variant_id"] for item in VARIANTS)
        raise ValueError(f"unknown RQ2 variant {value!r}; expected {valid}")
    return dict(candidate)


def build_cells(
    densities: Sequence[float] = FORMAL_DENSITIES,
    seeds: Sequence[int] = FORMAL_SEEDS,
    qos_types: Sequence[str] = FORMAL_QOS,
    variants: Sequence[str | Mapping[str, Any]] = tuple(item["variant_id"] for item in VARIANTS),
) -> list[dict[str, Any]]:
                                                             

    cells: list[dict[str, Any]] = []
    for variant_value in variants:
        spec = _variant(variant_value)
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
                            "cell_id": _cell_id(spec["variant_id"], qos, density, int(seed)),
                            "variant_id": spec["variant_id"],
                            "variant": spec["name"],
                            "variant_bits": spec["bits"],
                            "paper_label": spec["paper_label"],
                            "qos": qos,
                            "density": density,
                            "seed": int(seed),
                            "use_cpsa": bool(spec["use_cpsa"]),
                            "use_pcs": bool(spec["use_pcs"]),
                            "use_carw_srr": bool(spec["use_carw_srr"]),
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
    for got, frozen, name in zip(supplied, expected, ("densities", "seeds", "qos", "slices")):
        if got is not None and got != tuple(frozen) if name != "slices" else got is not None and got != frozen:
            raise ValueError(
                f"RQ2 {'smoke' if smoke else 'formal'} {name} are frozen; "
                f"requested {got!r}, expected {frozen!r}"
            )
    return tuple(expected[0]), tuple(expected[1]), tuple(expected[2]), int(expected[3])


def _load_config(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config.get("shape_time_user_service", ())) != SHAPE:
        raise ValueError("RQ2 GPU runner requires WSDream Dataset 3 shape 64x142x4500")
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
        "gpu_model": ROOT / "code" / "core" / "gpu_ablation_training.py",
        "protocol": ROOT / "code" / "core" / "protocol.py",
        "accountant": ROOT / "code" / "core" / "sdq8_attribution_controls.py",
        "multicast_hook": ROOT / "code" / "core" / "request_aggregated_multicast.py",
        "config": Path(config_path).resolve(),
    }
    return {name: _sha256(path) for name, path in files.items()}


def _new_model(config: Mapping[str, Any], seed: int, variant: Mapping[str, Any]) -> Any:
    from core.gpu_ablation_training import GPUAblationModel

    return GPUAblationModel(
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
        variant=str(variant["name"]),
    )


def _new_accountant(config: Mapping[str, Any]) -> Any:
    from core.hmp_sr16_codec import PRECISION_INT8
    from core.sdq8_attribution_controls import FixedLinkPrecisionCommunicationAccountant

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


def _model_cuda_available() -> bool:
    from core.gpu_ablation_training import cuda_available

    return bool(cuda_available())


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    property_value = getattr(value, "as_dict", None)
    if callable(property_value):
        return dict(property_value())
    if isinstance(property_value, Mapping):
        return dict(property_value)
    raise TypeError(f"expected mapping or as_dict report, got {type(value).__name__}")


def _profile_dict(profile: Any) -> dict[str, Any]:
    if profile is None:
        raise AssertionError("GPU training profile missing")
    if hasattr(profile, "as_dict") and callable(profile.as_dict):
        raw = dict(profile.as_dict())
    elif isinstance(profile, Mapping):
        raw = dict(profile)
    else:
        raw = {name: getattr(profile, name) for name in PROFILE_FIELDS if hasattr(profile, name)}
                                                                           
                                                  
    if "aggregation_seconds" not in raw and "pcs_aggregation_seconds" in raw:
        raw["aggregation_seconds"] = raw["pcs_aggregation_seconds"]
    return raw


def _state_hash(model: Any) -> str:
    try:
        from core.gpu_ablation_training import state_hash

        return str(state_hash(model))
    except Exception:
        snapshot = model.full_state_snapshot()
        digest = hashlib.sha256()
        for name, raw in sorted(snapshot.items()):
            value = np.asarray(raw)
            digest.update(str(name).encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def _completed(
    cell_dir: Path,
    cell: Mapping[str, Any],
    slices: int,
    source_sha256: Mapping[str, str] | None = None,
    config_sha256: str | None = None,
    data_sha256: str | None = None,
) -> bool:
    summary = _read_json(Path(cell_dir) / "summary.json")
    if summary is None:
        return False
    if summary.get("status") != "COMPLETED":
        return False
    for key in ("cell_id", "variant_id", "variant", "variant_bits", "qos", "seed"):
        if str(summary.get(key)) != str(cell.get(key)):
            return False
    if not np.isclose(float(summary.get("density", -1.0)), float(cell["density"])):
        return False
    if int(summary.get("time_slices_completed", -1)) != int(slices):
        return False
    if not bool(summary.get("finite_metrics_all_slices")):
        return False
    if bool(summary.get("query_or_future_used")) or bool(summary.get("query_or_future_used_for_update")):
        return False
    mask_audit = summary.get("mask_audit")
    if not isinstance(mask_audit, Mapping):
        return False
    if int(mask_audit.get("count", -1)) != int(slices):
        return False
    if int(mask_audit.get("unique_count", -1)) != int(slices):
        return False
    for metric in METRICS:
        value = summary.get("metrics", {}).get("pooled", {}).get(metric)
        if value is None or not np.isfinite(float(value)):
            return False
    if source_sha256 is not None and dict(summary.get("source_sha256", {})) != dict(source_sha256):
        return False
    if config_sha256 is not None and str(summary.get("config_sha256")) != str(config_sha256):
        return False
    if data_sha256 is not None and str(summary.get("data_sha256")) != str(data_sha256):
        return False
    return True


def _next_attempt(cell_dir: Path) -> Path:
    attempts = Path(cell_dir) / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for path in attempts.glob("attempt_*"):
        try:
            indices.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    attempt = attempts / f"attempt_{max(indices, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _summarize_metrics(rows: Sequence[Mapping[str, Any]], relative_chunks: Sequence[np.ndarray]) -> dict[str, Any]:
    if not rows or not relative_chunks:
        raise ValueError("cannot summarize an empty cell")
    count = sum(int(row["query_entries"]) for row in rows)
    absolute = sum(float(row["absolute_error_sum"]) for row in rows)
    squared = sum(float(row["squared_error_sum"]) for row in rows)
    pooled = np.concatenate([np.asarray(chunk, dtype=np.float64) for chunk in relative_chunks])
    if count <= 0 or pooled.size != count or not np.all(np.isfinite(pooled)):
        raise FloatingPointError("invalid metric accumulation")
    pooled.sort()
    time_mean = {
        name: float(np.mean([float(row[name]) for row in rows], dtype=np.float64))
        for name in METRICS
    }
    return {
        "pooled": {
            "mae": float(absolute / count),
            "rmse": float(np.sqrt(squared / count)),
            "mre": float(np.median(pooled)),
            "npre": float(pooled[min(int(0.9 * count), count - 1)]),
            "count": int(count),
        },
        "time_mean": time_mean,
        "definitions": {
            "mre_per_slice": "median(sorted(abs(pred-truth)/truth))",
            "npre_per_slice": "sorted_relative[min(int(0.9*n), n-1)]",
            "time_mean": "unweighted arithmetic mean over completed time slices",
            "pooled": "computed over all query observations concatenated across slices",
        },
    }


def _delivery_projection(
    support_users: np.ndarray,
    support_services: np.ndarray,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
                                                                      

                                                                         
                                                                           
                                                                             
               
       

    try:
        from core.request_aggregated_multicast import plan_delivery

        plan = plan_delivery(
            support_users,
            support_services,
            num_users=SHAPE[1],
            num_services=SHAPE[2],
            num_edges=int(config["num_edges"]),
            rank=int(config["rank"]),
        )
    except Exception as exc:
        return {
            "hook_status": "UNAVAILABLE",
            "hook_error": f"{type(exc).__name__}: {exc}",
            "UNCAST": None,
            "NAIVE_MULTICAST": None,
            "RAMD": None,
        }
    base_delivery = int(report.get("edge_client_touched_row_downlink_bytes", 0))
    base_total = int(report.get("edge_cache_total_bytes", report.get("flat_total_bytes", 0)))
    values = {
        "UNCAST": int(plan.unicast_bytes),
        "NAIVE_MULTICAST": int(plan.always_multicast_bytes),
        "RAMD": int(plan.racm_bytes),
    }
    projections = {
        strategy: {
            "delivery_downlink_bytes": int(delivery),
            "edge_cache_total_bytes": int(base_total - base_delivery + delivery),
            "edge_cache_steady_state_total_bytes": int(
                base_total - base_delivery + delivery
                - int(report.get("cloud_edge_bootstrap_bytes", 0))
            ),
            "request_count": int(plan.request_count),
            "delivery_group_count": int(plan.group_count),
            "multicast_groups": int(plan.multicast_groups),
            "singleton_groups": int(plan.singleton_groups),
        }
        for strategy, delivery in values.items()
    }
    return {"hook_status": "AVAILABLE", "hook_error": "", **projections}


def _ledger_rows_from_summary(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    totals = summary.get("communication_strategy_totals")
    if not isinstance(totals, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for strategy in ("UNCAST", "NAIVE_MULTICAST", "RAMD"):
        value = totals.get(strategy)
        if not isinstance(value, Mapping):
            continue
        rows.append(
            {
                "protocol": str(summary.get("protocol", PROTOCOL)),
                "cell_id": str(summary.get("cell_id", "")),
                "variant_id": str(summary.get("variant_id", "A5")),
                "variant": str(summary.get("variant", "full")),
                "qos": str(summary.get("qos", "")),
                "density": float(summary.get("density", 0.0)),
                "seed": int(summary.get("seed", 0)),
                "strategy": strategy,
                "time_slices": int(summary.get("time_slices_completed", 0)),
                "delivery_downlink_bytes": int(value.get("delivery_downlink_bytes", 0)),
                "edge_cache_total_bytes": int(value.get("edge_cache_total_bytes", 0)),
                "edge_cache_steady_state_total_bytes": int(value.get("edge_cache_steady_state_total_bytes", 0)),
                "delivery_group_count": int(value.get("delivery_group_count", 0)),
                "multicast_groups": int(value.get("multicast_groups", 0)),
                "singleton_groups": int(value.get("singleton_groups", 0)),
                "hook_status": str(value.get("hook_status", "")),
                "source_sha256_runner": str(summary.get("source_sha256", {}).get("runner", "")),
                "config_sha256": str(summary.get("config_sha256", "")),
                "data_sha256": str(summary.get("data_sha256", "")),
            }
        )
    return rows


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
    model_factory: Callable[[Mapping[str, Any], int, Mapping[str, Any]], Any] | None = None,
    accountant_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
                                                                   

    cell_dir = _cell_dir(Path(output), cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if _completed(cell_dir, cell, slices, source_sha256, config_sha256, data_sha256):
        if resume:
            summary = _read_json(cell_dir / "summary.json") or {}
            return {
                "status": "SKIPPED_COMPLETED",
                "summary": summary,
                "ledger_rows": _ledger_rows_from_summary(summary),
            }
        raise FileExistsError(f"completed cell already exists; use --resume: {cell_dir}")
    attempts_dir = cell_dir / "attempts"
    has_attempts = attempts_dir.exists() and any(attempts_dir.glob("attempt_*"))
    if has_attempts and not resume:
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
        "gpu_only_model_arithmetic": True,
        "cpu_reference_model_constructed": False,
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)

    process = psutil.Process(os.getpid()) if psutil is not None else None
    start_cpu = process.cpu_times() if process is not None else None
    peak_rss = int(process.memory_info().rss) if process is not None else 0
    rows: list[dict[str, Any]] = []
    relative_chunks: list[np.ndarray] = []
    mask_hashes: list[str] = []
    strategy_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "delivery_downlink_bytes": 0,
            "edge_cache_total_bytes": 0,
            "edge_cache_steady_state_total_bytes": 0,
            "delivery_group_count": 0,
            "multicast_groups": 0,
            "singleton_groups": 0,
        }
    )
    hook_status = "NOT_REQUESTED"
    hook_error = ""
    model_factory = model_factory or _new_model
    accountant_factory = accountant_factory or _new_accountant
    variant = _variant(str(cell["variant_id"]))
    model: Any = None
    started_cell = time.perf_counter()
    try:
        model = model_factory(config, int(cell["seed"]), variant)
        accountant = accountant_factory(config)
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
            from core.gpu_ablation_training import prediction_metrics

            metrics = prediction_metrics(split.query_values, prediction)
            truth = np.asarray(split.query_values, dtype=np.float64)
            estimate = np.asarray(prediction, dtype=np.float64)
            relative = np.abs(estimate - truth) / truth
            if not np.all(np.isfinite(relative)):
                raise FloatingPointError("non-finite relative error")
            relative_chunks.append(relative)
            metric_seconds = time.perf_counter() - metric_started

            accounting_started = time.perf_counter()
            communication = _as_dict(accountant.account_round(
                split.support_users,
                split.support_services,
                result,
            ))
            accounting_seconds = time.perf_counter() - accounting_started
            missing_comm = [name for name in COMMUNICATION_FIELDS if name not in communication]
            if missing_comm:
                raise KeyError(f"accountant report missing fields: {missing_comm}")
            profile = _profile_dict(getattr(model, "last_profile", None))
            if not all(np.isfinite(float(metrics[name])) for name in METRICS):
                raise FloatingPointError("non-finite RQ2 metric")
            mask_hashes.append(str(split.mask_sha256))
            if str(cell["variant_id"]) == "A5":
                projection = _delivery_projection(
                    split.support_users,
                    split.support_services,
                    communication,
                    config,
                )
                hook_status = str(projection.get("hook_status", "UNAVAILABLE"))
                hook_error = str(projection.get("hook_error", ""))
                for strategy in ("UNCAST", "NAIVE_MULTICAST", "RAMD"):
                    value = projection.get(strategy)
                    if not isinstance(value, Mapping):
                        continue
                    for key in strategy_totals[strategy]:
                        strategy_totals[strategy][key] += int(value.get(key, 0))

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
                "communication_accounting_seconds": float(accounting_seconds),
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
                "query_or_future_used": False,
                "query_or_future_used_for_update": False,
                "gpu_profile_json": json.dumps(profile, sort_keys=True),
                "communication_json": json.dumps(communication, sort_keys=True),
            }
            for name in COMMUNICATION_FIELDS:
                row[name] = int(communication[name])
            for name in PROFILE_FIELDS:
                if name in profile:
                    row[f"gpu_{name}"] = float(profile[name])
            rows.append(row)
            _atomic_csv(attempt / "per_slice.csv", rows)
            status["time_slices_completed"] = int(time_slice + 1)
            status["last_mask_sha256"] = str(split.mask_sha256)
            status["updated_at"] = _utc_now()
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)

        digest = hashlib.sha256()
        for value in mask_hashes:
            digest.update(bytes.fromhex(value))
        metrics_summary = _summarize_metrics(rows, relative_chunks)
        end_cpu = process.cpu_times() if process is not None else None
        communication_totals = {
            name: int(sum(int(row[name]) for row in rows)) for name in COMMUNICATION_FIELDS
        }
        strategies: dict[str, Any] = {}
        if str(cell["variant_id"]) == "A5":
            for strategy in ("UNCAST", "NAIVE_MULTICAST", "RAMD"):
                strategies[strategy] = {
                    **strategy_totals[strategy],
                    "hook_status": hook_status,
                    "hook_error": hook_error,
                }
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
            "gpu_only_model_arithmetic": True,
            "cpu_reference_model_constructed": False,
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
            "communication_strategy_totals": strategies,
            "timing_seconds": {
                name: float(sum(float(row.get(name, 0.0)) for row in rows))
                for name in (
                    "split_seconds", "update_wall_seconds", "predict_wall_seconds",
                    "metric_seconds", "communication_accounting_seconds", "slice_wall_seconds",
                )
            },
            "cell_wall_seconds": float(time.perf_counter() - started_cell),
            "gpu_peak_memory_used_bytes": int(max((int(row.get("gpu_peak_memory_used_bytes", 0)) for row in rows), default=0)),
            "process_peak_rss_bytes": int(peak_rss),
            "process_cpu_user_seconds": float(end_cpu.user - start_cpu.user) if end_cpu is not None and start_cpu is not None else 0.0,
            "process_cpu_system_seconds": float(end_cpu.system - start_cpu.system) if end_cpu is not None and start_cpu is not None else 0.0,
            "final_state_sha256": _state_hash(model),
            "attempt_dir": str(attempt),
            "completed_at": _utc_now(),
        }
        if str(cell["variant_id"]) == "A5":
            _atomic_csv(attempt / "communication_ledger.csv", _ledger_rows_from_summary(summary))
        _atomic_json(attempt / "summary.json", summary)
        _atomic_json(attempt / "status.json", summary)
        _atomic_json(cell_dir / "summary.json", summary)
        _atomic_json(cell_dir / "status.json", summary)
        return {"status": "COMPLETED", "summary": summary, "ledger_rows": _ledger_rows_from_summary(summary)}
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


def _summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    pooled = metrics.get("pooled", {}) if isinstance(metrics, Mapping) else {}
    time_mean = metrics.get("time_mean", {}) if isinstance(metrics, Mapping) else {}
    mask_audit = summary.get("mask_audit", {})
    source = summary.get("source_sha256", {})
    row: dict[str, Any] = {
        "protocol": str(summary.get("protocol", PROTOCOL)),
        "execution_mode": str(summary.get("execution_mode", EXECUTION_MODE)),
        "cell_id": str(summary.get("cell_id", "")),
        "variant_id": str(summary.get("variant_id", "")),
        "variant": str(summary.get("variant", "")),
        "variant_bits": str(summary.get("variant_bits", "")),
        "paper_label": str(summary.get("paper_label", "")),
        "qos": str(summary.get("qos", "")),
        "density": float(summary.get("density", 0.0)),
        "seed": int(summary.get("seed", 0)),
        "time_slices_completed": int(summary.get("time_slices_completed", 0)),
        "finite_metrics_all_slices": bool(summary.get("finite_metrics_all_slices", False)),
        "query_or_future_used": bool(summary.get("query_or_future_used", True)),
        "query_or_future_used_for_update": bool(summary.get("query_or_future_used_for_update", True)),
        "query_entries": int(pooled.get("count", 0)),
        "mask_count": int(mask_audit.get("count", 0)),
        "mask_unique_count": int(mask_audit.get("unique_count", 0)),
        "mask_stream_sha256": str(mask_audit.get("mask_stream_sha256", "")),
        "config_sha256": str(summary.get("config_sha256", "")),
        "data_sha256": str(summary.get("data_sha256", "")),
        "source_sha256_runner": str(source.get("runner", "")) if isinstance(source, Mapping) else "",
        "source_sha256_gpu_model": str(source.get("gpu_model", "")) if isinstance(source, Mapping) else "",
    }
    for metric in METRICS:
        row[f"pooled_{metric}"] = float(pooled.get(metric, float("nan")))
        row[f"time_mean_{metric}"] = float(time_mean.get(metric, float("nan")))
    communication = summary.get("communication_totals", {})
    if isinstance(communication, Mapping):
        for name in COMMUNICATION_FIELDS:
            row[f"communication_{name}"] = int(communication.get(name, 0))
    return row


def _aggregate_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[(str(row["variant_id"]), str(row["qos"]), float(row["density"]))].append(row)
    result: list[dict[str, Any]] = []
    for (variant_id, qos, density), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        spec = VARIANT_BY_ID[variant_id]
        aggregate: dict[str, Any] = {
            "protocol": PROTOCOL,
            "execution_mode": EXECUTION_MODE,
            "variant_id": variant_id,
            "variant": spec["name"],
            "variant_bits": spec["bits"],
            "paper_label": spec["paper_label"],
            "qos": qos,
            "density": density,
            "n_seeds": len(rows),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda value: int(value["seed"]))),
        }
        for metric in METRICS:
            values = np.asarray([float(row[f"pooled_{metric}"]) for row in rows], dtype=np.float64)
            aggregate[f"pooled_{metric}_mean"] = float(np.mean(values))
            aggregate[f"pooled_{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            values_time = np.asarray([float(row[f"time_mean_{metric}"]) for row in rows], dtype=np.float64)
            aggregate[f"time_mean_{metric}_mean"] = float(np.mean(values_time))
            aggregate[f"time_mean_{metric}_std"] = float(np.std(values_time, ddof=1)) if values_time.size > 1 else 0.0
        result.append(aggregate)
    return result


def _batch_gate(
    cells: Sequence[Mapping[str, Any]],
    output: Path,
    slices: int,
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: Mapping[str, str],
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    summaries: list[dict[str, Any]] = []
    invalid: list[str] = []
    for cell in cells:
        summary = _read_json(_cell_dir(output, cell) / "summary.json")
        qos_hash = str(data_sha256.get(str(cell["qos"]), ""))
        if summary is None or not _completed(
            _cell_dir(output, cell), cell, slices, source_sha256, config_sha256, qos_hash
        ):
            invalid.append(str(cell["cell_id"]))
            continue
        summaries.append(summary)
    valid = len(invalid) == 0 and len(summaries) == len(cells)
    return valid, summaries, invalid


def _validate_existing_manifest(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    immutable = (
        "protocol", "execution_mode", "scope", "shape", "densities", "seeds",
        "qos_types", "time_slices", "cell_count", "variant_ids", "variants",
    )
    for key in immutable:
        if existing.get(key) != current.get(key):
            raise ValueError(f"existing RQ2 manifest mismatch at {key}; start a new output")
    if existing.get("source_sha256") != current.get("source_sha256"):
        raise ValueError("existing RQ2 manifest source hash mismatch; do not resume across source changes")
    if existing.get("config_sha256") != current.get("config_sha256"):
        raise ValueError("existing RQ2 manifest config hash mismatch; do not resume across config changes")


def run_queue(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path | None = None,
    smoke: bool = False,
    densities: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    qos_types: Sequence[str] | None = None,
    slices: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
    model_factory: Callable[[Mapping[str, Any], int, Mapping[str, Any]], Any] | None = None,
    accountant_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
                                                     

    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    densities, seeds, qos_types, slices = _scope(smoke, densities, seeds, qos_types, slices)
    output = Path(output or (DEFAULT_SMOKE_OUTPUT if smoke else DEFAULT_FORMAL_OUTPUT)).resolve()
    cells = build_cells(densities, seeds, qos_types)
    source_sha256 = _source_hashes(config_path)
    config_sha256 = source_sha256["config"]
    variant_ids = [str(item["variant_id"]) for item in VARIANTS]
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
        "variant_ids": variant_ids,
        "variants": [dict(item) for item in VARIANTS],
        "cell_count": len(cells),
        "cells": cells,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "data_sha256": {},
        "query_or_future_used": False,
        "gpu_only_model_arithmetic": True,
        "cpu_reference_model_constructed": False,
        "backend": {},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "communication_ledger": {
            "variant": "A5",
            "strategies": ["UNCAST", "NAIVE_MULTICAST", "RAMD"],
            "expected_rows": len(densities) * len(seeds) * len(qos_types) * 3,
            "hook_is_accounting_only": True,
        },
    }
    if dry_run:
        return {"status": "DRY_RUN", "manifest": manifest, "cells": cells}
    if not _model_cuda_available() and model_factory is None:
        raise RuntimeError("a usable Numba CUDA device is required for RQ2 accuracy ablation")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = _read_json(manifest_path)
    if existing is not None:
        if not resume:
            raise FileExistsError(f"formal/smoke output already initialized; use --resume: {output}")
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
            raise ValueError("existing RQ2 manifest data hash mismatch; do not resume")
    manifest["data_sha256"] = data_sha256
    try:
        from core.gpu_ablation_training import cuda_device_info

        manifest["backend"] = cuda_device_info()
    except Exception as exc:
        manifest["backend"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    _atomic_json(manifest_path, manifest)

    queue_rows: list[dict[str, Any]] = [
        {"cell_index": index, **dict(cell), "status": "PENDING", "error": ""}
        for index, cell in enumerate(cells)
    ]
    queue_path = output / "queue_cells.csv"
    queue_started = time.perf_counter()
    completed_now = skipped = failed = 0

    def write_queue() -> None:
        _atomic_csv(queue_path, queue_rows)

    write_queue()
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
                model_factory=model_factory,
                accountant_factory=accountant_factory,
            )
            queue_rows[index]["status"] = str(result["status"])
            queue_rows[index]["attempt_dir"] = str(result.get("summary", {}).get("attempt_dir", ""))
            if result["status"] == "SKIPPED_COMPLETED":
                skipped += 1
            else:
                completed_now += 1
        except Exception as exc:
            failed += 1
            queue_rows[index]["status"] = "FAILED"
            queue_rows[index]["error"] = f"{type(exc).__name__}: {exc}"
            if stop_on_error:
                write_queue()
                raise
        write_queue()
        _atomic_json(output / "queue_status.json", {
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
        })

    valid, summaries, invalid = _batch_gate(
        cells, output, slices, source_sha256, config_sha256, data_sha256
    )
    summary_rows = [_summary_row(value) for value in summaries]
    _atomic_csv(output / "final_summary.csv", summary_rows)
    _atomic_csv(output / "aggregate_mean_std.csv", _aggregate_rows(summary_rows))
    ledger_rows: list[dict[str, Any]] = []
    for summary in summaries:
        ledger_rows.extend(_ledger_rows_from_summary(summary))
    if ledger_rows:
        _atomic_csv(output / "communication_ledger.csv", ledger_rows)
    elif (output / "communication_ledger.csv").exists():
        (output / "communication_ledger.csv").unlink()

    expected_ledger_rows = len(densities) * len(seeds) * len(qos_types) * 3
    ledger_complete = len(ledger_rows) == expected_ledger_rows
    terminal_valid = valid and ledger_complete
    if terminal_valid:
        marker = {
            "status": "COMPLETED" if smoke else "BATCH_COMPLETED",
            "protocol": PROTOCOL,
            "scope": "smoke" if smoke else "formal",
            "cells_expected": len(cells),
            "cells_completed": len(summaries),
            "time_slices": int(slices),
            "final_summary_sha256": _sha256(output / "final_summary.csv"),
            "aggregate_mean_std_sha256": _sha256(output / "aggregate_mean_std.csv"),
            "communication_ledger_rows": len(ledger_rows),
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
        "communication_ledger_rows": len(ledger_rows),
        "communication_ledger_rows_expected": expected_ledger_rows,
        "communication_ledger_complete": ledger_complete,
        "invalid_cells": invalid,
        "elapsed_seconds": float(time.perf_counter() - queue_started),
        "completed_at": _utc_now(),
    }
    _atomic_json(output / "queue_status.json", final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="run the explicit D10/seed42/RT/1-slice smoke scope")
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

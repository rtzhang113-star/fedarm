                                           

                                                                         
                                                                            
                                                                          
                                                                          
                                                                         

                                                                        
                                                                          
                                                                            
   

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import types

import numpy as np

try:                                                                  
    import psutil
except Exception:                                                 
    psutil = None                            


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_protocol_api() -> Any:
                                                                         

    try:
        return importlib.import_module(".protocol", package=__package__)
    except (ImportError, TypeError, ValueError):
        try:
            return importlib.import_module("protocol")
        except ImportError:
            return None


_PROTOCOL_API = _load_protocol_api()
ROOT = Path(getattr(_PROTOCOL_API, "ROOT", REPO_ROOT)).resolve()
DEFAULT_PROTOCOL = Path(
    getattr(_PROTOCOL_API, "DEFAULT_PROTOCOL", HERE / "rq3_robustness_protocol.json")
).resolve()
load_protocol = getattr(_PROTOCOL_API, "load_protocol", None)
build_cells = getattr(_PROTOCOL_API, "build_cells", None)
scope_slices = getattr(_PROTOCOL_API, "scope_slices", None)
intervene_support = getattr(_PROTOCOL_API, "intervene_support", None)

                                                                     
                                                                             
                                                                 
if importlib.util.find_spec("torch") is None and "continual_qos" not in sys.modules:
    _continual_qos_stub = types.ModuleType("continual_qos")
    _continual_qos_stub.__path__ = [str(ROOT / "continual_qos")]                              
    sys.modules["continual_qos"] = _continual_qos_stub


METRICS = ("mae", "rmse", "mre", "npre")
DEFAULT_PROTOCOL_ID = "fedarm_rq3_robustness_v1"
                                                                           
                                                                        
                                                  
SUPPORT_GROUPS = ("support_0", "support_1", "support_2_4", "support_ge5")
PARTICIPATION_GROUPS = ("selected_clients", "not_selected_clients")
GROUP_KEYS = (("all", "all"),) + tuple(
    ("support", group) for group in SUPPORT_GROUPS
) + tuple(("participation", group) for group in PARTICIPATION_GROUPS)
TERMINAL_STATUS = "BATCH_COMPLETED"
ACCEPTED_CELL_STATUSES = {"COMPLETED", "SKIPPED_COMPLETED"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _jsonable(value: Any) -> Any:
                                                                             

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row:
                if str(key) not in names:
                    names.append(str(key))
    else:
        names = [str(item) for item in fieldnames]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return _sha256_bytes(
        str(array.dtype).encode("ascii"),
        np.asarray(array.shape, dtype=np.int64).tobytes(),
        array.tobytes(),
    )


def _hash_indices(values: Sequence[int] | np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return _hash_array(array)


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))
    return text.strip("._") or "unknown"


def density_code(density: float) -> str:
    return f"d{int(round(float(density) * 100)):02d}"


def _require_protocol_api() -> None:
    if any(item is None for item in (load_protocol, build_cells, scope_slices, intervene_support)):
        raise ImportError(
            "code/rq3/robustness/protocol.py must provide load_protocol, build_cells, "
            "scope_slices, and intervene_support"
        )


def _model_api() -> Any:
    try:
        return importlib.import_module(".model", package=__package__)
    except (ImportError, TypeError, ValueError):
        try:
            return importlib.import_module("model")
        except ImportError as exc:
            raise ImportError("code/rq3/robustness/model.py is unavailable") from exc


def _dataset_spec(protocol: Mapping[str, Any]) -> tuple[tuple[int, int, int], Path]:
    dataset = protocol.get("dataset")
    dataset = dataset if isinstance(dataset, Mapping) else {}
    raw_shape = dataset.get(
        "shape_time_user_service",
        protocol.get("shape_time_user_service", (64, 142, 4500)),
    )
    shape = tuple(int(item) for item in raw_shape)
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise ValueError(f"invalid RQ3 dataset shape: {raw_shape!r}")
    raw_cache = dataset.get("cache_dir", protocol.get("cache_dir", "data/cache"))
    cache_dir = Path(str(raw_cache))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return (shape[0], shape[1], shape[2]), cache_dir.resolve()


def _config_path(protocol: Mapping[str, Any]) -> Path:
    raw = protocol.get("base_config")
    if isinstance(raw, Mapping):
        raw = raw.get("path")
    if not raw:
        raise ValueError("RQ3 protocol must specify base_config")
    path = Path(str(raw))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_config(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = _config_path(protocol)
    value = _read_json(path)
    if value is None:
        raise ValueError(f"invalid base config JSON: {path}")
    shape, _ = _dataset_spec(protocol)
    if tuple(value.get("shape_time_user_service", ())) != shape:
        raise ValueError("RQ3 base config shape differs from protocol dataset shape")
    required = (
        "rank",
        "bias_eta",
        "factor_eta",
        "l2",
        "beta",
        "factor_init_std",
        "num_edges",
        "bias_epochs",
        "factor_epochs",
        "raw_prediction_caps",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"RQ3 base config missing fields: {missing}")
    return value, path


def _qos_types(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    matrix = protocol.get("matrix")
    matrix = matrix if isinstance(matrix, Mapping) else {}
    raw = matrix.get("qos_types", protocol.get("qos_types", ("rt", "tp")))
    result = tuple(str(item).lower() for item in raw)
    if not result or any(item not in {"rt", "tp"} for item in result):
        raise ValueError(f"invalid RQ3 QoS types: {raw!r}")
    return result


def _source_hashes(protocol: Mapping[str, Any], config_path: Path) -> dict[str, str]:
    files: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "rq3_model": HERE / "model.py",
        "rq3_protocol": Path(getattr(_PROTOCOL_API, "__file__", HERE / "protocol.py")).resolve(),
                                                                        
                                                                             
                                         
        "original_gpu_model": (ROOT / "continual_qos" / "gpu_ablation_training.py").resolve(),
        "reference_full_gpu_model": (ROOT / "continual_qos" / "gpu_full_training.py").resolve(),
        "causal_protocol": (ROOT / "continual_qos" / "protocol.py").resolve(),
        "data_loader": (ROOT / "continual_qos" / "data.py").resolve(),
        "base_config": Path(config_path).resolve(),
    }
                                                                             
                                                                              
    result: dict[str, str] = {"protocol_content": hashlib.sha256(
        json.dumps(_jsonable(protocol), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()}
    for name, path in files.items():
        if path.is_file():
            result[name] = sha256_file(path)
        else:
            raise FileNotFoundError(f"source file for hash is missing: {path}")
    return result


def _load_caches(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    shape: tuple[int, int, int],
    qos_types: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    _, cache_dir = _dataset_spec(protocol)
    try:
        from continual_qos.data import load_value_cache
    except ImportError as exc:                                                  
        raise ImportError("continual_qos.data.load_value_cache is required") from exc
    cache: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for qos in qos_types:
        path = cache_dir / f"{qos}_values.npy"
        cache[qos] = load_value_cache(path, shape)
        hashes[qos] = sha256_file(path)
    return cache, hashes


def _model_kwargs(config: Mapping[str, Any], shape: tuple[int, int, int], cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "num_users": int(shape[1]),
        "num_services": int(shape[2]),
        "rank": int(config["rank"]),
        "seed": int(cell["seed"]),
        "bias_eta": float(config["bias_eta"]),
        "factor_eta": float(config["factor_eta"]),
        "l2": float(config["l2"]),
        "beta": float(config["beta"]),
        "factor_init_std": float(config["factor_init_std"]),
        "num_edges": int(config["num_edges"]),
        "variant": str(cell["variant"]),
    }


def _new_model(config: Mapping[str, Any], shape: tuple[int, int, int], cell: Mapping[str, Any]) -> Any:
    model_api = _model_api()
    constructor = getattr(model_api, "GPURobustnessModel", None)
    if constructor is None:
        raise AttributeError("RQ3 model.py must export GPURobustnessModel")
    return constructor(**_model_kwargs(config, shape, cell))


def _raw_cap(config: Mapping[str, Any], qos: str) -> float:
    caps = config.get("raw_prediction_caps")
    if not isinstance(caps, Mapping) or qos not in caps:
        raise KeyError(f"raw_prediction_caps lacks {qos}")
    value = float(caps[qos])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid raw prediction cap for {qos}: {value}")
    return value


def _state_hash(model: Any) -> str:
    try:
        model_api = _model_api()
        function = getattr(model_api, "state_hash", None)
        if callable(function):
            return str(function(model, full=True))
    except Exception:
        pass
    snapshot_function = getattr(model, "full_state_snapshot", None)
    if not callable(snapshot_function):
        snapshot_function = getattr(model, "parameter_snapshot", None)
    if not callable(snapshot_function):
        raise AttributeError("model must expose full_state_snapshot or parameter_snapshot")
    snapshot = snapshot_function()
    digest = hashlib.sha256()
    for name, raw in sorted(dict(snapshot).items()):
        value = np.ascontiguousarray(np.asarray(raw))
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


PRIVATE_STATE_KEYS = frozenset(
    {"user_factors", "user_bias", "user_error", "client_service_adapter"}
)


def _private_snapshot(model: Any) -> dict[str, np.ndarray]:
                                                                               

    snapshot_function = getattr(model, "full_state_snapshot", None)
    if not callable(snapshot_function):
        snapshot_function = getattr(model, "parameter_snapshot", None)
    if callable(snapshot_function):
        raw = snapshot_function()
        if isinstance(raw, Mapping):
            selected: dict[str, np.ndarray] = {}
            for name, value in raw.items():
                text = str(name)
                if text in PRIVATE_STATE_KEYS or text.startswith("carw_history_"):
                    selected[text] = np.asarray(value).copy()
            if PRIVATE_STATE_KEYS.issubset(selected):
                return selected
    property_value = getattr(model, "client_service_adapter", None)
    if property_value is not None:
                                                                       
                                                                             
                                                       
        return {"client_service_adapter": np.asarray(property_value).copy()}
    raise AttributeError(
        "model must expose full_state_snapshot/parameter_snapshot with "
        "user_factors, user_bias, user_error, and client_service_adapter"
    )


def _private_state_unchanged(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], *, unobserved: np.ndarray, num_users: int
) -> bool:
                                                                                

    if not PRIVATE_STATE_KEYS.issubset(before) or not PRIVATE_STATE_KEYS.issubset(after):
        return False
    for name in PRIVATE_STATE_KEYS:
        left = np.asarray(before[name])
        right = np.asarray(after[name])
        if left.shape != right.shape or left.ndim == 0 or left.shape[0] != int(num_users):
            return False
        if not np.array_equal(left[unobserved], right[unobserved]):
            return False
    history_users = set()
    for name in set(before) | set(after):
        if not name.startswith("carw_history_"):
            continue
        try:
            history_users.add(int(name.rsplit("_", 1)[1]))
        except ValueError:
            return False
    for user in history_users:
        if user < 0 or user >= int(num_users) or not bool(unobserved[user]):
            continue
        left = before.get(f"carw_history_{user}")
        right = after.get(f"carw_history_{user}")
        if (left is None) != (right is None):
            return False
        if left is not None and not np.array_equal(np.asarray(left), np.asarray(right)):
            return False
    return True


def _profile_dict(model: Any) -> dict[str, Any]:
    profile = getattr(model, "last_profile", None)
    if profile is None:
        return {}
    as_dict = getattr(profile, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    if isinstance(profile, Mapping):
        return dict(profile)
    return {
        str(name): _jsonable(getattr(profile, name))
        for name in dir(profile)
        if not str(name).startswith("_")
        and isinstance(getattr(profile, name), (int, float, np.integer, np.floating))
    }


def _prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    try:
        function = getattr(_model_api(), "prediction_metrics", None)
        if callable(function):
            return dict(function(truth, prediction))
    except Exception:
        pass
    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if target.shape != estimate.shape or target.size == 0:
        raise ValueError("truth and prediction must be non-empty equal-shaped arrays")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(estimate)):
        raise FloatingPointError("metrics require finite truth and prediction")
    if np.any(target <= 0.0):
        raise ValueError("MRE/NPRE require strictly positive truth")
    error = estimate - target
    relative = np.sort(np.abs(error) / target)
    count = int(target.size)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mre": float(np.median(relative)),
        "npre": float(relative[min(int(0.9 * count), count - 1)]),
        "absolute_error_sum": float(np.sum(np.abs(error), dtype=np.float64)),
        "squared_error_sum": float(np.sum(np.square(error), dtype=np.float64)),
        "count": count,
    }


def _metric_values(truth: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    metrics = _prediction_metrics(truth, prediction)
    relative = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) / np.asarray(
        truth, dtype=np.float64
    )
    if not np.all(np.isfinite(relative)):
        raise FloatingPointError("non-finite relative error")
    return metrics, relative


def _empty_accumulator() -> dict[str, Any]:
    return {"count": 0, "absolute_error_sum": 0.0, "squared_error_sum": 0.0, "relative_chunks": []}


def _add_accumulator(accumulator: dict[str, Any], metrics: Mapping[str, Any], relative: np.ndarray) -> None:
    accumulator["count"] += int(metrics.get("count", relative.size))
    accumulator["absolute_error_sum"] += float(metrics.get("absolute_error_sum", 0.0))
    accumulator["squared_error_sum"] += float(metrics.get("squared_error_sum", 0.0))
    if relative.size:
        accumulator["relative_chunks"].append(np.asarray(relative, dtype=np.float64).copy())


def _summarize_accumulator(accumulator: Mapping[str, Any], *, time_mean: Mapping[str, float] | None = None) -> dict[str, Any]:
    count = int(accumulator.get("count", 0))
    chunks = [np.asarray(item, dtype=np.float64) for item in accumulator.get("relative_chunks", [])]
    if count <= 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "mre": None,
            "npre": None,
            "time_mean": None if time_mean is None else dict(time_mean),
        }
    relative = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    if relative.size != count or not np.all(np.isfinite(relative)):
        raise FloatingPointError("invalid pooled relative-error accumulator")
    relative.sort()
    result: dict[str, Any] = {
        "count": count,
        "mae": float(accumulator["absolute_error_sum"] / count),
        "rmse": float(math.sqrt(accumulator["squared_error_sum"] / count)),
        "mre": float(np.median(relative)),
        "npre": float(relative[min(int(0.9 * count), count - 1)]),
    }
    if time_mean is not None:
        result["time_mean"] = dict(time_mean)
    return result


def _support_group(count: int) -> str:
    count = int(count)
    if count <= 0:
        return "support_0"
    if count == 1:
        return "support_1"
    if count <= 4:
        return "support_2_4"
    return "support_ge5"


def _intervention_details(
    intervention: Any,
    *,
    original_users: np.ndarray,
    original_services: np.ndarray,
    num_users: int,
    num_services: int,
) -> dict[str, Any]:
    indices = np.asarray(getattr(intervention, "indices"), dtype=np.int64)
    selected = np.asarray(getattr(intervention, "selected_clients"), dtype=np.int64)
    observed = np.asarray(getattr(intervention, "observed_clients"), dtype=np.int64)
    service_counts = np.asarray(getattr(intervention, "service_counts"), dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= original_users.size):
        raise ValueError("intervention.indices is outside the original support")
    if indices.size and np.any(np.diff(indices) <= 0):
        raise ValueError("intervention.indices must be a strictly increasing subset")
    if selected.ndim != 1 or observed.ndim != 1 or service_counts.ndim != 1:
        raise ValueError("intervention client/service arrays must be one-dimensional")
    if service_counts.size != int(num_services) or np.any(service_counts < 0):
        raise ValueError("intervention.service_counts has the wrong shape or negative values")
    for name, values in (("selected_clients", selected), ("observed_clients", observed)):
        if np.any(values < 0) or np.any(values >= int(num_users)):
            raise ValueError(f"intervention.{name} contains an invalid client")
        if np.unique(values).size != values.size:
            raise ValueError(f"intervention.{name} contains duplicates")
    effective_users = original_users[indices]
    effective_services = original_services[indices]
    expected_observed = np.unique(effective_users).astype(np.int64, copy=False)
    if not np.array_equal(observed, expected_observed):
        raise ValueError("intervention.observed_clients does not match effective support")
    expected_counts = np.bincount(effective_services, minlength=int(num_services)).astype(np.int64)
    if not np.array_equal(service_counts, expected_counts):
        raise ValueError("intervention.service_counts does not match effective support")
    supplied_audit = getattr(intervention, "audit", {})
    supplied_audit = dict(supplied_audit) if isinstance(supplied_audit, Mapping) else {}
    audit = {
        **supplied_audit,
        "original_support_count": int(original_users.size),
        "effective_support_count": int(indices.size),
        "dropped_support_count": int(original_users.size - indices.size),
        "effective_support_indices_sha256": _hash_indices(indices),
        "selected_clients_sha256": _hash_indices(selected),
        "observed_clients_sha256": _hash_indices(observed),
        "service_counts_sha256": _hash_indices(service_counts),
        "observed_client_count": int(observed.size),
        "selected_client_count": int(selected.size),
        "service_support_gini": float(_gini(service_counts)),
    }
    return {
        "indices": indices,
        "selected_clients": selected,
        "observed_clients": observed,
        "service_counts": service_counts,
        "effective_users": effective_users,
        "effective_services": effective_services,
        "audit": audit,
    }


def _gini(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0 or np.sum(array) <= 0.0:
        return 0.0
    sorted_values = np.sort(array)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * np.sum(sorted_values))) - (n + 1.0) / n)


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    return (
        Path(output)
        / _slug(cell.get("scenario_id", cell.get("scenario", {}).get("id", "scenario")))
        / _slug(cell.get("variant_id", "variant"))
        / _slug(cell.get("qos", "qos"))
        / density_code(float(cell["density"]))
        / f"seed{int(cell['seed'])}"
    )


def _cell_id(cell: Mapping[str, Any]) -> str:
    return str(cell.get("cell_id", ""))


def _cell_manifest_row(cell: Mapping[str, Any], *, status: str = "PLANNED", error: str = "") -> dict[str, Any]:
    scenario = cell.get("scenario", {})
    scenario_id = cell.get("scenario_id")
    if scenario_id is None and isinstance(scenario, Mapping):
        scenario_id = scenario.get("id", "")
    return {
        "record_type": "accuracy_cell",
        "cell_id": _cell_id(cell),
        "scenario_id": str(scenario_id or ""),
        "scenario_json": json.dumps(_jsonable(scenario), ensure_ascii=False, sort_keys=True),
        "variant_id": str(cell.get("variant_id", "")),
        "variant": str(cell.get("variant", "")),
        "qos": str(cell.get("qos", "")),
        "density": float(cell.get("density", 0.0)),
        "seed": int(cell.get("seed", 0)),
        "time_slices": int(cell.get("time_slices", 0)),
        "training_required": True,
        "projection_only": False,
        "status": status,
        "error": error,
    }


def _new_attempt(cell_dir: Path) -> Path:
    attempts_dir = Path(cell_dir) / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in attempts_dir.glob("attempt_*"):
        try:
            numbers.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = attempts_dir / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _cell_artifacts_valid(directory: Path, summary: Mapping[str, Any], slices: int) -> bool:
                                                                                
    try:
        attempt = Path(str(summary["attempt_dir"])).resolve()
        if attempt.parent != (Path(directory) / "attempts").resolve():
            return False
        if _read_json(attempt / "summary.json") != dict(summary):
            return False
        if _read_json(attempt / "status.json") != dict(summary):
            return False
        expected = summary.get("artifact_sha256", {})
        for name in ("per_slice.csv", "group_metrics_per_slice.csv"):
            if expected.get(name) != sha256_file(attempt / name):
                return False
        rows = _read_csv(attempt / "per_slice.csv")
        if [int(row["time_slice"]) for row in rows] != list(range(int(slices))):
            return False
        for row in rows:
            for key in ("cell_id", "scenario_id", "variant_id", "variant", "qos", "seed"):
                if str(row[key]) != str(summary[key]):
                    return False
            if float(row["density"]) != float(summary["density"]):
                return False
            if row["query_state_hash_before"] != row["query_state_hash_after"]:
                return False
            index = str(row["time_slice"])
            if row["original_mask_sha256"] != summary["mask_audit"]["mask_sha256_by_slice"][index]:
                return False
            if row["effective_support_indices_sha256"] != summary["mask_audit"]["effective_support_indices_sha256_by_slice"][index]:
                return False
        groups = _read_csv(attempt / "group_metrics_per_slice.csv")
        expected_keys = {(t, axis, group) for t in range(int(slices)) for axis, group in GROUP_KEYS}
        group_keys = [(int(row["time_slice"]), row["group_axis"], row["group"]) for row in groups]
        if len(group_keys) != len(expected_keys) or set(group_keys) != expected_keys:
            return False
        for row in groups:
            count = int(row["count"])
            if count < 0 or row["cell_id"] != summary["cell_id"]:
                return False
            if count == 0:
                if any(row[name] != "" for name in METRICS):
                    return False
            elif any(not np.isfinite(float(row[name])) for name in METRICS):
                return False
        for t, row in enumerate(rows):
            for axis in ("all", "support", "participation"):
                if sum(int(g["count"]) for g in groups if int(g["time_slice"]) == t and g["group_axis"] == axis) != int(row["query_entries"]):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _completed(
    cell_dir: Path,
    cell: Mapping[str, Any],
    slices: int,
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: Mapping[str, str],
) -> bool:
    summary = _read_json(Path(cell_dir) / "summary.json")
    status = _read_json(Path(cell_dir) / "status.json")
    if summary is None or status is None or summary.get("status") != "COMPLETED" or status.get("status") != "COMPLETED":
        return False
    if not _cell_artifacts_valid(Path(cell_dir), summary, slices):
        return False
    for key in ("cell_id", "scenario_id", "variant_id", "variant", "qos", "seed"):
        if str(summary.get(key, "")) != str(cell.get(key, "")):
            return False
    if not np.isclose(float(summary.get("density", -1.0)), float(cell.get("density", -2.0))):
        return False
    if int(summary.get("time_slices_completed", -1)) != int(slices):
        return False
    if not bool(summary.get("finite_metrics_all_slices")):
        return False
    if summary.get("query_or_future_used") or summary.get("query_or_future_used_for_update"):
        return False
    if summary.get("source_sha256") != dict(source_sha256):
        return False
    if str(summary.get("config_sha256", "")) != str(config_sha256):
        return False
    if summary.get("data_sha256") != dict(data_sha256):
        return False
    if not bool(summary.get("final_state_sha256")):
        return False
    mask_audit = summary.get("mask_audit")
    if not isinstance(mask_audit, Mapping) or int(mask_audit.get("count", -1)) != int(slices):
        return False
    if len(dict(mask_audit.get("effective_support_indices_sha256_by_slice", {}))) != int(slices):
        return False
    pooled = summary.get("metrics", {}).get("pooled", {})
    if not isinstance(pooled, Mapping) or int(pooled.get("count", 0)) <= 0:
        return False
    if any(pooled.get(name) is None or not np.isfinite(float(pooled[name])) for name in METRICS):
        return False
    per_slice = Path(str(summary["attempt_dir"])) / "per_slice.csv"
    if not per_slice.is_file():
        return False
    try:
        rows = _read_csv(per_slice)
    except Exception:
        return False
    if len(rows) != int(slices):
        return False
    for row in rows:
        if any(row.get(name, "") == "" or not np.isfinite(float(row[name])) for name in METRICS):
            return False
        if row.get("query_or_future_used", "true").lower() != "false":
            return False
        if row.get("query_state_unchanged", "false").lower() != "true":
            return False
        if row.get("unobserved_private_state_unchanged", "false").lower() != "true":
            return False
        if not row.get("original_mask_sha256") or not row.get("effective_support_indices_sha256"):
            return False
    return True


def _group_row(
    cell: Mapping[str, Any],
    *,
    time_slice: int,
    axis: str,
    group: str,
    metrics: Mapping[str, Any],
    service_support_count: int | None = None,
) -> dict[str, Any]:
    row = {
        "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
        "cell_id": _cell_id(cell),
        "scenario_id": str(cell.get("scenario_id", "")),
        "variant_id": str(cell.get("variant_id", "")),
        "variant": str(cell.get("variant", "")),
        "qos": str(cell.get("qos", "")),
        "density": float(cell.get("density", 0.0)),
        "seed": int(cell.get("seed", 0)),
        "scope": "per_slice",
        "time_slice": int(time_slice),
        "group_axis": axis,
        "group": group,
        "count": int(metrics.get("count", 0)),
        "service_support_count": service_support_count,
    }
    for name in METRICS:
        row[name] = metrics.get(name)
    return row


def run_cell(
    *,
    cell: Mapping[str, Any],
    values: np.ndarray,
    config: Mapping[str, Any],
    shape: tuple[int, int, int],
    output: Path,
    slices: int,
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: Mapping[str, str],
    resume: bool,
    model_factory: Callable[[Mapping[str, Any], tuple[int, int, int], Mapping[str, Any]], Any] | None = None,
    intervention_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
                                                                             

    cell_dir = _cell_dir(Path(output), cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if _completed(cell_dir, cell, slices, source_sha256, config_sha256, data_sha256):
        if not resume:
            raise FileExistsError(f"completed RQ3 cell already exists; use --resume: {cell_dir}")
        summary = _read_json(cell_dir / "summary.json") or {}
        return {"status": "SKIPPED_COMPLETED", "summary": summary}
    attempts_dir = cell_dir / "attempts"
    if attempts_dir.is_dir() and any(attempts_dir.glob("attempt_*")) and not resume:
        raise FileExistsError(f"incomplete RQ3 attempt exists; use --resume: {cell_dir}")

    model_factory = model_factory or _new_model
    intervention_factory = intervention_factory or intervene_support
    if intervention_factory is None:
        _require_protocol_api()
    attempt = _new_attempt(cell_dir)
    status: dict[str, Any] = {
        "status": "RUNNING",
        "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
        **_jsonable(dict(cell)),
        "scenario": _jsonable(cell.get("scenario", {})),
        "time_slices_expected": int(slices),
        "time_slices_completed": 0,
        "started_at": _utc_now(),
        "attempt_dir": str(attempt),
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": dict(data_sha256),
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
    group_rows: list[dict[str, Any]] = []
    accumulators: dict[tuple[str, str], dict[str, Any]] = {
        key: _empty_accumulator() for key in GROUP_KEYS
    }
    overall_accumulator = _empty_accumulator()
    time_metric_values: dict[str, list[float]] = {name: [] for name in METRICS}
    mask_hashes: list[str] = []
    effective_hashes: list[str] = []
    selected_hashes: list[str] = []
    observed_hashes: list[str] = []
    service_count_hashes: list[str] = []
    query_state_checks: list[bool] = []
    private_state_checks: list[bool] = []
    cell_started = time.perf_counter()
    model: Any = None
    try:
        model = model_factory(config, shape, cell)
        raw_cap = _raw_cap(config, str(cell["qos"]))
        for time_slice in range(int(slices)):
            slice_started = time.perf_counter()
            base_split_started = time.perf_counter()
            try:
                from continual_qos.protocol import make_causal_split
            except ImportError as exc:                                                  
                raise ImportError("continual_qos.protocol.make_causal_split is required") from exc
            split = make_causal_split(
                np.asarray(values[time_slice]),
                float(cell["density"]),
                int(cell["seed"]),
                int(time_slice),
            )
            split_seconds = time.perf_counter() - base_split_started
            if split.support_count <= 0 or split.query_count <= 0:
                raise ValueError("causal split must have non-empty support and query sets")
            base_support_keys = split.support_users.astype(np.int64) * int(shape[2]) + split.support_services
            query_keys = split.query_users.astype(np.int64) * int(shape[2]) + split.query_services
            if np.intersect1d(base_support_keys, query_keys).size:
                raise AssertionError("base support/query masks overlap")
            if not np.all(np.isfinite(split.support_values)) or not np.all(np.isfinite(split.query_values)):
                raise FloatingPointError("causal split contains non-finite values")
            if np.any(split.support_values <= 0) or np.any(split.query_values <= 0):
                raise ValueError("causal split contains non-positive QoS values")

            intervention = intervention_factory(
                split.support_users,
                split.support_services,
                num_users=int(shape[1]),
                num_services=int(shape[2]),
                seed=int(cell["seed"]),
                time_slice=int(time_slice),
                scenario=cell.get("scenario", {}),
            )
            details = _intervention_details(
                intervention,
                original_users=split.support_users,
                original_services=split.support_services,
                num_users=int(shape[1]),
                num_services=int(shape[2]),
            )
            indices = details["indices"]
            support_users = split.support_users[indices]
            support_services = split.support_services[indices]
            support_values = split.support_values[indices]
            effective_keys = support_users.astype(np.int64) * int(shape[2]) + support_services
            if np.intersect1d(effective_keys, query_keys).size:
                raise AssertionError("effective support/query masks overlap")

            private_before = _private_snapshot(model)
            if indices.size:
                update_started = time.perf_counter()
                model.advance_support(
                    support_users,
                    support_services,
                    support_values,
                    raw_cap,
                    int(config["bias_epochs"]),
                    int(config["factor_epochs"]),
                    int(cell["seed"]) * 1_000_003 + int(time_slice),
                )
                update_seconds = time.perf_counter() - update_started
                profile = _profile_dict(model)
            else:
                                                                              
                                                                              
                                                                      
                update_seconds = 0.0
                profile = {}
            private_after = _private_snapshot(model)
            observed = np.asarray(details["observed_clients"], dtype=np.int64)
            unobserved_mask = np.ones(int(shape[1]), dtype=bool)
            unobserved_mask[observed] = False
            private_unchanged = _private_state_unchanged(
                private_before,
                private_after,
                unobserved=unobserved_mask,
                num_users=int(shape[1]),
            )
            if not private_unchanged:
                raise AssertionError("an unobserved client's private state changed during update")

            state_before_query = _state_hash(model)
            predict_started = time.perf_counter()
            prediction, prediction_timing = model.predict(
                split.query_users,
                split.query_services,
                raw_cap,
            )
            predict_seconds = time.perf_counter() - predict_started
            state_after_query = _state_hash(model)
            query_unchanged = state_before_query == state_after_query
            if not query_unchanged:
                raise AssertionError("read-only query prediction changed model state")
            truth = np.asarray(split.query_values, dtype=np.float64)
            estimate = np.asarray(prediction, dtype=np.float64)
            if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
                raise FloatingPointError("prediction is non-finite or has the wrong shape")
            if np.any(estimate < 0.0) or np.any(estimate > raw_cap):
                raise ValueError("prediction is outside the configured raw QoS cap")
            metrics, relative = _metric_values(truth, estimate)
            _add_accumulator(overall_accumulator, metrics, relative)
            for name in METRICS:
                value = float(metrics[name])
                if not np.isfinite(value):
                    raise FloatingPointError(f"non-finite RQ3 metric: {name}")
                time_metric_values[name].append(value)

            service_counts = np.asarray(details["service_counts"], dtype=np.int64)
            service_labels = np.asarray([_support_group(service_counts[int(service)]) for service in split.query_services])
            selected = np.asarray(details["selected_clients"], dtype=np.int64)
            selected_mask = np.isin(split.query_users, selected)
            effective_audit = details["audit"]
            per_slice_group_rows: list[dict[str, Any]] = []
            all_metrics = dict(metrics)
            _add_accumulator(accumulators[("all", "all")], all_metrics, relative)
            per_slice_group_rows.append(
                _group_row(
                    cell,
                    time_slice=time_slice,
                    axis="all",
                    group="all",
                    metrics=all_metrics,
                )
            )
            for group in SUPPORT_GROUPS:
                group_mask = service_labels == group
                group_metrics, group_relative = _metric_values(truth[group_mask], estimate[group_mask]) if np.any(group_mask) else (
                    {"count": 0, "mae": None, "rmse": None, "mre": None, "npre": None},
                    np.empty(0, dtype=np.float64),
                )
                _add_accumulator(accumulators[("support", group)], group_metrics, group_relative)
                per_slice_group_rows.append(
                    _group_row(
                        cell,
                        time_slice=time_slice,
                        axis="support",
                        group=group,
                        metrics=group_metrics,
                    )
                )
            for group, group_mask in (
                ("selected_clients", selected_mask),
                ("not_selected_clients", ~selected_mask),
            ):
                group_metrics, group_relative = _metric_values(truth[group_mask], estimate[group_mask]) if np.any(group_mask) else (
                    {"count": 0, "mae": None, "rmse": None, "mre": None, "npre": None},
                    np.empty(0, dtype=np.float64),
                )
                _add_accumulator(accumulators[("participation", group)], group_metrics, group_relative)
                per_slice_group_rows.append(
                    _group_row(
                        cell,
                        time_slice=time_slice,
                        axis="participation",
                        group=group,
                        metrics=group_metrics,
                    )
                )
            group_rows.extend(per_slice_group_rows)
            row: dict[str, Any] = {
                "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
                "cell_id": _cell_id(cell),
                "scenario_id": str(cell.get("scenario_id", "")),
                "variant_id": str(cell.get("variant_id", "")),
                "variant": str(cell.get("variant", "")),
                "qos": str(cell.get("qos", "")),
                "density": float(cell["density"]),
                "seed": int(cell["seed"]),
                "time_slice": int(time_slice),
                "original_support_count": int(split.support_count),
                "effective_support_count": int(indices.size),
                "dropped_support_count": int(split.support_count - indices.size),
                "selected_client_count": int(details["selected_clients"].size),
                "observed_client_count": int(details["observed_clients"].size),
                "planned_participation_rate": float(details["selected_clients"].size / int(shape[1])),
                "effective_observation_ratio": float(indices.size / max(1, split.support_count + split.query_count)),
                "query_entries": int(split.query_count),
                "original_mask_sha256": str(split.mask_sha256),
                "base_support_sha256": str(effective_audit.get("base_support_sha256", _hash_array(np.column_stack((split.support_users, split.support_services))))),
                "effective_support_sha256": str(effective_audit.get("effective_support_sha256", _hash_array(np.column_stack((support_users, support_services))))),
                "effective_support_indices_sha256": str(effective_audit["effective_support_indices_sha256"]),
                "selected_clients_sha256": str(effective_audit["selected_clients_sha256"]),
                "observed_clients_sha256": str(effective_audit["observed_clients_sha256"]),
                "service_counts_sha256": str(effective_audit["service_counts_sha256"]),
                "service_support_gini": float(effective_audit["service_support_gini"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mre": float(metrics["mre"]),
                "npre": float(metrics["npre"]),
                "absolute_error_sum": float(metrics["absolute_error_sum"]),
                "squared_error_sum": float(metrics["squared_error_sum"]),
                "split_seconds": float(split_seconds),
                "update_seconds": float(update_seconds),
                "predict_seconds": float(predict_seconds),
                "predict_kernel_seconds": float(dict(prediction_timing).get("kernel_seconds", 0.0)),
                "query_state_hash_before": state_before_query,
                "query_state_hash_after": state_after_query,
                "query_state_unchanged": query_unchanged,
                "unobserved_private_state_unchanged": private_unchanged,
                "query_or_future_used": False,
                "query_or_future_used_for_update": False,
                "gpu_profile_json": json.dumps(_jsonable(profile), ensure_ascii=False, sort_keys=True),
                "intervention_audit_json": json.dumps(_jsonable(effective_audit), ensure_ascii=False, sort_keys=True),
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
            }
            rows.append(row)
            _atomic_csv(attempt / "per_slice.csv", rows)
            _atomic_csv(attempt / "group_metrics_per_slice.csv", group_rows)
            mask_hashes.append(str(split.mask_sha256))
            effective_hashes.append(str(effective_audit["effective_support_indices_sha256"]))
            selected_hashes.append(str(effective_audit["selected_clients_sha256"]))
            observed_hashes.append(str(effective_audit["observed_clients_sha256"]))
            service_count_hashes.append(str(effective_audit["service_counts_sha256"]))
            query_state_checks.append(query_unchanged)
            private_state_checks.append(private_unchanged)
            if process is not None:
                peak_rss = max(peak_rss, int(process.memory_info().rss))
            status.update(
                {
                    "time_slices_completed": int(time_slice + 1),
                    "last_original_mask_sha256": str(split.mask_sha256),
                    "last_effective_support_indices_sha256": str(effective_audit["effective_support_indices_sha256"]),
                    "updated_at": _utc_now(),
                }
            )
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)

        end_cpu = process.cpu_times() if process is not None else None
        digest = hashlib.sha256()
        for value in mask_hashes:
            digest.update(bytes.fromhex(value))
        effective_digest = hashlib.sha256()
        for value in effective_hashes:
            effective_digest.update(bytes.fromhex(value))
        selected_digest = hashlib.sha256()
        for value in selected_hashes:
            selected_digest.update(bytes.fromhex(value))
        observed_digest = hashlib.sha256()
        for value in observed_hashes:
            observed_digest.update(bytes.fromhex(value))
        service_digest = hashlib.sha256()
        for value in service_count_hashes:
            service_digest.update(bytes.fromhex(value))
        pooled = _summarize_accumulator(overall_accumulator)
        time_mean = {
            name: float(np.mean(values_for_metric, dtype=np.float64))
            for name, values_for_metric in time_metric_values.items()
        }
        summary: dict[str, Any] = {
            "status": "COMPLETED",
            "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
            **_jsonable(dict(cell)),
            "scenario": _jsonable(cell.get("scenario", {})),
            "time_slices_completed": len(rows),
            "metrics": {"pooled": pooled, "time_mean": time_mean},
            "finite_metrics_all_slices": True,
            "query_or_future_used": False,
            "query_or_future_used_for_update": False,
            "gpu_only_model_arithmetic": True,
            "cpu_reference_model_constructed": False,
            "source_sha256": dict(source_sha256),
            "config_sha256": str(config_sha256),
            "data_sha256": dict(data_sha256),
            "mask_audit": {
                "count": len(mask_hashes),
                "unique_count": len(set(mask_hashes)),
                "mask_sha256_by_slice": {str(index): value for index, value in enumerate(mask_hashes)},
                "mask_stream_sha256": digest.hexdigest(),
                "effective_support_indices_sha256_by_slice": {
                    str(index): value for index, value in enumerate(effective_hashes)
                },
                "effective_support_stream_sha256": effective_digest.hexdigest(),
                "selected_clients_stream_sha256": selected_digest.hexdigest(),
                "observed_clients_stream_sha256": observed_digest.hexdigest(),
                "service_counts_stream_sha256": service_digest.hexdigest(),
                "support_query_partition_checked": True,
            },
            "query_state_unchanged_all_slices": bool(all(query_state_checks)),
            "unobserved_private_state_unchanged_all_slices": bool(all(private_state_checks)),
            "group_metrics": {
                f"{axis}:{group}": _summarize_accumulator(accumulator)
                for (axis, group), accumulator in sorted(accumulators.items())
            },
            "timing_seconds": {
                name: float(sum(float(row.get(name, 0.0)) for row in rows))
                for name in ("split_seconds", "update_seconds", "predict_seconds", "slice_wall_seconds")
            },
            "cell_wall_seconds": float(time.perf_counter() - cell_started),
            "gpu_peak_memory_used_bytes": max(
                (int(profile["peak_device_memory_used_bytes"])
                 for profile in (json.loads(row["gpu_profile_json"]) for row in rows)
                 if profile.get("peak_device_memory_used_bytes") is not None),
                default=None,
            ),
            "process_peak_rss_bytes": int(peak_rss),
            "process_cpu_user_seconds": float(end_cpu.user - start_cpu.user) if end_cpu is not None and start_cpu is not None else 0.0,
            "process_cpu_system_seconds": float(end_cpu.system - start_cpu.system) if end_cpu is not None and start_cpu is not None else 0.0,
            "final_state_sha256": _state_hash(model),
            "attempt_dir": str(attempt),
            "artifact_sha256": {
                name: sha256_file(attempt / name)
                for name in ("per_slice.csv", "group_metrics_per_slice.csv")
            },
            "completed_at": _utc_now(),
        }
        _atomic_json(attempt / "summary.json", summary)
        _atomic_json(attempt / "status.json", summary)
        _atomic_json(cell_dir / "summary.json", summary)
        _atomic_json(cell_dir / "status.json", summary)
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
        _atomic_json(attempt / "failure.json", failed)
        _atomic_json(cell_dir / "status.json", failed)
        raise


def _summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    pooled = summary.get("metrics", {}).get("pooled", {})
    time_mean = summary.get("metrics", {}).get("time_mean", {})
    row: dict[str, Any] = {
        "protocol": str(summary.get("protocol", "")),
        "cell_id": str(summary.get("cell_id", "")),
        "scenario_id": str(summary.get("scenario_id", "")),
        "variant_id": str(summary.get("variant_id", "")),
        "variant": str(summary.get("variant", "")),
        "qos": str(summary.get("qos", "")),
        "density": float(summary.get("density", 0.0)),
        "seed": int(summary.get("seed", 0)),
        "time_slices_completed": int(summary.get("time_slices_completed", 0)),
        "query_or_future_used": bool(summary.get("query_or_future_used", True)),
        "query_or_future_used_for_update": bool(summary.get("query_or_future_used_for_update", True)),
        "query_state_unchanged_all_slices": bool(summary.get("query_state_unchanged_all_slices", False)),
        "unobserved_private_state_unchanged_all_slices": bool(summary.get("unobserved_private_state_unchanged_all_slices", False)),
        "finite_metrics_all_slices": bool(summary.get("finite_metrics_all_slices", False)),
        "query_entries": int(pooled.get("count", 0)) if isinstance(pooled, Mapping) else 0,
        "source_sha256_runner": str(summary.get("source_sha256", {}).get("runner", "")),
        "config_sha256": str(summary.get("config_sha256", "")),
        "final_state_sha256": str(summary.get("final_state_sha256", "")),
    }
    for name in METRICS:
        row[f"pooled_{name}"] = pooled.get(name) if isinstance(pooled, Mapping) else None
        row[f"time_mean_{name}"] = time_mean.get(name) if isinstance(time_mean, Mapping) else None
    return row


def _aggregate_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[(str(row["scenario_id"]), str(row["variant_id"]), str(row["qos"]), float(row["density"]))].append(row)
    result: list[dict[str, Any]] = []
    for (scenario_id, variant_id, qos, density), rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "scenario_id": scenario_id,
            "variant_id": variant_id,
            "qos": qos,
            "density": density,
            "n_seeds": len(rows),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda item: int(item["seed"]))),
        }
        for name in METRICS:
            values = np.asarray([float(row[f"pooled_{name}"]) for row in rows if row.get(f"pooled_{name}") is not None], dtype=np.float64)
            time_values = np.asarray([float(row[f"time_mean_{name}"]) for row in rows if row.get(f"time_mean_{name}") is not None], dtype=np.float64)
            aggregate[f"pooled_{name}_mean"] = float(np.mean(values)) if values.size else None
            aggregate[f"pooled_{name}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size else None)
            aggregate[f"time_mean_{name}_mean"] = float(np.mean(time_values)) if time_values.size else None
            aggregate[f"time_mean_{name}_std"] = float(np.std(time_values, ddof=1)) if time_values.size > 1 else (0.0 if time_values.size else None)
        result.append(aggregate)
    return result


def _group_terminal_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for key, metrics in sorted(dict(summary.get("group_metrics", {})).items()):
            if ":" not in key:
                continue
            axis, group = key.split(":", 1)
            row = {
                "protocol": str(summary.get("protocol", "")),
                "cell_id": str(summary.get("cell_id", "")),
                "scenario_id": str(summary.get("scenario_id", "")),
                "variant_id": str(summary.get("variant_id", "")),
                "variant": str(summary.get("variant", "")),
                "qos": str(summary.get("qos", "")),
                "density": float(summary.get("density", 0.0)),
                "seed": int(summary.get("seed", 0)),
                "scope": "pooled",
                "time_slice": None,
                "group_axis": axis,
                "group": group,
            }
            row.update({"count": metrics.get("count", 0), **{name: metrics.get(name) for name in METRICS}})
            rows.append(row)
    return rows


def _paired_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["scenario_id"]), str(row["qos"]), float(row["density"]), int(row["seed"]), str(row["variant_id"])): row for row in summary_rows}
    keys = sorted({key[:-1] for key in by_key})
    result: list[dict[str, Any]] = []
    for key in keys:
        left = by_key.get((*key, "R0"))
        right = by_key.get((*key, "R1"))
        if left is None or right is None:
            continue
        row: dict[str, Any] = {
            "scenario_id": key[0],
            "qos": key[1],
            "density": key[2],
            "seed": key[3],
            "reference_variant_id": "R0",
            "matched_variant_id": "R1",
            "reference_cell_id": left["cell_id"],
            "matched_cell_id": right["cell_id"],
        }
        for name in METRICS:
            base = float(left[f"pooled_{name}"])
            control = float(right[f"pooled_{name}"])
            row[f"R0_{name}"] = base
            row[f"R1_{name}"] = control
            row[f"R1_minus_R0_{name}"] = control - base
            row[f"R0_improvement_vs_R1_percent_{name}"] = (100.0 * (control - base) / control) if control != 0.0 else None
        result.append(row)
    return result


def _configured_output(protocol: Mapping[str, Any], scope: str, override: Path | None = None) -> Path:
    if override is not None:
        path = Path(override)
        if not path.is_absolute():
            path = ROOT / path
    else:
        outputs = protocol.get("outputs", {})
        raw: Any = outputs.get(scope) if isinstance(outputs, Mapping) else None
        if isinstance(raw, Mapping):
            raw = raw.get("root", raw.get("path"))
        if raw is None:
            raw = protocol.get(f"{scope}_output")
        if raw is None:
            raw = Path("results") / f"rq4_robustness_{scope}"
        path = Path(str(raw))
        if not path.is_absolute():
            path = ROOT / path
    path = path.resolve()
    allowed = (ROOT / "results").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError(f"RQ4 output must stay below {allowed}: {path}")
    if not path.name.startswith("rq4_robustness_"):
        raise ValueError(f"RQ4 output leaf must start with rq4_robustness_: {path}")
    return path


def _assert_fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty RQ3 output; use --resume: {path}")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:
            return True
                                                                            
                                                                            
    return True


class _BatchLock:
                                                                         

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.payload = dict(payload)
        self.owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = _read_json(self.path) or {}
            pid = int(existing.get("pid", -1)) if str(existing.get("pid", "")).lstrip("-").isdigit() else -1
            if _pid_is_alive(pid):
                raise RuntimeError(f"RQ3 output is locked by active pid {pid}: {self.path}") from exc
            raise RuntimeError(
                f"RQ3 output has a stale lock at {self.path}; verify pid {pid} and rerun with --unlock"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(self.payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self.owned = True

    def release(self) -> None:
        if not self.owned:
            return
        try:
            current = _read_json(self.path) or {}
            if int(current.get("pid", -1)) == int(self.payload.get("pid", -2)):
                self.path.unlink(missing_ok=True)
        finally:
            self.owned = False


def _acquire_batch_lock(output: Path, *, scope: str, protocol_id: str) -> _BatchLock:
    lock = _BatchLock(
        Path(output) / ".rq3_runner.lock",
        {
            "pid": os.getpid(),
            "scope": scope,
            "protocol_id": protocol_id,
            "started_at": _utc_now(),
        },
    )
    lock.acquire()
    return lock


def unlock_output(output: Path) -> dict[str, Any]:
                                                                     

    lock_path = Path(output).resolve() / ".rq3_runner.lock"
    existing = _read_json(lock_path)
    if existing is None:
        return {"status": "NO_LOCK", "lock": str(lock_path)}
    raw_pid = existing.get("pid", -1)
    pid = int(raw_pid) if str(raw_pid).lstrip("-").isdigit() else -1
    if _pid_is_alive(pid):
        raise RuntimeError(f"refusing to unlock active pid {pid}: {lock_path}")
    lock_path.unlink()
    return {"status": "UNLOCKED", "lock": str(lock_path), "previous_pid": pid}


def _protocol_matrix(protocol: Mapping[str, Any], smoke: bool) -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...]]:
    matrix = protocol.get("smoke" if smoke else "matrix", {})
    matrix = matrix if isinstance(matrix, Mapping) else {}
    return (
        tuple(str(item) for item in matrix.get("qos_types", _qos_types(protocol))),
        tuple(float(item) for item in matrix.get("densities", (0.10,) if smoke else (0.01, 0.05, 0.10, 0.15, 0.20))),
        tuple(int(item) for item in matrix.get("seeds", (42,))),
    )


def _build_manifest(
    protocol: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    output: Path,
    shape: tuple[int, int, int],
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: Mapping[str, str],
    slices: int,
) -> dict[str, Any]:
    return {
        "manifest_kind": "rq3_robustness_batch_manifest",
        "status": "PLANNED",
        "protocol_id": str(protocol.get("protocol_id", "")),
        "protocol": str(protocol.get("protocol_id", "")),
        "rq": "RQ3",
        "scope": scope,
        "output": str(Path(output).resolve()),
        "created_at": _utc_now(),
        "formal_authorized": bool(protocol.get("formal_authorized", False)),
        "protocol_status": str(protocol.get("status", "")),
        "shape": list(shape),
        "qos_types": list(_qos_types(protocol)),
        "densities": [float(item) for item in _protocol_matrix(protocol, scope == "smoke")[1]],
        "seeds": [int(item) for item in _protocol_matrix(protocol, scope == "smoke")[2]],
        "time_slices": int(slices),
        "cells": [_jsonable(dict(cell)) for cell in cells],
        "cell_count": len(cells),
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": dict(data_sha256),
        "query_or_future_used": False,
        "training_required": True,
        "terminal_artifacts": [
            "manifest.json",
            "queue_cells.csv",
            "queue_status.json",
            "final_summary.csv",
            "aggregate_mean_std.csv",
            "group_metrics.csv",
            "paired_comparisons.csv",
            "BATCH_COMPLETED",
        ],
        "gates": {
            "cell_slices": int(slices),
            "metrics": list(METRICS),
            "support_groups": list(SUPPORT_GROUPS),
            "participation_groups": list(PARTICIPATION_GROUPS),
            "query_is_read_only": True,
            "dropped_support_not_added_to_query": True,
        },
    }


def write_plan(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    output: Path,
    cells: Sequence[Mapping[str, Any]],
    shape: tuple[int, int, int],
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: Mapping[str, str],
    slices: int,
) -> dict[str, Any]:
    output = _configured_output(protocol, "smoke" if smoke else "formal", output)
    _assert_fresh(output)
    manifest = _build_manifest(
        protocol,
        cells,
        scope="smoke" if smoke else "formal",
        output=output,
        shape=shape,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        slices=slices,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    _atomic_json(output / "manifest.json", manifest)
    queue_rows = [_cell_manifest_row(cell) for cell in cells]
    _atomic_csv(output / "queue_cells.csv", queue_rows)
    _atomic_json(
        output / "queue_status.json",
        {
            "status": "PLANNED",
            "protocol_id": manifest["protocol_id"],
            "scope": manifest["scope"],
            "cells_expected": len(cells),
            "cells_visited": 0,
            "cells_completed": 0,
            "cells_failed": 0,
            "active": None,
            "process_count": 0,
            "created_at": manifest["created_at"],
            "updated_at": _utc_now(),
            "required_terminal_artifacts": manifest["terminal_artifacts"],
        },
    )
    return {
        "status": "PLANNED",
        "output": str(output),
        "cells": len(cells),
        "time_slices": int(slices),
        "formal_authorized": bool(protocol.get("formal_authorized", False)),
        "protocol_status": str(protocol.get("status", "")),
        "training_launched": False,
    }


def _validate_existing_manifest(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    immutable = (
        "manifest_kind",
        "protocol_id",
        "scope",
        "shape",
        "qos_types",
        "densities",
        "seeds",
        "time_slices",
        "cells",
        "cell_count",
        "source_sha256",
        "config_sha256",
        "data_sha256",
    )
    for key in immutable:
        if existing.get(key) != current.get(key):
            raise ValueError(f"existing RQ3 manifest mismatch at {key}; start a new output")


def _audit_cell(
    output: Path, cell: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return _audit_cell_impl(output, cell, manifest)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"cell_id": _cell_id(cell), "accepted": False,
                "reason": f"invalid cell artifact: {type(exc).__name__}: {exc}"}


def _audit_cell_impl(
    output: Path,
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    directory = _cell_dir(output, cell)
    summary = _read_json(directory / "summary.json")
    status = _read_json(directory / "status.json")
    result: dict[str, Any] = {"cell_id": _cell_id(cell), "path": str(directory), "accepted": False}
    if summary is None or status is None:
        result["reason"] = "missing summary/status"
        return result
    checks: dict[str, bool] = {
        "summary_completed": summary.get("status") == "COMPLETED",
        "status_completed": status.get("status") == "COMPLETED",
        "cell_id": str(summary.get("cell_id", "")) == _cell_id(cell),
        "scenario_id": str(summary.get("scenario_id", "")) == str(cell.get("scenario_id", "")),
        "variant_id": str(summary.get("variant_id", "")) == str(cell.get("variant_id", "")),
        "qos": str(summary.get("qos", "")) == str(cell.get("qos", "")),
        "density": np.isclose(float(summary.get("density", -1.0)), float(cell.get("density", -2.0))),
        "seed": int(summary.get("seed", -1)) == int(cell.get("seed", -2)),
        "time_slices": int(summary.get("time_slices_completed", -1)) == int(manifest.get("time_slices", -2)),
        "finite_metrics": bool(summary.get("finite_metrics_all_slices")),
        "query_flags": not bool(summary.get("query_or_future_used")) and not bool(summary.get("query_or_future_used_for_update")),
        "query_state": bool(summary.get("query_state_unchanged_all_slices")),
        "private_state": bool(summary.get("unobserved_private_state_unchanged_all_slices")),
        "final_state_sha256": bool(summary.get("final_state_sha256")),
        "source_sha256": summary.get("source_sha256") == manifest.get("source_sha256"),
        "config_sha256": str(summary.get("config_sha256", "")) == str(manifest.get("config_sha256", "")),
        "data_sha256": summary.get("data_sha256") == manifest.get("data_sha256"),
    }
    pooled = summary.get("metrics", {}).get("pooled", {})
    checks["pooled_metrics"] = isinstance(pooled, Mapping) and int(pooled.get("count", 0)) > 0 and all(
        pooled.get(name) is not None and np.isfinite(float(pooled[name])) for name in METRICS
    )
    per_slice_path = Path(str(summary.get("attempt_dir", ""))) / "per_slice.csv"
    per_slice_ok = False
    if per_slice_path.is_file():
        try:
            rows = _read_csv(per_slice_path)
            per_slice_ok = len(rows) == int(manifest.get("time_slices", -1)) and all(
                all(row.get(name, "") != "" and np.isfinite(float(row[name])) for name in METRICS)
                and row.get("query_or_future_used", "true").lower() == "false"
                and row.get("query_state_unchanged", "false").lower() == "true"
                and row.get("unobserved_private_state_unchanged", "false").lower() == "true"
                and bool(row.get("original_mask_sha256"))
                and bool(row.get("effective_support_indices_sha256"))
                for row in rows
            )
        except Exception:
            per_slice_ok = False
    checks["per_slice"] = per_slice_ok
    checks["artifact_integrity"] = _cell_artifacts_valid(
        directory, summary, int(manifest.get("time_slices", -1))
    )
    result["checks"] = checks
    result["accepted"] = all(checks.values())
    result["reason"] = "strict complete" if result["accepted"] else "; ".join(name for name, passed in checks.items() if not passed)
    return result


def _audit_pairs(output: Path, cells: Sequence[Mapping[str, Any]], slices: int) -> dict[str, Any]:
    by_key: dict[tuple[str, str, float, int, str], Mapping[str, Any]] = {}
    for cell in cells:
        by_key[(str(cell.get("scenario_id", "")), str(cell.get("qos", "")), float(cell["density"]), int(cell["seed"]), str(cell.get("variant_id", "")))] = cell
    mismatches: list[str] = []
    pairs = 0
    expected_keys = sorted({item[:-1] for item in by_key})

    def _latest_per_slice(cell: Mapping[str, Any]) -> list[dict[str, str]] | None:
        directory = _cell_dir(output, cell)
        root_summary = _read_json(directory / "summary.json")
        candidates: list[Path] = []
        if isinstance(root_summary, Mapping) and root_summary.get("attempt_dir"):
            candidates.append(Path(str(root_summary["attempt_dir"])) / "per_slice.csv")
        attempts = sorted((directory / "attempts").glob("attempt_*") if (directory / "attempts").is_dir() else [])
        candidates.extend((attempt / "per_slice.csv") for attempt in reversed(attempts))
        for candidate in candidates:
            if candidate.is_file():
                try:
                    rows = _read_csv(candidate)
                except Exception:
                    continue
                if len(rows) == int(slices):
                    return rows
        return None

    for key in expected_keys:
        left = by_key.get((*key, "R0"))
        right = by_key.get((*key, "R1"))
        if left is None or right is None:
            mismatches.append("/".join(map(str, key)) + ":missing_variant")
            continue
        left_rows = _latest_per_slice(left)
        right_rows = _latest_per_slice(right)
        if left_rows is None or right_rows is None:
            mismatches.append("/".join(map(str, key)) + ":missing_per_slice")
            continue
        pairs += 1
        for index, (lrow, rrow) in enumerate(zip(left_rows, right_rows)):
            for field in ("original_mask_sha256", "effective_support_indices_sha256", "selected_clients_sha256", "observed_clients_sha256", "service_counts_sha256"):
                if lrow.get(field) != rrow.get(field):
                    mismatches.append("/".join(map(str, key)) + f":t{index:03d}:{field}")
    return {
        "pairs_expected": len(expected_keys),
        "pairs_checked": pairs,
        "mismatches": mismatches,
        "valid": pairs == len(expected_keys) and not mismatches,
    }


def audit_batch(
    protocol: Mapping[str, Any],
    *,
    output: Path,
    smoke: bool,
) -> dict[str, Any]:
                                                                            

    output = Path(output).resolve()
    manifest = _read_json(output / "manifest.json")
    queue_path = output / "queue_cells.csv"
    status = _read_json(output / "queue_status.json")
    if manifest is None or status is None or not queue_path.is_file():
        return {"status": "MISSING", "output": str(output)}
    shape, _ = _dataset_spec(protocol)
    cells = list(build_cells(protocol, smoke=smoke)) if build_cells is not None else []
    current_slices = int(scope_slices(protocol, smoke)) if scope_slices is not None else int(manifest.get("time_slices", -1))
    current_config, config_path = _load_config(protocol)
    del current_config
    current_source = _source_hashes(protocol, config_path)
    _, current_data = _load_caches(protocol, _read_json(config_path) or {}, shape, _qos_types(protocol))
    current_manifest = _build_manifest(
        protocol,
        cells,
        scope="smoke" if smoke else "formal",
        output=output,
        shape=shape,
        source_sha256=current_source,
        config_sha256=sha256_file(config_path),
        data_sha256=current_data,
        slices=current_slices,
    )
    manifest_match = True
    try:
        _validate_existing_manifest(manifest, current_manifest)
    except ValueError:
        manifest_match = False
    queue_rows = _read_csv(queue_path)
    expected_ids = [_cell_id(cell) for cell in cells]
    observed_ids = [str(row.get("cell_id", "")) for row in queue_rows]
    duplicate_ids = sorted({item for item in observed_ids if observed_ids.count(item) > 1})
    missing_ids = sorted(set(expected_ids) - set(observed_ids))
    unexpected_ids = sorted(set(observed_ids) - set(expected_ids))
    expected_by_id = {_cell_id(cell): cell for cell in cells}
    cell_audits = [
        _audit_cell(output, expected_by_id[row["cell_id"]], manifest)
        for row in queue_rows
        if row.get("cell_id") in expected_by_id and row.get("status") in ACCEPTED_CELL_STATUSES
    ]
    terminal_paths = [output / name for name in manifest.get("terminal_artifacts", [])]
    terminal_present = all(path.is_file() and path.stat().st_size > 0 for path in terminal_paths)
    marker = _read_json(output / "BATCH_COMPLETED") or {}
    terminal_integrity = (
        marker.get("status") == TERMINAL_STATUS
        and marker.get("protocol_id") == manifest.get("protocol_id")
        and marker.get("scope") == manifest.get("scope")
    )
    terminal_integrity = terminal_integrity and marker.get("cells_completed") == len(cells)
    for name, key in (
        ("final_summary.csv", "final_summary_sha256"),
        ("aggregate_mean_std.csv", "aggregate_mean_std_sha256"),
        ("group_metrics.csv", "group_metrics_sha256"),
        ("paired_comparisons.csv", "paired_comparisons_sha256"),
    ):
        target = output / name
        terminal_integrity = terminal_integrity and target.is_file() and marker.get(key) == sha256_file(target)
    expected_summaries = marker.get("cell_summary_sha256", {})
    for cell in cells:
        target = _cell_dir(output, cell) / "summary.json"
        terminal_integrity = terminal_integrity and target.is_file() and expected_summaries.get(_cell_id(cell)) == sha256_file(target)
    all_queue_accepted = (
        manifest_match
        and len(queue_rows) == len(cells)
        and not duplicate_ids
        and not missing_ids
        and not unexpected_ids
        and all(row.get("status") in ACCEPTED_CELL_STATUSES for row in queue_rows)
    )
    all_cells = len(cell_audits) == len(cells) and bool(cell_audits) and all(item["accepted"] for item in cell_audits)
    pair_audit = _audit_pairs(output, cells, current_slices) if all_cells else {"pairs_checked": 0, "mismatches": [], "valid": False}
    complete = (
        all_queue_accepted
        and all_cells
        and pair_audit["valid"]
        and status.get("status") == TERMINAL_STATUS
        and status.get("active") is None
        and int(status.get("process_count", -1)) == 0
        and terminal_present
        and terminal_integrity
    )
    if complete:
        batch_status = "COMPLETE"
    elif manifest_match and status.get("status") == "PLANNED" and not cell_audits:
        batch_status = "PLANNED"
    else:
        batch_status = "INCOMPLETE"
    return {
        "status": batch_status,
        "output": str(output),
        "manifest_match": manifest_match,
        "expected_cells": len(cells),
        "queued_cells": len(queue_rows),
        "completed_queue_rows": sum(row.get("status") in ACCEPTED_CELL_STATUSES for row in queue_rows),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "all_queue_rows_accepted": all_queue_accepted,
        "all_cell_gates": all_cells,
        "terminal_artifacts_present": terminal_present,
        "terminal_artifact_integrity": bool(terminal_integrity),
        "paired_mask_audit": pair_audit,
        "cell_audits": cell_audits,
    }


def _run_queue_unlocked(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    output: Path | None = None,
    resume: bool = False,
    run_formal: bool = False,
    stop_on_error: bool = False,
    model_factory: Callable[[Mapping[str, Any], tuple[int, int, int], Mapping[str, Any]], Any] | None = None,
    intervention_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
                                                                                      

    if not smoke:
        if not run_formal:
            raise PermissionError("formal RQ3 execution requires --run-formal")
        if str(protocol.get("status", "")) != "FROZEN" or not bool(protocol.get("formal_authorized", False)):
            raise PermissionError("formal RQ3 execution requires protocol status FROZEN and formal_authorized=true")
    _require_protocol_api()
    shape, _ = _dataset_spec(protocol)
    config, config_path = _load_config(protocol)
    qos_types = _qos_types(protocol)
    cache, data_sha256 = _load_caches(protocol, config, shape, qos_types)
    source_sha256 = _source_hashes(protocol, config_path)
    config_sha256 = sha256_file(config_path)
    cells = list(build_cells(protocol, smoke=smoke))
    slices = int(scope_slices(protocol, smoke))
    scope = "smoke" if smoke else "formal"
    output = _configured_output(protocol, scope, output)
    current_manifest = _build_manifest(
        protocol,
        cells,
        scope=scope,
        output=output,
        shape=shape,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        slices=slices,
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = _read_json(manifest_path)
    if existing is not None:
        if not resume:
            raise FileExistsError(f"RQ3 output is initialized; use --resume: {output}")
        _validate_existing_manifest(existing, current_manifest)
        current_manifest["created_at"] = existing.get("created_at", current_manifest["created_at"])
    elif any(path.name != ".rq3_runner.lock" for path in output.iterdir()):
        raise FileExistsError(f"RQ3 output is non-empty without a manifest: {output}")
    _atomic_json(manifest_path, current_manifest)
    queue_rows = [_cell_manifest_row(cell) for cell in cells]
    queue_path = output / "queue_cells.csv"
    queue_started = time.perf_counter()
    completed = skipped = failed = 0

    def write_status(*, status_value: str, active: Mapping[str, Any] | None, visited: int) -> None:
        _atomic_json(
            output / "queue_status.json",
            {
                "status": status_value,
                "protocol_id": current_manifest["protocol_id"],
                "scope": scope,
                "cells_expected": len(cells),
                "cells_visited": int(visited),
                "cells_completed": int(completed + skipped),
                "cells_completed_this_invocation": int(completed),
                "cells_skipped_completed": int(skipped),
                "cells_failed": int(failed),
                "active": _jsonable(active),
                "process_count": 1 if active is not None else 0,
                "updated_at": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - queue_started),
                "required_terminal_artifacts": current_manifest["terminal_artifacts"],
            },
        )

    _atomic_csv(queue_path, queue_rows)
    write_status(status_value="RUNNING", active=None, visited=0)
    for index, cell in enumerate(cells):
        queue_rows[index]["status"] = "RUNNING"
        queue_rows[index]["attempt_dir"] = ""
        _atomic_csv(queue_path, queue_rows)
        write_status(status_value="RUNNING", active={"cell_id": _cell_id(cell), "pid": os.getpid()}, visited=index)
        try:
            cell_for_run = {**dict(cell), "protocol": current_manifest["protocol_id"]}
            result = run_cell(
                cell=cell_for_run,
                values=cache[str(cell["qos"])],
                config=config,
                shape=shape,
                output=output,
                slices=slices,
                source_sha256=source_sha256,
                config_sha256=config_sha256,
                data_sha256=data_sha256,
                resume=resume,
                model_factory=model_factory,
                intervention_factory=intervention_factory,
            )
            queue_rows[index]["status"] = str(result["status"])
            queue_rows[index]["attempt_dir"] = str(result.get("summary", {}).get("attempt_dir", ""))
            if result["status"] == "SKIPPED_COMPLETED":
                skipped += 1
            else:
                completed += 1
        except Exception as exc:
            failed += 1
            queue_rows[index]["status"] = "FAILED"
            queue_rows[index]["error"] = f"{type(exc).__name__}: {exc}"
            if stop_on_error:
                _atomic_csv(queue_path, queue_rows)
                write_status(status_value="INCOMPLETE", active=None, visited=index + 1)
                raise
        _atomic_csv(queue_path, queue_rows)
        write_status(status_value="RUNNING", active=None, visited=index + 1)

    audits = [_audit_cell(output, cell, current_manifest) for cell in cells]
    valid_cells = all(item["accepted"] for item in audits) and len(audits) == len(cells)
    pair_audit = _audit_pairs(output, cells, slices) if valid_cells else {"pairs_checked": 0, "mismatches": [], "valid": False}
    terminal_valid = valid_cells and pair_audit["valid"]
    if terminal_valid:
        summaries = [_read_json(_cell_dir(output, cell) / "summary.json") for cell in cells]
        summaries = [item for item in summaries if isinstance(item, Mapping)]
        summary_rows = [_summary_row(summary) for summary in summaries]
        _atomic_csv(output / "final_summary.csv", summary_rows)
        _atomic_csv(output / "aggregate_mean_std.csv", _aggregate_rows(summary_rows))
        _atomic_csv(output / "group_metrics.csv", _group_terminal_rows(summaries))
        _atomic_csv(output / "paired_comparisons.csv", _paired_rows(summary_rows))
        marker = {
            "status": TERMINAL_STATUS,
            "protocol_id": current_manifest["protocol_id"],
            "scope": scope,
            "cells_expected": len(cells),
            "cells_completed": len(summaries),
            "paired_mask_audit": pair_audit,
            "final_summary_sha256": sha256_file(output / "final_summary.csv"),
            "aggregate_mean_std_sha256": sha256_file(output / "aggregate_mean_std.csv"),
            "group_metrics_sha256": sha256_file(output / "group_metrics.csv"),
            "paired_comparisons_sha256": sha256_file(output / "paired_comparisons.csv"),
            "cell_summary_sha256": {
                _cell_id(cell): sha256_file(_cell_dir(output, cell) / "summary.json")
                for cell in cells
            },
            "completed_at": _utc_now(),
        }
        _atomic_json(output / "BATCH_COMPLETED", marker)
        final_status = TERMINAL_STATUS
    else:
        final_status = "COMPLETED_WITH_FAILURES"
    final = {
        "status": final_status,
        "protocol_id": current_manifest["protocol_id"],
        "scope": scope,
        "cells_expected": len(cells),
        "cells_valid": sum(bool(item["accepted"]) for item in audits),
        "cells_completed_this_invocation": completed,
        "cells_skipped_completed": skipped,
        "cells_failed": failed,
        "invalid_cells": [item["cell_id"] for item in audits if not item["accepted"]],
        "paired_mask_audit": pair_audit,
        "elapsed_seconds": float(time.perf_counter() - queue_started),
        "completed_at": _utc_now(),
    }
    write_status(status_value=final_status, active=None, visited=len(cells))
    return final


def run_queue(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    output: Path | None = None,
    resume: bool = False,
    run_formal: bool = False,
    stop_on_error: bool = False,
    model_factory: Callable[[Mapping[str, Any], tuple[int, int, int], Mapping[str, Any]], Any] | None = None,
    intervention_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
                                                          

    if not smoke and (
        not run_formal
        or str(protocol.get("status", "")) != "FROZEN"
        or not bool(protocol.get("formal_authorized", False))
    ):
        raise PermissionError(
            "formal RQ3 execution requires --run-formal and authorized FROZEN protocol"
        )
    scope = "smoke" if smoke else "formal"
    resolved_output = _configured_output(protocol, scope, output)
    lock = _acquire_batch_lock(
        resolved_output,
        scope=scope,
        protocol_id=str(protocol.get("protocol_id", "")),
    )
    try:
        return _run_queue_unlocked(
            protocol,
            smoke=smoke,
            output=resolved_output,
            resume=resume,
            run_formal=run_formal,
            stop_on_error=stop_on_error,
            model_factory=model_factory,
            intervention_factory=intervention_factory,
        )
    finally:
        lock.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="inspect scope without writing or training")
    parser.add_argument("--write-plan", action="store_true", help="write PLANNED queues only")
    parser.add_argument("--smoke", action="store_true", help="run the independent smoke scope")
    parser.add_argument("--run-formal", action="store_true", help="explicitly authorize execution of the formal scope")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--audit",
        nargs="?",
        const=Path(""),
        default=None,
        type=Path,
        help="audit an existing RQ3 output root (default: protocol output)",
    )
    parser.add_argument("--unlock", action="store_true", help="explicitly remove an inactive batch lock after checking its PID")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if sum(bool(v) for v in (args.dry_run, args.write_plan, args.run_formal,
                             args.audit is not None, args.unlock)) > 1:
        raise ValueError("choose exactly one action: inspect, plan, audit, unlock, or run")
    if args.run_formal and args.smoke:
        raise ValueError("--run-formal and --smoke cannot be combined")
    _require_protocol_api()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    smoke = bool(args.smoke)
    if not any((args.dry_run, args.write_plan, args.smoke, args.run_formal, args.audit is not None, args.unlock)):
                                                                            
                                                    
        args.dry_run = True
    scope = "smoke" if smoke else "formal"
    if args.unlock:
        output = _configured_output(protocol, scope, args.output)
        result = unlock_output(output)
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        return 0
    if args.audit is not None:
        audit_arg = args.output if str(args.audit) in {"", "."} else Path(args.audit)
        audit_output = _configured_output(protocol, scope, audit_arg)
        result = audit_batch(protocol, output=audit_output, smoke=smoke)
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {"COMPLETE", "PLANNED"} else 1
    cells = list(build_cells(protocol, smoke=smoke))
    slices = int(scope_slices(protocol, smoke))
    shape, _ = _dataset_spec(protocol)
    if args.dry_run:
        result = {
            "status": "DRY_RUN",
            "protocol_id": protocol.get("protocol_id"),
            "scope": scope,
            "cells": len(cells),
            "time_slices": slices,
            "shape": list(shape),
            "formal_authorized": bool(protocol.get("formal_authorized", False)),
            "protocol_status": protocol.get("status"),
            "training_launched": False,
            "output": str(_configured_output(protocol, scope, args.output)),
        }
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        return 0
    if not smoke and args.run_formal and (
        str(protocol.get("status", "")) != "FROZEN"
        or not bool(protocol.get("formal_authorized", False))
    ):
                                                                             
        raise PermissionError(
            "formal RQ3 execution requires protocol status FROZEN and formal_authorized=true"
        )
    config, config_path = _load_config(protocol)
    del config
    source_sha256 = _source_hashes(protocol, config_path)
    config_sha256 = sha256_file(config_path)
    _, data_sha256 = _load_caches(protocol, _read_json(config_path) or {}, shape, _qos_types(protocol))
    output = _configured_output(protocol, scope, args.output)
    if args.write_plan:
        result = write_plan(
            protocol,
            smoke=smoke,
            output=output,
            cells=cells,
            shape=shape,
            source_sha256=source_sha256,
            config_sha256=config_sha256,
            data_sha256=data_sha256,
            slices=slices,
        )
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        return 0
    if not smoke and not args.run_formal:
        raise SystemExit("formal execution requires --run-formal; use --dry-run or --write-plan for preparation")
    result = run_queue(
        protocol,
        smoke=smoke,
        output=output,
        resume=bool(args.resume),
        run_formal=bool(args.run_formal),
        stop_on_error=bool(args.stop_on_error),
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") == TERMINAL_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())

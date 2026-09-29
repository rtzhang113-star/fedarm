                                                      

                                                                        
                                                                          
                                                                             
                                                                             

                                          

                 
                                                                             

                
                                                                            
                                                                              
           

                                                                          
                                                                          
                                                                    
   

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import traceback
import types
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:                                                              
    import psutil
except Exception:                                                           
    psutil = None                            


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO_ROOT = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

                                                                          
                                                                       
if importlib.util.find_spec("torch") is None and "core" not in sys.modules:
    _code_core_stub = types.ModuleType("core")
    _code_core_stub.__path__ = [str(ROOT / "code" / "core")]
    sys.modules["core"] = _code_core_stub

from core.data import load_value_cache
from core.gpu_ablation_training import prediction_metrics
from core.protocol import make_causal_split


PROTOCOL = "fedarm_rq5_retention_historysafe_v1"
SHAPE = (64, 142, 4500)
FORMAL_DENSITIES = (0.01, 0.10, 0.25)
SUPPLEMENT_DENSITIES = (0.05, 0.15, 0.20)
FORMAL_SEEDS = (42, 3407, 1024, 1, 0)
FORMAL_QOS = ("rt", "tp")
METHODS = ("fedsaa_retain", "fedsaa_reset")
METHOD_DISPLAY = {
    "fedsaa_retain": "FedARM-Retain",
    "fedsaa_reset": "FedARM-Reset",
}
METRICS = ("mae", "rmse", "mre", "npre")
PRIMARY_FORGETTING_METRICS = ("mae", "rmse")
TERMINAL_STATUS = "BATCH_COMPLETED"
ACCEPTED_CELL_STATUSES = {"COMPLETED", "SKIPPED_COMPLETED"}
CURRENT_FIELDS = (
    "protocol",
    "cell_id",
    "method",
    "method_display",
    "qos",
    "density",
    "seed",
    "time_slice",
    "support_entries",
    "query_entries",
    "mask_sha256",
    "query_indices_sha256",
    "theta0_state_sha256",
    "initial_or_inherited_state_sha256",
    "post_update_state_sha256",
    "query_state_hash_before",
    "query_state_hash_after",
    "query_state_unchanged",
    "query_or_future_used",
    "query_labels_used_for_update",
    "future_slices_used",
    "reset_matches_fixed_theta0",
    "retain_inherits_previous_state",
    "model_state_bytes",
    "update_wall_seconds",
    "predict_wall_seconds",
    "history_replay_wall_seconds",
    "gpu_peak_memory_used_bytes",
    "process_rss_bytes",
    "active_clients",
    "proposal_count",
    *METRICS,
    "absolute_error_sum",
    "squared_error_sum",
    "query_count",
)
HISTORY_FIELDS = (
    "protocol",
    "cell_id",
    "method",
    "method_display",
    "qos",
    "density",
    "seed",
    "checkpoint_slice",
    "historical_slice",
    "historical_mask_sha256",
    "query_entries",
    "history_replay_state_hash_before",
    "history_replay_state_hash_after",
    "history_replay_batch_read_only",
    "query_labels_evaluator_only",
    "future_slices_used",
    *METRICS,
    "absolute_error_sum",
    "squared_error_sum",
    "query_count",
)

_MODEL_MODULE_NAME = "_rq5_retention_historysafe_model_v1"
_MODEL_MODULE: Any | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
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
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    return sha256_file(path) if Path(path).is_file() else None


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return _hash_bytes(
        str(array.dtype).encode("ascii"),
        np.asarray(array.shape, dtype=np.int64).tobytes(),
        array.tobytes(),
    )


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def resolve_output(path: str | Path) -> Path:
                                                      

    output = _resolve(path)
    allowed = (ROOT / "results").resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("RQ5 output must be below results") from exc
    if output == allowed or not output.name.startswith("rq5_retention_"):
        raise ValueError("RQ5 output leaf must start with rq5_retention_")
    return output


def _load_model_api() -> Any:
                                                                          

    global _MODEL_MODULE
    if _MODEL_MODULE is not None:
        return _MODEL_MODULE
    model_path = ROOT / "experiment" / "rq4" / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    spec = importlib.util.spec_from_file_location(_MODEL_MODULE_NAME, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load history-safe model: {model_path}")
    module = importlib.util.module_from_spec(spec)
                                                                             
                                                                 
    sys.modules[_MODEL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _MODEL_MODULE = module
    return module


def _new_model(base: Mapping[str, Any], seed: int) -> Any:
    api = _load_model_api()
    return api.GPURobustnessModel(
        num_users=SHAPE[1],
        num_services=SHAPE[2],
        rank=int(base["rank"]),
        seed=int(seed),
        bias_eta=float(base["bias_eta"]),
        factor_eta=float(base["factor_eta"]),
        l2=float(base["l2"]),
        beta=float(base["beta"]),
        factor_init_std=float(base["factor_init_std"]),
        num_edges=int(base["num_edges"]),
        variant="full",
    )


def model_state_snapshot(model: Any) -> dict[str, np.ndarray]:
    snapshot = model.full_state_snapshot()
    if not isinstance(snapshot, Mapping):
        raise TypeError("model.full_state_snapshot() must return a mapping")
    return {
        str(name): np.ascontiguousarray(np.asarray(value)).copy()
        for name, value in snapshot.items()
    }


def model_state_hash_and_bytes(model: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for name, value in sorted(model_state_snapshot(model).items()):
        array = np.ascontiguousarray(value)
        total_bytes += int(array.nbytes)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest(), total_bytes


def pack_query_indices(users: np.ndarray, services: np.ndarray) -> np.ndarray:
    users = np.asarray(users, dtype=np.int64)
    services = np.asarray(services, dtype=np.int64)
    if users.ndim != 1 or services.ndim != 1 or users.shape != services.shape:
        raise ValueError("query users/services must be equal one-dimensional arrays")
    flat = users * SHAPE[2] + services
    if np.any(flat < 0) or np.any(flat >= SHAPE[1] * SHAPE[2]):
        raise IndexError("query index outside Dataset 3 dimensions")
    return flat.astype(np.uint32, copy=False)


def unpack_query_indices(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(flat, dtype=np.int64)
    return indices // SHAPE[2], indices % SHAPE[2]


def _metric_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{name: float(metrics[name]) for name in METRICS},
        "absolute_error_sum": float(metrics["absolute_error_sum"]),
        "squared_error_sum": float(metrics["squared_error_sum"]),
        "query_count": int(metrics["count"]),
    }


def _metric_values(
    truth: np.ndarray, estimate: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    target = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(estimate, dtype=np.float64)
    metrics = prediction_metrics(target, prediction)
    relative = np.abs(prediction - target) / target
    return metrics, np.asarray(relative, dtype=np.float64)


def summarize_current_metrics(
    rows: Sequence[Mapping[str, Any]], relative_chunks: Sequence[np.ndarray]
) -> dict[str, Any]:
    count = sum(int(row["query_count"]) for row in rows)
    if count <= 0 or not relative_chunks:
        raise ValueError("no current-query observations to summarize")
    relative = np.concatenate(
        [np.asarray(chunk, dtype=np.float64).reshape(-1) for chunk in relative_chunks]
    )
    relative.sort()
    absolute = sum(float(row["absolute_error_sum"]) for row in rows)
    squared = sum(float(row["squared_error_sum"]) for row in rows)
    return {
        "pooled": {
            "mae": float(absolute / count),
            "rmse": float(np.sqrt(squared / count)),
            "mre": float(np.median(relative)),
            "npre": float(relative[int(0.9 * count)]),
            "count": int(count),
        },
        "time_mean": {
            name: float(np.mean([float(row[name]) for row in rows], dtype=np.float64))
            for name in METRICS
        },
        "definitions": {
            "pooled_mae_rmse": "pooled over all current query observations",
            "pooled_mre": "median pooled absolute relative error",
            "pooled_npre": "pooled absolute relative error at index int(0.9*n)",
            "time_mean": "unweighted mean over time slices",
        },
    }


def compute_forgetting_summary(
    history_rows: Sequence[Mapping[str, Any]], slices: int
) -> dict[str, Any]:
                                                            

                                                      
                                                                          
                                                                      
       

    if slices < 2:
        return {"eligible": False, "reason": "at least two slices are required", "metrics": {}}
    lookup: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in history_rows:
        key = (int(row["checkpoint_slice"]), int(row["historical_slice"]))
        if key in lookup:
            raise ValueError(f"duplicate history key: {key}")
        lookup[key] = row
    expected = slices * (slices + 1) // 2
    if len(lookup) != expected:
        raise ValueError(f"history matrix is incomplete: {len(lookup)} != {expected}")
    result: dict[str, Any] = {}
    final_t = slices - 1
    for metric in METRICS:
        per_history: list[dict[str, Any]] = []
        for historical_slice in range(slices - 1):
            trajectory = [
                float(lookup[(checkpoint, historical_slice)][metric])
                for checkpoint in range(historical_slice, slices)
            ]
            diagonal = float(lookup[(historical_slice, historical_slice)][metric])
            final = float(lookup[(final_t, historical_slice)][metric])
            best = min(trajectory)
            forgetting = final - best
            final_change = final - diagonal
            per_history.append(
                {
                    "historical_slice": historical_slice,
                    "best_error": best,
                    "diagonal_error": diagonal,
                    "final_error": final,
                    "forgetting": forgetting,
                    "forgetting_percent_of_best": 100.0 * forgetting / max(abs(best), 1e-12),
                    "final_history_change": final_change,
                    "final_history_change_percent": 100.0
                    * final_change
                    / max(abs(diagonal), 1e-12),
                }
            )
        result[metric] = {
            "average_forgetting": float(
                np.mean([r["forgetting"] for r in per_history], dtype=np.float64)
            ),
            "average_forgetting_percent": float(
                np.mean(
                    [r["forgetting_percent_of_best"] for r in per_history], dtype=np.float64
                )
            ),
            "mean_final_history_change": float(
                np.mean([r["final_history_change"] for r in per_history], dtype=np.float64)
            ),
            "mean_final_history_change_percent": float(
                np.mean(
                    [r["final_history_change_percent"] for r in per_history], dtype=np.float64
                )
            ),
            "error_bwt_score": float(
                -np.mean([r["final_history_change"] for r in per_history], dtype=np.float64)
            ),
            "per_historical_slice": per_history,
        }
    return {
        "eligible": True,
        "primary_metrics": list(PRIMARY_FORGETTING_METRICS),
        "auxiliary_metrics": ["mre", "npre"],
        "sign_convention": {
            "average_forgetting": "nonnegative; lower is better",
            "final_history_change": "positive is worse, negative is backward improvement",
            "error_bwt_score": "negative of final_history_change; higher is better",
        },
        "metrics": result,
    }


def _validate_base_config(base: Mapping[str, Any]) -> None:
    if tuple(base.get("shape_time_user_service", ())) != SHAPE:
        raise ValueError("base config must use Dataset 3 shape 64x142x4500")
    required = (
        "cache_dir",
        "rank",
        "bias_eta",
        "factor_eta",
        "bias_epochs",
        "factor_epochs",
        "l2",
        "beta",
        "factor_init_std",
        "num_edges",
        "raw_prediction_caps",
    )
    missing = [name for name in required if name not in base]
    if missing:
        raise ValueError(f"base config is missing frozen fields: {missing}")
    if base["raw_prediction_caps"] != {"rt": 20.0, "tp": 10000.0}:
        raise ValueError("base config raw prediction caps differ from frozen v1")


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL}")
    if config.get("rq", "RQ5") != "RQ5":
        raise ValueError("RQ5 config must identify rq=RQ5")
    if not config.get("base_config") or not config.get("output_dir"):
        raise ValueError("base_config and output_dir are required")
    status = str(config.get("status", "PREPARED_NOT_AUTHORIZED"))
    if status not in {"PREPARED_NOT_AUTHORIZED", "FROZEN"}:
        raise ValueError("status must be PREPARED_NOT_AUTHORIZED or FROZEN")
    densities = tuple(float(value) for value in config.get("densities", ()))
    seeds = tuple(int(value) for value in config.get("seeds", ()))
    qos_types = tuple(str(value).lower() for value in config.get("qos_types", ()))
    slices = int(config.get("run_time_slices", 0))
    if len(set(densities)) != len(densities):
        raise ValueError("densities must not contain duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if len(set(qos_types)) != len(qos_types):
        raise ValueError("qos_types must not contain duplicates")
    if not densities or any(value not in FORMAL_DENSITIES + SUPPLEMENT_DENSITIES for value in densities):
        raise ValueError("densities must be a non-empty subset of D01/D05/D10/D15/D20/D25")
    if not seeds or any(value not in FORMAL_SEEDS for value in seeds):
        raise ValueError("seeds must be a non-empty subset of the frozen seeds")
    if not qos_types or any(value not in FORMAL_QOS for value in qos_types):
        raise ValueError("qos_types must be a non-empty subset of rt/tp")
    if slices < 2 or slices > SHAPE[0]:
        raise ValueError("run_time_slices must lie in [2,64] for the historical matrix")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError(f"methods must be exactly {METHODS}")
    if config.get("history_replay") is not True:
        raise ValueError("history_replay must be explicitly true")
    if config.get("query_evaluator_only") is not True:
        raise ValueError("query_evaluator_only must be explicitly true")
    if bool(config.get("pilot_authorized", False)):
        raise ValueError("pilot_authorized is not supported for RQ5")
    if not isinstance(config.get("formal_authorized", False), bool):
        raise ValueError("formal_authorized must be boolean")
                                                                             
                                                                             
                                                                            


def _load_json_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value is None:
        raise ValueError(f"invalid JSON object: {path}")
    return value


def load_config(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = _load_json_object(path)
    validate_experiment_config(config)
    base_path = _resolve(str(config["base_config"]))
    base = _load_json_object(base_path)
    _validate_base_config(base)
    return config, base, base_path


def build_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for density in (float(value) for value in config["densities"]):
        for seed in (int(value) for value in config["seeds"]):
            for qos in (str(value).lower() for value in config["qos_types"]):
                cell_id = f"{qos}_d{int(round(density * 100)):02d}_seed{seed}"
                cells.append(
                    {
                        "protocol": PROTOCOL,
                        "cell_id": cell_id,
                        "qos": qos,
                        "density": density,
                        "seed": seed,
                        "time_slices": int(config["run_time_slices"]),
                    }
                )
    return cells


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    return (
        Path(output)
        / str(cell["qos"])
        / f"d{int(round(float(cell['density']) * 100)):02d}"
        / f"seed{int(cell['seed'])}"
    )


def _new_attempt(cell_dir: Path) -> Path:
    attempts = Path(cell_dir) / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in attempts.glob("attempt_*"):
        try:
            numbers.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = attempts / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _profile_dict(model: Any) -> dict[str, Any]:
    profile = getattr(model, "last_profile", None)
    if profile is None:
        return {}
    if hasattr(profile, "__dict__"):
        result = dict(profile.__dict__)
    else:
        result = {}
        for name in (
            "support_count",
            "active_clients",
            "total_seconds",
            "peak_device_memory_used_bytes",
        ):
            if hasattr(profile, name):
                result[name] = getattr(profile, name)
    return _jsonable(result)


def _raw_cap(base: Mapping[str, Any], qos: str) -> float:
    caps = base["raw_prediction_caps"]
    return float(caps[str(qos).lower()])


def _current_row(
    *,
    cell: Mapping[str, Any],
    method: str,
    time_slice: int,
    split: Any,
    mask_query_hash: str,
    theta0_hash: str,
    initial_hash: str,
    post_hash: str,
    query_before: str,
    query_after: str,
    metrics: Mapping[str, Any],
    profile: Mapping[str, Any],
    update_seconds: float,
    predict_seconds: float,
    history_seconds: float,
    reset_ok: bool,
    retain_ok: bool,
    process_rss: int,
    active_clients: Sequence[int],
    proposal_count: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol": PROTOCOL,
        "cell_id": str(cell["cell_id"]),
        "method": method,
        "method_display": METHOD_DISPLAY[method],
        "qos": str(cell["qos"]),
        "density": float(cell["density"]),
        "seed": int(cell["seed"]),
        "time_slice": int(time_slice),
        "support_entries": int(split.support_count),
        "query_entries": int(split.query_count),
        "mask_sha256": str(split.mask_sha256),
        "query_indices_sha256": mask_query_hash,
        "theta0_state_sha256": theta0_hash,
        "initial_or_inherited_state_sha256": initial_hash,
        "post_update_state_sha256": post_hash,
        "query_state_hash_before": query_before,
        "query_state_hash_after": query_after,
        "query_state_unchanged": query_before == query_after,
        "query_or_future_used": False,
        "query_labels_used_for_update": False,
        "future_slices_used": False,
        "reset_matches_fixed_theta0": bool(reset_ok),
        "retain_inherits_previous_state": bool(retain_ok),
        "model_state_bytes": int(profile.get("model_state_bytes", 0)),
        "update_wall_seconds": float(update_seconds),
        "predict_wall_seconds": float(predict_seconds),
        "history_replay_wall_seconds": float(history_seconds),
        "gpu_peak_memory_used_bytes": int(
            profile.get("peak_device_memory_used_bytes", 0)
        ),
        "process_rss_bytes": int(process_rss),
        "active_clients": json.dumps(list(active_clients)),
        "proposal_count": int(proposal_count),
    }
    row.update(_metric_row(metrics))
    row["query_count"] = int(metrics["count"])
    return row


def _history_row(
    *,
    cell: Mapping[str, Any],
    method: str,
    checkpoint_slice: int,
    historical_slice: int,
    mask_sha256: str,
    query_entries: int,
    before: str,
    after: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "cell_id": str(cell["cell_id"]),
        "method": method,
        "method_display": METHOD_DISPLAY[method],
        "qos": str(cell["qos"]),
        "density": float(cell["density"]),
        "seed": int(cell["seed"]),
        "checkpoint_slice": int(checkpoint_slice),
        "historical_slice": int(historical_slice),
        "historical_mask_sha256": str(mask_sha256),
        "query_entries": int(query_entries),
        "history_replay_state_hash_before": before,
        "history_replay_state_hash_after": after,
        "history_replay_batch_read_only": before == after,
        "query_labels_evaluator_only": True,
        "future_slices_used": False,
        **_metric_row(metrics),
    }


def _validate_prediction(
    truth: np.ndarray, prediction: np.ndarray, raw_cap: float
) -> None:
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("prediction and query truth have different or empty shapes")
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("prediction contains non-finite values")
    if np.any(prediction < 0.0) or np.any(prediction > raw_cap):
        raise ValueError("prediction is outside configured raw QoS cap")


def _run_cell(
    *,
    cell: Mapping[str, Any],
    values: np.ndarray,
    base: Mapping[str, Any],
    output: Path,
    source_sha256: Mapping[str, str | None],
    config_sha256: str,
    data_sha256: Mapping[str, str | None],
    resume: bool = False,
    model_factory: Callable[[Mapping[str, Any], int], Any] | None = None,
) -> dict[str, Any]:
                                                                        

    del resume                                                               
    cell_dir = _cell_dir(output, cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    slices = int(cell["time_slices"])
    if model_factory is None:
        model_factory = _new_model
    attempt = _new_attempt(cell_dir)
    status: dict[str, Any] = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        **dict(cell),
        "time_slices_expected": slices,
        "time_slices_completed": 0,
        "attempt_dir": str(attempt.resolve()),
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": dict(data_sha256),
        "query_or_future_used": False,
        "query_or_future_used_for_update": False,
        "history_replay": True,
        "query_evaluator_only": True,
        "created_at": _utc_now(),
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)

    process = psutil.Process(os.getpid()) if psutil is not None else None
    peak_rss = int(process.memory_info().rss) if process is not None else 0
    started = time.perf_counter()
    retain_model: Any | None = None
    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    relative_chunks: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    current_by_method: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    history_by_method: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    query_history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    previous_retain_hash: str | None = None
    theta0_hash: str | None = None
    theta0_bytes: int | None = None
    raw_cap = _raw_cap(base, str(cell["qos"]))

    try:
        retain_model = model_factory(base, int(cell["seed"]))
        theta0_hash, theta0_bytes = model_state_hash_and_bytes(retain_model)
        for time_slice in range(slices):
            split = make_causal_split(
                np.asarray(values[time_slice]),
                float(cell["density"]),
                int(cell["seed"]),
                int(time_slice),
            )
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

            query_flat = pack_query_indices(split.query_users, split.query_services)
            query_hash = _hash_array(query_flat)
            query_history.append(
                (
                    split.query_users.copy(),
                    split.query_services.copy(),
                    np.asarray(split.query_values, dtype=np.float64).copy(),
                    str(split.mask_sha256),
                )
            )
            reset_model = model_factory(base, int(cell["seed"]))
            reset_initial_hash, reset_initial_bytes = model_state_hash_and_bytes(reset_model)
            if reset_initial_hash != theta0_hash or reset_initial_bytes != theta0_bytes:
                raise AssertionError("FedARM-Reset did not reconstruct fixed theta0")
            retain_pre_hash, _ = model_state_hash_and_bytes(retain_model)
            if time_slice == 0:
                retain_ok = retain_pre_hash == theta0_hash
            else:
                retain_ok = retain_pre_hash == previous_retain_hash
            if not retain_ok:
                raise AssertionError("FedARM-Retain state continuity failed")

            for method, model in (
                ("fedsaa_retain", retain_model),
                ("fedsaa_reset", reset_model),
            ):
                initial_hash = retain_pre_hash if method == "fedsaa_retain" else reset_initial_hash
                update_started = time.perf_counter()
                result = model.advance_support(
                    split.support_users,
                    split.support_services,
                    split.support_values,
                    raw_cap,
                    int(base["bias_epochs"]),
                    int(base["factor_epochs"]),
                    int(cell["seed"]) * 1_000_003 + int(time_slice),
                )
                update_seconds = time.perf_counter() - update_started
                post_hash, post_bytes = model_state_hash_and_bytes(model)
                predict_started = time.perf_counter()
                prediction, _prediction_timing = model.predict(
                    split.query_users, split.query_services, raw_cap
                )
                predict_seconds = time.perf_counter() - predict_started
                query_after, query_bytes = model_state_hash_and_bytes(model)
                if post_hash != query_after or post_bytes != query_bytes:
                    raise AssertionError("current query evaluation mutated model state")
                truth = np.asarray(split.query_values, dtype=np.float64)
                estimate = np.asarray(prediction, dtype=np.float64)
                _validate_prediction(truth, estimate, raw_cap)
                metrics, relative = _metric_values(truth, estimate)
                relative_chunks[method].append(relative)

                replay_started = time.perf_counter()
                replay_before, replay_before_bytes = model_state_hash_and_bytes(model)
                method_history: list[dict[str, Any]] = []
                for historical_slice, (
                    historical_users,
                    historical_services,
                    historical_truth,
                    historical_mask,
                ) in enumerate(query_history):
                    historical_prediction, _ = model.predict(
                        historical_users, historical_services, raw_cap
                    )
                    historical_estimate = np.asarray(historical_prediction, dtype=np.float64)
                    _validate_prediction(historical_truth, historical_estimate, raw_cap)
                    historical_metrics, _ = _metric_values(
                        historical_truth, historical_estimate
                    )
                    method_history.append(
                        _history_row(
                            cell=cell,
                            method=method,
                            checkpoint_slice=time_slice,
                            historical_slice=historical_slice,
                            mask_sha256=historical_mask,
                            query_entries=int(historical_truth.size),
                            before=replay_before,
                            after=replay_before,
                            metrics=historical_metrics,
                        )
                    )
                replay_after, replay_after_bytes = model_state_hash_and_bytes(model)
                if replay_before != replay_after or replay_before_bytes != replay_after_bytes:
                    raise AssertionError("historical replay mutated model state")
                replay_seconds = time.perf_counter() - replay_started
                for row in method_history:
                    row["history_replay_state_hash_after"] = replay_after
                    row["history_replay_batch_read_only"] = True
                    row["history_replay_batch_wall_seconds"] = float(replay_seconds)
                history_by_method[method].extend(method_history)
                history_rows.extend(method_history)

                profile = _profile_dict(model)
                profile["model_state_bytes"] = int(post_bytes)
                diagnostics = getattr(model, "last_diagnostics", {})
                active_clients = tuple(int(value) for value in diagnostics.get("active_clients", []))
                proposal_count = int(diagnostics.get("proposal_count", split.support_count))
                rss = int(process.memory_info().rss) if process is not None else 0
                peak_rss = max(peak_rss, rss)
                row = _current_row(
                    cell=cell,
                    method=method,
                    time_slice=time_slice,
                    split=split,
                    mask_query_hash=query_hash,
                    theta0_hash=str(theta0_hash),
                    initial_hash=initial_hash,
                    post_hash=post_hash,
                    query_before=post_hash,
                    query_after=query_after,
                    metrics=metrics,
                    profile=profile,
                    update_seconds=update_seconds,
                    predict_seconds=predict_seconds,
                    history_seconds=replay_seconds,
                    reset_ok=(method != "fedsaa_reset" or reset_initial_hash == theta0_hash),
                    retain_ok=(method != "fedsaa_retain" or retain_ok),
                    process_rss=rss,
                    active_clients=active_clients,
                    proposal_count=proposal_count,
                )
                current_by_method[method].append(row)
                rows.append(row)
                if method == "fedsaa_retain":
                    previous_retain_hash = replay_after

            _atomic_csv(attempt / "current_per_slice.csv", rows, CURRENT_FIELDS)
            _atomic_csv(attempt / "history_matrix.csv", history_rows, HISTORY_FIELDS)
            status = {
                **status,
                "time_slices_completed": time_slice + 1,
                "updated_at": _utc_now(),
            }
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)
            del reset_model
            gc.collect()

        summaries: dict[str, Any] = {}
        for method in METHODS:
            summaries[method] = {
                "method_display": METHOD_DISPLAY[method],
                "current_query_metrics": summarize_current_metrics(
                    current_by_method[method], relative_chunks[method]
                ),
                "historical_retention": compute_forgetting_summary(
                    history_by_method[method], slices
                ),
                "update_wall_seconds": float(
                    sum(float(row["update_wall_seconds"]) for row in current_by_method[method])
                ),
                "current_predict_wall_seconds": float(
                    sum(float(row["predict_wall_seconds"]) for row in current_by_method[method])
                ),
                "history_replay_wall_seconds": float(
                    sum(float(row["history_replay_wall_seconds"]) for row in current_by_method[method])
                ),
                "peak_model_state_bytes": int(
                    max(int(row["model_state_bytes"]) for row in current_by_method[method])
                ),
                "peak_gpu_memory_used_bytes": int(
                    max(int(row["gpu_peak_memory_used_bytes"]) for row in current_by_method[method])
                ),
            }
        summary: dict[str, Any] = {
            "status": "COMPLETED",
            "protocol": PROTOCOL,
            **dict(cell),
            "attempt_dir": str(attempt.resolve()),
            "time_slices_completed": slices,
            "method_trajectories": len(METHODS),
            "history_points_per_method": slices * (slices + 1) // 2,
            "theta0_state_sha256": str(theta0_hash),
            "theta0_state_bytes": int(theta0_bytes or 0),
            "all_reset_initial_states_match_theta0": all(
                bool(row["reset_matches_fixed_theta0"])
                for row in current_by_method["fedsaa_reset"]
            ),
            "retain_state_continuity_passed": all(
                bool(row["retain_inherits_previous_state"])
                for row in current_by_method["fedsaa_retain"]
            ),
            "all_current_queries_read_only": all(
                bool(row["query_state_unchanged"]) for row in rows
            ),
            "all_history_replay_batches_read_only": all(
                bool(row["history_replay_batch_read_only"]) for row in history_rows
            ),
            "final_state_sha256_by_method": {
                method: current_by_method[method][-1]["post_update_state_sha256"]
                for method in METHODS
            },
            "query_or_future_used_for_update": False,
            "finite_metrics_all_slices": all(
                np.isfinite(float(row[name])) for row in rows for name in METRICS
            ),
            "process_peak_rss_bytes": int(peak_rss),
            "cell_wall_seconds": float(time.perf_counter() - started),
            "source_sha256": dict(source_sha256),
            "config_sha256": str(config_sha256),
            "data_sha256": dict(data_sha256),
            "methods": summaries,
            "interpretation_boundary": {
                "retain_vs_reset": (
                    "Tests the value of cross-slice state accumulation; it does not alone "
                    "establish catastrophic-forgetting mitigation."
                ),
                "history_matrix": (
                    "Evaluates each fixed historical query set after every checkpoint; "
                    "query labels are evaluator-only."
                ),
            },
            "completed_at": _utc_now(),
        }
        summary["artifact_sha256"] = {
            "current_per_slice.csv": sha256_file(attempt / "current_per_slice.csv"),
            "history_matrix.csv": sha256_file(attempt / "history_matrix.csv"),
        }
        _atomic_json(attempt / "summary.json", summary)
        _atomic_json(attempt / "status.json", summary)
        _atomic_json(cell_dir / "summary.json", summary)
        _atomic_json(cell_dir / "status.json", summary)
        return summary
    except Exception as exc:
        failure = {
            **status,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "failed_at": _utc_now(),
        }
        _atomic_json(attempt / "status.json", failure)
        _atomic_json(cell_dir / "status.json", failure)
        raise
    finally:
        if retain_model is not None:
            del retain_model
        gc.collect()


def _cell_artifacts_valid(
    cell_dir: Path,
    summary: Mapping[str, Any],
    slices: int,
) -> bool:
                                                                       

    try:
        attempt = Path(str(summary["attempt_dir"])).resolve()
        if attempt.parent != (Path(cell_dir) / "attempts").resolve():
            return False
        if _read_json(attempt / "summary.json") != dict(summary):
            return False
        if _read_json(attempt / "status.json") != dict(summary):
            return False
        artifacts = summary.get("artifact_sha256")
        if not isinstance(artifacts, Mapping):
            return False
        for name in ("current_per_slice.csv", "history_matrix.csv"):
            if artifacts.get(name) != sha256_file(attempt / name):
                return False
        current = _read_csv(attempt / "current_per_slice.csv")
        expected_current = {(method, t) for method in METHODS for t in range(slices)}
        actual_current = {(row["method"], int(row["time_slice"])) for row in current}
        if len(current) != len(expected_current) or actual_current != expected_current:
            return False
        current_by_key = {
            (row["method"], int(row["time_slice"])): row for row in current
        }
        for row in current:
            if row.get("protocol") != PROTOCOL or row.get("cell_id") != summary.get("cell_id"):
                return False
            if row.get("method_display") != METHOD_DISPLAY.get(row.get("method", "")):
                return False
            if row.get("qos") != str(summary.get("qos")):
                return False
            if int(row.get("seed", -1)) != int(summary.get("seed", -2)):
                return False
            if not np.isclose(float(row.get("density", -1)), float(summary.get("density", -2))):
                return False
            if any(row.get(name, "") == "" for name in METRICS):
                return False
            if any(not np.isfinite(float(row[name])) for name in METRICS):
                return False
            if int(row.get("time_slice", -1)) < 0 or int(row["time_slice"]) >= slices:
                return False
            if row.get("query_state_hash_before") != row.get("query_state_hash_after"):
                return False
            if row.get("post_update_state_sha256") != row.get("query_state_hash_before"):
                return False
            if row.get("query_state_unchanged", "false").lower() != "true":
                return False
            if row.get("query_or_future_used", "true").lower() != "false":
                return False
            if row.get("query_labels_used_for_update", "true").lower() != "false":
                return False
            if row.get("future_slices_used", "true").lower() != "false":
                return False
            if row["method"] == "fedsaa_reset" and row.get(
                "reset_matches_fixed_theta0", "false"
            ).lower() != "true":
                return False
            if row["method"] == "fedsaa_retain" and row.get(
                "retain_inherits_previous_state", "false"
            ).lower() != "true":
                return False
            if row.get("theta0_state_sha256") != summary.get("theta0_state_sha256"):
                return False
            if int(row.get("query_entries", -1)) != int(row.get("query_count", -2)):
                return False

                                                                            
                                                                 
        for time_slice in range(slices):
            retain = current_by_key[("fedsaa_retain", time_slice)]
            reset = current_by_key[("fedsaa_reset", time_slice)]
            if retain["mask_sha256"] != reset["mask_sha256"]:
                return False
            if retain["query_indices_sha256"] != reset["query_indices_sha256"]:
                return False
            if retain["query_entries"] != reset["query_entries"]:
                return False
            if reset["initial_or_inherited_state_sha256"] != summary.get(
                "theta0_state_sha256"
            ):
                return False
            expected_retain_initial = (
                summary.get("theta0_state_sha256")
                if time_slice == 0
                else current_by_key[("fedsaa_retain", time_slice - 1)][
                    "post_update_state_sha256"
                ]
            )
            if retain["initial_or_inherited_state_sha256"] != expected_retain_initial:
                return False
        history = _read_csv(attempt / "history_matrix.csv")
        expected_history = {
            (method, checkpoint, historical)
            for method in METHODS
            for checkpoint in range(slices)
            for historical in range(checkpoint + 1)
        }
        actual_history = {
            (row["method"], int(row["checkpoint_slice"]), int(row["historical_slice"]))
            for row in history
        }
        if len(history) != len(expected_history) or actual_history != expected_history:
            return False
        history_by_method: dict[str, list[dict[str, str]]] = {name: [] for name in METHODS}
        for row in history:
            if row.get("protocol") != PROTOCOL or row.get("cell_id") != summary.get("cell_id"):
                return False
            if row.get("method_display") != METHOD_DISPLAY.get(row.get("method", "")):
                return False
            checkpoint = int(row.get("checkpoint_slice", -1))
            historical = int(row.get("historical_slice", -1))
            if checkpoint < 0 or checkpoint >= slices or historical < 0 or historical > checkpoint:
                return False
            linked_current = current_by_key[(row["method"], historical)]
            checkpoint_current = current_by_key[(row["method"], checkpoint)]
            if row.get("historical_mask_sha256") != linked_current.get("mask_sha256"):
                return False
            if int(row.get("query_entries", -1)) != int(linked_current.get("query_entries", -2)):
                return False
            if row.get("history_replay_state_hash_before") != checkpoint_current.get(
                "post_update_state_sha256"
            ):
                return False
            if row.get("history_replay_state_hash_after") != checkpoint_current.get(
                "post_update_state_sha256"
            ):
                return False
            if any(row.get(name, "") == "" for name in METRICS):
                return False
            if any(not np.isfinite(float(row[name])) for name in METRICS):
                return False
            if row.get("history_replay_state_hash_before") != row.get(
                "history_replay_state_hash_after"
            ):
                return False
            if row.get("history_replay_batch_read_only", "false").lower() != "true":
                return False
            if row.get("query_labels_evaluator_only", "false").lower() != "true":
                return False
            if row.get("future_slices_used", "true").lower() != "false":
                return False
            history_by_method[row["method"]].append(row)

                                                                        
                                                                            
        methods_summary = summary.get("methods")
        if not isinstance(methods_summary, Mapping):
            return False
        for method in METHODS:
            expected_retention = compute_forgetting_summary(history_by_method[method], slices)
            stored_method = methods_summary.get(method)
            if not isinstance(stored_method, Mapping):
                return False
            stored_retention = stored_method.get("historical_retention")
            if not isinstance(stored_retention, Mapping):
                return False
            for metric in METRICS:
                expected_metric = expected_retention["metrics"][metric]
                stored_metric = stored_retention.get("metrics", {}).get(metric)
                if not isinstance(stored_metric, Mapping):
                    return False
                for key in (
                    "average_forgetting",
                    "average_forgetting_percent",
                    "mean_final_history_change",
                    "mean_final_history_change_percent",
                    "error_bwt_score",
                ):
                    if not np.isclose(
                        float(stored_metric.get(key, np.nan)),
                        float(expected_metric[key]),
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        return False
            final_row = current_by_key[(method, slices - 1)]
            final_hashes = summary.get("final_state_sha256_by_method", {})
            if not isinstance(final_hashes, Mapping) or final_hashes.get(method) != final_row.get(
                "post_update_state_sha256"
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, FloatingPointError):
        return False


def _completed(
    cell_dir: Path,
    cell: Mapping[str, Any],
    source_sha256: Mapping[str, str | None],
    config_sha256: str,
    data_sha256: Mapping[str, str | None],
) -> bool:
    summary = _read_json(Path(cell_dir) / "summary.json")
    status = _read_json(Path(cell_dir) / "status.json")
    if summary is None or status is None:
        return False
    if summary.get("status") != "COMPLETED" or status.get("status") != "COMPLETED":
        return False
    if summary.get("protocol") != PROTOCOL:
        return False
    for key in ("cell_id", "qos", "seed"):
        if str(summary.get(key, "")) != str(cell.get(key, "")):
            return False
    if not np.isclose(float(summary.get("density", -1)), float(cell.get("density", -2))):
        return False
    slices = int(cell["time_slices"])
    if int(summary.get("time_slices_completed", -1)) != slices:
        return False
    if summary.get("source_sha256") != dict(source_sha256):
        return False
    if summary.get("config_sha256") != str(config_sha256):
        return False
    if summary.get("data_sha256") != dict(data_sha256):
        return False
    if not summary.get("finite_metrics_all_slices"):
        return False
    if not summary.get("all_current_queries_read_only"):
        return False
    if not summary.get("all_history_replay_batches_read_only"):
        return False
    if not summary.get("retain_state_continuity_passed"):
        return False
    if not summary.get("all_reset_initial_states_match_theta0"):
        return False
    if summary.get("query_or_future_used_for_update"):
        return False
    return _cell_artifacts_valid(Path(cell_dir), summary, slices)


def _source_paths() -> dict[str, Path]:
    return {
        "isolated_engine": HERE / "engine.py",
        "isolated_runner": HERE / "run_retention.py",
        "historysafe_model": ROOT / "experiment" / "rq4" / "model.py",
        "frozen_gpu_ablation": ROOT / "code" / "core" / "gpu_ablation_training.py",
        "frozen_gpu_full": ROOT / "code" / "core" / "gpu_full_training.py",
        "frozen_causal_split": ROOT / "code" / "core" / "protocol.py",
        "frozen_data_loader": ROOT / "code" / "core" / "data.py",
                                                                        
                                                                            
        "frozen_sdq8_accounting": ROOT
        / "code"
        / "core"
        / "sdq8_attribution_controls.py",
        "frozen_saq8_codec": ROOT / "code" / "core" / "hmp_sr16_codec.py",
    }


def _snapshot_sources(
    output: Path,
    *,
    config_path: Path,
    base_path: Path,
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
                                                                              

    source_paths = dict(source_paths or _source_paths())
    source_paths["experiment_config"] = Path(config_path)
    source_paths["base_config"] = Path(base_path)
    snapshot_dir = Path(output) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    for name, source in source_paths.items():
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = snapshot_dir / f"{name}{source.suffix}"
        shutil.copyfile(source, destination)
        entries[name] = str(destination.resolve())
    return entries


def _snapshots_match(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    config_path: Path,
    base_path: Path,
) -> bool:
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, Mapping):
        return False
    expected: dict[str, str | None] = {
        **dict(manifest.get("source_sha256", {})),
        "experiment_config": manifest.get("config_sha256"),
        "base_config": manifest.get("base_config_sha256"),
    }
    for name, expected_hash in expected.items():
        if not isinstance(expected_hash, str):
            return False
        raw_path = Path(str(snapshots.get(name, "")))
        if not raw_path.is_file() or sha256_file(raw_path) != expected_hash:
            return False
                                                                         
                                                                        
    snapshot_root = (Path(output) / "snapshots").resolve()
    for raw in snapshots.values():
        try:
            Path(str(raw)).resolve().relative_to(snapshot_root)
        except ValueError:
            return False
    del config_path, base_path                                                  
    return True


def _base_cache_dir(base: Mapping[str, Any]) -> Path:
    return _resolve(str(base["cache_dir"]))


def build_manifest(
    config_path: str | Path,
    config: Mapping[str, Any],
    base_path: Path,
    base: Mapping[str, Any],
    *,
    read_data: bool = True,
) -> dict[str, Any]:
    cache_dir = _base_cache_dir(base)
    data_sha256 = {
        str(qos): (
            _sha256_if_file(cache_dir / f"{str(qos).lower()}_values.npy")
            if read_data
            else None
        )
        for qos in config["qos_types"]
    }
    source_sha256 = {
        name: _sha256_if_file(path) for name, path in _source_paths().items()
    }
    return {
        "protocol": PROTOCOL,
        "status": str(config.get("status", "PREPARED_NOT_AUTHORIZED")),
        "created_at": _utc_now(),
        "workspace": str(ROOT),
        "output": str(resolve_output(str(config["output_dir"]))),
        "shape_time_user_service": list(SHAPE),
        "methods": list(METHODS),
        "method_display": dict(METHOD_DISPLAY),
        "densities": [float(value) for value in config["densities"]],
        "seeds": [int(value) for value in config["seeds"]],
        "qos_types": [str(value).lower() for value in config["qos_types"]],
        "time_slices": int(config["run_time_slices"]),
        "stream_conditions": len(build_cells(config)),
        "method_trajectories": len(build_cells(config)) * len(METHODS),
        "history_points_per_method_trajectory": int(config["run_time_slices"])
        * (int(config["run_time_slices"]) + 1)
        // 2,
        "formal_authorized": bool(config.get("formal_authorized", False)),
        "formal_execution": int(config["run_time_slices"]) == SHAPE[0],
        "history_replay": True,
        "query_evaluator_only": True,
        "causal_boundary": (
            "current support labels update state; current and historical query labels "
            "are evaluator-only; future slices are not accessed"
        ),
        "state_hash_scope": "complete predictive state plus committed CARW history",
        "source_sha256": source_sha256,
        "config_sha256": sha256_file(Path(config_path).resolve()),
        "base_config_sha256": sha256_file(base_path),
        "data_sha256": data_sha256,
        "snapshot_contract": "source/config/base byte snapshots are stored below snapshots/",
        "backend": "Numba CUDA via history-safe RQ5 model wrapper",
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
    }


class RunnerLock:
                                                                     

    def __init__(self, output: Path) -> None:
        self.path = Path(output) / ".rq5_runner.lock"
        self.fd: int | None = None

    def __enter__(self) -> "RunnerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "created_at": _utc_now(),
            "purpose": "RQ5 retention runner",
        }
        try:
            self.fd = os.open(
                str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            os.write(self.fd, (json.dumps(payload) + "\n").encode("utf-8"))
            os.close(self.fd)
            self.fd = None
        except FileExistsError as exc:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            raise RuntimeError(
                f"RQ5 runner lock exists; inspect/remove only after confirming no active runner: {self.path}"
            ) from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _read_manifest(output: Path) -> dict[str, Any]:
    manifest = _read_json(Path(output) / "manifest.json")
    if manifest is None:
        raise ValueError(f"missing or invalid manifest: {output / 'manifest.json'}")
    return manifest


def _manifest_matches(existing: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    keys = (
        "protocol",
        "methods",
        "densities",
        "seeds",
        "qos_types",
        "time_slices",
        "source_sha256",
        "config_sha256",
        "base_config_sha256",
        "data_sha256",
    )
    return all(existing.get(key) == current.get(key) for key in keys)


def _queue_status(
    output: Path,
    *,
    status: str,
    cells: Sequence[Mapping[str, Any]],
    completed: int,
    failed: int,
    errors: Sequence[str] = (),
) -> None:
    _atomic_json(
        Path(output) / "queue_status.json",
        {
            "status": status,
            "protocol": PROTOCOL,
            "total_cells": len(cells),
            "completed_cells": int(completed),
            "failed_cells": int(failed),
            "pending_cells": max(0, len(cells) - completed - failed),
            "errors": list(errors),
            "updated_at": _utc_now(),
        },
    )


def _aggregate_terminal(output: Path, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current_all: list[dict[str, Any]] = []
    retention_all: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    for cell in cells:
        cell_dir = _cell_dir(output, cell)
        summary = _read_json(cell_dir / "summary.json")
        if summary is None:
            raise ValueError(f"missing completed cell summary: {cell_dir}")
        cell_summaries.append(summary)
        attempt = Path(str(summary["attempt_dir"]))
        current_all.extend(_read_csv(attempt / "current_per_slice.csv"))
        for method in METHODS:
            retention = summary["methods"][method]["historical_retention"]
            for metric in METRICS:
                details = retention["metrics"][metric]
                retention_all.append(
                    {
                        "protocol": PROTOCOL,
                        "cell_id": cell["cell_id"],
                        "qos": cell["qos"],
                        "density": cell["density"],
                        "seed": cell["seed"],
                        "method": method,
                        "method_display": METHOD_DISPLAY[method],
                        "metric": metric,
                        "average_forgetting": details["average_forgetting"],
                        "average_forgetting_percent": details["average_forgetting_percent"],
                        "mean_final_history_change": details["mean_final_history_change"],
                        "mean_final_history_change_percent": details[
                            "mean_final_history_change_percent"
                        ],
                        "error_bwt_score": details["error_bwt_score"],
                    }
                )
    _atomic_csv(
        output / "current_metrics.csv",
        current_all,
        CURRENT_FIELDS,
    )
    _atomic_csv(
        output / "retention_summary.csv",
        retention_all,
        (
            "protocol",
            "cell_id",
            "qos",
            "density",
            "seed",
            "method",
            "method_display",
            "metric",
            "average_forgetting",
            "average_forgetting_percent",
            "mean_final_history_change",
            "mean_final_history_change_percent",
            "error_bwt_score",
        ),
    )
    final = {
        "status": "COMPLETED",
        "protocol": PROTOCOL,
        "stream_conditions_completed": len(cell_summaries),
        "method_trajectories_completed": len(cell_summaries) * len(METHODS),
        "all_structural_gates_passed": all(
            bool(summary.get("finite_metrics_all_slices"))
            and bool(summary.get("all_current_queries_read_only"))
            and bool(summary.get("all_history_replay_batches_read_only"))
            and bool(summary.get("retain_state_continuity_passed"))
            and bool(summary.get("all_reset_initial_states_match_theta0"))
            and not bool(summary.get("query_or_future_used_for_update"))
            for summary in cell_summaries
        ),
        "completed_at": _utc_now(),
    }
    _atomic_json(output / "final_summary.json", final)
    terminal_hashes = {
        name: sha256_file(output / name)
        for name in ("final_summary.json", "current_metrics.csv", "retention_summary.csv")
    }
    marker = {
        "status": TERMINAL_STATUS,
        "protocol": PROTOCOL,
        "scope_cells": len(cells),
        "completed_cells": len(cell_summaries),
        "all_cells_completed": len(cell_summaries) == len(cells),
        "artifact_sha256": terminal_hashes,
        "cell_summary_sha256": {
            str(summary["cell_id"]): sha256_file(
                _cell_dir(output, next(cell for cell in cells if cell["cell_id"] == summary["cell_id"]))
                / "summary.json"
            )
            for summary in cell_summaries
        },
        "created_at": _utc_now(),
    }
    _atomic_json(output / "BATCH_COMPLETED", marker)
    return final


def audit_batch(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
                                                                

    config, base, base_path = load_config(config_path)
    current_manifest = build_manifest(config_path, config, base_path, base, read_data=True)
    output_path = resolve_output(output or str(config["output_dir"]))
    if not output_path.is_dir():
        return {
            "status": "MISSING",
            "protocol": PROTOCOL,
            "output": str(output_path),
        }
    existing = _read_manifest(output_path)
    manifest_match = _manifest_matches(existing, current_manifest)
    snapshots_match = _snapshots_match(
        output_path,
        existing,
        config_path=Path(config_path).resolve(),
        base_path=base_path,
    )
    cells = build_cells(config)
    valid_cells = [
        _completed(
            _cell_dir(output_path, cell),
            cell,
            current_manifest["source_sha256"],
            current_manifest["config_sha256"],
            current_manifest["data_sha256"],
        )
        for cell in cells
    ]
    marker = _read_json(output_path / "BATCH_COMPLETED")
    terminal_integrity = False
    if marker is not None:
        try:
            terminal_integrity = (
                marker.get("status") == TERMINAL_STATUS
                and marker.get("protocol") == PROTOCOL
                and marker.get("scope_cells") == len(cells)
                and marker.get("completed_cells") == len(cells)
                and isinstance(marker.get("cell_summary_sha256"), Mapping)
                and all(
                    marker["cell_summary_sha256"].get(cell["cell_id"])
                    == sha256_file(_cell_dir(output_path, cell) / "summary.json")
                    for cell in cells
                )
                and all(
                    marker["artifact_sha256"][name] == sha256_file(output_path / name)
                    for name in (
                        "final_summary.json",
                        "current_metrics.csv",
                        "retention_summary.csv",
                    )
                )
            )
        except (KeyError, OSError, TypeError):
            terminal_integrity = False
    return {
        "status": "COMPLETE"
        if all(valid_cells) and terminal_integrity and manifest_match and snapshots_match
        else "INCOMPLETE",
        "protocol": PROTOCOL,
        "output": str(output_path),
        "manifest_match": manifest_match,
        "snapshots_match": snapshots_match,
        "valid_cells": int(sum(valid_cells)),
        "total_cells": len(cells),
        "terminal_integrity": terminal_integrity,
        "marker": marker,
    }


def run_queue(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    run_formal: bool = False,
    resume: bool = False,
    audit: bool = False,
    output: str | Path | None = None,
    stop_on_error: bool = True,
) -> dict[str, Any]:
                                                              

    config_path = Path(config_path).resolve()
    config, base, base_path = load_config(config_path)
    if audit:
        return audit_batch(config_path, output=output)
    cells = build_cells(config)
    is_formal = int(config["run_time_slices"]) == SHAPE[0]
    if dry_run:
                                                                      
                                                                            
                                                           
        manifest = build_manifest(
            config_path, config, base_path, base, read_data=False
        )
        return {
            "status": "DRY_RUN",
            "protocol": PROTOCOL,
            "formal_scope": is_formal,
            "formal_authorized": bool(config.get("formal_authorized", False)),
            "config_status": config.get("status"),
            "total_cells": len(cells),
            "method_trajectories": len(cells) * len(METHODS),
            "history_points_per_method_trajectory": manifest[
                "history_points_per_method_trajectory"
            ],
            "output": manifest["output"],
            "manifest": manifest,
        }
    if is_formal and (
        not run_formal
        or config.get("status") != "FROZEN"
        or config.get("formal_authorized") is not True
    ):
        raise PermissionError(
            "formal RQ5 execution requires --run-formal, status=FROZEN, and formal_authorized=true"
        )
    if run_formal and not is_formal:
        raise ValueError("--run-formal requires the 64-slice formal configuration")
    if is_formal and (
        tuple(float(value) for value in config["densities"]) not in (FORMAL_DENSITIES, SUPPLEMENT_DENSITIES)
        or tuple(int(value) for value in config["seeds"]) != FORMAL_SEEDS
        or tuple(str(value).lower() for value in config["qos_types"]) != FORMAL_QOS
    ):
        raise ValueError("formal execution requires complete D01/D10/D25 or D05/D15/D20 x five-seed x RT/TP matrix")
    if not is_formal:
        smoke_shape = (
            tuple(float(value) for value in config["densities"]) == (0.01,)
            and tuple(int(value) for value in config["seeds"]) == (42,)
            and tuple(str(value).lower() for value in config["qos_types"]) == FORMAL_QOS
            and 2 <= int(config["run_time_slices"]) <= 20
        )
        if not smoke_shape:
            raise ValueError(
                "non-formal execution is restricted to smoke D01/seed42/RT+TP/at most 20 slices"
            )
                                                                               
                                              
    manifest = build_manifest(config_path, config, base_path, base, read_data=True)
    if any(value is None for value in manifest["data_sha256"].values()):
        raise FileNotFoundError("one or more configured QoS cache files are missing")
    output_path = resolve_output(output or str(config["output_dir"]))
    cache_dir = _base_cache_dir(base)
    for qos in config["qos_types"]:
        cache_path = cache_dir / f"{str(qos).lower()}_values.npy"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
    with RunnerLock(output_path):
        if output_path.exists() and any(
            item.name != ".rq5_runner.lock" for item in output_path.iterdir()
        ):
            if not resume:
                raise FileExistsError(f"RQ5 output exists; use --resume: {output_path}")
            existing = _read_manifest(output_path)
            if not _manifest_matches(existing, manifest):
                raise ValueError("RQ5 resume rejected: source/config/cache hashes differ")
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            manifest = {
                **manifest,
                "snapshots": _snapshot_sources(
                    output_path,
                    config_path=config_path,
                    base_path=base_path,
                ),
            }
            _atomic_json(output_path / "manifest.json", manifest)
        if not (output_path / "manifest.json").is_file():
            _atomic_json(output_path / "manifest.json", manifest)
        else:
                                                                          
                                                                
            existing_manifest = _read_manifest(output_path)
            if not _snapshots_match(
                output_path,
                existing_manifest,
                config_path=config_path,
                base_path=base_path,
            ):
                raise ValueError("RQ5 output source snapshots are missing or corrupted")
        summaries: list[dict[str, Any]] = []
        errors: list[str] = []
        completed_count = 0
        for cell in cells:
            cell_dir = _cell_dir(output_path, cell)
            if _completed(
                cell_dir,
                cell,
                manifest["source_sha256"],
                manifest["config_sha256"],
                manifest["data_sha256"],
            ):
                summary = _read_json(cell_dir / "summary.json") or {}
                summaries.append(summary)
                completed_count += 1
                _queue_status(
                    output_path,
                    status="RUNNING",
                    cells=cells,
                    completed=completed_count,
                    failed=len(errors),
                    errors=errors,
                )
                continue
            values = load_value_cache(
                cache_dir / f"{str(cell['qos']).lower()}_values.npy", SHAPE
            )
            try:
                summary = _run_cell(
                    cell=cell,
                    values=values,
                    base=base,
                    output=output_path,
                    source_sha256=manifest["source_sha256"],
                    config_sha256=manifest["config_sha256"],
                    data_sha256=manifest["data_sha256"],
                    resume=resume,
                )
                summaries.append(summary)
                completed_count += 1
            except Exception as exc:
                errors.append(f"{cell['cell_id']}: {type(exc).__name__}: {exc}")
                _queue_status(
                    output_path,
                    status="FAILED",
                    cells=cells,
                    completed=completed_count,
                    failed=len(errors),
                    errors=errors,
                )
                if stop_on_error:
                    raise
            finally:
                del values
                gc.collect()
            _queue_status(
                output_path,
                status="RUNNING",
                cells=cells,
                completed=completed_count,
                failed=len(errors),
                errors=errors,
            )
        if errors or completed_count != len(cells):
            _queue_status(
                output_path,
                status="FAILED",
                cells=cells,
                completed=completed_count,
                failed=len(errors),
                errors=errors,
            )
            return {
                "status": "INCOMPLETE",
                "protocol": PROTOCOL,
                "completed_cells": completed_count,
                "total_cells": len(cells),
                "errors": errors,
                "output": str(output_path),
            }
        final = _aggregate_terminal(output_path, cells)
        _queue_status(
            output_path,
            status=TERMINAL_STATUS,
            cells=cells,
            completed=len(cells),
            failed=0,
        )
        return {**final, "output": str(output_path)}


def run_experiment(config_path: str | Path, *, dry_run: bool = False, **kwargs: Any) -> dict[str, Any]:
                                                                        

    return run_queue(config_path, dry_run=dry_run, **kwargs)


__all__ = [
    "ACCEPTED_CELL_STATUSES",
    "FORMAL_DENSITIES",
    "FORMAL_QOS",
    "FORMAL_SEEDS",
    "METHODS",
    "METHOD_DISPLAY",
    "METRICS",
    "PROTOCOL",
    "ROOT",
    "SHAPE",
    "_cell_artifacts_valid",
    "_completed",
    "_run_cell",
    "audit_batch",
    "build_cells",
    "build_manifest",
    "compute_forgetting_summary",
    "load_config",
    "model_state_hash_and_bytes",
    "pack_query_indices",
    "resolve_output",
    "run_experiment",
    "run_queue",
    "sha256_file",
    "summarize_current_metrics",
    "unpack_query_indices",
    "validate_experiment_config",
]

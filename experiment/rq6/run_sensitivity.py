                                                              

                                                                            
                                                                         
                                                                             
                                                        

                                                                     
                                                                   
                                                                         
                                                                            
                                                                             
                                                               
   

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
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


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))


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
    getattr(_PROTOCOL_API, "DEFAULT_PROTOCOL", HERE / "rq6_sensitivity_protocol.json")
).resolve()
load_protocol = getattr(_PROTOCOL_API, "load_protocol", None)
build_cells = getattr(_PROTOCOL_API, "build_cells", None)
scope_slices = getattr(_PROTOCOL_API, "scope_slices", None)

                                                                             
                                                                             
                                                            
if importlib.util.find_spec("torch") is None and "core" not in sys.modules:
    _code_core_stub = type(sys)("core")
    _code_core_stub.__path__ = [str(ROOT / "code" / "core")]
    sys.modules["core"] = _code_core_stub


METRICS = ("mae", "rmse", "mre", "npre")
TERMINAL_STATUS = "BATCH_COMPLETED"
ACCEPTED_CELL_STATUSES = {"COMPLETED", "REUSED", "SKIPPED_COMPLETED", "SKIPPED_REUSED"}
FORMAL_SCOPE = "formal"
SMOKE_SCOPE = "smoke"
LOCK_NAME = ".rq6_runner.lock"
DEFAULT_PROTOCOL_ID = "fedarm_rq6_sensitivity_v1"


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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _stream_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        try:
            digest.update(bytes.fromhex(str(value)))
        except ValueError:
            digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def _slug(value: Any) -> str:
    text = str(value).strip()
    out = "".join(char if (char.isalnum() or char in "._-") else "_" for char in text)
    return out or "unknown"


def density_code(density: float) -> str:
                                                                         
                                                                            
                                                                           
    return f"d{int(round(float(density) * 100)):02d}"


def _require_protocol_api() -> None:
    global load_protocol, build_cells, scope_slices
    if _PROTOCOL_API is None:
        raise ImportError("experiment/rq6/protocol.py is unavailable")
    load_protocol = getattr(_PROTOCOL_API, "load_protocol", load_protocol)
    build_cells = getattr(_PROTOCOL_API, "build_cells", build_cells)
    scope_slices = getattr(_PROTOCOL_API, "scope_slices", scope_slices)
    if not all(callable(item) for item in (load_protocol, build_cells, scope_slices)):
        raise ImportError(
            "RQ6 protocol.py must provide load_protocol, build_cells, and scope_slices"
        )


def _model_api() -> Any:
    try:
        return importlib.import_module(".model", package=__package__)
    except (ImportError, TypeError, ValueError):
        try:
            return importlib.import_module("model")
        except ImportError as exc:
            raise ImportError("experiment/rq6/model.py is unavailable") from exc


def _dataset_spec(protocol: Mapping[str, Any]) -> tuple[tuple[int, int, int], Path]:
    dataset = protocol.get("dataset")
    dataset = dataset if isinstance(dataset, Mapping) else {}
    raw_shape = dataset.get(
        "shape_time_user_service",
        protocol.get("shape_time_user_service", (64, 142, 4500)),
    )
    shape = tuple(int(item) for item in raw_shape)
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise ValueError(f"invalid RQ6 dataset shape: {raw_shape!r}")
    raw_cache = dataset.get("cache_dir", protocol.get("cache_dir", "dataset/cache"))
    cache_dir = Path(str(raw_cache))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return (shape[0], shape[1], shape[2]), cache_dir.resolve()


def _config_path(protocol: Mapping[str, Any]) -> Path:
    raw = protocol.get("base_config")
    if isinstance(raw, Mapping):
        raw = raw.get("path")
    if not raw:
        raise ValueError("RQ6 protocol must specify base_config")
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
        raise ValueError("RQ6 base config shape differs from protocol dataset shape")
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
        raise ValueError(f"RQ6 base config missing fields: {missing}")
    return value, path


def _qos_types(protocol: Mapping[str, Any], *, smoke: bool = False) -> tuple[str, ...]:
    scope = protocol.get("smoke" if smoke else "matrix", {})
    if not isinstance(scope, Mapping):
        raise ValueError("RQ6 protocol scope must be an object")
    result = tuple(str(item).lower() for item in scope.get("qos_types", ("rt", "tp")))
    if not result or any(item not in {"rt", "tp"} for item in result):
        raise ValueError(f"invalid RQ6 QoS types: {result!r}")
    return result


def _load_caches(
    protocol: Mapping[str, Any],
    shape: tuple[int, int, int],
    qos_types: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    _, cache_dir = _dataset_spec(protocol)
    try:
        from core.data import load_value_cache
    except ImportError as exc:                    
        raise ImportError("core.data.load_value_cache is required") from exc
    cache: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for qos in qos_types:
        path = cache_dir / f"{qos}_values.npy"
        cache[qos] = load_value_cache(path, shape)
        hashes[qos] = sha256_file(path)
    return cache, hashes


def _protocol_file() -> Path:
    path = getattr(_PROTOCOL_API, "__file__", HERE / "protocol.py")
    return Path(path).resolve()


def _source_hashes(protocol: Mapping[str, Any], config_path: Path) -> dict[str, str]:
    files: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "rq6_model": HERE / "model.py",
        "rq6_protocol": _protocol_file(),
        "base_config": Path(config_path).resolve(),
        "causal_protocol": ROOT / "code" / "core" / "protocol.py",
        "data_loader": ROOT / "code" / "core" / "data.py",
        "gpu_implementation": ROOT / "code" / "core" / "gpu_ablation_training.py",
        "gpu_kernels": ROOT / "experiment" / "rq6" / "kernels.py",
        "federated_residual": ROOT / "code" / "core" / "federated_residual.py",
                                                                             
                                                                            
                                                    
        "rq4_reference_runner": ROOT / "experiment" / "rq4" / "run_robustness.py",
        "rq4_reference_model": ROOT / "experiment" / "rq4" / "model.py",
        "rq4_reference_protocol": ROOT / "experiment" / "rq4" / "protocol.py",
        "reference_full_gpu_model": ROOT / "code" / "core" / "gpu_full_training.py",
    }
    result: dict[str, str] = {
        "protocol_content": hashlib.sha256(
            json.dumps(
                _jsonable(protocol),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    }
    for name, path in files.items():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source file for hash is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _snapshot_sources(
    output: Path,
    protocol: Mapping[str, Any],
    config_path: Path,
    source_sha256: Mapping[str, str],
    data_sha256: Mapping[str, str],
    *,
    scope: str,
) -> dict[str, Any]:
                                                                               

    snapshots = Path(output) / "snapshots"
    if snapshots.exists():
        raise FileExistsError(f"RQ6 snapshot directory already exists: {snapshots}")
    source_files: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "rq6_model": (HERE / "model.py").resolve(),
        "rq6_protocol": _protocol_file(),
        "base_config": Path(config_path).resolve(),
        "causal_protocol": (ROOT / "code" / "core" / "protocol.py").resolve(),
        "data_loader": (ROOT / "code" / "core" / "data.py").resolve(),
        "gpu_implementation": (ROOT / "code" / "core" / "gpu_ablation_training.py").resolve(),
        "gpu_kernels": (HERE / "kernels.py").resolve(),
        "federated_residual": (ROOT / "code" / "core" / "federated_residual.py").resolve(),
        "rq4_reference_runner": (ROOT / "experiment" / "rq4" / "run_robustness.py").resolve(),
        "rq4_reference_model": (ROOT / "experiment" / "rq4" / "model.py").resolve(),
        "rq4_reference_protocol": (ROOT / "experiment" / "rq4" / "protocol.py").resolve(),
        "reference_full_gpu_model": (ROOT / "code" / "core" / "gpu_full_training.py").resolve(),
    }
    snapshot_targets = {
        "runner": "experiment/rq6/run_sensitivity.py",
        "rq6_model": "experiment/rq6/model.py",
        "rq6_protocol": "experiment/rq6/protocol.py",
        "base_config": "code/config/base_config.json",
        "causal_protocol": "code/core/protocol.py",
        "data_loader": "code/core/data.py",
        "gpu_implementation": "code/core/gpu_ablation_training.py",
        "gpu_kernels": "experiment/rq6/kernels.py",
        "federated_residual": "code/core/federated_residual.py",
        "rq4_reference_runner": "experiment/rq4/run_robustness.py",
        "rq4_reference_model": "experiment/rq4/model.py",
        "rq4_reference_protocol": "experiment/rq4/protocol.py",
        "reference_full_gpu_model": "code/core/gpu_full_training.py",
    }
    snapshot_hashes: dict[str, str] = {}
    for name, source in source_files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = snapshot_targets[name]
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshot_hashes[relative] = sha256_file(target)
    manifest = {
        "snapshot_kind": "rq6_sensitivity_source_snapshot",
        "created_at": _utc_now(),
        "source_sha256": dict(source_sha256),
        "data_sha256": dict(data_sha256),
        "snapshot_sha256": snapshot_hashes,
        "source_files": {name: str(path) for name, path in source_files.items()},
        "base_config": str(config_path),
        "scope": str(scope),
        "training_allowed": bool(scope == SMOKE_SCOPE or bool(protocol.get("formal_authorized", False))),
        "training_launched": False,
        "measurement_scope": "GPU trajectory training plus causal read-only metrics; local timing/memory telemetry; no network latency or energy",
        "python": sys.version,
        "platform": platform.platform(),
    }
    _atomic_json(snapshots / "manifest.json", manifest)
    return manifest


def verify_sources(manifest: Mapping[str, Any]) -> bool:
                                                                                  

    expected = manifest.get("source_sha256")
    paths = manifest.get("source_files")
    if not isinstance(expected, Mapping) or not isinstance(paths, Mapping):
        return False
    try:
        for name, raw_path in paths.items():
            path = Path(str(raw_path)).resolve()
            if not path.is_file() or str(expected.get(name, "")) == "":
                return False
            if sha256_file(path) != str(expected[name]):
                return False
    except (OSError, KeyError, TypeError):
        return False
    return True


def _snapshot_valid(output: Path, source_sha256: Mapping[str, str]) -> bool:
                                                                          

    manifest = _read_json(Path(output) / "snapshots" / "manifest.json")
    if manifest is None or manifest.get("source_sha256") != dict(source_sha256):
        return False
    if not verify_sources(manifest):
        return False
    snapshot_hashes = manifest.get("snapshot_sha256")
    if not isinstance(snapshot_hashes, Mapping):
        return False
    try:
        for relative, expected in snapshot_hashes.items():
            target = Path(output) / "snapshots" / str(relative)
            if not target.is_file() or sha256_file(target) != str(expected):
                return False
    except OSError:
        return False
    return True


def _model_kwargs(
    config: Mapping[str, Any],
    shape: tuple[int, int, int],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    params = cell.get("parameters", {})
    params = params if isinstance(params, Mapping) else {}
    return {
        "num_users": int(shape[1]),
        "num_services": int(shape[2]),
        "rank": int(params.get("rank", config["rank"])),
        "seed": int(cell["seed"]),
        "bias_eta": float(config["bias_eta"]),
        "factor_eta": float(config["factor_eta"]),
        "l2": float(config["l2"]),
        "beta": float(config["beta"]),
        "factor_init_std": float(config["factor_init_std"]),
        "num_edges": int(config["num_edges"]),
        "cpsa_gain": float(params.get("cpsa_gain", 0.1)),
        "count_exponent": float(params.get("count_exponent", 1.0)),
    }


def _new_model(
    config: Mapping[str, Any],
    shape: tuple[int, int, int],
    cell: Mapping[str, Any],
) -> Any:
    constructor = getattr(_model_api(), "GPUSensitivityModel", None)
    if constructor is None:
        raise AttributeError("RQ6 model.py must export GPUSensitivityModel")
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
        function = getattr(_model_api(), "state_hash", None)
        if callable(function):
            return str(function(model, full=True))
    except Exception:
        pass
    snapshot_function = getattr(model, "full_state_snapshot", None)
    if not callable(snapshot_function):
        snapshot_function = getattr(model, "parameter_snapshot", None)
    if not callable(snapshot_function):
        raise AttributeError("model must expose full_state_snapshot or parameter_snapshot")
    digest = hashlib.sha256()
    for name, raw in sorted(dict(snapshot_function()).items()):
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
        "model must expose full_state_snapshot/parameter_snapshot with private state"
    )


def _private_state_unchanged(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    *,
    unobserved: np.ndarray,
    num_users: int,
) -> bool:
    if PRIVATE_STATE_KEYS.issubset(before) and PRIVATE_STATE_KEYS.issubset(after):
        for name in PRIVATE_STATE_KEYS:
            left = np.asarray(before[name])
            right = np.asarray(after[name])
            if left.shape != right.shape or left.ndim == 0 or left.shape[0] != int(num_users):
                return False
            if not np.array_equal(left[unobserved], right[unobserved]):
                return False
    else:
                                                                       
        for name in set(before) | set(after):
            if name.startswith("carw_history_"):
                continue
            if name not in before or name not in after:
                return False
            left = np.asarray(before[name])
            right = np.asarray(after[name])
            if left.shape != right.shape:
                return False
            if left.ndim > 0 and left.shape[0] == int(num_users) and not np.array_equal(
                left[unobserved], right[unobserved]
            ):
                return False
    history_users: set[int] = set()
    for name in set(before) | set(after):
        if not name.startswith("carw_history_"):
            continue
        try:
            history_users.add(int(name.rsplit("_", 1)[1]))
        except ValueError:
            return False
    for user in history_users:
        if 0 <= user < int(num_users) and bool(unobserved[user]):
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
    function = getattr(profile, "as_dict", None)
    if callable(function):
        return dict(function())
    if isinstance(profile, Mapping):
        return dict(profile)
    names = (
        "support_count",
        "active_clients",
        "input_upload_seconds",
        "cpsa_seconds",
        "shared_broadcast_seconds",
        "local_update_seconds",
        "carw_srr_seconds",
        "proposal_quantization_seconds",
        "aggregation_seconds",
        "pcs_aggregation_seconds",
        "aggregate_quantization_seconds",
        "total_seconds",
        "peak_device_memory_used_bytes",
    )
    return {name: _jsonable(getattr(profile, name)) for name in names if hasattr(profile, name)}


def _diagnostics_dict(model: Any) -> dict[str, Any]:
    value = getattr(model, "last_diagnostics", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _validate_profile(profile: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
                                                                           

    if not isinstance(profile, Mapping) or not profile:
        raise AssertionError("RQ6 GPU training profile missing")
    params = _parameters(cell)
    resolved = profile.get("resolved_parameters")
    if not isinstance(resolved, Mapping):
        raise AssertionError("RQ6 profile missing resolved_parameters")
    for name, expected in params.items():
        if name not in resolved or not math.isclose(
            float(resolved[name]), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(f"RQ6 profile parameter not resolved: {name}")
    if not math.isclose(float(resolved.get("trust_exponent", math.nan)), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("RQ6 profile trust_exponent drifted")
    if not math.isclose(float(resolved.get("cpsa_l2", math.nan)), 0.0003, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("RQ6 profile cpsa_l2 drifted")
    expected_decay = 1.0 - float(params["cpsa_gain"]) * 0.0003
    if not math.isclose(float(resolved.get("cpsa_decay", math.nan)), expected_decay, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("RQ6 profile cpsa_decay drifted")
    if profile.get("state_numeric_finite") is not True:
        raise AssertionError("RQ6 committed model state is not finite")
    for name in (
        "pcs_applied_scale_count",
        "pcs_trust_clip_count",
        "pcs_fallback_count",
    ):
        try:
            if int(profile[name]) < 0:
                raise AssertionError(f"RQ6 profile {name} is negative")
        except (KeyError, TypeError, ValueError) as exc:
            raise AssertionError(f"RQ6 profile missing {name}") from exc
    count = int(profile["pcs_applied_scale_count"])
    if int(profile["pcs_trust_clip_count"]) > count or int(profile["pcs_fallback_count"]) > count:
        raise AssertionError("RQ6 profile clipping/fallback counts exceed applied count")
    try:
        scale_min = float(profile["pcs_applied_scale_min"])
        scale_max = float(profile["pcs_applied_scale_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError("RQ6 profile missing PCS scale bounds") from exc
    if not (np.isfinite(scale_min) and np.isfinite(scale_max) and 0.0 <= scale_min <= scale_max <= 1.0):
        raise AssertionError("RQ6 applied PCS scale bounds are invalid")


def _profile_is_valid(profile: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    try:
        _validate_profile(profile, summary)
    except (AssertionError, KeyError, TypeError, ValueError):
        return False
    return True


def _resolved_parameters(model: Any, cell: Mapping[str, Any]) -> dict[str, Any]:
    expected = cell.get("parameters", {})
    expected = dict(expected) if isinstance(expected, Mapping) else {}
    value = getattr(model, "resolved_parameters", None)
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        result = dict(value)
        result.setdefault("requested", expected)
        return _jsonable(result)
    return _jsonable({"requested": expected})


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
        raise ValueError("MRE/NPRE require strictly positive QoS truth")
    error = estimate - target
    relative = np.sort(np.abs(error) / target)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mre": float(np.median(relative)),
        "npre": float(relative[int(0.9 * relative.size)]),
        "absolute_error_sum": float(np.sum(np.abs(error), dtype=np.float64)),
        "squared_error_sum": float(np.sum(np.square(error), dtype=np.float64)),
        "count": int(error.size),
    }


def _metric_values(truth: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    metrics = _prediction_metrics(truth, prediction)
    relative = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) / np.asarray(truth, dtype=np.float64)
    return metrics, relative


def _empty_accumulator() -> dict[str, Any]:
    return {"count": 0, "absolute_error_sum": 0.0, "squared_error_sum": 0.0, "relative": []}


def _add_accumulator(accumulator: dict[str, Any], metrics: Mapping[str, Any], relative: np.ndarray) -> None:
    accumulator["count"] += int(metrics.get("count", relative.size))
    accumulator["absolute_error_sum"] += float(metrics.get("absolute_error_sum", np.sum(relative)))
    accumulator["squared_error_sum"] += float(metrics.get("squared_error_sum", 0.0))
    accumulator["relative"].extend(float(value) for value in np.asarray(relative).ravel())


def _summarize_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    count = int(accumulator.get("count", 0))
    relative = np.asarray(accumulator.get("relative", []), dtype=np.float64)
    if count <= 0:
        return {"count": 0, "mae": None, "rmse": None, "mre": None, "npre": None}
    return {
        "count": count,
        "mae": float(float(accumulator["absolute_error_sum"]) / count),
        "rmse": float(math.sqrt(float(accumulator["squared_error_sum"]) / count)),
        "mre": float(np.median(relative)),
        "npre": float(np.sort(relative)[int(0.9 * relative.size)]),
    }


def _cell_id(cell: Mapping[str, Any]) -> str:
    return str(cell.get("cell_id", ""))


def _cell_dir(output: Path, cell: Mapping[str, Any]) -> Path:
    setting = cell.get("setting_id", cell.get("axis", "setting"))
    return (
        Path(output)
        / _slug(setting)
        / _slug(cell.get("variant_id", "R0"))
        / _slug(cell.get("qos", "qos"))
        / density_code(float(cell.get("density", 0.0)))
        / f"seed{int(cell.get('seed', 0))}"
    )


def _parameters(cell: Mapping[str, Any]) -> dict[str, Any]:
    raw = cell.get("parameters", {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"cell {cell.get('cell_id')} has no parameter mapping")
    required = ("rank", "cpsa_gain", "count_exponent")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"cell {cell.get('cell_id')} missing parameters: {missing}")
    rank = int(raw["rank"])
    cpsa_gain = float(raw["cpsa_gain"])
    count_exponent = float(raw["count_exponent"])
    if rank <= 0 or not np.isfinite(cpsa_gain) or not np.isfinite(count_exponent):
        raise ValueError(f"invalid parameters for cell {cell.get('cell_id')}")
    return {
        "rank": rank,
        "cpsa_gain": cpsa_gain,
        "count_exponent": count_exponent,
    }


def _cell_manifest_row(cell: Mapping[str, Any], *, status: str = "PLANNED", error: str = "") -> dict[str, Any]:
    params = _parameters(cell)
    return {
        "record_type": "sensitivity_cell",
        "cell_id": _cell_id(cell),
        "setting_id": str(cell.get("setting_id", "")),
        "axis": str(cell.get("axis", "")),
        "axis_value": cell.get("axis_value"),
        "parameters_json": json.dumps(params, sort_keys=True),
        "qos": str(cell.get("qos", "")),
        "density": float(cell.get("density", 0.0)),
        "seed": int(cell.get("seed", 0)),
        "time_slices": int(cell.get("time_slices", 0)),
        "training_required": not _is_default_cell(cell),
        "projection_only": False,
        "status": status,
        "error": error,
    }


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


def _is_default_cell(cell: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> bool:
    params = _parameters(cell)
    defaults = None
    if protocol is not None:
        raw = protocol.get("defaults", {})
        defaults = raw if isinstance(raw, Mapping) else None
    if defaults is None:
        defaults = {"rank": 20, "cpsa_gain": 0.1, "count_exponent": 1.0}
    try:
        return (
            int(params["rank"]) == int(defaults["rank"])
            and math.isclose(float(params["cpsa_gain"]), float(defaults["cpsa_gain"]), rel_tol=0, abs_tol=1e-12)
            and math.isclose(float(params["count_exponent"]), float(defaults["count_exponent"]), rel_tol=0, abs_tol=1e-12)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _artifact_valid(
    cell_dir: Path,
    summary: Mapping[str, Any],
    slices: int,
    *,
    accepted_statuses: set[str] | None = None,
) -> bool:
    accepted_statuses = accepted_statuses or {"COMPLETED", "REUSED"}
    try:
        if str(summary.get("status")) not in accepted_statuses:
            return False
        attempt = Path(str(summary["attempt_dir"])).resolve()
        if attempt.parent != (Path(cell_dir) / "attempts").resolve():
            return False
        if _read_json(attempt / "summary.json") != dict(summary):
            return False
        if _read_json(attempt / "status.json") != dict(summary):
            return False
        expected = summary.get("artifact_sha256", {})
        if not isinstance(expected, Mapping):
            return False
        for name in ("per_slice.csv",):
            if expected.get(name) != sha256_file(attempt / name):
                return False
        rows = _read_csv(attempt / "per_slice.csv")
        if len(rows) != int(slices) or [int(row["time_slice"]) for row in rows] != list(range(int(slices))):
            return False
        for row in rows:
            for key in ("cell_id", "setting_id", "axis", "qos", "seed"):
                if str(row.get(key, "")) != str(summary.get(key, "")):
                    return False
            if not math.isclose(float(row["density"]), float(summary["density"]), rel_tol=0, abs_tol=1e-12):
                return False
            for metric in METRICS:
                if row.get(metric, "") == "" or not np.isfinite(float(row[metric])):
                    return False
            if row.get("query_or_future_used", "true").lower() != "false":
                return False
            if row.get("query_state_unchanged", "false").lower() != "true":
                return False
            if str(summary.get("status")) == "COMPLETED":
                profile = json.loads(row.get("gpu_profile_json", "{}"))
                if not _profile_is_valid(profile, summary):
                    return False
        return bool(summary.get("finite_metrics_all_slices")) and bool(summary.get("final_state_sha256"))
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
    if summary is None or status is None:
        return False
    if str(summary.get("status")) not in {"COMPLETED", "REUSED"} or status != summary:
        return False
    for key in ("cell_id", "setting_id", "axis", "qos", "seed"):
        if str(summary.get(key, "")) != str(cell.get(key, "")):
            return False
    if not math.isclose(float(summary.get("density", -1)), float(cell.get("density", -2)), rel_tol=0, abs_tol=1e-12):
        return False
    if summary.get("parameters") != _parameters(cell):
        return False
    if int(summary.get("time_slices_completed", -1)) != int(slices):
        return False
    if summary.get("source_sha256") != dict(source_sha256):
        return False
    if str(summary.get("config_sha256")) != str(config_sha256):
        return False
    if summary.get("data_sha256") != dict(data_sha256):
        return False
    if summary.get("query_or_future_used") or summary.get("query_or_future_used_for_update"):
        return False
    return _artifact_valid(Path(cell_dir), summary, slices)


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accumulator = _empty_accumulator()
    for row in rows:
        metrics = {name: float(row[name]) for name in METRICS}
        metrics["count"] = int(row["query_entries"])
        metrics["absolute_error_sum"] = float(row["absolute_error_sum"])
        metrics["squared_error_sum"] = float(row["squared_error_sum"])
                                                                           
                                                                              
        _add_accumulator(accumulator, metrics, np.asarray([float(row["mre"])]))
    return _summarize_accumulator(accumulator)


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
) -> dict[str, Any]:
                                                                                    

    cell_dir = _cell_dir(Path(output), cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if _completed(cell_dir, cell, slices, source_sha256, config_sha256, data_sha256):
        if not resume:
            raise FileExistsError(f"completed RQ6 cell already exists; use --resume: {cell_dir}")
        summary = _read_json(cell_dir / "summary.json") or {}
        return {"status": "SKIPPED_REUSED" if summary.get("status") == "REUSED" else "SKIPPED_COMPLETED", "summary": summary}
    attempts_dir = cell_dir / "attempts"
    if attempts_dir.is_dir() and any(attempts_dir.glob("attempt_*")) and not resume:
        raise FileExistsError(f"incomplete RQ6 attempt exists; use --resume: {cell_dir}")

    model_factory = model_factory or _new_model
    attempt = _new_attempt(cell_dir)
    params = _parameters(cell)
    status: dict[str, Any] = {
        "status": "RUNNING",
        "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
        **_jsonable(dict(cell)),
        "parameters": params,
        "time_slices_expected": int(slices),
        "time_slices_completed": 0,
        "started_at": _utc_now(),
        "attempt_dir": str(attempt),
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": dict(data_sha256),
        "query_or_future_used": False,
        "query_or_future_used_for_update": False,
        "training_launched": True,
        "reuse_status": "NOT_REUSED",
    }
    _atomic_json(attempt / "status.json", status)
    _atomic_json(cell_dir / "status.json", status)

    process = psutil.Process(os.getpid()) if psutil is not None else None
    start_cpu = process.cpu_times() if process is not None else None
    peak_rss = int(process.memory_info().rss) if process is not None else 0
    rows: list[dict[str, Any]] = []
    mask_hashes: list[str] = []
    state_hashes: list[str] = []
    overall = _empty_accumulator()
    per_slice_metric_values: dict[str, list[float]] = {name: [] for name in METRICS}
    started_cell = time.perf_counter()
    model: Any = None
    try:
        model = model_factory(config, shape, cell)
        raw_cap = _raw_cap(config, str(cell["qos"]))
        resolved = _resolved_parameters(model, cell)
        for time_slice in range(int(slices)):
            slice_started = time.perf_counter()
            try:
                from core.protocol import make_causal_split
            except ImportError as exc:                    
                raise ImportError("core.protocol.make_causal_split is required") from exc
            split_started = time.perf_counter()
            split = make_causal_split(
                np.asarray(values[time_slice]),
                float(cell["density"]),
                int(cell["seed"]),
                int(time_slice),
            )
            split_seconds = time.perf_counter() - split_started
            if split.support_count <= 0 or split.query_count <= 0:
                raise ValueError("causal split must have non-empty support and query sets")
            support_users = np.asarray(split.support_users, dtype=np.int64)
            support_services = np.asarray(split.support_services, dtype=np.int64)
            support_values = np.asarray(split.support_values, dtype=np.float64)
            query_users = np.asarray(split.query_users, dtype=np.int64)
            query_services = np.asarray(split.query_services, dtype=np.int64)
            truth = np.asarray(split.query_values, dtype=np.float64)
            support_keys = support_users * int(shape[2]) + support_services
            query_keys = query_users * int(shape[2]) + query_services
            if np.intersect1d(support_keys, query_keys).size:
                raise AssertionError("support/query masks overlap")
            if np.unique(support_keys).size != support_keys.size or np.unique(query_keys).size != query_keys.size:
                raise AssertionError("causal split contains duplicate user-service keys")
            if not np.all(np.isfinite(support_values)) or not np.all(np.isfinite(truth)):
                raise FloatingPointError("causal split contains non-finite QoS values")
            if np.any(support_values <= 0) or np.any(truth <= 0):
                raise ValueError("causal split contains non-positive QoS values")

            private_before = _private_snapshot(model)
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
            _validate_profile(profile, cell)
            diagnostics = _diagnostics_dict(model)
            private_after = _private_snapshot(model)
            observed = np.unique(support_users).astype(np.int64)
            unobserved = np.ones(int(shape[1]), dtype=bool)
            unobserved[observed] = False
            private_unchanged = _private_state_unchanged(
                private_before, private_after, unobserved=unobserved, num_users=int(shape[1])
            )
            if not private_unchanged:
                raise AssertionError("an unobserved client's private state changed during update")

            before_query = _state_hash(model)
            predict_started = time.perf_counter()
            prediction, prediction_timing = model.predict(query_users, query_services, raw_cap)
            predict_seconds = time.perf_counter() - predict_started
            after_query = _state_hash(model)
            query_unchanged = before_query == after_query
            if not query_unchanged:
                raise AssertionError("read-only query prediction changed model state")
            estimate = np.asarray(prediction, dtype=np.float64)
            if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
                raise FloatingPointError("prediction is non-finite or has the wrong shape")
            if np.any(estimate < 0.0) or np.any(estimate > raw_cap):
                raise ValueError("prediction is outside the configured raw QoS cap")
            metrics, relative = _metric_values(truth, estimate)
            _add_accumulator(overall, metrics, relative)
            for name in METRICS:
                value = float(metrics[name])
                if not np.isfinite(value):
                    raise FloatingPointError(f"non-finite RQ6 metric: {name}")
                per_slice_metric_values[name].append(value)
            row: dict[str, Any] = {
                "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
                "cell_id": _cell_id(cell),
                "setting_id": str(cell.get("setting_id", "")),
                "axis": str(cell.get("axis", "")),
                "axis_value": cell.get("axis_value"),
                "parameters_json": json.dumps(params, sort_keys=True),
                "resolved_parameters_json": json.dumps(resolved, ensure_ascii=False, sort_keys=True),
                "qos": str(cell["qos"]),
                "density": float(cell["density"]),
                "seed": int(cell["seed"]),
                "time_slice": int(time_slice),
                "support_entries": int(split.support_count),
                "query_entries": int(split.query_count),
                "support_users": int(np.unique(support_users).size),
                "original_mask_sha256": str(split.mask_sha256),
                "support_keys_sha256": _hash_array(np.column_stack((support_users, support_services))),
                "query_keys_sha256": _hash_array(np.column_stack((query_users, query_services))),
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
                "slice_wall_seconds": float(time.perf_counter() - slice_started),
                "query_state_hash_before": before_query,
                "query_state_hash_after": after_query,
                "query_state_unchanged": bool(query_unchanged),
                "unobserved_private_state_unchanged": bool(private_unchanged),
                "query_or_future_used": False,
                "query_or_future_used_for_update": False,
                "gpu_profile_json": json.dumps(_jsonable(profile), ensure_ascii=False, sort_keys=True),
                "model_diagnostics_json": json.dumps(_jsonable(diagnostics), ensure_ascii=False, sort_keys=True),
            }
            rows.append(row)
            _atomic_csv(attempt / "per_slice.csv", rows)
            mask_hashes.append(str(split.mask_sha256))
            state_hashes.append(after_query)
            if process is not None:
                peak_rss = max(peak_rss, int(process.memory_info().rss))
            status.update(
                {
                    "time_slices_completed": int(time_slice + 1),
                    "last_mask_sha256": str(split.mask_sha256),
                    "updated_at": _utc_now(),
                }
            )
            _atomic_json(attempt / "status.json", status)
            _atomic_json(cell_dir / "status.json", status)

        end_cpu = process.cpu_times() if process is not None else None
        pooled = _summarize_accumulator(overall)
        time_mean = {
            name: float(np.mean(values_for_metric, dtype=np.float64))
            for name, values_for_metric in per_slice_metric_values.items()
        }
        summary: dict[str, Any] = {
            "status": "COMPLETED",
            "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
            **_jsonable(dict(cell)),
            "parameters": params,
            "resolved_parameters": resolved,
            "time_slices_completed": len(rows),
            "metrics": {"pooled": pooled, "time_mean": time_mean},
            "finite_metrics_all_slices": True,
            "query_or_future_used": False,
            "query_or_future_used_for_update": False,
            "training_launched": True,
            "reuse_status": "NOT_REUSED",
            "source_sha256": dict(source_sha256),
            "config_sha256": str(config_sha256),
            "data_sha256": dict(data_sha256),
            "mask_audit": {
                "count": len(mask_hashes),
                "unique_count": len(set(mask_hashes)),
                "mask_sha256_by_slice": {str(index): value for index, value in enumerate(mask_hashes)},
                "mask_stream_sha256": _stream_hash(mask_hashes),
                "state_hash_by_slice": {str(index): value for index, value in enumerate(state_hashes)},
                "support_query_partition_checked": True,
            },
            "query_state_unchanged_all_slices": bool(all(bool(row["query_state_unchanged"]) for row in rows)),
            "unobserved_private_state_unchanged_all_slices": bool(all(bool(row["unobserved_private_state_unchanged"]) for row in rows)),
            "timing_seconds": {
                name: float(sum(float(row.get(name, 0.0)) for row in rows))
                for name in ("split_seconds", "update_seconds", "predict_seconds", "slice_wall_seconds")
            },
            "cell_wall_seconds": float(time.perf_counter() - started_cell),
            "gpu_peak_memory_used_bytes": max(
                (int(json.loads(row["gpu_profile_json"]).get("peak_device_memory_used_bytes"))
                 for row in rows
                 if json.loads(row["gpu_profile_json"]).get("peak_device_memory_used_bytes") is not None),
                default=None,
            ),
            "process_peak_rss_bytes": int(peak_rss),
            "process_cpu_user_seconds": float(end_cpu.user - start_cpu.user) if end_cpu is not None and start_cpu is not None else 0.0,
            "process_cpu_system_seconds": float(end_cpu.system - start_cpu.system) if end_cpu is not None and start_cpu is not None else 0.0,
            "final_state_sha256": _state_hash(model),
            "attempt_dir": str(attempt),
            "artifact_sha256": {"per_slice.csv": sha256_file(attempt / "per_slice.csv")},
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
    params = summary.get("parameters", {})
    return {
        "protocol": str(summary.get("protocol", "")),
        "cell_id": str(summary.get("cell_id", "")),
        "setting_id": str(summary.get("setting_id", "")),
        "axis": str(summary.get("axis", "")),
        "axis_value": summary.get("axis_value"),
        "rank": params.get("rank"),
        "cpsa_gain": params.get("cpsa_gain"),
        "count_exponent": params.get("count_exponent"),
        "qos": str(summary.get("qos", "")),
        "density": float(summary.get("density", 0.0)),
        "seed": int(summary.get("seed", 0)),
        "cell_status": str(summary.get("status", "")),
        "reuse_status": str(summary.get("reuse_status", "")),
        "training_launched": bool(summary.get("training_launched", False)),
        "time_slices_completed": int(summary.get("time_slices_completed", 0)),
        **{f"pooled_{name}": pooled.get(name) for name in METRICS},
        **{f"time_mean_{name}": time_mean.get(name) for name in METRICS},
        "cell_wall_seconds": summary.get("cell_wall_seconds"),
        "process_peak_rss_bytes": summary.get("process_peak_rss_bytes"),
        "gpu_peak_memory_used_bytes": summary.get("gpu_peak_memory_used_bytes"),
        "finite_metrics_all_slices": bool(summary.get("finite_metrics_all_slices", False)),
    }


def _summary_memberships(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
                                                                             

    if str(summary.get("setting_id", "")) != "default":
        return [{"axis": str(summary.get("axis", "")), "axis_value": summary.get("axis_value")}]
    api = _PROTOCOL_API
    function = getattr(api, "axis_memberships", None)
    if callable(function):
        try:
            return [dict(item) for item in function(summary)]
        except Exception:
            pass
    return [
        {"axis": "rank", "axis_value": 20},
        {"axis": "cpsa_gain", "axis_value": 0.1},
        {"axis": "count_exponent", "axis_value": 1.0},
    ]


def _aggregate_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
                                                                         
                                                                           
                                                                
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        for membership in _summary_memberships(row):
            grouped[(str(membership.get("axis", "")), str(membership.get("axis_value")), str(row["qos"]), float(row["density"]))].append(row)
    output: list[dict[str, Any]] = []
    for (axis, axis_value, qos, density), rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "axis": axis,
            "axis_value": axis_value,
            "qos": qos,
            "density": density,
            "cell_count": len(rows),
        }
        for field in ("pooled_mae", "pooled_rmse", "pooled_mre", "pooled_npre", "time_mean_mae", "time_mean_rmse", "time_mean_mre", "time_mean_npre"):
            values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
            record[f"mean_{field}"] = float(np.mean(values)) if values else None
            record[f"std_{field}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        output.append(record)
    return output


def _paired_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
                                                                               

    defaults: dict[tuple[str, float, int], Mapping[str, Any]] = {}
    for row in summary_rows:
        if str(row.get("setting_id")) == "default":
            defaults[(str(row["qos"]), float(row["density"]), int(row["seed"]))] = row
    output: list[dict[str, Any]] = []
    for candidate in summary_rows:
        if str(candidate.get("setting_id")) == "default":
            continue
        key = (str(candidate["qos"]), float(candidate["density"]), int(candidate["seed"]))
        reference = defaults.get(key)
        if reference is None:
            continue
        record: dict[str, Any] = {
            "setting_id": str(candidate.get("setting_id", "")),
            "axis": str(candidate.get("axis", "")),
            "axis_value": candidate.get("axis_value"),
            "default_setting_id": "default",
            "qos": str(candidate["qos"]),
            "density": float(candidate["density"]),
            "seed": int(candidate["seed"]),
        }
        for metric in METRICS:
            right = float(reference[f"pooled_{metric}"])
            left = float(candidate[f"pooled_{metric}"])
            delta = left - right
            record[f"candidate_pooled_{metric}"] = left
            record[f"default_pooled_{metric}"] = right
            record[f"delta_pooled_{metric}"] = delta
            record[f"relative_change_pct_{metric}"] = float(100.0 * delta / right) if right != 0.0 else None
        output.append(record)
    return output


def _configured_output(protocol: Mapping[str, Any], scope: str, override: Path | None = None) -> Path:
    if scope not in {SMOKE_SCOPE, FORMAL_SCOPE}:
        raise ValueError(f"unknown RQ6 output scope {scope}")
    if override is None:
        outputs = protocol.get("outputs", {})
        raw = outputs.get(scope) if isinstance(outputs, Mapping) else None
        if not raw:
            raw = f"results/rq6_sensitivity_{scope}"
        override = Path(str(raw))
    output = Path(override)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    experiment_root = (ROOT / "results").resolve()
    try:
        output.relative_to(experiment_root)
    except ValueError as exc:
        raise ValueError(f"RQ6 output must be under {experiment_root}: {output}") from exc
    if output == experiment_root:
        raise ValueError("RQ6 output must be a dedicated child directory")
    return output


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            if not process.is_running():
                return False
            try:
                return process.status() != psutil.STATUS_ZOMBIE
            except psutil.AccessDenied:
                                                                         
                return True
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True
        except Exception:
            return True
    if os.name == "nt":
                                                                             
                                                                       
                                                        
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            query_limited = 0x1000
            handle = kernel32.OpenProcess(query_limited, False, int(pid))
            if not handle:
                error = ctypes.get_last_error()
                                                                         
                                                                    
                return int(error) != 87
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(not ok or int(exit_code.value) == 259)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _BatchLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.held = False


def _acquire_batch_lock(output: Path, *, scope: str, protocol_id: str) -> _BatchLock:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / LOCK_NAME
    payload = {
        "pid": os.getpid(),
        "scope": scope,
        "protocol_id": protocol_id,
        "started_at": _utc_now(),
    }
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = _read_json(lock_path) or {}
        pid = int(existing.get("pid", 0)) if str(existing.get("pid", "0")).isdigit() else 0
        if _pid_is_alive(pid):
            raise RuntimeError(f"RQ6 output locked by active pid {pid}: {output}") from exc
        raise RuntimeError(f"stale RQ6 lock exists; remove it only after inspection: {lock_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return _BatchLock(lock_path)


def unlock_output(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    lock_path = output / LOCK_NAME
    lock = _read_json(lock_path)
    if lock is None:
        return {"status": "UNLOCKED", "removed": False, "output": str(output)}
    pid = int(lock.get("pid", 0)) if str(lock.get("pid", "0")).isdigit() else 0
    if _pid_is_alive(pid):
        raise RuntimeError(f"cannot unlock active RQ6 pid {pid}")
    lock_path.unlink()
    return {"status": "UNLOCKED", "removed": True, "output": str(output)}


def _protocol_matrix(protocol: Mapping[str, Any], smoke: bool) -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...], int]:
    scope = protocol.get("smoke" if smoke else "matrix")
    if not isinstance(scope, Mapping):
        raise ValueError("RQ6 scope missing")
    qos = tuple(str(item).lower() for item in scope.get("qos_types", ("rt", "tp")))
    densities = tuple(float(item) for item in scope.get("densities", ()))
    seeds = tuple(int(item) for item in scope.get("seeds", ()))
    slices = int(scope.get("time_slices", 0))
    return qos, densities, seeds, slices


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
        "manifest_kind": "rq6_sensitivity_batch_manifest",
        "protocol_id": str(protocol.get("protocol_id", DEFAULT_PROTOCOL_ID)),
        "protocol_status": str(protocol.get("status", "")),
        "scope": scope,
        "output": str(Path(output).resolve()),
        "shape_time_user_service": list(shape),
        "time_slices": int(slices),
        "cells_expected": len(cells),
        "cell_ids": [_cell_id(cell) for cell in cells],
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": dict(data_sha256),
        "terminal_artifacts": [
            "manifest.json",
            "queue_cells.csv",
            "queue_status.json",
            "final_summary.csv",
            "aggregate_mean_std.csv",
            "paired_comparisons.csv",
            "BATCH_COMPLETED",
        ],
        "metrics": list(METRICS),
        "formal_authorized": bool(protocol.get("formal_authorized", False)),
        "training_allowed": bool(scope == SMOKE_SCOPE or bool(protocol.get("formal_authorized", False))),
        "training_launched": False,
        "measurement_scope": "GPU trajectory training plus causal read-only metrics; local timing/memory telemetry; no network latency or energy",
        "created_at": _utc_now(),
    }


def _validate_existing_manifest(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    keys = (
        "manifest_kind",
        "protocol_id",
        "scope",
        "shape_time_user_service",
        "time_slices",
        "cells_expected",
        "cell_ids",
        "source_sha256",
        "config_sha256",
        "data_sha256",
        "training_allowed",
    )
    for key in keys:
        if existing.get(key) != current.get(key):
            raise ValueError(f"existing RQ6 manifest mismatch at {key}; start a new output")


def _reference_condition_path(root: Path, cell: Mapping[str, Any]) -> Path:
    density = float(cell["density"])
    tags = (density_code(density), f"d{int(round(density * 100)):02d}")
    candidates = []
    for tag in tags:
        candidates.append(root / "reference" / "R0" / str(cell["qos"]) / tag / f"seed{int(cell['seed'])}")
                                                                         
                                                                              
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _reference_provenance(
    summary: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    qos: str,
) -> None:
                                                                        

    source = summary.get("source_sha256")
    if not isinstance(source, Mapping):
        raise ValueError("RQ3 reference source hashes are missing")
    reference_files = {
        "runner": ROOT / "experiment" / "rq4" / "run_robustness.py",
        "rq4_model": ROOT / "experiment" / "rq4" / "model.py",
        "rq4_protocol": ROOT / "experiment" / "rq4" / "protocol.py",
        "original_gpu_model": ROOT / "code" / "core" / "gpu_ablation_training.py",
        "reference_full_gpu_model": ROOT / "code" / "core" / "gpu_full_training.py",
        "causal_protocol": ROOT / "code" / "core" / "protocol.py",
        "data_loader": ROOT / "code" / "core" / "data.py",
    }
    for name, path in reference_files.items():
        if not path.is_file() or str(source.get(name, "")) != sha256_file(path):
            raise ValueError(f"RQ3 reference {name} source hash mismatch")
    config_path = _config_path(protocol)
    if str(source.get("base_config", "")) != sha256_file(config_path):
        raise ValueError("RQ3 reference base configuration hash mismatch")
    if str(summary.get("config_sha256", "")) != sha256_file(config_path):
        raise ValueError("RQ3 reference config_sha256 mismatch")
    _, cache_dir = _dataset_spec(protocol)
    data_hashes = summary.get("data_sha256")
    data_path = cache_dir / f"{str(qos).lower()}_values.npy"
    if not isinstance(data_hashes, Mapping) or str(data_hashes.get(str(qos).lower(), "")) != sha256_file(data_path):
        raise ValueError("RQ3 reference input data hash mismatch")
    if summary.get("variant_id") != "R0" or summary.get("variant") != "full":
        raise ValueError("RQ3 reuse source is not R0/full")
    if summary.get("scenario_id") != "reference":
        raise ValueError("RQ3 reuse source is not the reference scenario")


def _load_accepted_reference(
    path: Path,
    slices: int = 64,
    *,
    protocol: Mapping[str, Any] | None = None,
    qos: str | None = None,
) -> tuple[dict[str, Any], Path]:
    summary = _read_json(path / "summary.json")
    if summary is None or summary.get("status") != "COMPLETED":
        raise ValueError(f"RQ3 reference summary is not completed: {path}")
    if int(summary.get("time_slices_completed", -1)) != int(slices):
        raise ValueError(f"RQ3 reference does not cover {slices} slices: {path}")
    attempt = Path(str(summary.get("attempt_dir", ""))).resolve()
    if attempt.parent != (path / "attempts").resolve() or not attempt.is_dir():
        raise ValueError(f"RQ3 reference accepted attempt is outside its cell: {path}")
    attempt_summary = _read_json(attempt / "summary.json")
    attempt_status = _read_json(attempt / "status.json")
    if attempt_summary != summary or attempt_status != summary:
        raise ValueError(f"RQ3 reference attempt summary/status mismatch: {path}")
    artifacts = summary.get("artifact_sha256", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"RQ3 reference artifacts missing: {path}")
    for name in ("per_slice.csv", "group_metrics_per_slice.csv"):
        target = attempt / name
        if artifacts.get(name) != sha256_file(target):
            raise ValueError(f"RQ3 reference artifact hash mismatch: {target}")
    if summary.get("query_or_future_used") or summary.get("query_or_future_used_for_update"):
        raise ValueError(f"RQ3 reference violates read-only provenance: {path}")
    rows = _read_csv(attempt / "per_slice.csv")
    if len(rows) != int(slices) or [int(row["time_slice"]) for row in rows] != list(range(int(slices))):
        raise ValueError(f"RQ3 reference per-slice coverage invalid: {path}")
    if protocol is not None and qos is not None:
        _reference_provenance(summary, protocol=protocol, qos=qos)
    return summary, attempt


def _equivalence_check(
    smoke_output: Path,
    cell: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    *,
    slices: int,
    current_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
                                                                             

    smoke_dir = _cell_dir(smoke_output, cell)
    if current_source_sha256 is not None:
        smoke_manifest = _read_json(Path(smoke_output) / "manifest.json")
        smoke_source = smoke_manifest.get("source_sha256", {}) if smoke_manifest else {}
        if not isinstance(smoke_source, Mapping):
            raise ValueError("default smoke source provenance is unavailable")
                                                                            
                                                                          
                                                           
        ignored = {"protocol_content"}
        for key, expected in current_source_sha256.items():
            if key not in ignored and str(smoke_source.get(key, "")) != str(expected):
                raise ValueError(f"default smoke execution source changed: {key}")
    smoke_summary = _read_json(smoke_dir / "summary.json")
    if smoke_summary is None or smoke_summary.get("status") != "COMPLETED":
        raise ValueError(f"default smoke equivalence result is unavailable: {smoke_dir}")
    attempt = Path(str(smoke_summary["attempt_dir"]))
    smoke_rows = _read_csv(attempt / "per_slice.csv")
    reference_attempt = Path(str(reference_summary["attempt_dir"]))
    reference_rows = _read_csv(reference_attempt / "per_slice.csv")
    if len(smoke_rows) != slices or len(reference_rows) < slices:
        raise ValueError("equivalence slice coverage is incomplete")
    mismatches: list[str] = []
    for index in range(int(slices)):
        left = smoke_rows[index]
        right = reference_rows[index]
        for field in (
            "original_mask_sha256",
            "query_state_hash_before",
            "query_state_hash_after",
        ):
            if str(left.get(field)) != str(right.get(field)):
                mismatches.append(f"t{index}:{field}")
                                                                               
                                                                      
    for metric in METRICS:
        smoke_values = [float(row[metric]) for row in smoke_rows]
        ref_values = [float(row[metric]) for row in reference_rows[:slices]]
        if smoke_values != ref_values:
            mismatches.append(metric)
    if mismatches:
        raise ValueError("RQ6 default smoke is not equivalent to the RQ3 reference: " + ",".join(mismatches))
    return {
        "status": "PASSED",
        "slices_compared": int(slices),
        "reference_summary_sha256": sha256_file(reference_attempt / "summary.json"),
        "smoke_summary_sha256": sha256_file(smoke_dir / "summary.json"),
        "mismatches": [],
    }


def _reuse_reference_cell(
    *,
    cell: Mapping[str, Any],
    output: Path,
    reference_root: Path,
    equivalence_output: Path,
    source_sha256: Mapping[str, str],
    config_sha256: str,
    data_sha256: Mapping[str, str],
    slices: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
                                                                            

    ref_dir = _reference_condition_path(reference_root, cell)
    reference_summary, reference_attempt = _load_accepted_reference(
        ref_dir, 64, protocol=protocol, qos=str(cell["qos"])
    )
                                                                             
                                                                            
                                                                           
                                                                            
                                       
    smoke_cells = list(build_cells(protocol, smoke=True))
    gate_cell = next(
        candidate
        for candidate in smoke_cells
        if candidate.get("setting_id") == "default"
        and str(candidate.get("qos")) == str(cell.get("qos"))
    )
    gate_ref_cell = {**dict(cell), "density": 0.01, "seed": 42}
    gate_ref_dir = _reference_condition_path(reference_root, gate_ref_cell)
    gate_reference_summary, _ = _load_accepted_reference(
        gate_ref_dir, 64, protocol=protocol, qos=str(cell["qos"])
    )
    equivalence = _equivalence_check(
        equivalence_output,
        gate_cell,
        gate_reference_summary,
        slices=min(int(slices), int(protocol.get("smoke", {}).get("time_slices", 20))),
        current_source_sha256=source_sha256,
    )
    cell_dir = _cell_dir(output, cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if _completed(cell_dir, {**dict(cell)}, slices, source_sha256, config_sha256, data_sha256):
        summary = _read_json(cell_dir / "summary.json") or {}
        return {"status": "SKIPPED_REUSED", "summary": summary}
    if any((cell_dir / "attempts").glob("attempt_*")):
        raise FileExistsError(f"incomplete RQ6 reuse attempt exists; use --resume: {cell_dir}")
    attempt = _new_attempt(cell_dir)
    reference_rows = _read_csv(reference_attempt / "per_slice.csv")
    rows: list[dict[str, Any]] = []
    for old in reference_rows:
        row = dict(old)
        row.update(
            {
                "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
                "cell_id": _cell_id(cell),
                "setting_id": str(cell.get("setting_id", "")),
                "axis": str(cell.get("axis", "")),
                "axis_value": cell.get("axis_value"),
                "parameters_json": json.dumps(_parameters(cell), sort_keys=True),
                "resolved_parameters_json": json.dumps({"reused_from_rq4": True, "requested": _parameters(cell)}, sort_keys=True),
                "reuse_status": "REUSED",
                "query_or_future_used": False,
                "query_or_future_used_for_update": False,
            }
        )
        rows.append(row)
    _atomic_csv(attempt / "per_slice.csv", rows)
                                                                               
                                                              
    source_group = reference_attempt / "group_metrics_per_slice.csv"
    if source_group.is_file():
        shutil.copy2(source_group, attempt / "group_metrics_per_slice.csv")
    source_pooled = reference_summary.get("metrics", {}).get("pooled", {})
    source_time = reference_summary.get("metrics", {}).get("time_mean", {})
    summary: dict[str, Any] = {
        "status": "REUSED",
        "protocol": str(cell.get("protocol", DEFAULT_PROTOCOL_ID)),
        **_jsonable(dict(cell)),
        "parameters": _parameters(cell),
        "resolved_parameters": {"reused_from_rq4": True, "requested": _parameters(cell)},
        "time_slices_completed": int(slices),
        "metrics": {"pooled": dict(source_pooled), "time_mean": dict(source_time)},
        "finite_metrics_all_slices": True,
        "query_or_future_used": False,
        "query_or_future_used_for_update": False,
        "training_launched": False,
        "reuse_status": "REUSED",
        "reuse_source_root": str(reference_root.resolve()),
        "reuse_source_cell": str(ref_dir.resolve()),
        "reuse_source_attempt": str(reference_attempt.resolve()),
        "reuse_source_summary_sha256": sha256_file(reference_attempt / "summary.json"),
        "reuse_equivalence": equivalence,
        "source_sha256": dict(source_sha256),
        "config_sha256": str(config_sha256),
        "data_sha256": dict(data_sha256),
        "mask_audit": reference_summary.get("mask_audit", {}),
        "query_state_unchanged_all_slices": True,
        "unobserved_private_state_unchanged_all_slices": True,
        "timing_seconds": dict(reference_summary.get("timing_seconds", {})),
        "cell_wall_seconds": reference_summary.get("cell_wall_seconds"),
        "gpu_peak_memory_used_bytes": reference_summary.get("gpu_peak_memory_used_bytes"),
        "process_peak_rss_bytes": reference_summary.get("process_peak_rss_bytes"),
        "process_cpu_user_seconds": reference_summary.get("process_cpu_user_seconds"),
        "process_cpu_system_seconds": reference_summary.get("process_cpu_system_seconds"),
        "final_state_sha256": reference_summary.get("final_state_sha256"),
        "attempt_dir": str(attempt),
        "artifact_sha256": {
            "per_slice.csv": sha256_file(attempt / "per_slice.csv"),
            **({"group_metrics_per_slice.csv": sha256_file(attempt / "group_metrics_per_slice.csv")} if (attempt / "group_metrics_per_slice.csv").is_file() else {}),
        },
        "completed_at": _utc_now(),
    }
    _atomic_json(attempt / "summary.json", summary)
    _atomic_json(attempt / "status.json", summary)
    _atomic_json(cell_dir / "summary.json", summary)
    _atomic_json(cell_dir / "status.json", summary)
    return {"status": "REUSED", "summary": summary}


def _audit_cell(output: Path, cell: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    directory = _cell_dir(output, cell)
    summary = _read_json(directory / "summary.json")
    status = _read_json(directory / "status.json")
    checks: dict[str, bool] = {
        "summary_present": summary is not None,
        "status_present": status is not None,
        "status_summary_match": summary is not None and status == summary,
    }
    if summary is None:
        checks["artifact_integrity"] = False
        return {"cell_id": _cell_id(cell), "accepted": False, "checks": checks}
    checks.update(
        {
            "identity": all(str(summary.get(key, "")) == str(cell.get(key, "")) for key in ("cell_id", "setting_id", "axis", "qos", "seed")),
            "density": math.isclose(float(summary.get("density", -1)), float(cell.get("density", -2)), rel_tol=0, abs_tol=1e-12),
            "parameters": summary.get("parameters") == _parameters(cell),
            "time_slices": int(summary.get("time_slices_completed", -1)) == int(manifest.get("time_slices", -2)),
            "finite_metrics": bool(summary.get("finite_metrics_all_slices")),
            "read_only": not bool(summary.get("query_or_future_used") or summary.get("query_or_future_used_for_update")),
            "source_sha256": summary.get("source_sha256") == manifest.get("source_sha256"),
            "config_sha256": str(summary.get("config_sha256", "")) == str(manifest.get("config_sha256", "")),
            "data_sha256": summary.get("data_sha256") == manifest.get("data_sha256"),
            "reuse_label": (summary.get("status") != "REUSED") or (summary.get("reuse_status") == "REUSED" and not bool(summary.get("training_launched"))),
        }
    )
    checks["artifact_integrity"] = _artifact_valid(directory, summary, int(manifest.get("time_slices", -1)))
    return {"cell_id": _cell_id(cell), "accepted": bool(all(checks.values())), "checks": checks, "status": summary.get("status")}


def audit_batch(protocol: Mapping[str, Any], *, output: Path, smoke: bool) -> dict[str, Any]:
                                                                               

    output = Path(output).resolve()
    manifest = _read_json(output / "manifest.json")
    status = _read_json(output / "queue_status.json")
    queue_path = output / "queue_cells.csv"
    if manifest is None or status is None or not queue_path.is_file():
        return {"status": "MISSING", "output": str(output)}
    _require_protocol_api()
    cells = list(build_cells(protocol, smoke=smoke))
    slices = int(scope_slices(protocol, smoke))
    try:
        shape, _ = _dataset_spec(protocol)
        config, config_path = _load_config(protocol)
        del config
        source = _source_hashes(protocol, config_path)
        _, data = _load_caches(protocol, shape, _qos_types(protocol, smoke=smoke))
        current = _build_manifest(protocol, cells, scope=SMOKE_SCOPE if smoke else FORMAL_SCOPE, output=output, shape=shape, source_sha256=source, config_sha256=sha256_file(config_path), data_sha256=data, slices=slices)
        _validate_existing_manifest(manifest, current)
        manifest_match = True
        snapshot_match = _snapshot_valid(output, source)
    except Exception:
        manifest_match = False
        snapshot_match = False
    queue_rows = _read_csv(queue_path)
    expected_ids = [_cell_id(cell) for cell in cells]
    observed_ids = [str(row.get("cell_id", "")) for row in queue_rows]
    duplicates = sorted({value for value in observed_ids if observed_ids.count(value) > 1})
    missing = sorted(set(expected_ids) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(expected_ids))
    by_id = {_cell_id(cell): cell for cell in cells}
    cell_audits = [_audit_cell(output, by_id[row["cell_id"]], manifest) for row in queue_rows if row.get("cell_id") in by_id]
    terminal_paths = [output / name for name in manifest.get("terminal_artifacts", []) if name != "BATCH_COMPLETED"]
    terminal_present = all(path.is_file() and path.stat().st_size > 0 for path in terminal_paths)
    marker = _read_json(output / "BATCH_COMPLETED") or {}
    terminal_integrity = marker.get("status") == TERMINAL_STATUS and marker.get("protocol_id") == manifest.get("protocol_id") and marker.get("scope") == manifest.get("scope")
    for name, key in (
        ("final_summary.csv", "final_summary_sha256"),
        ("aggregate_mean_std.csv", "aggregate_mean_std_sha256"),
        ("paired_comparisons.csv", "paired_comparisons_sha256"),
    ):
        terminal_integrity = terminal_integrity and (output / name).is_file() and marker.get(key) == sha256_file(output / name)
    expected_summaries = marker.get("cell_summary_sha256", {})
    for cell in cells:
        summary_path = _cell_dir(output, cell) / "summary.json"
        terminal_integrity = terminal_integrity and summary_path.is_file() and expected_summaries.get(_cell_id(cell)) == sha256_file(summary_path)
    all_queue = manifest_match and len(queue_rows) == len(cells) and not duplicates and not missing and not unexpected and all(row.get("status") in ACCEPTED_CELL_STATUSES for row in queue_rows)
    all_cells = len(cell_audits) == len(cells) and bool(cell_audits) and all(item["accepted"] for item in cell_audits)
    complete = all_queue and all_cells and snapshot_match and status.get("status") == TERMINAL_STATUS and status.get("active") is None and int(status.get("process_count", -1)) == 0 and terminal_present and terminal_integrity
    return {
        "status": "COMPLETE" if complete else "PLANNED" if manifest_match and status.get("status") == "PLANNED" and not cell_audits else "INCOMPLETE",
        "output": str(output),
        "manifest_match": manifest_match,
        "snapshot_match": snapshot_match,
        "expected_cells": len(cells),
        "queued_cells": len(queue_rows),
        "completed_queue_rows": sum(row.get("status") in ACCEPTED_CELL_STATUSES for row in queue_rows),
        "duplicate_ids": duplicates,
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "all_queue_rows_accepted": all_queue,
        "all_cell_gates": all_cells,
        "terminal_artifacts_present": terminal_present,
        "terminal_artifact_integrity": bool(terminal_integrity),
        "cell_audits": cell_audits,
    }


def _run_queue_unlocked(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    output: Path,
    resume: bool = False,
    run_formal: bool = False,
    stop_on_error: bool = False,
    model_factory: Callable[[Mapping[str, Any], tuple[int, int, int], Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    if not smoke:
        if not run_formal:
            raise PermissionError("formal RQ6 execution requires --run-formal")
        if str(protocol.get("status")) != "FROZEN" or not bool(protocol.get("formal_authorized")):
            raise PermissionError("formal RQ6 execution requires protocol status FROZEN and formal_authorized=true")
    _require_protocol_api()
    shape, _ = _dataset_spec(protocol)
    config, config_path = _load_config(protocol)
    qos_types = _qos_types(protocol, smoke=smoke)
    cache, data_sha256 = _load_caches(protocol, shape, qos_types)
    source_sha256 = _source_hashes(protocol, config_path)
    config_sha256 = sha256_file(config_path)
    cells = list(build_cells(protocol, smoke=smoke))
    slices = int(scope_slices(protocol, smoke))
    scope = SMOKE_SCOPE if smoke else FORMAL_SCOPE
    output = _configured_output(protocol, scope, output)
    current_manifest = _build_manifest(protocol, cells, scope=scope, output=output, shape=shape, source_sha256=source_sha256, config_sha256=config_sha256, data_sha256=data_sha256, slices=slices)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = _read_json(manifest_path)
    if existing is not None:
        if not resume:
            raise FileExistsError(f"RQ6 output is initialized; use --resume: {output}")
        _validate_existing_manifest(existing, current_manifest)
        current_manifest["created_at"] = existing.get("created_at", current_manifest["created_at"])
    elif any(path.name != LOCK_NAME for path in output.iterdir()):
        raise FileExistsError(f"RQ6 output is non-empty without a manifest: {output}")
    _atomic_json(manifest_path, current_manifest)
    if not (output / "snapshots").exists():
        _snapshot_sources(
            output,
            protocol,
            config_path,
            source_sha256,
            data_sha256,
            scope=scope,
        )
    elif not _snapshot_valid(output, source_sha256):
        raise ValueError("existing RQ6 source snapshot/hash mismatch; do not resume across source changes")
    queue_path = output / "queue_cells.csv"
    if resume and queue_path.is_file():
        queue_rows = _read_csv(queue_path)
        if [str(row.get("cell_id", "")) for row in queue_rows] != [_cell_id(cell) for cell in cells]:
            raise ValueError("existing RQ6 queue coverage/order mismatch; start a new output")
    else:
        queue_rows = [_cell_manifest_row(cell, status="PLANNED") for cell in cells]
        _atomic_csv(queue_path, queue_rows)
    started = time.perf_counter()
    completed = skipped = reused = failed = 0

    def write_status(status_value: str, active: Mapping[str, Any] | None, visited: int) -> None:
        _atomic_json(
            output / "queue_status.json",
            {
                "status": status_value,
                "protocol_id": current_manifest["protocol_id"],
                "scope": scope,
                "cells_expected": len(cells),
                "cells_visited": int(visited),
                "cells_completed": int(completed + skipped + reused),
                "cells_completed_this_invocation": int(completed + reused),
                "cells_skipped_completed": int(skipped),
                "cells_reused": int(reused),
                "cells_failed": int(failed),
                "active": _jsonable(active),
                "process_count": 1 if active is not None else 0,
                "updated_at": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )

    write_status("RUNNING", None, 0)
    for index, cell in enumerate(cells):
        queue_rows[index]["status"] = "RUNNING"
        queue_rows[index]["attempt_dir"] = ""
        _atomic_csv(queue_path, queue_rows)
        write_status("RUNNING", {"cell_id": _cell_id(cell), "pid": os.getpid()}, index)
        try:
            cell_for_run = {**dict(cell), "protocol": current_manifest["protocol_id"]}
            result: dict[str, Any]
            if (not smoke and _is_default_cell(cell, protocol)):
                reuse = protocol.get("baseline_reuse", {})
                if not isinstance(reuse, Mapping) or not bool(reuse.get("enabled", False)):
                    raise ValueError("default RQ6 cells require explicit baseline_reuse.enabled=true")
                root = Path(str(reuse.get("source_root", "")))
                if not root.is_absolute():
                    root = ROOT / root
                eq = Path(str(reuse.get("equivalence_output", "")))
                if not eq.is_absolute():
                    eq = ROOT / eq
                result = _reuse_reference_cell(cell=cell_for_run, output=output, reference_root=root.resolve(), equivalence_output=eq.resolve(), source_sha256=source_sha256, config_sha256=config_sha256, data_sha256=data_sha256, slices=slices, protocol=protocol)
            else:
                result = run_cell(cell=cell_for_run, values=cache[str(cell["qos"])], config=config, shape=shape, output=output, slices=slices, source_sha256=source_sha256, config_sha256=config_sha256, data_sha256=data_sha256, resume=resume, model_factory=model_factory)
            queue_rows[index]["status"] = str(result["status"])
            queue_rows[index]["attempt_dir"] = str(result.get("summary", {}).get("attempt_dir", ""))
            if result["status"] in {"SKIPPED_COMPLETED", "SKIPPED_REUSED"}:
                skipped += 1
            elif result["status"] == "REUSED":
                reused += 1
            else:
                completed += 1
        except Exception as exc:
            failed += 1
            queue_rows[index]["status"] = "FAILED"
            queue_rows[index]["error"] = f"{type(exc).__name__}: {exc}"
            if stop_on_error:
                _atomic_csv(queue_path, queue_rows)
                write_status("INCOMPLETE", None, index + 1)
                raise
        _atomic_csv(queue_path, queue_rows)
        write_status("RUNNING", None, index + 1)

    audits = [_audit_cell(output, cell, current_manifest) for cell in cells]
    valid_cells = len(audits) == len(cells) and all(item["accepted"] for item in audits)
    terminal_valid = valid_cells
    summaries = [_read_json(_cell_dir(output, cell) / "summary.json") for cell in cells] if terminal_valid else []
    summaries = [item for item in summaries if isinstance(item, Mapping)]
    if terminal_valid:
        summary_rows = [_summary_row(summary) for summary in summaries]
        _atomic_csv(output / "final_summary.csv", summary_rows)
        _atomic_csv(output / "aggregate_mean_std.csv", _aggregate_rows(summary_rows))
        _atomic_csv(output / "paired_comparisons.csv", _paired_rows(summary_rows))
        marker = {
            "status": TERMINAL_STATUS,
            "protocol_id": current_manifest["protocol_id"],
            "scope": scope,
            "cells_expected": len(cells),
            "cells_completed": len(summaries),
            "final_summary_sha256": sha256_file(output / "final_summary.csv"),
            "aggregate_mean_std_sha256": sha256_file(output / "aggregate_mean_std.csv"),
            "paired_comparisons_sha256": sha256_file(output / "paired_comparisons.csv"),
            "cell_summary_sha256": {_cell_id(cell): sha256_file(_cell_dir(output, cell) / "summary.json") for cell in cells},
            "completed_at": _utc_now(),
        }
        _atomic_json(output / "BATCH_COMPLETED", marker)
        final_status = TERMINAL_STATUS
    else:
        final_status = "COMPLETED_WITH_FAILURES"
    write_status(final_status, None, len(cells))
    return {
        "status": final_status,
        "protocol_id": current_manifest["protocol_id"],
        "scope": scope,
        "cells_expected": len(cells),
        "cells_valid": sum(bool(item["accepted"]) for item in audits),
        "cells_completed_this_invocation": completed,
        "cells_skipped_completed": skipped,
        "cells_reused": reused,
        "cells_failed": failed,
        "invalid_cells": [item["cell_id"] for item in audits if not item["accepted"]],
        "elapsed_seconds": float(time.perf_counter() - started),
        "completed_at": _utc_now(),
    }


def run_queue(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    output: Path | None = None,
    resume: bool = False,
    run_formal: bool = False,
    stop_on_error: bool = False,
    model_factory: Callable[[Mapping[str, Any], tuple[int, int, int], Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    if not smoke and (not run_formal or str(protocol.get("status")) != "FROZEN" or not bool(protocol.get("formal_authorized"))):
        raise PermissionError("formal RQ6 execution requires --run-formal and authorized FROZEN protocol")
    scope = SMOKE_SCOPE if smoke else FORMAL_SCOPE
    resolved = _configured_output(protocol, scope, output)
    lock = _acquire_batch_lock(resolved, scope=scope, protocol_id=str(protocol.get("protocol_id", "")))
    try:
        return _run_queue_unlocked(protocol, smoke=smoke, output=resolved, resume=resume, run_formal=run_formal, stop_on_error=stop_on_error, model_factory=model_factory)
    finally:
        lock.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="inspect scope without writing or training")
    parser.add_argument("--smoke", action="store_true", help="run the protocol-defined smoke scope")
    parser.add_argument("--run-formal", action="store_true", help="explicitly authorize the formal scope")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--audit", nargs="?", const=Path(""), default=None, type=Path, help="audit an existing RQ6 output")
    parser.add_argument("--unlock", action="store_true", help="remove an inactive RQ6 lock after checking its PID")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    actions = sum(bool(value) for value in (args.dry_run, args.run_formal, args.audit is not None, args.unlock))
    if actions > 1:
        raise ValueError("choose exactly one action: dry-run, audit, unlock, or run")
    if args.run_formal and args.smoke:
        raise ValueError("--run-formal and --smoke cannot be combined")
    _require_protocol_api()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    smoke = bool(args.smoke)
    if not any((args.dry_run, args.smoke, args.run_formal, args.audit is not None, args.unlock)):
        args.dry_run = True
    scope = SMOKE_SCOPE if smoke else FORMAL_SCOPE
    output = _configured_output(protocol, scope, args.output)
    if args.unlock:
        print(json.dumps(_jsonable(unlock_output(output)), ensure_ascii=False, indent=2))
        return 0
    if args.audit is not None:
        audit_arg = args.output if str(args.audit) in {"", "."} else Path(args.audit)
        result = audit_batch(protocol, output=_configured_output(protocol, scope, audit_arg), smoke=smoke)
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
            "output": str(output),
        }
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        return 0
    result = run_queue(protocol, smoke=smoke, output=output, resume=args.resume, run_formal=args.run_formal, stop_on_error=args.stop_on_error)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") == TERMINAL_STATUS else 1


if __name__ == "__main__":                    
    raise SystemExit(main())

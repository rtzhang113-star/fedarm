                                                                       

                                                                             
                                                                              
                                                                               
                                                                           
                                                                    
   

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_value_cache
from .protocol import make_causal_split
from .residual_model import CausalResidualBiasMF


FROZEN_SEED = 42
FROZEN_DENSITY = 0.1
FROZEN_RANK = 20
FROZEN_FACTOR_ETA = 0.01
FROZEN_TIME_SLICES = 64
FROZEN_QOS_TYPES = ("rt", "tp")
FROZEN_HISTORY_WINDOW = 8
FROZEN_MIN_HISTORY = 8
FROZEN_ROBUST_Z_THRESHOLD = 3.5
FROZEN_USER_OBSERVATION_THRESHOLD = 32
FROZEN_SERVICE_OBSERVATION_THRESHOLD = 8
FROZEN_RESIDUAL_CLIP = 3.0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return _sha256(path)


def _jsonable_finite(value: Any) -> bool:
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (int, np.integer, bool, str)) or value is None:
        return True
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, dict):
        return all(_jsonable_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_jsonable_finite(item) for item in value)
    return True


def raw_values_to_logit(raw_values: np.ndarray, raw_cap: float) -> np.ndarray:
                                                                               

    values = np.asarray(raw_values, dtype=np.float64)
    if not np.isfinite(raw_cap) or raw_cap <= 0:
        raise ValueError("raw_cap must be finite and positive")
    if not np.all(np.isfinite(values)) or not np.all((values > 0) & (values < raw_cap)):
        raise ValueError("support values must be finite and inside the fixed QoS cap")
    normalized = np.log1p(values) / np.log1p(float(raw_cap))
    if not np.all((normalized > 0.0) & (normalized < 1.0)):
        raise ValueError("normalized support targets must lie strictly inside (0, 1)")
    return np.log(normalized / (1.0 - normalized))


def pre_update_logit_residuals(
    model: CausalResidualBiasMF,
    users: np.ndarray,
    services: np.ndarray,
    raw_values: np.ndarray,
    raw_cap: float,
) -> np.ndarray:
                                                                          

                                                                              
                                                                    
       

    users = np.asarray(users, dtype=np.int64)
    services = np.asarray(services, dtype=np.int64)
    raw_values = np.asarray(raw_values)
    if users.ndim != 1 or services.ndim != 1 or raw_values.ndim != 1:
        raise ValueError("support arrays must be one-dimensional")
    if not (users.size == services.size == raw_values.size) or users.size == 0:
        raise ValueError("support arrays must be non-empty and equally sized")
    target_logits = raw_values_to_logit(raw_values, raw_cap)
    model_logits = model.user_bias[users] + model.service_bias[services]
    if model.interaction:
        model_logits = model_logits + np.sum(
            model.user_factors[users] * model.service_factors[services], axis=1
        )
    residuals = target_logits - np.asarray(model_logits, dtype=np.float64)
    if not np.all(np.isfinite(residuals)):
        raise FloatingPointError("non-finite pre-update logit residual")
                                                                             
                                                    
    return np.clip(residuals, -FROZEN_RESIDUAL_CLIP, FROZEN_RESIDUAL_CLIP)


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("RMS requires a non-empty finite array")
    return float(np.sqrt(np.mean(np.square(values))))


class RobustDriftTracker:
                                                                    

                                                                            
                                                                            
                                                                          
                                                             
       

    def __init__(
        self,
        history_window: int = FROZEN_HISTORY_WINDOW,
        min_history: int = FROZEN_MIN_HISTORY,
        robust_z_threshold: float = FROZEN_ROBUST_Z_THRESHOLD,
    ) -> None:
        if history_window <= 0 or min_history <= 0:
            raise ValueError("history_window and min_history must be positive")
        if min_history > history_window:
            raise ValueError("min_history cannot exceed history_window")
        if not np.isfinite(robust_z_threshold):
            raise ValueError("robust_z_threshold must be finite")
        self.history_window = int(history_window)
        self.min_history = int(min_history)
        self.robust_z_threshold = float(robust_z_threshold)
        self.histories: dict[str, dict[str, deque[float]]] = {
            "global": defaultdict(lambda: deque(maxlen=self.history_window)),
            "user": defaultdict(lambda: deque(maxlen=self.history_window)),
            "service": defaultdict(lambda: deque(maxlen=self.history_window)),
        }

    @staticmethod
    def _key(qos_type: str, entity_id: int) -> str:
        return f"{qos_type}:{int(entity_id)}"

    def score(
        self,
        granularity: str,
        qos_type: str,
        entity_id: int,
        time_slice: int,
        score: float,
        current_observations: int,
        observation_threshold: int,
        mask_sha256: str,
    ) -> dict[str, Any]:
        if granularity not in self.histories:
            raise ValueError(f"unknown granularity={granularity!r}")
        if not np.isfinite(score) or score < 0:
            raise ValueError("score must be finite and non-negative")
        history = self.histories[granularity][self._key(qos_type, entity_id)]
        past = np.asarray(list(history), dtype=np.float64)
        ready = int(past.size) >= self.min_history
        median: float | None = None
        mad: float | None = None
        robust_scale: float | None = None
        robust_z: float | None = None
        flag = False
        if ready:
            median = float(np.median(past))
            mad = float(np.median(np.abs(past - median)))
            robust_scale = float(max(1.4826 * mad, 1e-12))
            robust_z = float((float(score) - median) / robust_scale)
            flag = bool(robust_z > self.robust_z_threshold)
        row = {
            "qos_type": str(qos_type),
            "granularity": str(granularity),
            "entity_id": int(entity_id),
            "time_slice": int(time_slice),
            "current_observations": int(current_observations),
            "observation_threshold": int(observation_threshold),
            "score_rms_pre_update_logit_residual": float(score),
            "history_count_before": int(past.size),
            "history_window": int(self.history_window),
            "min_history": int(self.min_history),
            "history_ready": bool(ready),
            "history_median": median,
            "history_mad": mad,
            "robust_scale": robust_scale,
            "robust_z": robust_z,
            "flag": bool(flag),
            "mask_sha256": str(mask_sha256),
        }
        if not _jsonable_finite(row):
            raise FloatingPointError("non-finite robust drift row")
                                                                                
                                                                      
        history.append(float(score))
        return row

    def export_state(self) -> dict[str, Any]:
        return {
            "history_window": self.history_window,
            "min_history": self.min_history,
            "robust_z_threshold": self.robust_z_threshold,
            "histories": {
                granularity: {
                    key: [float(value) for value in history]
                    for key, history in sorted(mapping.items())
                    if history
                }
                for granularity, mapping in self.histories.items()
            },
        }


def _model_checkpoint(path: Path, model: CausalResidualBiasMF) -> str:
    snapshot = model.parameter_snapshot()
    np.savez(path, **{name: np.asarray(value, dtype=np.float64) for name, value in snapshot.items()})
    return _sha256(path)


def _state_digest(model: CausalResidualBiasMF) -> str:
    digest = hashlib.sha256()
    snapshot = model.parameter_snapshot()
    for name in sorted(snapshot):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(snapshot[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event_intervals(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        if bool(row.get("flag", False)):
            grouped[tuple(row[field] for field in key_fields)].append(int(row["time_slice"]))
    intervals: list[dict[str, Any]] = []
    for key, times in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        times = sorted(set(times))
        start = previous = times[0]
        for current in times[1:]:
            if current != previous + 1:
                intervals.append({**dict(zip(key_fields, key)), "start": start, "end": previous, "length": previous - start + 1})
                start = current
            previous = current
        intervals.append({**dict(zip(key_fields, key)), "start": start, "end": previous, "length": previous - start + 1})
    return intervals


def _top_entities(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["qos_type"]), int(row["entity_id"]))].append(row)
    top: list[dict[str, Any]] = []
    for (qos_type, entity_id), selected in grouped.items():
        z_values = [float(row["robust_z"]) for row in selected if row.get("robust_z") is not None]
        score_values = [float(row["score_rms_pre_update_logit_residual"]) for row in selected]
        top.append(
            {
                "qos_type": qos_type,
                "entity_id": entity_id,
                "flag_count": int(sum(bool(row["flag"]) for row in selected)),
                "max_robust_z": float(max(z_values)) if z_values else None,
                "max_score": float(max(score_values)),
                "scored_slices": len(selected),
            }
        )
    top.sort(
        key=lambda row: (
            row["flag_count"],
            -float("inf") if row["max_robust_z"] is None else row["max_robust_z"],
            row["max_score"],
            row["qos_type"],
            -row["entity_id"],
        ),
        reverse=True,
    )
    return top[: int(limit)]


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    left, right = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _query_audit(
    query_audit_dir: Path | None,
    qos_types: tuple[str, ...],
    mask_by_qos: dict[str, dict[int, str]],
    expected_slices: int,
) -> dict[str, Any]:
                                                                          

    audit: dict[str, Any] = {
        "status": "NOT_REQUESTED" if query_audit_dir is None else "NOT_AVAILABLE",
        "used_for_drift": False,
        "used_for_threshold": False,
        "used_for_training": False,
        "descriptive_only": True,
        "mask_consistent": None,
        "per_qos": {},
    }
    if query_audit_dir is None:
        return audit
    all_consistent = True
    any_available = False
    for qos_type in qos_types:
        path = query_audit_dir / qos_type / "per_slice_metrics.csv"
        if not path.is_file():
            audit["per_qos"][qos_type] = {"status": "NOT_AVAILABLE", "path": str(path.resolve())}
            continue
        any_available = True
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_slices: set[int] = set()
        mask_ok = True
        for row in rows:
            time_slice = int(row["time_slice"])
            seen_slices.add(time_slice)
            expected_mask = mask_by_qos[qos_type].get(time_slice)
            mask_ok &= expected_mask is not None and str(row.get("mask_sha256")) == expected_mask
            method_rows[str(row["method"])].append(row)
        mask_ok &= seen_slices == set(range(int(expected_slices)))
        mask_ok &= seen_slices == set(mask_by_qos[qos_type])
        all_consistent &= bool(mask_ok)
        audit["per_qos"][qos_type] = {
            "status": "AVAILABLE",
            "path": str(path.resolve()),
            "row_count": len(rows),
            "mask_consistent": bool(mask_ok),
            "methods": sorted(method_rows),
        }
    audit["status"] = "COMPLETED" if any_available else "NOT_AVAILABLE"
    audit["mask_consistent"] = bool(all_consistent) if any_available else None
    return audit


def _query_audit_with_scores(
    query_audit_dir: Path | None,
    qos_types: tuple[str, ...],
    mask_by_qos: dict[str, dict[int, str]],
    global_rows_by_qos: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
                                                                                      

    audit = _query_audit(query_audit_dir, qos_types, mask_by_qos, FROZEN_TIME_SLICES)
    if audit["status"] != "COMPLETED" or audit["mask_consistent"] is not True:
        return audit
    for qos_type in qos_types:
        path = Path(audit["per_qos"][qos_type]["path"])
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        score_rows = {
            int(row["time_slice"]): float(row["score_rms_pre_update_logit_residual"])
            for row in global_rows_by_qos[qos_type]
        }
        by_method: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
        for row in rows:
            time_slice = int(row["time_slice"])
            if time_slice in score_rows:
                by_method[str(row["method"])].append(
                    (time_slice, score_rows[time_slice], float(row["mae"]), float(row["rmse"]))
                )
        correlations: dict[str, Any] = {}
        for method, selected in sorted(by_method.items()):
            selected.sort()
            post_warmup = [item for item in selected if item[0] >= FROZEN_MIN_HISTORY]
            score_changes = [
                post_warmup[index][1] - post_warmup[index - 1][1]
                for index in range(1, len(post_warmup))
            ]
            mae_changes = [
                post_warmup[index][2] - post_warmup[index - 1][2]
                for index in range(1, len(post_warmup))
            ]
            rmse_changes = [
                post_warmup[index][3] - post_warmup[index - 1][3]
                for index in range(1, len(post_warmup))
            ]
            correlations[method] = {
                "slices": len(selected),
                "pearson_score_vs_query_mae": _pearson(
                    [item[1] for item in selected], [item[2] for item in selected]
                ),
                "pearson_score_vs_query_rmse": _pearson(
                    [item[1] for item in selected], [item[3] for item in selected]
                ),
                "post_warmup_first_difference_slices": len(score_changes),
                "pearson_first_difference_score_vs_query_mae": _pearson(
                    score_changes, mae_changes
                ),
                "pearson_first_difference_score_vs_query_rmse": _pearson(
                    score_changes, rmse_changes
                ),
                "interpretation_boundary": (
                    "raw correlation includes shared temporal trend; first-difference "
                    "correlation is the conservative short-term association"
                ),
            }
        audit["per_qos"][qos_type]["pearson_correlations"] = correlations
    return audit


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("seed", -1)) != FROZEN_SEED:
        raise ValueError("drift diagnostic is frozen to seed 42")
    if not np.isclose(float(config.get("density", -1.0)), FROZEN_DENSITY):
        raise ValueError("drift diagnostic is frozen to D10")
    if tuple(config.get("qos_types", ())) != FROZEN_QOS_TYPES:
        raise ValueError("drift diagnostic requires RT and TP in this order")
    shape = tuple(int(value) for value in config.get("shape_time_user_service", ()))
    if len(shape) != 3 or shape[0] != FROZEN_TIME_SLICES:
        raise ValueError("shape_time_user_service must contain 64 time slices")
    if int(config.get("expected_time_slices", -1)) != FROZEN_TIME_SLICES:
        raise ValueError("expected_time_slices must be 64")
    if int(config.get("rank", -1)) != FROZEN_RANK:
        raise ValueError("drift diagnostic is frozen to rank 20")
    if not np.isclose(float(config.get("factor_eta", -1.0)), FROZEN_FACTOR_ETA):
        raise ValueError("drift diagnostic is frozen to factor_eta=0.01")
    if int(config.get("history_window", -1)) != FROZEN_HISTORY_WINDOW:
        raise ValueError("history_window must be 8")
    if int(config.get("min_history", -1)) != FROZEN_MIN_HISTORY:
        raise ValueError("min_history must be 8")
    if not np.isclose(float(config.get("robust_z_threshold", -1.0)), FROZEN_ROBUST_Z_THRESHOLD):
        raise ValueError("robust z threshold must be 3.5")
    thresholds = config.get("entity_observation_thresholds", {})
    if int(thresholds.get("user", -1)) != FROZEN_USER_OBSERVATION_THRESHOLD:
        raise ValueError("user observation threshold must be 32")
    if int(thresholds.get("service", -1)) != FROZEN_SERVICE_OBSERVATION_THRESHOLD:
        raise ValueError("service observation threshold must be 8")
    if int(config.get("minimum_user_observations", FROZEN_USER_OBSERVATION_THRESHOLD)) != FROZEN_USER_OBSERVATION_THRESHOLD:
        raise ValueError("minimum_user_observations must be 32")
    if int(config.get("minimum_service_observations", FROZEN_SERVICE_OBSERVATION_THRESHOLD)) != FROZEN_SERVICE_OBSERVATION_THRESHOLD:
        raise ValueError("minimum_service_observations must be 8")
    clip = config.get("residual_clip", (-FROZEN_RESIDUAL_CLIP, FROZEN_RESIDUAL_CLIP))
    if len(clip) != 2 or not np.allclose(np.asarray(clip, dtype=np.float64), [-FROZEN_RESIDUAL_CLIP, FROZEN_RESIDUAL_CLIP]):
        raise ValueError("residual_clip must remain [-3, 3]")


def _run_qos(
    values: np.ndarray,
    qos_type: str,
    config: dict[str, Any],
    output_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, str], list[dict[str, Any]]]:
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] != FROZEN_TIME_SLICES:
        raise ValueError(f"{qos_type} cache must have 64 time x user x service shape")
    num_times, num_users, num_services = values.shape
    common = dict(
        num_users=num_users,
        num_services=num_services,
        rank=int(config["rank"]),
        seed=int(config["seed"]),
        bias_eta=float(config["bias_eta"]),
        factor_eta=float(config["factor_eta"]),
        l2=float(config["l2"]),
        beta=float(config["beta"]),
        factor_init_std=float(config["factor_init_std"]),
    )
    model = CausalResidualBiasMF(interaction=True, **common)
    tracker = RobustDriftTracker(
        history_window=int(config["history_window"]),
        min_history=int(config["min_history"]),
        robust_z_threshold=float(config["robust_z_threshold"]),
    )
    raw_cap = float(config["raw_prediction_caps"][qos_type])
    thresholds = config["entity_observation_thresholds"]
    global_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    service_rows: list[dict[str, Any]] = []
    mask_by_slice: dict[int, str] = {}
    mask_digest = hashlib.sha256()
    started = time.perf_counter()

    for time_slice in range(num_times):
        split = make_causal_split(values[time_slice], float(config["density"]), int(config["seed"]), time_slice)
        if split.support_count <= 0:
            raise RuntimeError(f"empty support at {qos_type} time={time_slice}")
        mask_by_slice[time_slice] = split.mask_sha256
        mask_digest.update(bytes.fromhex(split.mask_sha256))

                                                                                
                                                                       
        support_users = split.support_users
        support_services = split.support_services
        support_values = split.support_values
        residuals = pre_update_logit_residuals(
            model, support_users, support_services, support_values, raw_cap
        )
        global_score = _rms(residuals)
        global_rows.append(
            tracker.score(
                "global", qos_type, 0, time_slice, global_score,
                int(residuals.size), int(residuals.size), split.mask_sha256,
            )
        )

        residual_squares = np.square(residuals, dtype=np.float64)
        for granularity, ids, entity_count, threshold, target_rows in (
            ("user", support_users, num_users, int(thresholds["user"]), user_rows),
            ("service", support_services, num_services, int(thresholds["service"]), service_rows),
        ):
            counts = np.bincount(ids, minlength=entity_count)
            squared_sums = np.bincount(ids, weights=residual_squares, minlength=entity_count)
            for entity_id in np.flatnonzero(counts >= threshold):
                count = int(counts[entity_id])
                target_rows.append(
                    tracker.score(
                        granularity,
                        qos_type,
                        int(entity_id),
                        time_slice,
                        float(np.sqrt(squared_sums[entity_id] / count)),
                        count,
                        threshold,
                        split.mask_sha256,
                    )
                )

                                                                              
        model.fit_support(
            support_users,
            support_services,
            support_values,
            raw_cap=raw_cap,
            bias_epochs=int(config["bias_epochs"]),
            factor_epochs=int(config["factor_epochs"]),
            shuffle_seed=int(config["seed"]) * 1_000_003 + time_slice,
        )
        if not _jsonable_finite(model.parameter_snapshot()):
            raise FloatingPointError(f"non-finite model state at {qos_type} time={time_slice}")

    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "global_per_slice.csv"
    user_path = output_dir / "user_per_slice.csv"
    service_path = output_dir / "service_per_slice.csv"
    fields = list(global_rows[0])
    _write_csv(global_path, global_rows, fields)
    _write_csv(user_path, user_rows, fields)
    _write_csv(service_path, service_rows, fields)

    model_path = output_dir / "residual_mf_final.npz"
    model_hash = _model_checkpoint(model_path, model)
    tracker_state = tracker.export_state()
    tracker_path = output_dir / "tracker_state.json"
    tracker_hash = _write_json(tracker_path, tracker_state)
    summary = {
        "status": "COMPLETED",
        "qos_type": qos_type,
        "evidence_level": "seed_42_causal_pre_update_drift_diagnostic",
        "seed": int(config["seed"]),
        "density": float(config["density"]),
        "time_slices_completed": num_times,
        "time_slices_expected": FROZEN_TIME_SLICES,
        "support_only": True,
        "query_labels_used_for_drift": False,
        "query_labels_used_for_threshold": False,
        "query_labels_used_for_training": False,
        "future_slices_used": False,
        "diagnostic_score": "RMS(clip(target_logit - pre_update_model_logit, -3, 3))",
        "residual_clip": [-FROZEN_RESIDUAL_CLIP, FROZEN_RESIDUAL_CLIP],
        "fixed_raw_prediction_cap": raw_cap,
        "history_window": int(config["history_window"]),
        "min_history": int(config["min_history"]),
        "robust_z_threshold": float(config["robust_z_threshold"]),
        "entity_observation_thresholds": {
            "user": int(thresholds["user"]),
            "service": int(thresholds["service"]),
        },
        "minimum_user_observations": int(thresholds["user"]),
        "minimum_service_observations": int(thresholds["service"]),
        "support_entries_total": int(sum(int(row["current_observations"]) for row in global_rows)),
        "global_flags": [int(row["time_slice"]) for row in global_rows if row["flag"]],
        "flag_counts_by_slice": {
            "global": [
                {"time_slice": int(row["time_slice"]), "flag_count": int(bool(row["flag"]))}
                for row in global_rows
            ],
            "user": [
                {"time_slice": time_slice, "flag_count": int(sum(bool(row["flag"]) for row in user_rows if int(row["time_slice"]) == time_slice))}
                for time_slice in range(num_times)
            ],
            "service": [
                {"time_slice": time_slice, "flag_count": int(sum(bool(row["flag"]) for row in service_rows if int(row["time_slice"]) == time_slice))}
                for time_slice in range(num_times)
            ],
        },
        "top_entities": {
            "user": _top_entities(user_rows),
            "service": _top_entities(service_rows),
        },
        "continuous_event_intervals": {
            "global": _event_intervals(global_rows, ("qos_type",)),
            "user": _event_intervals(user_rows, ("qos_type", "entity_id")),
            "service": _event_intervals(service_rows, ("qos_type", "entity_id")),
        },
        "mask_stream_sha256": mask_digest.hexdigest(),
        "mask_sha256_by_slice": mask_by_slice,
        "data_manifest": manifest or {},
        "final_model_state_sha256": _state_digest(model),
        "final_tracker_state_sha256": tracker_hash,
        "artifacts": {
            "global_per_slice": str(global_path.resolve()),
            "user_per_slice": str(user_path.resolve()),
            "service_per_slice": str(service_path.resolve()),
            "final_model_state": str(model_path.resolve()),
            "tracker_state": str(tracker_path.resolve()),
            "final_model_checkpoint_sha256": model_hash,
            "tracker_state_sha256": tracker_hash,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    mmap_handle = getattr(values, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()
    return summary, mask_by_slice, global_rows


def run_drift_diagnostic(config_path: Path, dry_run: bool = False) -> dict[str, Any]:
                                                                               

    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    output_root = _resolve(root, config["output_dir"])
    if dry_run:
        return {
            "status": "DRY_RUN",
            "execution_started": False,
            "expected_time_slices": FROZEN_TIME_SLICES,
            "expected_cells": FROZEN_TIME_SLICES * len(FROZEN_QOS_TYPES),
            "seed": FROZEN_SEED,
            "density": FROZEN_DENSITY,
            "history_window": FROZEN_HISTORY_WINDOW,
            "min_history": FROZEN_MIN_HISTORY,
            "robust_z_threshold": FROZEN_ROBUST_Z_THRESHOLD,
            "minimum_user_observations": FROZEN_USER_OBSERVATION_THRESHOLD,
            "minimum_service_observations": FROZEN_SERVICE_OBSERVATION_THRESHOLD,
            "residual_clip": [-FROZEN_RESIDUAL_CLIP, FROZEN_RESIDUAL_CLIP],
            "query_labels_used_for_drift": False,
            "output_dir": str(output_root),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    _write_json(status_path, {"status": "RUNNING", "execution_started": True})
    started = time.perf_counter()
    try:
        cache_dir = _resolve(root, config["cache_dir"])
        summaries: dict[str, dict[str, Any]] = {}
        masks: dict[str, dict[int, str]] = {}
        global_rows: dict[str, list[dict[str, Any]]] = {}
        for qos_type in FROZEN_QOS_TYPES:
            cache_path = cache_dir / f"{qos_type}_values.npy"
            manifest_path = cache_dir / f"{qos_type}_data_manifest.json"
            values = load_value_cache(cache_path)
            if tuple(values.shape) != tuple(int(value) for value in config["shape_time_user_service"]):
                raise ValueError(f"{qos_type} cache shape {values.shape} does not match frozen config")
            manifest = None
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["manifest_sha256"] = _sha256(manifest_path)
            summary, mask_by_slice, rows = _run_qos(
                values, qos_type, config, output_root / qos_type, manifest
            )
            summaries[qos_type] = summary
            masks[qos_type] = mask_by_slice
            global_rows[qos_type] = rows

        query_audit_dir = None
        if config.get("query_audit_dir"):
            query_audit_dir = _resolve(root, config["query_audit_dir"])
        query_audit = _query_audit_with_scores(
            query_audit_dir, FROZEN_QOS_TYPES, masks, global_rows
        )
        result = {
            "status": "COMPLETED",
            "execution_started": True,
            "evidence_level": "seed_42_causal_pre_update_drift_diagnostic",
            "seed": FROZEN_SEED,
            "density": FROZEN_DENSITY,
            "qos_types_completed": list(FROZEN_QOS_TYPES),
            "time_slices_completed": FROZEN_TIME_SLICES,
            "time_slices_expected": FROZEN_TIME_SLICES,
            "query_labels_used_for_drift": False,
            "query_labels_used_for_threshold": False,
            "query_labels_used_for_training": False,
            "future_slices_used": False,
            "causal_boundary": {
                "drift_input": "current slice support only",
                "update_order": "score current support before fit_support",
                "query_metrics": "isolated post-hoc descriptive audit only",
                "future_slices": "not read before their slice",
            },
            "diagnostic_formula": "RMS(clip(logit(log1p(y)/log1p(cap)) - pre_update_model_logit, -3, 3))",
            "residual_clip": [-FROZEN_RESIDUAL_CLIP, FROZEN_RESIDUAL_CLIP],
            "history_window": FROZEN_HISTORY_WINDOW,
            "min_history": FROZEN_MIN_HISTORY,
            "robust_z_threshold": FROZEN_ROBUST_Z_THRESHOLD,
            "entity_observation_thresholds": config["entity_observation_thresholds"],
            "minimum_user_observations": int(config["entity_observation_thresholds"]["user"]),
            "minimum_service_observations": int(config["entity_observation_thresholds"]["service"]),
            "global_flags": {
                qos_type: summaries[qos_type]["global_flags"] for qos_type in FROZEN_QOS_TYPES
            },
            "flag_counts_by_slice": {
                qos_type: summaries[qos_type]["flag_counts_by_slice"] for qos_type in FROZEN_QOS_TYPES
            },
            "top_entities": {
                qos_type: summaries[qos_type]["top_entities"] for qos_type in FROZEN_QOS_TYPES
            },
            "continuous_event_intervals": {
                qos_type: summaries[qos_type]["continuous_event_intervals"]
                for qos_type in FROZEN_QOS_TYPES
            },
            "mask_stream_sha256": {
                qos_type: summaries[qos_type]["mask_stream_sha256"] for qos_type in FROZEN_QOS_TYPES
            },
            "summaries": summaries,
            "query_audit": query_audit,
            "config_sha256": _sha256(config_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "summary.json", result)
        _write_json(status_path, result)
        return result
    except Exception as exc:
        _write_json(
            status_path,
            {
                "status": "FAILED",
                "execution_started": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

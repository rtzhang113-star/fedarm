                                                                   

                                                                            
                                                                            
                                                                     
                                                                            
                                                                          
                      
   

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


DEFAULT_MIN_NORM = 1e-12


def _finite_delta(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class StablePlasticProjection:
                                                                            

    current_delta: np.ndarray
    projected_delta: np.ndarray
    previous_delta: np.ndarray | None
    reason: str
    applied: bool
    conflict: bool
    dot_product: float | None
    current_norm: float
    previous_norm: float | None
    projected_norm: float
    projected_norm_ratio: float
    removed_norm: float
    removed_fraction: float
    history_updated: bool = False

    def with_history_updated(self, value: bool) -> "StablePlasticProjection":
        return replace(self, history_updated=bool(value))

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "applied": bool(self.applied),
            "conflict": bool(self.conflict),
            "dot_product": self.dot_product,
            "current_norm": float(self.current_norm),
            "previous_norm": self.previous_norm,
            "projected_norm": float(self.projected_norm),
            "projected_norm_ratio": float(self.projected_norm_ratio),
            "removed_norm": float(self.removed_norm),
            "removed_fraction": float(self.removed_fraction),
            "history_updated": bool(self.history_updated),
        }


def project_conflicting_factor_delta(
    current_delta: Any,
    previous_delta: Any | None,
    *,
    min_norm: float = DEFAULT_MIN_NORM,
) -> StablePlasticProjection:
                                                                            

                                                                         
                                                                           
                                                                            
               
       

    threshold = float(min_norm)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("min_norm must be a positive finite number")
    current = _finite_delta(current_delta, "current_delta")
    previous = None if previous_delta is None else _finite_delta(previous_delta, "previous_delta")
    if previous is not None and previous.shape != current.shape:
        raise ValueError("current_delta and previous_delta must have the same shape")
    previous_norm = None if previous is None else float(np.linalg.norm(previous))
    current_norm = float(np.linalg.norm(current))
    if current_norm <= threshold:
        return StablePlasticProjection(
            current_delta=current, projected_delta=current.copy(), previous_delta=previous,
            reason="current_delta_too_small", applied=False, conflict=False,
            dot_product=None, current_norm=current_norm, previous_norm=previous_norm,
            projected_norm=current_norm, projected_norm_ratio=1.0,
            removed_norm=0.0, removed_fraction=0.0,
        )
    if previous is None:
        return StablePlasticProjection(
            current_delta=current, projected_delta=current.copy(), previous_delta=None,
            reason="first_observation", applied=False, conflict=False,
            dot_product=None, current_norm=current_norm, previous_norm=None,
            projected_norm=current_norm, projected_norm_ratio=1.0,
            removed_norm=0.0, removed_fraction=0.0,
        )

    assert previous_norm is not None
    if previous_norm <= threshold:
        return StablePlasticProjection(
            current_delta=current, projected_delta=current.copy(), previous_delta=previous,
            reason="previous_delta_too_small", applied=False, conflict=False,
            dot_product=None, current_norm=current_norm, previous_norm=previous_norm,
            projected_norm=current_norm, projected_norm_ratio=1.0,
            removed_norm=0.0, removed_fraction=0.0,
        )

    dot = float(np.dot(current, previous))
    if dot >= 0.0:
        return StablePlasticProjection(
            current_delta=current, projected_delta=current.copy(), previous_delta=previous,
            reason="non_conflicting", applied=False, conflict=False,
            dot_product=dot, current_norm=current_norm, previous_norm=previous_norm,
            projected_norm=current_norm, projected_norm_ratio=1.0,
            removed_norm=0.0, removed_fraction=0.0,
        )

    coefficient = dot / (previous_norm * previous_norm)
    projected = current - coefficient * previous
    projected_norm = float(np.linalg.norm(projected))
    removed = current - projected
    removed_norm = float(np.linalg.norm(removed))
    return StablePlasticProjection(
        current_delta=current, projected_delta=projected, previous_delta=previous,
        reason="conflicting_component_removed", applied=True, conflict=True,
        dot_product=dot, current_norm=current_norm, previous_norm=previous_norm,
        projected_norm=projected_norm, projected_norm_ratio=projected_norm / current_norm,
        removed_norm=removed_norm, removed_fraction=removed_norm / current_norm,
    )


class StablePlasticProjectionTracker:
                                                                               

    def __init__(self, *, min_norm: float = DEFAULT_MIN_NORM) -> None:
        threshold = float(min_norm)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("min_norm must be a positive finite number")
        self.min_norm = threshold
        self._history: dict[int, tuple[int, np.ndarray]] = {}

    def commit(
        self,
        service_id: int,
        time_slice: int,
        current_delta: Any,
    ) -> StablePlasticProjection:
                                                                                  

        service = int(service_id)
        previous_record = self._history.get(service)
        previous = None if previous_record is None else previous_record[1]
        result = project_conflicting_factor_delta(
            current_delta, previous, min_norm=self.min_norm
        )
        if result.projected_norm > self.min_norm:
            self._history[service] = (int(time_slice), result.projected_delta.copy())
            result = result.with_history_updated(True)
        return result

    def last_committed_delta(self, service_id: int) -> np.ndarray | None:
        record = self._history.get(int(service_id))
        return None if record is None else record[1].copy()

    def history_snapshot(self) -> dict[int, dict[str, object]]:
        return {
            int(service): {"time_slice": int(time_slice), "delta": delta.copy()}
            for service, (time_slice, delta) in self._history.items()
        }


                                                                           
            
StablePlasticTracker = StablePlasticProjectionTracker
project_stable_plastic_delta = project_conflicting_factor_delta


__all__ = [
    "DEFAULT_MIN_NORM",
    "StablePlasticProjection",
    "StablePlasticProjectionTracker",
    "StablePlasticTracker",
    "project_conflicting_factor_delta",
    "project_stable_plastic_delta",
]

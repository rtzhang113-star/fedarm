                                                                          

                                                                        
                                                                      
                                                                            
                                                                         
                                                              
   

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .federated_residual import FederatedResidualBiasMF, ServiceProposal
from .stable_plastic_projection import (
    DEFAULT_MIN_NORM,
    StablePlasticProjection,
    StablePlasticProjectionTracker,
)


FROZEN_MIN_NORM = DEFAULT_MIN_NORM


@dataclass(frozen=True)
class StablePlasticServiceDiagnostic:
                                                                        

    service_id: int
    proposal_count: int
    client_ids: tuple[int, ...]
    time_slice: int
    factor_raw_delta_norm: float
    factor_previous_delta_norm: float | None
    factor_dot_previous: float | None
    factor_projected_delta_norm: float
    factor_projected_norm_ratio: float
    factor_removed_fraction: float
    factor_projected_dot_previous: float | None
    factor_norm_nonincreasing: bool
    projection_triggered: bool
    projection_applied: bool
    projection_reason: str
    history_updated: bool
    bias_average: float
    error_average: float

    def as_dict(self) -> dict[str, object]:
        return {
            "service_id": int(self.service_id),
            "proposal_count": int(self.proposal_count),
            "client_ids": [int(value) for value in self.client_ids],
            "time_slice": int(self.time_slice),
            "factor_raw_delta_norm": float(self.factor_raw_delta_norm),
            "factor_previous_delta_norm": self.factor_previous_delta_norm,
            "factor_dot_previous": self.factor_dot_previous,
            "factor_projected_delta_norm": float(self.factor_projected_delta_norm),
            "factor_projected_norm_ratio": float(self.factor_projected_norm_ratio),
            "factor_removed_fraction": float(self.factor_removed_fraction),
            "factor_projected_dot_previous": self.factor_projected_dot_previous,
            "factor_norm_nonincreasing": bool(self.factor_norm_nonincreasing),
            "projection_triggered": bool(self.projection_triggered),
            "projection_applied": bool(self.projection_applied),
            "projection_reason": str(self.projection_reason),
            "history_updated": bool(self.history_updated),
            "bias_average": float(self.bias_average),
            "error_average": float(self.error_average),
        }


def _proposal_arrays(
    proposals: tuple[ServiceProposal, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                                                         

    factors = np.stack(
        [proposal.service_factors for proposal in proposals], axis=0
    ).astype(np.float64, copy=False)
    biases = np.asarray(
        [proposal.service_bias for proposal in proposals], dtype=np.float64
    )
    errors = np.asarray(
        [proposal.service_error for proposal in proposals], dtype=np.float64
    )
    return factors, biases, errors


class StablePlasticFederatedResidualBiasMF(FederatedResidualBiasMF):
                                                                            

    def __init__(
        self,
        *args: object,
        min_norm: float = FROZEN_MIN_NORM,
        **kwargs: object,
    ) -> None:
        threshold = float(min_norm)
        if not np.isfinite(threshold) or threshold != FROZEN_MIN_NORM:
            raise ValueError(f"min_norm is frozen to {FROZEN_MIN_NORM}")
        super().__init__(*args, **kwargs)
        self.min_norm = threshold
        self._projection_tracker = StablePlasticProjectionTracker(min_norm=threshold)
        self._round_index = 0

    @property
    def projection_tracker(self) -> StablePlasticProjectionTracker:
        return self._projection_tracker

    @property
    def round_index(self) -> int:
        return int(self._round_index)

    def advance_support(self, *args: object, **kwargs: object):
                                                                            

        tracker_before = copy.deepcopy(self._projection_tracker)
        round_before = int(self._round_index)
        try:
            result = super().advance_support(*args, **kwargs)
        except Exception:
            self._projection_tracker = tracker_before
            self._round_index = round_before
            raise
        self._round_index = round_before + 1
        return result

    def _commit_service_aggregates(
        self,
        by_service: dict[int, list[ServiceProposal]],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[StablePlasticServiceDiagnostic, ...]:
        diagnostics: list[StablePlasticServiceDiagnostic] = []
        for service in sorted(by_service):
            proposals = tuple(
                sorted(by_service[service], key=lambda item: int(item.client_id))
            )
            if not proposals:
                continue
            factors, biases, errors = _proposal_arrays(proposals)
            factor_average = np.mean(factors, axis=0, dtype=np.float64)
            bias_average = float(np.mean(biases, dtype=np.float64))
            error_average = float(np.mean(errors, dtype=np.float64))
            if not (
                np.all(np.isfinite(factor_average))
                and np.isfinite(bias_average)
                and np.isfinite(error_average)
            ):
                raise FloatingPointError(f"non-finite service mean for service {service}")

            raw_delta = factor_average - np.asarray(
                shared_snapshot["service_factors"][service], dtype=np.float64
            )
            projection: StablePlasticProjection = self._projection_tracker.commit(
                int(service), self._round_index, raw_delta
            )
            if projection.applied:
                                                                          
                                                                        
                                                                        
                                   
                committed_factors = np.asarray(
                    shared_snapshot["service_factors"][service], dtype=np.float64
                ) + projection.projected_delta
            else:
                                                                           
                                                                          
                                                                         
                                                                         
                committed_factors = factor_average.copy()
            if not np.all(np.isfinite(committed_factors)):
                raise FloatingPointError(f"non-finite projected service factors for {service}")

                                                                      
                                                                              
            self.service_factors[service] = committed_factors
            self.service_bias[service] = bias_average
            self.service_error[service] = error_average
            diagnostics.append(
                StablePlasticServiceDiagnostic(
                    service_id=int(service),
                    proposal_count=len(proposals),
                    client_ids=tuple(int(item.client_id) for item in proposals),
                    time_slice=int(self._round_index),
                    factor_raw_delta_norm=float(projection.current_norm),
                    factor_previous_delta_norm=projection.previous_norm,
                    factor_dot_previous=projection.dot_product,
                    factor_projected_delta_norm=float(projection.projected_norm),
                    factor_projected_norm_ratio=float(projection.projected_norm_ratio),
                    factor_removed_fraction=float(projection.removed_fraction),
                    factor_projected_dot_previous=(
                        None
                        if projection.previous_delta is None
                        else float(np.dot(projection.projected_delta, projection.previous_delta))
                    ),
                    factor_norm_nonincreasing=bool(
                        projection.projected_norm <= projection.current_norm + 1e-12
                    ),
                    projection_triggered=bool(projection.conflict),
                    projection_applied=bool(projection.applied),
                    projection_reason=str(projection.reason),
                    history_updated=bool(projection.history_updated),
                    bias_average=bias_average,
                    error_average=error_average,
                )
            )
        return tuple(diagnostics)


StablePlasticFederatedResidualMF = StablePlasticFederatedResidualBiasMF


__all__ = [
    "FROZEN_MIN_NORM",
    "StablePlasticFederatedResidualBiasMF",
    "StablePlasticFederatedResidualMF",
    "StablePlasticServiceDiagnostic",
]

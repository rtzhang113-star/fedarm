                                                                           

                                                                             
                                                                            
                                                                           
                                                                            
                                                                         
                                                                

                                                                            
                                                                   
                                                                              
                                                                             
                        
   

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .drift_diagnostic import FROZEN_RESIDUAL_CLIP, raw_values_to_logit
from .drift_residual_recenter import (
    SupportResidualRecenterDiagnostic,
    SupportResidualRecenterFedOBRMF,
    SupportResidualRecenterHierSPFedOBRMF,
    _SupportResidualRecenterLocalModel,
)
from .pcs_srr_combo_fedobr import ProposalCountScaledSupportResidualRecenterFedOBRMF
from .residual_model import CausalResidualBiasMF


                                                                           
                                                                         
FROZEN_CARW_WINDOW = 16
FROZEN_CARW_MIN_OBSERVATIONS = 8
FROZEN_CARW_MIN_SEGMENT = 4
FROZEN_CARW_DELTA = 0.05
FROZEN_CARW_RESIDUAL_RANGE = 3.0


def _validated_rms_values(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    name: str,
    allow_window_overflow: bool = False,
) -> tuple[float, ...]:
                                                                             

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(
        array > FROZEN_CARW_RESIDUAL_RANGE
    ):
        raise ValueError(
            f"{name} must contain finite RMS values in [0, {FROZEN_CARW_RESIDUAL_RANGE}]"
        )
    if not allow_window_overflow and array.size > FROZEN_CARW_WINDOW:
        raise ValueError(f"{name} exceeds the {FROZEN_CARW_WINDOW}-slice window")
    return tuple(float(value) for value in array)


def carw_hoeffding_epsilon(n0: int, n1: int) -> float:
                                                        

                                                                             
                                                                         
       

    n0 = int(n0)
    n1 = int(n1)
    if n0 <= 0 or n1 <= 0:
        raise ValueError("both Hoeffding segments must be non-empty")
    epsilon = FROZEN_CARW_RESIDUAL_RANGE * np.sqrt(
        0.5 * np.log(4.0 / FROZEN_CARW_DELTA) * (1.0 / n0 + 1.0 / n1)
    )
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise FloatingPointError("non-finite Hoeffding threshold")
    return float(epsilon)


@dataclass(frozen=True)
class AdaptiveResidualWindowDecision:
                                                               

    current_rms: float
    history_before: tuple[float, ...]
    combined_count: int
    gain: float
    change_detected: bool
    change_cut: int | None
    max_excess: float
    max_abs_diff: float
    epsilon_for_max_diff: float
    retained_history_before_current: tuple[float, ...]

    @property
    def history_count(self) -> int:
        return int(len(self.history_before))

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_rms": float(self.current_rms),
            "history_before": [float(value) for value in self.history_before],
            "history_count_before": int(self.history_count),
            "combined_count": int(self.combined_count),
            "gain": float(self.gain),
            "change_detected": bool(self.change_detected),
            "change_cut": None if self.change_cut is None else int(self.change_cut),
            "max_excess": float(self.max_excess),
            "max_abs_diff": float(self.max_abs_diff),
            "epsilon_for_max_diff": float(self.epsilon_for_max_diff),
            "retained_history_before_current": [
                float(value) for value in self.retained_history_before_current
            ],
            "history_decision_order": "strict_past_window_then_current_decision_then_append",
            "query_or_future_used": False,
        }


def adaptive_residual_window_decision(
    past_rms: np.ndarray | list[float] | tuple[float, ...],
    current_rms: float,
) -> AdaptiveResidualWindowDecision:
                                                                          

                                                                              
                                                                             
                                                                        
                                                                             
                                                                              
                                                       
       

    history = _validated_rms_values(past_rms, name="past_rms")
    current_array = np.asarray([current_rms], dtype=np.float64)
    current = _validated_rms_values(current_array, name="current_rms")[0]
    combined = np.asarray(history + (current,), dtype=np.float64)
    combined_count = int(combined.size)

    max_excess = 0.0
    max_abs_diff = 0.0
    epsilon_for_max_diff = 0.0
    detected_cuts: list[int] = []

    if combined_count >= FROZEN_CARW_MIN_OBSERVATIONS:
        first_cut = FROZEN_CARW_MIN_SEGMENT
        last_cut = combined_count - FROZEN_CARW_MIN_SEGMENT
        for cut in range(first_cut, last_cut + 1):
            left = combined[:cut]
            right = combined[cut:]
            n0, n1 = int(left.size), int(right.size)
            epsilon = carw_hoeffding_epsilon(n0, n1)
            difference = float(abs(np.mean(left, dtype=np.float64) - np.mean(right, dtype=np.float64)))
            excess = float(max(0.0, difference - epsilon))
            if difference > max_abs_diff:
                max_abs_diff = difference
                epsilon_for_max_diff = epsilon
            max_excess = max(max_excess, excess)
            if difference > epsilon:
                detected_cuts.append(int(cut))

    change_detected = bool(detected_cuts)
                                                                             
                                                                           
                                                                          
                        
    change_cut = int(max(detected_cuts)) if change_detected else None
    if change_cut is None:
        retained_history = history
    else:
        retained_history = tuple(
            float(value) for value in combined[change_cut:-1]
        )

    history_count = len(history)
    continuity_floor = float(1.0 / np.sqrt(1.0 + history_count))
    if history_count == 0:
        gain = 1.0
    elif change_detected:
        gain = 1.0
    else:
        denominator = max_abs_diff + epsilon_for_max_diff
        evidence_ratio = 0.0 if denominator <= 0.0 else min(1.0, max_excess / denominator)
        gain = max(continuity_floor, evidence_ratio)
    if not np.isfinite(gain) or not 0.0 < gain <= 1.0:
        raise FloatingPointError("adaptive residual gain is outside (0, 1]")

    return AdaptiveResidualWindowDecision(
        current_rms=float(current),
        history_before=history,
        combined_count=combined_count,
        gain=float(gain),
        change_detected=change_detected,
        change_cut=change_cut,
        max_excess=float(max_excess),
        max_abs_diff=float(max_abs_diff),
        epsilon_for_max_diff=float(epsilon_for_max_diff),
        retained_history_before_current=retained_history,
    )


@dataclass(frozen=True)
class AdaptiveResidualWindowDiagnostic:
                                                                          

    client_id: int
    support_entries: int
    touched_services: int
    current_client_rms: float
    history_count_before: int
    combined_count: int
    gain: float
    change_detected: bool
    change_cut: int | None
    max_excess: float
    max_abs_diff: float
    epsilon_for_max_diff: float
    retained_history_count_before_current: int
    correction_l2: float
    correction_max_abs: float
    history_before: tuple[float, ...]
    retained_history_before_current: tuple[float, ...]

    @property
    def gated_entries(self) -> int:
                                                                          

        return int(self.support_entries if self.gain > 0.0 else 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_id": int(self.client_id),
            "support_entries": int(self.support_entries),
            "touched_services": int(self.touched_services),
            "current_client_rms": float(self.current_client_rms),
            "history_count_before": int(self.history_count_before),
            "combined_count": int(self.combined_count),
            "gain": float(self.gain),
            "change_detected": bool(self.change_detected),
            "change_cut": None if self.change_cut is None else int(self.change_cut),
            "max_excess": float(self.max_excess),
            "max_abs_diff": float(self.max_abs_diff),
            "epsilon_for_max_diff": float(self.epsilon_for_max_diff),
            "retained_history_count_before_current": int(
                self.retained_history_count_before_current
            ),
            "correction_l2": float(self.correction_l2),
            "correction_max_abs": float(self.correction_max_abs),
            "history_before": [float(value) for value in self.history_before],
            "retained_history_before_current": [
                float(value) for value in self.retained_history_before_current
            ],
            "history_decision_order": "decide_before_current_history_append",
            "residual_range": [0.0, float(FROZEN_CARW_RESIDUAL_RANGE)],
            "query_or_future_used": False,
            "communication_added_bytes": 0,
        }


@dataclass(frozen=True)
class _AdaptiveResidualObservation:
    client_id: int
    current_rms: float
    retained_history_before_current: tuple[float, ...]


class _AdaptiveResidualWindowLocalModel(_SupportResidualRecenterLocalModel):
                                                                   

    _arw_client_history: tuple[float, ...]
    _arw_sink: Any
                                                                            
                                                                            
                                                    
    _arw_residual_offset: np.ndarray | None = None

    def fit_support(
        self,
        users: np.ndarray,
        services: np.ndarray,
        raw_values: np.ndarray,
        raw_cap: float,
        bias_epochs: int = 1,
        factor_epochs: int = 1,
        shuffle_seed: int = 42,
    ) -> float:
                                                                              
                                                             
        loss = CausalResidualBiasMF.fit_support(
            self,
            users,
            services,
            raw_values,
            raw_cap,
            bias_epochs,
            factor_epochs,
            shuffle_seed,
        )
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        raw_values = np.asarray(raw_values)
        target_logits = raw_values_to_logit(raw_values, float(raw_cap))
        model_logits = self.user_bias[users] + self.service_bias[services]
        model_logits = model_logits + np.sum(
            self.user_factors[users] * self.service_factors[services], axis=1
        )
        residual_offset = getattr(self, "_arw_residual_offset", None)
        if residual_offset is not None:
            residual_offset = np.asarray(residual_offset, dtype=np.float64)
            if residual_offset.shape != (self.num_services,):
                raise ValueError("adaptive residual offset must match service dimensions")
            if not np.all(np.isfinite(residual_offset)):
                raise FloatingPointError("non-finite adaptive residual offset")
            model_logits = model_logits + residual_offset[services]
        clipped = np.clip(
            target_logits - model_logits,
            -FROZEN_RESIDUAL_CLIP,
            FROZEN_RESIDUAL_CLIP,
        ).astype(np.float64, copy=False)
        if clipped.size == 0:
            raise ValueError("adaptive residual window requires non-empty support")
        if not np.all(np.isfinite(clipped)) or np.any(
            np.abs(clipped) > FROZEN_CARW_RESIDUAL_RANGE
        ):
            raise FloatingPointError("non-finite or out-of-range clipped residual")

        current_rms = float(np.sqrt(np.mean(np.square(clipped), dtype=np.float64)))
        decision = adaptive_residual_window_decision(
            self._arw_client_history,
            current_rms,
        )
        gain = float(decision.gain)

        counts = np.bincount(services, minlength=self.num_services)
        sums = np.bincount(
            services,
            weights=clipped,
            minlength=self.num_services,
        )
        touched = counts > 0
        correction = np.zeros(self.num_services, dtype=np.float64)
        correction[touched] = gain * sums[touched] / counts[touched]
        self.service_bias[touched] += correction[touched]
        if not np.all(np.isfinite(self.service_bias)):
            raise FloatingPointError("non-finite service bias after adaptive residual gating")

        client_id = int(users[0]) if users.size else -1
        diagnostic = AdaptiveResidualWindowDiagnostic(
            client_id=client_id,
            support_entries=int(users.size),
            touched_services=int(np.count_nonzero(touched)),
            current_client_rms=current_rms,
            history_count_before=int(len(self._arw_client_history)),
            combined_count=int(decision.combined_count),
            gain=gain,
            change_detected=bool(decision.change_detected),
            change_cut=decision.change_cut,
            max_excess=float(decision.max_excess),
            max_abs_diff=float(decision.max_abs_diff),
            epsilon_for_max_diff=float(decision.epsilon_for_max_diff),
            retained_history_count_before_current=int(
                len(decision.retained_history_before_current)
            ),
            correction_l2=float(np.linalg.norm(correction[touched])),
            correction_max_abs=(
                float(np.max(np.abs(correction[touched]))) if np.any(touched) else 0.0
            ),
            history_before=tuple(self._arw_client_history),
            retained_history_before_current=tuple(
                decision.retained_history_before_current
            ),
        )
        sink = self._arw_sink
        sink._pending_adaptive_observations.append(
            _AdaptiveResidualObservation(
                client_id=client_id,
                current_rms=current_rms,
                retained_history_before_current=tuple(
                    decision.retained_history_before_current
                ),
            )
        )
        sink._pending_adaptive_diagnostics.append(diagnostic)

                                                                              
                                                                            
        recenter_diagnostic = SupportResidualRecenterDiagnostic(
            client_id=client_id,
            touched_services=int(np.count_nonzero(touched)),
            support_entries=int(users.size),
            residual_rms_before_recenter=current_rms,
            residual_mean_abs=float(np.mean(np.abs(clipped))),
            correction_l2=float(np.linalg.norm(correction[touched])),
            correction_max_abs=(
                float(np.max(np.abs(correction[touched]))) if np.any(touched) else 0.0
            ),
            gain=gain,
        )
        sink._pending_recenter_diagnostics.append(recenter_diagnostic)
        self._last_recenter_diagnostic = recenter_diagnostic
        return loss


class AdaptiveResidualWindowMixin:
                                                                         

    def __init__(
        self,
        *args: object,
        residual_clip: tuple[float, float] | list[float] = (
            -FROZEN_CARW_RESIDUAL_RANGE,
            FROZEN_CARW_RESIDUAL_RANGE,
        ),
        window_min_observations: int = FROZEN_CARW_MIN_OBSERVATIONS,
        window_min_side_observations: int = FROZEN_CARW_MIN_SEGMENT,
        window_confidence: float = FROZEN_CARW_DELTA,
        window_bound_multiplier: float = FROZEN_CARW_RESIDUAL_RANGE,
        **kwargs: object,
    ) -> None:
                                                                             
                                                                       
                                                                             
                                                        
        clip = tuple(float(value) for value in residual_clip)
        if clip != (-FROZEN_CARW_RESIDUAL_RANGE, FROZEN_CARW_RESIDUAL_RANGE):
            raise ValueError("residual_clip is frozen to [-3, 3]")
        if int(window_min_observations) != FROZEN_CARW_MIN_OBSERVATIONS:
            raise ValueError(
                f"window_min_observations is frozen to {FROZEN_CARW_MIN_OBSERVATIONS}"
            )
        if int(window_min_side_observations) != FROZEN_CARW_MIN_SEGMENT:
            raise ValueError(
                f"window_min_side_observations is frozen to {FROZEN_CARW_MIN_SEGMENT}"
            )
        if not np.isfinite(float(window_confidence)) or float(window_confidence) != FROZEN_CARW_DELTA:
            raise ValueError(f"window_confidence is frozen to {FROZEN_CARW_DELTA}")
        if not np.isfinite(float(window_bound_multiplier)) or float(window_bound_multiplier) != FROZEN_CARW_RESIDUAL_RANGE:
            raise ValueError(
                f"window_bound_multiplier is frozen to {FROZEN_CARW_RESIDUAL_RANGE}"
            )
        super().__init__(*args, **kwargs)
        self._adaptive_residual_history: dict[int, list[float]] = {}
        self._pending_adaptive_observations: list[_AdaptiveResidualObservation] = []
        self._pending_adaptive_diagnostics: list[AdaptiveResidualWindowDiagnostic] = []
        self._last_adaptive_diagnostics: tuple[AdaptiveResidualWindowDiagnostic, ...] = ()

    def _local_model_from_snapshot(
        self,
        user: int,
        shared_snapshot: dict[str, np.ndarray],
    ) -> CausalResidualBiasMF:
        local = super()._local_model_from_snapshot(user, shared_snapshot)
        local.__class__ = _AdaptiveResidualWindowLocalModel
        local._arw_client_history = tuple(
            float(value)
            for value in self._adaptive_residual_history.get(int(user), ())
        )
        local._arw_sink = self
        return local

    def advance_support(self, *args: object, **kwargs: object):
                                                                                

        self._pending_adaptive_observations = []
        self._pending_adaptive_diagnostics = []
        before_parameters = self.parameter_snapshot()
        before_history = {
            int(client): [float(value) for value in values]
            for client, values in self._adaptive_residual_history.items()
        }
        try:
            result = super().advance_support(*args, **kwargs)
                                                                              
                                                                            
            diagnostics = tuple(self._pending_adaptive_diagnostics)
            self._commit_adaptive_observations()
            self._last_adaptive_diagnostics = diagnostics
            return result
        except Exception:
                                                                               
                                                                          
                                                        
            self._restore_snapshot(before_parameters)
            self._adaptive_residual_history = before_history
            self._pending_adaptive_observations = []
            self._pending_adaptive_diagnostics = []
            self._last_adaptive_diagnostics = ()
            if hasattr(self, "_last_recenter_diagnostics"):
                self._last_recenter_diagnostics = ()
            raise

    def _commit_adaptive_observations(self) -> None:
        observations = sorted(
            self._pending_adaptive_observations,
            key=lambda item: int(item.client_id),
        )
        new_history = {
            int(client): [float(value) for value in values]
            for client, values in self._adaptive_residual_history.items()
        }
        seen_clients: set[int] = set()
        for item in observations:
            client_id = int(item.client_id)
            if client_id in seen_clients:
                raise RuntimeError("duplicate current-slice adaptive history observation")
            seen_clients.add(client_id)
            retained = list(
                _validated_rms_values(
                    item.retained_history_before_current,
                    name="retained_history_before_current",
                )
            )
            current = _validated_rms_values(
                [item.current_rms],
                name="current_rms",
            )[0]
            retained.append(float(current))
            if len(retained) > FROZEN_CARW_WINDOW:
                retained = retained[-FROZEN_CARW_WINDOW:]
            if not retained or len(retained) > FROZEN_CARW_WINDOW:
                raise FloatingPointError("invalid adaptive residual history commit")
            new_history[client_id] = retained
        self._adaptive_residual_history = new_history
        self._pending_adaptive_observations = []
        self._pending_adaptive_diagnostics = []

    @property
    def last_adaptive_residual_diagnostics(
        self,
    ) -> tuple[AdaptiveResidualWindowDiagnostic, ...]:
        return self._last_adaptive_diagnostics

    @property
    def last_carw_diagnostics(
        self,
    ) -> tuple[AdaptiveResidualWindowDiagnostic, ...]:
                                                          

        return self._last_adaptive_diagnostics

    def adaptive_residual_history_snapshot(self) -> dict[int, tuple[float, ...]]:
                                                                         

        return {
            int(client): tuple(float(value) for value in values)
            for client, values in sorted(self._adaptive_residual_history.items())
        }

    def carw_history_snapshot(self) -> dict[int, tuple[float, ...]]:
        return self.adaptive_residual_history_snapshot()

    def adaptive_residual_history_counts(self) -> dict[str, int]:
        snapshot = self.adaptive_residual_history_snapshot()
        return {
            "client_entities": int(len(snapshot)),
            "client_observations": int(sum(len(values) for values in snapshot.values())),
            "max_window_length": int(max((len(values) for values in snapshot.values()), default=0)),
        }


class AdaptiveResidualWindowFedOBRMF(
    AdaptiveResidualWindowMixin,
    SupportResidualRecenterFedOBRMF,
):
                                                                       

    model_name = "FedOBR-CARW-MF"
    aggregation_algorithm = "FedAvg with causal adaptive residual-window recentering"


class AdaptiveResidualWindowHierSPFedOBRMF(
    AdaptiveResidualWindowMixin,
    SupportResidualRecenterHierSPFedOBRMF,
):
                                                                       

    model_name = "Hier-SP-FedOBR-CARW-MF"
    aggregation_algorithm = (
        "hierarchical FedAvg plus stable-plastic projection and causal adaptive residual-window recentering"
    )


class CausalAdaptiveResidualWindowFedOBRMF(
    AdaptiveResidualWindowMixin,
    ProposalCountScaledSupportResidualRecenterFedOBRMF,
):
                                                                  

                                                                            
                                                                     
                                                                            
                   
       

    model_name = "PCS-CARW-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "proposal-count-scaled service-factor aggregation with causal adaptive residual-window SRR"
    )


                                                                 
CausalAdaptiveResidualWindowSRRFedOBRMF = CausalAdaptiveResidualWindowFedOBRMF
CARWSRRFedOBRMF = CausalAdaptiveResidualWindowFedOBRMF


                                                                         
CARWFedOBRMF = CausalAdaptiveResidualWindowFedOBRMF
ARWFedOBRMF = CausalAdaptiveResidualWindowFedOBRMF


__all__ = [
    "ARWFedOBRMF",
    "AdaptiveResidualWindowDecision",
    "AdaptiveResidualWindowDiagnostic",
    "AdaptiveResidualWindowFedOBRMF",
    "AdaptiveResidualWindowHierSPFedOBRMF",
    "AdaptiveResidualWindowMixin",
    "CARWFedOBRMF",
    "CARWSRRFedOBRMF",
    "CausalAdaptiveResidualWindowFedOBRMF",
    "CausalAdaptiveResidualWindowSRRFedOBRMF",
    "FROZEN_CARW_DELTA",
    "FROZEN_CARW_MIN_OBSERVATIONS",
    "FROZEN_CARW_MIN_SEGMENT",
    "FROZEN_CARW_RESIDUAL_RANGE",
    "FROZEN_CARW_WINDOW",
    "adaptive_residual_window_decision",
    "carw_hoeffding_epsilon",
]

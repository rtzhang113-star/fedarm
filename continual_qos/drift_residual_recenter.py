                                                                       

                                                                           
                                                                        
                                                                           
                                                                       

                                                                           
                                                                        
                                                                    
                                                                           
                                                                        
                                                                            
                         
   

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .drift_diagnostic import FROZEN_RESIDUAL_CLIP, raw_values_to_logit
from .federated_residual import FederatedResidualBiasMF
from .fedobr_mf import FedOBRMF
from .hierarchical_stable_plastic import HierarchicalStablePlasticFedOBRMF
from .residual_model import CausalResidualBiasMF


FROZEN_RECENTER_GAIN = 1.0


@dataclass(frozen=True)
class SupportResidualRecenterDiagnostic:
                                                          

    client_id: int
    touched_services: int
    support_entries: int
    residual_rms_before_recenter: float
    residual_mean_abs: float
    correction_l2: float
    correction_max_abs: float
    gain: float = FROZEN_RECENTER_GAIN

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_id": int(self.client_id),
            "touched_services": int(self.touched_services),
            "support_entries": int(self.support_entries),
            "residual_rms_before_recenter": float(self.residual_rms_before_recenter),
            "residual_mean_abs": float(self.residual_mean_abs),
            "correction_l2": float(self.correction_l2),
            "correction_max_abs": float(self.correction_max_abs),
            "gain": float(self.gain),
        }


class _SupportResidualRecenterLocalModel(CausalResidualBiasMF):
                                                                           

    recenter_gain: float

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
        loss = super().fit_support(
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
        clipped = np.clip(
            target_logits - model_logits,
            -FROZEN_RESIDUAL_CLIP,
            FROZEN_RESIDUAL_CLIP,
        )
        counts = np.bincount(services, minlength=self.num_services)
        sums = np.bincount(services, weights=clipped, minlength=self.num_services)
        touched = counts > 0
        correction = np.zeros(self.num_services, dtype=np.float64)
        correction[touched] = float(self.recenter_gain) * sums[touched] / counts[touched]
        self.service_bias[touched] += correction[touched]
        if not np.all(np.isfinite(self.service_bias)):
            raise FloatingPointError("non-finite service bias after support recentering")
        diagnostic = SupportResidualRecenterDiagnostic(
            client_id=int(users[0]) if users.size else -1,
            touched_services=int(np.count_nonzero(touched)),
            support_entries=int(users.size),
            residual_rms_before_recenter=float(np.sqrt(np.mean(np.square(clipped)))),
            residual_mean_abs=float(np.mean(np.abs(clipped))),
            correction_l2=float(np.linalg.norm(correction[touched])),
            correction_max_abs=float(np.max(np.abs(correction[touched]))) if np.any(touched) else 0.0,
        )
        sink = getattr(self, "_recenter_sink", None)
        if sink is not None:
            sink._pending_recenter_diagnostics.append(diagnostic)
        self._last_recenter_diagnostic = diagnostic
        return loss


class SupportResidualRecenterMixin:
                                                                       

                                                                              
                                                                        
                                                                         
       

    recenter_gain: float

    def __init__(
        self,
        *args: object,
        recenter_gain: float = FROZEN_RECENTER_GAIN,
        **kwargs: object,
    ) -> None:
        gain = float(recenter_gain)
        if not np.isfinite(gain) or gain != FROZEN_RECENTER_GAIN:
            raise ValueError(f"recenter_gain is frozen to {FROZEN_RECENTER_GAIN}")
        super().__init__(*args, **kwargs)
        self.recenter_gain = gain
        self._last_recenter_diagnostics: tuple[SupportResidualRecenterDiagnostic, ...] = ()
        self._pending_recenter_diagnostics: list[SupportResidualRecenterDiagnostic] = []

    def _local_model_from_snapshot(
        self,
        user: int,
        shared_snapshot: dict[str, np.ndarray],
    ) -> CausalResidualBiasMF:
                                                                            
                                                                              
                                                                             
        local = super()._local_model_from_snapshot(user, shared_snapshot)
        local.__class__ = _SupportResidualRecenterLocalModel
        local.recenter_gain = float(self.recenter_gain)
        local._last_recenter_diagnostic = None
        local._recenter_sink = self
        return local

    def advance_support(self, *args: object, **kwargs: object):
        self._pending_recenter_diagnostics = []
        try:
            result = super().advance_support(*args, **kwargs)
            for proposal in result.proposals:
                                                                           
                                                                       
                if not np.all(np.isfinite(proposal.service_factors)):
                    raise FloatingPointError("non-finite corrected service proposal")
            self._last_recenter_diagnostics = tuple(self._pending_recenter_diagnostics)
            return result
        except Exception:
            self._pending_recenter_diagnostics = []
            self._last_recenter_diagnostics = ()
            raise

    @property
    def last_recenter_diagnostics(self) -> tuple[SupportResidualRecenterDiagnostic, ...]:
        return self._last_recenter_diagnostics


class SupportResidualRecenterFedOBRMF(SupportResidualRecenterMixin, FedOBRMF):
                                                                   

    model_name = "FedOBR-SRR-MF"
    aggregation_algorithm = "FedAvg with support residual recentering"


class SupportResidualRecenterHierSPFedOBRMF(
    SupportResidualRecenterMixin,
    HierarchicalStablePlasticFedOBRMF,
):
                                                                   

    model_name = "Hier-SP-FedOBR-SRR-MF"
    aggregation_algorithm = "hierarchical FedAvg plus stable-plastic projection and support residual recentering"


__all__ = [
    "FROZEN_RECENTER_GAIN",
    "SupportResidualRecenterDiagnostic",
    "SupportResidualRecenterMixin",
    "SupportResidualRecenterFedOBRMF",
    "SupportResidualRecenterHierSPFedOBRMF",
]

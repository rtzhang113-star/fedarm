                                                                 

                                   

                                                                 

                                                                            
                                                                         
                                                                    
                                                   

                                                                             
                                                                             
                                                                          
                                                                           
                                                                            
                                                                            
   

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .federated_residual import ServiceProposal
from .hierarchical_fedobr import HierarchicalFedOBRMF


DEFAULT_NUM_EDGES = 6
DEFAULT_MIN_PROPOSALS = 2
DEFAULT_MIN_NORM = 1e-12
DEFAULT_COUNT_EXPONENT = 1.0
DEFAULT_TRUST_EXPONENT = 0.5
DEFAULT_EQUIVALENCE_ATOL = 1e-12


@dataclass(frozen=True)
class ProposalCountScaleDiagnostic:
    service_id: int
    proposal_count: int
    participating_edges: tuple[int, ...]
    edge_proposal_counts: tuple[int, ...]
    count_exponent: float
    raw_scale: float
    raw_delta_norm: float
    median_proposal_delta_norm: float
    trust_region_norm: float
    applied_scale: float
    committed_delta_norm: float
    fallback: bool
    fallback_reason: str | None
    cloud_max_abs_difference_from_direct: float

    def as_dict(self) -> dict[str, object]:
        return {
            "service_id": int(self.service_id),
            "proposal_count": int(self.proposal_count),
            "participating_edges": [int(value) for value in self.participating_edges],
            "edge_proposal_counts": [int(value) for value in self.edge_proposal_counts],
            "count_exponent": float(self.count_exponent),
            "raw_scale": float(self.raw_scale),
            "raw_delta_norm": float(self.raw_delta_norm),
            "median_proposal_delta_norm": float(self.median_proposal_delta_norm),
            "trust_region_norm": float(self.trust_region_norm),
            "applied_scale": float(self.applied_scale),
            "committed_delta_norm": float(self.committed_delta_norm),
            "fallback": bool(self.fallback),
            "fallback_reason": self.fallback_reason,
            "cloud_max_abs_difference_from_direct": float(
                self.cloud_max_abs_difference_from_direct
            ),
            "query_labels_used": False,
            "query_predictions_used": False,
            "future_labels_used": False,
            "additional_communication_bytes": 0,
        }


class ProposalCountScaledFedOBRMF(HierarchicalFedOBRMF):
                                                                        

    model_name = "PCS-FedOBR-MF"
    aggregation_algorithm = "proposal-count-scaled service-factor aggregation"
    partition_semantics = "synthetic_user_id_modulo"

    def __init__(
        self,
        *args: object,
        num_edges: int = DEFAULT_NUM_EDGES,
        equivalence_atol: float = DEFAULT_EQUIVALENCE_ATOL,
        min_proposals: int = DEFAULT_MIN_PROPOSALS,
        min_norm: float = DEFAULT_MIN_NORM,
        count_exponent: float = DEFAULT_COUNT_EXPONENT,
        trust_exponent: float = DEFAULT_TRUST_EXPONENT,
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            num_edges=num_edges,
            equivalence_atol=equivalence_atol,
            **kwargs,
        )
        if isinstance(min_proposals, bool) or int(min_proposals) != DEFAULT_MIN_PROPOSALS:
            raise ValueError(f"min_proposals is frozen to {DEFAULT_MIN_PROPOSALS}")
        if not np.isfinite(float(min_norm)) or float(min_norm) != DEFAULT_MIN_NORM:
            raise ValueError(f"min_norm is frozen to {DEFAULT_MIN_NORM}")
        if not np.isfinite(float(count_exponent)) or float(count_exponent) != DEFAULT_COUNT_EXPONENT:
            raise ValueError(f"count_exponent is frozen to {DEFAULT_COUNT_EXPONENT}")
        if not np.isfinite(float(trust_exponent)) or float(trust_exponent) != DEFAULT_TRUST_EXPONENT:
            raise ValueError(f"trust_exponent is frozen to {DEFAULT_TRUST_EXPONENT}")
        self.min_proposals = int(min_proposals)
        self.min_norm = float(min_norm)
        self.count_exponent = float(count_exponent)
        self.trust_exponent = float(trust_exponent)

    def _commit_service_aggregates(
        self,
        by_service: dict[int, list[ServiceProposal]],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[ProposalCountScaleDiagnostic, ...]:
        diagnostics: list[ProposalCountScaleDiagnostic] = []
        pending: list[tuple[int, np.ndarray, float, float]] = []
        for service in sorted(by_service):
            proposals = tuple(sorted(by_service[service], key=lambda item: int(item.client_id)))
            count = len(proposals)
            factors = np.stack([proposal.service_factors for proposal in proposals], axis=0).astype(
                np.float64, copy=False
            )
            biases = np.asarray([proposal.service_bias for proposal in proposals], dtype=np.float64)
            errors = np.asarray([proposal.service_error for proposal in proposals], dtype=np.float64)
            snapshot = np.asarray(shared_snapshot["service_factors"][service], dtype=np.float64)
            deltas = factors - snapshot[None, :]
            norms = np.linalg.norm(deltas, axis=1)
            mean_factor = np.mean(factors, axis=0, dtype=np.float64)
            mean_bias = float(np.mean(biases, dtype=np.float64))
            mean_error = float(np.mean(errors, dtype=np.float64))
            fallback = False
            reason: str | None = None
            raw_scale = float(count ** self.count_exponent)
            median_norm = float(np.median(norms)) if count else 0.0
            trust_norm = float(count ** self.trust_exponent * median_norm)
            raw_delta = raw_scale * np.mean(deltas, axis=0, dtype=np.float64)
            raw_norm = float(np.linalg.norm(raw_delta))
            applied_scale = 1.0
            committed_factor = mean_factor.copy()
            if count < self.min_proposals:
                fallback = True
                reason = "insufficient_proposals"
            elif not (
                np.all(np.isfinite(factors))
                and np.all(np.isfinite(biases))
                and np.all(np.isfinite(errors))
                and np.all(np.isfinite(snapshot))
            ):
                fallback = True
                reason = "nonfinite_proposal_state"
            elif median_norm <= self.min_norm:
                fallback = True
                reason = "zero_proposal_delta_scale"
            elif not np.isfinite(raw_norm) or not np.isfinite(trust_norm):
                fallback = True
                reason = "invalid_scaled_delta"
            else:
                if raw_norm > trust_norm:
                    applied_scale = trust_norm / raw_norm
                committed_factor = snapshot + applied_scale * raw_delta
                if not np.all(np.isfinite(committed_factor)):
                    fallback = True
                    reason = "nonfinite_trust_region_commit"
                    committed_factor = mean_factor.copy()
            if fallback:
                applied_scale = 0.0
                committed_factor = mean_factor.copy()

                                                                             
                                                                            
                                                                               
            grouped: dict[int, list[int]] = {}
            for index, proposal in enumerate(proposals):
                grouped.setdefault(self.edge_id(proposal.client_id), []).append(index)
            edges = tuple(sorted(grouped))
            edge_counts = tuple(len(grouped[edge]) for edge in edges)
            edge_delta_sum = np.sum(
                np.stack(
                    [np.sum(deltas[grouped[edge]], axis=0, dtype=np.float64) for edge in edges],
                    axis=0,
                ),
                axis=0,
                dtype=np.float64,
            )
            edge_raw_delta = raw_scale / float(count) * edge_delta_sum
            edge_raw_norm = float(np.linalg.norm(edge_raw_delta))
            edge_applied_scale = 0.0 if fallback else (
                trust_norm / edge_raw_norm if edge_raw_norm > trust_norm else 1.0
            )
            edge_factor = mean_factor if fallback else snapshot + edge_applied_scale * edge_raw_delta
            direct_difference = float(np.max(np.abs(edge_factor - committed_factor)))
            if direct_difference > self.equivalence_atol:
                raise AssertionError(
                    f"edge/cloud scaled service aggregate differs for service {service}: "
                    f"{direct_difference}"
                )
            pending.append((int(service), committed_factor.copy(), mean_bias, mean_error))
            diagnostics.append(
                ProposalCountScaleDiagnostic(
                    service_id=int(service),
                    proposal_count=count,
                    participating_edges=edges,
                    edge_proposal_counts=edge_counts,
                    count_exponent=self.count_exponent,
                    raw_scale=raw_scale,
                    raw_delta_norm=raw_norm,
                    median_proposal_delta_norm=median_norm,
                    trust_region_norm=trust_norm,
                    applied_scale=applied_scale,
                    committed_delta_norm=float(np.linalg.norm(committed_factor - snapshot)),
                    fallback=bool(fallback),
                    fallback_reason=reason,
                    cloud_max_abs_difference_from_direct=direct_difference,
                )
            )
        for service, factors, bias, error in pending:
            self.service_factors[service] = factors
            self.service_bias[service] = bias
            self.service_error[service] = error
        return tuple(diagnostics)


__all__ = [
    "DEFAULT_COUNT_EXPONENT",
    "DEFAULT_EQUIVALENCE_ATOL",
    "DEFAULT_MIN_NORM",
    "DEFAULT_MIN_PROPOSALS",
    "DEFAULT_NUM_EDGES",
    "DEFAULT_TRUST_EXPONENT",
    "ProposalCountScaleDiagnostic",
    "ProposalCountScaledFedOBRMF",
]

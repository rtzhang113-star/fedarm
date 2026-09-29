                                                               

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fedobr_mf import FedOBRMF
from .federated_residual import ServiceProposal


@dataclass(frozen=True)
class HierarchicalServiceDiagnostic:
    service_id: int
    proposal_count: int
    participating_edges: tuple[int, ...]
    edge_proposal_counts: tuple[int, ...]
    cloud_max_abs_difference_from_flat: float
    cloud_weighted_equivalent: bool
    committed_flat_reference: bool = True

    @property
    def edge_summary_count(self) -> int:
        return len(self.participating_edges)


class HierarchicalFedOBRMF(FedOBRMF):
                                                                   

                                                                              
                                                                             
                                                                             
                                                                           
                                                                  
       

    model_name = "Hier-FedOBR-MF"
    aggregation_algorithm = "Hierarchical FedAvg"
    partition_semantics = "synthetic_user_id_modulo"

    def __init__(self, *args: object, num_edges: int = 6, equivalence_atol: float = 1e-12, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if int(num_edges) <= 1 or int(num_edges) > self.num_users:
            raise ValueError("num_edges must be in [2, num_users]")
        if not np.isfinite(equivalence_atol) or float(equivalence_atol) != 1e-12:
            raise ValueError("equivalence_atol is frozen to 1e-12")
        self.num_edges = int(num_edges)
        self.equivalence_atol = float(equivalence_atol)

    def edge_id(self, client_id: int) -> int:
        client = int(client_id)
        if not 0 <= client < self.num_users:
            raise IndexError("client id outside model dimensions")
        return client % self.num_edges

    def edge_assignment(self) -> np.ndarray:
        return np.arange(self.num_users, dtype=np.int64) % self.num_edges

    @property
    def edge_summary_payload_bytes(self) -> int:
                                                                                
        return int(8 + 8 * (self.rank + 2))

    def _commit_service_aggregates(
        self,
        by_service: dict[int, list[ServiceProposal]],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[HierarchicalServiceDiagnostic, ...]:
        del shared_snapshot
        diagnostics: list[HierarchicalServiceDiagnostic] = []
        pending: list[tuple[int, np.ndarray, float, float]] = []
        for service in sorted(by_service):
            proposals = tuple(sorted(by_service[service], key=lambda item: int(item.client_id)))
            factors = np.stack([item.service_factors for item in proposals]).astype(np.float64, copy=False)
            biases = np.asarray([item.service_bias for item in proposals], dtype=np.float64)
            errors = np.asarray([item.service_error for item in proposals], dtype=np.float64)
            flat_factor = np.mean(factors, axis=0, dtype=np.float64)
            flat_bias = float(np.mean(biases, dtype=np.float64))
            flat_error = float(np.mean(errors, dtype=np.float64))

            grouped: dict[int, list[int]] = {}
            for index, proposal in enumerate(proposals):
                grouped.setdefault(self.edge_id(proposal.client_id), []).append(index)
            edges = tuple(sorted(grouped))
            counts = np.asarray([len(grouped[edge]) for edge in edges], dtype=np.float64)
            edge_factors = np.stack([np.mean(factors[grouped[edge]], axis=0, dtype=np.float64) for edge in edges])
            edge_biases = np.asarray([np.mean(biases[grouped[edge]], dtype=np.float64) for edge in edges])
            edge_errors = np.asarray([np.mean(errors[grouped[edge]], dtype=np.float64) for edge in edges])
            cloud_factor = np.average(edge_factors, axis=0, weights=counts)
            cloud_bias = float(np.average(edge_biases, weights=counts))
            cloud_error = float(np.average(edge_errors, weights=counts))
            difference = float(max(np.max(np.abs(cloud_factor - flat_factor)), abs(cloud_bias - flat_bias), abs(cloud_error - flat_error)))
            equivalent = bool(difference <= self.equivalence_atol)
            if not equivalent:
                raise AssertionError(f"hierarchical FedAvg differs from flat mean for service {service}: {difference}")
            pending.append((int(service), flat_factor.copy(), flat_bias, flat_error))
            diagnostics.append(HierarchicalServiceDiagnostic(
                service_id=int(service), proposal_count=len(proposals), participating_edges=edges,
                edge_proposal_counts=tuple(int(value) for value in counts),
                cloud_max_abs_difference_from_flat=difference, cloud_weighted_equivalent=equivalent,
            ))
        for service, factors, bias, error in pending:
            self.service_factors[service] = factors
            self.service_bias[service] = bias
            self.service_error[service] = error
        return tuple(diagnostics)


__all__ = ["HierarchicalFedOBRMF", "HierarchicalServiceDiagnostic"]

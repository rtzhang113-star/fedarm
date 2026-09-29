                                                                      

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .federated_residual import ServiceProposal
from .stable_plastic_federated import (
    StablePlasticFederatedResidualBiasMF,
    StablePlasticServiceDiagnostic,
)


@dataclass(frozen=True)
class HierarchicalStablePlasticDiagnostic:
    service_id: int
    participating_edges: tuple[int, ...]
    edge_proposal_counts: tuple[int, ...]
    cloud_max_abs_difference_from_flat: float
    cloud_weighted_equivalent: bool
    stable_plastic: StablePlasticServiceDiagnostic

    @property
    def edge_summary_count(self) -> int:
        return len(self.participating_edges)


class HierarchicalStablePlasticFedOBRMF(StablePlasticFederatedResidualBiasMF):
                                                                            

                                                                        
                                                                               
                                                                            
                                                                          
       

    model_name = "Hier-SP-FedOBR-MF"
    aggregation_algorithm = "hierarchical FedAvg plus cloud stable-plastic projection"
    partition_semantics = "synthetic_user_id_modulo"

    def __init__(self, *args: object, num_edges: int = 6, equivalence_atol: float = 1e-12, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if int(num_edges) <= 1 or int(num_edges) > self.num_users:
            raise ValueError("num_edges must be in [2, num_users]")
        if float(equivalence_atol) != 1e-12:
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

    def _commit_service_aggregates(
        self,
        by_service: dict[int, list[ServiceProposal]],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[HierarchicalStablePlasticDiagnostic, ...]:
        hierarchy: dict[int, tuple[tuple[int, ...], tuple[int, ...], float]] = {}
        for service in sorted(by_service):
            proposals = tuple(sorted(by_service[service], key=lambda item: int(item.client_id)))
            factors = np.stack([item.service_factors for item in proposals]).astype(np.float64, copy=False)
            biases = np.asarray([item.service_bias for item in proposals], dtype=np.float64)
            errors = np.asarray([item.service_error for item in proposals], dtype=np.float64)
            flat = (np.mean(factors, axis=0, dtype=np.float64), float(np.mean(biases)), float(np.mean(errors)))
            grouped: dict[int, list[int]] = {}
            for index, proposal in enumerate(proposals):
                grouped.setdefault(self.edge_id(proposal.client_id), []).append(index)
            edges = tuple(sorted(grouped)); counts = np.asarray([len(grouped[edge]) for edge in edges], dtype=np.float64)
            edge_factors = np.stack([np.mean(factors[grouped[edge]], axis=0, dtype=np.float64) for edge in edges])
            edge_biases = np.asarray([np.mean(biases[grouped[edge]], dtype=np.float64) for edge in edges])
            edge_errors = np.asarray([np.mean(errors[grouped[edge]], dtype=np.float64) for edge in edges])
            cloud = (np.average(edge_factors, axis=0, weights=counts), float(np.average(edge_biases, weights=counts)), float(np.average(edge_errors, weights=counts)))
            difference = float(max(np.max(np.abs(cloud[0]-flat[0])), abs(cloud[1]-flat[1]), abs(cloud[2]-flat[2])))
            if difference > self.equivalence_atol:
                raise AssertionError(f"edge/cloud mean mismatch for service {service}")
            hierarchy[int(service)] = (edges, tuple(int(value) for value in counts), difference)
        stable = super()._commit_service_aggregates(by_service, shared_snapshot)
        return tuple(HierarchicalStablePlasticDiagnostic(
            service_id=item.service_id,
            participating_edges=hierarchy[item.service_id][0],
            edge_proposal_counts=hierarchy[item.service_id][1],
            cloud_max_abs_difference_from_flat=hierarchy[item.service_id][2],
            cloud_weighted_equivalent=True,
            stable_plastic=item,
        ) for item in stable)


__all__ = ["HierarchicalStablePlasticFedOBRMF", "HierarchicalStablePlasticDiagnostic"]

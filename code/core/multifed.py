                                                  

                                                                        
                                                                           
                                                                         
                                                                         
        

                                         
                                                                           
                                                                
                                                                              
        

                                                                  
                                                                          
                                                                         
                                            

                                                                          
                                                                              
                                                                            
                                                                             
                                   
   

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .federated_residual import (
    FederatedResidualBiasMF,
    FederatedRoundResult,
    ServiceProposal,
)
from .residual_model import CausalResidualBiasMF


@dataclass(frozen=True)
class MultiFedCommunicationReport:
                                                              

                                                                     
                                                                      
                                                                           
                                              
       

    active_clients: tuple[int, ...]
    active_edges: tuple[int, ...]
    support_counts: tuple[tuple[int, int], ...]
    proposal_count: int
    internal_aggregation_rounds: int
    client_edge_uplink_bytes: int
    edge_cloud_uplink_bytes: int
    cloud_edge_downlink_bytes: int
    edge_client_downlink_bytes: int

    @property
    def uplink_bytes(self) -> int:
        return int(self.client_edge_uplink_bytes + self.edge_cloud_uplink_bytes)

    @property
    def downlink_bytes(self) -> int:
        return int(self.cloud_edge_downlink_bytes + self.edge_client_downlink_bytes)

    @property
    def total_bytes(self) -> int:
        return int(self.uplink_bytes + self.downlink_bytes)

    @property
    def support_count_bytes(self) -> int:
        return int(4 * len(self.active_clients) * self.internal_aggregation_rounds)

    @property
    def proposal_payload_bytes(self) -> int:
        return int(self.client_edge_uplink_bytes - self.support_count_bytes)

    def as_dict(self) -> dict[str, object]:
        return {
            "active_clients": list(self.active_clients),
            "active_edges": list(self.active_edges),
            "support_counts": [list(item) for item in self.support_counts],
            "proposal_count": int(self.proposal_count),
            "internal_aggregation_rounds": int(self.internal_aggregation_rounds),
            "client_edge_uplink_bytes": int(self.client_edge_uplink_bytes),
            "edge_cloud_uplink_bytes": int(self.edge_cloud_uplink_bytes),
            "cloud_edge_downlink_bytes": int(self.cloud_edge_downlink_bytes),
            "edge_client_downlink_bytes": int(self.edge_client_downlink_bytes),
            "uplink_bytes": int(self.uplink_bytes),
            "downlink_bytes": int(self.downlink_bytes),
            "total_bytes": int(self.total_bytes),
        }


@dataclass(frozen=True)
class MultiFedServiceDiagnostic:
                                                                       

    service_id: int
    participating_edges: tuple[int, ...]
    edge_proposal_counts: tuple[int, ...]
    internal_rounds: int
    external_weight: float
    cloud_max_abs_difference_before_mix: float
    cloud_weighted_equivalent_before_mix: bool
    edge_max_abs_difference_after_mix: float

    @property
    def edge_summary_count(self) -> int:
        return len(self.participating_edges)


class MultiFedFedOBRMF(FederatedResidualBiasMF):
                                                            

                                                                              
                                                                          
                                                                               
                                                                            
                                                                             
                                                                             
                          

                                                                            
                                                                             
                                                                           
                           
       

    model_name = "MultiFed-FedOBR-MF"
    backbone_name = "OBR-MF"
    aggregation_algorithm = "MultiFed internal/external aggregation"
    partition_semantics = "synthetic_user_id_modulo"
    protocol = "causal_cloud_edge_multifed_v1"

    def __init__(
        self,
        *args: object,
        num_edges: int = 6,
        external_weight: float = 0.5,
        internal_aggregations: int = 1,
        equivalence_atol: float = 1e-12,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(num_edges) <= 1 or int(num_edges) > self.num_users:
            raise ValueError("num_edges must be in [2, num_users]")
        if not np.isfinite(float(external_weight)) or not 0.0 <= float(external_weight) <= 1.0:
            raise ValueError("external_weight must be finite and lie in [0, 1]")
        if isinstance(internal_aggregations, bool) or int(internal_aggregations) <= 0:
            raise ValueError("internal_aggregations must be a positive integer")
        if not np.isfinite(float(equivalence_atol)) or float(equivalence_atol) != 1e-12:
            raise ValueError("equivalence_atol is frozen to 1e-12")

        self.num_edges = int(num_edges)
        self.external_weight = float(external_weight)
        self.internal_aggregations = int(internal_aggregations)
        self.equivalence_atol = float(equivalence_atol)

                                                                             
                                                                               
                                                           
        self.edge_service_factors = np.repeat(
            self.service_factors[None, :, :], self.num_edges, axis=0
        ).copy()
        self.edge_service_bias = np.repeat(
            self.service_bias[None, :], self.num_edges, axis=0
        ).copy()
        self.edge_service_error = np.repeat(
            self.service_error[None, :], self.num_edges, axis=0
        ).copy()
        self._require_finite_state()

                                                                        
                                           
                                                                        
    def edge_id(self, client_id: int) -> int:
        client = int(client_id)
        if not 0 <= client < self.num_users:
            raise IndexError("client id outside model dimensions")
        return client % self.num_edges

    def edge_assignment(self) -> np.ndarray:
        return np.arange(self.num_users, dtype=np.int64) % self.num_edges

    def edge_state_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "service_factors": self.edge_service_factors.copy(),
            "service_bias": self.edge_service_bias.copy(),
            "service_error": self.edge_service_error.copy(),
        }

    def private_state_snapshot(self) -> dict[str, np.ndarray]:
        return super().private_state_snapshot()

    def shared_state_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "cloud_service_factors": self.service_factors.copy(),
            "cloud_service_bias": self.service_bias.copy(),
            "cloud_service_error": self.service_error.copy(),
            "edge_service_factors": self.edge_service_factors.copy(),
            "edge_service_bias": self.edge_service_bias.copy(),
            "edge_service_error": self.edge_service_error.copy(),
        }

    def parameter_snapshot(self) -> dict[str, np.ndarray]:
        snapshot = super().parameter_snapshot()
        snapshot.update(
            {
                "edge_service_factors": self.edge_service_factors.copy(),
                "edge_service_bias": self.edge_service_bias.copy(),
                "edge_service_error": self.edge_service_error.copy(),
            }
        )
        return snapshot

    def clone(self) -> "MultiFedFedOBRMF":
        return copy.deepcopy(self)

    def _require_finite_state(self) -> None:
        super()._require_finite_state()
        for name in ("edge_service_factors", "edge_service_bias", "edge_service_error"):
            if not hasattr(self, name):
                continue
            state = np.asarray(getattr(self, name))
            if state.dtype != np.float64 or not np.all(np.isfinite(state)):
                raise FloatingPointError(f"non-finite or non-float64 model state: {name}")

    def _restore_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        super()._restore_snapshot(
            {name: snapshot[name] for name in ("user_factors", "user_bias", "user_error", "service_factors", "service_bias", "service_error")}
        )
        for name in ("edge_service_factors", "edge_service_bias", "edge_service_error"):
            setattr(self, name, np.asarray(snapshot[name], dtype=np.float64).copy())

                                                                        
                                      
                                                                        
    def predict(
        self,
        users: np.ndarray,
        services: np.ndarray,
        raw_cap: float,
    ) -> np.ndarray:
        users = self._index_array(users, "users")
        services = self._index_array(services, "services")
        if users.shape != services.shape:
            raise ValueError("query users/services must have equal one-dimensional shapes")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("query user id is outside the model dimensions")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("query service id is outside the model dimensions")
        raw_cap = self._validate_raw_cap(raw_cap)
        self._require_finite_state()

        edges = users % self.num_edges
        logits = self.user_bias[users] + self.edge_service_bias[edges, services]
        logits += np.sum(
            self.user_factors[users] * self.edge_service_factors[edges, services],
            axis=1,
        )
        normalized = CausalResidualBiasMF._sigmoid(logits)
        prediction = np.expm1(normalized * np.log1p(raw_cap))
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("non-finite prediction")
        return prediction.astype(np.float32)

                                                                        
                           
                                                                        
    def advance_support(
        self,
        support_users: np.ndarray,
        support_services: np.ndarray,
        support_values: np.ndarray,
        raw_cap: float,
        bias_epochs: int = 1,
        factor_epochs: int = 1,
        shuffle_seed: int = 42,
        *,
        client_order: Sequence[int] | None = None,
    ) -> FederatedRoundResult:
                                                                              

                                                                             
                                                                     
                                                                       
                                                                            
                                                                             
                                          
           

        raw_cap = self._validate_raw_cap(raw_cap)
        bias_epochs = self._validate_epochs(bias_epochs, "bias_epochs")
        factor_epochs = self._validate_epochs(factor_epochs, "factor_epochs")
        if not np.isfinite(shuffle_seed):
            raise ValueError("shuffle_seed must be finite")
        shuffle_seed = int(shuffle_seed)
        users, services, values = self._validate_and_canonicalize_support(
            support_users, support_services, support_values, raw_cap
        )
        self._require_finite_state()
        before = self.parameter_snapshot()
        active = tuple(int(value) for value in np.unique(users))
        processing_order = self._validate_client_order(active, client_order)
        support_positions = {user: np.flatnonzero(users == user) for user in active}
        active_edges = tuple(sorted({self.edge_id(user) for user in active}))
        all_proposals: list[ServiceProposal] = []
        last_round_proposals: tuple[ServiceProposal, ...] = ()
        diagnostics_by_service: dict[int, MultiFedServiceDiagnostic] = {}

        try:
            for internal_round in range(self.internal_aggregations):
                                                                             
                                                                         
                edge_snapshot = self.edge_state_snapshot()
                proposals_by_edge_service: dict[tuple[int, int], list[ServiceProposal]] = {}
                round_proposals: list[ServiceProposal] = []
                private_updates: dict[int, dict[str, np.ndarray]] = {}

                for user in processing_order:
                    positions = support_positions[user]
                    edge = self.edge_id(user)
                    shared_snapshot = {
                        "service_factors": edge_snapshot["service_factors"][edge].copy(),
                        "service_bias": edge_snapshot["service_bias"][edge].copy(),
                        "service_error": edge_snapshot["service_error"][edge].copy(),
                    }
                    local = self._local_model_from_snapshot(
                        user=user,
                        shared_snapshot=shared_snapshot,
                    )
                    local_users = np.full(positions.size, user, dtype=np.int64)
                    local.fit_support(
                        local_users,
                        services[positions],
                        values[positions],
                        raw_cap=raw_cap,
                        bias_epochs=bias_epochs,
                        factor_epochs=factor_epochs,
                        shuffle_seed=shuffle_seed + internal_round * 1_000_003,
                    )
                    local_state = local.parameter_snapshot()
                    if not all(np.all(np.isfinite(state)) for state in local_state.values()):
                        raise FloatingPointError(f"non-finite local update for client {user}")
                    private_updates[user] = {
                        "user_factors": local.user_factors[user].copy(),
                        "user_bias": np.asarray(local.user_bias[user], dtype=np.float64),
                        "user_error": np.asarray(local.user_error[user], dtype=np.float64),
                    }
                    client_candidates = tuple(
                        ServiceProposal(
                            client_id=user,
                            service_id=int(service),
                            service_factors=local.service_factors[int(service)].copy(),
                            service_bias=float(local.service_bias[int(service)]),
                            service_error=float(local.service_error[int(service)]),
                        )
                        for service in services[positions]
                    )
                    selected = tuple(
                        self._select_client_service_proposals(
                            client_id=user,
                            proposals=client_candidates,
                            shared_snapshot=shared_snapshot,
                        )
                    )
                    candidate_services = {int(item.service_id) for item in client_candidates}
                    selected_services: set[int] = set()
                    for proposal in selected:
                        if not isinstance(proposal, ServiceProposal):
                            raise TypeError("client proposal selection must return ServiceProposal objects")
                        service = int(proposal.service_id)
                        if int(proposal.client_id) != user:
                            raise ValueError("selected proposal client_id does not match active client")
                        if service not in candidate_services:
                            raise ValueError("selected proposal was not present in client's candidates")
                        if service in selected_services:
                            raise ValueError("client proposal selection returned a duplicate service")
                        selected_services.add(service)
                        round_proposals.append(proposal)
                        proposals_by_edge_service.setdefault((edge, service), []).append(proposal)

                                                                             
                                                                          
                                              
                for (edge, service), proposals in sorted(proposals_by_edge_service.items()):
                    ordered = tuple(sorted(proposals, key=lambda item: int(item.client_id)))
                    factors = np.stack([item.service_factors for item in ordered]).astype(np.float64, copy=False)
                    biases = np.asarray([item.service_bias for item in ordered], dtype=np.float64)
                    errors = np.asarray([item.service_error for item in ordered], dtype=np.float64)
                    factor_average = np.mean(factors, axis=0, dtype=np.float64)
                    bias_average = float(np.mean(biases, dtype=np.float64))
                    error_average = float(np.mean(errors, dtype=np.float64))
                    if not (np.all(np.isfinite(factor_average)) and np.isfinite(bias_average) and np.isfinite(error_average)):
                        raise FloatingPointError(f"non-finite internal aggregate for edge={edge}, service={service}")
                    self.edge_service_factors[edge, service] = factor_average
                    self.edge_service_bias[edge, service] = bias_average
                    self.edge_service_error[edge, service] = error_average

                for user, update in private_updates.items():
                    self.user_factors[user] = update["user_factors"]
                    self.user_bias[user] = float(update["user_bias"])
                    self.user_error[user] = float(update["user_error"])

                all_proposals.extend(round_proposals)
                last_round_proposals = tuple(round_proposals)

                                                                            
                                                                              
                                                                              
                          
            by_service_edge: dict[int, dict[int, int]] = {}
            for proposal in last_round_proposals:
                edge = self.edge_id(proposal.client_id)
                edge_counts = by_service_edge.setdefault(int(proposal.service_id), {})
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

            for service in sorted(by_service_edge):
                edge_counts = by_service_edge[service]
                edges = tuple(sorted(edge_counts))
                counts = np.asarray([edge_counts[edge] for edge in edges], dtype=np.float64)
                factors = np.stack([self.edge_service_factors[edge, service] for edge in edges]).astype(np.float64, copy=False)
                biases = np.asarray([self.edge_service_bias[edge, service] for edge in edges], dtype=np.float64)
                errors = np.asarray([self.edge_service_error[edge, service] for edge in edges], dtype=np.float64)
                weights = counts / float(np.sum(counts))
                cloud_factor = np.sum(factors * weights[:, None], axis=0, dtype=np.float64)
                cloud_bias = float(np.sum(biases * weights, dtype=np.float64))
                cloud_error = float(np.sum(errors * weights, dtype=np.float64))
                average_factor = np.average(factors, axis=0, weights=counts)
                average_bias = float(np.average(biases, weights=counts))
                average_error = float(np.average(errors, weights=counts))
                cloud_difference = float(
                    max(
                        np.max(np.abs(cloud_factor - average_factor)),
                        abs(cloud_bias - average_bias),
                        abs(cloud_error - average_error),
                    )
                )
                                                                           
                                                                         
                                                            
                cloud_equivalent = bool(cloud_difference <= self.equivalence_atol)
                if not cloud_equivalent:
                    raise AssertionError(f"cloud weighted aggregate is not self-consistent for service {service}")

                p = self.external_weight
                mixed_factors = np.empty_like(factors)
                mixed_biases = np.empty_like(biases)
                mixed_errors = np.empty_like(errors)
                for index, edge in enumerate(edges):
                    mixed_factors[index] = (1.0 - p) * factors[index] + p * cloud_factor
                    mixed_biases[index] = (1.0 - p) * biases[index] + p * cloud_bias
                    mixed_errors[index] = (1.0 - p) * errors[index] + p * cloud_error
                    self.edge_service_factors[edge, service] = mixed_factors[index]
                    self.edge_service_bias[edge, service] = mixed_biases[index]
                    self.edge_service_error[edge, service] = mixed_errors[index]

                                                                             
                                                                             
                                                                  
                self.service_factors[service] = np.average(mixed_factors, axis=0, weights=counts)
                self.service_bias[service] = float(np.average(mixed_biases, weights=counts))
                self.service_error[service] = float(np.average(mixed_errors, weights=counts))
                post_cloud_factor = self.service_factors[service]
                post_difference = float(
                    max(
                        np.max(np.abs(mixed_factors - post_cloud_factor[None, :])),
                        np.max(np.abs(mixed_biases - self.service_bias[service])),
                        np.max(np.abs(mixed_errors - self.service_error[service])),
                    )
                )
                diagnostics_by_service[service] = MultiFedServiceDiagnostic(
                    service_id=int(service),
                    participating_edges=edges,
                    edge_proposal_counts=tuple(int(value) for value in counts),
                    internal_rounds=self.internal_aggregations,
                    external_weight=p,
                    cloud_max_abs_difference_before_mix=cloud_difference,
                    cloud_weighted_equivalent_before_mix=cloud_equivalent,
                    edge_max_abs_difference_after_mix=post_difference,
                )

            self._require_finite_state()
        except Exception:
            self._restore_snapshot(before)
            raise

        proposal_count = len(all_proposals)
        proposal_payload = proposal_count * self.proposal_payload_bytes
        client_edge_uplink = len(active) * 4 * self.internal_aggregations + proposal_payload
        edge_summary_payload = self.edge_summary_payload_bytes
        edge_cloud_uplink = len(active_edges) * 4 + sum(
            len(edge_counts) * edge_summary_payload for edge_counts in by_service_edge.values()
        )
        dense = self.dense_service_state_bytes
        cloud_edge_downlink = len(active_edges) * dense
        edge_client_downlink = len(active) * dense * self.internal_aggregations
        communication = MultiFedCommunicationReport(
            active_clients=active,
            active_edges=active_edges,
            support_counts=tuple((user, int(support_positions[user].size)) for user in sorted(active)),
            proposal_count=proposal_count,
            internal_aggregation_rounds=self.internal_aggregations,
            client_edge_uplink_bytes=int(client_edge_uplink),
            edge_cloud_uplink_bytes=int(edge_cloud_uplink),
            cloud_edge_downlink_bytes=int(cloud_edge_downlink),
            edge_client_downlink_bytes=int(edge_client_downlink),
        )
        return FederatedRoundResult(
            active_clients=active,
                                                                              
                                                                              
                                                                              
            proposals=tuple(sorted(all_proposals, key=lambda item: (item.service_id, item.client_id))),
            communication=communication,                          
            service_aggregation_diagnostics=tuple(diagnostics_by_service[service] for service in sorted(diagnostics_by_service)),
        )

    @property
    def edge_summary_payload_bytes(self) -> int:
                                                                              
                                                                            
        return int(8 + 8 * (self.rank + 2))


                                                                         
                                                             
MultiFed = MultiFedFedOBRMF
MultiFedOBRMF = MultiFedFedOBRMF


__all__ = [
    "MultiFed",
    "MultiFedOBRMF",
    "MultiFedCommunicationReport",
    "MultiFedFedOBRMF",
    "MultiFedServiceDiagnostic",
]

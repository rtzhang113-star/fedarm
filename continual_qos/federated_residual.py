                                                                      

                                                                        
                                                                            
                                                                           
                                                                              
                                                                           

                                                                         
                                                                        
                                                                           
                                                                     
               
   

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .residual_model import CausalResidualBiasMF


_PRIVATE_NAMES = ("user_factors", "user_bias", "user_error")
_SHARED_NAMES = ("service_factors", "service_bias", "service_error")
_ALL_NAMES = _PRIVATE_NAMES + _SHARED_NAMES


@dataclass(frozen=True)
class ServiceProposal:
                                                                   

    client_id: int
    service_id: int
    service_factors: np.ndarray
    service_bias: float
    service_error: float

    @property
    def payload_bytes(self) -> int:
                                                          

                                                                             
                                                                            
                                                             
           

        return int(4 + 8 * (int(self.service_factors.size) + 2))


@dataclass(frozen=True)
class CommunicationReport:
                                                                

    active_clients: tuple[int, ...]
    support_counts: tuple[tuple[int, int], ...]
    proposal_count: int
    uplink_bytes: int
    downlink_bytes: int

    @property
    def total_bytes(self) -> int:
        return int(self.uplink_bytes + self.downlink_bytes)

    @property
    def support_count_bytes(self) -> int:
        return int(4 * len(self.active_clients))

    @property
    def proposal_payload_bytes(self) -> int:
        return int(self.uplink_bytes - self.support_count_bytes)

    def as_dict(self) -> dict[str, object]:
        return {
            "active_clients": list(self.active_clients),
            "support_counts": [list(item) for item in self.support_counts],
            "proposal_count": int(self.proposal_count),
            "uplink_bytes": int(self.uplink_bytes),
            "downlink_bytes": int(self.downlink_bytes),
            "total_bytes": int(self.total_bytes),
        }


@dataclass(frozen=True)
class FederatedRoundResult:
                                                                     

    active_clients: tuple[int, ...]
    proposals: tuple[ServiceProposal, ...]
    communication: CommunicationReport
                                                                           
                                                                         
                                            
    service_aggregation_diagnostics: tuple[object, ...] = ()

    @property
    def client_proposals(self) -> tuple[ServiceProposal, ...]:
                                                                               

        return self.proposals

    @property
    def aggregation_diagnostics(self) -> tuple[object, ...]:
                                                                           

        return self.service_aggregation_diagnostics


class FederatedResidualBiasMF:
                                                    

                                                                             
                                                                            
                                                                             
                                                         

                                                                              
                                                                            
       

    def __init__(
        self,
        num_users: int,
        num_services: int,
        rank: int = 20,
        seed: int = 42,
        bias_eta: float = 0.8,
        factor_eta: float = 0.01,
        l2: float = 3e-4,
        beta: float = 0.3,
        factor_init_std: float = 0.05,
        *,
        drift_hook: object | None = None,
        service_aggregation_hook: object | None = None,
        memory_hook: object | None = None,
    ) -> None:
        if drift_hook is not None or service_aggregation_hook is not None or memory_hook is not None:
            raise ValueError(
                "drift_hook, service_aggregation_hook, and memory_hook are reserved "
                "for later methods and must be None in the baseline"
            )
        for name, value in (
            ("bias_eta", bias_eta),
            ("factor_eta", factor_eta),
            ("l2", l2),
            ("beta", beta),
            ("factor_init_std", factor_init_std),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

                                                                           
                                                                       
                                                            
        initialized = CausalResidualBiasMF(
            num_users=num_users,
            num_services=num_services,
            rank=rank,
            seed=seed,
            bias_eta=bias_eta,
            factor_eta=factor_eta,
            l2=l2,
            beta=beta,
            factor_init_std=factor_init_std,
            interaction=True,
        )
        self.num_users = initialized.num_users
        self.num_services = initialized.num_services
        self.rank = initialized.rank
        self.seed = int(seed)
        self.bias_eta = initialized.bias_eta
        self.factor_eta = initialized.factor_eta
        self.l2 = initialized.l2
        self.beta = initialized.beta
        self.factor_init_std = float(factor_init_std)
        self.drift_hook = None
        self.service_aggregation_hook = None
        self.memory_hook = None

                                                                         
                                                                              
                                                                          
        for name in _ALL_NAMES:
            setattr(self, name, getattr(initialized, name).copy())

                                                                        
                                              
                                                                        
    def private_state_snapshot(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in _PRIVATE_NAMES}

    def shared_state_snapshot(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in _SHARED_NAMES}

    def parameter_snapshot(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in _ALL_NAMES}

    def clone(self) -> "FederatedResidualBiasMF":
        return copy.deepcopy(self)

    def _require_finite_state(self) -> None:
        for name in _ALL_NAMES:
            state = np.asarray(getattr(self, name))
            if state.dtype != np.float64 or not np.all(np.isfinite(state)):
                raise FloatingPointError(f"non-finite or non-float64 model state: {name}")

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
        if np.any(users >= self.num_users) or np.any(services >= self.num_services):
            raise IndexError("query user/service id is outside the model dimensions")
        raw_cap = self._validate_raw_cap(raw_cap)
        self._require_finite_state()
        logits = self.user_bias[users] + self.service_bias[services]
        logits = logits + np.sum(
            self.user_factors[users] * self.service_factors[services], axis=1
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

        try:
            active = tuple(int(value) for value in np.unique(users))
            processing_order = self._validate_client_order(active, client_order)
            support_positions = {
                user: np.flatnonzero(users == user)
                for user in active
            }
                                                                            
            shared_snapshot = self.shared_state_snapshot()
            private_updates: dict[int, dict[str, np.ndarray]] = {}
            proposals: list[ServiceProposal] = []

            for user in processing_order:
                positions = support_positions[user]
                client_shared_snapshot = self._shared_snapshot_for_client(
                    user=user,
                    shared_snapshot=shared_snapshot,
                )
                local = self._local_model_from_snapshot(
                    user=user,
                    shared_snapshot=client_shared_snapshot,
                )
                local_users = np.full(positions.size, user, dtype=np.int64)
                local.fit_support(
                    local_users,
                    services[positions],
                    values[positions],
                    raw_cap=raw_cap,
                    bias_epochs=bias_epochs,
                    factor_epochs=factor_epochs,
                    shuffle_seed=shuffle_seed,
                )
                local_state = local.parameter_snapshot()
                if not all(np.all(np.isfinite(state)) for state in local_state.values()):
                    raise FloatingPointError(f"non-finite local update for client {user}")
                private_updates[user] = {
                    "user_factors": local.user_factors[user].copy(),
                    "user_bias": np.asarray(local.user_bias[user], dtype=np.float64),
                    "user_error": np.asarray(local.user_error[user], dtype=np.float64),
                }
                client_candidates: list[ServiceProposal] = []
                for service in services[positions]:
                    service = int(service)
                    client_candidates.append(
                        ServiceProposal(
                            client_id=user,
                            service_id=service,
                            service_factors=local.service_factors[service].copy(),
                            service_bias=float(local.service_bias[service]),
                            service_error=float(local.service_error[service]),
                        )
                    )
                                                                            
                                                                         
                                                                           
                                                                           
                                                                             
                                                                           
                                                                         
                                                                       
                selected = tuple(
                    self._select_client_service_proposals(
                        client_id=user,
                        proposals=tuple(client_candidates),
                        shared_snapshot=client_shared_snapshot,
                    )
                )
                candidate_services = {
                    int(proposal.service_id) for proposal in client_candidates
                }
                selected_services: set[int] = set()
                for proposal in selected:
                    if not isinstance(proposal, ServiceProposal):
                        raise TypeError(
                            "client proposal selection must return ServiceProposal objects"
                        )
                    service = int(proposal.service_id)
                    if int(proposal.client_id) != user:
                        raise ValueError(
                            "selected proposal client_id does not match the active client"
                        )
                    if service not in candidate_services:
                        raise ValueError(
                            "selected proposal was not present in the client's candidates"
                        )
                    if service in selected_services:
                        raise ValueError(
                            "client proposal selection returned a duplicate service"
                        )
                    selected_services.add(service)
                    proposals.append(proposal)

                                                                              
                                                                           
                                                                      
                                                                        
                                     
            by_service: dict[int, list[ServiceProposal]] = {}
            for proposal in proposals:
                by_service.setdefault(proposal.service_id, []).append(proposal)
            service_aggregation_diagnostics = self._commit_service_aggregates(
                by_service,
                shared_snapshot,
            )

                                                                              
                                                                              
            for user, update in private_updates.items():
                self.user_factors[user] = update["user_factors"]
                self.user_bias[user] = float(update["user_bias"])
                self.user_error[user] = float(update["user_error"])
            self._require_finite_state()
        except Exception:
            self._restore_snapshot(before)
            raise

        support_counts = tuple(
            (user, int(support_positions[user].size)) for user in sorted(active)
        )
        proposal_payload = len(proposals) * self.proposal_payload_bytes
        uplink = len(active) * 4 + proposal_payload
        downlink = len(active) * self.num_services * 8 * (self.rank + 2)
        communication = CommunicationReport(
            active_clients=active,
            support_counts=support_counts,
            proposal_count=len(proposals),
            uplink_bytes=int(uplink),
            downlink_bytes=int(downlink),
        )
        return FederatedRoundResult(
            active_clients=active,
            proposals=tuple(sorted(proposals, key=lambda item: (item.service_id, item.client_id))),
            communication=communication,
            service_aggregation_diagnostics=tuple(service_aggregation_diagnostics),
        )

    def _should_emit_service_proposal(self, service_id: int) -> bool:
                                                                          

                                                                          
                                                                               
                                                                          
                                     
           

        del service_id
        return True

    def _select_client_service_proposals(
        self,
        client_id: int,
        proposals: tuple[ServiceProposal, ...],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[ServiceProposal, ...]:
                                                                          

                                                                            
                                                                          
                                                                           
                                                                            
                                                                             
                                                                     

                                                                           
           

        del client_id, shared_snapshot
        return tuple(
            proposal
            for proposal in proposals
            if self._should_emit_service_proposal(int(proposal.service_id))
        )

    def _commit_service_aggregates(
        self,
        by_service: dict[int, list[ServiceProposal]],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[object, ...]:
                                                                 

                                                                       
                                                                            
                                                                       
                                                                           
                                                                        
           

        del shared_snapshot
        for service in by_service:
            by_service[service].sort(key=lambda item: item.client_id)
            service_proposals = by_service[service]
            factors = np.stack(
                [proposal.service_factors for proposal in service_proposals], axis=0
            ).astype(np.float64, copy=False)
            biases = np.asarray(
                [proposal.service_bias for proposal in service_proposals], dtype=np.float64
            )
            errors = np.asarray(
                [proposal.service_error for proposal in service_proposals], dtype=np.float64
            )
            factor_average = np.mean(factors, axis=0, dtype=np.float64)
            bias_average = float(np.mean(biases, dtype=np.float64))
            error_average = float(np.mean(errors, dtype=np.float64))
            if not (
                np.all(np.isfinite(factor_average))
                and np.isfinite(bias_average)
                and np.isfinite(error_average)
            ):
                raise FloatingPointError(f"non-finite service aggregate for service {service}")
            self.service_factors[service] = factor_average
            self.service_bias[service] = bias_average
            self.service_error[service] = error_average
        return ()

    @property
    def proposal_payload_bytes(self) -> int:
        return int(4 + 8 * (self.rank + 2))

    @property
    def dense_service_state_bytes(self) -> int:
        return int(self.num_services * 8 * (self.rank + 2))

                                                                        
                                            
                                                                        
    @staticmethod
    def _validate_raw_cap(raw_cap: float) -> float:
        raw_cap = float(raw_cap)
        if not np.isfinite(raw_cap) or raw_cap <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        return raw_cap

    @staticmethod
    def _validate_epochs(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @staticmethod
    def _index_array(values: np.ndarray, name: str) -> np.ndarray:
        original = np.asarray(values)
        if original.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if not np.issubdtype(original.dtype, np.integer):
            if not np.all(np.isfinite(original)) or not np.all(original == np.floor(original)):
                raise ValueError(f"{name} must contain integer ids")
        return original.astype(np.int64, copy=False)

    def _validate_and_canonicalize_support(
        self,
        support_users: np.ndarray,
        support_services: np.ndarray,
        support_values: np.ndarray,
        raw_cap: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = self._index_array(support_users, "support_users")
        services = self._index_array(support_services, "support_services")
        values = np.asarray(support_values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("support_values must be a one-dimensional array")
        if not (len(users) == len(services) == len(values)):
            raise ValueError("support arrays must have equal lengths")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("support user id is outside the model dimensions")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("support service id is outside the model dimensions")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= raw_cap):
            raise ValueError("support values must be finite and strictly inside raw_cap")
        if len(users) == 0:
            return users, services, values
        keys = users.astype(np.int64) * self.num_services + services
        if np.unique(keys).size != keys.size:
            raise ValueError("each user-service pair may occur at most once per slice")
                                                                         
                                                                         
                                                                       
                                  
        return users, services, values

    @staticmethod
    def _validate_client_order(
        active: tuple[int, ...],
        client_order: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if client_order is None:
            return active
        order = tuple(int(value) for value in client_order)
        if len(order) != len(active) or set(order) != set(active):
            raise ValueError("client_order must be a permutation of active client ids")
        return order

    def _local_model_from_snapshot(
        self,
        user: int,
        shared_snapshot: dict[str, np.ndarray],
    ) -> CausalResidualBiasMF:
                                                                           

        local = object.__new__(CausalResidualBiasMF)
        local.num_users = self.num_users
        local.num_services = self.num_services
        local.rank = self.rank
        local.bias_eta = self.bias_eta
        local.factor_eta = self.factor_eta
        local.l2 = self.l2
        local.beta = self.beta
        local.interaction = True
        local.user_factors = np.zeros_like(self.user_factors)
        local.user_bias = np.zeros_like(self.user_bias)
        local.user_error = np.ones_like(self.user_error)
        local.user_factors[user] = self.user_factors[user]
        local.user_bias[user] = self.user_bias[user]
        local.user_error[user] = self.user_error[user]
        local.service_factors = shared_snapshot["service_factors"].copy()
        local.service_bias = shared_snapshot["service_bias"].copy()
        local.service_error = shared_snapshot["service_error"].copy()
        return local

    def _shared_snapshot_for_client(
        self,
        user: int,
        shared_snapshot: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
                                                                         

                                                                         
                                                                          
                                                                            
                                   
           

        del user
        return shared_snapshot

    def _restore_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        for name in _ALL_NAMES:
            setattr(self, name, snapshot[name].copy())


                                                                           
                                                              
CausalFederatedResidualBiasMF = FederatedResidualBiasMF
FederatedResidualMF = FederatedResidualBiasMF


__all__ = [
    "CommunicationReport",
    "CausalFederatedResidualBiasMF",
    "FederatedResidualBiasMF",
    "FederatedResidualMF",
    "FederatedRoundResult",
    "ServiceProposal",
]

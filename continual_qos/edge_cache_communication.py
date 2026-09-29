                                                                 

                                                              
                                                                             
                                                                     
                 

                                                                           
                                                                
                                                        
                                                                             
        

                                                                       
                                                                         
                                                                             
                                                                           
   

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .federated_residual import FederatedRoundResult


@dataclass(frozen=True)
class EdgeCacheRoundReport:
                                                                         

    active_clients: tuple[int, ...]
    active_edges: tuple[int, ...]
    initialized_edges: tuple[int, ...]
    proposal_count: int
    touched_request_rows: int
    touched_service_rows: int
    edge_summary_count: int
    client_edge_uplink_bytes: int
    client_edge_request_bytes: int
    edge_cloud_uplink_bytes: int
    cloud_edge_bootstrap_rows: int
    cloud_edge_bootstrap_bytes: int
    cloud_edge_delta_rows: int
    cloud_edge_delta_bytes: int
    edge_client_touched_rows: int
    edge_client_touched_row_downlink_bytes: int
    cache_client_lookup_rows: int
    cache_client_hit_rows: int
    cache_bootstrap_miss_rows: int
    cache_refresh_rows: int
    flat_total_bytes: int
    naive_cloud_edge_dense_bytes: int
    naive_edge_client_dense_bytes: int
    naive_hierarchical_total_bytes: int
                                                                            
                                                                      
                                                                
    gate_control_bytes: int = 0

    @property
    def client_edge_total_uplink_bytes(self) -> int:
        return int(self.client_edge_uplink_bytes + self.client_edge_request_bytes)

    @property
    def cloud_edge_total_bytes(self) -> int:
        return int(self.cloud_edge_bootstrap_bytes + self.cloud_edge_delta_bytes)

    @property
    def edge_cache_total_bytes(self) -> int:
        return int(
            self.client_edge_total_uplink_bytes
            + self.edge_cloud_uplink_bytes
            + self.cloud_edge_total_bytes
            + self.edge_client_touched_row_downlink_bytes
            + self.gate_control_bytes
        )

    @property
    def edge_cache_steady_state_total_bytes(self) -> int:
                                                                

        return int(self.edge_cache_total_bytes - self.cloud_edge_bootstrap_bytes)

    @property
    def client_cache_hit_rate(self) -> float:
        if self.cache_client_lookup_rows == 0:
            return 1.0
        return float(self.cache_client_hit_rows / self.cache_client_lookup_rows)

    @property
    def as_dict(self) -> dict[str, object]:
        return {
            "active_clients": list(self.active_clients),
            "active_edges": list(self.active_edges),
            "initialized_edges": list(self.initialized_edges),
            "proposal_count": int(self.proposal_count),
            "touched_request_rows": int(self.touched_request_rows),
            "touched_service_rows": int(self.touched_service_rows),
            "edge_summary_count": int(self.edge_summary_count),
            "client_edge_uplink_bytes": int(self.client_edge_uplink_bytes),
            "client_edge_request_bytes": int(self.client_edge_request_bytes),
            "client_edge_total_uplink_bytes": int(self.client_edge_total_uplink_bytes),
            "edge_cloud_uplink_bytes": int(self.edge_cloud_uplink_bytes),
            "cloud_edge_bootstrap_rows": int(self.cloud_edge_bootstrap_rows),
            "cloud_edge_bootstrap_bytes": int(self.cloud_edge_bootstrap_bytes),
            "cloud_edge_delta_rows": int(self.cloud_edge_delta_rows),
            "cloud_edge_delta_bytes": int(self.cloud_edge_delta_bytes),
            "cloud_edge_total_bytes": int(self.cloud_edge_total_bytes),
            "edge_client_touched_rows": int(self.edge_client_touched_rows),
            "edge_client_touched_row_downlink_bytes": int(self.edge_client_touched_row_downlink_bytes),
            "cache_client_lookup_rows": int(self.cache_client_lookup_rows),
            "cache_client_hit_rows": int(self.cache_client_hit_rows),
            "cache_bootstrap_miss_rows": int(self.cache_bootstrap_miss_rows),
            "cache_refresh_rows": int(self.cache_refresh_rows),
            "client_cache_hit_rate": float(self.client_cache_hit_rate),
            "flat_total_bytes": int(self.flat_total_bytes),
            "naive_cloud_edge_dense_bytes": int(self.naive_cloud_edge_dense_bytes),
            "naive_edge_client_dense_bytes": int(self.naive_edge_client_dense_bytes),
            "naive_hierarchical_total_bytes": int(self.naive_hierarchical_total_bytes),
            "gate_control_bytes": int(self.gate_control_bytes),
            "edge_cache_total_bytes": int(self.edge_cache_total_bytes),
            "edge_cache_steady_state_total_bytes": int(self.edge_cache_steady_state_total_bytes),
        }


class EdgeCacheCommunicationAccountant:
                                                                           

                                                                        
                                                                             
                                                                          
                                                                           
                                                                       
                                                                             
                                                                         
               
       

    def __init__(
        self,
        *,
        num_users: int,
        num_services: int,
        rank: int,
        num_edges: int = 6,
        service_id_bytes: int = 4,
        float_bytes: int = 8,
    ) -> None:
        if int(num_users) <= 0 or int(num_services) <= 0:
            raise ValueError("num_users and num_services must be positive")
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if int(num_edges) <= 1 or int(num_edges) > int(num_users):
            raise ValueError("num_edges must be in [2, num_users]")
        if int(service_id_bytes) != 4 or int(float_bytes) != 8:
            raise ValueError("the payload format is frozen to uint32/float64")
        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.rank = int(rank)
        self.num_edges = int(num_edges)
        self.service_id_bytes = int(service_id_bytes)
        self.float_bytes = int(float_bytes)
        self._initialized_edges: set[int] = set()

    @property
    def service_row_payload_bytes(self) -> int:
                                                                       

        return int(self.service_id_bytes + self.float_bytes * (self.rank + 2))

    @property
    def dense_service_state_bytes(self) -> int:
                                                                            

        return int(self.num_services * self.float_bytes * (self.rank + 2))

    @property
    def edge_summary_payload_bytes(self) -> int:
                                                                            

        return int(2 * self.service_id_bytes + self.float_bytes * (self.rank + 2))

    @property
    def initialized_edges(self) -> tuple[int, ...]:
        return tuple(sorted(self._initialized_edges))

    def reset(self) -> None:
                                                                           

        self._initialized_edges.clear()

    def edge_id(self, user_id: int) -> int:
        user = int(user_id)
        if not 0 <= user < self.num_users:
            raise IndexError("user id outside model dimensions")
        return int(user % self.num_edges)

    def account_round(
        self,
        support_users: np.ndarray,
        support_services: np.ndarray,
        result: FederatedRoundResult,
        *,
        changed_services: Iterable[int] | None = None,
        summary_services: Iterable[int] | None = None,
        gate_control_bytes: int = 0,
    ) -> EdgeCacheRoundReport:
                                                                    

                                                                            
                                                                          
                                                                            
                                                                           
                                                                              
                                                                        
               
           

        users = self._index_array(support_users, "support_users")
        services = self._index_array(support_services, "support_services")
        if users.shape != services.shape:
            raise ValueError("support users/services must have equal shapes")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("support user id outside model dimensions")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("support service id outside model dimensions")
        keys = users.astype(np.int64) * self.num_services + services
        if np.unique(keys).size != keys.size:
            raise ValueError("each user-service support cell may occur once")

        active_clients = tuple(int(value) for value in np.unique(users))
        result_clients = tuple(int(value) for value in result.active_clients)
        if result_clients != active_clients:
            raise ValueError("round result active clients do not match support")
        active_edges = tuple(sorted({self.edge_id(value) for value in active_clients}))
        touched_by_edge: dict[int, set[int]] = {edge: set() for edge in active_edges}
        touched_by_service: dict[int, set[int]] = {}
        for user, service in zip(users.tolist(), services.tolist()):
            edge = self.edge_id(user)
            touched_by_edge[edge].add(int(service))
            touched_by_service.setdefault(int(service), set()).add(edge)

        new_edges = tuple(edge for edge in active_edges if edge not in self._initialized_edges)
        self._initialized_edges.update(active_edges)
        initialized_edges = self.initialized_edges

        if changed_services is None:
            changed = tuple(sorted({int(item.service_id) for item in result.proposals}))
        else:
            changed = tuple(sorted({int(value) for value in changed_services}))
        if any(value < 0 or value >= self.num_services for value in changed):
            raise IndexError("changed service id outside model dimensions")

        if summary_services is None:
            summary_service_set = set(touched_by_service)
        else:
            summary_service_set = {int(value) for value in summary_services}
            if any(value < 0 or value >= self.num_services for value in summary_service_set):
                raise IndexError("summary service id outside model dimensions")
            if not summary_service_set.issubset(touched_by_service):
                raise ValueError("summary_services must be touched by current support")
        if isinstance(gate_control_bytes, bool) or int(gate_control_bytes) != gate_control_bytes:
            raise ValueError("gate_control_bytes must be a non-negative integer")
        gate_control_bytes = int(gate_control_bytes)
        if gate_control_bytes < 0:
            raise ValueError("gate_control_bytes must be non-negative")
        edge_summary_count = sum(
            len(touched_by_service[service]) for service in summary_service_set
        )
        touched_request_rows = int(users.size)
        touched_service_rows = int(len(touched_by_service))
        client_edge_uplink = int(result.communication.uplink_bytes)
                                                                           
                                                                               
                                                                           
                                
        client_edge_request_bytes = int(touched_request_rows * self.service_id_bytes)
        edge_cloud_uplink = int(
            len(active_edges) * self.service_id_bytes
            + edge_summary_count * self.edge_summary_payload_bytes
        )
        bootstrap_rows = int(len(new_edges) * self.num_services)
        bootstrap_bytes = int(len(new_edges) * self.dense_service_state_bytes)
        delta_rows = int(len(changed) * len(initialized_edges))
        delta_bytes = int(delta_rows * self.service_row_payload_bytes)
        edge_client_rows = touched_request_rows
        edge_client_bytes = int(edge_client_rows * self.service_row_payload_bytes)
        cache_lookup_rows = touched_request_rows
        cache_hit_rows = touched_request_rows
        cache_refresh_rows = delta_rows

        naive_cloud_edge = int(len(active_edges) * self.dense_service_state_bytes)
        naive_edge_client = int(len(active_clients) * self.dense_service_state_bytes)
        naive_total = int(
            client_edge_uplink + edge_cloud_uplink + naive_cloud_edge + naive_edge_client
        )
        if int(result.communication.total_bytes) != int(
            result.communication.uplink_bytes + result.communication.downlink_bytes
        ):
            raise ValueError("model communication report is internally inconsistent")

        return EdgeCacheRoundReport(
            active_clients=active_clients,
            active_edges=active_edges,
            initialized_edges=initialized_edges,
            proposal_count=int(result.communication.proposal_count),
            touched_request_rows=touched_request_rows,
            touched_service_rows=touched_service_rows,
            edge_summary_count=int(edge_summary_count),
            client_edge_uplink_bytes=client_edge_uplink,
            client_edge_request_bytes=client_edge_request_bytes,
            edge_cloud_uplink_bytes=edge_cloud_uplink,
            cloud_edge_bootstrap_rows=bootstrap_rows,
            cloud_edge_bootstrap_bytes=bootstrap_bytes,
            cloud_edge_delta_rows=delta_rows,
            cloud_edge_delta_bytes=delta_bytes,
            edge_client_touched_rows=edge_client_rows,
            edge_client_touched_row_downlink_bytes=edge_client_bytes,
            cache_client_lookup_rows=cache_lookup_rows,
            cache_client_hit_rows=cache_hit_rows,
            cache_bootstrap_miss_rows=bootstrap_rows,
            cache_refresh_rows=cache_refresh_rows,
            flat_total_bytes=int(result.communication.total_bytes),
            naive_cloud_edge_dense_bytes=naive_cloud_edge,
            naive_edge_client_dense_bytes=naive_edge_client,
            naive_hierarchical_total_bytes=naive_total,
            gate_control_bytes=gate_control_bytes,
        )

    @staticmethod
    def _index_array(values: np.ndarray, name: str) -> np.ndarray:
        original = np.asarray(values)
        if original.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.issubdtype(original.dtype, np.integer):
            if not np.all(np.isfinite(original)) or not np.all(original == np.floor(original)):
                raise ValueError(f"{name} must contain integer ids")
        return original.astype(np.int64, copy=False)


__all__ = ["EdgeCacheCommunicationAccountant", "EdgeCacheRoundReport"]

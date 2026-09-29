                                                               

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .edge_cache_communication import EdgeCacheRoundReport
from .hmp_sr16_codec import PRECISION_INT8, expected_payload_bytes
from .sdq8_attribution_controls import FixedLinkPrecisionCommunicationAccountant


PROTOCOL = "request_aggregated_causal_multicast_v1"
MULTICAST_HEADER_BYTES = 8


@dataclass(frozen=True)
class DeliveryGroup:
    edge_id: int
    service_id: int
    recipients: tuple[int, ...]
    mode: str
    transmitted_bytes: int


@dataclass(frozen=True)
class DeliveryPlan:
    groups: tuple[DeliveryGroup, ...]
    request_count: int
    singleton_groups: int
    multicast_groups: int
    unicast_bytes: int
    always_multicast_bytes: int
    racm_bytes: int
    request_table_state_bytes: int

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def message_reduction_percent(self) -> float:
        if self.request_count == 0:
            return 0.0
        return 100.0 * (self.request_count - self.group_count) / self.request_count


def plan_delivery(
    users: np.ndarray,
    services: np.ndarray,
    *,
    num_users: int,
    num_services: int,
    num_edges: int = 6,
    rank: int = 20,
) -> DeliveryPlan:
    users = np.asarray(users, dtype=np.int64)
    services = np.asarray(services, dtype=np.int64)
    if users.ndim != 1 or services.ndim != 1 or users.shape != services.shape:
        raise ValueError("users and services must be equal one-dimensional arrays")
    if np.any(users < 0) or np.any(users >= int(num_users)):
        raise IndexError("user id outside model dimensions")
    if np.any(services < 0) or np.any(services >= int(num_services)):
        raise IndexError("service id outside model dimensions")
    if int(num_edges) <= 1 or int(num_edges) > int(num_users):
        raise ValueError("num_edges must be in [2, num_users]")
    keys = users * int(num_services) + services
    if np.unique(keys).size != keys.size:
        raise ValueError("each user-service request may occur once per round")
    row_bytes = int(expected_payload_bytes(PRECISION_INT8, rank=int(rank)))
    grouped: dict[tuple[int, int], list[int]] = {}
    for user, service in zip(users.tolist(), services.tolist()):
        grouped.setdefault((int(user) % int(num_edges), int(service)), []).append(int(user))
    groups: list[DeliveryGroup] = []
    singleton = 0
    multicast = 0
    racm_bytes = 0
    for (edge, service), recipients in sorted(grouped.items()):
        members = tuple(sorted(recipients))
        if len(members) == 1:
            mode = "unicast"
            transmitted = row_bytes
            singleton += 1
        else:
            mode = "multicast"
            transmitted = row_bytes + MULTICAST_HEADER_BYTES
            multicast += 1
        racm_bytes += transmitted
        groups.append(
            DeliveryGroup(
                edge_id=edge,
                service_id=service,
                recipients=members,
                mode=mode,
                transmitted_bytes=transmitted,
            )
        )
                                                                         
                                                                         
    request_table_state = int(len(groups) * 8 + users.size * 4)
    return DeliveryPlan(
        groups=tuple(groups),
        request_count=int(users.size),
        singleton_groups=int(singleton),
        multicast_groups=int(multicast),
        unicast_bytes=int(users.size * row_bytes),
        always_multicast_bytes=int(len(groups) * (row_bytes + MULTICAST_HEADER_BYTES)),
        racm_bytes=int(racm_bytes),
        request_table_state_bytes=request_table_state,
    )


class RACMCommunicationAccountant(FixedLinkPrecisionCommunicationAccountant):
                                                                        

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.last_delivery_plan: DeliveryPlan | None = None

    def account_round(self, support_users, support_services, result, **kwargs):
        base = super().account_round(support_users, support_services, result, **kwargs)
        plan = plan_delivery(
            support_users,
            support_services,
            num_users=self.num_users,
            num_services=self.num_services,
            num_edges=self.num_edges,
            rank=self.rank,
        )
        if int(base.edge_client_touched_rows) != int(plan.request_count):
            raise ValueError("request plan does not match baseline delivery rows")
        if int(base.edge_client_touched_row_downlink_bytes) != int(plan.unicast_bytes):
            raise ValueError("request plan does not match baseline delivery bytes")
        if any(
            group.transmitted_bytes
            > min(
                len(group.recipients) * expected_payload_bytes(PRECISION_INT8, rank=self.rank),
                expected_payload_bytes(PRECISION_INT8, rank=self.rank) + MULTICAST_HEADER_BYTES,
            )
            for group in plan.groups
        ):
            raise AssertionError("RACM selected a dominated delivery mode")
        self.last_delivery_plan = plan
        return replace(
            base,
            edge_client_touched_rows=int(plan.group_count),
            edge_client_touched_row_downlink_bytes=int(plan.racm_bytes),
        )


__all__ = [
    "DeliveryGroup",
    "DeliveryPlan",
    "MULTICAST_HEADER_BYTES",
    "PROTOCOL",
    "RACMCommunicationAccountant",
    "plan_delivery",
]

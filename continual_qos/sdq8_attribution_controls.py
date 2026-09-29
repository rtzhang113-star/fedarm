                                                                    

                                                                           
                                                                              
                                                                        
                                               
   

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .edge_cache_communication import (
    EdgeCacheCommunicationAccountant,
    EdgeCacheRoundReport,
)
from .federated_residual import (
    CommunicationReport,
    FederatedRoundResult,
    ServiceProposal,
)
from .hmp_sr16_codec import (
    PRECISION_FP32,
    PRECISION_FP64,
    PRECISION_INT16,
    PRECISION_INT8,
    encode_and_decode_service_row,
    expected_payload_bytes,
)
from .personalized_service_adapter import CPSAFedOBRMF


PROTOCOL = "sdq8_incremental_attribution_controls_v1"
_PRECISIONS = {PRECISION_FP64, PRECISION_FP32, PRECISION_INT16, PRECISION_INT8}


@dataclass(frozen=True)
class AbsoluteINT8Diagnostic:
    client_id: int
    proposal_count: int
    payload_bytes: int
    max_abs_decode_error: float


class AbsoluteProposalINT8CPSAFedOBRMF(CPSAFedOBRMF):
                                                                          

    model_name = "APQ8-CPSA-PCS-CARW-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "all-proposal client-edge absolute INT8 transport; unchanged PCS aggregation"
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._pending_absolute_diagnostics: list[AbsoluteINT8Diagnostic] | None = None
        self.last_absolute_int8_diagnostics: tuple[AbsoluteINT8Diagnostic, ...] = ()

    @property
    def proposal_payload_bytes(self) -> int:
        return int(expected_payload_bytes(PRECISION_INT8, rank=self.rank))

    def _select_client_service_proposals(
        self,
        client_id: int,
        proposals: tuple[ServiceProposal, ...],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[ServiceProposal, ...]:
        del shared_snapshot
        diagnostics = self._pending_absolute_diagnostics
        if diagnostics is None:
            raise RuntimeError("absolute INT8 encoding called outside an active round")
        transformed: list[ServiceProposal] = []
        payload_bytes = 0
        max_abs_error = 0.0
        for proposal in proposals:
            payload, row = encode_and_decode_service_row(
                int(proposal.service_id),
                proposal.service_factors,
                proposal.service_bias,
                proposal.service_error,
                precision=PRECISION_INT8,
                rank=self.rank,
            )
            original = np.concatenate(
                (
                    np.asarray(proposal.service_factors, dtype=np.float64),
                    np.asarray(
                        [proposal.service_bias, proposal.service_error], dtype=np.float64
                    ),
                )
            )
            decoded = np.concatenate(
                (
                    np.asarray(row.factors, dtype=np.float64),
                    np.asarray([row.bias, row.error], dtype=np.float64),
                )
            )
            if not np.all(np.isfinite(decoded)):
                raise FloatingPointError("non-finite absolute INT8 transport value")
            max_abs_error = max(max_abs_error, float(np.max(np.abs(original - decoded))))
            payload_bytes += len(payload)
            transformed.append(
                ServiceProposal(
                    client_id=int(proposal.client_id),
                    service_id=int(proposal.service_id),
                    service_factors=row.factors.copy(),
                    service_bias=float(row.bias),
                    service_error=float(row.error),
                )
            )
        diagnostics.append(
            AbsoluteINT8Diagnostic(
                client_id=int(client_id),
                proposal_count=len(proposals),
                payload_bytes=int(payload_bytes),
                max_abs_decode_error=float(max_abs_error),
            )
        )
        return tuple(transformed)

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
        if self._pending_absolute_diagnostics is not None:
            raise RuntimeError("absolute INT8 round is already active")
        self._pending_absolute_diagnostics = []
        self.last_absolute_int8_diagnostics = ()
        try:
            result = super().advance_support(
                support_users,
                support_services,
                support_values,
                raw_cap,
                bias_epochs,
                factor_epochs,
                shuffle_seed,
                client_order=client_order,
            )
            expected_uplink = int(
                len(result.active_clients) * 4
                + result.communication.proposal_count * self.proposal_payload_bytes
            )
            if result.communication.uplink_bytes != expected_uplink:
                raise AssertionError("absolute INT8 proposal wire ledger mismatch")
            self.last_absolute_int8_diagnostics = tuple(
                sorted(self._pending_absolute_diagnostics, key=lambda item: item.client_id)
            )
            communication = CommunicationReport(
                active_clients=result.communication.active_clients,
                support_counts=result.communication.support_counts,
                proposal_count=result.communication.proposal_count,
                uplink_bytes=result.communication.uplink_bytes,
                downlink_bytes=result.communication.downlink_bytes,
            )
            return FederatedRoundResult(
                active_clients=result.active_clients,
                proposals=result.proposals,
                communication=communication,
                service_aggregation_diagnostics=result.service_aggregation_diagnostics,
            )
        finally:
            self._pending_absolute_diagnostics = None


class StaticAllLinkINT8CPSAFedOBRMF(AbsoluteProposalINT8CPSAFedOBRMF):
                                                                      

    model_name = "SAQ8-CPSA-PCS-CARW-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "all-link absolute INT8 wire control; unchanged PCS aggregation"
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.last_static_aggregate_audits: tuple[dict[str, object], ...] = ()

    def _commit_service_aggregates(
        self,
        by_service: dict[int, list[ServiceProposal]],
        shared_snapshot: dict[str, np.ndarray],
    ) -> tuple[object, ...]:
        diagnostics = super()._commit_service_aggregates(by_service, shared_snapshot)
        audits: list[dict[str, object]] = []
        for service in sorted(by_service):
            payload, row = encode_and_decode_service_row(
                int(service),
                self.service_factors[service],
                self.service_bias[service],
                self.service_error[service],
                precision=PRECISION_INT8,
                rank=self.rank,
            )
            self.service_factors[service] = row.factors.copy()
            self.service_bias[service] = float(row.bias)
            self.service_error[service] = float(row.error)
            audits.append(
                {
                    "message_kind": "aggregate_refresh_delivery",
                    "service_id": int(service),
                    "precision": PRECISION_INT8,
                    "payload_bytes": int(len(payload)),
                }
            )
        self.last_static_aggregate_audits = tuple(audits)
        return diagnostics


class FixedLinkPrecisionCommunicationAccountant(EdgeCacheCommunicationAccountant):
                                                                           

    def __init__(
        self,
        *,
        proposal_precision: str,
        summary_precision: str,
        refresh_precision: str,
        delivery_precision: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        values = {
            "proposal": str(proposal_precision).upper(),
            "summary": str(summary_precision).upper(),
            "refresh": str(refresh_precision).upper(),
            "delivery": str(delivery_precision).upper(),
        }
        if any(value not in _PRECISIONS for value in values.values()):
            raise ValueError("invalid fixed link precision")
        self.link_precisions = values

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
        touched = {int(value) for value in services.tolist()}
        changed = touched if changed_services is None else {int(value) for value in changed_services}
        summary = touched if summary_services is None else {int(value) for value in summary_services}
        base = super().account_round(
            users,
            services,
            result,
            changed_services=changed,
            summary_services=summary,
            gate_control_bytes=gate_control_bytes,
        )
        active_clients = tuple(int(value) for value in np.unique(users))
        active_edges = tuple(sorted({self.edge_id(value) for value in active_clients}))
        touched_by_service: dict[int, set[int]] = {}
        for user, service in zip(users.tolist(), services.tolist()):
            touched_by_service.setdefault(int(service), set()).add(self.edge_id(int(user)))
        edge_summary_count = sum(len(touched_by_service[sid]) for sid in summary)
        proposal_bytes = int(
            len(active_clients) * self.service_id_bytes
            + sum(
                expected_payload_bytes(
                    self.link_precisions["proposal"], rank=self.rank, summary=False
                )
                for _ in result.proposals
            )
        )
        summary_bytes = int(
            len(active_edges) * self.service_id_bytes
            + sum(
                len(touched_by_service[sid])
                * expected_payload_bytes(
                    self.link_precisions["summary"], rank=self.rank, summary=True
                )
                for sid in summary
            )
        )
        refresh_bytes = int(
            len(base.initialized_edges)
            * len(changed)
            * expected_payload_bytes(
                self.link_precisions["refresh"], rank=self.rank, summary=False
            )
        )
        delivery_bytes = int(
            services.size
            * expected_payload_bytes(
                self.link_precisions["delivery"], rank=self.rank, summary=False
            )
        )
        return EdgeCacheRoundReport(
            active_clients=base.active_clients,
            active_edges=base.active_edges,
            initialized_edges=base.initialized_edges,
            proposal_count=base.proposal_count,
            touched_request_rows=base.touched_request_rows,
            touched_service_rows=base.touched_service_rows,
            edge_summary_count=int(edge_summary_count),
            client_edge_uplink_bytes=proposal_bytes,
            client_edge_request_bytes=base.client_edge_request_bytes,
            edge_cloud_uplink_bytes=summary_bytes,
            cloud_edge_bootstrap_rows=base.cloud_edge_bootstrap_rows,
            cloud_edge_bootstrap_bytes=base.cloud_edge_bootstrap_bytes,
            cloud_edge_delta_rows=base.cloud_edge_delta_rows,
            cloud_edge_delta_bytes=refresh_bytes,
            edge_client_touched_rows=base.edge_client_touched_rows,
            edge_client_touched_row_downlink_bytes=delivery_bytes,
            cache_client_lookup_rows=base.cache_client_lookup_rows,
            cache_client_hit_rows=base.cache_client_hit_rows,
            cache_bootstrap_miss_rows=base.cache_bootstrap_miss_rows,
            cache_refresh_rows=base.cache_refresh_rows,
            flat_total_bytes=base.flat_total_bytes,
            naive_cloud_edge_dense_bytes=base.naive_cloud_edge_dense_bytes,
            naive_edge_client_dense_bytes=base.naive_edge_client_dense_bytes,
            naive_hierarchical_total_bytes=base.naive_hierarchical_total_bytes,
            gate_control_bytes=base.gate_control_bytes,
        )


__all__ = [
    "AbsoluteINT8Diagnostic",
    "AbsoluteProposalINT8CPSAFedOBRMF",
    "FixedLinkPrecisionCommunicationAccountant",
    "PROTOCOL",
    "StaticAllLinkINT8CPSAFedOBRMF",
]

                                                                     

                                                                       
                                                                             
                                                                            
                                                               
                                                                        

                                                                            
                                                                           
                                                                          
                        
   

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .adaptive_residual_window import CausalAdaptiveResidualWindowFedOBRMF
from .drift_diagnostic import raw_values_to_logit
from .pcs_srr_combo_fedobr import ProposalCountScaledSupportResidualRecenterFedOBRMF
from .residual_model import CausalResidualBiasMF


                                                                         
                                                                            
                   
FROZEN_CPSA_GAIN = 0.1
FROZEN_CPSA_L2 = 0.0003
FROZEN_CPSA_RESIDUAL_CLIP = (-3.0, 3.0)
FROZEN_CPSA_DECAY = 1.0 - FROZEN_CPSA_GAIN * FROZEN_CPSA_L2
_ADAPTER_STATE_NAME = "client_service_adapter"


@dataclass(frozen=True)
class _AdapterTransaction:
                                                                       

                                                                             
                                                                             
                                                                            
                                                                 
       

    users: np.ndarray
    services: np.ndarray
    old_values: np.ndarray
    new_values: np.ndarray

    @property
    def touched_count(self) -> int:
        return int(self.users.size)


class ClientPrivateSparseAdditiveServiceAdapterMixin:
                                                   

                                                                        
                                                                           
                                                                     
                 

                                                                              

                                                                          
                                                                          
                                                                         
                                                                             
                                                    
       

    model_name = "CPSA-PCS-CARW-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "PCS-CARW-SRR-FedOBR-MF with client-private sparse additive logit adapters"
    )
    adapter_gain = float(FROZEN_CPSA_GAIN)
    adapter_l2 = float(FROZEN_CPSA_L2)
    adapter_residual_clip = FROZEN_CPSA_RESIDUAL_CLIP
    adapter_decay = float(FROZEN_CPSA_DECAY)

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.client_service_adapter = np.zeros(
            (self.num_users, self.num_services), dtype=np.float64
        )
        self._require_finite_state()

                                                                        
                             
                                                                        
    @property
    def adapter(self) -> np.ndarray:
                                                               

                                                      
                                                                           
                                                                           
           

        return self.client_service_adapter

    @property
    def service_adapter(self) -> np.ndarray:
                                                                       

        return self.client_service_adapter

    @property
    def private_service_adapter(self) -> np.ndarray:
                                                                  

        return self.client_service_adapter

    def adapter_state_snapshot(self) -> dict[str, np.ndarray]:
                                                                          

        return {
            _ADAPTER_STATE_NAME: np.asarray(self.client_service_adapter).copy()
        }

                                                    
    private_adapter_state_snapshot = adapter_state_snapshot
    client_private_state_snapshot = adapter_state_snapshot

    def full_state_snapshot(self) -> dict[str, np.ndarray]:
                                                                          

        snapshot = self.parameter_snapshot()
        snapshot.update(self.adapter_state_snapshot())
        return snapshot

    @property
    def adapter_storage_bytes(self) -> int:
                                                                             

        return int(self.client_service_adapter.nbytes)

    @property
    def private_adapter_storage_bytes(self) -> int:
                                                    

        return self.adapter_storage_bytes

    @property
    def additional_communication_bytes(self) -> int:
                                                                             

        return 0

    def _require_finite_state(self) -> None:
                                                                             

        super()._require_finite_state()
        adapter = np.asarray(self.client_service_adapter)
        if (
            adapter.shape != (self.num_users, self.num_services)
            or adapter.dtype != np.float64
        ):
            raise FloatingPointError(
                "client_service_adapter must have the model shape and float64 dtype"
            )
        if not np.all(np.isfinite(adapter)):
            raise FloatingPointError("non-finite client/service adapter state")

                                                                        
                               
                                                                        
    def _pre_round_adapter_update(
        self,
        users: np.ndarray,
        services: np.ndarray,
        values: np.ndarray,
        raw_cap: float,
    ) -> _AdapterTransaction:
                                                                        

                                                                            
                                                                             
                                                  
           

        if users.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return _AdapterTransaction(
                users=users.astype(np.int64, copy=True),
                services=services.astype(np.int64, copy=True),
                old_values=empty.copy(),
                new_values=empty.copy(),
            )

                                                                          
                                                                            
                                                                              
        state = self.parameter_snapshot()
        base_logit = np.asarray(state["user_bias"][users], dtype=np.float64)
        base_logit = base_logit + np.asarray(
            state["service_bias"][services], dtype=np.float64
        )
        base_logit = base_logit + np.sum(
            state["user_factors"][users] * state["service_factors"][services],
            axis=1,
            dtype=np.float64,
        )
        target_logit = raw_values_to_logit(values, raw_cap)
        old = np.asarray(
            self.client_service_adapter[users, services], dtype=np.float64
        ).copy()
        residual = target_logit - (base_logit + old)
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("non-finite pre-round CPSA residual")
        clipped = np.clip(
            residual,
            FROZEN_CPSA_RESIDUAL_CLIP[0],
            FROZEN_CPSA_RESIDUAL_CLIP[1],
        ).astype(np.float64, copy=False)
        updated = np.asarray(
            FROZEN_CPSA_DECAY * old + FROZEN_CPSA_GAIN * clipped,
            dtype=np.float64,
        )
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("non-finite CPSA adapter update")
                                                                             
                                                                 
        return _AdapterTransaction(
            users=users.astype(np.int64, copy=True),
            services=services.astype(np.int64, copy=True),
            old_values=old,
            new_values=np.asarray(updated, dtype=np.float64).copy(),
        )

    def _commit_adapter_transaction(self, transaction: _AdapterTransaction) -> None:
                                                                              

        if transaction.touched_count:
            self.client_service_adapter[
                transaction.users, transaction.services
            ] = transaction.new_values

    def _rollback_adapter_transaction(self, transaction: _AdapterTransaction) -> None:
                                                                             

        if transaction.touched_count:
            self.client_service_adapter[
                transaction.users, transaction.services
            ] = transaction.old_values

    @staticmethod
    def _auxiliary_state_names() -> tuple[str, ...]:
                                                                         

        return (
            "_adaptive_residual_history",
            "_pending_adaptive_observations",
            "_pending_adaptive_diagnostics",
            "_last_adaptive_diagnostics",
            "_pending_recenter_diagnostics",
            "_last_recenter_diagnostics",
        )

    def _capture_auxiliary_state(self) -> dict[str, object]:
        return {
            name: copy.deepcopy(getattr(self, name))
            for name in self._auxiliary_state_names()
            if hasattr(self, name)
        }

    def _restore_auxiliary_state(self, snapshot: dict[str, object]) -> None:
        for name, value in snapshot.items():
            setattr(self, name, copy.deepcopy(value))

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
    ):
                                                                     

                                                                          
                                                                       
                                                                          
                                                                            
                     
           

        raw_cap = self._validate_raw_cap(raw_cap)
        users, services, values = self._validate_and_canonicalize_support(
            support_users, support_services, support_values, raw_cap
        )
        self._require_finite_state()
        ordinary_before = self.parameter_snapshot()
        auxiliary_before = self._capture_auxiliary_state()
        transaction = self._pre_round_adapter_update(
            users, services, values, raw_cap
        )

        try:
            result = super().advance_support(
                users,
                services,
                values,
                raw_cap,
                bias_epochs,
                factor_epochs,
                shuffle_seed,
                client_order=client_order,
            )
                                                                           
                                                                              
                                                             
            self._commit_adapter_transaction(transaction)
            self._require_finite_state()
            return result
        except Exception:
                                                                             
                                                                             
                                                                          
                                                                           
                            
            self._restore_snapshot(ordinary_before)
            self._rollback_adapter_transaction(transaction)
            self._restore_auxiliary_state(auxiliary_before)
            raise

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
        logits = self.user_bias[users] + self.service_bias[services]
        logits = logits + np.sum(
            self.user_factors[users] * self.service_factors[services],
            axis=1,
            dtype=np.float64,
        )
        logits = logits + np.asarray(
            self.client_service_adapter[users, services], dtype=np.float64
        )
        normalized = CausalResidualBiasMF._sigmoid(logits)
        prediction = np.expm1(normalized * np.log1p(raw_cap))
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("non-finite CPSA prediction")
        return prediction.astype(np.float32)


class ClientPrivateSparseAdditiveServiceAdapterPCSSRRFedOBRMF(
    ClientPrivateSparseAdditiveServiceAdapterMixin,
    ProposalCountScaledSupportResidualRecenterFedOBRMF,
):
                                                                         

    model_name = "CPSA-PCS-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "PCS-SRR-FedOBR-MF with client-private sparse additive logit adapters"
    )


class ClientPrivateSparseAdditiveServiceAdapterFedOBRMF(
    ClientPrivateSparseAdditiveServiceAdapterMixin,
    CausalAdaptiveResidualWindowFedOBRMF,
):
                                                                           

                                                                        
                                                                        
                                                                     
       

    model_name = "CPSA-PCS-CARW-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "PCS-CARW-SRR-FedOBR-MF with client-private sparse additive logit adapters"
    )


                                                                            
                                                                           
CPSAFedOBRMF = ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
CPSAPCSSRRFedOBRMF = ClientPrivateSparseAdditiveServiceAdapterPCSSRRFedOBRMF
CPSAPersonalizedFedOBRMF = ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
PersonalizedServiceAdapterFedOBRMF = ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
ClientPrivateServiceAdapterFedOBRMF = ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
CPSA = ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
PersonalizedServiceAdapter = ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
ClientPrivateSparseAdditiveServiceAdapter = (
    ClientPrivateSparseAdditiveServiceAdapterFedOBRMF
)


__all__ = [
    "CPSA",
    "CPSAFedOBRMF",
    "CPSAPCSSRRFedOBRMF",
    "CPSAPersonalizedFedOBRMF",
    "ClientPrivateSparseAdditiveServiceAdapter",
    "ClientPrivateSparseAdditiveServiceAdapterMixin",
    "ClientPrivateSparseAdditiveServiceAdapterPCSSRRFedOBRMF",
    "ClientPrivateServiceAdapterFedOBRMF",
    "ClientPrivateSparseAdditiveServiceAdapterFedOBRMF",
    "FROZEN_CPSA_DECAY",
    "FROZEN_CPSA_GAIN",
    "FROZEN_CPSA_L2",
    "FROZEN_CPSA_RESIDUAL_CLIP",
    "PersonalizedServiceAdapter",
    "PersonalizedServiceAdapterFedOBRMF",
]

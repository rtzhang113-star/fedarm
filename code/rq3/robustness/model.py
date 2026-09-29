                                                         

                                                                              
                                                                          
                                                                            
                                                                       
         

                                                                            
                                                                          
                                                                         
                                                                        
                                                                            
                                          

                                                                          
                                                                        
                                                                         
                      
   

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from continual_qos.gpu_ablation_training import (
    CARW_WINDOW,
    GPUAblationAdvanceProfile,
    GPUAblationModel,
    cuda,
    cuda_available,
)


RQ3_VARIANTS = ("full", "service_mean_matched")
RQ3_VARIANT_BITS = {
                                                                           
    "full": "111",
    "service_mean_matched": "101",
}


class GPURobustnessModel(GPUAblationModel):
                                                                  

                                                                               
                                                                             
                                                                             
                                                                          
                                                                             
       

    model_name = "RQ3-Robustness-FedARM-GPU"
    aggregation_algorithm = "RQ3 Full FedARM on CUDA"

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
        num_edges: int = 6,
        variant: str = "full",
    ) -> None:
        key = str(variant).strip().lower()
        if key not in RQ3_VARIANTS:
            valid = ", ".join(RQ3_VARIANTS)
            raise ValueError(
                f"unknown RQ3 robustness variant {variant!r}; expected one of {valid}"
            )

                                                                             
                                                                              
        super().__init__(
            num_users=num_users,
            num_services=num_services,
            rank=rank,
            seed=seed,
            bias_eta=bias_eta,
            factor_eta=factor_eta,
            l2=l2,
            beta=beta,
            factor_init_std=factor_init_std,
            num_edges=num_edges,
            variant="full",
        )

                                                                       
                                                                        
                                                                           
                                                                            
                                                                             
                                                                        
                                                                           
                                                                           
        assert cuda is not None
        self._d_history = cuda.to_device(
            np.zeros((self.num_users, CARW_WINDOW + 1), dtype=np.float64)
        )
        self._d_new_history = cuda.device_array(
            (self.num_users, CARW_WINDOW + 1), dtype=np.float64
        )

        self.variant = key
        self.variant_bits = RQ3_VARIANT_BITS[key]
        self.use_cpsa = True
        self.use_carw_srr = True
        if key == "service_mean_matched":
            self.use_pcs = False
            self.aggregation_algorithm = (
                "service-level equal mean over effective proposals "
                "with CPSA and CARW/SRR on CUDA"
            )
        else:
            self.use_pcs = True
            self.aggregation_algorithm = "PCS with CPSA and CARW/SRR on CUDA"

                                                                              
                                                                             
                                                                         
        self.last_diagnostics.update(
            {
                "variant": self.variant,
                "variant_bits": self.variant_bits,
                "rq3_variant": self.variant,
                "use_cpsa": self.use_cpsa,
                "use_pcs": self.use_pcs,
                "use_carw_srr": self.use_carw_srr,
                "aggregation_semantics": (
                    "pcs_service_aggregation"
                    if self.use_pcs
                    else "service_level_effective_proposal_mean"
                ),
            }
        )

    def advance_support(
        self,
        support_users: np.ndarray,
        support_services: np.ndarray,
        support_values: np.ndarray,
        raw_cap: float,
        bias_epochs: int = 1,
        factor_epochs: int = 1,
        shuffle_seed: int = 42,
    ) -> SimpleNamespace:
                                                                     

                                                                              
                                                                      
                                                             
                                                   
           

        result = super().advance_support(
            support_users,
            support_services,
            support_values,
            raw_cap,
            bias_epochs,
            factor_epochs,
            shuffle_seed,
        )
        self.last_diagnostics.update(
            {
                "rq3_variant": self.variant,
                "aggregation_semantics": (
                    "pcs_service_aggregation"
                    if self.use_pcs
                    else "service_level_effective_proposal_mean"
                ),
            }
        )
        return result


__all__ = [
    "GPUAblationAdvanceProfile",
    "GPURobustnessModel",
    "RQ3_VARIANT_BITS",
    "RQ3_VARIANTS",
    "cuda_available",
]

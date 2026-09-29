                                                                 

from __future__ import annotations

from .federated_residual import FederatedResidualBiasMF


class FedOBRMF(FederatedResidualBiasMF):
                                                                       

                                                                       
                                                                             
                                                                            
                                                       
       

    model_name = "FedOBR-MF"
    backbone_name = "OBR-MF"
    aggregation_algorithm = "FedAvg"


__all__ = ["FedOBRMF"]

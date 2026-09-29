                                                                

                                                                  

                                                                            
                                                      
                                                                        
                                                        

                                                                     
                                                                            
                                                                          
                                                                          
   

from __future__ import annotations

from .drift_residual_recenter import SupportResidualRecenterMixin
from .proposal_count_scaled_fedobr import ProposalCountScaledFedOBRMF


class ProposalCountScaledSupportResidualRecenterFedOBRMF(
    SupportResidualRecenterMixin,
    ProposalCountScaledFedOBRMF,
):
                                                                           

    model_name = "PCS-SRR-FedOBR-MF"
    aggregation_algorithm = (
        "proposal-count-scaled service-factor aggregation with support residual recentering"
    )


                                                                          
                                 
PCSSRRFedOBRMF = ProposalCountScaledSupportResidualRecenterFedOBRMF
ProposalCountScaledSRRFedOBRMF = ProposalCountScaledSupportResidualRecenterFedOBRMF


__all__ = [
    "PCSSRRFedOBRMF",
    "ProposalCountScaledSRRFedOBRMF",
    "ProposalCountScaledSupportResidualRecenterFedOBRMF",
]

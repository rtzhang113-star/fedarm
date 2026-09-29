                                                              

                                                                           
                                                                              
                                                                             
                                                                             
                    
   

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:                                                                   
    from numba import cuda
except Exception:                                             
    cuda = None                            


MAX_RANK = 64
MAX_PROPOSALS_PER_SERVICE = 256
INT8_LIMIT = 127
SCALE_FLOOR = 2.0 ** -24


if cuda is not None:

    @cuda.jit
    def cpsa_update_kernel(
        users: Any,
        services: Any,
        values: Any,
        raw_cap: float,
        user_factors: Any,
        service_factors: Any,
        user_bias: Any,
        service_bias: Any,
        adapter: Any,
        cpsa_gain: float,
        cpsa_decay: float,
    ) -> None:
                                                                   

                                                                            
                                                                             
                                                                     
           

        index = cuda.grid(1)
        if index >= users.size:
            return
        user = users[index]
        service = services[index]
        normalized = math.log1p(values[index]) / math.log1p(raw_cap)
        target = math.log(normalized / (1.0 - normalized))
        model_logit = user_bias[user] + service_bias[service]
        for factor in range(user_factors.shape[1]):
            model_logit += user_factors[user, factor] * service_factors[service, factor]
        residual = target - (model_logit + adapter[user, service])
        if residual < -3.0:
            residual = -3.0
        elif residual > 3.0:
            residual = 3.0
        adapter[user, service] = (
            cpsa_decay * adapter[user, service] + cpsa_gain * residual
        )


    @cuda.jit
    def pcs_aggregate_kernel(
        touched_services: Any,
        service_offsets: Any,
        proposal_order: Any,
        proposal_factors: Any,
        proposal_bias: Any,
        proposal_error: Any,
        service_factors: Any,
        service_bias: Any,
        service_error: Any,
        count_exponent: float,
        trust_exponent: float,
        fallback_flags: Any,
        applied_scales: Any,
        count_scales: Any,
        raw_delta_norms: Any,
        trust_norms: Any,
        clip_flags: Any,
        committed_norms: Any,
    ) -> None:
                                                                       

                                                                      
                                                                           
                                                                           
                                                                             
                               
           

        service_index = cuda.grid(1)
        if service_index >= touched_services.size:
            return
        service = touched_services[service_index]
        start = service_offsets[service_index]
        end = service_offsets[service_index + 1]
        count = end - start
        if count <= 0 or count > MAX_PROPOSALS_PER_SERVICE:
            fallback_flags[service_index] = 1
            applied_scales[service_index] = 0.0
            count_scales[service_index] = 0.0
            raw_delta_norms[service_index] = 0.0
            trust_norms[service_index] = 0.0
            clip_flags[service_index] = 0
            committed_norms[service_index] = 0.0
            return

        mean_factor = cuda.local.array(MAX_RANK, dtype=np.float64)
        delta_sum = cuda.local.array(MAX_RANK, dtype=np.float64)
        norms = cuda.local.array(MAX_PROPOSALS_PER_SERVICE, dtype=np.float64)
        for factor in range(proposal_factors.shape[1]):
            mean_factor[factor] = 0.0
            delta_sum[factor] = 0.0
        mean_bias = 0.0
        mean_error = 0.0
        for cursor in range(start, end):
            proposal = proposal_order[cursor]
            mean_bias += proposal_bias[proposal]
            mean_error += proposal_error[proposal]
            norm_square = 0.0
            for factor in range(proposal_factors.shape[1]):
                value = proposal_factors[proposal, factor]
                mean_factor[factor] += value
                delta = value - service_factors[service, factor]
                delta_sum[factor] += delta
                norm_square += delta * delta
            norms[cursor - start] = math.sqrt(norm_square)
        for factor in range(proposal_factors.shape[1]):
            mean_factor[factor] /= count
        mean_bias /= count
        mean_error /= count

        for left in range(1, count):
            value = norms[left]
            right = left - 1
            while right >= 0 and norms[right] > value:
                norms[right + 1] = norms[right]
                right -= 1
            norms[right + 1] = value
        if count % 2 == 1:
            median_norm = norms[count // 2]
        else:
            median_norm = 0.5 * (norms[count // 2 - 1] + norms[count // 2])

                                                                           
                                                                             
        if count_exponent == 1.0 or count == 1:
            count_scale = 1.0
        else:
            count_scale = math.pow(float(count), count_exponent - 1.0)
        if trust_exponent == 0.5:
            trust_count_scale = math.sqrt(float(count))
        else:
            trust_count_scale = math.pow(float(count), trust_exponent)

        raw_norm_square = 0.0
        for factor in range(proposal_factors.shape[1]):
            scaled_delta = count_scale * delta_sum[factor]
            raw_norm_square += scaled_delta * scaled_delta
        raw_norm = math.sqrt(raw_norm_square)
        trust_norm = trust_count_scale * median_norm

        fallback = count < 2 or median_norm <= 1e-12
        applied = 1.0
        clipped = 0
        if not fallback:
            if raw_norm > trust_norm:
                applied = trust_norm / raw_norm
                clipped = 1
            if not math.isfinite(applied):
                fallback = True
        if fallback:
            applied = 0.0
            for factor in range(proposal_factors.shape[1]):
                service_factors[service, factor] = mean_factor[factor]
        else:
            for factor in range(proposal_factors.shape[1]):
                if count_exponent == 1.0 or count == 1:
                    raw_delta = delta_sum[factor]
                else:
                    raw_delta = count_scale * delta_sum[factor]
                service_factors[service, factor] += applied * raw_delta
        service_bias[service] = mean_bias
        service_error[service] = mean_error
        fallback_flags[service_index] = 1 if fallback else 0
        applied_scales[service_index] = applied
        count_scales[service_index] = count_scale
        raw_delta_norms[service_index] = raw_norm
        trust_norms[service_index] = trust_norm
        clip_flags[service_index] = clipped
        committed_square = 0.0
        for factor in range(proposal_factors.shape[1]):
            value = service_factors[service, factor]
            committed_square += value * value
        committed_norms[service_index] = math.sqrt(committed_square)


    @cuda.jit
    def finite_state_kernel(values: Any, flag: Any) -> None:
                                                                                 

        index = cuda.grid(1)
        if index >= values.size:
            return
        if not math.isfinite(values.flat[index]):
            flag[0] = 0

else:                                             
    cpsa_update_kernel = None
    pcs_aggregate_kernel = None
    finite_state_kernel = None


def pcs_count_scale(count: int, count_exponent: float) -> float:
                                                                            

    n = int(count)
    if n <= 0:
        raise ValueError("count must be positive")
    exponent = float(count_exponent)
    if not math.isfinite(exponent) or not 0.0 <= exponent <= 2.0:
        raise ValueError("count_exponent must be finite and in [0, 2]")
    if n == 1 or exponent == 1.0:
        return 1.0
    return float(math.pow(float(n), exponent - 1.0))


def pcs_trust_radius(count: int, median_norm: float, trust_exponent: float = 0.5) -> float:
                                                                          

    n = int(count)
    median = float(median_norm)
    exponent = float(trust_exponent)
    if n <= 0 or not math.isfinite(median) or median < 0.0:
        raise ValueError("count and median_norm must be finite and non-negative")
    if exponent != 0.5:
        raise ValueError("RQ6 fixes trust_exponent at 0.5")
    return float(math.sqrt(float(n)) * median)


__all__ = [
    "MAX_PROPOSALS_PER_SERVICE",
    "MAX_RANK",
    "cpsa_update_kernel",
    "finite_state_kernel",
    "pcs_aggregate_kernel",
    "pcs_count_scale",
    "pcs_trust_radius",
]

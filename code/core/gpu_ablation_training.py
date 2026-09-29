                                                                

                                                                            
                                                                           
                                                                            
                               

                   
                              
               
                                                                
              
                                                     
              
                  
                  
                                                 
          
                                                           

                                                                            
                                                                            
                                                                             
                              
   

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np

try:                                                                    
    from numba import cuda
except Exception:                                                        
    cuda = None                            


GPU_BACKEND = "numba-cuda"
EXECUTION_MODE = "ablation_gpu_training"
MAX_RANK = 64
MAX_PROPOSALS_PER_SERVICE = 256
CARW_WINDOW = 16
CARW_MIN_OBSERVATIONS = 8
CARW_MIN_SEGMENT = 4
CARW_DELTA = 0.05
CARW_RANGE = 3.0
CPSA_GAIN = 0.1
CPSA_L2 = 0.0003
CPSA_DECAY = 1.0 - CPSA_GAIN * CPSA_L2
INT8_LIMIT = 127
SCALE_FLOOR = 2.0 ** -24


@dataclass(frozen=True)
class VariantSpec:
                                                                 

    name: str
    bits: str
    use_cpsa: bool
    use_pcs: bool
    use_carw_srr: bool


VARIANT_SPECS: dict[str, VariantSpec] = {
    "backbone_mean": VariantSpec("backbone_mean", "000", False, False, False),
    "cpsa_mean": VariantSpec("cpsa_mean", "100", True, False, False),
    "pcs_only": VariantSpec("pcs_only", "010", False, True, False),
    "cpsa_pcs": VariantSpec("cpsa_pcs", "110", True, True, False),
    "pcs_carw_srr": VariantSpec("pcs_carw_srr", "011", False, True, True),
    "full": VariantSpec("full", "111", True, True, True),
}

_VARIANT_ALIASES = {
    "base": "backbone_mean",
    "000": "backbone_mean",
    "100": "cpsa_mean",
    "010": "pcs_only",
    "110": "cpsa_pcs",
    "011": "pcs_carw_srr",
    "111": "full",
}


def resolve_variant(value: str | VariantSpec) -> VariantSpec:
                                                                           

    if isinstance(value, VariantSpec):
        spec = value
    else:
        key = str(value).strip().lower()
        key = _VARIANT_ALIASES.get(key, key)
        try:
            spec = VARIANT_SPECS[key]
        except KeyError as exc:
            valid = ", ".join(sorted(VARIANT_SPECS))
            raise ValueError(
                f"unknown RQ2 variant {value!r}; expected one of {valid}"
            ) from exc
    if spec.bits not in {"000", "100", "010", "110", "011", "111"}:
        raise ValueError(
            "RQ2 variant bits must be one of 000, 100, 010, 110, 011, 111"
        )
    return spec


VARIANT_NAMES = tuple(VARIANT_SPECS)
VARIANT_BITS = {name: spec.bits for name, spec in VARIANT_SPECS.items()}


def cuda_available() -> bool:
                                                     

    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def cuda_device_info() -> dict[str, Any]:
                                                                    

    if not cuda_available():
        return {
            "backend": GPU_BACKEND,
            "available": False,
            "name": None,
            "compute_capability": None,
        }
    assert cuda is not None
    try:
        device = cuda.get_current_device()
        raw_name = getattr(device, "name", None)
        name = (
            raw_name.decode("utf-8", errors="replace")
            if isinstance(raw_name, bytes)
            else (None if raw_name is None else str(raw_name))
        )
        capability = getattr(device, "compute_capability", None)
        return {
            "backend": GPU_BACKEND,
            "available": True,
            "name": name,
            "compute_capability": (
                [int(item) for item in capability]
                if capability is not None
                else None
            ),
        }
    except Exception as exc:                                       
        return {
            "backend": GPU_BACKEND,
            "available": False,
            "name": None,
            "compute_capability": None,
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _sigmoid(value: float) -> float:
        if value < -30.0:
            value = -30.0
        elif value > 30.0:
            value = 30.0
        return 1.0 / (1.0 + math.exp(-value))


    @cuda.jit(device=True, inline=True)
    def _raw_to_logit(value: float, raw_cap: float) -> float:
        normalized = math.log1p(value) / math.log1p(raw_cap)
        return math.log(normalized / (1.0 - normalized))


    @cuda.jit(device=True, inline=True)
    def _round_even(value: float) -> float:
        lower = math.floor(value)
        fraction = value - lower
        if fraction < 0.5:
            return lower
        if fraction > 0.5:
            return lower + 1.0
        if int(lower) % 2 == 0:
            return lower
        return lower + 1.0


    @cuda.jit
    def _cpsa_update_kernel(
        users: Any,
        services: Any,
        values: Any,
        raw_cap: float,
        user_factors: Any,
        service_factors: Any,
        user_bias: Any,
        service_bias: Any,
        adapter: Any,
    ) -> None:
                                                               

        index = cuda.grid(1)
        if index >= users.size:
            return
        user = users[index]
        service = services[index]
        target = _raw_to_logit(values[index], raw_cap)
        model_logit = user_bias[user] + service_bias[service]
        for factor in range(user_factors.shape[1]):
            model_logit += user_factors[user, factor] * service_factors[service, factor]
        residual = target - (model_logit + adapter[user, service])
        if residual < -3.0:
            residual = -3.0
        elif residual > 3.0:
            residual = 3.0
        adapter[user, service] = CPSA_DECAY * adapter[user, service] + CPSA_GAIN * residual


    @cuda.jit
    def _broadcast_shared_kernel(
        active_count: int,
        num_services: int,
        global_factors: Any,
        global_bias: Any,
        global_error: Any,
        local_factors: Any,
        local_bias: Any,
        local_error: Any,
    ) -> None:
                                                                    

        index = cuda.grid(1)
        total = active_count * num_services
        if index >= total:
            return
        slot = index // num_services
        service = index - slot * num_services
        local_bias[slot, service] = global_bias[service]
        local_error[slot, service] = global_error[service]
        for factor in range(global_factors.shape[1]):
            local_factors[slot, service, factor] = global_factors[service, factor]


    @cuda.jit
    def _fit_local_kernel(
        active_users: Any,
        bias_offsets: Any,
        bias_order: Any,
        factor_offsets: Any,
        factor_order: Any,
        support_services: Any,
        support_values: Any,
        raw_cap: float,
        bias_eta: float,
        factor_eta: float,
        l2: float,
        beta: float,
        user_factors: Any,
        user_bias: Any,
        user_error: Any,
        local_factors: Any,
        local_bias: Any,
        local_error: Any,
        loss_by_client: Any,
    ) -> None:
                                                                        

        slot = cuda.blockIdx.x
        if slot >= active_users.size or cuda.threadIdx.x != 0:
            return
        user = active_users[slot]
        for cursor in range(bias_offsets[slot], bias_offsets[slot + 1]):
            position = bias_order[cursor]
            service = support_services[position]
            target = math.log1p(support_values[position]) / math.log1p(raw_cap)
            prediction = _sigmoid(user_bias[user] + local_bias[slot, service])
            relative_error = abs(prediction - target) / max(target, 1e-12)
            denominator = user_error[user] + local_error[slot, service]
            user_weight = user_error[user] / max(denominator, 1e-12)
            service_weight = 1.0 - user_weight
            user_error[user] = (
                beta * user_weight * relative_error
                + (1.0 - beta * user_weight) * user_error[user]
            )
            local_error[slot, service] = (
                beta * service_weight * relative_error
                + (1.0 - beta * service_weight) * local_error[slot, service]
            )
            gradient = (prediction - target) * prediction * (1.0 - prediction)
            user_bias[user] -= bias_eta * (
                user_weight * gradient + l2 * user_bias[user]
            )
            local_bias[slot, service] -= bias_eta * (
                service_weight * gradient + l2 * local_bias[slot, service]
            )

        before_user = cuda.local.array(MAX_RANK, dtype=np.float64)
        before_service = cuda.local.array(MAX_RANK, dtype=np.float64)
        loss = 0.0
        updates = 0
        for cursor in range(factor_offsets[slot], factor_offsets[slot + 1]):
            position = factor_order[cursor]
            service = support_services[position]
            target = math.log1p(support_values[position]) / math.log1p(raw_cap)
            target_logit = math.log(target / (1.0 - target))
            residual_target = target_logit - (
                user_bias[user] + local_bias[slot, service]
            )
            for factor in range(user_factors.shape[1]):
                before_user[factor] = user_factors[user, factor]
                before_service[factor] = local_factors[slot, service, factor]
            error = 0.0
            for factor in range(user_factors.shape[1]):
                error += before_user[factor] * before_service[factor]
            error -= residual_target
            for factor in range(user_factors.shape[1]):
                user_factors[user, factor] -= factor_eta * (
                    error * before_service[factor] + l2 * before_user[factor]
                )
                local_factors[slot, service, factor] -= factor_eta * (
                    error * before_user[factor] + l2 * before_service[factor]
                )
            loss += error * error
            updates += 1
        loss_by_client[slot] = loss / max(updates, 1)


    @cuda.jit
    def _zero_float_kernel(values: Any) -> None:
        index = cuda.grid(1)
        if index < values.size:
            values.flat[index] = 0.0


    @cuda.jit
    def _zero_int_kernel(values: Any) -> None:
        index = cuda.grid(1)
        if index < values.size:
            values.flat[index] = 0


    @cuda.jit
    def _residual_stats_kernel(
        active_users: Any,
        segment_offsets: Any,
        segment_order: Any,
        support_services: Any,
        support_values: Any,
        raw_cap: float,
        user_factors: Any,
        user_bias: Any,
        local_factors: Any,
        local_bias: Any,
        residuals: Any,
        current_rms: Any,
        sums: Any,
        counts: Any,
    ) -> None:
                                                                  

        slot = cuda.blockIdx.x
        if slot >= active_users.size or cuda.threadIdx.x != 0:
            return
        user = active_users[slot]
        square_sum = 0.0
        start = segment_offsets[slot]
        end = segment_offsets[slot + 1]
        for cursor in range(start, end):
            position = segment_order[cursor]
            service = support_services[position]
            target_logit = _raw_to_logit(support_values[position], raw_cap)
            model_logit = user_bias[user] + local_bias[slot, service]
            for factor in range(user_factors.shape[1]):
                model_logit += user_factors[user, factor] * local_factors[slot, service, factor]
            residual = target_logit - model_logit
            if residual < -3.0:
                residual = -3.0
            elif residual > 3.0:
                residual = 3.0
            residuals[position] = residual
            square_sum += residual * residual
            sums[slot, service] += residual
            counts[slot, service] += 1
        current_rms[slot] = math.sqrt(square_sum / max(end - start, 1))


    @cuda.jit
    def _carw_srr_kernel(
        active_users: Any,
        history: Any,
        history_length: Any,
        current_rms: Any,
        sums: Any,
        counts: Any,
        local_bias: Any,
        new_history: Any,
        new_history_length: Any,
        gains: Any,
    ) -> None:
                                                                           

        slot = cuda.blockIdx.x
        if slot >= active_users.size or cuda.threadIdx.x != 0:
            return
        user = active_users[slot]
        hcount = history_length[user]
        current = current_rms[slot]
        combined_count = hcount + 1
        max_excess = 0.0
        max_abs_diff = 0.0
        epsilon_for_max_diff = 0.0
        change_cut = -1

        if combined_count >= CARW_MIN_OBSERVATIONS:
            last_cut = combined_count - CARW_MIN_SEGMENT
            for cut in range(CARW_MIN_SEGMENT, last_cut + 1):
                left_sum = 0.0
                right_sum = 0.0
                for index in range(cut):
                    if index < hcount:
                        left_sum += history[user, index]
                    else:
                        left_sum += current
                for index in range(cut, combined_count):
                    if index < hcount:
                        right_sum += history[user, index]
                    else:
                        right_sum += current
                n0 = cut
                n1 = combined_count - cut
                epsilon = CARW_RANGE * math.sqrt(
                    0.5 * math.log(4.0 / CARW_DELTA) * (1.0 / n0 + 1.0 / n1)
                )
                difference = abs(left_sum / n0 - right_sum / n1)
                excess = difference - epsilon
                if excess < 0.0:
                    excess = 0.0
                if difference > max_abs_diff:
                    max_abs_diff = difference
                    epsilon_for_max_diff = epsilon
                if excess > max_excess:
                    max_excess = excess
                if difference > epsilon:
                    change_cut = cut

        retained = cuda.local.array(17, dtype=np.float64)
        if change_cut < 0:
            retained_length = hcount
            for index in range(retained_length):
                retained[index] = history[user, index]
        else:
            retained_length = combined_count - 1 - change_cut
            for index in range(retained_length):
                retained[index] = history[user, change_cut + index]

        if hcount == 0 or change_cut >= 0:
            gain = 1.0
        else:
            continuity_floor = 1.0 / math.sqrt(1.0 + hcount)
            denominator = max_abs_diff + epsilon_for_max_diff
            if denominator <= 0.0:
                evidence_ratio = 0.0
            else:
                evidence_ratio = max_excess / denominator
                if evidence_ratio > 1.0:
                    evidence_ratio = 1.0
            gain = max(continuity_floor, evidence_ratio)

        for service in range(local_bias.shape[1]):
            if counts[slot, service] > 0:
                local_bias[slot, service] += gain * (
                    sums[slot, service] / counts[slot, service]
                )

        output_length = retained_length + 1
        for index in range(retained_length):
            new_history[user, index] = retained[index]
        new_history[user, retained_length] = current
        if output_length > CARW_WINDOW:
            shift = output_length - CARW_WINDOW
            for index in range(CARW_WINDOW):
                new_history[user, index] = new_history[user, index + shift]
            output_length = CARW_WINDOW
        new_history_length[user] = output_length
        gains[slot] = gain


    @cuda.jit
    def _gather_proposals_kernel(
        support_users: Any,
        support_services: Any,
        user_to_slot: Any,
        user_factors: Any,
        user_bias: Any,
        user_error: Any,
        local_factors: Any,
        local_bias: Any,
        local_error: Any,
        proposal_factors: Any,
        proposal_bias: Any,
        proposal_error: Any,
        proposal_clients: Any,
        proposal_services: Any,
    ) -> None:
                                                                     

        index = cuda.grid(1)
        if index >= support_users.size:
            return
        user = support_users[index]
        service = support_services[index]
        slot = user_to_slot[user]
        proposal_clients[index] = user
        proposal_services[index] = service
        proposal_bias[index] = local_bias[slot, service]
        proposal_error[index] = local_error[slot, service]
        for factor in range(user_factors.shape[1]):
            proposal_factors[index, factor] = local_factors[slot, service, factor]


    @cuda.jit
    def _quantize_rows_kernel(
        factors: Any,
        biases: Any,
        errors: Any,
        factor_scales: Any,
        scalar_scales: Any,
    ) -> None:
                                                              

        index = cuda.grid(1)
        if index >= factors.shape[0]:
            return
        max_factor = 0.0
        for factor in range(factors.shape[1]):
            value = abs(factors[index, factor])
            if value > max_factor:
                max_factor = value
        factor_scale = max(max_factor / INT8_LIMIT, SCALE_FLOOR)
        factor_scales[index] = factor_scale
        factor_scale = float(factor_scales[index])
        for factor in range(factors.shape[1]):
            code = _round_even(factors[index, factor] / factor_scale)
            if code < -INT8_LIMIT:
                code = -INT8_LIMIT
            elif code > INT8_LIMIT:
                code = INT8_LIMIT
            factors[index, factor] = code * factor_scale

        max_scalar = max(abs(biases[index]), abs(errors[index]))
        scalar_scale = max(max_scalar / INT8_LIMIT, SCALE_FLOOR)
        scalar_scales[index] = scalar_scale
        scalar_scale = float(scalar_scales[index])
        bias_code = _round_even(biases[index] / scalar_scale)
        error_code = _round_even(errors[index] / scalar_scale)
        if bias_code < -INT8_LIMIT:
            bias_code = -INT8_LIMIT
        elif bias_code > INT8_LIMIT:
            bias_code = INT8_LIMIT
        if error_code < -INT8_LIMIT:
            error_code = -INT8_LIMIT
        elif error_code > INT8_LIMIT:
            error_code = INT8_LIMIT
        biases[index] = bias_code * scalar_scale
        errors[index] = error_code * scalar_scale


    @cuda.jit
    def _fedavg_aggregate_kernel(
        touched_services: Any,
        service_offsets: Any,
        proposal_order: Any,
        proposal_factors: Any,
        proposal_bias: Any,
        proposal_error: Any,
        service_factors: Any,
        service_bias: Any,
        service_error: Any,
    ) -> None:
                                                                               

        service_index = cuda.grid(1)
        if service_index >= touched_services.size:
            return
        service = touched_services[service_index]
        start = service_offsets[service_index]
        end = service_offsets[service_index + 1]
        count = end - start
        if count <= 0:
            return
        for factor in range(proposal_factors.shape[1]):
            total = 0.0
            for cursor in range(start, end):
                proposal = proposal_order[cursor]
                total += proposal_factors[proposal, factor]
            service_factors[service, factor] = total / count
        total_bias = 0.0
        total_error = 0.0
        for cursor in range(start, end):
            proposal = proposal_order[cursor]
            total_bias += proposal_bias[proposal]
            total_error += proposal_error[proposal]
        service_bias[service] = total_bias / count
        service_error[service] = total_error / count


    @cuda.jit
    def _pcs_aggregate_kernel(
        touched_services: Any,
        service_offsets: Any,
        proposal_order: Any,
        proposal_factors: Any,
        proposal_bias: Any,
        proposal_error: Any,
        service_factors: Any,
        service_bias: Any,
        service_error: Any,
        fallback_flags: Any,
        applied_scales: Any,
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
        raw_norm_square = 0.0
        for factor in range(proposal_factors.shape[1]):
            raw_norm_square += delta_sum[factor] * delta_sum[factor]
        raw_norm = math.sqrt(raw_norm_square)
        trust_norm = math.sqrt(count) * median_norm

        fallback = count < 2 or median_norm <= 1e-12
        applied = 1.0
        if not fallback:
            if raw_norm > trust_norm:
                applied = trust_norm / raw_norm
            if not math.isfinite(applied):
                fallback = True
        if fallback:
            applied = 0.0
            for factor in range(proposal_factors.shape[1]):
                service_factors[service, factor] = mean_factor[factor]
        else:
            for factor in range(proposal_factors.shape[1]):
                service_factors[service, factor] += applied * delta_sum[factor]
        service_bias[service] = mean_bias
        service_error[service] = mean_error
        fallback_flags[service_index] = 1 if fallback else 0
        applied_scales[service_index] = applied
        committed_square = 0.0
        for factor in range(proposal_factors.shape[1]):
            value = service_factors[service, factor]
            committed_square += value * value
        committed_norms[service_index] = math.sqrt(committed_square)


    @cuda.jit
    def _quantize_service_rows_kernel(
        touched_services: Any,
        service_factors: Any,
        service_bias: Any,
        service_error: Any,
        factor_scales: Any,
        scalar_scales: Any,
    ) -> None:
                                                              

        index = cuda.grid(1)
        if index >= touched_services.size:
            return
        service = touched_services[index]
        max_factor = 0.0
        for factor in range(service_factors.shape[1]):
            value = abs(service_factors[service, factor])
            if value > max_factor:
                max_factor = value
        factor_scale = max(max_factor / INT8_LIMIT, SCALE_FLOOR)
        factor_scales[index] = factor_scale
        factor_scale = float(factor_scales[index])
        for factor in range(service_factors.shape[1]):
            code = _round_even(service_factors[service, factor] / factor_scale)
            if code < -INT8_LIMIT:
                code = -INT8_LIMIT
            elif code > INT8_LIMIT:
                code = INT8_LIMIT
            service_factors[service, factor] = code * factor_scale

        max_scalar = max(abs(service_bias[service]), abs(service_error[service]))
        scalar_scale = max(max_scalar / INT8_LIMIT, SCALE_FLOOR)
        scalar_scales[index] = scalar_scale
        scalar_scale = float(scalar_scales[index])
        bias_code = _round_even(service_bias[service] / scalar_scale)
        error_code = _round_even(service_error[service] / scalar_scale)
        if bias_code < -INT8_LIMIT:
            bias_code = -INT8_LIMIT
        elif bias_code > INT8_LIMIT:
            bias_code = INT8_LIMIT
        if error_code < -INT8_LIMIT:
            error_code = -INT8_LIMIT
        elif error_code > INT8_LIMIT:
            error_code = INT8_LIMIT
        service_bias[service] = bias_code * scalar_scale
        service_error[service] = error_code * scalar_scale


    @cuda.jit
    def _predict_kernel(
        users: Any,
        services: Any,
        user_factors: Any,
        service_factors: Any,
        user_bias: Any,
        service_bias: Any,
        adapter: Any,
        log1p_cap: float,
        output: Any,
    ) -> None:
                                                                              

        index = cuda.grid(1)
        if index >= users.size:
            return
        user = users[index]
        service = services[index]
        logit = user_bias[user] + service_bias[service] + adapter[user, service]
        for factor in range(user_factors.shape[1]):
            logit += user_factors[user, factor] * service_factors[service, factor]
        output[index] = math.expm1(_sigmoid(logit) * log1p_cap)

else:                                             
    _sigmoid = _raw_to_logit = _round_even = None
    _cpsa_update_kernel = _broadcast_shared_kernel = _fit_local_kernel = None
    _zero_float_kernel = _zero_int_kernel = _residual_stats_kernel = None
    _carw_srr_kernel = _gather_proposals_kernel = None
    _quantize_rows_kernel = _fedavg_aggregate_kernel = None
    _pcs_aggregate_kernel = _quantize_service_rows_kernel = None
    _predict_kernel = None


@dataclass(frozen=True)
class GPUAblationAdvanceProfile:
                                                                        

    support_count: int
    active_clients: int
    input_upload_seconds: float
    cpsa_seconds: float
    shared_broadcast_seconds: float
    local_update_seconds: float
    carw_srr_seconds: float
    proposal_quantization_seconds: float
    aggregation_seconds: float
    aggregate_quantization_seconds: float
    total_seconds: float
    peak_device_memory_used_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_count": int(self.support_count),
            "active_clients": int(self.active_clients),
            "input_upload_seconds": float(self.input_upload_seconds),
            "cpsa_seconds": float(self.cpsa_seconds),
            "shared_broadcast_seconds": float(self.shared_broadcast_seconds),
            "local_update_seconds": float(self.local_update_seconds),
            "carw_srr_seconds": float(self.carw_srr_seconds),
            "proposal_quantization_seconds": float(self.proposal_quantization_seconds),
            "aggregation_seconds": float(self.aggregation_seconds),
            "aggregate_quantization_seconds": float(self.aggregate_quantization_seconds),
            "total_seconds": float(self.total_seconds),
            "peak_device_memory_used_bytes": int(self.peak_device_memory_used_bytes),
        }


def _metric_totals(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return (
        float(np.sum(np.abs(error), dtype=np.float64)),
        float(np.sum(np.square(error), dtype=np.float64)),
        int(error.size),
    )


def prediction_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
                                                                   

    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if target.shape != estimate.shape or target.size == 0:
        raise ValueError("truth and prediction must be non-empty equal-shaped arrays")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(estimate)):
        raise FloatingPointError("metrics require finite truth and prediction")
    if np.any(target <= 0.0):
        raise ValueError("MRE/NPRE require strictly positive QoS truth")
    absolute, squared, count = _metric_totals(target, estimate)
    relative = np.sort(np.abs(estimate - target) / target)
    return {
        "mae": float(absolute / count),
        "rmse": float(math.sqrt(squared / count)),
        "mre": float(np.median(relative)),
        "npre": float(relative[int(0.9 * count)]),
        "absolute_error_sum": float(absolute),
        "squared_error_sum": float(squared),
        "count": int(count),
    }


class GPUAblationModel:
                                                                     

                                                                        
                                                                           
                                                                             
       

    model_name = "RQ2-GPU-Ablation-FedOBR-MF"
    aggregation_algorithm = "variant-controlled SAQ8 FedOBR-MF on CUDA"

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
        variant: str | VariantSpec = "full",
    ) -> None:
        if not cuda_available():
            raise RuntimeError("a usable Numba CUDA device is required")
        spec = resolve_variant(variant)
        if int(rank) <= 0 or int(rank) > MAX_RANK:
            raise ValueError(f"rank must be in [1, {MAX_RANK}]")
        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.rank = int(rank)
        self.seed = int(seed)
        self.bias_eta = float(bias_eta)
        self.factor_eta = float(factor_eta)
        self.l2 = float(l2)
        self.beta = float(beta)
        self.factor_init_std = float(factor_init_std)
        self.num_edges = int(num_edges)
        self.variant = spec.name
        self.variant_bits = spec.bits
        self.use_cpsa = bool(spec.use_cpsa)
        self.use_pcs = bool(spec.use_pcs)
        self.use_carw_srr = bool(spec.use_carw_srr)
        if self.num_users <= 0 or self.num_services <= 0:
            raise ValueError("dimensions must be positive")
        if not all(
            np.isfinite(value)
            for value in (
                self.bias_eta,
                self.factor_eta,
                self.l2,
                self.beta,
                self.factor_init_std,
            )
        ):
            raise ValueError("hyperparameters must be finite")
        if self.bias_eta <= 0 or self.factor_eta <= 0 or self.factor_init_std <= 0:
            raise ValueError("learning rates and factor scale must be positive")
        if self.l2 < 0 or not 0 < self.beta <= 1:
            raise ValueError("invalid residual MF hyperparameters")
        if cuda is None:                                                 
            raise RuntimeError("Numba CUDA import failed")

        rng = np.random.default_rng(self.seed)
        user_factors = rng.normal(
            0.0, self.factor_init_std, (self.num_users, self.rank)
        ).astype(np.float64)
        service_factors = rng.normal(
            0.0, self.factor_init_std, (self.num_services, self.rank)
        ).astype(np.float64)
        zeros_user = np.zeros(self.num_users, dtype=np.float64)
        zeros_service = np.zeros(self.num_services, dtype=np.float64)
        ones_user = np.ones(self.num_users, dtype=np.float64)
        ones_service = np.ones(self.num_services, dtype=np.float64)
        self._d_user_factors = cuda.to_device(user_factors)
        self._d_service_factors = cuda.to_device(service_factors)
        self._d_user_bias = cuda.to_device(zeros_user)
        self._d_service_bias = cuda.to_device(zeros_service)
        self._d_user_error = cuda.to_device(ones_user)
        self._d_service_error = cuda.to_device(ones_service)
        self._d_adapter = cuda.to_device(
            np.zeros((self.num_users, self.num_services), dtype=np.float64)
        )
        self._d_history = cuda.to_device(
            np.zeros((self.num_users, CARW_WINDOW), dtype=np.float64)
        )
        self._d_history_length = cuda.to_device(
            np.zeros(self.num_users, dtype=np.int32)
        )
        self._d_new_history = cuda.device_array(
            (self.num_users, CARW_WINDOW), dtype=np.float64
        )
        self._d_new_history_length = cuda.device_array(
            self.num_users, dtype=np.int32
        )
        self.last_profile: GPUAblationAdvanceProfile | None = None
        self.last_diagnostics: dict[str, Any] = {
            "variant": self.variant,
            "variant_bits": self.variant_bits,
            "use_cpsa": self.use_cpsa,
            "use_pcs": self.use_pcs,
            "use_carw_srr": self.use_carw_srr,
            "query_or_future_used": False,
        }
        self._peak_memory = self._memory_used()

    def _memory_used(self) -> int:
        assert cuda is not None
        try:
            free, total = cuda.current_context().get_memory_info()
            return int(total - free)
        except Exception:
            return 0

    def _observe_memory(self) -> int:
        current = self._memory_used()
        self._peak_memory = max(self._peak_memory, current)
        return current

    @staticmethod
    def _upload(value: np.ndarray) -> Any:
        assert cuda is not None
        return cuda.to_device(np.ascontiguousarray(value))

    @staticmethod
    def _grid(length: int, threads: int = 128) -> tuple[int, int]:
        return ((int(length) + threads - 1) // threads, threads)

    def _prepare_orders(
        self,
        support_users: np.ndarray,
        support_services: np.ndarray,
        seed: int,
        active_users: np.ndarray,
        bias_epochs: int,
        factor_epochs: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
                                                                               

        segment_offsets = [0]
        segment_order: list[int] = []
        bias_offsets = [0]
        factor_offsets = [0]
        bias_order: list[int] = []
        factor_order: list[int] = []
        for user in active_users.tolist():
            positions = np.flatnonzero(support_users == int(user))
            segment_offsets.append(segment_offsets[-1] + int(positions.size))
            segment_order.extend(int(value) for value in positions.tolist())
            bias_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(user), 0xB1A5])
            )
            for _ in range(int(bias_epochs)):
                permutation = bias_rng.permutation(positions)
                bias_order.extend(int(value) for value in permutation.tolist())
            bias_offsets.append(len(bias_order))
            factor_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(user), 0xFAC7])
            )
            for _ in range(int(factor_epochs)):
                permutation = factor_rng.permutation(positions)
                factor_order.extend(int(value) for value in permutation.tolist())
            factor_offsets.append(len(factor_order))
        if support_services.size != support_users.size:
            raise ValueError("support users/services length mismatch")
        return (
            np.asarray(segment_offsets, dtype=np.int32),
            np.asarray(segment_order, dtype=np.int32),
            np.asarray(bias_offsets, dtype=np.int32),
            np.asarray(bias_order, dtype=np.int32),
            np.asarray(factor_offsets, dtype=np.int32),
            np.asarray(factor_order, dtype=np.int32),
        )

    def parameter_snapshot(self) -> dict[str, np.ndarray]:
                                                                          

        assert cuda is not None
        return {
            "user_factors": self._d_user_factors.copy_to_host(),
            "service_factors": self._d_service_factors.copy_to_host(),
            "user_bias": self._d_user_bias.copy_to_host(),
            "service_bias": self._d_service_bias.copy_to_host(),
            "user_error": self._d_user_error.copy_to_host(),
            "service_error": self._d_service_error.copy_to_host(),
            "client_service_adapter": self._d_adapter.copy_to_host(),
        }

    def full_state_snapshot(self) -> dict[str, np.ndarray]:
                                                                               

        assert cuda is not None
        snapshot = self.parameter_snapshot()
        history = self._d_history.copy_to_host()
        lengths = self._d_history_length.copy_to_host()
        for user in range(self.num_users):
            length = int(lengths[user])
            if length > 0:
                snapshot[f"carw_history_{user}"] = np.asarray(
                    history[user, :length], dtype=np.float64
                ).copy()
        return snapshot

    def adaptive_residual_history_snapshot(self) -> dict[int, tuple[float, ...]]:
                                                                      

        assert cuda is not None
        history = self._d_history.copy_to_host()
        lengths = self._d_history_length.copy_to_host()
        return {
            int(user): tuple(float(value) for value in history[user, : int(lengths[user])])
            for user in range(self.num_users)
            if int(lengths[user]) > 0
        }

    carw_history_snapshot = adaptive_residual_history_snapshot

    @property
    def client_service_adapter(self) -> np.ndarray:
                                                                   

        assert cuda is not None
        return self._d_adapter.copy_to_host()

    @property
    def adapter_storage_bytes(self) -> int:
        return int(self.num_users * self.num_services * np.dtype(np.float64).itemsize)

    @property
    def private_state_bytes(self) -> int:
                                                                         

        return int(
            self.num_users * (self.rank + 2) * np.dtype(np.float64).itemsize
            + self.adapter_storage_bytes
        )

    @property
    def proposal_payload_bytes(self) -> int:
                                                                           

        return int(4 + 1 + 8 + self.rank + 2)

    def predict(
        self,
        users: np.ndarray,
        services: np.ndarray,
        raw_cap: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
                                                     

        if _predict_kernel is None or cuda is None:
            raise RuntimeError("CUDA kernels are unavailable")
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.ndim != 1 or services.ndim != 1 or users.shape != services.shape:
            raise ValueError("query users/services must be equal one-dimensional arrays")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("query user outside model dimensions")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("query service outside model dimensions")
        if not np.isfinite(raw_cap) or float(raw_cap) <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        d_users = self._upload(users)
        d_services = self._upload(services)
        output = cuda.device_array(users.size, dtype=np.float32)
        started = time.perf_counter()
        if users.size:
            blocks, threads = self._grid(users.size)
            _predict_kernel[blocks, threads](
                d_users,
                d_services,
                self._d_user_factors,
                self._d_service_factors,
                self._d_user_bias,
                self._d_service_bias,
                self._d_adapter,
                math.log1p(float(raw_cap)),
                output,
            )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - started
        started_copy = time.perf_counter()
        host = output.copy_to_host()
        cuda.synchronize()
        copy_seconds = time.perf_counter() - started_copy
        self._observe_memory()
        if not np.all(np.isfinite(host)):
            raise FloatingPointError("GPU prediction produced non-finite values")
        return host, {
            "kernel_seconds": float(kernel_seconds),
            "device_to_host_seconds": float(copy_seconds),
            "total_seconds": float(kernel_seconds + copy_seconds),
        }

    @staticmethod
    def _validate_support(
        users: np.ndarray,
        services: np.ndarray,
        values: np.ndarray,
        num_users: int,
        num_services: int,
        raw_cap: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if users.ndim != 1 or services.ndim != 1 or values.ndim != 1:
            raise ValueError("support arrays must be one-dimensional")
        if not (users.size == services.size == values.size) or users.size == 0:
            raise ValueError("support arrays must be non-empty and equally sized")
        if np.any(users < 0) or np.any(users >= int(num_users)):
            raise IndexError("support user outside model dimensions")
        if np.any(services < 0) or np.any(services >= int(num_services)):
            raise IndexError("support service outside model dimensions")
        if (
            not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
            or np.any(values >= float(raw_cap))
        ):
            raise ValueError("support values must be finite and inside raw_cap")
        keys = users * int(num_services) + services
        if np.unique(keys).size != keys.size:
            raise ValueError("duplicate user-service support pair")
        return users, services, values

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
                                                                            

        if cuda is None:
            raise RuntimeError("CUDA kernels are unavailable")
        raw_cap = float(raw_cap)
        if not np.isfinite(raw_cap) or raw_cap <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        if isinstance(bias_epochs, bool) or isinstance(factor_epochs, bool):
            raise ValueError("epochs must be non-negative integers")
        bias_epochs = int(bias_epochs)
        factor_epochs = int(factor_epochs)
        if bias_epochs < 0 or factor_epochs < 0:
            raise ValueError("epochs must be non-negative integers")
        if not np.isfinite(shuffle_seed):
            raise ValueError("shuffle_seed must be finite")
        users, services, values = self._validate_support(
            support_users,
            support_services,
            support_values,
            self.num_users,
            self.num_services,
            raw_cap,
        )

        active = np.unique(users).astype(np.int64, copy=False)
        active_count = int(active.size)
        (
            segment_offsets,
            segment_order,
            bias_offsets,
            bias_order,
            factor_offsets,
            factor_order,
        ) = self._prepare_orders(
            users,
            services,
            int(shuffle_seed),
            active,
            bias_epochs,
            factor_epochs,
        )
        user_to_slot = np.full(self.num_users, -1, dtype=np.int32)
        user_to_slot[active] = np.arange(active_count, dtype=np.int32)

        started_total = time.perf_counter()
        started_upload = time.perf_counter()
        d_users = self._upload(users)
        d_services = self._upload(services)
        d_values = self._upload(values)
        d_active = self._upload(active)
        d_user_to_slot = self._upload(user_to_slot)
        d_segment_offsets = self._upload(segment_offsets)
        d_segment_order = self._upload(segment_order)
        d_bias_offsets = self._upload(bias_offsets)
        d_bias_order = self._upload(bias_order)
        d_factor_offsets = self._upload(factor_offsets)
        d_factor_order = self._upload(factor_order)
        input_upload_seconds = time.perf_counter() - started_upload

        if self.use_cpsa:
            started = time.perf_counter()
            blocks, threads = self._grid(users.size)
            _cpsa_update_kernel[blocks, threads](
                d_users,
                d_services,
                d_values,
                raw_cap,
                self._d_user_factors,
                self._d_service_factors,
                self._d_user_bias,
                self._d_service_bias,
                self._d_adapter,
            )
            cuda.synchronize()
            cpsa_seconds = time.perf_counter() - started
        else:
            cpsa_seconds = 0.0

        local_factors = cuda.device_array(
            (active_count, self.num_services, self.rank), dtype=np.float64
        )
        local_bias = cuda.device_array(
            (active_count, self.num_services), dtype=np.float64
        )
        local_error = cuda.device_array(
            (active_count, self.num_services), dtype=np.float64
        )
        started = time.perf_counter()
        blocks, threads = self._grid(active_count * self.num_services)
        _broadcast_shared_kernel[blocks, threads](
            active_count,
            self.num_services,
            self._d_service_factors,
            self._d_service_bias,
            self._d_service_error,
            local_factors,
            local_bias,
            local_error,
        )
        cuda.synchronize()
        shared_broadcast_seconds = time.perf_counter() - started

        loss_by_client = cuda.device_array(active_count, dtype=np.float64)
        started = time.perf_counter()
        _fit_local_kernel[active_count, 1](
            d_active,
            d_bias_offsets,
            d_bias_order,
            d_factor_offsets,
            d_factor_order,
            d_services,
            d_values,
            raw_cap,
            self.bias_eta,
            self.factor_eta,
            self.l2,
            self.beta,
            self._d_user_factors,
            self._d_user_bias,
            self._d_user_error,
            local_factors,
            local_bias,
            local_error,
            loss_by_client,
        )
        cuda.synchronize()
        local_update_seconds = time.perf_counter() - started

        if self.use_carw_srr:
            residuals = cuda.device_array(users.size, dtype=np.float64)
            current_rms = cuda.device_array(active_count, dtype=np.float64)
            sums = cuda.device_array(
                (active_count, self.num_services), dtype=np.float64
            )
            counts = cuda.device_array(
                (active_count, self.num_services), dtype=np.int32
            )
            blocks, threads = self._grid(active_count * self.num_services)
            _zero_float_kernel[blocks, threads](sums)
            _zero_int_kernel[blocks, threads](counts)
            started = time.perf_counter()
            _residual_stats_kernel[active_count, 1](
                d_active,
                d_segment_offsets,
                d_segment_order,
                d_services,
                d_values,
                raw_cap,
                self._d_user_factors,
                self._d_user_bias,
                local_factors,
                local_bias,
                residuals,
                current_rms,
                sums,
                counts,
            )
            self._d_new_history.copy_to_device(self._d_history)
            self._d_new_history_length.copy_to_device(self._d_history_length)
            gains = cuda.device_array(active_count, dtype=np.float64)
            _carw_srr_kernel[active_count, 1](
                d_active,
                self._d_history,
                self._d_history_length,
                current_rms,
                sums,
                counts,
                local_bias,
                self._d_new_history,
                self._d_new_history_length,
                gains,
            )
            cuda.synchronize()
            carw_srr_seconds = time.perf_counter() - started
        else:
            current_rms = None
            gains = None
            carw_srr_seconds = 0.0

        proposal_factors = cuda.device_array(
            (users.size, self.rank), dtype=np.float64
        )
        proposal_bias = cuda.device_array(users.size, dtype=np.float64)
        proposal_error = cuda.device_array(users.size, dtype=np.float64)
        proposal_clients = cuda.device_array(users.size, dtype=np.int64)
        proposal_services = cuda.device_array(users.size, dtype=np.int64)
        started = time.perf_counter()
        blocks, threads = self._grid(users.size)
        _gather_proposals_kernel[blocks, threads](
            d_users,
            d_services,
            d_user_to_slot,
            self._d_user_factors,
            self._d_user_bias,
            self._d_user_error,
            local_factors,
            local_bias,
            local_error,
            proposal_factors,
            proposal_bias,
            proposal_error,
            proposal_clients,
            proposal_services,
        )
        factor_scales = cuda.device_array(users.size, dtype=np.float32)
        scalar_scales = cuda.device_array(users.size, dtype=np.float32)
        _quantize_rows_kernel[blocks, threads](
            proposal_factors,
            proposal_bias,
            proposal_error,
            factor_scales,
            scalar_scales,
        )
        cuda.synchronize()
        proposal_quantization_seconds = time.perf_counter() - started

        touched_services = np.unique(services).astype(np.int64)
        service_offsets = [0]
        proposal_order: list[int] = []
        for service in touched_services.tolist():
            indices = np.flatnonzero(services == int(service))
            client_order = np.argsort(users[indices], kind="mergesort")
            indices = indices[client_order]
            proposal_order.extend(int(value) for value in indices.tolist())
            service_offsets.append(len(proposal_order))
        d_touched_services = self._upload(touched_services)
        d_service_offsets = self._upload(
            np.asarray(service_offsets, dtype=np.int32)
        )
        d_proposal_order = self._upload(
            np.asarray(proposal_order, dtype=np.int32)
        )
        fallback_flags = cuda.device_array(
            touched_services.size, dtype=np.int8
        )
        applied_scales = cuda.device_array(
            touched_services.size, dtype=np.float64
        )
        committed_norms = cuda.device_array(
            touched_services.size, dtype=np.float64
        )
        started = time.perf_counter()
        blocks_services, threads_services = self._grid(touched_services.size)
        if self.use_pcs:
            _pcs_aggregate_kernel[blocks_services, threads_services](
                d_touched_services,
                d_service_offsets,
                d_proposal_order,
                proposal_factors,
                proposal_bias,
                proposal_error,
                self._d_service_factors,
                self._d_service_bias,
                self._d_service_error,
                fallback_flags,
                applied_scales,
                committed_norms,
            )
        else:
            _fedavg_aggregate_kernel[blocks_services, threads_services](
                d_touched_services,
                d_service_offsets,
                d_proposal_order,
                proposal_factors,
                proposal_bias,
                proposal_error,
                self._d_service_factors,
                self._d_service_bias,
                self._d_service_error,
            )
        cuda.synchronize()
        aggregation_seconds = time.perf_counter() - started

        aggregate_factor_scales = cuda.device_array(
            touched_services.size, dtype=np.float32
        )
        aggregate_scalar_scales = cuda.device_array(
            touched_services.size, dtype=np.float32
        )
        started = time.perf_counter()
        _quantize_service_rows_kernel[blocks_services, threads_services](
            d_touched_services,
            self._d_service_factors,
            self._d_service_bias,
            self._d_service_error,
            aggregate_factor_scales,
            aggregate_scalar_scales,
        )
        cuda.synchronize()
        aggregate_quantization_seconds = time.perf_counter() - started

        if self.use_carw_srr:
            self._d_history, self._d_new_history = (
                self._d_new_history,
                self._d_history,
            )
            self._d_history_length, self._d_new_history_length = (
                self._d_new_history_length,
                self._d_history_length,
            )
        cuda.synchronize()
        self._observe_memory()

        active_host = tuple(int(value) for value in active.tolist())
        proposal_service_ids = proposal_services.copy_to_host()
        proposals = tuple(
            SimpleNamespace(
                client_id=int(users[index]),
                service_id=int(proposal_service_ids[index]),
            )
            for index in range(users.size)
        )
        from .federated_residual import CommunicationReport

        communication = CommunicationReport(
            active_clients=active_host,
            support_counts=tuple(
                (int(user), int(np.count_nonzero(users == user)))
                for user in active_host
            ),
            proposal_count=int(users.size),
            uplink_bytes=int(
                len(active_host) * 4 + users.size * self.proposal_payload_bytes
            ),
            downlink_bytes=int(
                len(active_host) * self.num_services * 8 * (self.rank + 2)
            ),
        )
        diagnostics: dict[str, Any] = {
            "variant": self.variant,
            "variant_bits": self.variant_bits,
            "use_cpsa": self.use_cpsa,
            "use_pcs": self.use_pcs,
            "use_carw_srr": self.use_carw_srr,
            "active_clients": list(active_host),
            "proposal_count": int(users.size),
            "touched_services": int(touched_services.size),
            "query_or_future_used": False,
        }
        if current_rms is not None:
            diagnostics["carw_current_rms"] = [
                float(value) for value in current_rms.copy_to_host().tolist()
            ]
        if gains is not None:
            diagnostics["carw_gains"] = [
                float(value) for value in gains.copy_to_host().tolist()
            ]
        diagnostics.update({
            "pcs_fallback_count": int(
                np.count_nonzero(fallback_flags.copy_to_host())
            ) if self.use_pcs else 0,
            "pcs_applied_scales": [
                float(value) for value in applied_scales.copy_to_host().tolist()
            ] if self.use_pcs else [],
            "proposal_factor_scales": [
                float(value) for value in factor_scales.copy_to_host().tolist()
            ],
            "aggregate_factor_scales": [
                float(value)
                for value in aggregate_factor_scales.copy_to_host().tolist()
            ],
        })
        self.last_diagnostics = diagnostics
        self.last_profile = GPUAblationAdvanceProfile(
            support_count=int(users.size),
            active_clients=active_count,
            input_upload_seconds=float(input_upload_seconds),
            cpsa_seconds=float(cpsa_seconds),
            shared_broadcast_seconds=float(shared_broadcast_seconds),
            local_update_seconds=float(local_update_seconds),
            carw_srr_seconds=float(carw_srr_seconds),
            proposal_quantization_seconds=float(proposal_quantization_seconds),
            aggregation_seconds=float(aggregation_seconds),
            aggregate_quantization_seconds=float(aggregate_quantization_seconds),
            total_seconds=float(time.perf_counter() - started_total),
            peak_device_memory_used_bytes=int(self._peak_memory),
        )
        return SimpleNamespace(
            active_clients=active_host,
            proposals=proposals,
            communication=communication,
            service_aggregation_diagnostics=(),
        )


def state_hash(model: GPUAblationModel, *, full: bool = True) -> str:
                                                                    

    snapshot = model.full_state_snapshot() if full else model.parameter_snapshot()
    digest = hashlib.sha256()
    for name, raw in sorted(snapshot.items()):
        value = np.asarray(raw)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def state_difference(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> dict[str, Any]:
                                                                          

    result: dict[str, Any] = {}
    for name in sorted(set(reference).union(candidate)):
        if name not in reference or name not in candidate:
            result[name] = {"missing": True}
            continue
        left = np.asarray(reference[name], dtype=np.float64)
        right = np.asarray(candidate[name], dtype=np.float64)
        if left.shape != right.shape:
            result[name] = {
                "shape_mismatch": [list(left.shape), list(right.shape)]
            }
            continue
        delta = np.abs(right - left)
        result[name] = {
            "max_abs": float(np.max(delta)) if delta.size else 0.0,
            "mean_abs": float(np.mean(delta, dtype=np.float64)) if delta.size else 0.0,
            "count": int(delta.size),
        }
    return result


                                                                           
GPUAblationStackModel = GPUAblationModel


__all__ = [
    "CARW_DELTA",
    "CARW_MIN_OBSERVATIONS",
    "CARW_MIN_SEGMENT",
    "CARW_RANGE",
    "CARW_WINDOW",
    "CPSA_DECAY",
    "CPSA_GAIN",
    "CPSA_L2",
    "EXECUTION_MODE",
    "GPUAblationAdvanceProfile",
    "GPUAblationModel",
    "GPUAblationStackModel",
    "GPU_BACKEND",
    "MAX_RANK",
    "VARIANT_BITS",
    "VARIANT_NAMES",
    "VARIANT_SPECS",
    "VariantSpec",
    "cuda_available",
    "cuda_device_info",
    "prediction_metrics",
    "resolve_variant",
    "state_difference",
    "state_hash",
]

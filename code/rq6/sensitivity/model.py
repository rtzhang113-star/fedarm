                                                                    

                                                                            
                                                                             
                                                                          
                                                                            
                                                                       

                                                                            
                                                                             
                                                                         
                                                                            
                                     
   

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from continual_qos import gpu_ablation_training as _base

try:                                                                          
    from .kernels import (
        MAX_PROPOSALS_PER_SERVICE,
        MAX_RANK,
        cpsa_update_kernel,
        finite_state_kernel,
        pcs_aggregate_kernel,
    )
except ImportError:                                                        
    from kernels import (                          
        MAX_PROPOSALS_PER_SERVICE,
        MAX_RANK,
        cpsa_update_kernel,
        finite_state_kernel,
        pcs_aggregate_kernel,
    )


cuda = _base.cuda
CARW_WINDOW = _base.CARW_WINDOW
GPUAblationAdvanceProfile = _base.GPUAblationAdvanceProfile

RQ6_VARIANTS = ("full",)
RQ6_VARIANT_BITS = {"full": "111"}
CPSA_GAIN_DEFAULT = 0.1
CPSA_L2 = 0.0003
TRUST_EXPONENT = 0.5
MIN_GAIN = 0.0
MAX_GAIN = 1.0
MIN_COUNT_EXPONENT = 0.0
MAX_COUNT_EXPONENT = 2.0


def _validate_integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or number != math.floor(number):
        raise ValueError(f"{name} must be an integer")
    result = int(number)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _validate_finite_range(
    value: Any,
    name: str,
    lower: float,
    upper: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [{lower:g}, {upper:g}]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [{lower:g}, {upper:g}]") from exc
    if not math.isfinite(result) or result < lower or result > upper:
        raise ValueError(f"{name} must be finite and in [{lower:g}, {upper:g}]")
    return result


def resolve_sensitivity_parameters(
    *, cpsa_gain: Any = CPSA_GAIN_DEFAULT, count_exponent: Any = 1.0
) -> dict[str, float]:
                                                                             

    gain = _validate_finite_range(cpsa_gain, "cpsa_gain", MIN_GAIN, MAX_GAIN)
    count = _validate_finite_range(
        count_exponent,
        "count_exponent",
        MIN_COUNT_EXPONENT,
        MAX_COUNT_EXPONENT,
    )
    decay = 1.0 - gain * CPSA_L2
    if not math.isfinite(decay):                                                
        raise FloatingPointError("resolved CPSA decay is non-finite")
    return {
        "cpsa_gain": float(gain),
        "cpsa_l2": float(CPSA_L2),
        "cpsa_decay": float(decay),
        "count_exponent": float(count),
        "trust_exponent": float(TRUST_EXPONENT),
    }


@dataclass(frozen=True)
class GPUSensitivityAdvanceProfile(GPUAblationAdvanceProfile):
                                                                     

    resolved_parameters: Mapping[str, float]
    pcs_applied_scale_count: int
    pcs_applied_scale_min: float
    pcs_applied_scale_max: float
    pcs_applied_scale_mean: float
    pcs_count_scale_min: float
    pcs_count_scale_max: float
    pcs_trust_clip_count: int
    pcs_fallback_count: int
    state_numeric_finite: bool

    def as_dict(self) -> dict[str, Any]:
        result = super().as_dict()
        result.update(
            {
                "resolved_parameters": {
                    str(key): float(value)
                    for key, value in self.resolved_parameters.items()
                },
                "pcs_applied_scale_count": int(self.pcs_applied_scale_count),
                "pcs_applied_scale_min": float(self.pcs_applied_scale_min),
                "pcs_applied_scale_max": float(self.pcs_applied_scale_max),
                "pcs_applied_scale_mean": float(self.pcs_applied_scale_mean),
                "pcs_count_scale_min": float(self.pcs_count_scale_min),
                "pcs_count_scale_max": float(self.pcs_count_scale_max),
                "pcs_trust_clip_count": int(self.pcs_trust_clip_count),
                "pcs_fallback_count": int(self.pcs_fallback_count),
                "state_numeric_finite": bool(self.state_numeric_finite),
            }
        )
        return result


class GPUSensitivityModel(_base.GPUAblationModel):
                                                                  

                                                   
                                                                            
                                                                           
                                                 
       

    model_name = "RQ6-Sensitivity-FedARM-GPU"
    aggregation_algorithm = "RQ6 parameterized PCS/CPSA FedARM on CUDA"

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
        cpsa_gain: float = CPSA_GAIN_DEFAULT,
        count_exponent: float = 1.0,
    ) -> None:
        key = str(variant).strip().lower()
        if key not in RQ6_VARIANTS:
            raise ValueError(
                f"unknown RQ6 sensitivity variant {variant!r}; expected full"
            )
        resolved = resolve_sensitivity_parameters(
            cpsa_gain=cpsa_gain, count_exponent=count_exponent
        )
        rank_value = _validate_integer(rank, "rank", minimum=1)
        if rank_value > MAX_RANK:
            raise ValueError(f"rank must be in [1, {MAX_RANK}]")

                                                                               
                                                                            
                                                  
        super().__init__(
            num_users=num_users,
            num_services=num_services,
            rank=rank_value,
            seed=seed,
            bias_eta=bias_eta,
            factor_eta=factor_eta,
            l2=l2,
            beta=beta,
            factor_init_std=factor_init_std,
            num_edges=num_edges,
            variant="full",
        )

        self.variant = key
        self.variant_bits = RQ6_VARIANT_BITS[key]
        self.use_cpsa = True
        self.use_pcs = True
        self.use_carw_srr = True
        self.cpsa_gain = float(resolved["cpsa_gain"])
        self.cpsa_l2 = float(resolved["cpsa_l2"])
        self.cpsa_decay = float(resolved["cpsa_decay"])
        self.count_exponent = float(resolved["count_exponent"])
        self.trust_exponent = float(resolved["trust_exponent"])

                                                                            
                                                                              
                                                      
        assert cuda is not None
        self._d_history = cuda.to_device(
            np.zeros((self.num_users, CARW_WINDOW + 1), dtype=np.float64)
        )
        self._d_new_history = cuda.device_array(
            (self.num_users, CARW_WINDOW + 1), dtype=np.float64
        )

        self.last_diagnostics.update(
            {
                "variant": self.variant,
                "variant_bits": self.variant_bits,
                "resolved_parameters": self.resolved_parameters,
                "query_or_future_used": False,
            }
        )

    @property
    def resolved_parameters(self) -> dict[str, float]:
                                                                 

        return {
            "rank": float(self.rank),
            "cpsa_gain": float(self.cpsa_gain),
            "cpsa_l2": float(self.cpsa_l2),
            "cpsa_decay": float(self.cpsa_decay),
            "count_exponent": float(self.count_exponent),
            "trust_exponent": float(self.trust_exponent),
        }

    def _state_numeric_finite(self) -> bool:
                                                                                 

                                                                            
                                                                        
                                                                            
                                                          
           

        if cuda is None or finite_state_kernel is None:
            return False
        flag = cuda.to_device(np.asarray([1], dtype=np.int8))
        arrays = (
            self._d_user_factors,
            self._d_service_factors,
            self._d_user_bias,
            self._d_service_bias,
            self._d_user_error,
            self._d_service_error,
            self._d_adapter,
        )
        for values in arrays:
            size = int(values.size)
            if size <= 0:
                continue
            blocks, threads = self._grid(size)
            finite_state_kernel[blocks, threads](values, flag)
        cuda.synchronize()
        if not bool(int(flag.copy_to_host()[0])):
            return False
        history = self._d_history.copy_to_host()
        lengths = self._d_history_length.copy_to_host()
        if np.any(lengths < 0) or np.any(lengths > CARW_WINDOW):
            return False
        for user, length in enumerate(lengths.tolist()):
            if length and not np.all(np.isfinite(history[user, : int(length)])):
                return False
        return True

    @staticmethod
    def _summary(values: np.ndarray) -> tuple[int, float, float, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return 0, 0.0, 0.0, 0.0
        return (
            int(array.size),
            float(np.min(array)),
            float(np.max(array)),
            float(np.mean(array, dtype=np.float64)),
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
                                                                             

        if cuda is None or cpsa_update_kernel is None or pcs_aggregate_kernel is None:
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

        started = time.perf_counter()
        blocks, threads = self._grid(users.size)
        cpsa_update_kernel[blocks, threads](
            d_users,
            d_services,
            d_values,
            raw_cap,
            self._d_user_factors,
            self._d_service_factors,
            self._d_user_bias,
            self._d_service_bias,
            self._d_adapter,
            self.cpsa_gain,
            self.cpsa_decay,
        )
        cuda.synchronize()
        cpsa_seconds = time.perf_counter() - started

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
        _base._broadcast_shared_kernel[blocks, threads](
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
        _base._fit_local_kernel[active_count, 1](
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

        residuals = cuda.device_array(users.size, dtype=np.float64)
        current_rms = cuda.device_array(active_count, dtype=np.float64)
        sums = cuda.device_array(
            (active_count, self.num_services), dtype=np.float64
        )
        counts = cuda.device_array(
            (active_count, self.num_services), dtype=np.int32
        )
        blocks, threads = self._grid(active_count * self.num_services)
        _base._zero_float_kernel[blocks, threads](sums)
        _base._zero_int_kernel[blocks, threads](counts)
        started = time.perf_counter()
        _base._residual_stats_kernel[active_count, 1](
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
        _base._carw_srr_kernel[active_count, 1](
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

        proposal_factors = cuda.device_array(
            (users.size, self.rank), dtype=np.float64
        )
        proposal_bias = cuda.device_array(users.size, dtype=np.float64)
        proposal_error = cuda.device_array(users.size, dtype=np.float64)
        proposal_clients = cuda.device_array(users.size, dtype=np.int64)
        proposal_services = cuda.device_array(users.size, dtype=np.int64)
        started = time.perf_counter()
        blocks, threads = self._grid(users.size)
        _base._gather_proposals_kernel[blocks, threads](
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
        _base._quantize_rows_kernel[blocks, threads](
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
        count_scales = cuda.device_array(
            touched_services.size, dtype=np.float64
        )
        raw_delta_norms = cuda.device_array(
            touched_services.size, dtype=np.float64
        )
        trust_norms = cuda.device_array(
            touched_services.size, dtype=np.float64
        )
        clip_flags = cuda.device_array(
            touched_services.size, dtype=np.int8
        )
        committed_norms = cuda.device_array(
            touched_services.size, dtype=np.float64
        )
        started = time.perf_counter()
        blocks_services, threads_services = self._grid(touched_services.size)
        pcs_aggregate_kernel[blocks_services, threads_services](
            d_touched_services,
            d_service_offsets,
            d_proposal_order,
            proposal_factors,
            proposal_bias,
            proposal_error,
            self._d_service_factors,
            self._d_service_bias,
            self._d_service_error,
            self.count_exponent,
            self.trust_exponent,
            fallback_flags,
            applied_scales,
            count_scales,
            raw_delta_norms,
            trust_norms,
            clip_flags,
            committed_norms,
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
        _base._quantize_service_rows_kernel[blocks_services, threads_services](
            d_touched_services,
            self._d_service_factors,
            self._d_service_bias,
            self._d_service_error,
            aggregate_factor_scales,
            aggregate_scalar_scales,
        )
        cuda.synchronize()
        aggregate_quantization_seconds = time.perf_counter() - started

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

        fallback_host = fallback_flags.copy_to_host()
        applied_host = applied_scales.copy_to_host()
        count_scale_host = count_scales.copy_to_host()
        raw_norm_host = raw_delta_norms.copy_to_host()
        trust_norm_host = trust_norms.copy_to_host()
        clip_host = clip_flags.copy_to_host()
        applied_count, applied_min, applied_max, applied_mean = self._summary(
            applied_host
        )
        _, count_min, count_max, _ = self._summary(count_scale_host)
        pcs_fallback_count = int(np.count_nonzero(fallback_host))
        pcs_trust_clip_count = int(np.count_nonzero(clip_host))
        state_numeric_finite = self._state_numeric_finite()
        active_host = tuple(int(value) for value in active.tolist())
        proposal_service_ids = proposal_services.copy_to_host()
        proposals = tuple(
            SimpleNamespace(
                client_id=int(users[index]),
                service_id=int(proposal_service_ids[index]),
            )
            for index in range(users.size)
        )
        from continual_qos.federated_residual import CommunicationReport

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
        resolved = self.resolved_parameters
        diagnostics: dict[str, Any] = {
            "variant": self.variant,
            "variant_bits": self.variant_bits,
            "use_cpsa": True,
            "use_pcs": True,
            "use_carw_srr": True,
            "resolved_parameters": resolved,
            "active_clients": list(active_host),
            "proposal_count": int(users.size),
            "touched_services": int(touched_services.size),
            "pcs_fallback_count": pcs_fallback_count,
            "pcs_trust_clip_count": pcs_trust_clip_count,
                                                                          
                                                                             
                                                                     
            "pcs_applied_scale_summary": {
                "count": int(applied_count),
                "min": float(applied_min),
                "max": float(applied_max),
                "mean": float(applied_mean),
            },
            "pcs_count_scale_summary": {
                "count": int(count_scale_host.size),
                "min": float(count_min),
                "max": float(count_max),
                "mean": float(np.mean(count_scale_host, dtype=np.float64))
                if count_scale_host.size
                else 0.0,
            },
            "pcs_raw_delta_norm_summary": {
                "count": int(raw_norm_host.size),
                "min": float(np.min(raw_norm_host)) if raw_norm_host.size else 0.0,
                "max": float(np.max(raw_norm_host)) if raw_norm_host.size else 0.0,
                "mean": float(np.mean(raw_norm_host, dtype=np.float64))
                if raw_norm_host.size
                else 0.0,
            },
            "pcs_trust_norm_summary": {
                "count": int(trust_norm_host.size),
                "min": float(np.min(trust_norm_host)) if trust_norm_host.size else 0.0,
                "max": float(np.max(trust_norm_host)) if trust_norm_host.size else 0.0,
                "mean": float(np.mean(trust_norm_host, dtype=np.float64))
                if trust_norm_host.size
                else 0.0,
            },
            "state_numeric_finite": bool(state_numeric_finite),
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
        self.last_diagnostics = diagnostics
        self.last_profile = GPUSensitivityAdvanceProfile(
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
            resolved_parameters=resolved,
            pcs_applied_scale_count=applied_count,
            pcs_applied_scale_min=applied_min,
            pcs_applied_scale_max=applied_max,
            pcs_applied_scale_mean=applied_mean,
            pcs_count_scale_min=count_min,
            pcs_count_scale_max=count_max,
            pcs_trust_clip_count=pcs_trust_clip_count,
            pcs_fallback_count=pcs_fallback_count,
            state_numeric_finite=state_numeric_finite,
        )
        return SimpleNamespace(
            active_clients=active_host,
            proposals=proposals,
            communication=communication,
            service_aggregation_diagnostics=(),
            profile=self.last_profile,
        )


                                                                         
                                                                             
                    
GPURobustnessModel = GPUSensitivityModel


def state_hash(model: GPUSensitivityModel, *, full: bool = True) -> str:
                                                                              

    return _base.state_hash(model, full=full)


def prediction_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
                                                                        

    return _base.prediction_metrics(truth, prediction)


__all__ = [
    "CARW_WINDOW",
    "CPSA_GAIN_DEFAULT",
    "CPSA_L2",
    "GPUAblationAdvanceProfile",
    "GPUSensitivityAdvanceProfile",
    "GPUSensitivityModel",
    "GPURobustnessModel",
    "MAX_COUNT_EXPONENT",
    "MAX_GAIN",
    "MAX_RANK",
    "MIN_COUNT_EXPONENT",
    "MIN_GAIN",
    "RQ6_VARIANT_BITS",
    "RQ6_VARIANTS",
    "TRUST_EXPONENT",
    "cuda_available",
    "cuda_device_info",
    "prediction_metrics",
    "resolve_sensitivity_parameters",
    "state_difference",
    "state_hash",
]


cuda_available = _base.cuda_available
cuda_device_info = _base.cuda_device_info
state_difference = _base.state_difference

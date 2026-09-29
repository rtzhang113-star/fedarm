                                                                 

                                                                         
                                                                               
                                                                           
                                                                            
                                                                             
             
   

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    from numba import cuda
except Exception:                                                             
    cuda = None                            


GPU_BACKEND = "numba-cuda"
EXECUTION_MODE = "numba_cuda_training_and_prediction"
SUPPORT_INITIALIZATION = "slice0_support_mean_global_intercept"
DEFAULT_INITIALIZATION_CLIP_EPSILON = 1e-4


def cuda_available() -> bool:
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def cuda_device_info() -> dict[str, Any]:
    if not cuda_available():
        return {"backend": GPU_BACKEND, "available": False, "name": None}
    assert cuda is not None
    try:
        device = cuda.get_current_device()
        raw_name = getattr(device, "name", None)
        name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        capability = getattr(device, "compute_capability", None)
        return {
            "backend": GPU_BACKEND,
            "available": True,
            "name": name,
            "compute_capability": None if capability is None else [int(v) for v in capability],
        }
    except Exception as exc:                                       
        return {
            "backend": GPU_BACKEND,
            "available": False,
            "name": None,
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


@dataclass(frozen=True)
class GPUProfile:
    update_kernel_seconds: float = 0.0
    prediction_kernel_seconds: float = 0.0
    host_to_device_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    device_memory_used_bytes: int = 0
    peak_device_memory_used_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "update_kernel_seconds": float(self.update_kernel_seconds),
            "prediction_kernel_seconds": float(self.prediction_kernel_seconds),
            "host_to_device_seconds": float(self.host_to_device_seconds),
            "device_to_host_seconds": float(self.device_to_host_seconds),
            "device_memory_used_bytes": int(self.device_memory_used_bytes),
            "peak_device_memory_used_bytes": int(self.peak_device_memory_used_bytes),
        }


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _sigmoid(value: float) -> float:
        if value < -30.0:
            value = -30.0
        elif value > 30.0:
            value = 30.0
        return 1.0 / (1.0 + math.exp(-value))


    @cuda.jit
    def _zero_1d(values: Any) -> None:
        index = cuda.grid(1)
        if index < values.size:
            values[index] = 0


    @cuda.jit
    def _zero_2d(values: Any) -> None:
        index = cuda.grid(1)
        if index < values.size:
            columns = values.shape[1]
            values[index // columns, index % columns] = 0


    @cuda.jit
    def _zero_3d(values: Any) -> None:
        index = cuda.grid(1)
        if index < values.size:
            middle = values.shape[1]
            last = values.shape[2]
            first_index = index // (middle * last)
            remainder = index - first_index * middle * last
            values[first_index, remainder // last, remainder % last] = 0


    @cuda.jit
    def _fed_bootstrap_bias_kernel(user_biases: Any, service_biases: Any, bias_value: float) -> None:
                                                                            
        index = cuda.grid(1)
        if index < user_biases.size:
            user_biases[index] = bias_value
        if index < service_biases.size:
            service_biases[index] = bias_value


    @cuda.jit
    def _fed_local_kernel(
        active_users: Any,
        offsets: Any,
        services: Any,
        targets: Any,
        user_factors: Any,
        service_factors: Any,
        user_biases: Any,
        service_biases: Any,
        user_delta: Any,
        service_delta: Any,
        user_bias_delta: Any,
        service_bias_delta: Any,
        learning_rate: float,
        l2: float,
        mu: float,
        local_epochs: int,
    ) -> None:
        slot = cuda.grid(1)
        if slot >= active_users.size:
            return
        user = active_users[slot]
        for _ in range(local_epochs):
            for position in range(offsets[slot], offsets[slot + 1]):
                service = services[position]
                logit = user_biases[user] + user_bias_delta[slot]
                logit += service_biases[service] + service_bias_delta[slot, service]
                for factor in range(user_factors.shape[1]):
                    local_user = user_factors[user, factor] + user_delta[slot, factor]
                    local_service = service_factors[service, factor] + service_delta[slot, service, factor]
                    logit += local_user * local_service
                prediction = _sigmoid(logit)
                gradient = (prediction - targets[position]) * prediction * (1.0 - prediction)
                for factor in range(user_factors.shape[1]):
                    before_user = user_factors[user, factor] + user_delta[slot, factor]
                    before_service = service_factors[service, factor] + service_delta[slot, service, factor]
                    prox_user = mu * user_delta[slot, factor]
                    prox_service = mu * service_delta[slot, service, factor]
                    user_delta[slot, factor] -= learning_rate * (
                        gradient * before_service + l2 * before_user + prox_user
                    )
                    service_delta[slot, service, factor] -= learning_rate * (
                        gradient * before_user + l2 * before_service + prox_service
                    )
                before_user_bias = user_biases[user] + user_bias_delta[slot]
                before_service_bias = service_biases[service] + service_bias_delta[slot, service]
                user_bias_delta[slot] -= learning_rate * (
                    gradient + l2 * before_user_bias + mu * user_bias_delta[slot]
                )
                service_bias_delta[slot, service] -= learning_rate * (
                    gradient + l2 * before_service_bias + mu * service_bias_delta[slot, service]
                )


    @cuda.jit
    def _fed_aggregate_users(
        active_users: Any,
        counts: Any,
        total_count: int,
        user_delta: Any,
        user_bias_delta: Any,
        user_factors: Any,
        user_biases: Any,
    ) -> None:
        slot = cuda.grid(1)
        if slot >= active_users.size:
            return
        user = active_users[slot]
        weight = counts[slot] / total_count
        for factor in range(user_factors.shape[1]):
            user_factors[user, factor] += weight * user_delta[slot, factor]
        user_biases[user] += weight * user_bias_delta[slot]


    @cuda.jit
    def _fed_aggregate_services(
        counts: Any,
        total_count: int,
        service_delta: Any,
        service_factors: Any,
    ) -> None:
        index = cuda.grid(1)
        if index >= service_factors.size:
            return
        rank = service_factors.shape[1]
        service = index // rank
        factor = index - service * rank
        shift = 0.0
        for slot in range(counts.size):
            shift += (counts[slot] / total_count) * service_delta[slot, service, factor]
        service_factors[service, factor] += shift


    @cuda.jit
    def _fed_aggregate_service_bias(
        counts: Any,
        total_count: int,
        service_bias_delta: Any,
        service_biases: Any,
    ) -> None:
        service = cuda.grid(1)
        if service >= service_biases.size:
            return
        shift = 0.0
        for slot in range(counts.size):
            shift += (counts[slot] / total_count) * service_bias_delta[slot, service]
        service_biases[service] += shift


    @cuda.jit
    def _fed_predict_kernel(
        users: Any,
        services: Any,
        user_factors: Any,
        service_factors: Any,
        user_biases: Any,
        service_biases: Any,
        log1p_cap: float,
        output: Any,
    ) -> None:
        index = cuda.grid(1)
        if index >= users.size:
            return
        user = users[index]
        service = services[index]
        logit = user_biases[user] + service_biases[service]
        for factor in range(user_factors.shape[1]):
            logit += user_factors[user, factor] * service_factors[service, factor]
        output[index] = math.expm1(_sigmoid(logit) * log1p_cap)


    @cuda.jit
    def _perfed_local_kernel(
        active_clients: Any,
        offsets: Any,
        users: Any,
        services: Any,
        targets: Any,
        time_slice: int,
        private_user: Any,
        local_service: Any,
        local_time: Any,
        global_service: Any,
        global_time: Any,
        learning_rate: float,
        gamma: float,
        l2: float,
        local_epochs: int,
    ) -> None:
        slot = cuda.grid(1)
        if slot >= active_clients.size:
            return
        client = active_clients[slot]
        for _ in range(local_epochs):
            for position in range(offsets[slot], offsets[slot + 1]):
                user = users[position]
                service = services[position]
                prediction = 0.0
                for factor in range(private_user.shape[1]):
                    prediction += (
                        private_user[user, factor]
                        * local_service[client, service, factor]
                        * local_time[client, time_slice, factor]
                    )
                error = prediction - targets[position]
                for factor in range(private_user.shape[1]):
                    before_user = private_user[user, factor]
                    before_service = local_service[client, service, factor]
                    before_time = local_time[client, time_slice, factor]
                    private_user[user, factor] = before_user - learning_rate * (
                        error * before_service * before_time + l2 * before_user
                    )
                    local_service[client, service, factor] = before_service - learning_rate * (
                        error * before_user * before_time
                        + gamma * (before_service - global_service[service, factor])
                        + l2 * before_service
                    )
                    local_time[client, time_slice, factor] = before_time - learning_rate * (
                        error * before_user * before_service
                        + gamma * (before_time - global_time[time_slice, factor])
                        + l2 * before_time
                    )


    @cuda.jit
    def _perfed_center_service(
        active_clients: Any,
        counts: Any,
        total_count: int,
        local_service: Any,
        global_service: Any,
        coupling_step: float,
    ) -> None:
        index = cuda.grid(1)
        if index >= global_service.size:
            return
        rank = global_service.shape[1]
        service = index // rank
        factor = index - service * rank
        center = 0.0
        for slot in range(active_clients.size):
            center += (counts[slot] / total_count) * local_service[active_clients[slot], service, factor]
        global_service[service, factor] += coupling_step * (center - global_service[service, factor])


    @cuda.jit
    def _perfed_center_time(
        active_clients: Any,
        counts: Any,
        total_count: int,
        local_time: Any,
        global_time: Any,
        coupling_step: float,
    ) -> None:
        index = cuda.grid(1)
        if index >= global_time.size:
            return
        rank = global_time.shape[1]
        time_index = index // rank
        factor = index - time_index * rank
        center = 0.0
        for slot in range(active_clients.size):
            center += (counts[slot] / total_count) * local_time[active_clients[slot], time_index, factor]
        global_time[time_index, factor] += coupling_step * (center - global_time[time_index, factor])


    @cuda.jit
    def _perfed_predict_kernel(
        users: Any,
        services: Any,
        time_slice: int,
        user_clients: Any,
        private_user: Any,
        local_service: Any,
        local_time: Any,
        log1p_cap: float,
        output: Any,
    ) -> None:
        index = cuda.grid(1)
        if index >= users.size:
            return
        user = users[index]
        service = services[index]
        client = user_clients[user]
        normalized = 0.0
        for factor in range(private_user.shape[1]):
            normalized += (
                private_user[user, factor]
                * local_service[client, service, factor]
                * local_time[client, time_slice, factor]
            )
        if normalized < 0.0:
            normalized = 0.0
        elif normalized > 1.0:
            normalized = 1.0
        output[index] = math.expm1(normalized * log1p_cap)


    @cuda.jit
    def _fl_lft_update_kernel(
        users: Any,
        services: Any,
        targets: Any,
        time_slice: int,
        private_user: Any,
        shared_service: Any,
        shared_time: Any,
        learning_rate: float,
        regularization: float,
        local_epochs: int,
    ) -> None:
        if cuda.grid(1) != 0:
            return
        for _ in range(local_epochs):
            for position in range(users.size):
                user = users[position]
                service = services[position]
                prediction = 0.0
                for factor in range(private_user.shape[1]):
                    prediction += (
                        private_user[user, factor]
                        * shared_service[service, factor]
                        * shared_time[time_slice, factor]
                    )
                error = prediction - targets[position]
                for factor in range(private_user.shape[1]):
                    before_user = private_user[user, factor]
                    before_service = shared_service[service, factor]
                    before_time = shared_time[time_slice, factor]
                    updated_user = before_user - learning_rate * (
                        error * before_service * before_time + regularization * before_user
                    )
                    private_user[user, factor] = updated_user
                    shared_service[service, factor] = before_service - learning_rate * (
                        error * updated_user * before_time + regularization * before_service
                    )
                    shared_time[time_slice, factor] = before_time - learning_rate * (
                        error * updated_user * before_service + regularization * before_time
                    )


    @cuda.jit
    def _fl_lft_predict_kernel(
        users: Any,
        services: Any,
        time_slice: int,
        private_user: Any,
        shared_service: Any,
        shared_time: Any,
        log1p_cap: float,
        output: Any,
    ) -> None:
        index = cuda.grid(1)
        if index >= users.size:
            return
        normalized = 0.0
        user = users[index]
        service = services[index]
        for factor in range(private_user.shape[1]):
            normalized += (
                private_user[user, factor]
                * shared_service[service, factor]
                * shared_time[time_slice, factor]
            )
        if normalized < 0.0:
            normalized = 0.0
        elif normalized > 1.0:
            normalized = 1.0
        output[index] = math.expm1(normalized * log1p_cap)

else:                    
    _zero_1d = _zero_2d = _zero_3d = None
    _fed_bootstrap_bias_kernel = None
    _fed_local_kernel = _fed_aggregate_users = _fed_aggregate_services = None
    _fed_aggregate_service_bias = _fed_predict_kernel = None
    _perfed_local_kernel = _perfed_center_service = _perfed_center_time = None
    _perfed_predict_kernel = _fl_lft_update_kernel = _fl_lft_predict_kernel = None


class _GPUModelBase:
    def __init__(self) -> None:
        if not cuda_available():
            raise RuntimeError("a usable Numba CUDA device is required")
        self._last_profile = GPUProfile()
        self._last_communication: dict[str, int] = {}
        self._peak_memory = 0

    @property
    def last_profile(self) -> GPUProfile:
        return self._last_profile

    def communication(self) -> dict[str, int]:
        return dict(self._last_communication)

    def _memory_used(self) -> int:
        assert cuda is not None
        try:
            free, total = cuda.current_context().get_memory_info()
            used = int(total - free)
        except Exception:
            used = 0
        self._peak_memory = max(self._peak_memory, used)
        return used

    @staticmethod
    def _validate_indices(users: Any, services: Any, values: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
                                                                          
                                                                        
                                                                           
        users_array = np.ascontiguousarray(users, dtype=np.int64)
        services_array = np.ascontiguousarray(services, dtype=np.int64)
        if users_array.ndim != 1 or services_array.shape != users_array.shape:
            raise ValueError("users and services must be aligned one-dimensional arrays")
        values_array = None
        if values is not None:
            values_array = np.ascontiguousarray(values, dtype=np.float32)
            if values_array.shape != users_array.shape or not np.all(np.isfinite(values_array)):
                raise ValueError("support values must be finite and aligned")
        return users_array, services_array, values_array

    @staticmethod
    def _normalize(values: np.ndarray, raw_cap: float, dtype: Any = np.float32) -> np.ndarray:
        if not np.isfinite(raw_cap) or raw_cap <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        if np.any(values <= 0.0):
            raise ValueError("support values must be positive")
                                                                         
                                                                          
        values64 = np.asarray(values, dtype=np.float64)
        return (np.log1p(values64) / np.log1p(float(raw_cap))).astype(dtype)


class GPUFedAvgProx(_GPUModelBase):
    def __init__(self, config: Mapping[str, Any], seed: int, mu: float) -> None:
        super().__init__()
        assert cuda is not None
        self.num_users = int(config["num_users"])
        self.num_services = int(config["num_services"])
        self.rank = int(config["rank"])
        self.learning_rate = float(config["learning_rate"])
        self.l2 = float(config["l2"])
        self.local_epochs = int(config["local_epochs"])
        self.mu = float(mu)
        self.support_initialization = str(config.get("support_initialization", ""))
        if self.support_initialization != SUPPORT_INITIALIZATION:
            raise ValueError(
                "FedAvg/FedProx requires support_initialization="
                f"{SUPPORT_INITIALIZATION!r}"
            )
        try:
            configured_epsilon = config["initialization_clip_epsilon"]
        except KeyError as exc:
            raise ValueError(
                "FedAvg/FedProx requires initialization_clip_epsilon"
            ) from exc
        if isinstance(configured_epsilon, bool):
            raise ValueError("initialization_clip_epsilon must be finite and positive")
        try:
            configured_epsilon = float(configured_epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "initialization_clip_epsilon must be finite and positive"
            ) from exc
        if (
            not np.isfinite(configured_epsilon)
            or configured_epsilon <= 0.0
            or configured_epsilon >= 0.5
        ):
            raise ValueError("initialization_clip_epsilon must be in (0, 0.5)")
        self.initialization_clip_epsilon = configured_epsilon
        self._support_intercept_initialized = False
        self._support_intercept_initialization_count = 0
        self._support_intercept_metadata: dict[str, Any] = {
            "initialized": False,
            "source": "support_only_normalized_target_mean",
            "statistic": "support_normalized_mean",
            "policy": self.support_initialization,
            "clip_epsilon": self.initialization_clip_epsilon,
            "support_entries": 0,
            "support_target_mean": None,
            "clipped_target_mean": None,
            "intercept": None,
            "bias_value": None,
            "initialization_uplink_bytes": 0,
            "initialization_participating_clients": 0,
            "initialization_count": 0,
        }
        if min(self.num_users, self.num_services, self.rank, self.local_epochs) <= 0:
            raise ValueError("invalid FedAvg/FedProx dimensions or epochs")
        if self.learning_rate <= 0.0 or self.l2 < 0.0 or self.mu < 0.0:
            raise ValueError("invalid FedAvg/FedProx optimizer settings")
        rng = np.random.default_rng(int(seed))
        started = time.perf_counter()
        self.d_user_factors = cuda.to_device(
            rng.normal(0.0, 0.05, (self.num_users, self.rank)).astype(np.float32)
        )
        self.d_service_factors = cuda.to_device(
            rng.normal(0.0, 0.05, (self.num_services, self.rank)).astype(np.float32)
        )
        self.d_user_biases = cuda.to_device(np.zeros(self.num_users, dtype=np.float32))
        self.d_service_biases = cuda.to_device(np.zeros(self.num_services, dtype=np.float32))
        upload = time.perf_counter() - started
        used = self._memory_used()
        self._last_profile = GPUProfile(host_to_device_seconds=upload, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)

    def initialization_metadata(self) -> dict[str, Any]:
                                                                                      
        return dict(self._support_intercept_metadata)

    def _bootstrap_support_intercept(self, normalized: np.ndarray, participating_clients: int) -> int:
                                                                                  

                                                                              
                                                                            
                                                                          
           
        if self._support_intercept_initialized or normalized.size == 0:
            return 0
        assert cuda is not None and _fed_bootstrap_bias_kernel is not None
        target_mean = float(np.mean(normalized, dtype=np.float64))
        clipped_mean = float(
            np.clip(
                target_mean,
                self.initialization_clip_epsilon,
                1.0 - self.initialization_clip_epsilon,
            )
        )
        intercept = float(np.log(clipped_mean / (1.0 - clipped_mean)))
        bias_value = np.float32(intercept / 2.0)
        threads = 128
        width = max(self.num_users, self.num_services)
        _fed_bootstrap_bias_kernel[(width + threads - 1) // threads, threads](
            self.d_user_biases,
            self.d_service_biases,
            bias_value,
        )
        cuda.synchronize()
        initialization_bytes = int(participating_clients) * (np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize)
        self._support_intercept_initialized = True
        self._support_intercept_initialization_count += 1
        self._support_intercept_metadata = {
            "initialized": True,
            "source": "support_only_normalized_target_mean",
            "statistic": "support_normalized_mean",
            "policy": self.support_initialization,
            "clip_epsilon": self.initialization_clip_epsilon,
            "support_entries": int(normalized.size),
            "support_target_mean": target_mean,
            "clipped_target_mean": clipped_mean,
            "intercept": intercept,
            "bias_value": float(bias_value),
            "initialization_uplink_bytes": initialization_bytes,
            "initialization_participating_clients": int(participating_clients),
            "initialization_count": int(self._support_intercept_initialization_count),
        }
        return initialization_bytes

    @property
    def parameter_count(self) -> int:
        return int(
            self.num_users * self.rank + self.num_services * self.rank
            + self.num_users + self.num_services
        )

    def advance_support(self, users: Any, services: Any, raw_values: Any, time_slice: int, raw_cap: float, round_seed: int) -> dict[str, Any]:
        del time_slice, round_seed
        assert cuda is not None and _fed_local_kernel is not None
        users_array, services_array, values_array = self._validate_indices(users, services, raw_values)
        assert values_array is not None
        if users_array.size == 0:
            self._last_communication = {
                "uplink_bytes": 0,
                "downlink_bytes": 0,
                "total_bytes": 0,
                "initialization_uplink_bytes": 0,
                "participating_clients": 0,
            }
            return {
                "communication": self.communication(),
                "support_entries": 0,
                "initialization": self.initialization_metadata(),
            }
        if np.any(users_array < 0) or np.any(users_array >= self.num_users) or np.any(services_array < 0) or np.any(services_array >= self.num_services):
            raise IndexError("support index out of range")
        order = np.lexsort((services_array, users_array))
        users_array = users_array[order]
        services_array = services_array[order]
        normalized = self._normalize(values_array[order], raw_cap, np.float32)
        active_users, first, counts = np.unique(users_array, return_index=True, return_counts=True)
        offsets = np.concatenate((first, np.asarray([len(users_array)], dtype=np.int64)))

        upload_started = time.perf_counter()
        d_active = cuda.to_device(active_users.astype(np.int64))
        d_offsets = cuda.to_device(offsets.astype(np.int64))
        d_counts = cuda.to_device(counts.astype(np.int64))
        d_services = cuda.to_device(services_array.astype(np.int64))
        d_targets = cuda.to_device(normalized)
        active_count = int(active_users.size)
        d_user_delta = cuda.device_array((active_count, self.rank), dtype=np.float32)
        d_service_delta = cuda.device_array((active_count, self.num_services, self.rank), dtype=np.float32)
        d_user_bias_delta = cuda.device_array(active_count, dtype=np.float32)
        d_service_bias_delta = cuda.device_array((active_count, self.num_services), dtype=np.float32)
        upload_seconds = time.perf_counter() - upload_started
        initialization_uplink_bytes = self._bootstrap_support_intercept(normalized, active_count)

        threads = 128
        _zero_2d[(d_user_delta.size + threads - 1) // threads, threads](d_user_delta)
        _zero_3d[(d_service_delta.size + threads - 1) // threads, threads](d_service_delta)
        _zero_1d[(d_user_bias_delta.size + threads - 1) // threads, threads](d_user_bias_delta)
        _zero_2d[(d_service_bias_delta.size + threads - 1) // threads, threads](d_service_bias_delta)
        cuda.synchronize()
        kernel_started = time.perf_counter()
        _fed_local_kernel[(active_count + threads - 1) // threads, threads](
            d_active, d_offsets, d_services, d_targets,
            self.d_user_factors, self.d_service_factors,
            self.d_user_biases, self.d_service_biases,
            d_user_delta, d_service_delta, d_user_bias_delta, d_service_bias_delta,
            self.learning_rate, self.l2, self.mu, self.local_epochs,
        )
        _fed_aggregate_users[(active_count + threads - 1) // threads, threads](
            d_active, d_counts, int(users_array.size), d_user_delta, d_user_bias_delta,
            self.d_user_factors, self.d_user_biases,
        )
        _fed_aggregate_services[(self.d_service_factors.size + threads - 1) // threads, threads](
            d_counts, int(users_array.size), d_service_delta, self.d_service_factors
        )
        _fed_aggregate_service_bias[(self.num_services + threads - 1) // threads, threads](
            d_counts, int(users_array.size), d_service_bias_delta, self.d_service_biases
        )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        used = self._memory_used()
        bytes_per_model = self.parameter_count * np.dtype(np.float32).itemsize
        self._last_communication = {
            "uplink_bytes": int(active_count * bytes_per_model + initialization_uplink_bytes),
            "downlink_bytes": int(active_count * bytes_per_model),
            "total_bytes": int(2 * active_count * bytes_per_model + initialization_uplink_bytes),
            "initialization_uplink_bytes": int(initialization_uplink_bytes),
            "uploaded_parameters": int(active_count * self.parameter_count),
            "downloaded_parameters": int(active_count * self.parameter_count),
            "participating_clients": active_count,
        }
        self._last_profile = GPUProfile(
            update_kernel_seconds=kernel_seconds,
            host_to_device_seconds=upload_seconds,
            device_memory_used_bytes=used,
            peak_device_memory_used_bytes=self._peak_memory,
        )
        return {
            "communication": self.communication(),
            "support_entries": int(users_array.size),
            "proximal_mu": float(self.mu),
            "local_epochs": int(self.local_epochs),
            "complete_model_aggregation": True,
            "initialization": self.initialization_metadata(),
        }

    def predict(self, users: Any, services: Any, time_slice: int, raw_cap: float) -> np.ndarray:
        del time_slice
        assert cuda is not None and _fed_predict_kernel is not None
        users_array, services_array, _ = self._validate_indices(users, services)
        upload_started = time.perf_counter()
        d_users = cuda.to_device(users_array)
        d_services = cuda.to_device(services_array)
        d_output = cuda.device_array(users_array.size, dtype=np.float64)
        upload_seconds = time.perf_counter() - upload_started
        threads = 256
        kernel_started = time.perf_counter()
        _fed_predict_kernel[(users_array.size + threads - 1) // threads, threads](
            d_users, d_services, self.d_user_factors, self.d_service_factors,
            self.d_user_biases, self.d_service_biases, math.log1p(float(raw_cap)), d_output,
        )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        download_started = time.perf_counter()
        output = d_output.copy_to_host()
        download_seconds = time.perf_counter() - download_started
        used = self._memory_used()
        self._last_profile = GPUProfile(
            update_kernel_seconds=self._last_profile.update_kernel_seconds,
            prediction_kernel_seconds=kernel_seconds,
            host_to_device_seconds=self._last_profile.host_to_device_seconds + upload_seconds,
            device_to_host_seconds=download_seconds,
            device_memory_used_bytes=used,
            peak_device_memory_used_bytes=self._peak_memory,
        )
        return output

    def state_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "user_factors": self.d_user_factors.copy_to_host(),
            "service_factors": self.d_service_factors.copy_to_host(),
            "user_biases": self.d_user_biases.copy_to_host(),
            "service_biases": self.d_service_biases.copy_to_host(),
        }


class GPUPerFed(_GPUModelBase):
    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        super().__init__()
        assert cuda is not None
        self.num_users = int(config["num_users"])
        self.num_services = int(config["num_services"])
        self.num_times = int(config["num_times"])
        self.num_clients = int(config["num_clients"])
        self.rank = int(config["rank"])
        self.learning_rate = float(config["learning_rate"])
        self.gamma = float(config["gamma"])
        self.l2 = float(config.get("l2", 0.0))
        self.local_epochs = int(config["local_epochs"])
        rng = np.random.default_rng(int(seed))
        mapping = np.empty(self.num_users, dtype=np.int64)
        for client, group in enumerate(np.array_split(rng.permutation(self.num_users), self.num_clients)):
            mapping[group] = client
        scale = float(self.rank ** (-1.0 / 3.0))
        started = time.perf_counter()
        self.user_clients_host = mapping
        self.d_user_clients = cuda.to_device(mapping)
        self.d_private_user = cuda.to_device(rng.uniform(0.0, scale, (self.num_users, self.rank)).astype(np.float32))
        self.d_global_service = cuda.to_device(rng.uniform(0.0, scale, (self.num_services, self.rank)).astype(np.float32))
        self.d_global_time = cuda.to_device(rng.uniform(0.0, scale, (self.num_times, self.rank)).astype(np.float32))
        self.d_local_service = cuda.to_device(rng.uniform(0.0, scale, (self.num_clients, self.num_services, self.rank)).astype(np.float32))
        self.d_local_time = cuda.to_device(rng.uniform(0.0, scale, (self.num_clients, self.num_times, self.rank)).astype(np.float32))
        upload = time.perf_counter() - started
        used = self._memory_used()
        self._last_profile = GPUProfile(host_to_device_seconds=upload, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)

    @property
    def public_parameter_count(self) -> int:
        return int((self.num_services + self.num_times) * self.rank)

    def advance_support(self, users: Any, services: Any, raw_values: Any, time_slice: int, raw_cap: float, round_seed: int) -> dict[str, Any]:
        del round_seed
        assert cuda is not None and _perfed_local_kernel is not None
        users_array, services_array, values_array = self._validate_indices(users, services, raw_values)
        assert values_array is not None
        if users_array.size == 0:
            self._last_communication = {"uplink_bytes": 0, "downlink_bytes": 0, "total_bytes": 0, "participating_clients": 0}
            return {"communication": self.communication(), "support_entries": 0}
        clients = self.user_clients_host[users_array]
        order = np.lexsort((services_array, users_array, clients))
        clients = clients[order]
        users_array = users_array[order]
        services_array = services_array[order]
        normalized = self._normalize(values_array[order], raw_cap, np.float32)
        active, first, counts = np.unique(clients, return_index=True, return_counts=True)
        offsets = np.concatenate((first, np.asarray([len(users_array)], dtype=np.int64)))
        upload_started = time.perf_counter()
        d_active = cuda.to_device(active.astype(np.int64))
        d_offsets = cuda.to_device(offsets.astype(np.int64))
        d_counts = cuda.to_device(counts.astype(np.int64))
        d_users = cuda.to_device(users_array)
        d_services = cuda.to_device(services_array)
        d_targets = cuda.to_device(normalized)
        upload_seconds = time.perf_counter() - upload_started
        threads = 128
        kernel_started = time.perf_counter()
        _perfed_local_kernel[(active.size + threads - 1) // threads, threads](
            d_active, d_offsets, d_users, d_services, d_targets, int(time_slice),
            self.d_private_user, self.d_local_service, self.d_local_time,
            self.d_global_service, self.d_global_time,
            self.learning_rate, self.gamma, self.l2, self.local_epochs,
        )
        coupling = self.learning_rate * self.gamma
        _perfed_center_service[(self.d_global_service.size + threads - 1) // threads, threads](
            d_active, d_counts, int(users_array.size), self.d_local_service,
            self.d_global_service, coupling,
        )
        _perfed_center_time[(self.d_global_time.size + threads - 1) // threads, threads](
            d_active, d_counts, int(users_array.size), self.d_local_time,
            self.d_global_time, coupling,
        )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        used = self._memory_used()
        payload = self.public_parameter_count * np.dtype(np.float32).itemsize
        active_count = int(active.size)
        self._last_communication = {
            "uplink_bytes": int(active_count * payload),
            "downlink_bytes": int(active_count * payload),
            "total_bytes": int(2 * active_count * payload),
            "uploaded_parameters": int(active_count * self.public_parameter_count),
            "downloaded_parameters": int(active_count * self.public_parameter_count),
            "participating_clients": active_count,
        }
        self._last_profile = GPUProfile(update_kernel_seconds=kernel_seconds, host_to_device_seconds=upload_seconds, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)
        return {
            "communication": self.communication(),
            "support_entries": int(users_array.size),
            "synthetic_edge_partition": True,
            "private_user_factors_uploaded": False,
        }

    def predict(self, users: Any, services: Any, time_slice: int, raw_cap: float) -> np.ndarray:
        assert cuda is not None and _perfed_predict_kernel is not None
        users_array, services_array, _ = self._validate_indices(users, services)
        upload_started = time.perf_counter()
        d_users = cuda.to_device(users_array)
        d_services = cuda.to_device(services_array)
        d_output = cuda.device_array(users_array.size, dtype=np.float64)
        upload_seconds = time.perf_counter() - upload_started
        threads = 256
        kernel_started = time.perf_counter()
        _perfed_predict_kernel[(users_array.size + threads - 1) // threads, threads](
            d_users, d_services, int(time_slice), self.d_user_clients,
            self.d_private_user, self.d_local_service, self.d_local_time,
            math.log1p(float(raw_cap)), d_output,
        )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        download_started = time.perf_counter()
        output = d_output.copy_to_host()
        download_seconds = time.perf_counter() - download_started
        used = self._memory_used()
        self._last_profile = GPUProfile(update_kernel_seconds=self._last_profile.update_kernel_seconds, prediction_kernel_seconds=kernel_seconds, host_to_device_seconds=self._last_profile.host_to_device_seconds + upload_seconds, device_to_host_seconds=download_seconds, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)
        return output

    def state_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "user_clients": self.d_user_clients.copy_to_host(),
            "private_user_factors": self.d_private_user.copy_to_host(),
            "global_service_factors": self.d_global_service.copy_to_host(),
            "global_time_factors": self.d_global_time.copy_to_host(),
            "local_service_factors": self.d_local_service.copy_to_host(),
            "local_time_factors": self.d_local_time.copy_to_host(),
        }


class GPUFLLFT(_GPUModelBase):
    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        super().__init__()
        self.num_users = int(config["num_users"])
        self.num_services = int(config["num_services"])
        self.num_times = int(config["num_times"])
        self.rank = int(config["rank"])
        self.learning_rate = float(config["learning_rate"])
        self.regularization = float(config["regularization"])
        self.local_epochs = int(config["local_epochs"])
        self.seed = int(seed)
        self._initialized = False
        self._initialization_info: dict[str, Any] = {}

    @property
    def public_parameter_count(self) -> int:
        return int((self.num_services + self.num_times) * self.rank)

    def _initialize(self, normalized: np.ndarray) -> None:
        assert cuda is not None
        support_mean = float(np.mean(normalized, dtype=np.float64)) if normalized.size else 0.5
        effective = max(support_mean, 1e-8)
        magnitude = float(np.cbrt(effective / self.rank))
        noise = float(min(1e-2, max(1e-6, 0.05 * magnitude)))
        rng = np.random.default_rng(self.seed)
                                                                       
                                                                            
                                                                        
                                                                          
                                                      
        rng.normal(0.0, 0.05, (self.num_users, self.rank))
        rng.normal(0.0, 0.05, (self.num_services, self.rank))
        rng.normal(0.0, 0.05, (self.num_times, self.rank))
        started = time.perf_counter()
        self.d_private_user = cuda.to_device((magnitude + rng.normal(0.0, noise, (self.num_users, self.rank))).astype(np.float64))
        self.d_shared_service = cuda.to_device((magnitude + rng.normal(0.0, noise, (self.num_services, self.rank))).astype(np.float64))
        self.d_shared_time = cuda.to_device((magnitude + rng.normal(0.0, noise, (self.num_times, self.rank))).astype(np.float64))
        upload = time.perf_counter() - started
        self._initialized = True
        self._initialization_info = {
            "source": "slice0_support_mean" if normalized.size else "neutral_empty_slice0_support",
            "support_entries": int(normalized.size),
            "support_target_mean": support_mean,
            "factor_magnitude": magnitude,
            "noise_scale": noise,
        }
        used = self._memory_used()
        self._last_profile = GPUProfile(host_to_device_seconds=upload, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)

    def advance_support(self, users: Any, services: Any, raw_values: Any, time_slice: int, raw_cap: float, round_seed: int) -> dict[str, Any]:
        del round_seed
        assert cuda is not None and _fl_lft_update_kernel is not None
        users_array, services_array, values_array = self._validate_indices(users, services, raw_values)
        assert values_array is not None
        normalized = self._normalize(values_array, raw_cap, np.float64)
        if not self._initialized:
            self._initialize(normalized)
        if users_array.size == 0:
            self._last_communication = {"uplink_bytes": 0, "downlink_bytes": 0, "total_bytes": 0, "participating_clients": 0}
            return {"communication": self.communication(), "support_entries": 0, "initialization": self._initialization_info}
        order = np.lexsort((services_array, users_array))
        users_array = users_array[order]
        services_array = services_array[order]
        normalized = normalized[order]
        upload_started = time.perf_counter()
        d_users = cuda.to_device(users_array)
        d_services = cuda.to_device(services_array)
        d_targets = cuda.to_device(normalized)
        upload_seconds = time.perf_counter() - upload_started
        kernel_started = time.perf_counter()
        _fl_lft_update_kernel[1, 1](
            d_users, d_services, d_targets, int(time_slice),
            self.d_private_user, self.d_shared_service, self.d_shared_time,
            self.learning_rate, self.regularization, self.local_epochs,
        )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        used = self._memory_used()
        active_users = int(np.unique(users_array).size)
        uploaded_parameters = int(users_array.size * self.local_epochs * 2 * self.rank)
        downloaded_parameters = int(active_users * self.public_parameter_count)
        scalar_bytes = np.dtype(np.float64).itemsize
        self._last_communication = {
            "uplink_bytes": int(uploaded_parameters * scalar_bytes),
            "downlink_bytes": int(downloaded_parameters * scalar_bytes),
            "total_bytes": int((uploaded_parameters + downloaded_parameters) * scalar_bytes),
            "uploaded_parameters": uploaded_parameters,
            "downloaded_parameters": downloaded_parameters,
            "participating_clients": active_users,
        }
        self._last_profile = GPUProfile(update_kernel_seconds=kernel_seconds, host_to_device_seconds=upload_seconds, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)
        return {"communication": self.communication(), "support_entries": int(users_array.size), "initialization": self._initialization_info}

    def predict(self, users: Any, services: Any, time_slice: int, raw_cap: float) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("FL-LFT must receive first-slice support before prediction")
        assert cuda is not None and _fl_lft_predict_kernel is not None
        users_array, services_array, _ = self._validate_indices(users, services)
        upload_started = time.perf_counter()
        d_users = cuda.to_device(users_array)
        d_services = cuda.to_device(services_array)
        d_output = cuda.device_array(users_array.size, dtype=np.float64)
        upload_seconds = time.perf_counter() - upload_started
        threads = 256
        kernel_started = time.perf_counter()
        _fl_lft_predict_kernel[(users_array.size + threads - 1) // threads, threads](
            d_users, d_services, int(time_slice), self.d_private_user,
            self.d_shared_service, self.d_shared_time, math.log1p(float(raw_cap)), d_output,
        )
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        download_started = time.perf_counter()
        output = d_output.copy_to_host()
        download_seconds = time.perf_counter() - download_started
        used = self._memory_used()
        self._last_profile = GPUProfile(update_kernel_seconds=self._last_profile.update_kernel_seconds, prediction_kernel_seconds=kernel_seconds, host_to_device_seconds=self._last_profile.host_to_device_seconds + upload_seconds, device_to_host_seconds=download_seconds, device_memory_used_bytes=used, peak_device_memory_used_bytes=self._peak_memory)
        return output

    def state_snapshot(self) -> dict[str, np.ndarray]:
        if not self._initialized:
            raise RuntimeError("FL-LFT state is not initialized")
        return {
            "private_user_factors": self.d_private_user.copy_to_host(),
            "shared_service_factors": self.d_shared_service.copy_to_host(),
            "shared_time_factors": self.d_shared_time.copy_to_host(),
        }


def create_model(method: str, config: Mapping[str, Any], seed: int) -> _GPUModelBase:
    method = str(method).lower()
    if method == "fedavg":
        return GPUFedAvgProx(config, seed, mu=0.0)
    if method == "fl_lft":
        return GPUFLLFT(config, seed)
    raise ValueError(f"unknown baseline method: {method!r}")


__all__ = [
    "GPU_BACKEND",
    "EXECUTION_MODE",
    "SUPPORT_INITIALIZATION",
    "DEFAULT_INITIALIZATION_CLIP_EPSILON",
    "GPUProfile",
    "GPUFedAvgProx",
    "GPUFLLFT",
    "create_model",
    "cuda_available",
    "cuda_device_info",
]

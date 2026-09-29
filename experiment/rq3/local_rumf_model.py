                                                  

                                                                            
                                                                            
                                                                        
                                                                            
                                                                     

                                                                          
                                                                             
                                                                            
   

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

try:                                                               
    from numba import cuda
except Exception:                                                        
    cuda = None                            


MAX_RANK = 64
CPSA_GAIN = 0.1
CPSA_L2 = 0.0003


@dataclass(frozen=True)
class LocalRUMFAdvanceProfile:
                                                              

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
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class LocalCommunicationReport:
                                                                   

    active_clients: tuple[int, ...]
    support_counts: tuple[tuple[int, int], ...]
    proposal_count: int = 0
    uplink_bytes: int = 0
    downlink_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_clients": list(self.active_clients),
            "support_counts": [list(item) for item in self.support_counts],
            "proposal_count": int(self.proposal_count),
            "uplink_bytes": int(self.uplink_bytes),
            "downlink_bytes": int(self.downlink_bytes),
        }


def cuda_available() -> bool:
                                                     

    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def cuda_device_info() -> dict[str, Any]:
                                                            

    if not cuda_available():
        return {"backend": "numba-cuda", "available": False, "name": None}
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
            "backend": "numba-cuda",
            "available": True,
            "name": name,
            "compute_capability": (
                [int(value) for value in capability]
                if capability is not None
                else None
            ),
        }
    except Exception as exc:                                       
        return {
            "backend": "numba-cuda",
            "available": False,
            "name": None,
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


    @cuda.jit
    def _local_cpsa_update_kernel(
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
        model_logit = user_bias[user] + service_bias[user, service]
        for factor in range(user_factors.shape[1]):
            model_logit += (
                user_factors[user, factor]
                * service_factors[user, service, factor]
            )
        residual = target - (model_logit + adapter[user, service])
        if residual < -3.0:
            residual = -3.0
        elif residual > 3.0:
            residual = 3.0
        adapter[user, service] = (
            (1.0 - CPSA_GAIN * CPSA_L2) * adapter[user, service]
            + CPSA_GAIN * residual
        )


    @cuda.jit
    def _local_fit_kernel(
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
        service_factors: Any,
        service_bias: Any,
        service_error: Any,
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
            prediction = _sigmoid(
                user_bias[user] + service_bias[user, service]
            )
            relative_error = abs(prediction - target) / max(target, 1e-12)
            denominator = user_error[user] + service_error[user, service]
            user_weight = user_error[user] / max(denominator, 1e-12)
            service_weight = 1.0 - user_weight
            user_error[user] = (
                beta * user_weight * relative_error
                + (1.0 - beta * user_weight) * user_error[user]
            )
            service_error[user, service] = (
                beta * service_weight * relative_error
                + (1.0 - beta * service_weight)
                * service_error[user, service]
            )
            gradient = (prediction - target) * prediction * (1.0 - prediction)
            user_bias[user] -= bias_eta * (
                user_weight * gradient + l2 * user_bias[user]
            )
            service_bias[user, service] -= bias_eta * (
                service_weight * gradient
                + l2 * service_bias[user, service]
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
                user_bias[user] + service_bias[user, service]
            )
            for factor in range(user_factors.shape[1]):
                before_user[factor] = user_factors[user, factor]
                before_service[factor] = service_factors[user, service, factor]
            error = 0.0
            for factor in range(user_factors.shape[1]):
                error += before_user[factor] * before_service[factor]
            error -= residual_target
            for factor in range(user_factors.shape[1]):
                user_factors[user, factor] -= factor_eta * (
                    error * before_service[factor]
                    + l2 * before_user[factor]
                )
                service_factors[user, service, factor] -= factor_eta * (
                    error * before_user[factor]
                    + l2 * before_service[factor]
                )
            loss += error * error
            updates += 1
        loss_by_client[slot] = loss / max(updates, 1)


    @cuda.jit
    def _local_predict_kernel(
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
        logit = (
            user_bias[user]
            + service_bias[user, service]
            + adapter[user, service]
        )
        for factor in range(user_factors.shape[1]):
            logit += user_factors[user, factor] * service_factors[
                user, service, factor
            ]
        output[index] = math.expm1(_sigmoid(logit) * log1p_cap)

else:                                             
    _sigmoid = _raw_to_logit = None
    _local_cpsa_update_kernel = _local_fit_kernel = _local_predict_kernel = None


class LocalRUMFModel:
                                                            

    model_name = "RQ2-Local-RUMF"
    aggregation_algorithm = "local-only; no federated aggregation"

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
    ) -> None:
        del num_edges                                                         
        if not cuda_available():
            raise RuntimeError("a usable Numba CUDA device is required")
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
            raise ValueError("invalid RUMF hyperparameters")
        rng = np.random.default_rng(self.seed)
        user_factors = rng.normal(
            0.0, self.factor_init_std, (self.num_users, self.rank)
        ).astype(np.float64)
        service_factors = rng.normal(
            0.0,
            self.factor_init_std,
            (self.num_users, self.num_services, self.rank),
        ).astype(np.float64)
        self._d_user_factors = cuda.to_device(user_factors)
        self._d_service_factors = cuda.to_device(service_factors)
        self._d_user_bias = cuda.to_device(np.zeros(self.num_users, dtype=np.float64))
        self._d_service_bias = cuda.to_device(
            np.zeros((self.num_users, self.num_services), dtype=np.float64)
        )
        self._d_user_error = cuda.to_device(np.ones(self.num_users, dtype=np.float64))
        self._d_service_error = cuda.to_device(
            np.ones((self.num_users, self.num_services), dtype=np.float64)
        )
        self._d_adapter = cuda.to_device(
            np.zeros((self.num_users, self.num_services), dtype=np.float64)
        )
        self.last_profile: LocalRUMFAdvanceProfile | None = None
        self.last_diagnostics: dict[str, Any] = {
            "variant": "LOCAL_RUMF",
            "aggregation": "local",
            "server_aggregation": False,
            "shared_broadcast": False,
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
        self._peak_memory = max(self._peak_memory, self._memory_used())
        return self._memory_used()

    @staticmethod
    def _grid(length: int, threads: int = 128) -> tuple[int, int]:
        return ((int(length) + threads - 1) // threads, threads)

    @staticmethod
    def _upload(value: np.ndarray) -> Any:
        assert cuda is not None
        return cuda.to_device(np.ascontiguousarray(value))

    def _prepare_orders(
        self,
        support_users: np.ndarray,
        bias_epochs: int,
        factor_epochs: int,
        shuffle_seed: int,
        active_users: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                np.random.SeedSequence([int(shuffle_seed), int(user), 0xB1A5])
            )
            for _ in range(int(bias_epochs)):
                bias_order.extend(int(value) for value in bias_rng.permutation(positions))
            bias_offsets.append(len(bias_order))
            factor_rng = np.random.default_rng(
                np.random.SeedSequence([int(shuffle_seed), int(user), 0xFAC7])
            )
            for _ in range(int(factor_epochs)):
                factor_order.extend(int(value) for value in factor_rng.permutation(positions))
            factor_offsets.append(len(factor_order))
        return (
            np.asarray(segment_offsets, dtype=np.int32),
            np.asarray(segment_order, dtype=np.int32),
            np.asarray(bias_offsets, dtype=np.int32),
            np.asarray(bias_order, dtype=np.int32),
            np.asarray(factor_offsets, dtype=np.int32),
            np.asarray(factor_order, dtype=np.int32),
        )

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

    def predict(
        self,
        users: np.ndarray,
        services: np.ndarray,
        raw_cap: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if cuda is None or _local_predict_kernel is None:
            raise RuntimeError("CUDA kernels are unavailable")
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.ndim != 1 or services.ndim != 1 or users.shape != services.shape:
            raise ValueError("query users/services must be equal one-dimensional arrays")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("query user outside model dimensions")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("query service outside model dimensions")
        raw_cap = float(raw_cap)
        if not np.isfinite(raw_cap) or raw_cap <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        d_users = self._upload(users)
        d_services = self._upload(services)
        output = cuda.device_array(users.size, dtype=np.float32)
        started = time.perf_counter()
        if users.size:
            blocks, threads = self._grid(users.size)
            _local_predict_kernel[blocks, threads](
                d_users,
                d_services,
                self._d_user_factors,
                self._d_service_factors,
                self._d_user_bias,
                self._d_service_bias,
                self._d_adapter,
                math.log1p(raw_cap),
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
                                                                                

        if cuda is None or _local_cpsa_update_kernel is None or _local_fit_kernel is None:
            raise RuntimeError("CUDA kernels are unavailable")
        raw_cap = float(raw_cap)
        bias_epochs = int(bias_epochs)
        factor_epochs = int(factor_epochs)
        if not np.isfinite(raw_cap) or raw_cap <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        if bias_epochs < 0 or factor_epochs < 0:
            raise ValueError("epochs must be non-negative integers")
        users, services, values = self._validate_support(
            support_users,
            support_services,
            support_values,
            self.num_users,
            self.num_services,
            raw_cap,
        )
        active = np.unique(users).astype(np.int64, copy=False)
        (
            _segment_offsets,
            _segment_order,
            bias_offsets,
            bias_order,
            factor_offsets,
            factor_order,
        ) = self._prepare_orders(
            users,
            bias_epochs,
            factor_epochs,
            int(shuffle_seed),
            active,
        )
        del _segment_offsets, _segment_order
        started_total = time.perf_counter()
        started_upload = time.perf_counter()
        d_users = self._upload(users)
        d_services = self._upload(services)
        d_values = self._upload(values)
        d_active = self._upload(active)
        d_bias_offsets = self._upload(bias_offsets)
        d_bias_order = self._upload(bias_order)
        d_factor_offsets = self._upload(factor_offsets)
        d_factor_order = self._upload(factor_order)
        input_upload_seconds = time.perf_counter() - started_upload

        started = time.perf_counter()
        blocks, threads = self._grid(users.size)
        _local_cpsa_update_kernel[blocks, threads](
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

        loss_by_client = cuda.device_array(active.size, dtype=np.float64)
        started = time.perf_counter()
        _local_fit_kernel[active.size, 1](
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
            self._d_service_factors,
            self._d_service_bias,
            self._d_service_error,
            loss_by_client,
        )
        cuda.synchronize()
        local_update_seconds = time.perf_counter() - started
        self._observe_memory()

        active_host = tuple(int(value) for value in active.tolist())
        communication = LocalCommunicationReport(
            active_clients=active_host,
            support_counts=tuple(
                (int(user), int(np.count_nonzero(users == user)))
                for user in active_host
            ),
        )
        self.last_diagnostics = {
            "variant": "LOCAL_RUMF",
            "aggregation": "local",
            "server_aggregation": False,
            "shared_broadcast": False,
            "active_clients": list(active_host),
            "support_count": int(users.size),
            "proposal_count": 0,
            "uplink_bytes": 0,
            "downlink_bytes": 0,
            "query_or_future_used": False,
        }
        self.last_profile = LocalRUMFAdvanceProfile(
            support_count=int(users.size),
            active_clients=int(active.size),
            input_upload_seconds=float(input_upload_seconds),
            cpsa_seconds=float(cpsa_seconds),
            shared_broadcast_seconds=0.0,
            local_update_seconds=float(local_update_seconds),
            carw_srr_seconds=0.0,
            proposal_quantization_seconds=0.0,
            aggregation_seconds=0.0,
            aggregate_quantization_seconds=0.0,
            total_seconds=float(time.perf_counter() - started_total),
            peak_device_memory_used_bytes=int(self._peak_memory),
        )
        return SimpleNamespace(
            active_clients=active_host,
            proposals=(),
            communication=communication,
            service_aggregation_diagnostics=(),
        )

    def parameter_snapshot(self) -> dict[str, np.ndarray]:
                                                     

        assert cuda is not None
        return {
            "user_factors": self._d_user_factors.copy_to_host(),
            "local_service_factors": self._d_service_factors.copy_to_host(),
            "user_bias": self._d_user_bias.copy_to_host(),
            "local_service_bias": self._d_service_bias.copy_to_host(),
            "user_error": self._d_user_error.copy_to_host(),
            "local_service_error": self._d_service_error.copy_to_host(),
            "client_service_adapter": self._d_adapter.copy_to_host(),
        }

    full_state_snapshot = parameter_snapshot

    @property
    def local_state_bytes(self) -> int:
        itemsize = np.dtype(np.float64).itemsize
        return int(
            self.num_users * self.rank * itemsize
            + self.num_users * self.num_services * self.rank * itemsize
            + 2 * self.num_users * itemsize
            + 3 * self.num_users * self.num_services * itemsize
        )

    @property
    def private_state_bytes(self) -> int:
        return self.local_state_bytes


def state_hash(model: LocalRUMFModel) -> str:
                                                                     

    digest = hashlib.sha256()
    for name, raw in sorted(model.full_state_snapshot().items()):
        value = np.asarray(raw)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
                                                                 

    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 1:
        raise ValueError("truth and prediction must be equal one-dimensional arrays")
    if target.size == 0 or not np.all(np.isfinite(target)) or not np.all(np.isfinite(estimate)):
        raise ValueError("metrics require finite non-empty arrays")
    if np.any(target <= 0.0):
        raise ValueError("metrics require positive truth values")
    error = estimate - target
    absolute = np.abs(error)
    relative = np.sort(absolute / target)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mre": float(np.median(relative)),
        "npre": float(relative[int(0.9 * relative.size)]),
        "absolute_error_sum": float(np.sum(absolute)),
        "squared_error_sum": float(np.sum(np.square(error))),
    }

                                                           

                                                                         
                                                                        
                                                                            
                                                                        

                                                                          
                                                                            
                                                                          
                                                                  
                                                                             
                                     
   

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


                                                                            
                                                                                 
                                                                     
DEFAULT_NUM_USERS = 142
DEFAULT_NUM_SERVICES = 4500
DEFAULT_DIMENSION = 10
PARAMETER_COUNT = (
    DEFAULT_NUM_USERS * DEFAULT_DIMENSION
    + DEFAULT_NUM_SERVICES * DEFAULT_DIMENSION
    + DEFAULT_NUM_USERS
    + DEFAULT_NUM_SERVICES
)
PARAMETER_BYTES_FP32 = PARAMETER_COUNT * 4
                                                                          
                                                                        
                      
INTERCEPT_CLIP_EPSILON = 1e-4


def sigmoid(value: float) -> float:
                                                      

    return float(1.0 / (1.0 + np.exp(-np.clip(float(value), -30.0, 30.0))))


@dataclass(frozen=True)
class FedAvgClientDelta:
                                                                 

                                                                       
                                                                           
                         
       

    user: int
    service_indices: np.ndarray
    user_vector_delta: np.ndarray
    service_vector_deltas: np.ndarray
    user_bias_delta: np.float32
    service_bias_deltas: np.ndarray
    support_entries: int


class FullAMFModel:
                                                                     

    def __init__(self, user_vectors, service_vectors, user_biases, service_biases):
        self.user_vectors = np.asarray(user_vectors, dtype=np.float32)
        self.service_vectors = np.asarray(service_vectors, dtype=np.float32)
        self.user_biases = np.asarray(user_biases, dtype=np.float32)
        self.service_biases = np.asarray(service_biases, dtype=np.float32)
        if self.user_vectors.ndim != 2 or self.service_vectors.ndim != 2:
            raise ValueError("embedding tensors must be two-dimensional")
        if self.user_vectors.shape[1] != self.service_vectors.shape[1]:
            raise ValueError("user and service dimensions must match")
        if self.user_biases.shape != (self.user_vectors.shape[0],):
            raise ValueError("user bias shape mismatch")
        if self.service_biases.shape != (self.service_vectors.shape[0],):
            raise ValueError("service bias shape mismatch")

    @classmethod
    def random(cls, num_users: int, num_services: int, dimension: int, seed: int):
                                                             

        if min(num_users, num_services, dimension) <= 0:
            raise ValueError("model dimensions must be positive")
        rng = np.random.default_rng(int(seed))
        return cls(
            rng.normal(0.0, 0.05, (num_users, dimension)).astype(np.float32),
            rng.normal(0.0, 0.05, (num_services, dimension)).astype(np.float32),
            np.zeros(num_users, dtype=np.float32),
            np.zeros(num_services, dtype=np.float32),
        )

    @property
    def dimension(self) -> int:
        return int(self.user_vectors.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(
            self.user_vectors.size
            + self.service_vectors.size
            + self.user_biases.size
            + self.service_biases.size
        )

    @property
    def parameter_bytes_fp32(self) -> int:
                                                       

        return int(self.parameter_count * np.dtype(np.float32).itemsize)

    def copy(self) -> "FullAMFModel":
        return FullAMFModel(
            self.user_vectors.copy(),
            self.service_vectors.copy(),
            self.user_biases.copy(),
            self.service_biases.copy(),
        )

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.user_vectors,
            self.service_vectors,
            self.user_biases,
            self.service_biases,
        )

    def predict_normalized(self, users, services) -> np.ndarray:
                                                                         

        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.ndim != 1 or services.ndim != 1 or users.shape != services.shape:
            raise ValueError("prediction indices must have the same 1-D shape")
        if np.any(users < 0) or np.any(users >= self.user_vectors.shape[0]):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.service_vectors.shape[0]):
            raise IndexError("service index out of range")
        logits = (
            np.sum(self.user_vectors[users] * self.service_vectors[services], axis=-1)
            + self.user_biases[users]
            + self.service_biases[services]
        )
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(
            np.float32,
            copy=False,
        )


def initialize_global_intercept(
    model: FullAMFModel,
    normalized_support_targets,
) -> dict[str, object]:
                                                                       

                                                                   
                                                              
                                                                            
                                                                     
                                                       

                                                                            
                                                                             
                                                                       
                                     
       

    if not isinstance(model, FullAMFModel):
        raise TypeError("model must be a FullAMFModel")
    try:
        targets = np.asarray(normalized_support_targets, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normalized support targets must be numeric"
        ) from exc
    if targets.ndim != 1:
        raise ValueError(
            "normalized support targets must be a one-dimensional vector"
        )
    if targets.size == 0:
        raise ValueError("normalized support targets must be non-empty")
    if not np.all(np.isfinite(targets)):
        raise ValueError("normalized support targets must be finite")
    if np.any(targets < 0.0) or np.any(targets > 1.0):
        raise ValueError(
            "normalized support targets must lie in [0, 1]"
        )
    if not model.user_biases.flags.writeable or not model.service_biases.flags.writeable:
        raise ValueError("model bias arrays must be writable")

    support_target_mean = float(np.mean(targets, dtype=np.float64))
    clipped_target_mean = float(
        np.clip(
            support_target_mean,
            INTERCEPT_CLIP_EPSILON,
            1.0 - INTERCEPT_CLIP_EPSILON,
        )
    )
    intercept = float(
        np.log(clipped_target_mean / (1.0 - clipped_target_mean))
    )
    half_intercept = np.float32(intercept / 2.0)
    model.user_biases[...] = half_intercept
    model.service_biases[...] = half_intercept
    return {
        "initialized": True,
        "support_entries": int(targets.size),
        "support_target_mean": support_target_mean,
        "clipped_target_mean": clipped_target_mean,
        "intercept": intercept,
        "target_logit": intercept,
        "user_bias": float(half_intercept),
        "service_bias": float(half_intercept),
        "clip_epsilon": float(INTERCEPT_CLIP_EPSILON),
    }


                                                                         
                                                    
initialize_support_intercept = initialize_global_intercept


def client_sgd_delta(
    global_model: FullAMFModel,
    user: int,
    services,
    targets,
    eta: float = 0.8,
    l2: float = 3e-4,
    local_epochs: int = 1,
) -> FedAvgClientDelta:
                                                                    

                                                                             
                                                                             
                                                                          
                             
       

    if not (0 <= int(user) < global_model.user_vectors.shape[0]):
        raise IndexError("user index out of range")
    if not np.isfinite(float(eta)) or float(eta) <= 0.0:
        raise ValueError("eta must be finite and positive")
    if not np.isfinite(float(l2)) or float(l2) < 0.0:
        raise ValueError("l2 must be finite and non-negative")
    if isinstance(local_epochs, bool) or int(local_epochs) <= 0:
        raise ValueError("local_epochs must be a positive integer")

    service_indices = np.asarray(services, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float32)
    if service_indices.ndim != 1 or targets.shape != service_indices.shape:
        raise ValueError("services and targets must be aligned vectors")
    if len(service_indices) == 0:
        raise ValueError("a participating client must have support entries")
    if np.any(service_indices < 0) or np.any(
        service_indices >= global_model.service_vectors.shape[0]
    ):
        raise IndexError("service index out of range")
    if np.any(service_indices[1:] < service_indices[:-1]):
        raise ValueError("services must be supplied in ascending order")
    if len(np.unique(service_indices)) != len(service_indices):
        raise ValueError("services must be unique within a time slice")
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite")

    old_user = global_model.user_vectors[user].copy()
    old_user_bias = np.float32(global_model.user_biases[user])
    local_user = old_user.copy()
    local_user_bias = np.float32(old_user_bias)
    local_services = global_model.service_vectors[service_indices].copy()
    local_service_biases = global_model.service_biases[service_indices].copy()
    eta32 = np.float32(eta)
    l2_32 = np.float32(l2)

    for _ in range(int(local_epochs)):
        for index, target in enumerate(targets):
            before_user = local_user.copy()
            before_user_bias = np.float32(local_user_bias)
            before_service = local_services[index].copy()
            before_service_bias = np.float32(local_service_biases[index])
            prediction = sigmoid(
                np.dot(before_user, before_service)
                + float(before_user_bias)
                + float(before_service_bias)
            )
            gradient = np.float32(
                (prediction - float(target)) * prediction * (1.0 - prediction)
            )
            local_user -= eta32 * (
                gradient * before_service + l2_32 * before_user
            )
            local_services[index] -= eta32 * (
                gradient * before_user + l2_32 * before_service
            )
            local_user_bias -= eta32 * (
                gradient + l2_32 * before_user_bias
            )
            local_service_biases[index] -= eta32 * (
                gradient + l2_32 * before_service_bias
            )

    return FedAvgClientDelta(
        user=int(user),
        service_indices=service_indices.copy(),
        user_vector_delta=(local_user - old_user).astype(np.float32),
        service_vector_deltas=(
            local_services - global_model.service_vectors[service_indices]
        ).astype(np.float32),
        user_bias_delta=np.float32(local_user_bias - old_user_bias),
        service_bias_deltas=(
            local_service_biases - global_model.service_biases[service_indices]
        ).astype(np.float32),
                                                                              
                                                                         
        support_entries=int(len(service_indices)),
    )


def aggregate_sparse_deltas(
    global_model: FullAMFModel,
    client_deltas: list[FedAvgClientDelta],
) -> FullAMFModel:
                                                                             

    if not client_deltas:
        raise ValueError("at least one client is required")
    total = sum(int(delta.support_entries) for delta in client_deltas)
    if total <= 0:
        raise ValueError("client support weights must be positive")
    result = global_model.copy()
    for delta in client_deltas:
        weight = np.float64(delta.support_entries) / np.float64(total)
        result.user_vectors[delta.user] += (
            np.float32(weight) * delta.user_vector_delta
        )
        result.user_biases[delta.user] += (
            np.float32(weight) * delta.user_bias_delta
        )
        result.service_vectors[delta.service_indices] += (
            np.float32(weight) * delta.service_vector_deltas
        )
        result.service_biases[delta.service_indices] += (
            np.float32(weight) * delta.service_bias_deltas
        )
    return result


def aggregate_full_models(
    global_model: FullAMFModel,
    client_models: list[FullAMFModel],
    support_counts,
) -> FullAMFModel:
                                                                               

    if not client_models or len(client_models) != len(support_counts):
        raise ValueError("client models and support counts must be non-empty and aligned")
    counts = np.asarray(support_counts, dtype=np.float64)
    if np.any(counts <= 0) or not np.all(np.isfinite(counts)):
        raise ValueError("support counts must be finite and positive")
    weights = counts / counts.sum()
    for model in client_models:
        if model.parameter_count != global_model.parameter_count:
            raise ValueError("client model shape mismatch")
    arrays = []
    for index in range(4):
        arrays.append(
            np.asarray(
                sum(
                    weight * model.state_arrays()[index]
                    for weight, model in zip(weights, client_models)
                ),
                dtype=np.float32,
            )
        )
    return FullAMFModel(*arrays)


__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_NUM_SERVICES",
    "DEFAULT_NUM_USERS",
    "FedAvgClientDelta",
    "FullAMFModel",
    "INTERCEPT_CLIP_EPSILON",
    "PARAMETER_BYTES_FP32",
    "PARAMETER_COUNT",
    "aggregate_full_models",
    "aggregate_sparse_deltas",
    "client_sgd_delta",
    "initialize_global_intercept",
    "initialize_support_intercept",
    "sigmoid",
]

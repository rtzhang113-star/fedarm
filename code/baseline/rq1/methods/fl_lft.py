                                                                     

                                                                       
                                                                          
                                                                           
   
from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np


_SCALE_EPS = 1e-8


class FederatedLatentTensorFactorization:
    def __init__(
        self,
        num_users: int,
        num_services: int,
        num_times: int,
        rank: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        seed: int = 0,
    ):
        if min(num_users, num_services, num_times, rank) <= 0:
            raise ValueError("tensor dimensions and rank must be positive")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError("invalid FL-LFT optimization parameters")
        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.num_times = int(num_times)
        self.rank = int(rank)
        self._rng = np.random.default_rng(seed)
        self.private_user_factors = self._rng.normal(0.0, 0.05, (num_users, rank))
        self.shared_service_factors = self._rng.normal(0.0, 0.05, (num_services, rank))
        self.shared_time_factors = self._rng.normal(0.0, 0.05, (num_times, rank))
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
                                                                        
                                                                            
                                                                            
                                                                         
                                                                            
                                                                 
        self._support_scale_initialized = False
        self._support_scale_initialization_count = 0
        self._support_scale_info = {
            "initialized": False,
            "source": None,
            "support_entries": 0,
            "support_target_mean": None,
            "effective_target_mean": None,
            "factor_magnitude": None,
            "noise_scale": None,
            "rank": int(rank),
            "mode": "positive_balanced_trilinear_scale",
        }

    @property
    def dimension(self) -> int:
                                                                               

        return int(self.rank)

    @property
    def parameter_count(self) -> int:
                                                                     

        return int(
            self.private_user_factors.size
            + self.shared_service_factors.size
            + self.shared_time_factors.size
        )

    @property
    def public_parameter_count(self) -> int:
                                                           

        return int(self.shared_service_factors.size + self.shared_time_factors.size)

    @property
    def public_parameter_bytes(self) -> int:
                                                               

        return int(self.public_parameter_count * np.dtype(np.float64).itemsize)

    @classmethod
    def random(cls, num_users: int, num_services: int, num_times: int, **kwargs):
                                                                    

        return cls(num_users, num_services, num_times, **kwargs)

    def copy(self):
                                                                       

        return copy.deepcopy(self)

    def state_arrays(self) -> tuple[np.ndarray, ...]:
                                                                               

        return tuple(
            array.copy()
            for array in (
                self.private_user_factors,
                self.shared_service_factors,
                self.shared_time_factors,
            )
        )

    def state_unchanged(self, snapshot) -> bool:
                                                                             

        current = self.state_arrays()
        snapshot = tuple(snapshot)
        return len(snapshot) == len(current) and all(
            np.array_equal(before, after) for before, after in zip(snapshot, current)
        )

    def read_only_audit(self, operation: Callable, *args, **kwargs) -> bool:
                                                                               

        before = self.state_arrays()
        operation(*args, **kwargs)
        return self.state_unchanged(before)

    @property
    def support_scale_initialization_count(self) -> int:
                                                                      
        return int(self._support_scale_initialization_count)

    def support_scale_initialization_info(self) -> dict:
                                                                            
        return dict(self._support_scale_info)

    def initialize_from_support_targets(self, targets) -> dict:
                                                                          

                                                                      
                                                                  
                                                                             
                                                                           
                                                                           

                                                                           
                                                                           
                                                                         
                                                
           
        if self._support_scale_initialized:
            raise RuntimeError("support-scale initialization may run only once")

        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError("support targets must be a one-dimensional vector")
        if targets.size and (
            not np.all(np.isfinite(targets))
            or np.any(targets < 0.0)
            or np.any(targets > 1.0)
        ):
            raise ValueError("support targets must be finite normalized values in [0, 1]")

        if targets.size:
            support_mean = float(np.mean(targets, dtype=np.float64))
            source = "slice0_support_mean"
        else:
            support_mean = 0.5
            source = "neutral_empty_slice0_support"

                                                                         
                                                                             
        effective_mean = float(max(support_mean, _SCALE_EPS))
        magnitude = float(np.cbrt(effective_mean / float(self.rank)))
                                                                            
                                                                              
                                                                          
        noise_scale = float(min(1e-2, max(1e-6, 0.05 * magnitude)))
        self.private_user_factors = (
            magnitude + self._rng.normal(0.0, noise_scale, self.private_user_factors.shape)
        ).astype(np.float64)
        self.shared_service_factors = (
            magnitude + self._rng.normal(0.0, noise_scale, self.shared_service_factors.shape)
        ).astype(np.float64)
        self.shared_time_factors = (
            magnitude + self._rng.normal(0.0, noise_scale, self.shared_time_factors.shape)
        ).astype(np.float64)
        if not all(np.all(np.isfinite(array)) for array in (
            self.private_user_factors, self.shared_service_factors, self.shared_time_factors
        )):
            raise FloatingPointError("support-only FL-LFT scale initialization was not finite")

        self._support_scale_initialized = True
        self._support_scale_initialization_count += 1
        self._support_scale_info = {
            "initialized": True,
            "source": source,
            "support_entries": int(targets.size),
            "support_target_mean": support_mean,
            "effective_target_mean": effective_mean,
            "factor_magnitude": magnitude,
            "noise_scale": noise_scale,
            "rank": int(self.rank),
            "mode": "positive_balanced_trilinear_scale",
        }
        return self.support_scale_initialization_info()

    def support_reconstruction_metrics(self, users, services, time_slices, values) -> dict:
                                                                              
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if not (users.ndim == services.ndim == time_slices.ndim == values.ndim == 1):
            raise ValueError("support audit inputs must be one-dimensional vectors")
        if not (users.shape == services.shape == time_slices.shape == values.shape):
            raise ValueError("support audit inputs must have aligned shapes")
        if not np.all(np.isfinite(values)):
            raise ValueError("support audit targets must be finite")
        if not len(values):
            return {
                "support_target_mean": 0.0,
                "support_prediction_mean": 0.0,
                "support_loss": 0.0,
            }
        predictions = self.predict(users, services, time_slices)
        residual = predictions.astype(np.float64) - values
        return {
            "support_target_mean": float(np.mean(values, dtype=np.float64)),
            "support_prediction_mean": float(np.mean(predictions, dtype=np.float64)),
            "support_loss": float(np.mean(residual * residual, dtype=np.float64)),
        }

    def _validate_support(self, user: int, services, time_slices, values, epochs: int):
        if isinstance(user, bool) or int(user) != user:
            raise ValueError("user must be an integer")
        user = int(user)
        if user < 0 or user >= self.num_users:
            raise IndexError("user index out of range")
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) <= 0:
            raise ValueError("epochs must be a positive integer")
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if services.ndim != 1 or time_slices.ndim != 1 or values.ndim != 1:
            raise ValueError("services, time_slices and values must be one-dimensional")
        if services.shape != time_slices.shape or services.shape != values.shape:
            raise ValueError("services, time_slices and values must be aligned vectors")
        if services.size == 0:
            raise ValueError("a participating client must have support entries")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if np.any(time_slices < 0) or np.any(time_slices >= self.num_times):
            raise IndexError("time index out of range")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        return user, services.copy(), time_slices.copy(), values.copy(), int(epochs)

    def update_client(self, user: int, services, time_slices, values, epochs: int = 1) -> dict:
        user, services, time_slices, values, epochs = self._validate_support(
            user, services, time_slices, values, epochs
        )
        lr, reg = self.learning_rate, self.regularization
        uploaded_scalars = 0
        for _ in range(int(epochs)):
            for service, time_slice, target in zip(services, time_slices, values):
                d = self.private_user_factors[user].copy()
                e = self.shared_service_factors[service].copy()
                t = self.shared_time_factors[time_slice].copy()
                prediction = float(np.sum(d * e * t))
                error = prediction - target
                self.private_user_factors[user] -= lr * (error * e * t + reg * d)
                d_updated = self.private_user_factors[user].copy()
                grad_e = error * d_updated * t + reg * e
                grad_t = error * d_updated * e + reg * t
                self.shared_service_factors[service] -= lr * grad_e
                self.shared_time_factors[time_slice] -= lr * grad_t
                uploaded_scalars += grad_e.size + grad_t.size
        downloaded_scalars = self.shared_service_factors.size + self.shared_time_factors.size
        return {
            "uploaded_bytes": int(uploaded_scalars * self.shared_service_factors.dtype.itemsize),
            "downloaded_bytes": int(downloaded_scalars * self.shared_service_factors.dtype.itemsize),
            "support_entries": int(len(values) * epochs),
        }

    def predict(self, users, services, time_slices) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        if users.ndim != 1 or users.shape != services.shape or users.shape != time_slices.shape:
            raise ValueError("prediction indices must have the same shape")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if np.any(time_slices < 0) or np.any(time_slices >= self.num_times):
            raise IndexError("time index out of range")
        return np.sum(
            self.private_user_factors[users]
            * self.shared_service_factors[services]
            * self.shared_time_factors[time_slices],
            axis=-1,
        )

                                                      

from __future__ import annotations

import copy

import numpy as np


class CausalResidualBiasMF:
                                                                                  

                                                                               
                                                                               
       

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
        interaction: bool = True,
    ) -> None:
        if min(num_users, num_services, rank) <= 0:
            raise ValueError("dimensions must be positive")
        if min(bias_eta, factor_eta, factor_init_std) <= 0:
            raise ValueError("learning rates and factor scale must be positive")
        if l2 < 0 or not (0 < beta <= 1):
            raise ValueError("invalid residual MF hyperparameters")
        rng = np.random.default_rng(int(seed))
        self.num_users, self.num_services, self.rank = int(num_users), int(num_services), int(rank)
        self.bias_eta, self.factor_eta = float(bias_eta), float(factor_eta)
        self.l2, self.beta = float(l2), float(beta)
        self.interaction = bool(interaction)
        self.user_factors = rng.normal(0.0, factor_init_std, (num_users, rank)).astype(np.float64)
        self.service_factors = rng.normal(0.0, factor_init_std, (num_services, rank)).astype(np.float64)
        self.user_bias = np.zeros(num_users, dtype=np.float64)
        self.service_bias = np.zeros(num_services, dtype=np.float64)
        self.user_error = np.ones(num_users, dtype=np.float64)
        self.service_error = np.ones(num_services, dtype=np.float64)

    @staticmethod
    def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))

    @staticmethod
    def _normalize(raw_values: np.ndarray, raw_cap: float) -> np.ndarray:
        values = np.asarray(raw_values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or not np.all((values > 0) & (values < raw_cap)):
            raise ValueError("support values must be finite and inside the fixed QoS cap")
        return np.log1p(values) / np.log1p(float(raw_cap))

    def _bias_stage(
        self,
        users: np.ndarray,
        services: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        shuffle_seed: int,
    ) -> None:
        for user in np.unique(users):
            positions = np.flatnonzero(users == user)
            rng = np.random.default_rng(np.random.SeedSequence([int(shuffle_seed), int(user), 0xB1A5]))
            for _ in range(int(epochs)):
                for position in rng.permutation(positions):
                    service = int(services[position])
                    target = float(targets[position])
                    prediction = float(self._sigmoid(self.user_bias[user] + self.service_bias[service]))
                    relative_error = abs(prediction - target) / max(target, 1e-12)
                    denominator = self.user_error[user] + self.service_error[service]
                    user_weight = self.user_error[user] / max(denominator, 1e-12)
                    service_weight = 1.0 - user_weight
                    self.user_error[user] = (
                        self.beta * user_weight * relative_error
                        + (1.0 - self.beta * user_weight) * self.user_error[user]
                    )
                    self.service_error[service] = (
                        self.beta * service_weight * relative_error
                        + (1.0 - self.beta * service_weight) * self.service_error[service]
                    )
                    gradient = (prediction - target) * prediction * (1.0 - prediction)
                    self.user_bias[user] -= self.bias_eta * (
                        user_weight * gradient + self.l2 * self.user_bias[user]
                    )
                    self.service_bias[service] -= self.bias_eta * (
                        service_weight * gradient + self.l2 * self.service_bias[service]
                    )

    def _factor_stage(
        self,
        users: np.ndarray,
        services: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        shuffle_seed: int,
    ) -> float:
        if not np.all((targets > 0.0) & (targets < 1.0)):
            raise ValueError("normalized support targets must lie strictly inside (0, 1)")
        target_logits = np.log(targets / (1.0 - targets))
        bias_logits = self.user_bias[users] + self.service_bias[services]
        residual_targets = target_logits - bias_logits
        squared_error_sum = 0.0
        updates = 0
        for user in np.unique(users):
            positions = np.flatnonzero(users == user)
            rng = np.random.default_rng(np.random.SeedSequence([int(shuffle_seed), int(user), 0xFAC7]))
            for _ in range(int(epochs)):
                for position in rng.permutation(positions):
                    service = int(services[position])
                    before_user = self.user_factors[user].copy()
                    before_service = self.service_factors[service].copy()
                    error = float(before_user @ before_service - residual_targets[position])
                    self.user_factors[user] -= self.factor_eta * (
                        error * before_service + self.l2 * before_user
                    )
                    self.service_factors[service] -= self.factor_eta * (
                        error * before_user + self.l2 * before_service
                    )
                    squared_error_sum += error * error
                    updates += 1
        return squared_error_sum / max(updates, 1)

    def fit_support(
        self,
        users: np.ndarray,
        services: np.ndarray,
        raw_values: np.ndarray,
        raw_cap: float,
        bias_epochs: int,
        factor_epochs: int,
        shuffle_seed: int,
    ) -> float:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if not (len(users) == len(services) == len(raw_values)) or len(raw_values) == 0:
            raise ValueError("support arrays must be non-empty and equally sized")
        targets = self._normalize(raw_values, raw_cap)
        self._bias_stage(users, services, targets, bias_epochs, shuffle_seed)
        if not self.interaction:
            return 0.0
        return self._factor_stage(users, services, targets, factor_epochs, shuffle_seed)

    def predict(self, users: np.ndarray, services: np.ndarray, raw_cap: float) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.shape != services.shape or users.ndim != 1:
            raise ValueError("query users/services must be equal one-dimensional arrays")
        logits = self.user_bias[users] + self.service_bias[services]
        if self.interaction:
            logits = logits + np.sum(self.user_factors[users] * self.service_factors[services], axis=1)
        normalized = self._sigmoid(logits)
        return np.expm1(normalized * np.log1p(float(raw_cap))).astype(np.float32)

    def bias_state_snapshot(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name).copy()
            for name in ("user_bias", "service_bias", "user_error", "service_error")
        }

    def parameter_snapshot(self) -> dict[str, np.ndarray]:
        return {
            **self.bias_state_snapshot(),
            "user_factors": self.user_factors.copy(),
            "service_factors": self.service_factors.copy(),
        }

    def clone(self) -> "CausalResidualBiasMF":
        return copy.deepcopy(self)

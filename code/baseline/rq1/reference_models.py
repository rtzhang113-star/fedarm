                                                                         

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np

                                                                          
                                                                              
                                                                     
ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_ROOT))
if importlib.util.find_spec("torch") is None and "continual_qos" not in sys.modules:
    package = types.ModuleType("continual_qos")
    package.__path__ = [str(ROOT / "continual_qos")]                              
    sys.modules["continual_qos"] = package

from code.baseline.rq1.methods.fedem import FedEMQoSModel
from code.baseline.rq1.methods.fedrecgel import FedRecGELQoS
from continual_qos.multifed import MultiFedFedOBRMF


def _hash_arrays(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    if not values:
        raise ValueError("state mapping must not be empty")
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise FloatingPointError(f"non-finite state array: {name}")
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class _ReferenceAdapter:
    def query_snapshot(self) -> Any:
        raise NotImplementedError

    def query_unchanged(self, snapshot: Any) -> bool:
        raise NotImplementedError

    def state_hash(self) -> str:
        raise NotImplementedError


class _FedRecGELAdapter(_ReferenceAdapter):
    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        self.model = FedRecGELQoS(
            num_users=int(config["num_users"]),
            num_services=int(config["num_services"]),
            num_times=int(config["num_times"]),
            factor_num=int(config.get("factor_num", config.get("rank", 32))),
            num_layers=int(config.get("num_layers", 3)),
            local_lr=float(config.get("local_lr", 0.01)),
            global_lr=float(config.get("global_lr", 0.001)),
            rho_sh=float(config.get("rho_sh", 0.02)),
            rho_ns=float(config.get("rho_ns", 0.02)),
            local_epochs=int(config.get("local_epochs", 1)),
            client_fraction=float(config.get("client_fraction", 0.1)),
            seed=int(seed),
            communication_enabled=True,
        )

    def advance_support(
        self,
        users: Any,
        services: Any,
        values: Any,
        time_slice: int,
        raw_cap: float,
        round_seed: int,
    ) -> dict[str, Any]:
        return self.model.advance_support(
            users, services, values, time_slice, raw_cap, round_seed
        )

    def predict(
        self,
        users: Any,
        services: Any,
        time_slice: int,
        raw_cap: float,
    ) -> np.ndarray:
        return self.model.predict(users, services, time_slice, raw_cap)

    def query_snapshot(self) -> Any:
        return self.model.state_copy()

    def query_unchanged(self, snapshot: Any) -> bool:
        return bool(self.model.state_unchanged(snapshot))

    def state_hash(self) -> str:
        return _hash_arrays(self.model.state_snapshot())


class _FedEMAdapter(_ReferenceAdapter):
    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        self.model = FedEMQoSModel(
            num_users=int(config["num_users"]),
            num_services=int(config["num_services"]),
            num_times=int(config["num_times"]),
            rank=int(config.get("rank", 16)),
            learning_rate=float(config.get("learning_rate", 0.1)),
            regularization=float(config.get("regularization", 0.0)),
            clients_sample_ratio=float(config.get("clients_sample_ratio", 1.0)),
            agg_clients_ratio=float(config.get("agg_clients_ratio", 0.1)),
            agg_mode=str(config.get("agg_mode", "both")),
            sim_mode=str(config.get("sim_mode", "l2")),
            similarity_source=str(config.get("similarity_source", "persistent_pca")),
            alpha=float(config.get("alpha", 0.9)),
            adapter_learning_rate=float(config.get("adapter_learning_rate", 0.1)),
            adapter_regularization=float(config.get("adapter_regularization", 0.0)),
            local_epochs=int(config.get("local_epochs", 10)),
            batch_size=int(config.get("batch_size", 256)),
            seed=int(seed),
        )
        self.local_epochs = int(config.get("local_epochs", 10))

    @staticmethod
    def _normalize(values: Any, raw_cap: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(raw_cap) or raw_cap <= 0.0:
            raise ValueError("raw_cap must be finite and positive")
        if values.size and (not np.all(np.isfinite(values)) or np.any(values <= 0.0)):
            raise ValueError("support values must be finite and positive")
        return (
            np.log1p(values.astype(np.float64)) / np.log1p(float(raw_cap))
        ).astype(np.float32)

    def advance_support(
        self,
        users: Any,
        services: Any,
        values: Any,
        time_slice: int,
        raw_cap: float,
        round_seed: int,
    ) -> dict[str, Any]:
        del round_seed
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        targets = self._normalize(values, raw_cap)
        if users.ndim != 1 or services.shape != users.shape or targets.shape != users.shape:
            raise ValueError("support arrays must be aligned vectors")
        if users.size == 0:
            return {
                "support_entries": 0,
                "communication": {
                    "uplink_bytes": 0,
                    "downlink_bytes": 0,
                    "total_bytes": 0,
                    "participating_clients": 0,
                },
            }
        batches = []
        for user in np.unique(users):
            selected = users == int(user)
            count = int(np.count_nonzero(selected))
            batches.append(
                (
                    int(user),
                    np.full(count, int(user), dtype=np.int64),
                    services[selected],
                    np.full(count, int(time_slice), dtype=np.int64),
                    targets[selected],
                )
            )
        result = self.model.sync_round(batches, epochs=self.local_epochs)
        uplink = int(result["uplink_bytes"])
        downlink = int(result["downlink_bytes"])
        return {
            "support_entries": int(users.size),
            "aggregation_clients": int(result["aggregation_clients"]),
            "aggregation_client_ids": list(result["aggregation_client_ids"]),
            "communication": {
                "uplink_bytes": uplink,
                "downlink_bytes": downlink,
                "total_bytes": uplink + downlink,
                "participating_clients": int(result["participating_clients"]),
            },
        }

    def predict(
        self,
        users: Any,
        services: Any,
        time_slice: int,
        raw_cap: float,
    ) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        times = np.full(users.size, int(time_slice), dtype=np.int64)
        normalized = self.model.predict(users, services, times)
        return np.expm1(
            np.clip(normalized.astype(np.float64), 0.0, 1.0)
            * np.log1p(float(raw_cap))
        ).astype(np.float32)

    def query_snapshot(self) -> Any:
        return self.model.state_copy()

    def query_unchanged(self, snapshot: Any) -> bool:
        return bool(self.model.state_unchanged(snapshot))

    def state_hash(self) -> str:
        arrays = {f"array_{index:03d}": value for index, value in enumerate(self.model.state_arrays())}
        arrays["server_available"] = self.model.server_available
        arrays["server_adapter_available"] = self.model.server_adapter_available
        arrays["round_index"] = np.asarray([self.model.round_index], dtype=np.int64)
        return _hash_arrays(arrays)


class _MultiFedAdapter(_ReferenceAdapter):
    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        self.model = MultiFedFedOBRMF(
            num_users=int(config["num_users"]),
            num_services=int(config["num_services"]),
            rank=int(config.get("rank", 20)),
            seed=int(seed),
            bias_eta=float(config.get("bias_eta", 0.8)),
            factor_eta=float(config.get("factor_eta", 0.01)),
            l2=float(config.get("l2", 0.0003)),
            beta=float(config.get("beta", 0.3)),
            factor_init_std=float(config.get("factor_init_std", 0.05)),
            num_edges=int(config.get("num_edges", 6)),
            external_weight=float(config.get("external_weight", 0.5)),
            internal_aggregations=int(config.get("internal_aggregations", 1)),
            equivalence_atol=float(config.get("equivalence_atol", 1e-12)),
        )
        self.bias_epochs = int(config.get("bias_epochs", 1))
        self.factor_epochs = int(config.get("factor_epochs", 1))

    def advance_support(
        self,
        users: Any,
        services: Any,
        values: Any,
        time_slice: int,
        raw_cap: float,
        round_seed: int,
    ) -> dict[str, Any]:
        del time_slice
        result = self.model.advance_support(
            np.asarray(users, dtype=np.int64),
            np.asarray(services, dtype=np.int64),
            np.asarray(values, dtype=np.float64),
            float(raw_cap),
            self.bias_epochs,
            self.factor_epochs,
            int(round_seed),
        )
        communication = result.communication.as_dict()
        communication["participating_clients"] = len(result.active_clients)
        return {
            "support_entries": int(np.asarray(values).size),
            "active_clients": list(result.active_clients),
            "proposal_count": len(result.proposals),
            "communication": communication,
        }

    def predict(
        self,
        users: Any,
        services: Any,
        time_slice: int,
        raw_cap: float,
    ) -> np.ndarray:
        del time_slice
        return self.model.predict(
            np.asarray(users, dtype=np.int64),
            np.asarray(services, dtype=np.int64),
            float(raw_cap),
        )

    def query_snapshot(self) -> Any:
        return self.model.parameter_snapshot()

    def query_unchanged(self, snapshot: Any) -> bool:
        current = self.model.parameter_snapshot()
        return set(snapshot) == set(current) and all(
            np.array_equal(snapshot[name], current[name]) for name in snapshot
        )

    def state_hash(self) -> str:
        return _hash_arrays(self.model.parameter_snapshot())


def create_reference_model(
    method: str,
    config: Mapping[str, Any],
    seed: int,
) -> _ReferenceAdapter:
    normalized = str(method).lower()
    if normalized == "multifed":
        return _MultiFedAdapter(config, seed)
    if normalized == "fedrecgel":
        return _FedRecGELAdapter(config, seed)
    if normalized == "fedem":
        return _FedEMAdapter(config, seed)
    raise ValueError(f"unknown reference baseline method: {method!r}")


__all__ = ["create_reference_model"]
